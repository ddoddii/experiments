#!/usr/bin/env python3
"""
Per-turn TTFT / throughput vs growing context length -- ShareGPT workload.

Same per-turn schema as sglang_perturn_ctxlen.py (BFCL), so plot_perturn_ctxlen.py
and run_perturn_compare.sh work unchanged -- only the workload differs. ShareGPT is
plain multi-turn chat (long free-text replies), a different agentic pattern than
BFCL's short tool calls: the assistant reply is long and recurs verbatim in the next
turn, so generated KV is reusable. We take only the human turns, let the model
generate each assistant reply, append it verbatim, and (like BFCL) sleep a think-time
gap between turns to reproduce idle-gap eviction.

Context length per turn comes from streaming usage (prompt_tokens); no client
tokenizer. Output: results/{CONFIG}.json with per-item `turns` + a flat `points` list
[{ctx, ttft, tput, turn}].

데이터 준비 (server17, 한 번):
  cd ~/experiments/data && wget https://huggingface.co/datasets/anon8231489123/\
ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

환경변수: CONFIG, CONCURRENCY(기본 16), MAX_ITEMS(기본 200), TOOL_DELAY(기본 3),
  MIN_TURNS(3), MAX_TURNS(6), MAX_TOKENS(512), MAX_PROMPT_CHARS(6000), DATA_PATH, ROUTER_URL.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CONFIG = os.environ.get("CONFIG", "perturn_sharegpt")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
_mi = int(os.environ.get("MAX_ITEMS", "200"))
MAX_ITEMS = _mi if _mi > 0 else 200
MIN_TURNS = int(os.environ.get("MIN_TURNS", "3"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "6000"))
TIMEOUT = int(os.environ.get("TIMEOUT", "180"))
TOOL_DELAY = float(os.environ.get("TOOL_DELAY", "3"))
DATA_PATH = os.environ.get("DATA_PATH", "data/ShareGPT_V3_unfiltered_cleaned_split.json")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")

_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "chatgpt": "assistant",
         "system": "system", "bard": "assistant", "assistant": "assistant"}


def load_conversations(path):
    """ShareGPT json -> per-conversation ordered human turns (>=MIN_TURNS, bounded)."""
    with open(path) as f:
        raw = json.load(f)
    items = []
    for conv in raw:
        turns = conv.get("conversations") or conv.get("conversation") or []
        user_turns = []
        for t in turns:
            role = _ROLE.get(str(t.get("from", "")).lower())
            val = (t.get("value") or "").strip()
            if role == "user" and val:
                user_turns.append(val)
        if len(user_turns) < MIN_TURNS or len(user_turns[0]) > MAX_PROMPT_CHARS:
            continue
        items.append({"id": conv.get("id", f"conv_{len(items)}"),
                      "user_turns": user_turns[:MAX_TURNS]})
        if len(items) >= MAX_ITEMS:
            break
    return items


def run_turn(conversation):
    """One streaming turn -> (metrics dict incl. context length, assistant_content)."""
    payload = {
        "model": MODEL, "messages": conversation, "max_tokens": MAX_TOKENS,
        "stream": True, "temperature": 0.0,
        "stream_options": {"include_usage": True},  # -> prompt_tokens (context length)
    }
    t_request = time.perf_counter()
    resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()

    t_first = t_last = None
    token_count = 0
    content = ""
    prompt_tokens = completion_tokens = None
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        piece = choices[0].get("delta", {}).get("content")
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
            token_count += 1
            t_last = time.perf_counter()
            content += piece

    n_out = completion_tokens if completion_tokens else token_count
    ttft = (t_first - t_request) if t_first else None
    e2e = (t_last - t_request) if t_last else None
    decode_time = (t_last - t_first) if (t_last and token_count > 1) else None
    real = decode_time is not None and decode_time >= 5e-3 and token_count >= 4
    tpot = (decode_time / (token_count - 1)) if real else None
    tput = ((token_count - 1) / decode_time) if real else None
    return ({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_out,
        "ttft_s": round(ttft, 4) if ttft else None,
        "tpot_s": round(tpot, 4) if tpot else None,
        "e2e_latency_s": round(e2e, 4) if e2e else None,
        "throughput_tok_per_s": round(tput, 2) if tput else None,
    }, content)


def process_item(item):
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []
    turn_metrics = []
    for turn_idx, user_msg in enumerate(item["user_turns"]):
        if TOOL_DELAY > 0 and turn_idx > 0:
            time.sleep(TOOL_DELAY)  # think-time gap (idle-gap eviction, as in BFCL)
        conversation.append({"role": "user", "content": user_msg})
        try:
            m, assistant_content = run_turn(conversation)
            m["turn"] = turn_idx
            turn_metrics.append(m)
            conversation.append({"role": "assistant", "content": assistant_content or ""})
        except Exception as e:  # noqa: BLE001
            turn_metrics.append({"turn": turn_idx, "error": str(e)})
            break

    valid = [t for t in turn_metrics if t.get("ttft_s")]
    return {
        "id": item["id"],
        "num_turns": len(item["user_turns"]),
        "turns": turn_metrics,
        "avg_ttft_s": round(sum(t["ttft_s"] for t in valid) / len(valid), 4) if valid else None,
        "total_output_tokens": sum(t["completion_tokens"] for t in turn_metrics if t.get("completion_tokens")),
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"ShareGPT 데이터 없음: {DATA_PATH}\n"
            "다운로드: cd data && wget https://huggingface.co/datasets/"
            "anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/"
            "ShareGPT_V3_unfiltered_cleaned_split.json"
        )
    items = load_conversations(DATA_PATH)
    print(f"[perturn-sharegpt] CONFIG={CONFIG} C={CONCURRENCY} items={len(items)} "
          f"delay={TOOL_DELAY}s (turns {MIN_TURNS}-{MAX_TURNS}) url={ROUTER_URL}")
    if not items:
        raise SystemExit("no multi-turn conversations matched (adjust MIN_TURNS/MAX_PROMPT_CHARS).")

    results = []
    lock = threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(process_item, it): it for it in items}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="conv"):
            r = fut.result()
            with lock:
                results.append(r)
    wall = time.perf_counter() - t0

    points = []
    for r in results:
        for t in r["turns"]:
            if t.get("prompt_tokens") and t.get("ttft_s"):
                points.append({"ctx": t["prompt_tokens"], "ttft": t["ttft_s"],
                               "tput": t.get("throughput_tok_per_s"), "turn": t.get("turn")})
    total_out = sum(r["total_output_tokens"] for r in results)
    valid = [r for r in results if r.get("avg_ttft_s")]
    summary = {
        "config": CONFIG, "workload": "sharegpt", "concurrency": CONCURRENCY,
        "tool_delay_s": TOOL_DELAY, "total_items": len(results),
        "success_items": len(valid), "error_items": len(results) - len(valid),
        "n_points": len(points), "total_output_tokens": total_out,
        "total_wall_time_s": round(wall, 2),
        "overall_throughput_tok_per_s": round(total_out / wall, 2) if wall else None,
        "avg_ttft_s": round(sum(r["avg_ttft_s"] for r in valid) / len(valid), 4) if valid else None,
    }
    output = {"summary": summary, "points": points, "results": results}
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{CONFIG}.json"
    json.dump(output, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"완료: {summary['total_items']}개 / 에러 {summary['error_items']}개 (C={CONCURRENCY}, points={len(points)})")
    print(f"전체 소요 {wall:.2f}s  overall throughput {summary['overall_throughput_tok_per_s']} tok/s  "
          f"avg TTFT {summary['avg_ttft_s']}s")
    if points:
        cs = sorted(p["ctx"] for p in points)
        print(f"context length 범위: {cs[0]} ~ {cs[-1]} tok (median {cs[len(cs)//2]})")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()

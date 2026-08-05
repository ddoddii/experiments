#!/usr/bin/env python3
"""
SGLang long-context multi-turn 벤치마크 — prefill-bound testbed (Phase 2 S0).

목적: §7-6에서 "생성-KV 재사용은 prefill이 병목일 때만 serving에 도움"임이 드러났다.
ShareGPT C=32는 decode-bound라 틀린 시험대였다. 이 벤치는 의도적으로 **prefill이 병목**이
되도록 만든다 — 그래야 KV offload/prefetch(재-prefill 회피 + 전송 은닉)의 이득이 TTFT에 나타난다.

prefill-bound 만드는 법 (phase2_predictive_offload.md §5):
  - 세션마다 **긴 고유 문서 prefix** (PREFIX_WORDS) → prefill 토큰이 크고 재-prefill이 비쌈.
  - **짧은 출력** (MAX_TOKENS ~24) → decode 저렴 → prefill이 지배.
  - 누적 멀티턴: 대화가 턴마다 길어지며 그 긴 prefix를 매턴 다시 필요로 함.
  - turn 사이 **tool-gap** (TOOL_DELAY) → (Phase 2에서) offload/prefetch 기회 + 예측 신호.
  - 높은 동시성 + pool 압박(--max-total-tokens 작게)과 함께 쓰면 prefix 축출→재-prefill 유발.

문서는 세션 index 시드로 만든 **의사난수 텍스트**라 세션마다 distinct (cross-session 캐시공유로
prefill 비용이 가짜로 줄어드는 것을 방지). 질문은 일반형(답 품질이 아니라 prefill 비용이 관심).

출력 스키마는 BFCL/ShareGPT 동시성 러너와 동일 → phase0_metrics_scraper.py / head_to_head_analyze.py /
run_head_to_head.sh 그대로 물림 (BENCH=이 파일로 지정).

환경변수:
  CONFIG        결과 태그 (기본 longctx)
  CONCURRENCY   동시 세션 수 (기본 16)
  MAX_ITEMS     세션 수 (기본 128)
  NUM_TURNS     세션당 user turn 수 (기본 4)
  PREFIX_WORDS  세션 고유 문서 prefix 단어 수 (기본 3000; 클수록 prefill-bound↑)
  MAX_TOKENS    응답 최대 토큰 (기본 24; 작을수록 prefill-bound↑)
  TOOL_DELAY    turn 사이 think-time 초 (기본 3)
  SEED          문서 생성 시드 (기본 1234)
  ROUTER_URL / MODEL / TIMEOUT
"""
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CONFIG = os.environ.get("CONFIG", "longctx")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
_max_items = int(os.environ.get("MAX_ITEMS", "128"))
# 0/미설정 -> 128. run_head_to_head.sh가 MAX_ITEMS=0을 export하므로(BFCL의 0="전체" 관례) 방어.
MAX_ITEMS = _max_items if _max_items > 0 else 128
NUM_TURNS = int(os.environ.get("NUM_TURNS", "4"))
PREFIX_WORDS = int(os.environ.get("PREFIX_WORDS", "3000"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "24"))
TOOL_DELAY = float(os.environ.get("TOOL_DELAY", "3"))
TIMEOUT = int(os.environ.get("TIMEOUT", "180"))
SEED = int(os.environ.get("SEED", "1234"))
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant analyzing a document.")

# 의사난수 텍스트용 소형 어휘 (토큰이 실제로 생기게 평범한 영단어).
_VOCAB = (
    "system model request buffer memory cache token prefix decode prefill latency "
    "throughput schedule tensor kernel stream device pointer offset layer weight "
    "gradient batch sequence context window attention residual channel matrix vector "
    "policy budget region segment cluster node replica shard index queue worker signal "
    "packet payload handle session document section paragraph summary analysis result "
    "value metric baseline pressure eviction survival transfer bandwidth topology bridge"
).split()

_QUESTIONS = [
    "Briefly, what is the document mainly about?",
    "List two key terms that appear frequently.",
    "Summarize the overall theme in one sentence.",
    "What section would you read first and why?",
    "Name one concept and give a short definition.",
    "What is the shortest useful takeaway?",
]


def make_document(session_idx: int) -> str:
    """세션 고유의 긴 의사난수 문서 (PREFIX_WORDS 단어). 시드로 재현 가능, 세션마다 distinct."""
    rng = random.Random(SEED * 100003 + session_idx)
    words = [rng.choice(_VOCAB) for _ in range(PREFIX_WORDS)]
    # 문단으로 쪼개 문서처럼 보이게 (토큰 수엔 큰 영향 없음).
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i : i + 30]))
        i += 30
    return "Document:\n" + "\n".join(out)


def build_items():
    items = []
    for idx in range(MAX_ITEMS):
        doc = make_document(idx)
        turns = []
        for t in range(NUM_TURNS):
            q = _QUESTIONS[t % len(_QUESTIONS)]
            # 첫 turn에만 긴 문서를 붙인다 (이후 턴은 짧은 질문; prefix는 대화 누적으로 유지).
            turns.append((doc + "\n\n" + q) if t == 0 else q)
        items.append({"id": f"longctx_{idx}", "session_id": f"longctx_{idx}", "user_turns": turns})
    return items


def run_turn(conversation):
    payload = {
        "model": MODEL, "messages": conversation,
        "max_tokens": MAX_TOKENS, "stream": True, "temperature": 0.0,
    }
    t_request = time.perf_counter()
    resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()

    t_first = t_last = None
    token_count = 0
    content = ""
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
        delta = chunk["choices"][0]["delta"]
        piece = delta.get("content")
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
            token_count += 1
            t_last = time.perf_counter()
            content += piece

    ttft = (t_first - t_request) if t_first else None
    e2e = (t_last - t_request) if t_last else None
    decode_time = (t_last - t_first) if (t_last and token_count > 1) else None
    tpot = (decode_time / (token_count - 1)) if (decode_time and token_count > 1) else None
    tput = ((token_count - 1) / decode_time) if (decode_time and token_count > 1) else None
    return ({
        "ttft_s": round(ttft, 4) if ttft else None,
        "tpot_s": round(tpot, 4) if tpot else None,
        "output_tokens": token_count,
        "e2e_latency_s": round(e2e, 4) if e2e else None,
        "throughput_tok_per_s": round(tput, 2) if tput else None,
    }, content)


def process_item(item):
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []
    turn_metrics = []
    for turn_idx, user_msg in enumerate(item["user_turns"]):
        if TOOL_DELAY > 0 and turn_idx > 0:
            time.sleep(TOOL_DELAY)  # tool-call gap 모사 (Phase 2 예측 신호가 붙을 지점)
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
        "session_id": item["session_id"],
        "num_turns": len(item["user_turns"]),
        "turns": turn_metrics,
        "avg_ttft_s": round(sum(t["ttft_s"] for t in valid) / len(valid), 4) if valid else None,
        "avg_tpot_s": round(sum(t["tpot_s"] for t in valid if t.get("tpot_s")) / max(1, sum(1 for t in valid if t.get("tpot_s"))), 4) if valid else None,
        "avg_throughput_tok_per_s": round(sum(t["throughput_tok_per_s"] for t in valid if t.get("throughput_tok_per_s")) / max(1, sum(1 for t in valid if t.get("throughput_tok_per_s"))), 2) if valid else None,
        "total_output_tokens": sum(t["output_tokens"] for t in turn_metrics if t.get("output_tokens")),
    }


def main():
    items = build_items()
    print(f"[longctx] CONFIG={CONFIG} C={CONCURRENCY} items={len(items)} "
          f"turns={NUM_TURNS} prefix_words={PREFIX_WORDS} max_tokens={MAX_TOKENS} "
          f"delay={TOOL_DELAY}s url={ROUTER_URL}")

    results = []
    lock = threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(process_item, it): it for it in items}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="items"):
            r = fut.result()
            with lock:
                results.append(r)
    wall = time.perf_counter() - t0

    total_out = sum(r["total_output_tokens"] for r in results)
    valid = [r for r in results if r.get("avg_ttft_s")]
    summary = {
        "config": CONFIG,
        "concurrency": CONCURRENCY,
        "tool_delay_s": TOOL_DELAY,
        "num_turns": NUM_TURNS,
        "prefix_words": PREFIX_WORDS,
        "max_tokens": MAX_TOKENS,
        "total_items": len(results),
        "success_items": len(valid),
        "error_items": len(results) - len(valid),
        "total_output_tokens": total_out,
        "total_wall_time_s": round(wall, 2),
        "overall_throughput_tok_per_s": round(total_out / wall, 2) if wall else None,
        "avg_ttft_s": round(sum(r["avg_ttft_s"] for r in valid) / len(valid), 4) if valid else None,
        "avg_tpot_s": round(sum(r["avg_tpot_s"] for r in valid if r["avg_tpot_s"]) / len(valid), 4) if valid else None,
        "avg_throughput_tok_per_s": round(sum(r["avg_throughput_tok_per_s"] for r in valid if r["avg_throughput_tok_per_s"]) / len(valid), 2) if valid else None,
    }
    # Percentiles over EVERY turn, not over per-session means.
    #
    # Without these the only TTFT number this bench produced was a mean of means, and a
    # single run of it was being read as a result. On the sharegpt workload the same
    # config re-run gave p95 values spanning 1.45-6.58 s, so a mean with no spread cannot
    # tell a 15% effect from run-to-run noise. Per-turn is the right population because
    # the cache acts per turn: a session whose first turn misses and whose next three hit
    # is averaged into invisibility by the per-session mean.
    all_ttft = sorted(t["ttft_s"] for r in results for t in r.get("turns", [])
                      if t.get("ttft_s"))
    if all_ttft:
        def pct(p):
            if len(all_ttft) == 1:
                return round(all_ttft[0], 4)
            i = p / 100 * (len(all_ttft) - 1)
            lo, hi = int(i), min(int(i) + 1, len(all_ttft) - 1)
            return round(all_ttft[lo] + (all_ttft[hi] - all_ttft[lo]) * (i - lo), 4)
        summary.update({
            "n_ttft_samples": len(all_ttft),
            "ttft_p50_s": pct(50), "ttft_p90_s": pct(90),
            "ttft_p95_s": pct(95), "ttft_p99_s": pct(99),
            # Mean over turns, next to the mean-of-session-means above. They answer
            # different questions and the two arms can rank differently under them.
            "ttft_mean_per_turn_s": round(sum(all_ttft) / len(all_ttft), 4),
        })
    output = {"summary": summary, "results": results}
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{CONFIG}.json"  # CONFIG가 실험 이름 (데이터/구성 반영)
    json.dump(output, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"완료: {summary['total_items']}개 / 에러: {summary['error_items']}개  (C={CONCURRENCY})")
    print(f"전체 소요시간: {wall:.2f}s  overall throughput: {summary['overall_throughput_tok_per_s']} tok/s")
    print(f"평균 TTFT: {summary['avg_ttft_s']}s  평균 TPOT: {summary['avg_tpot_s']}s")
    print(f"(prefill-bound 확인: TTFT가 크고 TPOT는 작아야 함)")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()

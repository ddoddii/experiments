#!/usr/bin/env python3
"""
SGLang ShareGPT multi-turn 벤치마크 — 동시성(concurrent) 버전.

목적: "긴 assistant 응답을 다음 turn이 그대로 재사용" 가설을 재는 워크로드.
BFCL(짧은 tool-call, 재직렬화로 토큰 불일치)과 달리, ShareGPT는 응답이 길고 평문이라
다음 turn 프롬프트에 verbatim으로 들어간다 → decode가 생성한 KV를 다음 prefill이 재사용 가능.
park의 SGLANG_KV_PARK_GEN(생성 KV 파킹) 기여를 여기서 크게 볼 수 있는지 측정한다.

핵심 설계:
  - 데이터셋에서는 **human(user) turn만** 가져온다.
  - assistant 응답은 **모델이 생성**하고, 그 출력을 대화에 **verbatim으로 append**한다
    (dataset의 gpt 응답이 아니라 모델 자기 출력 → 다음 turn 토큰이 생성분과 일치 → 생성 KV 재사용).
  - turn 사이에 TOOL_DELAY(think-time)만큼 쉬어 idle-gap eviction을 모사(BFCL과 동일).

출력 스키마는 BFCL 동시성 러너와 동일 → phase0_metrics_scraper.py / run_park_gen_ab.sh 그대로 물림.

데이터 준비 (한 번, server17에서):
  # 표준 ShareGPT (vLLM 벤치가 쓰는 파일)
  cd ~/experiments/data
  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

환경변수:
  DATA_PATH     ShareGPT json 경로 (기본 data/ShareGPT_V3_unfiltered_cleaned_split.json)
  CONFIG        결과 태그 (기본 sharegpt)
  CONCURRENCY   동시 대화 수 (기본 8)
  MAX_ITEMS     처리할 대화 수 (기본 200)
  MIN_TURNS     포함할 최소 user turn 수 (기본 3) — 멀티턴만
  MAX_TURNS     대화당 user turn 상한 (기본 6) — 길이 bound
  MAX_TOKENS    응답 최대 토큰 (기본 512; 길수록 생성-KV 재사용 효과↑)
  MAX_PROMPT_CHARS  첫 user turn이 이보다 길면 스킵 (기본 6000, context 폭주 방지)
  TOOL_DELAY    turn 사이 think-time 초 (기본 3)
  ROUTER_URL / MODEL / TIMEOUT
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
CONFIG = os.environ.get("CONFIG", "sharegpt")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
_max_items = int(os.environ.get("MAX_ITEMS", "200"))
# 0/미설정 -> 200 (ShareGPT는 대화가 수만 개라 "전체"는 무의미; BFCL 러너의 0="전체" 관례와
# 충돌하므로 여기서 sane default로 매핑. run_park_gen_ab.sh가 MAX_ITEMS=0을 export함.)
MAX_ITEMS = _max_items if _max_items > 0 else 200
MIN_TURNS = int(os.environ.get("MIN_TURNS", "3"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "6000"))
TIMEOUT = int(os.environ.get("TIMEOUT", "180"))
TOOL_DELAY = float(os.environ.get("TOOL_DELAY", "3"))
DATA_PATH = os.environ.get("DATA_PATH", "data/ShareGPT_V3_unfiltered_cleaned_split.json")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")
# Harmless on models without a thinking mode: SGLang passes chat_template_kwargs to the
# template, which ignores keys it does not use.
# Measured on Qwen3-14B: passing enable_thinking=False is what BREAKS append-only
# rendering. The template then emits "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# as the generation prompt but renders history as "<|im_start|>assistant\n<content>",
# leaving turn N's prompt four tokens longer than the head of turn N+1's. Not passing it
# is append-only and passes the check. So this now defaults OFF.
# (Echoing the block back into the stored reply does NOT work -- the template strips
#  <think>...</think> from history content, which is why that repair failed.)
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "0") != "0"
# Prepended to every stored assistant reply so the re-rendered history reproduces what
# the generation prompt actually contained. Empty for append-only templates (Llama).
# Use the literal "\n" escapes: ASSISTANT_PREFIX='<think>\n\n</think>\n\n'
ASSISTANT_PREFIX = os.environ.get("ASSISTANT_PREFIX", "").replace("\\n", "\n")
# DEFAULTS OFF, and the measurement is why. I assumed a thinking model would need its
# <think> block removed before the reply was stored; the probe says the opposite. Qwen3's
# template strips <think>...</think> from history ONLY when enable_thinking=False. With
# the kwarg absent nothing is injected into the generation prompt and nothing is stripped
# from history, so storing the reply EXACTLY as generated is what keeps the rendering
# append-only -- and stripping it would make the stored text differ from the tokens that
# were actually generated, breaking reuse of the GENERATED KV (the n_full boundary in the
# park index) while doing nothing for prompt continuity.
#
# Kept as a knob for a template that does strip history unconditionally; verify with
#   python benchmark/check_prefix_continuity.py --model <path> --try-fixes
# whose "thinking on, RAW reply stored" candidate covers exactly this case.
#
# The remaining cost is a workload one, not a correctness one: thinking tokens consume
# the MAX_TOKENS budget, so a thinking model yields less visible content per turn and its
# conversation grows more slowly than a non-thinking model at the same setting.
STRIP_THINK = os.environ.get("STRIP_THINK", "0") != "0"
_THINK_RE = __import__("re").compile(r"<think>.*?</think>\s*", __import__("re").DOTALL)


def _store_reply(text: str) -> str:
    """The assistant text as it must be STORED so the next turn re-renders identically."""
    if STRIP_THINK and text:
        text = _THINK_RE.sub("", text)
    return ASSISTANT_PREFIX + (text or "")

_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "chatgpt": "assistant",
         "system": "system", "bard": "assistant", "assistant": "assistant"}


def _model_from_server(url, timeout=5.0):
    """Ask the server what it actually loaded. SGLang IGNORES the request's `model`
    field, so a wrong MODEL here does not fail -- it silently mislabels the run. A
    Llama-3.1-8B run was once filed under Qwen3-14B for exactly this reason."""
    base = url.split("/v1/")[0].rstrip("/")
    try:
        import urllib.request

        with urllib.request.urlopen(f"{base}/get_model_info", timeout=timeout) as r:
            info = json.loads(r.read().decode())
        return info.get("model_path") or info.get("model") or None
    except Exception:  # noqa: BLE001
        return None


_served = _model_from_server(ROUTER_URL)
if _served:
    if os.path.basename(_served.rstrip("/")) != os.path.basename(MODEL.rstrip("/")):
        print(f"[model] MODEL env says {os.path.basename(MODEL.rstrip('/'))} but the "
              f"server serves {os.path.basename(_served.rstrip('/'))} -- using the "
              f"server's.")
    MODEL = _served
else:
    print(f"[model] WARNING: /get_model_info unreachable; trusting MODEL={MODEL}.")


def load_conversations(path):
    """ShareGPT json -> 각 대화의 ordered user(human) turn 리스트만 추출.

    응답(gpt)은 버린다 (모델이 직접 생성). 멀티턴(>=MIN_TURNS user)만, 첫 turn이 너무 길면 스킵.
    """
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
        if len(user_turns) < MIN_TURNS:
            continue
        if len(user_turns[0]) > MAX_PROMPT_CHARS:
            continue
        items.append({"id": conv.get("id", f"conv_{len(items)}"),
                      "user_turns": user_turns[:MAX_TURNS]})
        if len(items) >= MAX_ITEMS:
            break
    return items


def run_turn(conversation):
    """한 turn 요청(스트리밍) → (metrics, assistant_content)."""
    payload = {
        "model": MODEL, "messages": conversation,
        "max_tokens": MAX_TOKENS, "stream": True, "temperature": 0.0,
        # include_usage gives the SERVER's prompt_tokens on the final chunk. Without it
        # there is no per-turn context length in the output, and context length is the
        # precondition for reading any TTFT comparison: the most KV reuse can save on a
        # turn is what a full re-prefill of that context costs. A run whose context never
        # grew cannot show a reuse benefit no matter how well placement works, and that
        # is not visible after the fact if the length was never recorded.
        "stream_options": {"include_usage": True},
    }
    # Qwen3 ships with hybrid thinking ON by default, so every response would open with a
    # <think> block that eats the MAX_TOKENS budget and makes the conversation grow at a
    # completely different rate from Llama's. That is a different workload, not a
    # different model, and it would show up in the cross-model figure as if it were a
    # property of the placement policy. DISABLE_THINKING=0 to keep thinking on.
    if DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t_request = time.perf_counter()
    resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()

    t_first = t_last = None
    token_count = 0
    prompt_tokens = None
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
        usage = chunk.get("usage")
        if usage and usage.get("prompt_tokens"):
            prompt_tokens = usage["prompt_tokens"]
        if not chunk.get("choices"):
            continue          # the usage-only final chunk carries no choices
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
        # server-reported; falls back to a chars/4 estimate so the field is never absent
        "prompt_tokens": prompt_tokens if prompt_tokens else int(
            sum(len(str(m.get("content", "") or "")) for m in conversation) / 4),
        "prompt_tokens_exact": prompt_tokens is not None,
        "e2e_latency_s": round(e2e, 4) if e2e else None,
        "throughput_tok_per_s": round(tput, 2) if tput else None,
    }, content)


def process_item(item):
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []
    turn_metrics = []
    for turn_idx, user_msg in enumerate(item["user_turns"]):
        # tool-call 유휴시간 모사: 이전 assistant 이후 다음 user turn까지의 gap.
        if TOOL_DELAY > 0 and turn_idx > 0:
            time.sleep(TOOL_DELAY)
        conversation.append({"role": "user", "content": user_msg})
        try:
            m, assistant_content = run_turn(conversation)
            m["turn"] = turn_idx
            turn_metrics.append(m)
            # 모델의 생성 출력을 verbatim으로 append (다음 turn이 이 토큰을 그대로 재사용).
            # ASSISTANT_PREFIX repairs a template that appends a block to the assistant
            # slot when generating but drops it when re-rendering history. Qwen3 does
            # exactly that: with enable_thinking=False it emits
            # "<|im_start|>assistant\n<think>\n\n</think>\n\n" as the generation prompt,
            # and renders history as "<|im_start|>assistant\n<content>" -- so turn N's
            # prompt is four tokens longer than the head of turn N+1's and is NOT a
            # prefix of it. Prefix reuse then silently degrades and the parked-KV index,
            # which hashes the exact sequence, misses every lookup (measured: 1623
            # lookups, 0 hits, while the radix baseline still scored 45%).
            # Writing the block back into the stored reply makes the two agree.
            # Verify with: python benchmark/check_prefix_continuity.py --model <path>
            #              --try-fixes     (it names the setting to use)
            conversation.append({"role": "assistant",
                                 "content": _store_reply(assistant_content)})
        except Exception as e:  # noqa: BLE001
            turn_metrics.append({"turn": turn_idx, "error": str(e)})
            break

    valid = [t for t in turn_metrics if t.get("ttft_s")]
    return {
        "id": item["id"],
        "num_turns": len(item["user_turns"]),
        "turns": turn_metrics,
        "avg_ttft_s": round(sum(t["ttft_s"] for t in valid) / len(valid), 4) if valid else None,
        "avg_tpot_s": round(sum(t["tpot_s"] for t in valid if t.get("tpot_s")) / max(1, sum(1 for t in valid if t.get("tpot_s"))), 4) if valid else None,
        "avg_throughput_tok_per_s": round(sum(t["throughput_tok_per_s"] for t in valid if t.get("throughput_tok_per_s")) / max(1, sum(1 for t in valid if t.get("throughput_tok_per_s"))), 2) if valid else None,
        "total_output_tokens": sum(t["output_tokens"] for t in turn_metrics if t.get("output_tokens")),
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
    print(f"[sharegpt] CONFIG={CONFIG} C={CONCURRENCY} items={len(items)} "
          f"(min_turns={MIN_TURNS}, max_turns={MAX_TURNS}) url={ROUTER_URL}")
    if not items:
        raise SystemExit("조건에 맞는 멀티턴 대화가 없음 (MIN_TURNS/MAX_PROMPT_CHARS 조정).")

    results = []
    lock = threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(process_item, it): it for it in items}
        # mininterval=15: when stdout is a pipe rather than a terminal tqdm cannot
        # rewrite a line in place, so every refresh becomes a NEW line. At the default
        # 0.1 s that is thousands of lines of log; at 15 s it is a readable heartbeat
        # that also proves the run is alive.
        bar = tqdm(as_completed(futs), total=len(futs), desc="items",
                   mininterval=15, miniters=1)
        _ttfts, _errs = [], 0
        for fut in bar:
            r = fut.result()
            with lock:
                results.append(r)
            for t in (r.get("turns") or []):
                if t.get("error"):
                    _errs += 1
                elif t.get("ttft_s") is not None:
                    _ttfts.append(t["ttft_s"])
            # Carry the numbers that decide whether to abort: a median TTFT drifting
            # upward or an error count climbing means the run is already spoiled, and
            # waiting 45 minutes to discover that has happened here before.
            if _ttfts:
                med = sorted(_ttfts)[len(_ttfts) // 2]
                bar.set_postfix_str(f"turns={len(_ttfts)} med_ttft={med:.2f}s err={_errs}",
                                    refresh=False)
    wall = time.perf_counter() - t0

    total_out = sum(r["total_output_tokens"] for r in results)
    valid = [r for r in results if r.get("avg_ttft_s")]
    # Percentiles, not just the mean: tail TTFT is what the placement comparison turns on,
    # and collect_arm_metrics reads these keys by name (same schema as the BFCL runner).
    _ttfts = sorted(t["ttft_s"] for r in results for t in (r.get("turns") or [])
                    if t.get("ttft_s") is not None)

    def _pct(p):
        if not _ttfts:
            return None
        k = max(0, min(len(_ttfts) - 1, int(round((p / 100.0) * (len(_ttfts) - 1)))))
        return round(_ttfts[k], 4)

    summary = {
        "config": CONFIG,
        "model": MODEL,
        "ttft_p50_s": _pct(50),
        "ttft_p90_s": _pct(90),
        "ttft_p95_s": _pct(95),
        "ttft_p99_s": _pct(99),
        "n_ttft_samples": len(_ttfts),
        "concurrency": CONCURRENCY,
        "tool_delay_s": TOOL_DELAY,
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
    output = {"summary": summary, "results": results}
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{CONFIG}.json"  # CONFIG가 실험 이름 (데이터/구성 반영)
    json.dump(output, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"완료: {summary['total_items']}개 / 에러: {summary['error_items']}개  (C={CONCURRENCY})")
    print(f"전체 소요시간: {wall:.2f}s  overall throughput: {summary['overall_throughput_tok_per_s']} tok/s")
    print(f"평균 TTFT: {summary['avg_ttft_s']}s  평균 TPOT: {summary['avg_tpot_s']}s")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
KV Cache Tier TTFT 검증 스크립트
=================================
non-disagg hicache 서버에서 prefix cache가 실제로 동작하는지,
그리고 GPU KV가 evict된 후 L2/L3에서 reload 되는지를 TTFT로 측정한다.

3-way 비교:
  [A] Cold     : 완전 새 대화, 캐시 없음 → full prefill  (기준값)
  [B] L1 hit   : 직전 turn 재사용, GPU 캐시 hit         (A보다 훨씬 빠름)
  [C] After eviction: flood으로 GPU KV 밀어낸 후 재개
                      L2 DRAM fetch  → A보다 빠르고 B보다 느림 (L2 동작 확인)
                      full recompute → A ≈ C             (캐시 미동작)

사용법:
  python benchmark/test_cache_ttft.py
  SGLANG_URL=http://localhost:30001/v1/chat/completions python benchmark/test_cache_ttft.py
  PREFIX_TOKENS=5000 FLOOD_N=40 python benchmark/test_cache_ttft.py

환경변수:
  SGLANG_URL      서버 URL (default: http://localhost:30000/v1/chat/completions)
  MODEL           모델 이름 (default: Qwen/Qwen3-14B)
  PREFIX_TOKENS   prefix 길이 (토큰 수 근사, default: 3000)
  FLOOD_N         eviction용 flood 요청 수 (default: 30)
  FLOOD_TOKENS    flood 요청당 토큰 수 (default: 3000)
  REPEAT          각 측정 반복 횟수 (default: 3)
"""

import os
import sys
import threading
import time

import requests

URL    = os.environ.get("SGLANG_URL",   "http://localhost:30000/v1/chat/completions")
MODEL  = os.environ.get("MODEL",        "Qwen/Qwen3-14B")
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "3000"))
FLOOD_N       = int(os.environ.get("FLOOD_N",       "30"))
FLOOD_TOKENS  = int(os.environ.get("FLOOD_TOKENS",  "3000"))
REPEAT        = int(os.environ.get("REPEAT",        "3"))

# prefix 텍스트 생성 (PREFIX_TOKENS 개 토큰 근사: 1 token ≈ 4~5 chars)
_WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
          "nu xi omicron pi rho sigma tau upsilon phi chi psi omega ") * 200
PREFIX_TEXT = _WORDS[: PREFIX_TOKENS * 5]

# 대화 메시지 구성
MSG_TURN1 = [{"role": "user", "content": PREFIX_TEXT}]
MSG_TURN2 = MSG_TURN1 + [
    {"role": "assistant", "content": "Understood."},
    {"role": "user",      "content": "Please continue briefly."},
]


def ttft(messages: list, label: str = "") -> float:
    """Streaming 요청으로 TTFT(Time To First Token) 측정."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            URL,
            json={"model": MODEL, "messages": messages,
                  "max_tokens": 8, "temperature": 0, "stream": True},
            stream=True,
            timeout=120,
        )
        for raw in r.iter_lines():
            if raw and raw != b"data: [DONE]":
                line = raw.decode("utf-8", errors="replace").lstrip("data: ").strip()
                if line:
                    return time.perf_counter() - t0
    except Exception as e:
        print(f"  !! request error ({label}): {e}", file=sys.stderr)
    return float("nan")


def flood_evict(n: int = FLOOD_N, tokens: int = FLOOD_TOKENS):
    """GPU KV pool을 overflow시켜 기존 KV를 L2/L3으로 evict한다.

    각 요청은 고유한 seed prefix를 포함해 radix cache hit이 발생하지 않도록 한다.
    """
    print(f"  flooding: {n} req × {tokens} tok = {n*tokens:,} tokens "
          f"(goal: overflow GPU KV pool)...", end="", flush=True)
    filler = (_WORDS * 10)[: tokens * 5]

    def _one(seed: int):
        text = f"[flush seed={seed:08x}] " + filler
        try:
            requests.post(URL, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 4, "temperature": 0, "stream": False,
            }, timeout=90)
        except Exception:
            pass

    threads = [threading.Thread(target=_one, args=(s,), daemon=True) for s in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=95)
    print(f" done ({time.perf_counter()-t0:.1f}s)")


def measure(label: str, messages: list, repeat: int = REPEAT) -> tuple[float, list[float]]:
    samples = []
    for i in range(repeat):
        v = ttft(messages, label)
        samples.append(v)
        print(f"    [{i+1}/{repeat}] {v:.3f}s")
    valid = [v for v in samples if v == v]  # filter NaN
    mean = sum(valid) / len(valid) if valid else float("nan")
    return mean, samples


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" KV Cache Tier TTFT 검증")
    print("=" * 60)
    print(f" URL          : {URL}")
    print(f" Model        : {MODEL}")
    print(f" prefix tokens: ~{PREFIX_TOKENS:,} (text len={len(PREFIX_TEXT):,} chars)")
    print(f" flood        : {FLOOD_N} req × {FLOOD_TOKENS} tok = {FLOOD_N*FLOOD_TOKENS:,} tok")
    print(f" repeat       : {REPEAT}")
    print()

    # ── [A] Cold TTFT ────────────────────────────────────────────────────
    print("[A] Cold TTFT (no cache, full prefill baseline)")
    mean_a, _ = measure("cold", MSG_TURN1)
    print(f"    → mean = {mean_a:.3f}s")
    print()

    # ── [B] L1 GPU cache hit ─────────────────────────────────────────────
    print("[B] L1 GPU cache hit TTFT (same prefix, next turn)")
    print("    (서버가 방금 [A] 요청의 KV를 GPU에 보유 중)")
    mean_b, _ = measure("l1_hit", MSG_TURN2)
    print(f"    → mean = {mean_b:.3f}s")
    if mean_a > 0:
        ratio_b = mean_b / mean_a
        verdict = "✓ L1 cache 정상" if ratio_b < 0.5 else "✗ cache 미동작 의심"
        print(f"    → B/A = {ratio_b:.2f}  ({verdict})")
    print()

    # ── Flood eviction ────────────────────────────────────────────────────
    print("[*] KV eviction 유발 (flood)")
    flood_evict()
    print()

    # ── [C] After eviction TTFT ───────────────────────────────────────────
    print("[C] After eviction TTFT (prefix KV가 GPU에서 evict된 후)")
    print("    L2 DRAM fetch  → A > C > B 예상")
    print("    full recompute → C ≈ A     예상")
    mean_c, _ = measure("after_evict", MSG_TURN2)
    print(f"    → mean = {mean_c:.3f}s")
    print()

    # ── 결과 요약 ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(" 결과 요약")
    print("=" * 60)
    print(f"  [A] Cold          : {mean_a:.3f}s  (기준)")
    print(f"  [B] L1 GPU hit    : {mean_b:.3f}s  (B/A = {mean_b/mean_a:.2f})")
    print(f"  [C] After eviction: {mean_c:.3f}s  (C/A = {mean_c/mean_a:.2f})")
    print()

    if mean_b / mean_a < 0.5:
        print("  ✓ L1 prefix cache 동작 확인 (B << A)")
    else:
        print("  ✗ L1 prefix cache 미동작 (B ≈ A) — 서버 설정 확인 필요")

    if mean_b / mean_a < 0.5:  # L1이 동작하는 경우에만 eviction 판단 의미있음
        if mean_c / mean_a < 0.85:
            print(f"  ✓ L2/L3 hicache fetch 동작 (C < A, C/A={mean_c/mean_a:.2f})")
            print(f"    → eviction 후에도 재계산보다 빠름 = DRAM/SSD 에서 KV load 중")
        elif mean_c / mean_b > 1.5:
            print(f"  △ eviction 오버헤드 관찰됨 (C > B, C/B={mean_c/mean_b:.2f})")
            print(f"    → KV를 DRAM/SSD에서 로드하는 추가 지연 발생")
        else:
            print(f"  ? eviction 효과 불명확 (C ≈ A, 재계산과 구분 어려움)")
    print()

    # KV pool 크기 조언
    print("  ※ C ≈ A 이고 flood 후에도 변화 없다면:")
    print("     → GPU KV pool이 충분히 커서 eviction이 발생하지 않았을 수 있음")
    print("     → MEM_FRACTION 낮추거나 FLOOD_N/FLOOD_TOKENS 늘려서 재시도")


if __name__ == "__main__":
    main()

"""
vLLM 2P2D Warmup Script
========================
벤치마크 전에 실행하여 두 가지 cold-start 패널티를 제거:
  1. Triton JIT 컴파일 (_compute_slot_mapping_kernel)
  2. NCCL P2P 채널 초기화 (ncclCommInitRank, 모든 P→D 페어)

Usage:
  python scripts/vllm/warmup_2P2D.py
  # 또는 start_2P_2D.sh 에서 서버 ready 후 자동 실행

Warmup 요청 수:
  8개 sequential → 라운드로빈 라우팅이면 P1→D1, P1→D2, P2→D1, P2→D2 반복해서 모두 커버
"""

import json
import time
import requests
import sys

PROXY_URL   = "http://127.0.0.1:10001/v1/chat/completions"
MODEL       = "Llama"
N_WARMUP    = 8          # P→D 페어 4가지를 최소 2번씩
TIMEOUT     = 300        # 첫 요청에 NCCL init 있으므로 넉넉히

SHORT_PAYLOAD = {
    "model":      MODEL,
    "messages":   [{"role": "user", "content": "Say hi."}],
    "max_tokens": 8,
    "stream":     True,
}

def drain_stream(resp):
    tokens = 0
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                chunk = json.loads(line[6:])
                if chunk["choices"][0]["delta"].get("content"):
                    tokens += 1
            except Exception:
                pass
    return tokens

print(f"[warmup] target: {PROXY_URL}")
print(f"[warmup] sending {N_WARMUP} requests to trigger Triton JIT + NCCL init for all P→D pairs")
print(f"[warmup] first request may take 30-180s (cold start). subsequent ones will be fast.")
print()

for i in range(N_WARMUP):
    t0 = time.perf_counter()
    try:
        resp = requests.post(PROXY_URL, json=SHORT_PAYLOAD, stream=True, timeout=TIMEOUT)
        resp.raise_for_status()
        toks = drain_stream(resp)
        elapsed = time.perf_counter() - t0
        status = "✓"
    except Exception as e:
        elapsed = time.perf_counter() - t0
        toks    = 0
        status  = f"✗ {e}"
    print(f"  warmup [{i+1}/{N_WARMUP}] {elapsed:.1f}s  tokens={toks}  {status}")

print()
print("[warmup] done — JIT compiled, NCCL channels initialized.")
print("[warmup] Now run the benchmark:")
print("  python benchmark/vllm_2P2D_BFCL_v3_multi_turn_base.py")

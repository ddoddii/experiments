#!/usr/bin/env python3
"""
NVLink KV-transfer 마이크로벤치 (Phase 1 de-risk).

질문: "P(GPU0)↔D(GPU1) NVLink 56GB/s가 실제로 host-L2(PCIe)보다 빠른
       per-prefix park/fetch로 이어지는가?"

비교 (대화 prefix KV를 다음 턴 prefill 노드=P GPU에서 쓸 수 있게 만드는 비용):
  - NVLink park : GPU_D → GPU_P 직접 P2P 복사 (1홉). 이후 fetch=0 (P GPU 상주).
  - Host L2     : GPU_D → pinned host (PCIe) [park]  +  host → GPU_P (PCIe) [fetch] (2홉).
                  = 현재 sglang hierarchical cache의 device<->host 경로와 동일 하드웨어 길.

sglang 본체를 건드리지 않고 전송 하드웨어 상한을 잰다. (cross-process NIXL 핸드셰이크
오버헤드는 별도이며 512MB급 전송 대비 작다 — 여기선 대역폭 상한을 확인.)

주의: 서버가 GPU0/1을 점유 중이면 먼저 내리고 실행 (./scripts/sglang/stop_1P_1D.sh).

사용:
  python benchmark/nvlink_kv_transfer_microbench.py
  SRC_GPU=1 DST_GPU=0 python benchmark/nvlink_kv_transfer_microbench.py
"""
import json
import os
import time

import torch

SRC_GPU = int(os.environ.get("SRC_GPU", "1"))  # D node GPU
DST_GPU = int(os.environ.get("DST_GPU", "0"))  # P node GPU (parking target)
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))

# Llama-3.1-8B KV per token (fp16): 2(K,V) * layers * kv_heads * head_dim * 2 bytes
LAYERS = int(os.environ.get("LAYERS", "32"))
KV_HEADS = int(os.environ.get("KV_HEADS", "8"))
HEAD_DIM = int(os.environ.get("HEAD_DIM", "128"))
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2

TOKEN_COUNTS = [int(x) for x in os.environ.get("TOKENS", "512,1024,2048,4096").split(",")]

# 참고용: 4000토큰 prefill 재계산 대략치 (문서 4.1). 실측 TTFT로 대체 가능.
RECOMPUTE_REF_MS = float(os.environ.get("RECOMPUTE_REF_MS", "0"))  # 0=표시안함


def _sync(*tensors):
    for t in tensors:
        if t.device.type == "cuda":
            torch.cuda.synchronize(t.device)


def time_copy(dst, src, iters, warmup):
    """dst.copy_(src) 평균 시간(ms). CUDA device만 동기화 (host tensor 안전)."""
    for _ in range(warmup):
        dst.copy_(src)
    _sync(src, dst)
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    _sync(src, dst)
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    assert torch.cuda.is_available(), "CUDA required"
    ndev = torch.cuda.device_count()
    assert SRC_GPU < ndev and DST_GPU < ndev, f"need GPUs {SRC_GPU},{DST_GPU}; have {ndev}"

    p2p = torch.cuda.can_device_access_peer(DST_GPU, SRC_GPU)
    print(f"# device count={ndev}  SRC(D)=GPU{SRC_GPU}  DST(P)=GPU{DST_GPU}")
    print(f"# P2P access GPU{DST_GPU}<-GPU{SRC_GPU}: {p2p}  "
          f"({'NVLink/P2P direct' if p2p else 'NO P2P → cross-device copy staged via host!'})")
    print(f"# KV bytes/token (fp16) = {BYTES_PER_TOKEN} ({BYTES_PER_TOKEN/1024:.0f} KB)")
    print(f"# iters={ITERS} warmup={WARMUP}\n")

    rows = []
    for ntok in TOKEN_COUNTS:
        nbytes = ntok * BYTES_PER_TOKEN
        nelem = nbytes // 2  # fp16
        size_mb = nbytes / 1e6

        src = torch.empty(nelem, dtype=torch.float16, device=f"cuda:{SRC_GPU}")
        src.fill_(1.0)
        dst_gpu = torch.empty(nelem, dtype=torch.float16, device=f"cuda:{DST_GPU}")
        host = torch.empty(nelem, dtype=torch.float16, pin_memory=True)  # pinned for max PCIe BW

        # --- NVLink park: D GPU -> P GPU (1홉) ---
        nvlink_ms = time_copy(dst_gpu, src, ITERS, WARMUP)

        # --- Host L2: park D->host, fetch host->P ---
        host_park_ms = time_copy(host, src, ITERS, WARMUP)       # D GPU -> host (PCIe)
        host_fetch_ms = time_copy(dst_gpu, host, ITERS, WARMUP)  # host -> P GPU (PCIe)
        host_roundtrip_ms = host_park_ms + host_fetch_ms

        def bw(ms):
            return nbytes / (ms / 1000.0) / 1e9  # GB/s

        row = {
            "tokens": ntok,
            "size_mb": round(size_mb, 1),
            "nvlink_park_ms": round(nvlink_ms, 3),
            "nvlink_park_GBps": round(bw(nvlink_ms), 1),
            "host_park_ms": round(host_park_ms, 3),
            "host_fetch_ms": round(host_fetch_ms, 3),
            "host_roundtrip_ms": round(host_roundtrip_ms, 3),
            "host_park_GBps": round(bw(host_park_ms), 1),
            # 다음 턴 가용성 비용: NVLink=park(fetch 0) vs Host=park+fetch
            "speedup_vs_host": round(host_roundtrip_ms / nvlink_ms, 2),
        }
        rows.append(row)
        del src, dst_gpu, host
        torch.cuda.empty_cache()

    # --- 출력 ---
    print(f"{'tokens':>7} {'MB':>7} | {'NVLink park':>12} {'GB/s':>6} | "
          f"{'host park':>10} {'host fetch':>10} {'host RT':>9} | {'speedup':>7}")
    print("-" * 92)
    for r in rows:
        line = (f"{r['tokens']:>7} {r['size_mb']:>7} | "
                f"{r['nvlink_park_ms']:>10}ms {r['nvlink_park_GBps']:>6} | "
                f"{r['host_park_ms']:>8}ms {r['host_fetch_ms']:>8}ms {r['host_roundtrip_ms']:>7}ms | "
                f"{r['speedup_vs_host']:>6}x")
        print(line)

    print("\n해석:")
    print("  - NVLink park = 다음 턴에 prefix가 P GPU에 바로 상주 (fetch 0).")
    print("  - host RT = device→host + host→device (현재 hierarchical L2가 하는 왕복).")
    print("  - speedup = host_roundtrip / nvlink_park.")
    if RECOMPUTE_REF_MS > 0:
        print(f"  - 참고: 재계산(prefill) 대략 {RECOMPUTE_REF_MS}ms → 둘 다 이보다 훨씬 빠름.")
    print("  - radix insert는 페이지 포인터 메타데이터 조작(O(pages))이라 데이터 전송 대비 무시 가능.")

    out = os.environ.get("OUT", "results/nvlink_microbench.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"config": {
        "src_gpu": SRC_GPU, "dst_gpu": DST_GPU, "p2p": p2p,
        "bytes_per_token": BYTES_PER_TOKEN, "iters": ITERS,
    }, "rows": rows}, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

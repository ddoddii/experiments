#!/usr/bin/env python3
"""
Cross-process CUDA IPC + P2P 마이크로벤치 (Phase 1 슬라이스 2 de-risk).

질문: sglang의 D(prefill과 별개 프로세스)와 P가 서로 다른 프로세스인데,
      D(GPU1) -> P(GPU0) KV 전송을 CUDA IPC로 공유해 NVLink(P2P)로 실제로
      옮길 수 있는가? 단일 프로세스 벤치(52 GB/s)가 cross-process에서도 유지되나?

구성:
  - producer 프로세스: GPU1(D)에 KV-크기 텐서 할당 → CUDA IPC로 consumer에 공유.
  - consumer 프로세스: GPU0(P)에서 그 텐서를 IPC로 매핑 → dst(GPU0).copy_(src_ipc)
                       = D(GPU1) -> P(GPU0) NVLink P2P 전송. 이걸 반복 측정.

CUDA IPC 핸들 교환은 torch.multiprocessing(spawn) + Queue가 자동 처리한다.

주의: 서버가 GPU0/1을 점유 중이면 먼저 내리고 실행 (./scripts/sglang/stop_1P_1D.sh).

사용:
  python benchmark/nvlink_cross_process_p2p_microbench.py
  SRC_GPU=1 DST_GPU=0 TOKENS=512,1024,2048,4096 python benchmark/nvlink_cross_process_p2p_microbench.py
"""
import json
import os
import time

import torch
import torch.multiprocessing as mp

SRC_GPU = int(os.environ.get("SRC_GPU", "1"))  # D node GPU (producer)
DST_GPU = int(os.environ.get("DST_GPU", "0"))  # P node GPU (consumer / parking target)
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))

LAYERS = int(os.environ.get("LAYERS", "32"))
KV_HEADS = int(os.environ.get("KV_HEADS", "8"))
HEAD_DIM = int(os.environ.get("HEAD_DIM", "128"))
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2  # K,V * layers * heads * dim * fp16
TOKEN_COUNTS = [int(x) for x in os.environ.get("TOKENS", "512,1024,2048,4096").split(",")]


def producer(nelem, src_gpu, q_out, q_done):
    torch.cuda.set_device(src_gpu)
    src = torch.ones(nelem, dtype=torch.float16, device=f"cuda:{src_gpu}")
    torch.cuda.synchronize(src_gpu)
    q_out.put(src)          # CUDA IPC 공유 (spawn + torch reductions)
    q_done.get()            # consumer가 끝날 때까지 텐서(및 프로세스) 살려둠


def consumer(nelem, dst_gpu, src_gpu, iters, warmup, q_in, q_done, q_res):
    torch.cuda.set_device(dst_gpu)
    res = {"p2p": torch.cuda.can_device_access_peer(dst_gpu, src_gpu)}
    t_get0 = time.perf_counter()
    src = q_in.get()        # IPC로 매핑된 producer의 GPU1 텐서
    setup_ms = (time.perf_counter() - t_get0) * 1000.0
    dst = torch.empty(nelem, dtype=torch.float16, device=f"cuda:{dst_gpu}")
    try:
        for _ in range(warmup):
            dst.copy_(src)
        torch.cuda.synchronize(dst_gpu)
        torch.cuda.synchronize(src_gpu)
        t0 = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch.cuda.synchronize(dst_gpu)
        torch.cuda.synchronize(src_gpu)
        res["ms"] = (time.perf_counter() - t0) / iters * 1000.0
        res["setup_ms"] = setup_ms
        res["correct"] = bool(dst[0].item() == 1.0 and dst[-1].item() == 1.0)
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    q_res.put(res)
    q_done.put(1)


def main():
    assert torch.cuda.device_count() > max(SRC_GPU, DST_GPU), "need both GPUs"
    ctx = mp.get_context("spawn")
    print(f"# cross-process CUDA IPC + P2P  SRC(D)=GPU{SRC_GPU} -> DST(P)=GPU{DST_GPU}")
    print(f"# KV bytes/token (fp16) = {BYTES_PER_TOKEN} ({BYTES_PER_TOKEN/1024:.0f} KB)")
    print(f"# iters={ITERS} warmup={WARMUP}\n")

    rows = []
    for ntok in TOKEN_COUNTS:
        nbytes = ntok * BYTES_PER_TOKEN
        nelem = nbytes // 2
        q_out, q_done, q_res = ctx.Queue(), ctx.Queue(), ctx.Queue()
        p = ctx.Process(target=producer, args=(nelem, SRC_GPU, q_out, q_done))
        c = ctx.Process(
            target=consumer,
            args=(nelem, DST_GPU, SRC_GPU, ITERS, WARMUP, q_out, q_done, q_res),
        )
        p.start()
        c.start()
        res = q_res.get()   # consumer 결과
        p.join(timeout=30)
        c.join(timeout=30)

        if "error" in res:
            print(f"  {ntok} tok: ERROR {res['error']}")
            rows.append({"tokens": ntok, "error": res["error"], "p2p": res.get("p2p")})
            continue
        ms = res["ms"]
        gbps = nbytes / (ms / 1000.0) / 1e9
        row = {
            "tokens": ntok,
            "size_mb": round(nbytes / 1e6, 1),
            "p2p": res["p2p"],
            "xproc_p2p_ms": round(ms, 3),
            "xproc_p2p_GBps": round(gbps, 1),
            "ipc_setup_ms": round(res["setup_ms"], 2),
            "correct": res["correct"],
        }
        rows.append(row)

    print(f"{'tokens':>7} {'MB':>7} {'p2p':>5} | {'x-proc copy':>12} {'GB/s':>6} "
          f"{'ipc setup':>10} {'ok':>4}")
    print("-" * 68)
    for r in rows:
        if "error" in r:
            print(f"{r['tokens']:>7}  ERROR: {r['error']}")
            continue
        print(f"{r['tokens']:>7} {r['size_mb']:>7} {str(r['p2p']):>5} | "
              f"{r['xproc_p2p_ms']:>10}ms {r['xproc_p2p_GBps']:>6} "
              f"{r['ipc_setup_ms']:>8}ms {str(r['correct']):>4}")

    print("\n해석:")
    print("  - p2p=True + GB/s ~50 이면: cross-process에서도 NVLink 직결 전송 성공 → 슬라이스 2 CUDA IPC 방식 확정.")
    print("  - GB/s가 ~25 이하로 떨어지면: PCIe로 staged된 것 → NVLink 미사용, 원인 조사 필요.")
    print("  - ipc setup은 1회성 핸들 교환 비용(큐 지연 포함). 실제 park는 이 setup을 미리/유휴에 함.")

    out = os.environ.get("OUT", "results/nvlink_xproc_microbench.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"config": {"src_gpu": SRC_GPU, "dst_gpu": DST_GPU,
                          "bytes_per_token": BYTES_PER_TOKEN, "iters": ITERS},
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

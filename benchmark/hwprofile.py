#!/usr/bin/env python3
"""
Measure this machine's constants once, so the analysis stops carrying another one's.

WHY THIS IS NOT OPTIONAL. Several numbers baked into these scripts were measured on
4x RTX A6000 with Llama-3.1-8B, and they are load-bearing rather than cosmetic:

  peer / host bandwidth   decides the "link budget" -- how much of any observed gap the
                          medium could possibly explain. 52.7 and 26.3 GB/s here; an
                          NV12-linked A100 pair is several times the first number, which
                          moves the budget by the same factor.
  prefill ms/token        the value of a cache hit, and the input to the park module's
                          reuse-value eviction. An A100 prefills faster, so the SAME hit
                          rate is worth less wall-clock there.
  card size               drives park-pool sizing and the memory reserve.

Run this on each new machine and the analysis scripts pick the numbers up automatically.
Without it they fall back to the A6000 values and say so loudly -- the failure mode this
exists to prevent is a plausible-looking result computed from another box's hardware.

WHAT IT DOES NOT MEASURE: prefill ms/token needs a running server, so it is not here.
Get it from benchmark/ttft_ctx_sweep.py and pass --prefill-ms-per-tok, or leave it unset
and the field stays null rather than silently inheriting 0.131656.

Run with the servers DOWN; it allocates on every GPU.

사용:
  python benchmark/hwprofile.py --out results/hwprofile.json
  python benchmark/hwprofile.py --out results/hwprofile.json --prefill-ms-per-tok 0.045
"""
import argparse
import json
import os
import subprocess
import time

import torch

MB = float(os.environ.get("MB", "256"))
ITERS = int(os.environ.get("ITERS", "10"))


def bench_copy(src, dst, n_elem, host=False):
    if host:
        a = torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{src}")
        b = torch.empty(n_elem, dtype=torch.float16, pin_memory=True)
    else:
        a = torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{src}")
        b = torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{dst}")
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize(src)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        b.copy_(a)
    torch.cuda.synchronize(src)
    dt = (time.perf_counter() - t0) / ITERS
    del a, b
    torch.cuda.empty_cache()
    return n_elem * 2 / dt / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hwprofile.json")
    ap.add_argument("--prefill-ms-per-tok", type=float, default=None,
                    help="from benchmark/ttft_ctx_sweep.py on THIS machine")
    ap.add_argument("--prefill-ms-per-tok2", type=float, default=None)
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if n < 1:
        raise SystemExit("no CUDA devices")
    n_elem = int(MB * 2**20 // 2)
    props = torch.cuda.get_device_properties(0)

    peer, best_peer = {}, 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                ok = torch.cuda.can_device_access_peer(i, j)
                bw = bench_copy(i, j, n_elem)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] gpu{i}->gpu{j}: {e!r}")
                continue
            peer[f"{i}->{j}"] = {"can_peer": bool(ok), "gbps": round(bw, 1)}
            best_peer = max(best_peer, bw)

    host_bw = round(bench_copy(0, 0, n_elem, host=True), 1)

    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        topo = ""

    prof = {
        "gpu_name": props.name,
        "gpu_count": n,
        "card_gb": round(props.total_memory / 2**30, 1),
        # The FASTEST peer pair, since placement prefers it. Slower pairs are in `peer`.
        "peer_gpu_gbps": round(best_peer, 1),
        "host_gbps": host_bw,
        "peer": peer,
        # Left null unless supplied: inheriting another machine's prefill cost would
        # silently change what every cache hit appears to be worth.
        "prefill_ms_per_tok": args.prefill_ms_per_tok,
        "prefill_ms_per_tok2": args.prefill_ms_per_tok2,
        "topo": topo.strip().splitlines()[:12],
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(prof, open(args.out, "w"), indent=1)

    print(f"\n{props.name} x{n}, {prof['card_gb']:.0f} GB each")
    print(f"  fastest peer link : {prof['peer_gpu_gbps']:>7.1f} GB/s")
    print(f"  host (pinned)     : {prof['host_gbps']:>7.1f} GB/s")
    print(f"  peer / host ratio : {prof['peer_gpu_gbps'] / host_bw:>7.1f}x")
    slow = [k for k, v in peer.items() if v["gbps"] < 0.6 * best_peer]
    if slow:
        print(f"  slower peer pairs : {', '.join(slow)}")
        print("    Park candidates spanning fast and slow pairs mix two tiers under one")
        print("    'peer' label; keep the lists to the fast pairs or report per source.")
    if args.prefill_ms_per_tok is None:
        print("\n  prefill_ms_per_tok NOT SET. Measure it with ttft_ctx_sweep.py on this")
        print("  machine and re-run with --prefill-ms-per-tok, or every 'what a hit is")
        print("  worth' number stays unavailable rather than wrong.")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

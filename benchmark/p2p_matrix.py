#!/usr/bin/env python3
"""
Which GPU pairs can actually do peer-to-peer, and what each pair really achieves.

WHY THIS IS NOT ANSWERED BY `nvidia-smi topo -m`. topo prints the WIRING (NV4, NODE,
PHB). It does not say whether CUDA will use it. When cudaDeviceCanAccessPeer is false --
common for NODE/PHB pairs, where the path crosses a PCIe host bridge and the IOMMU may
forbid it -- PyTorch does not fail. It silently STAGES THE COPY THROUGH HOST MEMORY. The
fetch becomes GPU -> host -> GPU while the code, the logs and the telemetry all still call
it a peer-GPU hit.

That distinction decides whether "park on an idle GPU" is what the design claims. The
serving path measured 10.2 GB/s against a 52.7 GB/s peer ceiling, which is about what a
host-staged copy would look like, so it has to be checked rather than assumed.

WHAT IS MEASURED, and why both columns are needed:
  can_peer   cudaDeviceCanAccessPeer. Capability, not behaviour.
  GB/s       an actual timed copy of a KV-sized block, which is the behaviour. A pair can
             report can_peer=True and still run slowly, and a pair with can_peer=False
             still returns a number -- the number of a host round trip.

Run it with the servers DOWN; it allocates on every GPU.

사용:
  python benchmark/p2p_matrix.py
  MB=512 python benchmark/p2p_matrix.py
"""
import os
import time

import torch

MB = float(os.environ.get("MB", "256"))
ITERS = int(os.environ.get("ITERS", "10"))
WARMUP = int(os.environ.get("WARMUP", "3"))
# results/nvlink_microbench.json measured 52.7 GB/s on an NV4 pair; anything far below
# that on a pair reporting can_peer=True means the copy, not the link.
NVLINK_REF = float(os.environ.get("NVLINK_REF", "52.7"))


def bench(src, dst, n_elem):
    a = torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{src}")
    b = torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{dst}")
    for _ in range(WARMUP):
        b.copy_(a)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        b.copy_(a)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    dt = (time.perf_counter() - t0) / ITERS
    del a, b
    torch.cuda.empty_cache()
    return a_bytes(n_elem) / dt / 1e9, dt * 1e3


def a_bytes(n_elem):
    return n_elem * 2


def main():
    n = torch.cuda.device_count()
    if n < 2:
        raise SystemExit(f"need >= 2 GPUs, found {n}")
    n_elem = int(MB * 2**20 // 2)
    print(f"{n} GPUs, {MB:.0f} MB per copy, {ITERS} iters\n")

    peer = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                peer[(i, j)] = torch.cuda.can_device_access_peer(i, j)
            except Exception:  # noqa: BLE001
                peer[(i, j)] = None

    print("can_device_access_peer (CUDA capability, NOT the wiring):")
    print("      " + "".join(f"{'gpu'+str(j):>9}" for j in range(n)))
    for i in range(n):
        row = "".join(f"{'-' if i == j else str(peer[(i, j)]):>9}" for j in range(n))
        print(f"gpu{i}  {row}")

    print(f"\nmeasured device-to-device copy (GB/s):")
    print("      " + "".join(f"{'gpu'+str(j):>9}" for j in range(n)))
    got = {}
    for i in range(n):
        cells = []
        for j in range(n):
            if i == j:
                cells.append(f"{'-':>9}")
                continue
            try:
                bw, ms = bench(i, j, n_elem)
                got[(i, j)] = bw
                cells.append(f"{bw:>9.1f}")
            except Exception as e:  # noqa: BLE001
                cells.append(f"{'ERR':>9}")
                print(f"  [warn] gpu{i}->gpu{j}: {e!r}")
        print(f"gpu{i}  " + "".join(cells))

    print("\n=== reading it ===")
    staged = [(i, j) for (i, j), p in peer.items() if p is False]
    if staged:
        print(f"  {len(staged)} pair(s) have NO peer access: "
              + ", ".join(f"gpu{i}->gpu{j}" for i, j in staged))
        print("  Copies on those pairs are staged through host memory by PyTorch. Parking")
        print("  onto them is a GPU->host->GPU round trip that reports as a peer-GPU hit.")
        bws = [got[k] for k in staged if k in got]
        if bws:
            print(f"  Their measured bandwidth: {min(bws):.1f}-{max(bws):.1f} GB/s.")
    else:
        print("  Every pair reports peer access, so no fetch is being staged through host.")

    fast = [(k, v) for k, v in got.items() if peer.get(k) and v >= 0.6 * NVLINK_REF]
    slow = [(k, v) for k, v in got.items() if peer.get(k) and v < 0.6 * NVLINK_REF]
    if fast:
        print(f"\n  P2P pairs at full speed ({NVLINK_REF:.0f} GB/s class): "
              + ", ".join(f"gpu{i}<-gpu{j} {v:.0f}" for (i, j), v in sorted(fast)))
    if slow:
        print(f"  P2P pairs WELL BELOW that despite can_peer=True: "
              + ", ".join(f"gpu{i}<-gpu{j} {v:.0f}" for (i, j), v in sorted(slow)))
        print("  Capability is not throughput -- these are peer copies over PCIe, not NVLink.")

    print("\n=== park targets this implies ===")
    print("  Under PD_LAYOUT=b: P0=gpu0 D0=gpu1 P1=gpu2 D1=gpu3.")
    for p, d in ((0, 1), (2, 3)):
        v = got.get((p, d))
        print(f"  P{p // 2} on gpu{p} -> its decode gpu{d}: "
              + (f"{v:.1f} GB/s, peer={peer.get((p, d))}" if v else "not measured"))
    print("  Those are the NV4-bridged pairs. Any OTHER park target on this box crosses")
    print("  PCIe, so a candidate list spanning both mixes two very different tiers under")
    print("  one 'peer' label.")


if __name__ == "__main__":
    main()

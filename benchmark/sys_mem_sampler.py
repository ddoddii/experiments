#!/usr/bin/env python3
"""
System memory timeline sampler: GPU HBM (nvidia-smi) + host DRAM (/proc/meminfo),
sampled every --interval seconds to a CSV, so we can show how much GPU HBM vs host
DRAM each arm (park vs hicache) actually uses over the run.

Why bytes, not KV%: the point of park-vs-hicache is *where* the KV tier lives --
park spends GPU HBM (the park pool), hicache spends host DRAM (the offload tier).
This captures the real footprint of each, on a wall-clock timeline.

CSV columns:
  t_unix, elapsed_s, gpu_used_mb, gpu0_mb, gpu1_mb, gpu2_mb, gpu3_mb,
  host_used_mb, host_cached_mb
(gpu_used_mb = sum over visible GPUs; host_used = MemTotal-MemAvailable; host_cached
 = Cached+Buffers, since a file-backed hicache tier lands in page cache.)

사용:
  python benchmark/sys_mem_sampler.py --out results/perturn/mem_park.csv --interval 1 &
  SAMP=$!;  ...run benchmark...;  kill $SAMP
"""
import argparse
import subprocess
import time


def gpu_used_mb():
    """Per-GPU used MiB via nvidia-smi; returns (total, [per-gpu...]) or (None, [])."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        vals = [int(x) for x in out.splitlines() if x.strip()]
        return sum(vals), vals
    except Exception:  # noqa: BLE001
        return None, []


def host_meminfo_mb():
    """(used_mb, cached_mb) from /proc/meminfo. used = MemTotal-MemAvailable."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = int(rest.split()[0])  # kB
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        cached = info.get("Cached", 0) + info.get("Buffers", 0)
        return (total - avail) / 1024.0, cached / 1024.0
    except Exception:  # noqa: BLE001
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0.0, help="max seconds (0=until killed)")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.out, "w") as f:
        f.write("t_unix,elapsed_s,gpu_used_mb,gpu0_mb,gpu1_mb,gpu2_mb,gpu3_mb,"
                "host_used_mb,host_cached_mb\n")
        f.flush()
        while True:
            now = time.time()
            elapsed = now - t0
            g_tot, g = gpu_used_mb()
            g = (g + [None, None, None, None])[:4]
            h_used, h_cached = host_meminfo_mb()
            row = [f"{now:.1f}", f"{elapsed:.1f}",
                   "" if g_tot is None else str(g_tot),
                   *["" if x is None else str(x) for x in g],
                   "" if h_used is None else f"{h_used:.0f}",
                   "" if h_cached is None else f"{h_cached:.0f}"]
            f.write(",".join(row) + "\n")
            f.flush()
            if args.duration and elapsed >= args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

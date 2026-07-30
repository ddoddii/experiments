#!/usr/bin/env python3
"""
Sample WHERE the parked KV physically lives, over time.

This produces the evidence for the placement claim. "Idle GPU HBM first, CPU DRAM only
on overflow" is a policy assertion until you can show, second by second, what fraction
of the reusable KV actually sat in GPU HBM versus in host memory -- and that the host
share only rises when the GPUs had no room.

Reads the telemetry each prefill process publishes to
/dev/shm/sglang_kv_parking/parked_gpu*.json (see _publish_parked_bytes in
idle_kv_parking.py) and appends one CSV row per tick. Run it in the background for the
whole experiment; it is read-only with respect to the serving stack.

Note that the per-GPU numbers are the park POOL occupancy on that GPU, which is what the
placement policy fills; host bytes are the live pinned allocations of the CPU DRAM
overflow tier. Both are reported in the same unit (bytes of KV) so their ratio is
meaningful.

사용:
  python benchmark/park_location_sampler.py --out results/agent/parked_park.csv &
  ...run the benchmark...
  kill %1
  python benchmark/plot_park_location.py --csv results/agent/parked_park.csv
"""
import argparse
import csv
import glob
import json
import os
import time

DEFAULT_DIR = os.environ.get("SGLANG_KV_PARK_DIR", "/dev/shm/sglang_kv_parking")


def read_all(park_dir, stale_s):
    """Merge every publisher's latest file. Returns (per_gpu_bytes, host_bytes, meta).

    Stale files are skipped rather than trusted: a crashed or restarted prefill would
    otherwise keep contributing its last-known occupancy forever and inflate the GPU
    share -- exactly the direction that would flatter the result."""
    now = time.time()
    # SUMMED across publishers: every field below is either a per-process byte count or a
    # per-process cumulative counter, so the node-level value is the sum. bytes_per_token
    # is the exception (identical for every publisher) and is overwritten, not added.
    SUM = ("host_blocks", "host_flushes", "host_evicted", "host_peak_bytes",
           "serving_bytes", "dropped_bytes", "host_evicted_bytes",
           "fetch_hits", "fetch_peer_hits", "fetch_host_hits", "fetch_local_hits",
           "fetch_miss", "fetch_already", "fetch_nospace",
           "fetched_tokens", "parked_tokens")
    per_gpu, host = {}, 0
    meta = {"writers": 0, "stale": 0, "bytes_per_token": 0}
    meta.update({k: 0 for k in SUM})
    # parked bytes split by whether the target GPU is the publisher's own serving GPU
    meta["park_local_bytes"] = 0
    meta["park_peer_bytes"] = 0
    for path in sorted(glob.glob(os.path.join(park_dir, "parked_gpu*.json"))):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:  # noqa: BLE001 (mid-replace or truncated) -> skip this tick
            continue
        if stale_s > 0 and now - float(d.get("ts", 0)) > stale_s:
            meta["stale"] += 1
            continue
        meta["writers"] += 1
        writer = int(d.get("writer_gpu", -1))
        for g, b in (d.get("gpu_bytes") or {}).items():
            # last writer wins per target GPU: each park pool has exactly one owner
            per_gpu[int(g)] = int(b)
            key = "park_local_bytes" if int(g) == writer else "park_peer_bytes"
            meta[key] += int(b)
        host += int(d.get("host_bytes") or 0)
        for k in SUM:
            meta[k] += int(d.get(k) or 0)
        meta["bytes_per_token"] = int(d.get("bytes_per_token") or 0)
    return per_gpu, host, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--gpus", nargs="*", type=int, default=None,
                    help="GPU columns to emit; default = whatever appears in the files")
    ap.add_argument("--stale-s", type=float, default=5.0,
                    help="ignore a publisher's file older than this (0 = never)")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until killed")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.isdir(args.dir):
        print(f"[warn] {args.dir} does not exist yet -- start the servers with "
              f"IDLE_KV_PARKING=1 first. Sampling anyway (files may appear later).")

    # Fix the GPU columns up front so the CSV has a stable header even though pools are
    # discovered lazily.
    gpus = args.gpus
    if gpus is None:
        found, _, _ = read_all(args.dir, args.stale_s)
        gpus = sorted(found) or [0, 1, 2, 3]
    cols = (["elapsed_s"] + [f"gpu{g}_gb" for g in gpus]
            + ["gpu_total_gb", "host_gb", "total_gb", "host_frac",
               "host_peak_gb", "host_blocks", "host_flushes", "host_evicted",
               # residency breakdown: serving (local radix) / park local / park peer /
               # host / dropped. The first four are instantaneous, dropped_* are
               # cumulative -- KV that no longer exists anywhere and must be re-prefilled.
               "serving_gb", "park_local_gb", "park_peer_gb",
               "dropped_gb", "host_evicted_gb", "peer_frac_of_parked",
               # fetch source split: the read-back counterpart of the residency split
               "fetch_hits", "fetch_local_hits", "fetch_peer_hits", "fetch_host_hits",
               "fetch_miss", "fetch_already", "fetch_nospace",
               "fetched_tokens", "parked_tokens",
               "writers", "stale"])

    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        print(f"[park-loc] -> {args.out}  gpus={gpus}  dir={args.dir}  Ctrl-C to stop")
        while True:
            now = time.time()
            per_gpu, host, meta = read_all(args.dir, args.stale_s)
            gvals = [per_gpu.get(g, 0) / 1e9 for g in gpus]
            gtot = sum(gvals)
            hgb = host / 1e9
            tot = gtot + hgb
            gb = lambda k: round(meta[k] / 1e9, 4)  # noqa: E731
            parked = meta["park_local_bytes"] + meta["park_peer_bytes"]
            w.writerow([round(now - t0, 1)] + [round(v, 4) for v in gvals]
                       + [round(gtot, 4), round(hgb, 4), round(tot, 4),
                          round(hgb / tot, 4) if tot > 0 else 0.0,
                          gb("host_peak_bytes"),
                          meta["host_blocks"], meta["host_flushes"],
                          meta["host_evicted"],
                          gb("serving_bytes"), gb("park_local_bytes"),
                          gb("park_peer_bytes"), gb("dropped_bytes"),
                          gb("host_evicted_bytes"),
                          round(meta["park_peer_bytes"] / parked, 4) if parked else 0.0,
                          meta["fetch_hits"], meta["fetch_local_hits"],
                          meta["fetch_peer_hits"], meta["fetch_host_hits"],
                          meta["fetch_miss"], meta["fetch_already"],
                          meta["fetch_nospace"], meta["fetched_tokens"],
                          meta["parked_tokens"],
                          meta["writers"], meta["stale"]])
            fh.flush()
            if args.duration and now - t0 >= args.duration:
                break
            time.sleep(max(0.05, args.interval - (time.time() - now)))


if __name__ == "__main__":
    main()

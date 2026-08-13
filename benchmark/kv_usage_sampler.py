#!/usr/bin/env python3
"""
Sample each node's KV-POOL UTILISATION over time, per role.

WHY THIS EXISTS RATHER THAN REUSING sys_mem_breakdown.py
  That sampler reads nvidia-smi, which reports what the process has ALLOCATED. SGLang
  reserves its whole KV pool at startup from --mem-fraction-static, so an nvidia-smi
  line is flat from the first second no matter how much KV is actually resident -- it
  can never show a cache filling up. The "prefill fills to 100% over the first few
  minutes" picture is pool UTILISATION, which only the server itself knows.

WHAT IS RECORDED, and why more than one number
  occupancy = (max_total - available) / max_total, i.e. every slot holding KV, whether
  it is pinned to a running request or is evictable prefix cache. That is the quantity
  that fills up.

  token_usage is NOT that: SGLang defines it as (max_total - available - evictable) /
  max_total, so a pool that is 100% full of reusable prefix cache and a pool that is
  genuinely empty both report ~0. Recording both, plus evictable_tokens, keeps the two
  distinguishable -- the same confusion probe_cache_metric.sh was written to settle.

사용:
  python benchmark/kv_usage_sampler.py --out results/agent_trace/kv_park_c4.csv \
      --interval 2 &
  ...run the benchmark...
  kill %1
"""
import argparse
import csv
import os
import time
import urllib.request

KEYS = ("sglang:max_total_num_tokens", "sglang:num_used_tokens",
        "sglang:evictable_tokens", "sglang:token_usage",
        "sglang:cache_occupancy", "sglang:cache_hit_rate")
DEFAULT_NODES = "prefill:30000,prefill:30001,decode:30002,decode:30003"


def scrape(port):
    vals = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001  server down / restarting -> row of blanks
        return vals
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        key = parts[0].split("{", 1)[0]
        if key in KEYS:
            try:
                vals[key] = float(parts[1])
            except ValueError:
                pass
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--nodes", default=DEFAULT_NODES,
                     help="role:port pairs, comma separated")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until killed")
    args = ap.parse_args()

    nodes = []
    for spec in args.nodes.split(","):
        role, _, port = spec.partition(":")
        if port:
            nodes.append((role.strip(), int(port)))
    if not nodes:
        raise SystemExit(f"could not parse --nodes {args.nodes!r}")

    cols = ["elapsed_s"]
    for role, port in nodes:
        cols += [f"{role}{port}_{k.split(':', 1)[1]}" for k in KEYS]
        cols.append(f"{role}{port}_occupancy_frac")
    # Role aggregates: the figure asks "how full are the PREFILL GPUs", and summing the
    # pools is the honest way to answer it -- averaging the per-node fractions would
    # weight a small pool the same as a large one.
    for role in dict.fromkeys(r for r, _ in nodes):
        cols += [f"{role}_used_tokens", f"{role}_max_tokens", f"{role}_occupancy_frac"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        print(f"[kv-usage] -> {args.out}  nodes={nodes}  Ctrl-C to stop", flush=True)
        while True:
            now = time.time()
            row = [round(now - t0, 1)]
            agg = {}
            for role, port in nodes:
                v = scrape(port)
                row += [v.get(k, "") for k in KEYS]
                mx = v.get("sglang:max_total_num_tokens")
                used = None
                if mx:
                    # Prefer the server's own gauge when the patched build exports it;
                    # otherwise reconstruct from used + evictable, which is the same
                    # quantity as long as both are present.
                    if v.get("sglang:cache_occupancy") is not None:
                        frac = v["sglang:cache_occupancy"]
                        used = frac * mx
                    else:
                        used = (v.get("sglang:num_used_tokens") or 0) + \
                               (v.get("sglang:evictable_tokens") or 0)
                        frac = used / mx
                    row.append(round(frac, 4))
                    a = agg.setdefault(role, [0.0, 0.0])
                    a[0] += used
                    a[1] += mx
                else:
                    row.append("")
            for role in dict.fromkeys(r for r, _ in nodes):
                u, m = agg.get(role, (0.0, 0.0))
                row += [round(u), round(m), round(u / m, 4) if m else ""]
            w.writerow(row)
            fh.flush()
            if args.duration and now - t0 >= args.duration:
                break
            time.sleep(max(0.05, args.interval - (time.time() - now)))


if __name__ == "__main__":
    main()

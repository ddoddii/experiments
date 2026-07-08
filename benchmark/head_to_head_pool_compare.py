#!/usr/bin/env python3
"""
Cross-pool comparison for the head-to-head pressure sweep.

Reads each pool's head_to_head_summary.json (written by head_to_head_analyze.py)
and tabulates radix/hicache/park TTFT + reuse across P-GPU pool sizes, so we can
see whether relaxing pressure gives fetch-on-hit room to actually beat recompute
(park TTFT < radix TTFT) before the reuse gap to hicache closes.

사용:
  python benchmark/head_to_head_pool_compare.py --pools "40000 60000 80000 120000" --c 8 --delay 3
"""
import argparse
import json
import os


def load(pool, c, delay):
    tag = f"h2h_p{pool}_c{c}_d{delay}"
    path = f"results/head_to_head/{tag}/head_to_head_summary.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def g(d, arm, key):
    try:
        return d["arms"][arm].get(key)
    except (KeyError, TypeError, AttributeError):
        return None


def f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else ("-" if v is None else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--c", default="8")
    ap.add_argument("--delay", default="3")
    args = ap.parse_args()

    pools = args.pools.split()
    print(f"\nCross-pool head-to-head  (C={args.c}, delay={args.delay}s)")
    header = (f"{'pool':>8} | {'radix TTFT':>10} {'reuse':>6} | "
              f"{'hicache TTFT':>12} {'reuse':>6} | {'park TTFT':>9} {'reuse':>6} | "
              f"{'park vs radix':>13} {'park vs hi':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for pool in pools:
        d = load(pool, args.c, args.delay)
        if d is None:
            print(f"{pool:>8} | (no summary — run pool {pool} first)")
            continue
        r_ttft, r_reuse = g(d, "radix", "avg_ttft_s"), g(d, "radix", "reuse_ratio")
        h_ttft, h_reuse = g(d, "hicache", "avg_ttft_s"), g(d, "hicache", "reuse_ratio")
        p_ttft, p_reuse = g(d, "park", "avg_ttft_s"), g(d, "park", "reuse_ratio")
        pv_radix = d.get("park_vs_radix_ttft_pct")
        pv_hi = d.get("park_vs_hicache_ttft_pct")
        print(f"{pool:>8} | {f(r_ttft):>10} {f(r_reuse,3):>6} | "
              f"{f(h_ttft):>12} {f(h_reuse,3):>6} | {f(p_ttft):>9} {f(p_reuse,3):>6} | "
              f"{f(pv_radix,1):>12}% {f(pv_hi,1):>9}%")
        rows.append({"pool": pool, "radix_ttft": r_ttft, "radix_reuse": r_reuse,
                     "hicache_ttft": h_ttft, "hicache_reuse": h_reuse,
                     "park_ttft": p_ttft, "park_reuse": p_reuse,
                     "park_vs_radix_pct": pv_radix, "park_vs_hicache_pct": pv_hi})

    print("\n해석 축:")
    print("  - park vs radix < 0  → fetch가 recompute를 이긴다 (순 TTFT 이득)")
    print("  - park reuse가 radix→hicache 사이 어디쯤인지 → fetch가 회수하는 reuse 비율")
    print("  - pool↑ 하면 radix reuse도 오르며 hicache와 격차가 닫힘 → park이 되찾을 여지도 감소")

    out = "results/head_to_head/pool_sweep_summary.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

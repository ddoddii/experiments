#!/usr/bin/env python3
"""
Phase 2 slice-2 validation figure: park pool capacity -> survival -> TTFT.

Reads run_2p2d_park_sweep.sh's CSV (arm,pool,rep,ttft,tput,wall,host_mb,survival) and
draws the paper's core figure:
  (left)  park pool size -> avg TTFT, with the hicache reference line. TTFT drops as the
          pool grows and meets hicache once the hot working set fits.
  (right) survival -> TTFT scatter (the causal story): higher park survival -> lower TTFT
          -> converges to hicache. Points colored by pool.
Repeats shown as mean + min/max band (left) and individual points (right).

사용: python benchmark/plot_park_sweep.py --csv results/park_sweep/sweep_summary.csv
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_PARK = "#E69F00"
C_HICACHE = "#009E73"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
# Okabe-Ito sequential-ish for pools (light->dark)
POOL_COLORS = ["#56B4E9", "#0072B2", "#E69F00", "#D55E00", "#CC79A7"]


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            def fnum(k):
                v = r.get(k, "")
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            rows.append({"arm": r["arm"], "pool": int(r["pool"]), "rep": r.get("rep"),
                         "ttft": fnum("ttft"), "tput": fnum("tput"),
                         "host": fnum("host_mb"), "surv": fnum("survival")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load(args.csv)

    park = [r for r in rows if r["arm"] == "park" and r["ttft"] is not None]
    hic = [r for r in rows if r["arm"] == "hicache" and r["ttft"] is not None]
    hic_ttft = sum(r["ttft"] for r in hic) / len(hic) if hic else None

    by_pool = defaultdict(list)
    for r in park:
        by_pool[r["pool"]].append(r)
    pools = sorted(by_pool)

    print(f"\n=== park pool sweep ({os.path.basename(args.csv)}) ===")
    if hic_ttft:
        hh = sum(r["host"] for r in hic if r["host"]) / max(1, sum(1 for r in hic if r["host"]))
        print(f"  hicache: TTFT={hic_ttft:.3f}s  host~{hh:.0f}MB  (n={len(hic)})")
    for p in pools:
        rs = by_pool[p]
        t = [r["ttft"] for r in rs]
        s = [r["surv"] for r in rs if r["surv"] is not None]
        h = [r["host"] for r in rs if r["host"] is not None]
        print(f"  pool {p:>7}: TTFT={sum(t)/len(t):.3f}s (n={len(t)})  "
              f"survival~{(sum(s)/len(s)) if s else float('nan'):.0f}%  host~{(sum(h)/len(h)) if h else float('nan'):.0f}MB")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Park capacity → survival → TTFT (2P2D, radix+park, host-RAM-free)",
                 fontsize=13, fontweight="bold")

    # left: pool -> TTFT (mean + min/max band) with hicache reference
    xs, means, los, his = [], [], [], []
    for p in pools:
        t = [r["ttft"] for r in by_pool[p]]
        xs.append(p / 1000.0); means.append(sum(t) / len(t)); los.append(min(t)); his.append(max(t))
    axL.plot(xs, means, "-o", color=C_PARK, lw=2, ms=7, label="park (radix+park)", zorder=3)
    axL.fill_between(xs, los, his, color=C_PARK, alpha=0.18, zorder=2)
    if hic_ttft:
        axL.axhline(hic_ttft, color=C_HICACHE, lw=2, ls="--", label=f"hicache ref ({hic_ttft:.2f}s)")
    axL.set_xlabel("park pool size (k tokens)"); axL.set_ylabel("avg TTFT (s, lower better)")
    axL.set_title("bigger pool → TTFT meets hicache", fontsize=10.5)
    axL.legend(frameon=False, fontsize=9, loc="upper right")

    # right: survival -> TTFT scatter, colored by pool
    for i, p in enumerate(pools):
        rs = [r for r in by_pool[p] if r["surv"] is not None]
        if not rs:
            continue
        axR.scatter([r["surv"] for r in rs], [r["ttft"] for r in rs],
                    color=POOL_COLORS[i % len(POOL_COLORS)], s=70, zorder=3,
                    edgecolor="white", linewidth=0.8, label=f"{p//1000}k")
    if hic_ttft:
        axR.axhline(hic_ttft, color=C_HICACHE, lw=2, ls="--")
        axR.text(axR.get_xlim()[0], hic_ttft, " hicache", color=C_HICACHE, fontsize=8, va="bottom")
    axR.set_xlabel("park survival (%)"); axR.set_ylabel("avg TTFT (s)")
    axR.set_title("higher survival → lower TTFT (causal)", fontsize=10.5)
    axR.legend(frameon=False, fontsize=8, title="pool", loc="upper right")

    for ax in (axL, axR):
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    out = args.out or os.path.join(os.path.dirname(args.csv), "park_sweep_curve.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

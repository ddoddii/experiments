#!/usr/bin/env python3
"""
GPU HBM vs host DRAM usage over the run, park vs hicache. Two panels:
  (left)  GPU HBM used (GB) over elapsed time  -- park spends HBM (the park pool)
  (right) host DRAM used (GB) over elapsed time -- hicache spends DRAM (offload tier)

Reads sys_mem_sampler.py CSVs (one per arm). Shows the park<->hicache tradeoff as a
timeline: park's line sits high on HBM / flat on DRAM; hicache's sits flat on HBM /
rising on DRAM. Accepts multiple CSVs per arm (reps) -- drawn as thin faded lines with
the mean overlaid.

사용:
  python benchmark/plot_mem_timeline.py \
    --park results/perturn/mem_park_r*.csv --hicache results/perturn/mem_hicache_r*.csv \
    --out results/perturn/mem_timeline.png
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_HICACHE = "#2a78d6"   # blue
C_PARK = "#1baf7a"      # aqua
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load(paths, col):
    """Return list of (elapsed[], value_gb[]) series, one per file, for the given column."""
    series = []
    for p in paths:
        xs, ys = [], []
        try:
            with open(p) as f:
                for r in csv.DictReader(f):
                    e = r.get("elapsed_s"); v = r.get(col)
                    if not e or not v:
                        continue
                    xs.append(float(e)); ys.append(float(v) / 1024.0)  # MB -> GB
        except Exception as ex:  # noqa: BLE001
            print(f"[warn] skip {p}: {ex}")
            continue
        if xs:
            series.append((xs, ys))
    return series


def mean_series(series, dt=2.0):
    """Resample each series onto a common grid and average (for the bold mean line)."""
    if not series:
        return [], []
    tmax = max(xs[-1] for xs, _ in series)
    grid = [i * dt for i in range(int(tmax / dt) + 1)]
    means = []
    for g in grid:
        vals = []
        for xs, ys in series:
            # nearest sample at or before g
            v = None
            for x, y in zip(xs, ys):
                if x <= g:
                    v = y
                else:
                    break
            if v is not None:
                vals.append(v)
        means.append(sum(vals) / len(vals) if vals else None)
    gx = [g for g, m in zip(grid, means) if m is not None]
    gy = [m for m in means if m is not None]
    return gx, gy


def draw(ax, park, hic, ylabel, title):
    for series, color, label in ((hic, C_HICACHE, "hicache"),
                                 (park, C_PARK, "park (slab, host-RAM-free)")):
        for xs, ys in series:  # faint per-rep lines
            ax.plot(xs, ys, color=color, lw=0.9, alpha=0.28, zorder=2)
        mx, my = mean_series(series)
        if mx:
            ax.plot(mx, my, color=color, lw=2.4, zorder=3, label=label)
    ax.set_xlabel("elapsed time (s)", color=SECOND, fontsize=10.5)
    ax.set_ylabel(ylabel, color=SECOND, fontsize=10.5)
    ax.set_title(title, fontsize=11, color=INK, fontweight="bold")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="best")


def peak(series):
    return max((max(ys) for _, ys in series), default=float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park", nargs="+", required=True)
    ap.add_argument("--hicache", nargs="+", required=True)
    ap.add_argument("--out", default="results/perturn/mem_timeline.png")
    args = ap.parse_args()

    park_gpu = load(args.park, "gpu_used_mb")
    hic_gpu = load(args.hicache, "gpu_used_mb")
    park_host = load(args.park, "host_used_mb")
    hic_host = load(args.hicache, "host_used_mb")

    print("=== peak usage (GB) ===")
    print(f"  GPU HBM   park={peak(park_gpu):.1f}   hicache={peak(hic_gpu):.1f}")
    print(f"  host DRAM park={peak(park_host):.1f}  hicache={peak(hic_host):.1f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Memory footprint over the run: GPU HBM vs host DRAM  (2P2D, BFCL multi-turn)",
                 fontsize=13, fontweight="bold", color=INK)
    draw(axL, park_gpu, hic_gpu, "GPU HBM used (GB, all 4 GPUs)", "GPU HBM  (park spends this)")
    draw(axR, park_host, hic_host, "host DRAM used (GB)", "host DRAM  (hicache spends this)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

C_HICACHE = PALETTE["hicache"]
C_PARK = PALETTE["park"]
C_BOTH = PALETTE["both"]
INK, MUTED = PALETTE["ink"], PALETTE["muted"]


def _rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception as ex:  # noqa: BLE001
        print(f"[warn] skip {path}: {ex}")
        return []


def load(paths, col):
    """Return list of (elapsed[], value_gb[]) series, one per file, for the given column."""
    series = []
    for p in paths:
        xs, ys = [], []
        for r in _rows(p):
            e, v = r.get("elapsed_s"), r.get(col)
            if not e or not v:
                continue
            xs.append(float(e)); ys.append(float(v) / 1024.0)  # MB -> GB
        if xs:
            series.append((xs, ys))
    return series


def load_sum(paths, cols):
    """Like load() but sums several columns per row (e.g. used + cached)."""
    series = []
    for p in paths:
        xs, ys = [], []
        for r in _rows(p):
            e = r.get("elapsed_s")
            vals = [r.get(c) for c in cols]
            if not e or any(v in (None, "") for v in vals):
                continue
            xs.append(float(e)); ys.append(sum(float(v) for v in vals) / 1024.0)
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


def draw(ax, arms, ylabel, title):
    """arms = list of (series, color, label, key), drawn back-to-front."""
    ymax = 0.0
    for series, color, label, key in arms:
        for xs, ys in series:  # faint per-rep lines
            ax.plot(xs, ys, color=color, lw=0.6, alpha=0.25, zorder=2)
        mx, my = mean_series(series)
        if not mx:
            continue
        ax.plot(mx, my, color=color, lw=1.5, ls=STYLE[key]["ls"], zorder=3, label=label)
        pk = peak(series)
        ymax = max(ymax, pk)
        # direct value label at the right end of the mean line (peak GB)
        ax.annotate(f"{pk:.0f} GB", xy=(mx[-1], my[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    color=color, fontsize=7.5, fontweight="bold")
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, ymax * 1.18 if ymax else None)  # headroom so top line + label fit
    x1 = max((xs[-1] for series, *_ in arms for xs, _ in series), default=1)
    ax.set_xlim(0, x1 * 1.13)
    style_axes(ax)
    ax.legend(loc="center left")


def peak(series):
    return max((max(ys) for _, ys in series), default=float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park", nargs="+", required=True)
    ap.add_argument("--hicache", nargs="+", required=True)
    ap.add_argument("--both", nargs="+", default=[], help="optional coexistence arm (hicache+victim)")
    ap.add_argument("--out", default="results/perturn/mem_timeline.png")
    args = ap.parse_args()

    # host footprint = used + page cache. hicache's file backend lands its KV in page
    # cache, so "used" alone understates its host cost -- sum for the honest picture.
    HOST = ["host_used_mb", "host_cached_mb"]
    gpu = {"hic": load(args.hicache, "gpu_used_mb"), "park": load(args.park, "gpu_used_mb")}
    host = {"hic": load_sum(args.hicache, HOST), "park": load_sum(args.park, HOST)}
    if args.both:
        gpu["both"] = load(args.both, "gpu_used_mb")
        host["both"] = load_sum(args.both, HOST)

    print("=== peak usage (GB) ===")
    for k in ("hic", "park", "both"):
        if k in gpu:
            print(f"  {k:5} GPU={peak(gpu[k]):.1f}  host(used+cache)={peak(host[k]):.1f}")

    # arm draw order: hicache (baseline) first, then park, then coexistence on top
    def arms(d):
        out = [(d["hic"], C_HICACHE, "HiCache (host tier)", "hicache"),
               (d["park"], C_PARK, "KV victim cache (GPU, host-free)", "park")]
        if "both" in d:
            out.append((d["both"], C_BOTH, "HiCache + victim (layered)", "both"))
        return out

    use_paper_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    draw(axL, arms(gpu), "GPU HBM used (GB, 4 GPUs)", "(a) GPU HBM footprint")
    draw(axR, arms(host), "host RAM used + page cache (GB)", "(b) host RAM footprint")
    fig.tight_layout()
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

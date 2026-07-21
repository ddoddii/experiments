#!/usr/bin/env python3
"""
Combined cross-workload figure: BFCL (tool-call) and ShareGPT (long-form) in one.
2 rows (workloads) x 3 columns:
  (col 1) per-turn TTFT vs context length         -- smoothed (sliding-window median)
  (col 2) per-turn effective throughput vs context -- smoothed
  (col 3) host RAM footprint (used + page cache)   -- SGLang vs KV Victim Cache bars
Arms: SGLang (HiCache baseline) vs KV Victim Cache (ours). One boxed legend below.

Shows the parity (TTFT/throughput) + the host-RAM win holding across two different
agentic workloads. Each row keeps its own axes (BFCL and ShareGPT differ in scale).

사용:
  python benchmark/plot_perturn_combined.py \
    --bfcl-park results/perturn_park_r*.json --bfcl-hic results/perturn_hicache_r*.json \
    --bfcl-mem-park results/perturn/mem_park_r*.csv --bfcl-mem-hic results/perturn/mem_hicache_r*.csv \
    --sg-park results/perturn_sharegpt_park_r*.json --sg-hic results/perturn_sharegpt_hicache_r*.json \
    --sg-mem-park results/perturn_sharegpt/mem_park_r*.csv --sg-mem-hic results/perturn_sharegpt/mem_hicache_r*.csv \
    --out results/fig_combined
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig
from plot_perturn_ctxlen import load_points, sliding
from plot_mem_timeline import load_sum, peak, mean_series

C_HIC, C_PARK = PALETTE["hicache"], PALETTE["park"]
INK, MUTED = PALETTE["ink"], PALETTE["muted"]
L_HIC, L_PARK = "SGLang", "KV Victim Cache"


def line_panel(ax, park_pts, hic_pts, key, win, ylabel, title):
    for pts, col, lab, sty in ((hic_pts, C_HIC, L_HIC, "hicache"),
                               (park_pts, C_PARK, L_PARK, "park")):
        rows = sliding(pts, key, win, max(1, win // 6), 3)
        if not rows:
            continue
        xs = [r[0] / 1000.0 for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, color=col, lw=1.3, ls=STYLE[sty]["ls"], marker=STYLE[sty]["marker"],
                ms=3, label=lab, zorder=3, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel("context length (k tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)


def host_panel(ax, mem_park, mem_hic, title):
    """Host RAM (used + page cache) over elapsed time -- same style as fig_mem's
    panel (b): faint per-rep lines + bold mean line + a direct GB label at the end."""
    HOST = ["host_used_mb", "host_cached_mb"]
    ymax = 0.0
    for mem_paths, col, lab, sty in ((mem_hic, C_HIC, L_HIC, "hicache"),
                                     (mem_park, C_PARK, L_PARK, "park")):
        series = load_sum(mem_paths, HOST)
        for xs, ys in series:
            ax.plot(xs, ys, color=col, lw=0.5, alpha=0.25, zorder=2)
        mx, my = mean_series(series)
        if not mx:
            continue
        ax.plot(mx, my, color=col, lw=1.3, ls=STYLE[sty]["ls"], zorder=3, label=lab)
        pk = peak(series)
        ymax = max(ymax, pk)
        ax.annotate(f"{pk:.0f} GB", xy=(mx[-1], my[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    color=col, fontsize=7, fontweight="bold")
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("host RAM (GB)")
    ax.set_title(title)
    ax.set_ylim(0, ymax * 1.18 if ymax else None)
    x1 = max((max(xs) for mem_paths in (mem_hic, mem_park)
              for xs, _ in load_sum(mem_paths, HOST)), default=1)
    ax.set_xlim(0, x1 * 1.16)  # headroom so the GB label doesn't clip
    style_axes(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-park", nargs="+", required=True)
    ap.add_argument("--bfcl-hic", nargs="+", required=True)
    ap.add_argument("--bfcl-mem-park", nargs="+", required=True)
    ap.add_argument("--bfcl-mem-hic", nargs="+", required=True)
    ap.add_argument("--sg-park", nargs="+", required=True)
    ap.add_argument("--sg-hic", nargs="+", required=True)
    ap.add_argument("--sg-mem-park", nargs="+", required=True)
    ap.add_argument("--sg-mem-hic", nargs="+", required=True)
    ap.add_argument("--out", default="results/fig_combined")
    args = ap.parse_args()

    bp, bh = load_points(args.bfcl_park), load_points(args.bfcl_hic)
    sp, sh = load_points(args.sg_park), load_points(args.sg_hic)

    use_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6))
    # row 0: BFCL (longer context -> wider smoothing window)
    line_panel(axes[0][0], bp, bh, "ttft", 1400, "median TTFT (s)", "TTFT (lower better)")
    line_panel(axes[0][1], bp, bh, "good", 1400, "eff. throughput (tok/s)", "Throughput (higher better)")
    host_panel(axes[0][2], args.bfcl_mem_park, args.bfcl_mem_hic, "host RAM (lower better)")
    # row 1: ShareGPT (shorter context -> narrower window)
    line_panel(axes[1][0], sp, sh, "ttft", 600, "median TTFT (s)", "")
    line_panel(axes[1][1], sp, sh, "good", 600, "eff. throughput (tok/s)", "")
    host_panel(axes[1][2], args.sg_mem_park, args.sg_mem_hic, "")

    handles, labels = axes[0][0].get_legend_handles_labels()
    # extra hspace so the (a)/(b) row captions have room between/below the rows
    fig.tight_layout(rect=[0.0, 0.09, 1, 1])
    fig.subplots_adjust(hspace=0.62)

    # (a)/(b) captions centered under each row (below its bottom edge, above the next
    # row / legend). Positions read from the actual laid-out axes so they never overlap.
    row_left = axes[0][0].get_position().x0
    row_right = axes[0][2].get_position().x1
    cap_x = (row_left + row_right) / 2
    cap_y0 = axes[0][0].get_position().y0 - 0.07
    cap_y1 = axes[1][0].get_position().y0 - 0.07
    fig.text(cap_x, cap_y0, "(a) BFCL v3", ha="center", va="top", fontsize=10.5, fontweight="bold")
    fig.text(cap_x, cap_y1, "(b) ShareGPT", ha="center", va="top", fontsize=10.5, fontweight="bold")

    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fancybox=False,
               edgecolor=MUTED, handlelength=2.6, columnspacing=2.2, bbox_to_anchor=(0.5, 0.0))
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

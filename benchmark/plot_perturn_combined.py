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
from plot_mem_timeline import load_sum, peak

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
    HOST = ["host_used_mb", "host_cached_mb"]
    ph = peak(load_sum(mem_park, HOST))
    hh = peak(load_sum(mem_hic, HOST))
    xs = [0, 1]
    ax.bar(xs, [hh, ph], width=0.62, color=[C_HIC, C_PARK],
           edgecolor="white", linewidth=0.6, zorder=3)
    for x, v in zip(xs, [hh, ph]):
        ax.text(x, v + max(hh, ph) * 0.02, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([L_HIC, L_PARK], fontsize=7)
    ax.set_ylabel("host RAM (GB)")
    ax.set_title(title)
    ax.set_ylim(0, max(hh, ph) * 1.22)
    style_axes(ax)
    ax.grid(axis="x", visible=False)


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
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 4.7))
    # row 0: BFCL (longer context -> wider smoothing window)
    line_panel(axes[0][0], bp, bh, "ttft", 1400, "median TTFT (s)", "TTFT (lower better)")
    line_panel(axes[0][1], bp, bh, "good", 1400, "eff. throughput (tok/s)", "Throughput (higher better)")
    host_panel(axes[0][2], args.bfcl_mem_park, args.bfcl_mem_hic, "host RAM (lower better)")
    # row 1: ShareGPT (shorter context -> narrower window)
    line_panel(axes[1][0], sp, sh, "ttft", 600, "median TTFT (s)", "")
    line_panel(axes[1][1], sp, sh, "good", 600, "eff. throughput (tok/s)", "")
    host_panel(axes[1][2], args.sg_mem_park, args.sg_mem_hic, "")

    # row labels on the far left
    axes[0][0].annotate("BFCL\n(tool-call)", xy=(0, 0.5), xytext=(-52, 0),
                        xycoords="axes fraction", textcoords="offset points",
                        ha="center", va="center", rotation=90, fontsize=10, fontweight="bold")
    axes[1][0].annotate("ShareGPT\n(long-form)", xy=(0, 0.5), xytext=(-52, 0),
                        xycoords="axes fraction", textcoords="offset points",
                        ha="center", va="center", rotation=90, fontsize=10, fontweight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=[0.03, 0.07, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fancybox=False,
               edgecolor=MUTED, handlelength=2.6, columnspacing=2.2, bbox_to_anchor=(0.5, 0.0))
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

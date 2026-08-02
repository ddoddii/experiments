#!/usr/bin/env python3
"""
Open-loop sweep figure: (a) delivered throughput vs offered rate, (b) TTFT vs offered
rate, (c) the latency-throughput curve the two together imply.

Panel (c) is the one that carries the claim. Plotting TTFT against DELIVERED THROUGHPUT
rather than against offered rate removes the x-axis the arms do not share -- an offered
rate one arm sustains and another does not is not a common operating point -- and leaves
the question a reader actually has: at a given latency budget, how much load does each
arm carry? The curve bends up at capacity, so the rightmost point of each arm at an
acceptable TTFT is its answer.

사용:
  python benchmark/plot_qps.py results/qps/qps_llama8b/qps.json --out .../fig_qps
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

ARMS = [
    ("recompute", "Recompute", "#8A9199", "recompute"),
    ("hicache",   "SGLang",    PALETTE["hicache"], "hicache"),
    ("hicache_memfrac", "SGLang (equal mem)", PALETTE["both"], "both"),
    ("park",      "Ours",      PALETTE["park"],    "park"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qps_json")
    ap.add_argument("--out", default="results/qps/fig_qps")
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=2.1)
    ap.add_argument("--ttft", default="ttft_p50_s", choices=["ttft_p50_s", "ttft_p95_s"])
    args = ap.parse_args()

    with open(args.qps_json) as fh:
        data = json.load(fh)["arms"]

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height))

    for key, label, color, skey in ARMS:
        rows = data.get(key)
        if not rows:
            continue
        st = STYLE[skey]
        # Points past saturation are kept but marked hollow: they are real measurements
        # of a server that could not keep up, and hiding them would hide the knee, which
        # is the result. Reading them as latency would be the mistake, not plotting them.
        def series(xk, yk):
            return ([r[xk] for r in rows if r.get(yk) is not None and r.get(xk) is not None],
                    [r[yk] for r in rows if r.get(yk) is not None and r.get(xk) is not None])

        x, y = series("rate", "throughput_tok_s")
        axes[0].plot(x, y, color=color, lw=1.5, ls=st["ls"], marker=st["marker"], ms=3.6,
                     label=label, zorder=3, markeredgecolor="white", markeredgewidth=0.5)

        x, y = series("rate", args.ttft)
        axes[1].plot(x, y, color=color, lw=1.5, ls=st["ls"], marker=st["marker"], ms=3.6,
                     zorder=3, markeredgecolor="white", markeredgewidth=0.5)

        pts = [r for r in rows
               if r.get("throughput_tok_s") is not None and r.get(args.ttft) is not None]
        axes[2].plot([r["throughput_tok_s"] for r in pts], [r[args.ttft] for r in pts],
                     color=color, lw=1.5, ls=st["ls"], marker=st["marker"], ms=3.6,
                     zorder=3, markeredgecolor="white", markeredgewidth=0.5)
        sat = [r for r in pts if r.get("past_saturation")]
        if sat:
            axes[2].plot([r["throughput_tok_s"] for r in sat], [r[args.ttft] for r in sat],
                         ls="none", marker=st["marker"], ms=5.5, mfc="none",
                         markeredgecolor=color, markeredgewidth=1.0, zorder=4)

    tl = "median" if args.ttft == "ttft_p50_s" else "P95"
    for ax, xlab, ylab, title in (
        (axes[0], "offered rate (sessions/s)", "delivered tok/s", "(a) Throughput"),
        (axes[1], "offered rate (sessions/s)", f"{tl} TTFT (s)", "(b) Latency"),
        (axes[2], "delivered tok/s", f"{tl} TTFT (s)", "(c) Latency vs. load"),
    ):
        ax.set_xlabel(xlab, fontsize=7)
        ax.set_ylabel(ylab, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(bottom=0)
        style_axes(ax)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.04), columnspacing=1.8, handlelength=2.0)
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=[0, 0, 1, 0.9])
    savefig(fig, args.out)
    print("hollow markers in (c) = past saturation: the server did not finish the "
          "offered load there, so that TTFT is queueing delay.")


if __name__ == "__main__":
    main()

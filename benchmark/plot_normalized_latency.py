#!/usr/bin/env python3
"""
Normalized latency (ms/token) vs delivered throughput (tok/s) -- the
FlashGen/CachedAttention-style "latency vs. load" figure: x = throughput, so points from
DIFFERENT offered rates land at their actual delivered load rather than the nominal one,
and the curve bends up sharply once an arm nears its capacity.

Normalized latency = per-SESSION serving time (sum of each turn's own request -> last
token span; TOOL_DELAY think-time between turns is excluded) divided by tokens generated
in that session. This differs from job_delay_s (which the JPS-vs-delay figure plots):
job delay includes the think-time gaps because a scheduler's job clock runs through them,
while normalized latency is meant to read as pure serving cost per token, comparable
across arms and workloads whose output lengths differ.

사용:
  python benchmark/plot_normalized_latency.py results/qps/qps_llama8b_bfcl/qps.json \
    --out results/qps/qps_llama8b_bfcl/fig_norm_latency
  python benchmark/plot_normalized_latency.py .../qps.json --stat p95 --ymax 200
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

ARMS = [
    ("recompute", "Recompute", "#8A9199", "recompute"),
    ("radix",     "Radix",     "#B0B6BD", "recompute"),
    ("hicache",   "SGLang",    PALETTE["hicache"], "hicache"),
    ("hicache_memfrac", "SGLang (equal mem)", PALETTE["both"], "both"),
    ("park",      "Ours",      PALETTE["park"],    "park"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qps_json")
    ap.add_argument("--out", default="results/qps/fig_norm_latency")
    ap.add_argument("--width", type=float, default=3.4)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--stat", default="mean", choices=["mean", "p50", "p95"])
    ap.add_argument("--ymax", type=float, default=None,
                    help="clip the y-axis so one saturated point does not flatten the "
                         "rest of the curve; unfinished points are still marked hollow.")
    args = ap.parse_args()

    ykey = f"norm_latency_{args.stat}_ms_tok" if args.stat != "mean" \
        else "norm_latency_mean_ms_tok"

    with open(args.qps_json) as fh:
        data = json.load(fh)["arms"]

    use_paper_style()
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    missing_metric = True
    for key, label, color, skey in ARMS:
        rows = data.get(key)
        if not rows:
            continue
        st = STYLE[skey]
        pts = [r for r in rows
               if r.get(ykey) is not None and r.get("throughput_tok_s") is not None]
        if not pts:
            continue
        missing_metric = False
        pts.sort(key=lambda r: r["throughput_tok_s"])
        x = [r["throughput_tok_s"] for r in pts]
        y = [r[ykey] for r in pts]
        ax.plot(x, y, color=color, lw=1.5, ls=st["ls"], marker=st["marker"], ms=3.8,
                label=label, zorder=3, markeredgecolor="white", markeredgewidth=0.5)
        # Same saturation convention as plot_qps.py / plot_job_delay.py: hollow markers
        # are real points from a server that could not drain the offered load, kept
        # rather than dropped, but not to be read as a converged latency.
        sat = [r for r in pts if r.get("past_saturation")]
        if sat:
            ax.plot([r["throughput_tok_s"] for r in sat], [r[ykey] for r in sat],
                    ls="none", marker=st["marker"], ms=6.0, mfc="none",
                    markeredgecolor=color, markeredgewidth=1.1, zorder=4)

    if missing_metric:
        print(f"[error] no arm in {args.qps_json} has {ykey}. This field only exists on "
              "runs made after normalized-latency tracking was added -- re-run "
              "collect_qps.py against fresh bench_*.json files, not an older qps.json.")
        return

    stat_label = {"mean": "mean", "p50": "median", "p95": "P95"}[args.stat]
    ax.set_xlabel("Throughput (tokens / s)", fontsize=8)
    ax.set_ylabel(f"Normalized latency (ms / token) [{stat_label}]", fontsize=8)
    if args.ymax:
        ax.set_ylim(0, args.ymax)
    else:
        ax.set_ylim(bottom=0)
    style_axes(ax)

    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper left", frameon=False, fontsize=7, handlelength=2.0)
    fig.tight_layout(pad=0.3)
    savefig(fig, args.out)
    print("hollow markers = past saturation: the server did not finish the offered load "
          "inside the window, so throughput there is a lower bound and latency is a "
          "still-growing queue, not a converged number.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Average Job Delay vs Jobs Per Second -- the figure the "Continuum"-style scheduling
papers plot (job delay = session arrival -> session completion, offered rate on x).

This is DELIBERATELY the plainer cousin of plot_qps.py's 3-panel figure. plot_qps.py's
panel (c) (TTFT vs delivered throughput) is the right comparison when the claim is about
serving capacity; this one is the right comparison when the reference figure being
matched plots delay against OFFERED rate directly, one line per arm, and the reader is
meant to read "how much load before this arm's delay blows up" at a glance.

사용:
  python benchmark/plot_job_delay.py results/qps/qps_llama8b_bfcl/qps.json \
    --out results/qps/qps_llama8b_bfcl/fig_job_delay
  python benchmark/plot_job_delay.py .../qps.json --stat p95 --ymax 1000
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
    ap.add_argument("--out", default="results/qps/fig_job_delay")
    ap.add_argument("--width", type=float, default=3.4)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--stat", default="mean", choices=["mean", "p50", "p95", "p99"])
    ap.add_argument("--ymax", type=float, default=None,
                    help="clip the y-axis so one saturated point does not flatten the "
                         "rest of the curve; unfinished points are still marked hollow.")
    args = ap.parse_args()

    ykey = f"job_delay_{args.stat}_s"

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
        pts = [r for r in rows if r.get(ykey) is not None and r.get("rate") is not None]
        if not pts:
            continue
        missing_metric = False
        pts.sort(key=lambda r: r["rate"])
        x = [r["rate"] for r in pts]
        y = [r[ykey] for r in pts]
        ax.plot(x, y, color=color, lw=1.5, ls=st["ls"], marker=st["marker"], ms=3.8,
                label=label, zorder=3, markeredgecolor="white", markeredgewidth=0.5)
        # Points past saturation are real measurements of a server that could not drain
        # the offered load in time -- kept, but marked hollow, same convention as
        # plot_qps.py: the knee IS the result, but reading it as steady-state delay
        # (rather than a growing queue) would be the mistake.
        sat = [r for r in pts if r.get("past_saturation")]
        if sat:
            ax.plot([r["rate"] for r in sat], [r[ykey] for r in sat],
                    ls="none", marker=st["marker"], ms=6.0, mfc="none",
                    markeredgecolor=color, markeredgewidth=1.1, zorder=4)

    if missing_metric:
        print(f"[error] no arm in {args.qps_json} has {ykey}. This field only exists on "
              "runs made after job-delay tracking was added to the BFCL/ShareGPT "
              "open-loop scripts -- re-run collect_qps.py against fresh bench_*.json "
              "files, not an older qps.json.")
        return

    stat_label = {"mean": "mean", "p50": "median", "p95": "P95", "p99": "P99"}[args.stat]
    ax.set_xlabel("Jobs Per Second (JPS)", fontsize=8)
    ax.set_ylabel(f"Average Job Delay (s) [{stat_label}]", fontsize=8)
    if args.ymax:
        ax.set_ylim(0, args.ymax)
    else:
        ax.set_ylim(bottom=0)
    style_axes(ax)

    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper left", frameon=False, fontsize=7, handlelength=2.0)
    fig.tight_layout(pad=0.3)
    savefig(fig, args.out)
    print("hollow markers = past saturation (sessions_unfinished_at_drain > 10%): the "
          "server did not finish the offered load inside the window, so that point's "
          "delay is a still-growing queue, not a converged number.")


if __name__ == "__main__":
    main()

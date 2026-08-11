#!/usr/bin/env python3
"""
TTFT vs context length, 3 arms -- companion figure to plot_host_gpu_traffic.py, same
x-axis convention (log2, token-count tick labels) and same source data shape (one JSON
per arm from ttft_ctx_probe.py).

recompute's TTFT is always a full cold prefill (no cache exists). hicache/park's TTFT
is phase 4's genuine post-eviction reload -- see ttft_ctx_probe.py's docstring for why
that's the plotted phase rather than phase 1/2 (still-resident hits every arm can do).

사용:
  python benchmark/plot_ttft_ctx.py \
      --recompute results/ttft_ctx/recompute.json \
      --hicache   results/ttft_ctx/hicache.json \
      --park      results/ttft_ctx/park.json \
      --out results/ttft_ctx/fig_ttft_ctx
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

ARMS = [
    ("recompute", "Recompute", PALETTE["recompute"], "recompute"),
    ("hicache", "SGLang", PALETTE["hicache"], "hicache"),
    # The bandwidth ablation: our own mechanism with every park forced into pinned host
    # DRAM over PCIe instead of a peer GPU (SGLANG_KV_PARK_FORCE_HOST=1). Only shows up
    # when that arm was actually run; it is what separates "the link is faster" from
    # "the policy is better", which park-vs-SGLang alone cannot do.
    ("park_host", "Ours (host DRAM)", PALETTE["both"], "both"),
    ("park", "Ours", PALETTE["park"], "park"),
]


def fmt_tokens(n):
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 1024:
        return f"{n / 1024:.1f}k"
    return str(n)


def main():
    ap = argparse.ArgumentParser()
    for key, _, _, _ in ARMS:
        ap.add_argument(f"--{key}", default=None, help=f"path to {key}.json")
    ap.add_argument("--out", default="results/ttft_ctx/fig_ttft_ctx")
    ap.add_argument("--stat", default="ttft_median_s",
                     choices=["ttft_median_s", "ttft_mean_s"])
    ap.add_argument(
        "--logy", action="store_true",
        help="log-scale y. TTFT here spans ~0.3s to ~60s; on a linear axis the whole "
             "4k-32k region -- where the cached arms actually separate from recompute "
             "-- collapses onto the baseline and only the 128k point is legible.",
    )
    ap.add_argument("--width", type=float, default=3.4)
    ap.add_argument("--height", type=float, default=2.4)
    args = ap.parse_args()

    use_paper_style()
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    all_x = set()
    plotted = 0
    for key, label, color, skey in ARMS:
        path = getattr(args, key)
        if not path:
            continue
        with open(path) as fh:
            data = json.load(fh)
        rows = sorted(data["rows"], key=lambda r: r["context_len"])
        rows = [r for r in rows if r.get(args.stat) is not None]
        if not rows:
            print(f"[warn] {key}: no rows with {args.stat} -- skipping")
            continue
        x = [r["context_len"] for r in rows]
        y = [r[args.stat] for r in rows]
        all_x.update(x)
        st = STYLE[skey]
        ax.plot(x, y, color=color, label=label, **st)
        plotted += 1

    if not plotted:
        raise SystemExit("no arms given -- pass at least one of "
                          "--recompute/--hicache/--park pointing at a ttft_ctx_probe JSON")

    ax.set_xscale("log", base=2)
    if args.logy:
        ax.set_yscale("log")
    xs = sorted(all_x)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt_tokens(v) for v in xs])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel(f"TTFT ({'median' if 'median' in args.stat else 'mean'}, s)")
    ax.set_title("Time to first token vs context length")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, args.out)


if __name__ == "__main__":
    main()

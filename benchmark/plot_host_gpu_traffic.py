#!/usr/bin/env python3
"""
Host(DRAM)->GPU KV-cache traffic vs context length -- RULER-style Figure 5(a), our
arms instead of quantization baselines.

Reads one JSON per arm (as written by host_gpu_traffic_probe.py) and plots GB moved
from host DRAM to GPU HBM against context length, log-x with token-count tick labels
(8k/16k/32k/...), matching the reference figure's axis convention.

사용:
  python benchmark/plot_host_gpu_traffic.py \
      --recompute results/host_gpu_traffic/recompute.json \
      --hicache   results/host_gpu_traffic/hicache.json \
      --park      results/host_gpu_traffic/park.json \
      --out results/host_gpu_traffic/fig_host_gpu_traffic
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
    ap.add_argument("--out", default="results/host_gpu_traffic/fig_host_gpu_traffic")
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
        if not rows:
            continue
        x = [r["context_len"] for r in rows]
        y = [r["host_gb"] for r in rows]
        all_x.update(x)
        st = STYLE[skey]
        ax.plot(x, y, color=color, label=label, **st)
        plotted += 1

    if not plotted:
        raise SystemExit("no arms given -- pass at least one of "
                          "--recompute/--hicache/--park pointing at a probe JSON")

    ax.set_xscale("log", base=2)
    xs = sorted(all_x)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt_tokens(v) for v in xs])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Host→GPU traffic (GB)")
    ax.set_title("Host-to-GPU KV traffic")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, args.out)


if __name__ == "__main__":
    main()

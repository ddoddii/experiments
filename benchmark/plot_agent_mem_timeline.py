#!/usr/bin/env python3
"""
Memory occupancy over time, SGLang vs Ours: prefill HBM, decode HBM, host DRAM.

The claim this figure is for is a PLACEMENT claim -- the same reusable KV ends up
somewhere different -- so it has to be read across all three panels at once. A single
panel can always be explained away: more prefill HBM alone could be a leak, less host
DRAM alone could be a smaller cache. Together they say the bytes moved.

WHICH GPUs ARE WHICH comes from PD_LAYOUT, not from the column order. start_2P_2D.sh
layout 'a' puts prefills on gpu0/gpu1 and decodes on gpu2/gpu3; layout 'b' interleaves
them (P0=0, D0=1, P1=2, D1=3) so that each prefill's NVLink partner is a DECODE gpu.
Reading the wrong mapping would silently swap the two GPU panels and invert the
conclusion, so it is an explicit argument rather than an assumption.

Sums the two GPUs of each role rather than plotting four lines: the question is how much
HBM the ROLE holds, and per-GPU detail is available in the CSV for anyone who wants it.

사용:
  python benchmark/plot_agent_mem_timeline.py --dir results/agent_trace --c 4 \
      --out results/agent_trace/fig_agent_mem
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

ARMS = [
    ("recompute", "Recompute", "#8A9199", (4, 2)),
    ("hicache", "SGLang", PALETTE["hicache"], (5, 2)),
    ("park_host", "Ours (host DRAM)", PALETTE["both"], (1, 1.5)),
    ("park", "Ours", PALETTE["park"], None),
]
LAYOUTS = {"a": {"prefill": (0, 1), "decode": (2, 3)},
           "b": {"prefill": (0, 2), "decode": (1, 3)}}


def load(path, gpus):
    """(elapsed, prefill_gb, decode_gb, host_anon_gb) from a sys_mem_breakdown CSV."""
    t, pre, dec, host = [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                t.append(float(r["elapsed_s"]))
                pre.append(sum(float(r[f"gpu{g}_hbm_mb"]) for g in gpus["prefill"]) / 1024)
                dec.append(sum(float(r[f"gpu{g}_hbm_mb"]) for g in gpus["decode"]) / 1024)
                # anonpages: system-wide, non-reclaimable. Not file_cache -- page cache
                # is reclaimable, so counting it would credit hicache's disk tier as
                # memory pressure it can give back under demand.
                host.append(float(r["anonpages_mb"]) / 1024)
            except (KeyError, ValueError):
                continue
    return t, pre, dec, host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/agent_trace")
    ap.add_argument("--c", type=int, default=4, help="concurrency point to plot")
    ap.add_argument("--layout", default="a", choices=["a", "b"],
                     help="PD_LAYOUT the run used; decides which GPUs are prefill")
    ap.add_argument("--out", default="results/agent_trace/fig_agent_mem")
    ap.add_argument("--hbm-total", type=float, default=49.0, help="GB per GPU")
    ap.add_argument("--width", type=float, default=9.5)
    ap.add_argument("--height", type=float, default=2.5)
    args = ap.parse_args()

    gpus = LAYOUTS[args.layout]
    series = {}
    for key, label, color, dash in ARMS:
        p = os.path.join(args.dir, f"mem_{key}_c{args.c}.csv")
        if os.path.exists(p):
            series[key] = (label, color, dash, load(p, gpus))
    if not series:
        raise SystemExit(f"no mem_*_c{args.c}.csv in {args.dir}")

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height))
    cap = 2 * args.hbm_total
    PANELS = [
        (axes[0], 1, "Prefill HBM (GB)", "(a)  Prefill GPUs", cap),
        (axes[1], 2, "Decode HBM (GB)", "(b)  Decode GPUs", cap),
        (axes[2], 3, "Host DRAM (GB)", "(c)  CPU memory (anon)", None),
    ]
    for ax, idx, ylab, title, ceiling in PANELS:
        for key, (label, color, dash, data) in series.items():
            kw = {"dashes": dash} if dash else {}
            ax.plot(data[0], data[idx], color=color, label=label, lw=1.4, **kw)
        if ceiling:
            ax.axhline(ceiling, color=PALETTE["muted"], lw=0.8, ls=":")
            ax.text(0.98, ceiling, " capacity", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6, color=PALETTE["muted"])
            ax.set_ylim(0, ceiling * 1.08)
        else:
            ax.set_ylim(0, None)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        style_axes(ax)
    axes[0].legend(loc="lower right")

    fig.tight_layout()
    savefig(fig, args.out)

    print(f"\npeak values (C={args.c}, layout {args.layout}):")
    for key, (label, _, _, d) in series.items():
        print(f"  {label:20s} prefill {max(d[1]):5.1f} GB   decode {max(d[2]):5.1f} GB   "
              f"host {max(d[3]):5.1f} GB")


if __name__ == "__main__":
    main()

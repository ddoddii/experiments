#!/usr/bin/env python3
"""
Figure: where the reusable KV physically lives over time (GPU HBM vs CPU DRAM).

The placement claim -- "idle GPU HBM first, CPU DRAM only on overflow" -- is an
assertion until this figure exists. Stacked areas give the absolute split; the dashed
line on the right axis gives the host share, which is the number to quote.

Reads park_location_sampler.py CSVs.

사용:
  python benchmark/plot_park_location.py --csv results/agent/parked_park.csv \
      --out results/agent/fig_park_location
  # compare arms (e.g. GPU-first vs a host-only policy):
  python benchmark/plot_park_location.py \
      --csv gpu_first=results/agent/parked_park.csv \
            host_only=results/agent/parked_hostonly.csv \
      --out results/agent/fig_park_location_cmp
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

MUTED = PALETTE["muted"]


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) if v not in ("", None) else 0.0
                             for k, v in r.items()})
            except ValueError:
                continue
    return rows


def gpu_cols(rows):
    if not rows:
        return []
    return sorted((k for k in rows[0] if k.startswith("gpu") and k.endswith("_gb")
                   and k != "gpu_total_gb"),
                  key=lambda k: int(k[3:-3]))


def summarize(label, rows):
    if not rows:
        print(f"  {label}: no samples")
        return
    n = len(rows)
    g = sum(r["gpu_total_gb"] for r in rows) / n
    h = sum(r["host_gb"] for r in rows) / n
    t = g + h
    # time-weighted share, and the share at peak total -- the moment placement is most
    # under pressure is the one a reviewer will ask about
    peak = max(rows, key=lambda r: r["total_gb"])
    print(f"  {label:>12}: mean GPU {g:6.2f} GB | mean host {h:6.2f} GB | "
          f"host share {100*h/t if t else 0:5.1f}% | "
          f"at peak total ({peak['total_gb']:.2f} GB) host share "
          f"{100*peak['host_frac']:5.1f}% | flushes {peak.get('host_flushes', 0):.0f}")


def draw_one(ax, rows, show_gpu_breakdown):
    t = [r["elapsed_s"] for r in rows]
    gcols = gpu_cols(rows)
    host = [r["host_gb"] for r in rows]

    if show_gpu_breakdown and gcols:
        # per-GPU shades of the park colour so "which idle GPU absorbed it" is visible
        base = PALETTE["park"]
        shades = ["#00543D", "#007B5A", "#009E73", "#4FC3A1"]
        bottom = [0.0] * len(t)
        for i, c in enumerate(gcols):
            vals = [r[c] for r in rows]
            top = [b + v for b, v in zip(bottom, vals)]
            ax.fill_between(t, bottom, top, color=shades[i % len(shades)], lw=0,
                            label=f"GPU{c[3:-3]} HBM", zorder=2)
            bottom = top
        gtot = bottom
    else:
        gtot = [r["gpu_total_gb"] for r in rows]
        ax.fill_between(t, 0, gtot, color=PALETTE["park"], lw=0,
                        label="idle GPU HBM", zorder=2)

    ax.fill_between(t, gtot, [g + h for g, h in zip(gtot, host)],
                    color=PALETTE["hicache"], lw=0, label="CPU DRAM (overflow)", zorder=3)
    ax.plot(t, [g + h for g, h in zip(gtot, host)], color="#000000", lw=0.5, zorder=4)

    ax2 = ax.twinx()
    ax2.plot(t, [100 * r["host_frac"] for r in rows], color=MUTED, lw=1.0,
             ls=(0, (4, 2.5)), zorder=5, label="host share")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("host share (%)", color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED, length=2.5, width=0.5)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    return ax2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True,
                    help="path, or label=path for multiple arms")
    ap.add_argument("--out", default="results/agent/fig_park_location")
    ap.add_argument("--per-gpu", action="store_true",
                    help="stack each GPU separately instead of one GPU total")
    ap.add_argument("--width", type=float, default=6.4)
    ap.add_argument("--height", type=float, default=2.2)
    args = ap.parse_args()

    arms = []
    for spec in args.csv:
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = os.path.splitext(os.path.basename(spec))[0], spec
        if not os.path.exists(path):
            print(f"[warn] skip {spec}")
            continue
        arms.append((label, load(path)))
    if not arms:
        print("[error] no CSVs loaded")
        return

    print("\n=== parked KV by physical location ===")
    for label, rows in arms:
        summarize(label, rows)

    use_paper_style()
    fig, axes = plt.subplots(len(arms), 1, sharex=True, squeeze=False,
                             figsize=(args.width, args.height * len(arms)))
    ax2_last = None
    for ax, (label, rows) in zip(axes[:, 0], arms):
        if not rows:
            continue
        ax2_last = draw_one(ax, rows, args.per_gpu)
        ax.set_ylabel("parked KV (GB)")
        if len(arms) > 1:
            ax.set_title(label, fontsize=8.5, pad=3)
        ax.set_xlim(min(r["elapsed_s"] for r in rows), max(r["elapsed_s"] for r in rows))
        style_axes(ax)
        ax.grid(False)
    axes[-1, 0].set_xlabel("time (s)")

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    if ax2_last is not None:
        h2, l2 = ax2_last.get_legend_handles_labels()
        h1, l1 = h1 + h2, l1 + l2
    fig.legend(h1, l1, loc="lower center", ncol=min(4, len(l1)), frameon=False,
               fontsize=6.6, bbox_to_anchor=(0.5, 0.99), columnspacing=1.4)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.97])
    savefig(fig, args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out)


if __name__ == "__main__":
    main()

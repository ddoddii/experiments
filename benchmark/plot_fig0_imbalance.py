#!/usr/bin/env python3
"""
Fig 0 (intro): KV capacity is unevenly distributed ACROSS THE FOUR GPUs.

Deliberately role-agnostic. The DIRECTION of the prefill/decode imbalance is not a fixed
property of PD disaggregation -- with a 20k prefill pool at C=24 the decode pools ran
hotter (0.75 vs 0.44 mean), with a 60k pool at C=16 the prefill pools do (0.89 vs 0.06).
A figure that names roles commits the paper to one of those and makes the other read as a
contradiction. The claim that survives both, and the one the mechanism actually rests on,
is weaker and sufficient:

    at essentially every instant, some GPU has run out of KV room while
    tens of GB sit free on another.

Two panels, because the fraction and the byte count say different things and the paper
needs both:

  (a) occupancy fraction per GPU -- shows the imbalance is persistent, not a transient
      of the warm-up. Fractions are what the placement policy compares.
  (b) FREE capacity in GB per GPU -- shows the imbalance is worth acting on. Panel (a)
      cannot: the pools are not the same size (a prefill capped at 60k tokens holds
      7.3 GB, a decode 22.6 GB), so "5% free" and "95% free" are not 19x apart in
      memory, and only the byte axis says how much there is to move.

사용:
  python benchmark/plot_fig0_imbalance.py \
      --csv results/exp2/pd_layoutb_p60000_c16/occ_hicache.csv \
      --rename P0=GPU0,D0=GPU1,P1=GPU2,D1=GPU3 \
      --out results/intro/fig0_imbalance
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

# Okabe-Ito, four hues that stay distinct in grayscale and for colour-blind readers.
# No role grouping: the point is four GPUs, not two pairs.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
STYLES = ["-", "-", "--", "--"]
INK, MUTED = PALETTE["ink"], PALETTE["muted"]


def load(path, kib):
    rows = [ln.rstrip("\n") for ln in open(path) if not ln.startswith("#")]
    r = csv.DictReader(rows)
    labels = [c[:-4] for c in r.fieldnames if c.endswith("_use")]
    t, occ, caps = [], {l: [] for l in labels}, {l: [] for l in labels}
    for row in r:
        try:
            t.append(float(row["t_s"]))
        except (ValueError, KeyError):
            continue
        for l in labels:
            v = row.get(f"{l}_use", "")
            occ[l].append(float(v) if v not in ("", None) else None)
            c = row.get(f"{l}_cap_tok", "")
            if c not in ("", None):
                caps[l].append(float(c))
    cap_gb = {}
    for l in labels:
        if not caps[l]:
            raise SystemExit(
                f"{path} has no {l}_cap_tok column. Panel (b) is the point of this "
                f"figure and cannot be drawn without pool capacity -- re-run with the "
                f"current kv_occupancy_timeseries.py."
            )
        cap_gb[l] = sorted(caps[l])[len(caps[l]) // 2] * kib / (1024 * 1024)
    return t, occ, cap_gb, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rename", default="", help="P0=GPU0,D0=GPU1,P1=GPU2,D1=GPU3")
    ap.add_argument("--kib-per-token", type=float, default=128.0)
    ap.add_argument("--hi", type=float, default=0.80)
    ap.add_argument("--lo", type=float, default=0.50)
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=3.2)
    ap.add_argument("--out", default="results/intro/fig0_imbalance")
    args = ap.parse_args()

    t, occ, cap_gb, labels = load(args.csv, args.kib_per_token)
    if args.rename:
        m = dict(kv.split("=", 1) for kv in args.rename.split(",") if "=" in kv)
        miss = [l for l in labels if l not in m]
        if miss:
            raise SystemExit(f"--rename does not cover {miss}")
        occ = {m[l]: v for l, v in occ.items()}
        cap_gb = {m[l]: v for l, v in cap_gb.items()}
        labels = sorted(m.values())

    # Free GB per GPU per sample, and the opportunity condition.
    free = {l: [(1.0 - u) * cap_gb[l] if u is not None else None for u in occ[l]]
            for l in labels}
    opp, stranded = [], []
    for i in range(len(t)):
        vals = [occ[l][i] for l in labels if occ[l][i] is not None]
        on = len(vals) >= 2 and max(vals) >= args.hi and min(vals) <= args.lo
        opp.append(on)
        if on:
            stranded.append(sum(free[l][i] for l in labels
                                if occ[l][i] is not None and occ[l][i] <= args.lo))
    frac = sum(opp) / len(opp) if opp else 0.0
    mean_stranded = sum(stranded) / len(stranded) if stranded else 0.0

    print(f"=== Fig 0 ({os.path.basename(args.csv)}) ===")
    for l in labels:
        vv = [x for x in occ[l] if x is not None]
        fv = [x for x in free[l] if x is not None]
        print(f"  {l}: pool {cap_gb[l]:5.2f} GB   occupancy mean {sum(vv)/len(vv):.3f}"
              f"   free {sum(fv)/len(fv):5.2f} GB")
    print(f"  imbalance present (some GPU >= {args.hi:g}, some <= {args.lo:g}): "
          f"{frac*100:.1f}% of the run")
    print(f"  stranded while present: {mean_stranded:.1f} GB on average")

    use_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(args.width, args.height), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})

    # (a) occupancy fraction. No opportunity shading: it holds for 87% of the run, so a
    # wash over 87% of the axes is a solid block that hides the lines it is annotating.
    # The number belongs in the title, where it can be read.
    for i, l in enumerate(labels):
        xs = [t[k] for k in range(len(t)) if occ[l][k] is not None]
        ys = [occ[l][k] for k in range(len(t)) if occ[l][k] is not None]
        ax1.plot(xs, ys, color=COLORS[i], ls=STYLES[i], lw=1.1, label=l, zorder=3)
    ax1.axhline(args.hi, color=MUTED, lw=0.7, ls=(0, (5, 3)))
    ax1.text(t[-1], args.hi + 0.02, f"saturated {args.hi:g}", color=MUTED,
             fontsize=6.5, va="bottom", ha="right")
    ax1.set_ylabel("KV pool\noccupancy")
    ax1.set_ylim(0, 1.18)          # headroom so the legend never sits on the traces
    ax1.set_title(f"(a) Occupancy is split: two GPUs pinned at capacity, two near empty "
                  f"— {frac*100:.0f}% of the run")
    ax1.legend(ncol=4, loc="upper left", handlelength=1.7, columnspacing=1.1,
               borderaxespad=0.3, frameon=False, fontsize=7)
    style_axes(ax1)

    # (b) the same instant in bytes. Stacked, so the top of the stack is the cluster's
    # total free KV capacity and the bands show which GPUs are holding it.
    xs = [t[k] for k in range(len(t)) if all(free[l][k] is not None for l in labels)]
    idx = [k for k in range(len(t)) if all(free[l][k] is not None for l in labels)]
    ax2.stackplot(xs, *[[free[l][k] for k in idx] for l in labels],
                  colors=COLORS, labels=labels, lw=0, alpha=0.85)
    ax2.set_ylabel("Free KV\ncapacity (GB)")
    ax2.set_xlabel("time (s)")
    # "on the other GPUs", not "on the GPUs that cannot use it": nothing stops a decode
    # node from using its own pool, it simply has no demand for it. The defensible claim
    # is about location, not capability.
    ax2.set_title(f"(b) …and the free capacity is all on the other GPUs "
                  f"— {mean_stranded:.0f} GB")
    style_axes(ax2)

    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(hspace=0.42)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

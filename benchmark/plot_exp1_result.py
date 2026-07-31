#!/usr/bin/env python3
"""
Exp 1 result figures: what it costs (TTFT) and what it saves (host DRAM).

Two panels, two independent x-axes, because the two claims live on different clocks:

  (a) TTFT vs TURN INDEX. Turn is the right axis, not wall time: the mechanism can only
      act once a conversation has a prefix to reuse, so the turn number IS the amount of
      reusable history. Turn 0 doubles as the control -- no arm can reuse or fetch there,
      so the gap at turn 0 is constant overhead and everything after it is the mechanism.

  (b) Host DRAM (AnonPages) vs WALL TIME. Time is the right axis here because the claim
      is about a RESERVATION: HiCache's host pool is allocated when the server starts and
      never shrinks, which is only visible as a flat line that begins high at t=0.

Deliberately NOT one chart with two y-axes. Two measures on two scales sharing an axis
invites a crossing point that means nothing.

AnonPages rather than process RSS: it is system-wide and non-reclaimable, so it is what
an adjacent agent process actually cannot have. Page cache is excluded -- it is
reclaimable, and its level here is contaminated by the order the arms ran in.

사용:
  python benchmark/plot_exp1_result.py --dir results/exp1/sharegpt_p60000_c8_m1024
  python benchmark/plot_exp1_result.py --dir ... --split     # two separate files
  python benchmark/plot_exp1_result.py --dir ... --width 3.4 # single-column
"""
import argparse
import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

# Fixed hue order, assigned per arm and never cycled. Validated as a categorical trio
# (worst adjacent CVD dE 18.0 protan / 18.7 normal). Amber sits below 3:1 against the
# surface, which is why every line is also directly labelled rather than legend-only.
ARM_STYLE = {
    "radix":   dict(color="#E69F00", marker="^", ls=":",  label="GPU-only (radix)"),
    "hicache": dict(color=PALETTE["hicache"], marker="o", ls="--", label="HiCache (host DRAM)"),
    "park":    dict(color=PALETTE["park"], marker="s", ls="-",  label="GPU-first (ours)"),
}
ORDER = ["radix", "hicache", "park"]


def load_ttft(path, min_n):
    """turn -> median TTFT, for turns with at least min_n samples.

    Median, not mean: TTFT has a heavy right tail (a handful of turn 3-6 outliers reach
    6 s), and a mean would report those outliers as if they were the typical turn."""
    with open(path) as fh:
        d = json.load(fh)
    by = {}
    for item in d.get("results") or []:
        for t in item.get("turns") or []:
            if t.get("ttft_s") is not None:
                by.setdefault(t["turn"], []).append(t["ttft_s"])
    return ({k: st.median(v) for k, v in sorted(by.items()) if len(v) >= min_n},
            {k: len(v) for k, v in sorted(by.items())})


def load_mem(path, col="anonpages_mb"):
    ts, ys = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                ts.append(float(r["elapsed_s"]) / 60.0)
                ys.append(float(r[col]) / 1024.0)
            except (KeyError, TypeError, ValueError):
                continue
    return ts, ys


def _label_ends(ax, items, min_gap_frac=0.075):
    """Direct-label each series at its last point, pushed apart so they never overlap.

    Present on every series: the legend alone would leave identity to colour, and one of
    the three hues sits below the 3:1 contrast floor against the surface. The separation
    is computed rather than hand-tuned because two arms can finish within 0.03 of each
    other (radix 0.412 vs hicache 0.439) and a fixed offset silently stops working when
    the data changes.

    items: list of (x, y, text, color), any order."""
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * min_gap_frac
    order = sorted(items, key=lambda it: it[1])          # bottom-up
    placed = []
    for x, y, text, color in order:
        ty = y if not placed else max(y, placed[-1][1] + min_gap)
        placed.append((x, ty, text, color, y))
    # if the stack pushed past the top, slide the whole stack down
    over = placed[-1][1] - (hi - min_gap * 0.3)
    if over > 0:
        placed = [(x, ty - over, t, c, y0) for x, ty, t, c, y0 in placed]
    for x, ty, text, color, y0 in placed:
        ax.annotate(text, xy=(x, ty), xytext=(4, 0), textcoords="offset points",
                    color=color, fontsize=6.6, ha="left", va="center",
                    fontweight="bold", annotation_clip=False)


def panel_ttft(ax, arms, min_n):
    ends = []
    for arm in ORDER:
        d = arms.get(arm)
        if not d:
            continue
        med, _ = d
        xs = sorted(med)
        ys = [med[k] for k in xs]
        s = ARM_STYLE[arm]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"], ls=s["ls"],
                label=s["label"], markeredgecolor="white", markeredgewidth=0.5,
                zorder=3, clip_on=False)
        ends.append((xs[-1], ys[-1], s["label"].split(" (")[0], s["color"]))
    ax.set_xlabel("turn index")
    ax.set_ylabel("median TTFT (s)")
    ax.set_xlim(-0.3, max(xs) + 0.3)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, max(xs) + 1))
    # Turn 0 is the control, so say so on the figure rather than only in the caption.
    ax.axvspan(-0.3, 0.5, color=PALETTE["grid"], alpha=0.55, lw=0, zorder=0)
    ax.annotate("no prefix\nto reuse", xy=(0.12, ax.get_ylim()[1] * 0.96),
                fontsize=6.2, color=PALETTE["muted"], ha="left", va="top")
    # Shade the saving rather than leaving the reader to difference two lines. Measured
    # against the BETTER of the two baselines at each turn, so the band never overstates.
    if "park" in arms and len(arms) > 1:
        pmed = arms["park"][0]
        base = {k: min(arms[a][0][k] for a in arms if a != "park" and k in arms[a][0])
                for k in pmed if any(k in arms[a][0] for a in arms if a != "park")}
        ks = [k for k in sorted(base) if k >= 1]           # turn 0 has nothing to reuse
        if ks:
            ax.fill_between(ks, [pmed[k] for k in ks], [base[k] for k in ks],
                            color=PALETTE["park"], alpha=0.13, lw=0, zorder=1)
    _label_ends(ax, ends)
    style_axes(ax)


def panel_mem(ax, mems):
    ends = []
    for arm in ORDER:
        if arm not in mems:
            continue
        ts, ys = mems[arm]
        s = ARM_STYLE[arm]
        ax.plot(ts, ys, color=s["color"], ls=s["ls"], lw=1.3, zorder=3)
        ends.append((ts[-1], ys[-1], s["label"].split(" (")[0], s["color"]))
    ax.set_xlabel("wall time (min)")
    ax.set_ylabel("host DRAM, AnonPages (GB)")
    xmax = max(max(t) for t, _ in mems.values())
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    # State the saving on the figure. Two flat lines make the reader subtract two axis
    # readings to get the one number the panel exists to deliver.
    if "hicache" in mems and "park" in mems:
        hi = max(mems["hicache"][1])
        pk = max(mems["park"][1])
        xa = xmax * 0.42
        ax.annotate("", xy=(xa, hi), xytext=(xa, pk),
                    arrowprops=dict(arrowstyle="<->", color=PALETTE["muted"],
                                    lw=0.7, shrinkA=0, shrinkB=0))
        ax.annotate(f"−{hi - pk:.1f} GB", xy=(xa, (hi + pk) / 2),
                    xytext=(4, 0), textcoords="offset points", fontsize=7,
                    fontweight="bold", color=PALETTE["ink"], ha="left", va="center")
    _label_ends(ax, ends)
    style_axes(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", nargs="+", default=ORDER)
    ap.add_argument("--out", default=None, help="stem; default <dir>/fig_exp1")
    ap.add_argument("--min-n", type=int, default=30,
                    help="drop turns with fewer samples than this (default 30)")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=2.3)
    ap.add_argument("--split", action="store_true", help="write two separate figures")
    ap.add_argument("--mem-col", default="anonpages_mb")
    args = ap.parse_args()

    arms, mems = {}, {}
    for arm in args.arms:
        b = os.path.join(args.dir, f"bench_{arm}.json")
        m = os.path.join(args.dir, f"mem_{arm}.csv")
        if os.path.exists(b):
            arms[arm] = load_ttft(b, args.min_n)
        if os.path.exists(m):
            mems[arm] = load_mem(m, args.mem_col)
    if not arms and not mems:
        print(f"[error] nothing loaded from {args.dir}")
        return

    print("=== TTFT median by turn ===")
    for arm in ORDER:
        if arm not in arms:
            continue
        med, n = arms[arm]
        print(f"  {arm:<8} " + " ".join(f"t{k}:{v:.3f}({n[k]})" for k, v in med.items()))
    print("=== host DRAM (AnonPages) ===")
    for arm in ORDER:
        if arm not in mems:
            continue
        _, ys = mems[arm]
        print(f"  {arm:<8} start {ys[0]:.1f} GB  peak {max(ys):.1f} GB  end {ys[-1]:.1f} GB")

    use_paper_style()
    stem = args.out or os.path.join(args.dir, "fig_exp1")
    if args.split:
        f1, a1 = plt.subplots(figsize=(args.width / 2, args.height))
        panel_ttft(a1, arms, args.min_n)
        a1.legend(loc="upper left", fontsize=6.4, handlelength=1.8, borderpad=0.2)
        f1.tight_layout(pad=0.3)
        savefig(f1, stem + "_ttft")

        f2, a2 = plt.subplots(figsize=(args.width / 2, args.height))
        panel_mem(a2, mems)
        f2.tight_layout(pad=0.3)
        savefig(f2, stem + "_mem")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(args.width, args.height))
    panel_ttft(ax1, arms, args.min_n)
    panel_mem(ax2, mems)
    ax1.set_title("(a) latency", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax2.set_title("(b) host memory", loc="left", fontsize=8, fontweight="bold", pad=3)
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.8, handlelength=2.0)
    fig.tight_layout(pad=0.4, rect=[0, 0.05, 1, 1])
    savefig(fig, stem)


if __name__ == "__main__":
    main()

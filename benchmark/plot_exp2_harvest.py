#!/usr/bin/env python3
"""
Exp 2 figure: how much otherwise-idle GPU memory the placement policy actually harvests.

The earlier version of this figure plotted the POLICY (which GPU was chosen, against
which were rejected). That answers "does the selector prefer idle GPUs", which is close
to a restatement of its source code, and it never showed the thing the experiment is
about: how much idle GPU capacity ends up holding useful KV. This one plots the
RESOURCE.

  (a) WHERE THE REUSABLE KV LIVES, over time, stacked by GPU. Under local-only parking
      the two decode GPUs stay flat at zero however idle they are; under GPU-first
      placement they carry most of the cache. This is the claim, drawn directly.

  (b) HOW MUCH IDLE MEMORY WAS HARVESTED, against how much was there. Two numbers matter
      and they say different things: the share of the cache that idle GPUs absorbed
      (2/3 here), and the share of the idle capacity that was used at all (a sixth here,
      because per-GPU park size is a configured cap, not a limit of the mechanism). The
      second is drawn as the unfilled remainder so the figure cannot be read as "all the
      idle memory is now in use".

  (c) WHETHER IT PAID. Hit rate and TTFT at p50 AND p95: the median improves while the
      tail worsens, and showing one without the other reports a win the other contradicts.

사용:
  python benchmark/plot_exp2_harvest.py --dir results/exp2/pd_layoutb_c32_pergpu10k \
      --out results/exp2/fig_exp2
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

BASE, OURS = "park_local", "park_pd"
C_BASE, C_OURS = "#BFC4C9", PALETTE["park"]
C_IDLE = PALETTE["hicache"]
INK, MUTED, GRID = PALETTE["ink"], PALETTE["muted"], PALETTE["grid"]


def residency(d, arm):
    """Parked KV per target GPU over time, plus the role of each GPU."""
    rows = list(csv.DictReader(open(os.path.join(d, f"parked_{arm}.csv"))))
    f = lambda r, k: float(r.get(k, "") or 0)
    t = [f(r, "elapsed_s") for r in rows]
    per = {g: [f(r, f"gpu{g}_gb") for r in rows] for g in (0, 1, 2, 3)}
    return t, per


def roles(d, arm):
    imb = json.load(open(os.path.join(d, "imbalance.json")))[arm]["per_gpu"]
    # label -> role, and the free KV capacity that label had on average
    return ({k: v["role"] for k, v in imb.items()},
            {k: v["free_gb_mean"] for k, v in imb.items()})


def stats(d, arm):
    rows = list(csv.DictReader(open(os.path.join(d, f"parked_{arm}.csv"))))
    f = lambda k: float(rows[-1].get(k, "") or 0)
    hits, miss = f("fetch_hits"), f("fetch_miss")
    s = json.load(open(os.path.join(d, f"bench_{arm}.json")))["summary"]
    return {"hit_rate": 100 * hits / (hits + miss) if hits + miss else 0.0,
            "p50": s["ttft_p50_s"], "p95": s["ttft_p95_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    # PD_LAYOUT=b: prefill on gpu0/gpu2, decode on gpu1/gpu3. Passed rather than inferred
    # so a layout-a run cannot be silently mislabelled.
    ap.add_argument("--prefill-gpus", nargs="+", type=int, default=[0, 2])
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=2.05)
    ap.add_argument("--out", default="results/exp2/fig_exp2")
    args = ap.parse_args()

    P = set(args.prefill_gpus)
    D = [g for g in (0, 1, 2, 3) if g not in P]
    sb, so = stats(args.dir, BASE), stats(args.dir, OURS)
    _, free_b = roles(args.dir, BASE)
    _, free_o = roles(args.dir, OURS)
    # Idle KV capacity on the non-prefill GPUs -- the resource on offer. Taken from the
    # arm being drawn, because allocating a park pool there shrinks the serving pool and
    # using the other arm's number would credit capacity that no longer exists.
    idle_o = sum(v for k, v in free_o.items() if k.startswith("D"))
    idle_b = sum(v for k, v in free_b.items() if k.startswith("D"))

    t_o, per_o = residency(args.dir, OURS)
    t_b, per_b = residency(args.dir, BASE)
    harv = {arm: {"P": max(sum(v) for v in zip(*[per[g] for g in sorted(P)])),
                  "D": max(sum(v) for v in zip(*[per[g] for g in D]))}
            for arm, per in ((BASE, per_b), (OURS, per_o))}

    print(f"  idle KV capacity on decode GPUs : {BASE} {idle_b:.1f} GB / {OURS} {idle_o:.1f} GB")
    for arm in (BASE, OURS):
        h = harv[arm]
        tot = h["P"] + h["D"]
        print(f"  {arm:11s} cache {tot:5.2f} GB = prefill {h['P']:.2f} + idle-GPU {h['D']:.2f}"
              f"   ({100*h['D']/tot if tot else 0:.0f}% on idle GPUs)")

    use_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(args.width, args.height),
                                        gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # (a) residency over time. Ours stacked by GPU; the baseline drawn as a single line,
    # since all of it sits on the prefill GPUs and a second stack would just be flat.
    cols = {g: (C_OURS if g in P else C_IDLE) for g in (0, 1, 2, 3)}
    order = sorted(P) + D
    ax1.stackplot(t_o, *[per_o[g] for g in order],
                  colors=[cols[g] for g in order],
                  labels=[f"GPU{g}" + (" (prefill)" if g in P else " (idle/decode)")
                          for g in order],
                  lw=0, alpha=[0.9, 0.7, 0.55, 0.4][:len(order)])
    base_tot = [sum(v) for v in zip(*[per_b[g] for g in (0, 1, 2, 3)])]
    ax1.plot(t_b, base_tot, color=INK, lw=1.1, ls="--", zorder=4, label="local-only (total)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("Reusable KV resident (GB)")
    ax1.set_title("(a) Idle GPUs take most of the cache")
    ax1.set_ylim(0, max(sum(v) for v in zip(*[per_o[g] for g in (0, 1, 2, 3)])) * 1.35)
    ax1.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.2,
               labelspacing=0.25, borderaxespad=0.2)
    style_axes(ax1)

    # (b) harvested against available. The unfilled bar is idle capacity left on the
    # table; without it the panel would imply the idle memory is now fully in use.
    for i, (arm, idle) in enumerate(((BASE, idle_b), (OURS, idle_o))):
        h = harv[arm]
        ax2.bar(i, idle, width=0.6, color="none", edgecolor=MUTED, linewidth=0.7,
                linestyle=(0, (2, 2)), zorder=2)
        ax2.bar(i, h["P"], width=0.6, color=C_BASE, edgecolor=INK, linewidth=0.35, zorder=3)
        ax2.bar(i, h["D"], width=0.6, bottom=h["P"], color=C_IDLE, edgecolor=INK,
                linewidth=0.35, zorder=3, label="on idle GPUs" if i else None)
        tot = h["P"] + h["D"]
        ax2.text(i, tot + idle * 0.02, f"{tot:.2f} GB", ha="center",
                 va="bottom", fontsize=7, color=INK, zorder=4)
    ax2.text(1, idle_o, f"idle capacity {idle_o:.0f} GB", ha="center", va="bottom",
             fontsize=5.8, color=MUTED)
    # The number that matters is not the share of idle memory consumed -- that is set by
    # the configured per-GPU park size, not by the mechanism -- but what borrowing it did
    # to the cache. Tripling the resident cache is what produced the hit-rate gain in (c).
    tb_, to_ = (harv[BASE]["P"] + harv[BASE]["D"]), (harv[OURS]["P"] + harv[OURS]["D"])
    if tb_:
        ax2.annotate("", xy=(1, to_), xytext=(0, tb_),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=0.8,
                                     connectionstyle="arc3,rad=-0.25"))
        ax2.text(0.5, (tb_ + to_) / 2 + idle_o * 0.10, f"{to_/tb_:.1f}× cache",
                 ha="center", va="bottom", fontsize=7, color=INK, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Local-only", "Ours"], fontsize=7)
    ax2.set_ylabel("KV on GPU (GB)")
    ax2.set_title("(b) Borrowed idle memory")
    ax2.set_ylim(0, max(idle_b, idle_o) * 1.22)
    style_axes(ax2)

    # (c) payoff, on two scales: hit rate as bars, TTFT as labelled text beneath, so the
    # tail regression is stated rather than hidden behind a second axis.
    w = 0.36
    for i, (lab, b, o) in enumerate((("hit rate (%)", sb["hit_rate"], so["hit_rate"]),)):
        ax3.bar(-w / 2, b, width=w, color=C_BASE, edgecolor=INK, linewidth=0.35, zorder=3)
        ax3.bar(w / 2, o, width=w, color=C_OURS, edgecolor=INK, linewidth=0.35, zorder=3)
        for x, v in ((-w / 2, b), (w / 2, o)):
            ax3.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax3.set_xlim(-0.55, 1.75)
    ax3.set_xticks([0])
    ax3.set_xticklabels(["Cache hit rate (%)"], fontsize=7)
    ax3.set_ylabel("%")
    ax3.set_ylim(0, max(sb["hit_rate"], so["hit_rate"]) * 1.45)
    ax3.set_title("(c) What it buys, and costs")
    y = ax3.get_ylim()[1]
    ax3.text(1.1, y * 0.92, "TTFT (s)", fontsize=6.5, color=INK, ha="center", va="top")
    for j, (k, lab) in enumerate((("p50", "p50"), ("p95", "p95"))):
        ax3.text(1.1, y * (0.78 - 0.17 * j),
                 f"{lab}  {sb[k]:.2f} → {so[k]:.2f}", fontsize=6.5,
                 color=INK if k == "p50" else PALETTE["recompute"], ha="center", va="top")
    ax3.text(1.1, y * 0.40, "median improves,\ntail does not", fontsize=5.8,
             color=MUTED, ha="center", va="top", style="italic")
    style_axes(ax3)

    fig.tight_layout(pad=0.5)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

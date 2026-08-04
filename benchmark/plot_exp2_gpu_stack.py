#!/usr/bin/env python3
"""
Exp 2 main figure: per-GPU KV memory, stacked as own cache / borrowed cache / still free.

Each GPU gets a pair of bars, local-only against GPU-first placement, stacked as:

    own KV      the cache the GPU holds for its own serving work
    parked KV   reusable cache another node placed there  <- the mechanism
    free        pool capacity still unused                <- the waste

Reading it as one picture: the prefill GPUs are full and their bars have almost no free
band, the decode GPUs are nearly empty and theirs is almost all free band, and under
GPU-first placement a coloured block appears inside that empty band. The waste and the
thing that fills it are in the same bar, at the same scale, which no ratio or bar of
aggregates can show.

Deliberately the KV-pool view, not total HBM. On a 49 GB card the parked 2.6 GB would be
a sliver beside 16 GB of model weights and the panel would say nothing; the pool is the
resource actually being contended for.

사용:
  python benchmark/plot_exp2_gpu_stack.py --dir results/exp2/pd_layoutb_c32_pergpu10k \
      --out results/exp2/fig_exp2
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

BASE, OURS = "park_local", "park_pd"
C_OWN = "#BFC4C9"          # the GPU's own working cache -- neutral, it is not the claim
C_PARK = PALETTE["park"]   # borrowed cache: the mechanism
C_FREE = "#FFFFFF"         # unused pool capacity, outlined so it reads as absence
INK, MUTED = PALETTE["ink"], PALETTE["muted"]


def arm_data(d, arm, order):
    imb = json.load(open(os.path.join(d, "imbalance.json")))[arm]["per_gpu"]
    rows = list(csv.DictReader(open(os.path.join(d, f"parked_{arm}.csv"))))
    f = lambda r, k: float(r.get(k, "") or 0)
    parked = {g: max(f(r, f"gpu{g}_gb") for r in rows) for g in range(4)}
    out = []
    for lab, g in order:
        v = imb[lab]
        cap = v["cap_gb"]
        own = v["use_mean"] * cap
        out.append({"label": lab, "gpu": g, "role": v["role"], "cap": cap,
                    "own": own, "free": max(0.0, cap - own), "parked": parked[g]})
    return out


def stats(d, arm):
    """PARK-FETCH hit rate: of the prefill requests the local radix could not serve,
    the share that a parked prefix rescued.

    Not the same quantity as Exp 1's "cache hit rate", and must not be printed under that
    name. Exp 1 reports reuse_ratio = cached_tokens/prompt_tokens -- TOKEN-weighted, over
    every request. This is REQUEST-weighted, and its denominator deliberately excludes
    fetch_already (requests whose prefix was still in the local radix, so parking was
    never consulted). Both are "hit rates" in English and neither is the other in units;
    labelling this one "cache hit rate" invites the reader to compare 50.1% against Exp 1's
    55.7% as if one were lower than the other.
    """
    rows = list(csv.DictReader(open(os.path.join(d, f"parked_{arm}.csv"))))
    f = lambda k: float(rows[-1].get(k, "") or 0)
    h, m, al, ns = (f("fetch_hits"), f("fetch_miss"),
                    f("fetch_already"), f("fetch_nospace"))
    s = json.load(open(os.path.join(d, f"bench_{arm}.json")))["summary"]
    return {"hit": 100 * h / (h + m) if h + m else 0.0,
            "hits": h, "miss": m, "already": al, "nospace": ns,
            "p50": s["ttft_p50_s"], "p95": s["ttft_p95_s"], "p99": s["ttft_p99_s"]}


def agg(dirs, arm):
    """Median over repeats, keeping the per-repeat values so the figure can show spread.

    Quoting one run here produced a caption that claimed a 3.3x tail improvement. Three
    repeats later the baseline's p95 alone ranged over 4.5x and the sign flipped twice:
    that figure was the baseline drawing its worst value, not an effect. Anything the
    panel states now is a median across repeats with the range attached.
    """
    per = [stats(d, arm) for d in dirs]
    out = {k: st.median([p[k] for p in per]) for k in per[0]}
    out["n"] = len(per)
    out["all"] = {k: [p[k] for p in per] for k in per[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    # Several repeat directories, not one run. The memory panel is drawn from the first
    # (the layout is configured and came out identical in all three); the payoff panel is
    # aggregated over all of them.
    ap.add_argument("--dirs", nargs="+", required=True)
    # PD_LAYOUT=b physical order, passed rather than inferred so a layout-a run cannot be
    # silently mislabelled.
    ap.add_argument("--order", default="P0:0,D0:1,P1:2,D1:3")
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=2.45)
    ap.add_argument("--out", default="results/exp2/fig_exp2")
    args = ap.parse_args()

    dirs = sorted(sum((glob.glob(x) for x in args.dirs), []))
    if not dirs:
        raise SystemExit(f"no directories matched {args.dirs}")
    order = [(t.split(":")[0], int(t.split(":")[1])) for t in args.order.split(",")]
    B, O = arm_data(dirs[0], BASE, order), arm_data(dirs[0], OURS, order)
    sb, so = agg(dirs, BASE), agg(dirs, OURS)
    print(f"  aggregating {len(dirs)} run(s): " + ", ".join(os.path.basename(d) for d in dirs))

    for name, rows in ((BASE, B), (OURS, O)):
        tot = sum(r["parked"] for r in rows)
        onD = sum(r["parked"] for r in rows if r["role"] == "D")
        print(f"  {name:11s} parked {tot:5.2f} GB  (on idle/decode GPUs {onD:.2f})  "
              + "  ".join(f"gpu{r['gpu']}:{r['parked']:.2f}" for r in rows))
    # Print the full partition, so the plotted rate can never be mistaken for a hit rate
    # over all requests. The four buckets are disjoint and cover every prefill request.
    for name, s in ((BASE, sb), (OURS, so)):
        n = s["hits"] + s["miss"] + s["already"] + s["nospace"]
        print(f"  {name:11s} prefill reqs {n:.0f} = park-hit {s['hits']:.0f} + miss "
              f"{s['miss']:.0f} + radix-already {s['already']:.0f} + nospace "
              f"{s['nospace']:.0f}   -> park-fetch hit rate {s['hit']:.1f}% "
              f"(denominator excludes radix-already)")

    use_paper_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(args.width, args.height),
                                  gridspec_kw={"width_ratios": [2.5, 1]})

    w, gap = 0.34, 0.04
    xs = range(len(order))
    for i, (b, o) in enumerate(zip(B, O)):
        for j, (r, off) in enumerate(((b, -(w + gap) / 2), (o, (w + gap) / 2))):
            x = i + off
            ax.bar(x, r["own"], width=w, color=C_OWN, edgecolor=INK, lw=0.4, zorder=3)
            ax.bar(x, r["parked"], width=w, bottom=r["own"], color=C_PARK,
                   edgecolor=INK, lw=0.4, zorder=3)
            # Free capacity drawn on top as an outlined white block: the gap between what
            # a GPU uses and what it has IS the waste this work is about, so it has to be
            # visible as area rather than implied by a short bar.
            ax.bar(x, r["free"], width=w, bottom=r["own"] + r["parked"], color=C_FREE,
                   edgecolor=MUTED, lw=0.5, linestyle=(0, (2, 2)), zorder=2)
            if r["parked"] > 0.4:
                ax.text(x, r["own"] + r["parked"] / 2, f"{r['parked']:.1f}", ha="center",
                        va="center", fontsize=6, color="white", zorder=4, fontweight="bold")

    minor, minor_lab = [], []
    for i in xs:
        minor += [i - (w + gap) / 2, i + (w + gap) / 2]
        minor_lab += ["local", "ours"]
    ax.set_xticks(minor, minor=True)
    ax.set_xticklabels(minor_lab, minor=True, fontsize=5.5, color=MUTED)
    ax.tick_params(axis="x", which="minor", length=0, pad=1)
    ax.tick_params(axis="x", which="major", length=0, pad=11)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"GPU{g}\n({lab[0]}{'refill' if lab[0]=='P' else 'ecode'})"
                        for lab, g in order], fontsize=7)
    ax.set_ylabel("KV pool (GB)")
    ax.set_title("Borrowed cache lands where the pool was empty", pad=16)
    # Bar height = serving pool + park pools on that GPU, i.e. all of its KV-capable
    # memory. The two arms come out nearly equal in total (22.64 vs 22.77 GB on a decode
    # GPU) because a park pool is carved from the same HBM the serving pool would have
    # taken -- which is the honest picture: nothing was conjured, it was re-partitioned.
    ax.set_ylim(0, max(r["cap"] + r["parked"] for r in B + O) * 1.02)
    ax.legend(handles=[Patch(facecolor=C_OWN, edgecolor=INK, lw=0.4, label="own cache"),
                       Patch(facecolor=C_PARK, edgecolor=INK, lw=0.4, label="parked (borrowed)"),
                       Patch(facecolor=C_FREE, edgecolor=MUTED, lw=0.5, label="unused")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=6.5, handlelength=1.2, columnspacing=1.4, borderaxespad=0.35)
    style_axes(ax)

    # Right: what borrowing bought, with every repeat's value drawn on the bar. The bar is
    # the median; the dots are the runs. A reader can see at a glance whether the gap
    # between the two bars is bigger than the scatter within each of them.
    for x, s, c in ((-0.19, sb, C_OWN), (0.19, so, C_PARK)):
        ax2.bar(x, s["hit"], width=0.36, color=c, edgecolor=INK, lw=0.4, zorder=3)
        ax2.scatter([x] * s["n"], s["all"]["hit"], s=7, color=INK, zorder=5,
                    linewidths=0, alpha=0.85)
        ax2.text(x, max(s["all"]["hit"]), f"{s['hit']:.1f}", ha="center", va="bottom",
                 fontsize=7, color=INK)
    # Room to the right of the bars for the TTFT block; at the old 1.5 the range
    # annotations ran back over the bars themselves.
    ax2.set_xlim(-0.55, 2.6)
    ax2.set_xticks([0])
    ax2.set_xticklabels([f"park-fetch hit rate (%)\nmedian of {sb['n']} runs"], fontsize=7)
    ax2.set_ylim(0, max(sb["hit"], so["hit"]) * 1.5)
    ax2.set_ylabel("%")
    ax2.set_title("What it buys, and costs")
    y = ax2.get_ylim()[1]
    ax2.text(1.5, y * 0.97, f"TTFT (s), median of {sb['n']}", fontsize=6.5, color=INK,
             ha="center", va="top")
    # Every TTFT line carries the baseline's own range across repeats. The single-run
    # version of this panel reported "p95 3.3x better"; repeated, the baseline's p95 alone
    # spanned 1.45-6.58 s and the sign of the difference flipped twice. A tail number
    # without its spread beside it is not reportable at this n.
    for j, k in enumerate(("p50", "p95", "p99")):
        lo, hi = min(sb["all"][k]), max(sb["all"][k])
        worse = so[k] > sb[k]
        unstable = (hi - lo) > abs(so[k] - sb[k])
        ax2.text(1.5, y * (0.84 - 0.15 * j),
                 f"{k}  {sb[k]:.2f} → {so[k]:.2f}"
                 + (f"   (base {lo:.1f}–{hi:.1f})" if unstable else ""),
                 fontsize=6.0,
                 color=MUTED if unstable else (PALETTE["recompute"] if worse else INK),
                 ha="center", va="top")
    # Derived, never remembered. The previous hardcoded caption survived a configuration
    # change and asserted a tail win that three repeats then contradicted.
    hit_consistent = (sb["n"] > 1
                      and all(o > b for b, o in zip(sb["all"]["hit"], so["all"]["hit"]))
                      and abs(so["hit"] - sb["hit"]) > (max(sb["all"]["hit"])
                                                        - min(sb["all"]["hit"])))
    tail_noisy = (max(sb["all"]["p95"]) - min(sb["all"]["p95"])) > abs(so["p95"] - sb["p95"])
    cap = ("hit rate holds" if hit_consistent else "hit rate varies by run") + ";\n" + \
          ("tail is inside run-to-run noise" if tail_noisy
           else ("tail worsens" if so["p95"] > sb["p95"] else "tail improves"))
    ax2.text(1.5, y * 0.36, cap, fontsize=5.8, color=MUTED, ha="center", va="top",
             style="italic")
    style_axes(ax2)

    print(f"  park-fetch hit rate  {sb['hit']:.1f} -> {so['hit']:.1f} "
          f"(base range {min(sb['all']['hit']):.1f}-{max(sb['all']['hit']):.1f})")
    for k in ("p50", "p95", "p99"):
        print(f"  TTFT {k}  {sb[k]:.3f} -> {so[k]:.3f}   base range "
              f"{min(sb['all'][k]):.2f}-{max(sb['all'][k]):.2f}   ours range "
              f"{min(so['all'][k]):.2f}-{max(so['all'][k]):.2f}")

    fig.tight_layout(pad=0.5)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

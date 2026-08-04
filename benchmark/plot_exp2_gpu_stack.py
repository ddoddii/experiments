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
import json
import os

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
            "p50": s["ttft_p50_s"], "p95": s["ttft_p95_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    # PD_LAYOUT=b physical order, passed rather than inferred so a layout-a run cannot be
    # silently mislabelled.
    ap.add_argument("--order", default="P0:0,D0:1,P1:2,D1:3")
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=2.45)
    ap.add_argument("--out", default="results/exp2/fig_exp2")
    args = ap.parse_args()

    order = [(t.split(":")[0], int(t.split(":")[1])) for t in args.order.split(",")]
    B, O = arm_data(args.dir, BASE, order), arm_data(args.dir, OURS, order)
    sb, so = stats(args.dir, BASE), stats(args.dir, OURS)

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

    # Right: what borrowing bought and what it cost, on one axis each.
    ax2.bar(0 - 0.19, sb["hit"], width=0.36, color=C_OWN, edgecolor=INK, lw=0.4, zorder=3)
    ax2.bar(0 + 0.19, so["hit"], width=0.36, color=C_PARK, edgecolor=INK, lw=0.4, zorder=3)
    for x, v in ((-0.19, sb["hit"]), (0.19, so["hit"])):
        ax2.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax2.set_xlim(-0.6, 1.5)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["park-fetch hit rate (%)"], fontsize=7)
    ax2.set_ylim(0, max(sb["hit"], so["hit"]) * 1.5)
    ax2.set_ylabel("%")
    ax2.set_title("What it buys, and costs")
    y = ax2.get_ylim()[1]
    ax2.text(0.95, y * 0.95, "TTFT (s)", fontsize=6.5, color=INK, ha="center", va="top")
    ax2.text(0.95, y * 0.80, f"p50  {sb['p50']:.2f} → {so['p50']:.2f}", fontsize=6.5,
             color=INK, ha="center", va="top")
    ax2.text(0.95, y * 0.65, f"p95  {sb['p95']:.2f} → {so['p95']:.2f}", fontsize=6.5,
             color=PALETTE["recompute"] if so["p95"] > sb["p95"] else INK,
             ha="center", va="top")
    # Colour and wording follow the DATA rather than a remembered conclusion. On the
    # capped configuration the tail regressed and this line said so; uncapped it improves,
    # and a hardcoded "the tail does not" would have quietly mislabelled the new result.
    worse = so["p95"] > sb["p95"]
    ax2.text(0.95, y * 0.44,
             "median improves,\nthe tail does not" if worse
             else f"tail improves too\n(p95 {sb['p95']/so['p95']:.1f}x better)",
             fontsize=5.8, color=MUTED, ha="center", va="top", style="italic")
    style_axes(ax2)

    fig.tight_layout(pad=0.5)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

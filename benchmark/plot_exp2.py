#!/usr/bin/env python3
"""
Exp 2 figure: placement follows headroom, what that buys, and what it costs.

Built on the DECISION LOG, not on M1/M2. Those two were computed from serving-pool
occupancy while parked KV lives in a separate pool no exported gauge observes, so they
could not move no matter how well placement worked -- park_pd's entire apparent M1 gain
(-5.03 GB) was the decode serving pool shrinking to make room for the park pool (-4.98
GB). See paper/exp2_results.md §3.

Three panels, because the honest story needs all three:

  (a) MECHANISM -- the occupancy of the GPUs the policy chose against the ones it
      rejected, at the instant of each choice. The rejected candidates are what make
      "placed where there was room" falsifiable: a run where every park lands somewhere
      idle proves nothing if every candidate was idle. Null models are drawn as
      reference lines, computed by replaying the same log.

  (b) BENEFIT -- cache hit rate. Only meaningful at C=32 with per-GPU capacity capped;
      at C=16 both arms missed an identical 179 requests, because the working set fitted
      in one local pool and there was nothing left to catch.

  (c) COST -- TTFT. Median improves, but 97.7% of parks are placed cross-GPU so nearly
      every fetch is a remote read, and the tail pays for it. Plotting p50 alone would
      report a win that the p95 contradicts.

사용:
  python benchmark/plot_exp2.py --dir results/exp2/pd_layoutb_c32_pergpu10k \
      --out results/exp2/fig_exp2
"""
import argparse
import csv
import glob
import json
import os
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

BASE, OURS = "park_local", "park_pd"
C_BASE, C_OURS = "#BFC4C9", PALETTE["park"]
C_REJ = PALETTE["recompute"]
INK, MUTED = PALETTE["ink"], PALETTE["muted"]


def load_decisions(d, arm):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, f"decisions_{arm}.*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass          # a line torn by SIGKILL
    return recs


def usage_split(recs, seed=0):
    """(chosen usages, rejected usages, null-model means). None telemetry counts as idle,
    the same convention the selector itself uses, so policy and nulls are scored alike."""
    rng = random.Random(seed)
    u = lambda c: 0.0 if c.get("use") is None else c["use"]
    chosen, rejected, nulls = [], [], {"random": [], "round_robin": [], "always_local": []}
    rr = {}
    for r in recs:
        cand, ch, src = r["cand"], str(r["chosen"]), str(r.get("src_gpu"))
        chosen.append(u(cand[ch]))
        rejected += [u(c) for g, c in cand.items() if g != ch]
        nulls["random"].append(u(cand[rng.choice(list(cand))]))
        keys = sorted(cand)
        i = rr.get(src, 0)
        nulls["round_robin"].append(u(cand[keys[i % len(keys)]]))
        rr[src] = i + 1
        if src in cand:
            nulls["always_local"].append(u(cand[src]))
    return chosen, rejected, {k: (st.mean(v) if v else None) for k, v in nulls.items()}


def fetch_stats(d, arm):
    rows = list(csv.DictReader(open(os.path.join(d, f"parked_{arm}.csv"))))
    f = lambda k: float(rows[-1].get(k, "") or 0)
    hits, miss = f("fetch_hits"), f("fetch_miss")
    return {"hits": hits, "miss": miss,
            "rate": 100 * hits / (hits + miss) if hits + miss else 0.0}


def bench(d, arm):
    s = json.load(open(os.path.join(d, f"bench_{arm}.json")))["summary"]
    return {k: s[f"ttft_{k}_s"] for k in ("p50", "p95")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=1.95)
    ap.add_argument("--out", default="results/exp2/fig_exp2")
    args = ap.parse_args()

    recs = load_decisions(args.dir, OURS)
    if not recs:
        raise SystemExit(f"no decision log for {OURS} in {args.dir}")
    chosen, rejected, nulls = usage_split(recs)
    fb, fo = fetch_stats(args.dir, BASE), fetch_stats(args.dir, OURS)
    tb, to = bench(args.dir, BASE), bench(args.dir, OURS)

    print(f"  parks {len(recs)}   chosen mean {st.mean(chosen):.3f}   "
          f"rejected mean {st.mean(rejected):.3f}   gap {st.mean(chosen)-st.mean(rejected):+.3f}")
    print(f"  nulls {nulls}")
    print(f"  hit rate {fb['rate']:.1f}% -> {fo['rate']:.1f}%   "
          f"miss {fb['miss']:.0f} -> {fo['miss']:.0f}")
    print(f"  ttft p50 {tb['p50']:.3f} -> {to['p50']:.3f}   p95 {tb['p95']:.3f} -> {to['p95']:.3f}")

    use_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(args.width, args.height))

    # (a) chosen vs rejected occupancy. Density, not counts: there are two rejected
    # candidates per park, so raw counts would make the rejected mass look twice as tall
    # for a reason that has nothing to do with the policy.
    bins = [i / 20 for i in range(21)]
    ax1.hist(rejected, bins=bins, density=True, color=C_REJ, alpha=0.55,
             label="rejected", zorder=2)
    ax1.hist(chosen, bins=bins, density=True, color=C_OURS, alpha=0.85,
             label="chosen", zorder=3)
    # Null models as reference lines. Merge any that land within 0.02 of each other --
    # random and round-robin differ by 0.003 here, so two separate labels would sit on
    # top of one another and read as a rendering fault rather than as two policies that
    # happen to agree.
    marks, order = [], sorted((v, k) for k, v in nulls.items() if v is not None)
    for v, k in order:
        if marks and abs(v - marks[-1][0]) < 0.02:
            marks[-1] = (marks[-1][0], marks[-1][1] + "/" + k.replace("_", "-"))
        else:
            marks.append((v, k.replace("_", "-")))
    ymax = max(h.max() for h in
               (ax1.hist(rejected, bins=bins, density=True, alpha=0)[0],
                ax1.hist(chosen, bins=bins, density=True, alpha=0)[0]))
    # Labels go in the VALLEY between the two modes, not below the axis: the distribution
    # is strongly bimodal, so the band between the peaks is empty and is the only place
    # a label neither covers data nor collides with the tick labels.
    for v, name in marks:
        ax1.axvline(v, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=4)
        ax1.text(v - 0.015, ymax * 0.52, name, rotation=90, fontsize=5.5, color=MUTED,
                 ha="right", va="center", zorder=5)
    ax1.set_xlabel("GPU KV usage at decision time")
    ax1.set_ylabel("density")
    ax1.set_title("(a) Placement follows headroom")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, ymax * 1.12)
    ax1.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.4,
               borderaxespad=0.1)
    style_axes(ax1)

    def pair(ax, vals, ylabel, title, fmt="{:.0f}"):
        xs = range(len(vals))
        for i, (lab, v, c) in enumerate(vals):
            ax.bar(i, v, width=0.62, color=c, edgecolor=INK, linewidth=0.35, zorder=3)
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=7,
                    color=INK, zorder=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([lab for lab, _, _ in vals], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(v for _, v, _ in vals) * 1.28)
        style_axes(ax)

    pair(ax2, [("Local", fb["rate"], C_BASE), ("Ours", fo["rate"], C_OURS)],
         "Cache Hit Rate (%)", "(b) Benefit", "{:.1f}")

    # (c) both percentiles, because they disagree. p50 improves and p95 does not, and a
    # panel showing only the first would report a win the second contradicts.
    w = 0.36
    for i, (lab, b, o) in enumerate((("p50", tb["p50"], to["p50"]),
                                     ("p95", tb["p95"], to["p95"]))):
        ax3.bar(i - w / 2, b, width=w, color=C_BASE, edgecolor=INK, linewidth=0.35,
                zorder=3, label="Local" if i == 0 else None)
        ax3.bar(i + w / 2, o, width=w, color=C_OURS, edgecolor=INK, linewidth=0.35,
                zorder=3, label="Ours" if i == 0 else None)
        for x, v in ((i - w / 2, b), (i + w / 2, o)):
            ax3.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["p50", "p95"], fontsize=7)
    ax3.set_ylabel("TTFT (s)")
    ax3.set_title("(c) Cost: the tail")
    ax3.set_ylim(0, max(tb["p95"], to["p95"]) * 1.3)
    ax3.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.2)
    style_axes(ax3)

    fig.tight_layout(pad=0.5)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

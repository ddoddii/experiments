#!/usr/bin/env python3
"""
Peak memory of the whole 2P2D deployment, stacked: prefill HBM + decode HBM + CPU DRAM,
one pair of bars (SGLang / Ours) per model.

WHAT COUNTS AS "CPU DRAM" -- the choice decides the headline, so it is written down:
  proc_pss_mb      <- USED. Proportional set size over the four server processes: shared
                      pages are charged once, so it is the right total for a multi-process
                      deployment.
  proc_rss_mb         rejected: sums shared pages four times (67-83 GB where PSS is 14-32).
  mem_used_mb         rejected: 123.9 GB on a 125 GB machine for every hicache run, i.e.
                      the page cache expanded to fill RAM. That is not a requirement.
  hicache_dir_mb      rejected as DRAM: it is hicache's L3 tier on /tmp, and it reached
                      182.7 GB on a 125 GB machine -- it is disk. Reported separately
                      because it is a real cost the park arm does not pay, but it does not
                      belong in a DRAM bar.

IS THE DRAM BAR REALLY CAPTURING HICACHE'S HOST POOL? Checked against what the config
reserves (pool_tokens x kv_bytes x hicache_ratio x 2 prefill nodes):
    llama8b   reserved 17.6 GiB   measured 16.4   93% resident
    llama13b  reserved 14.6 GiB   measured 13.9   95%
    qwen14b   reserved  7.3 GiB   measured  6.3   86%
So yes -- PSS tracks the host tier to within a tenth. The bar looks small for three
reasons, none of them measurement error:
  1. PSS charges shared pages once. Summing anonymous RSS over the 148 server processes
     gives 55-73 GB instead of 14-32, by counting the same pages repeatedly.
  2. These runs set HICACHE_RATIO=1.2 to fit a 125 GB machine, where SGLang's own default
     is 2.0. At the default the host pool would be 12-29 GiB rather than 7-18, so this
     figure UNDERSTATES hicache's DRAM cost relative to how it ships.
  3. The HBM side is four GPUs at ~42 GB each by mem-fraction-static, so anything
     host-side is a small share of the total no matter how it is measured.

WHAT THE HBM BARS DO AND DO NOT MEASURE: most of each GPU's HBM is reserved by
--mem-fraction-static at startup, so the bar heights are dominated by a configuration
knob rather than by demand. What the two arms genuinely differ by is the park pool
(prefill HBM) and hicache's host pool (DRAM). The totals are still worth plotting --
they are what an operator has to provision -- but the difference, not the height, is the
result.

사용:
  python benchmark/plot_peak_memory.py --root results/models \
      --models llama8b:Llama-3.1-8B llama13b:Llama-2-13B qwen14b:Qwen3-14B \
      --out results/models/fig_peak_memory
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

# PD_LAYOUT=a (start_2P_2D.sh default, which run_models_sweep.sh does not override):
# prefill on gpu0/gpu1, decode on gpu2/gpu3. Passed as a flag so a layout-b run cannot be
# silently mislabelled.
C_P = "#BFC4C9"
C_D = PALETTE["park"]
C_DRAM = "#FFFFFF"
INK, MUTED = PALETTE["ink"], PALETTE["muted"]
ARMS = [("hicache", "SGLang"), ("park", "Ours")]


def peak(root, model, arm, prefill_gpus, decode_gpus):
    path = os.path.join(root, model, f"mem_{arm}.csv")
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty")
    mx = lambda k: max(float(r.get(k, "") or 0) for r in rows) / 1024.0
    return {"prefill": sum(mx(f"gpu{g}_hbm_mb") for g in prefill_gpus),
            "decode": sum(mx(f"gpu{g}_hbm_mb") for g in decode_gpus),
            "dram": mx("proc_pss_mb"),
            "disk": mx("hicache_dir_mb"),
            "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/models")
    ap.add_argument("--models", nargs="+",
                    default=["llama8b:Llama-3.1-8B", "llama13b:Llama-2-13B",
                             "qwen14b:Qwen3-14B"],
                    help="dir:label pairs, in plot order")
    ap.add_argument("--prefill-gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--decode-gpus", nargs="+", type=int, default=[2, 3])
    # Same code path as the figure, so the deck cannot disagree with the PDF about which
    # GPUs are prefill or which column counts as DRAM.
    ap.add_argument("--export-json", default=None,
                    help="also write the drawn bars to JSON (for make_*_pptx.js)")
    ap.add_argument("--width", type=float, default=4.6)
    ap.add_argument("--height", type=float, default=3.1)
    ap.add_argument("--out", default="results/models/fig_peak_memory")
    args = ap.parse_args()

    models = [(m.split(":")[0], m.split(":", 1)[1] if ":" in m else m) for m in args.models]
    data = {}
    print(f"{'model':>12} {'arm':>8} {'P HBM':>7} {'D HBM':>7} {'DRAM':>7} {'TOTAL':>8}"
          f" {'hicache disk':>13}")
    for key, _ in models:
        for arm, _lab in ARMS:
            v = peak(args.root, key, arm, args.prefill_gpus, args.decode_gpus)
            data[(key, arm)] = v
            print(f"{key:>12} {arm:>8} {v['prefill']:>7.1f} {v['decode']:>7.1f} "
                  f"{v['dram']:>7.1f} {v['prefill'] + v['decode'] + v['dram']:>8.1f} "
                  f"{v['disk']:>11.1f} GB")
        h, p = data[(key, "hicache")], data[(key, "park")]
        th = h["prefill"] + h["decode"] + h["dram"]
        tp = p["prefill"] + p["decode"] + p["dram"]
        print(f"{'':>12} {'delta':>8} {p['prefill'] - h['prefill']:>+7.1f} "
              f"{p['decode'] - h['decode']:>+7.1f} {p['dram'] - h['dram']:>+7.1f} "
              f"{tp - th:>+8.1f}")

    if args.export_json:
        import json as _json
        _json.dump({
            "models": [{"key": k, "label": l,
                        "arms": {a: {"prefill": round(data[(k, a)]["prefill"], 3),
                                     "decode": round(data[(k, a)]["decode"], 3),
                                     "dram": round(data[(k, a)]["dram"], 3),
                                     "disk": round(data[(k, a)]["disk"], 3)}
                                 for a, _ in ARMS}} for k, l in models],
            "arm_labels": {a: lab for a, lab in ARMS},
            "prefill_gpus": args.prefill_gpus, "decode_gpus": args.decode_gpus,
            "dram_column": "proc_pss_mb",
        }, open(args.export_json, "w"), indent=1)
        print(f"[export] {args.export_json}")

    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(args.width, args.height))
    w, gap = 0.36, 0.06
    xt, xl = [], []
    for i, (key, label) in enumerate(models):
        for j, (arm, alab) in enumerate(ARMS):
            v = data[(key, arm)]
            x = i + (j - 0.5) * (w + gap)
            ax.bar(x, v["prefill"], width=w, color=C_P, edgecolor=INK, lw=0.4, zorder=3)
            ax.bar(x, v["decode"], width=w, bottom=v["prefill"], color=C_D,
                   edgecolor=INK, lw=0.4, zorder=3)
            ax.bar(x, v["dram"], width=w, bottom=v["prefill"] + v["decode"],
                   color=C_DRAM, edgecolor=INK, lw=0.5, zorder=3)
            xt.append(x)
            xl.append(alab)
        ax.text(i, -0.085 * max(sum((data[(k, a)][z] for z in ("prefill", "decode", "dram")))
                                for k, _ in models for a, _ in ARMS),
                label, ha="center", va="top", fontsize=7, transform=ax.transData)

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6.5, color=MUTED)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("Peak memory usage (GB)")
    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.legend(handles=[Patch(facecolor=C_P, edgecolor=INK, lw=0.4, label="Prefill HBM"),
                       Patch(facecolor=C_D, edgecolor=INK, lw=0.4, label="Decode HBM"),
                       Patch(facecolor=C_DRAM, edgecolor=INK, lw=0.5, label="CPU DRAM")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=6.5, handlelength=1.2, columnspacing=1.2, borderaxespad=0.3)
    style_axes(ax)

    # Caption derived, never remembered: the sign of the total flips across these models
    # (Ours is lower on the 8B and higher on both larger ones), so a fixed "Ours uses less
    # memory" would be false on two of the three bars in this very figure.
    deltas = {}
    for key, label in models:
        h, p = data[(key, "hicache")], data[(key, "park")]
        deltas[label] = ((p["prefill"] + p["decode"] + p["dram"])
                         - (h["prefill"] + h["decode"] + h["dram"]))
    wins = [l for l, x in deltas.items() if x < 0]
    if len(wins) == len(deltas):
        msg = f"Ours lower on all {len(deltas)} models"
    elif not wins:
        msg = f"Ours higher on all {len(deltas)} models"
    else:
        msg = f"Ours lower only on {', '.join(wins)}"
    ax.text(0.5, 1.0, "", transform=ax.transAxes)
    print(f"\n  total peak memory: " +
          "  ".join(f"{l} {x:+.1f} GB" for l, x in deltas.items()) + f"   -> {msg}")
    print("  DRAM saved:        " +
          "  ".join(f"{l} {data[(k, 'park')]['dram'] - data[(k, 'hicache')]['dram']:+.1f}"
                    for k, l in models))
    print("  HBM added:         " +
          "  ".join(f"{l} "
                    f"{(data[(k, 'park')]['prefill'] + data[(k, 'park')]['decode']) - (data[(k, 'hicache')]['prefill'] + data[(k, 'hicache')]['decode']):+.1f}"
                    for k, l in models))
    print("  hicache /tmp disk: " +
          "  ".join(f"{l} {data[(k, 'hicache')]['disk']:.0f} GB" for k, l in models)
          + "   (park writes none)")

    fig.tight_layout(pad=0.5)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

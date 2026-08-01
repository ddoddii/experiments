#!/usr/bin/env python3
"""
Cross-model small multiples: one panel per metric, x = model, grouped bars per arm.

The layout of Figures 13-16 in the reference paper: several narrow panels in a row, each
answering one question across the same model set, so a reader scans one row and sees
whether the effect holds as the model changes.

WHAT EACH PANEL MEANS, AND ONE THAT DOES NOT MEAN WHAT ITS NAME SUGGESTS

  cache hit rate   server-side prefix reuse ratio. Directly comparable across models.

  TTFT             median. Bigger models are slower in absolute terms, so read the RATIO
                   between arms within a model, not the height across models.

  prefill          prompt tokens SERVED FROM CACHE per second -- prefill the server did
                   not have to do. Higher is better, matching how a bar chart is read by
                   default and how the reference figure plots it. The `recompute_rate`
                   panel is the same information inverted (tokens that DID have to be
                   recomputed); it was the default once, and under the name "prefill
                   work" the best arm was read as the worst.
                   Neither is prompt_tokens/s: this is a closed-loop benchmark, every arm
                   is offered the same prompts and finishes in about the same wall time,
                   so prompt_tokens/s is fixed by the workload and comes out identical
                   across arms by construction (measured 421 / 406 / 415 t/s on
                   Llama-3.1-8B). A capability-style peak throughput needs a saturating
                   open-loop run -- see qps_sweep.py.

  host DRAM        peak AnonPages: system-wide, non-reclaimable, so it is what an
                   adjacent agent process actually cannot have.

Reads the table.json that collect_arm_metrics.py writes, one per model.

사용:
  python benchmark/plot_models.py \
      Llama-3.1-8B=results/exp1/sharegpt_p60000_c8_m1024/table.json \
      Qwen3-14B=results/exp1/sharegpt_qwen14b/table.json \
      Qwen3-30B=results/exp1/sharegpt_qwen30b/table.json \
      --out results/exp1/fig_models
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

# Same hue order as every other figure in this paper. Validated as a categorical trio
# (worst adjacent CVD dE 18.0 protan, 18.7 normal).
ARMS = [
    ("recompute", "Recompute", PALETTE["recompute"]),
    ("hicache",   "SGLang",    PALETTE["hicache"]),
    ("park",      "Ours",      PALETTE["park"]),
]

# key, axis label, scale, lower_is_better, ratio_vs
#
# ratio_vs names the arm the "N.NNx" label is computed against. It defaults to the BETTER
# of the two baselines, which is the honest choice for hit rate, TTFT and prefill work.
# It is NOT the honest choice for host DRAM: neither Recompute nor the GPU-only variant
# has a host tier at all, so such an arm wins that metric by definition and the label
# would read 0.98x -- as if the proposal saved nothing. The claim there is against the
# incumbent that actually offloads to host memory, so the mem panel is pinned to SGLang.
PANELS = {
    "hit":     ("cache_hit_rate_pct",  "Cache hit rate (%)",   1.0, False, None),
    "ttft":    ("ttft_p50_s",          "Median TTFT (s)",      1.0, True,  None),
    "prefill": ("prefill_served_tok_s", "Prefill throughput (t/s)", 1.0, False, None),
    "recompute_rate": ("prefill_work_tok_s", "Recomputed prefill (t/s)", 1.0, True, None),
    "mem":     ("peak_anonpages_gb",   "Host DRAM (GB)",       1.0, True,  "hicache"),
    "ttft95":  ("ttft_p95_s",          "P95 TTFT (s)",         1.0, True,  None),
    "recomp":  ("recomputed_tokens",   "Recomputed tokens (M)", 1e-6, True, None),
    "hbm":     ("peak_gpu_hbm_total_gb", "GPU HBM (GB)",       1.0, False, None),
}


def load(spec):
    label, sep, path = spec.partition("=")
    if not sep:
        label, path = os.path.basename(os.path.dirname(path or label)), (path or label)
    if not os.path.exists(path):
        print(f"[warn] missing {path} -- {label} will be blank")
        return label, {}
    with open(path) as fh:
        rows = json.load(fh)
    return label, {r["arm"]: r for r in rows}


def draw(ax, models, key, ylabel, scale, lower_better, ratio_vs, annotate):
    n = len(models)
    width = 0.8 / len(ARMS)
    xs = range(n)
    top = 0.0
    for i, (arm, arm_label, color) in enumerate(ARMS):
        vals = []
        for _, byarm in models:
            v = (byarm.get(arm) or {}).get(key)
            vals.append(v * scale if isinstance(v, (int, float)) else 0.0)
        top = max(top, max(vals) if vals else 0.0)
        off = (i - (len(ARMS) - 1) / 2) * width
        ax.bar([x + off for x in xs], vals, width * 0.92, label=arm_label,
               color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(list(xs))
    names = [m for m, _ in models]
    rot = 20 if (n > 2 and max(len(m) for m in names) > 8) else 0
    ax.set_xticklabels(names, fontsize=6.6, rotation=rot,
                       ha="right" if rot else "center",
                       rotation_mode="anchor" if rot else None)
    ax.set_ylabel(ylabel + (" \u2193" if lower_better else ""), fontsize=7.5)
    ax.set_ylim(0, top * 1.22 if top else 1)
    ax.grid(axis="x", visible=False)
    if annotate:
        # Improvement of GPU-first over the BETTER of the two baselines, per model.
        for xi, (_, byarm) in enumerate(models):
            pk = (byarm.get("park") or {}).get(key)
            if ratio_vs:
                base = [(byarm.get(ratio_vs) or {}).get(key)]
            else:
                base = [(byarm.get(a) or {}).get(key) for a, _, _ in ARMS if a != "park"]
            base = [b for b in base if isinstance(b, (int, float))]
            if not isinstance(pk, (int, float)) or not base or pk == 0:
                continue
            b = min(base) if (lower_better and not ratio_vs) else (
                max(base) if not lower_better else base[0])
            if b == 0:
                continue
            ratio = (b / pk) if lower_better else (pk / b)
            ax.annotate(f"{ratio:.2f}×", xy=(xi, top * 1.16), ha="center",
                        fontsize=6.6, fontweight="bold", color=PALETTE["ink"])
    style_axes(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="Label=path/to/table.json")
    ap.add_argument("--panels", nargs="+", default=["hit", "ttft", "prefill", "mem"],
                    choices=list(PANELS))
    ap.add_argument("--out", default="results/exp1/fig_models")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--no-ratio", action="store_true",
                    help="omit the NNx improvement labels above each model group")
    args = ap.parse_args()

    models = [load(s) for s in args.models]
    have = [m for m, d in models if d]
    if not have:
        print("[error] no table.json loaded")
        return
    print(f"models: {', '.join(have)}")
    for label, byarm in models:
        for arm, _, _ in ARMS:
            r = byarm.get(arm)
            if not r:
                continue
            print(f"  {label:<16} {arm:<8} hit={r.get('cache_hit_rate_pct')}% "
                  f"ttft={r.get('ttft_p50_s')}s prefill_work={r.get('prefill_work_tok_s')}t/s "
                  f"anon={r.get('peak_anonpages_gb')}GB")

    use_paper_style()
    fig, axes = plt.subplots(1, len(args.panels), squeeze=False,
                             figsize=(args.width, args.height))
    for ax, p in zip(axes[0], args.panels):
        key, ylabel, scale, lower, ratio_vs = PANELS[p]
        draw(ax, models, key, ylabel, scale, lower, ratio_vs, not args.no_ratio)
    axes[0][0].set_xlabel("")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(ARMS), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.05), columnspacing=2.0, handlelength=1.4)
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=[0, 0.06, 1, 1])
    savefig(fig, args.out)


if __name__ == "__main__":
    main()

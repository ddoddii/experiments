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
    # Recompute is the do-nothing control, so it is drawn in neutral grey: it is the
    # reference the other two are read against, not a competing design. Same convention
    # as the "Naive" bar in the reference figure.
    ("recompute", "Recompute", "#BFC4C9"),
    ("hicache",   "SGLang",    PALETTE["hicache"]),
    ("park",      "Ours",      PALETTE["park"]),
]

# ratio_vs names the arm the "N.NNx" label is computed against. It defaults to the BETTER
# of the two baselines, which is the honest choice for hit rate, TTFT and prefill work.
# It is NOT the honest choice for host DRAM: neither Recompute nor the GPU-only variant
# has a host tier at all, so such an arm wins that metric by definition and the label
# would read 0.98x -- as if the proposal saved nothing. The claim there is against the
# incumbent that actually offloads to host memory, so the mem panel is pinned to SGLang.
#
# norm_to divides every arm by that arm's own value PER MODEL. TTFT is the one metric
# whose absolute height is dominated by model size rather than by the mechanism -- a 13B
# model is slower than an 8B one no matter which arm runs it -- so an absolute axis
# invites reading across models, which says nothing. Normalising to Recompute puts every
# model on the same 1.0 baseline and makes the panel show only what the arms changed.
PANELS = {
    "hit":     dict(key="cache_hit_rate_pct",   label="Cache Hit Rate (%)",
                    lower_better=False, yticks=[0, 25, 50, 75, 100]),
    "ttft":    dict(key="ttft_p50_s",           label="Normalized TTFT",
                    lower_better=True, norm_to="recompute"),
    "ttft_abs": dict(key="ttft_p50_s",          label="Median TTFT (s)",
                     lower_better=True),
    "prefill": dict(key="prefill_served_tok_s", label="Throughput (t/s)",
                    lower_better=False),
    "recompute_rate": dict(key="prefill_work_tok_s", label="Recomputed prefill (t/s)",
                           lower_better=True),
    "mem":     dict(key="peak_anonpages_gb",    label="Host DRAM (GB)",
                    lower_better=True, ratio_vs="hicache"),
    "ttft95":  dict(key="ttft_p95_s",           label="P95 TTFT (s)", lower_better=True),
    "recomp":  dict(key="recomputed_tokens",    label="Recomputed tokens (M)",
                    scale=1e-6, lower_better=True),
    "hbm":     dict(key="peak_gpu_hbm_total_gb", label="GPU HBM (GB)", lower_better=False),
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


def draw(ax, models, spec, annotate):
    key = spec["key"]
    scale = spec.get("scale", 1.0)
    lower_better = spec.get("lower_better", False)
    ratio_vs = spec.get("ratio_vs")
    norm_to = spec.get("norm_to")
    n = len(models)
    # Bars touch inside a group and the groups stay well separated: a narrow group with
    # no internal gap reads as one unit per model, which is what the comparison is.
    group_w = 0.60
    width = group_w / len(ARMS)
    xs = range(n)
    top = 0.0

    # Per-model divisor, so normalisation never mixes one model's scale into another's.
    denom = []
    for _, byarm in models:
        d = (byarm.get(norm_to) or {}).get(key) if norm_to else None
        denom.append(d if isinstance(d, (int, float)) and d else None)

    for i, (arm, arm_label, color) in enumerate(ARMS):
        vals = []
        for mi, (_, byarm) in enumerate(models):
            v = (byarm.get(arm) or {}).get(key)
            v = v * scale if isinstance(v, (int, float)) else 0.0
            if norm_to:
                v = (v / denom[mi]) if denom[mi] else 0.0
            vals.append(v)
        top = max(top, max(vals) if vals else 0.0)
        off = (i - (len(ARMS) - 1) / 2) * width
        ax.bar([x + off for x in xs], vals, width, label=arm_label,
               color=color, edgecolor=PALETTE["ink"], linewidth=0.35, zorder=3)

    if norm_to:
        # Draw the baseline itself, so a bar below the line is unambiguously better.
        ax.axhline(1.0, color=PALETTE["ink"], ls="--", lw=0.6, zorder=4)

    ax.set_xticks(list(xs))
    names = [m for m, _ in models]
    rot = 12 if (n > 2 and max(len(m) for m in names) > 8) else 0
    ax.set_xticklabels(names, fontsize=6.0, rotation=rot,
                       ha="right" if rot else "center",
                       rotation_mode="anchor" if rot else None)
    ax.set_ylabel(spec["label"], fontsize=7.0, labelpad=2)
    if spec.get("yticks"):
        ax.set_yticks(spec["yticks"])
        ax.set_ylim(spec["yticks"][0], spec["yticks"][-1] * 1.20)
    else:
        ax.set_ylim(0, top * 1.22 if top else 1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(axis="x", visible=False)
    if annotate:
        # Improvement of GPU-first over the BETTER of the two baselines, per model.
        # Ratios are scale-free, so normalising a panel does not change them.
        ymin, ymax = ax.get_ylim()
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
            ax.annotate(f"{ratio:.2f}×", xy=(xi, ymin + (ymax - ymin) * 0.93),
                        ha="center", va="center", fontsize=5.8, fontweight="bold",
                        color=PALETTE["ink"])
    style_axes(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="Label=path/to/table.json")
    ap.add_argument("--panels", nargs="+", default=["hit", "ttft", "prefill", "mem"],
                    choices=list(PANELS))
    ap.add_argument("--out", default="results/exp1/fig_models")
    # Defaults are the IEEE two-column text width (7.16 in) at a height that keeps the
    # row shallow enough to sit above or below a paragraph rather than owning the page.
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=1.65)
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
        draw(ax, models, PANELS[p], not args.no_ratio)
    axes[0][0].set_xlabel("")
    # Legend above the row, as in the reference: at this height a bottom legend competes
    # with the rotated model names for the same strip of space.
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ARMS), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.045), columnspacing=1.8, handlelength=1.3,
               handletextpad=0.5)
    fig.tight_layout(pad=0.3, w_pad=1.0, rect=[0, 0, 1, 0.93])
    savefig(fig, args.out)


if __name__ == "__main__":
    main()

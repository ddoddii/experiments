import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = {
    "sglang_hicache": {
        "path": "results/sglang_hicache/bfcl_multiturn_results_2P_2D.json",
        "label": "sglang_hicache",
        "color": "#4C9BE8",
    },
    "vllm_2p2d": {
        "path": "results/vllm-ppd/bfcl_multiturn_results_2P_2D_vllmppd.json",
        "label": "vllm 2P_2D",
        "color": "#E86B4C",
    },
    "vllm_2p2pd": {
        "path": "results/vllm-ppd/bfcl_multiturn_results_2P_2pD_vllmppd.json",
        "label": "vllm-ppd 2P_2pD",
        "color": "#54C27D",
    },
}

METRICS = [
    ("avg_ttft_s",              "Avg TTFT (s)",         "lower is better", True),
    ("avg_tpot_s",              "Avg TPOT (s)",         "lower is better", True),
    ("avg_throughput_tok_per_s","Avg Throughput (tok/s)","higher is better", False),
]

# ── load summaries ──────────────────────────────────────────────────────────
summaries = {}
for key, cfg in DATA.items():
    with open(cfg["path"]) as f:
        summaries[key] = json.load(f)["summary"]

baseline_key = "vllm_2p2d"
order = ["sglang_hicache", "vllm_2p2d", "vllm_2p2pd"]
labels = [DATA[k]["label"] for k in order]
colors = [DATA[k]["color"] for k in order]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("SGLang HiCache vs vLLM PD — 2P2D / 2P2pD\n(BFCL v3 Multi-turn, 200 items, baseline = vllm 2P_2D)",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (metric_key, metric_label, note, lower_better) in zip(axes, METRICS):
    values = [summaries[k][metric_key] for k in order]
    baseline = summaries[baseline_key][metric_key]

    x = np.arange(len(order))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white", linewidth=1.2)

    # ── % delta boxes ───────────────────────────────────────────────────────
    for i, (bar, val, key) in enumerate(zip(bars, values, order)):
        if key == baseline_key:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.03,
                    "baseline", ha="center", va="bottom",
                    fontsize=8, color="#555555", style="italic")
            continue

        pct = (val - baseline) / baseline * 100
        # for "lower is better" metrics a negative pct is good (green), positive is bad (red)
        if lower_better:
            box_color = "#d4edda" if pct < 0 else "#f8d7da"
            text_color = "#155724" if pct < 0 else "#721c24"
        else:
            box_color = "#d4edda" if pct > 0 else "#f8d7da"
            text_color = "#155724" if pct > 0 else "#721c24"

        sign = "+" if pct >= 0 else ""
        label_text = f"{sign}{pct:.1f}%"

        y_pos = bar.get_height() + max(values) * 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                label_text, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=text_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color,
                          edgecolor=text_color, linewidth=0.8, alpha=0.9))

    # ── value labels on bars ────────────────────────────────────────────────
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{val:.4f}" if val < 0.1 else f"{val:.3f}",
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(f"{metric_label}\n({note})", fontsize=10, pad=12)
    ax.set_ylim(0, max(values) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

fig.tight_layout()
out_path = "results/comparison_2P2D.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")

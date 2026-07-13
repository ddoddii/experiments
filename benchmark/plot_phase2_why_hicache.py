#!/usr/bin/env python3
"""
Phase 2 종합 그림: "왜 KV-movement가 hicache를 못 이기는가"를 한 장으로.

Panel A (원인): 토큰당 비용 — prefill 재계산 vs KV host->GPU 로드. 재계산이 20-40배 비싸다.
                = prefix caching이 작동하는 이유이자, hicache가 '큰 이득(재계산 회피)'을 먹는 이유.
Panel B (결과): 강압박(BFCL pool 40k) TTFT — radix > hicache ≈ park. KV-movement(park)가 남긴
                이득(로드 은닉)은 너무 작아 hicache와 '동률'까지만; latency로 gap을 못 만든다.

색: 프로젝트 Okabe-Ito(색맹 안전). radix=#0072B2, park=#E69F00, hicache=#009E73.
출력: results/phase1_report/why_hicache_wins.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# --- Okabe-Ito ---
C_RADIX = "#0072B2"   # blue
C_PARK = "#E69F00"    # orange
C_HICACHE = "#009E73" # green
C_COMPUTE = "#D55E00" # vermillion (재계산)
C_LOAD = "#56B4E9"    # sky blue  (로드)
INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "figure.dpi": 130,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))
fig.suptitle("Why KV-tier-movement cannot beat hicache in PD disaggregation",
             fontsize=13, fontweight="bold", y=0.99)

# ===== Panel A: per-token cost (log) =====
# prefill compute: 2*8.03e9 FLOP/token / A6000 effective 70-155 TFLOP/s -> 0.10-0.23 ms/tok
# KV load: 128 KB/token / PCIe 25 GB/s -> 0.0052 ms/tok
comp_lo, comp_hi, comp_mid = 0.104, 0.229, 0.15
load_val = 0.0052
labels = ["prefill recompute\n(hicache avoids this)", "KV host→GPU load\n(the only slice\nmovement can hide)"]
xs = [0, 1]
axA.bar(0, comp_mid, width=0.55, color=C_COMPUTE, zorder=3,
        yerr=[[comp_mid - comp_lo], [comp_hi - comp_mid]], ecolor=MUTED, capsize=5)
axA.bar(1, load_val, width=0.55, color=C_LOAD, zorder=3)
axA.set_yscale("log")
axA.set_ylim(0.002, 0.6)
axA.set_xticks(xs); axA.set_xticklabels(labels, fontsize=9.5)
axA.set_ylabel("cost per token  (ms/token, log)")
axA.set_title("A.  Recompute is 20–40× costlier than loading", fontsize=11)
axA.text(0, comp_hi * 1.15, "~0.10–0.23", ha="center", va="bottom", fontsize=9.5, color=INK)
axA.text(1, load_val * 1.25, "~0.005", ha="center", va="bottom", fontsize=9.5, color=INK)
# ratio bracket
axA.annotate("", xy=(0, comp_mid), xytext=(1, load_val),
             arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
axA.text(0.5, 0.028, "20–40×", ha="center", va="center", fontsize=13, fontweight="bold",
         color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=MUTED, lw=0.8))
axA.grid(axis="y", color=GRID, lw=0.7, zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

# ===== Panel B: BFCL pool-40k (strong pressure) TTFT =====
# report.md §Figure 2 / results table (canonical): radix 1.83/0.38, hicache 1.38/0.74, park 1.35/0.74
arms = ["radix", "hicache", "park"]
ttft = [1.83, 1.38, 1.35]
reuse = [0.38, 0.74, 0.74]
cols = [C_RADIX, C_HICACHE, C_PARK]
xb = range(len(arms))
bars = axB.bar(xb, ttft, width=0.6, color=cols, zorder=3)
axB.set_xticks(list(xb)); axB.set_xticklabels(arms)
axB.set_ylabel("avg TTFT  (s, lower is better)")
axB.set_ylim(0, 2.15)
axB.set_title("B.  Under pressure (BFCL, pool 40k): park only ties hicache", fontsize=11)
for x, t, r in zip(xb, ttft, reuse):
    axB.text(x, t + 0.04, f"{t:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    axB.text(x, 0.08, f"reuse {r:.2f}", ha="center", va="bottom", fontsize=9, color="white", zorder=4)
# annotate deltas
axB.annotate("−26% vs radix; ties hicache (−2.5%)\nwin = host RAM 0 (spare GPU), not latency",
             xy=(2, 1.35), xytext=(1.55, 1.90),
             fontsize=8.8, color=INK, ha="center",
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
axB.grid(axis="y", color=GRID, lw=0.7, zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

fig.text(0.5, 0.005,
         "hicache captures the dominant win (recompute-avoidance, Panel A); the residual movement can add "
         "(load-hiding) is 20–40× smaller and prefix-length-independent → no latency gap (Panel B).",
         ha="center", va="bottom", fontsize=8.6, color=MUTED)

fig.tight_layout(rect=[0, 0.045, 1, 0.96])
out = "results/phase1_report/why_hicache_wins.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)

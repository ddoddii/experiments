#!/usr/bin/env python3
"""
Phase 1 보고용 그림 — idle KV parking의 catch-22.

Fig 1 (catch22_pressure.png): P GPU pool(=압박) 스윕에서 reuse와 TTFT.
  · 상단: reuse_ratio. radix가 pool 40k에서만 0.39로 떨어짐(회수 여지) → pool≥60k에선
    radix=park=hicache≈0.74로 수렴(회수 대상 없음). park은 어디서나 radix에 붙음.
  · 하단: TTFT. park≈radix 전 구간. hicache는 pool 40k에서만 이기고 이후 오버헤드.

Fig 2 (catch22_fetch_nospace.png): pool 40000에서 fetch 시도 747건의 결과 분해.
  hits 3.5% / already 37% / miss 28% / nospace 32% — nospace(병목 P GPU에 자리 없음)가
  fetch를 막는 지배 요인.

데이터 출처: results/head_to_head/h2h_p*/head_to_head_summary.json + pool40000 park DIAG.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Korean-capable font (labels/annotations are in Korean).
for _p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False  # 유니코드 마이너스 대신 ASCII

OUT = "results/phase1_report"
os.makedirs(OUT, exist_ok=True)

# --- Okabe-Ito 계열, CVD-safe (validator PASS, ΔE 51.6) -------------------
C_RADIX = "#0072B2"   # blue  — baseline(recompute)
C_PARK = "#E69F00"    # orange — 본 연구(초점)
C_HICACHE = "#009E73" # green — incumbent
INK = "#1a1a1a"
MUTED = "#6b7280"
GRID = "#e5e7eb"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white",
})

POOLS = [40000, 60000, 80000, 120000]
XL = [f"{p//1000}k" for p in POOLS]


def load():
    rows = {}
    for p in POOLS:
        d = json.load(open(f"results/head_to_head/h2h_p{p}_c8_d3/head_to_head_summary.json"))
        rows[p] = d["arms"]
    return rows


def series(rows, arm, key):
    return [rows[p][arm].get(key) for p in POOLS]


def fig1_pressure(rows):
    x = list(range(len(POOLS)))
    fig, (axr, axt) = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True,
                                   gridspec_kw={"hspace": 0.16})

    # ---- 상단: reuse ----
    for arm, c, lbl, lw in [("radix", C_RADIX, "radix (GPU prefix cache)", 2.0),
                            ("hicache", C_HICACHE, "hicache (host-DRAM fetch)", 2.0),
                            ("park", C_PARK, "park (GPU2 fetch, 본 연구)", 2.6)]:
        y = series(rows, arm, "reuse_ratio")
        axr.plot(x, y, "-o", color=c, lw=lw, ms=7, label=lbl, zorder=3,
                 mec="white", mew=1.2)
    axr.set_ylabel("KV reuse ratio\n(cached / prompt tokens)")
    axr.set_ylim(0.30, 0.82)
    axr.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    axr.set_axisbelow(True)
    axr.legend(loc="lower right", frameon=False, fontsize=9.5)
    axr.set_title("Spare-GPU parking ≈ host-DRAM hicache — 압박(P GPU pool) 스윕, BFCL C=8",
                  fontsize=12, fontweight="bold", color=INK, pad=10, loc="left")

    # ---- 하단: TTFT ----
    for arm, c, lw in [("radix", C_RADIX, 2.0), ("hicache", C_HICACHE, 2.0),
                       ("park", C_PARK, 2.6)]:
        y = series(rows, arm, "avg_ttft_s")
        axt.plot(x, y, "-o", color=c, lw=lw, ms=7, zorder=3, mec="white", mew=1.2)
    axt.set_ylabel("평균 TTFT (s)  ↓ 낮을수록 좋음")
    axt.set_xticks(x)
    axt.set_xticklabels(XL)
    axt.set_xlabel("P GPU KV pool 크기 (tokens)  →  압박 감소")
    axt.set_xlim(-0.25, len(POOLS) + 0.15)
    axt.set_ylim(1.10, 1.92)
    axt.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    axt.set_axisbelow(True)
    axt.legend(["radix", "hicache", "park"], loc="upper right", frameon=False, fontsize=9.5)

    fig.subplots_adjust(left=0.115, right=0.93, top=0.92, bottom=0.09)
    fig.savefig(f"{OUT}/catch22_pressure.png", dpi=200, facecolor="white")
    print(f"[saved] {OUT}/catch22_pressure.png")


def fig2_pressure_win(rows):
    """강압박(pool 40k) 결정 케이스: park/hicache가 축출된 prefix를 되찾아 radix를 이긴다."""
    r = rows[40000]
    arms = [("radix", C_RADIX, "radix (GPU prefix cache)"),
            ("hicache", C_HICACHE, "hicache (host-DRAM fetch)"),
            ("park", C_PARK, "park (GPU2 fetch, 본 연구)")]
    ys = list(range(len(arms)))[::-1]  # radix on top
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for y, (arm, c, lbl) in zip(ys, arms):
        ttft = r[arm]["avg_ttft_s"]
        reuse = r[arm]["reuse_ratio"]
        ax.barh(y, ttft, color=c, height=0.62, zorder=3, edgecolor="white", linewidth=2)
        ax.text(ttft - 0.03, y, f"{ttft:.2f}s", ha="right", va="center",
                fontsize=11, color="white", fontweight="bold")
        ax.text(0.02, y + 0.34, lbl, ha="left", va="bottom", fontsize=10, color=INK)
        ax.text(ttft + 0.03, y, f"reuse {reuse:.2f}", ha="left", va="center",
                fontsize=9.5, color=MUTED)
    rd = r["radix"]["avg_ttft_s"]
    for arm, c in [("hicache", C_HICACHE), ("park", C_PARK)]:
        d = (r[arm]["avg_ttft_s"] - rd) / rd * 100
        y = ys[[a[0] for a in arms].index(arm)]
        ax.text(r[arm]["avg_ttft_s"] + 0.42, y, f"({d:+.0f}% vs radix)",
                ha="left", va="center", fontsize=9.5, color=c, fontweight="bold")
    ax.set_xlim(0, rd * 1.28)
    ax.set_ylim(-0.5, len(arms) - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("평균 TTFT (s)  ↓ 낮을수록 좋음", fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("강압박(pool 40k): park·hicache가 축출된 prefix를 되찾아 radix를 26% 이긴다",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.85, bottom=0.16)
    fig.savefig(f"{OUT}/pressure_win_40k.png", dpi=200, facecolor="white")
    print(f"[saved] {OUT}/pressure_win_40k.png")


if __name__ == "__main__":
    rows = load()
    fig1_pressure(rows)
    fig2_pressure_win(rows)

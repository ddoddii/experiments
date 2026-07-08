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
    for arm, c, lbl, lw in [("radix", C_RADIX, "radix (recompute)", 2.0),
                            ("hicache", C_HICACHE, "hicache (host-DRAM fetch)", 2.0),
                            ("park", C_PARK, "park (GPU2 fetch, 본 연구)", 2.6)]:
        y = series(rows, arm, "reuse_ratio")
        axr.plot(x, y, "-o", color=c, lw=lw, ms=7, label=lbl, zorder=3,
                 mec="white", mew=1.2)
    # 회수 여지(=radix가 hicache보다 낮은 구간)만 pool40k. 강조.
    axr.annotate("회수 여지\n(radix가 evict → 재계산 발생)", xy=(0, 0.565),
                 xytext=(0.35, 0.52), fontsize=9.5, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axr.annotate("세 방식 수렴 (evict 없음 → 회수 대상 없음)", xy=(2, 0.744),
                 xytext=(1.15, 0.63), fontsize=9.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axr.set_ylabel("KV reuse ratio\n(cached / prompt tokens)")
    axr.set_ylim(0.30, 0.82)
    axr.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    axr.set_axisbelow(True)
    axr.legend(loc="lower right", frameon=False, fontsize=9.5)
    axr.set_title("Idle KV parking의 catch-22 — 압박(P GPU pool) 스윕, BFCL C=8",
                  fontsize=12.5, fontweight="bold", color=INK, pad=10, loc="left")

    # ---- 하단: TTFT ----
    for arm, c, lw in [("radix", C_RADIX, 2.0), ("hicache", C_HICACHE, 2.0),
                       ("park", C_PARK, 2.6)]:
        y = series(rows, arm, "avg_ttft_s")
        axt.plot(x, y, "-o", color=c, lw=lw, ms=7, zorder=3, mec="white", mew=1.2)
    # 직접 라벨: hicache는 분리돼 있어 개별 라벨, radix+park는 겹치므로 하나로 묶어 표기.
    axt.annotate("hicache", xy=(x[-1], series(rows, "hicache", "avg_ttft_s")[-1]),
                 xytext=(6, 0), textcoords="offset points", color=C_HICACHE,
                 fontsize=10, va="center", fontweight="bold")
    axt.annotate("park ≈ radix", xy=(x[-1], series(rows, "park", "avg_ttft_s")[-1]),
                 xytext=(6, -1), textcoords="offset points", color=C_PARK,
                 fontsize=10, va="center", fontweight="bold")
    axt.set_ylabel("평균 TTFT (s)  ↓ 낮을수록 좋음")
    axt.set_xticks(x)
    axt.set_xticklabels(XL)
    axt.set_xlabel("P GPU KV pool 크기 (tokens)  →  압박 감소")
    axt.set_xlim(-0.25, len(POOLS) + 0.15)
    axt.set_ylim(1.10, 1.92)
    axt.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    axt.set_axisbelow(True)
    # 주석은 상단 여백(y>1.5, x 0.6~2.6)에 배치 — 데이터 마크와 겹치지 않게.
    axt.annotate("park ≈ radix (전 구간)\nfetch가 순 TTFT 이득을 못 만듦",
                 xy=(1, series(rows, "park", "avg_ttft_s")[1]),
                 xytext=(0.55, 1.60), fontsize=9.5, color=C_PARK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=C_PARK, lw=0.8))
    axt.annotate("hicache: 강압박(40k)에서만 이득\n저압박에선 순수 오버헤드",
                 xy=(2.55, 1.383), xytext=(2.0, 1.70), fontsize=9.5, color=C_HICACHE,
                 ha="left", arrowprops=dict(arrowstyle="-", color=C_HICACHE, lw=0.8))

    fig.text(0.065, 0.012,
             "핵심: '회수 가치 있음'(강압박)과 '회수 자리 있음'(저압박)이 공존하지 않는다 "
             "→ park은 어느 pool에서도 radix를 못 이긴다.",
             fontsize=9, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.115, right=0.93, top=0.92, bottom=0.10)
    fig.savefig(f"{OUT}/catch22_pressure.png", dpi=200, facecolor="white")
    print(f"[saved] {OUT}/catch22_pressure.png")


def fig2_fetch(rows):
    # pool 40000 park DIAG: hits=26 already=278 miss=206 nospace=237 (of 747)
    # 좌→우 서사: 성공(hits) → fetch 불필요/불가한 정상 사유 → 병목(nospace, 빨강 펀치라인).
    cats = [("hits", 26, "#009E73", "fetch 성공", "white"),
            ("already", 278, "#9ca3af", "already\nP가 이미 보유 (정상)", INK),
            ("miss", 206, "#cbd5e1", "miss\nparked 없음", INK),
            ("nospace", 237, "#D55E00", "nospace\nP GPU 자리 없음", "white")]
    total = sum(c[1] for c in cats)
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    left = 0
    for name, n, color, lbl, txtc in cats:
        ax.barh(0, n, left=left, color=color, height=0.6, zorder=3,
                edgecolor="white", linewidth=2.0)
        pct = n / total * 100
        if n > 60:
            ax.text(left + n / 2, 0, f"{lbl}\n{n}  ({pct:.0f}%)", ha="center",
                    va="center", fontsize=10, color=txtc, fontweight="bold")
        else:  # hits: 작은 조각은 막대 아래에 라벨
            ax.text(left + n / 2, -0.45, f"{name} {n}\n({pct:.1f}%)", ha="center",
                    va="top", fontsize=9, color="#047857", fontweight="bold")
        left += n
    ax.set_xlim(0, total)
    ax.set_ylim(-0.95, 0.55)
    ax.axis("off")
    ax.set_title("강압박(pool 40k): fetch 시도 747건의 결과 — nospace 32%가 restore를 막는다",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    fig.text(0.02, 0.05,
             "핵심: fetch한 KV는 attention용으로 병목 P GPU를 다시 점유해야 함 "
             "→ 압박 구간엔 자리가 없다(nospace 32%). 저장은 offload돼도 restore는 병목이다.",
             fontsize=9, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.02, right=0.985, top=0.82, bottom=0.14)
    fig.savefig(f"{OUT}/catch22_fetch_nospace.png", dpi=200, facecolor="white")
    print(f"[saved] {OUT}/catch22_fetch_nospace.png")


if __name__ == "__main__":
    rows = load()
    fig1_pressure(rows)
    fig2_fetch(rows)

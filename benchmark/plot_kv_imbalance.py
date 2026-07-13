#!/usr/bin/env python3
"""
2P2D KV 점유 불균형 시계열 플롯 + 정량화 (Phase 2 slice-2 motivation).

kv_occupancy_timeseries.py가 남긴 CSV를 읽어:
  (위) 4개 GPU(P0/P1/D0/D1)의 KV 점유율 시계열.
  (아래) 매 순간의 불균형 = max-min 점유율 spread.
그리고 핵심 정량: **"한 GPU가 포화(>HI)인 동시에 다른 GPU는 여유(<LO)"인 시간 비율** =
라우터가 KV를 옮길 "기회"가 존재하는 시간 분율. 이게 크면 slice-2(pressure-balancing) 정당.

사용:
  python benchmark/plot_kv_imbalance.py --csv results/kv_ts/2p2d.csv \
      --hi 0.8 --lo 0.5
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito: prefill 파랑/하늘, decode 주황/빨강 (역할별 그룹 색)
COLORS = {"P0": "#0072B2", "P1": "#56B4E9", "D0": "#E69F00", "D1": "#D55E00"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"


def load(csv_path):
    rows = []
    labels = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line.rstrip("\n"))
    reader = csv.DictReader(rows)
    cols = reader.fieldnames
    labels = [c[:-4] for c in cols if c.endswith("_use")]
    data = {"t": []}
    for lab in labels:
        data[lab] = []
    for r in reader:
        try:
            data["t"].append(float(r["t_s"]))
        except (ValueError, KeyError):
            continue
        for lab in labels:
            v = r.get(f"{lab}_use", "")
            data[lab].append(float(v) if v not in ("", None) else None)
    return data, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--hi", type=float, default=0.8, help="saturation threshold")
    ap.add_argument("--lo", type=float, default=0.5, help="headroom threshold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, labels = load(args.csv)
    t = data["t"]
    if not t:
        raise SystemExit(f"no samples in {args.csv}")

    # per-instant spread + "opportunity" (a saturated GPU AND an idle GPU coexist)
    spread, opp = [], []
    for i in range(len(t)):
        vals = [data[l][i] for l in labels if data[l][i] is not None]
        if len(vals) >= 2:
            spread.append(max(vals) - min(vals))
            opp.append(1 if (max(vals) >= args.hi and min(vals) <= args.lo) else 0)
        else:
            spread.append(None); opp.append(0)
    valid_spread = [s for s in spread if s is not None]
    opp_frac = (sum(opp) / len(opp)) if opp else 0.0
    mean_spread = (sum(valid_spread) / len(valid_spread)) if valid_spread else 0.0
    max_spread = max(valid_spread) if valid_spread else 0.0

    print(f"\n=== 2P2D KV imbalance ({os.path.basename(args.csv)}) ===")
    print(f"  samples: {len(t)}   duration: {t[-1]:.0f}s")
    for l in labels:
        vv = [x for x in data[l] if x is not None]
        if vv:
            print(f"  {l}: mean={sum(vv)/len(vv):.2f}  max={max(vv):.2f}  min={min(vv):.2f}")
    print(f"  mean spread (max-min): {mean_spread:.2f}   max spread: {max_spread:.2f}")
    print(f"  OPPORTUNITY (some GPU>={args.hi} AND some GPU<={args.lo}): "
          f"{opp_frac*100:.1f}% of time")
    print("  -> 이 비율이 크면 라우터가 KV를 여유 GPU로 옮길 기회가 상시 존재 = slice-2 정당.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.4]})
    fig.suptitle("2P2D per-GPU KV occupancy over time — does imbalance exist?",
                 fontsize=13, fontweight="bold")
    for l in labels:
        ys = data[l]
        xs = [t[i] for i in range(len(t)) if ys[i] is not None]
        yv = [ys[i] for i in range(len(t)) if ys[i] is not None]
        ax1.plot(xs, yv, "-", color=COLORS.get(l, "#999999"), lw=1.6, label=l)
    ax1.axhline(args.hi, color=MUTED, lw=0.8, ls="--")
    ax1.axhline(args.lo, color=MUTED, lw=0.8, ls=":")
    ax1.text(t[-1], args.hi, f" sat {args.hi:g}", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax1.text(t[-1], args.lo, f" idle {args.lo:g}", color=MUTED, fontsize=8, va="top", ha="right")
    ax1.set_ylabel("KV pool usage (token_usage)")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, ncol=len(labels), fontsize=9, loc="upper center")
    # shade the "opportunity" instants
    for i in range(len(t)):
        if opp[i]:
            ax2.axvspan(t[i] - 0.25, t[i] + 0.25, color="#E69F00", alpha=0.25, lw=0)
    ax2.plot(t, [s if s is not None else 0 for s in spread], "-", color=INK, lw=1.4)
    ax2.set_ylabel("spread\n(max-min)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(0, 1.02)
    ax2.text(0.01, 0.9, f"opportunity {opp_frac*100:.0f}% of time (shaded)",
             transform=ax2.transAxes, fontsize=9, color=INK, va="top")
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    out = args.out or os.path.splitext(args.csv)[0] + "_imbalance.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

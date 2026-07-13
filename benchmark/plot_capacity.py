#!/usr/bin/env python3
"""
Phase 2 B — capacity sweep 분석/플롯.

run_capacity_sweep.sh가 arm×C 마다 남긴
  results/bfcl_multiturn_results_<TAG>_<arm>_c<C>.json  (벤치 summary)
  <OUTDIR>/metrics_<arm>_c<C>_delta.json                 (reuse delta)
를 모아 capacity 곡선을 그린다: TTFT vs C, throughput vs C (arm별 라인).

핵심 판정: 고정 pool에서 concurrency를 올릴 때
  - radix는 낮은 C에서 TTFT 폭발(축출 스래싱) = capacity 바닥.
  - hicache는 host DRAM으로 높은 C까지 버팀.
  - park가 hicache 곡선을 따라가면 = spare GPU로 같은 capacity를 host RAM 0으로.

사용:
  python benchmark/plot_capacity.py --outdir results/capacity/<TAG> \
     --tag <TAG> --arms "radix hicache park" --cvalues "8 12 16 24 32"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"radix": "#0072B2", "hicache": "#009E73", "park": "#E69F00",
          "park_nvlink": "#CC79A7", "decode_offload": "#D55E00"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect(tag, arm, C, outdir):
    bench = load_json(f"results/bfcl_multiturn_results_{tag}_{arm}_c{C}.json")
    delta = load_json(os.path.join(outdir, f"metrics_{arm}_c{C}_delta.json"))
    row = {"arm": arm, "C": C}
    if bench and "summary" in bench:
        s = bench["summary"]
        row["ttft"] = s.get("avg_ttft_s")
        row["tput"] = s.get("overall_throughput_tok_per_s")
        row["errors"] = s.get("error_items")
        row["success"] = s.get("success_items")
    if delta and delta.get("summary"):
        row["reuse"] = delta["summary"].get("reuse_ratio")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="radix hicache park")
    ap.add_argument("--cvalues", default="8 12 16 24 32")
    args = ap.parse_args()
    arms = args.arms.split()
    cvals = [int(x) for x in args.cvalues.split()]

    data = {a: [collect(args.tag, a, c, args.outdir) for c in cvals] for a in arms}

    # --- table ---
    print(f"\n{'arm':<9}{'C':>4}{'TTFT(s)':>10}{'tput':>9}{'reuse':>8}{'err':>5}")
    print("-" * 45)
    for a in arms:
        for r in data[a]:
            print(f"{a:<9}{r['C']:>4}{_f(r.get('ttft')):>10}{_f(r.get('tput'),1):>9}"
                  f"{_f(r.get('reuse')):>8}{_i(r.get('errors')):>5}")

    # --- plot: TTFT vs C, throughput vs C ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle(f"Capacity sweep at fixed GPU pool  ({args.tag})", fontsize=13, fontweight="bold")
    for a in arms:
        col = COLORS.get(a, "#999999")
        cs = [r["C"] for r in data[a] if r.get("ttft") is not None]
        ttfts = [r["ttft"] for r in data[a] if r.get("ttft") is not None]
        tputs = [r["tput"] for r in data[a] if r.get("tput") is not None]
        cs2 = [r["C"] for r in data[a] if r.get("tput") is not None]
        if ttfts:
            ax1.plot(cs, ttfts, "-o", color=col, label=a, lw=2, ms=7, zorder=3)
            ax1.text(cs[-1], ttfts[-1], f" {a}", color=col, fontsize=9, va="center")
        if tputs:
            ax2.plot(cs2, tputs, "-o", color=col, label=a, lw=2, ms=7, zorder=3)
            ax2.text(cs2[-1], tputs[-1], f" {a}", color=col, fontsize=9, va="center")

    ax1.set_xlabel("concurrency (sessions)"); ax1.set_ylabel("avg TTFT (s, lower better)")
    ax1.set_title("TTFT vs load — where each tier's capacity ends", fontsize=10.5)
    ax2.set_xlabel("concurrency (sessions)"); ax2.set_ylabel("overall throughput (tok/s, higher better)")
    ax2.set_title("Throughput vs load", fontsize=10.5)
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(args.outdir, "capacity_curves.png")
    fig.savefig(out, bbox_inches="tight", dpi=130)
    print(f"\n[saved] {out}")
    # host RAM per arm
    print("\nhost RAM used at ready (MB):")
    for a in arms:
        p = os.path.join(args.outdir, f"hostmem_{a}.txt")
        if os.path.exists(p):
            print(f"  {a}: {open(p).read().strip()} MB")


def _f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def _i(v):
    return str(v) if v is not None else "-"


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
동시성 축 선 그래프: 부하가 커질수록 victim cache 가 유리해지는가.

가로축 C, 세로축 recompute 대비 정규화 TTFT. 1.0 아래면 recompute 보다 빠르다는 뜻이다.

이 그림이 막대보다 나은 이유는 "이긴다/진다" 가 아니라 **어디서부터 이기는가** 를
보여주기 때문이다. victim cache 는 prefill 풀이 밀어낼 때만 일한다. 부하가 낮으면
워킹셋이 풀 안에 다 들어가서 밀려나는 게 없고, 그때는 어떤 victim cache 도 비용만
낸다 -- 그래서 곡선은 낮은 C 에서 1.0 근처(또는 위)에서 시작해 C 가 커지며 내려간다.
교차점이 곧 "이 배포에서 이 기법이 값을 하기 시작하는 부하" 이고, 그게 결과다.

전제 조건 하나: prefill 풀 P 를 모든 C 에서 고정해야 한다. C 마다 P 를 바꾸면 축이
둘이 되어 곡선이 무엇을 뜻하는지 말할 수 없다. (submit_bench.sh MODE=sweep 이 그렇게 한다.)

    python benchmark/plot_scaling.py --dirs 'results/a100/a100_bfcl_c*' \
        --out results/a100/fig_scaling

에러율이 높은 점은 그리지 않는다. 타임아웃된 요청은 평균에서 빠지는데 느린 것부터
빠지므로, 남은 평균은 실패한 arm 에 유리하게 편향된다 -- C=64 hicache 가 78% 실패하고도
TPOT 이 제일 좋아 보였던 것이 그 예다.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paperstyle import PALETTE, savefig, use_paper_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ARMS = [
    ("hicache", "SGLang (host DRAM)", PALETTE["hicache"], dict(marker="o", ls="--")),
    ("park_gpu", "Ours (idle GPU HBM)", PALETTE["park"], dict(marker="s", ls="-")),
    ("radix", "Radix only", PALETTE["both"], dict(marker="^", ls=":")),
]
MAX_ERROR_RATE = 0.05


def _summary(path):
    if not os.path.exists(path):
        return None
    try:
        j = json.load(open(path))
        return j.get("summary", j)
    except Exception:  # noqa: BLE001
        return None


def _point(d, arm):
    """(정규화 TTFT, 에러율). 못 쓰는 점이면 (None, 에러율)."""
    b = _summary(os.path.join(d, f"bench_{arm}.json"))
    base = _summary(os.path.join(d, "bench_recompute.json"))
    if not b or not base or not base.get("avg_ttft_s"):
        return None, None
    n, e = b.get("total_items") or 0, b.get("error_items") or 0
    rate = (e / n) if n else 0.0
    # recompute 자체가 실패한 C 는 기준선이 못 되므로 그 점 전체를 버린다.
    bn, be = base.get("total_items") or 0, base.get("error_items") or 0
    if bn and (be / bn) > MAX_ERROR_RATE:
        return None, rate
    if rate > MAX_ERROR_RATE:
        return None, rate
    return b["avg_ttft_s"] / base["avg_ttft_s"], rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True,
                    help="glob. 예: 'results/a100/a100_bfcl_c*'")
    ap.add_argument("--out", default="results/a100/fig_scaling")
    args = ap.parse_args()

    # 디렉터리 이름 끝의 c<숫자> 에서 동시성을 읽는다. .prev 는 제외.
    points = {}
    for d in sorted(glob.glob(args.dirs)):
        if d.endswith(".prev") or d.endswith("_probe") or not os.path.isdir(d):
            continue
        m = re.search(r"_c(\d+)$", os.path.basename(d.rstrip("/")))
        if not m:
            continue
        points[int(m.group(1))] = d
    if not points:
        print(f"동시성을 읽을 수 있는 디렉터리가 없다: {args.dirs}")
        return 1

    cs = sorted(points)
    print(f"찾은 점: C={cs}")

    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    dropped = []
    table = []

    for arm, label, color, style in ARMS:
        xs, ys = [], []
        for c in cs:
            y, rate = _point(points[c], arm)
            if y is None:
                if rate is not None and rate > MAX_ERROR_RATE:
                    dropped.append((c, label, rate))
                continue
            xs.append(c)
            ys.append(y)
            table.append((c, arm, y, rate))
        if xs:
            ax.plot(xs, ys, color=color, label=label, lw=1.4, ms=4, **style)

    ax.axhline(1.0, color="black", ls="-", lw=0.8)
    ax.text(cs[0], 1.02, "Recompute", fontsize=6.5, va="bottom", color="#444444")
    ax.set_xscale("log", base=2)
    ax.set_xticks(cs)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Normalized TTFT")
    ax.grid(color=PALETTE["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    fig.tight_layout()
    savefig(fig, args.out)

    for c, label, rate in dropped:
        print(f"[제외] C={c} {label}: {rate*100:.0f}% 실패 -> 점 없음 (TIMEOUT 을 올려라)")
    tsv = args.out + "_values.tsv"
    with open(tsv, "w") as f:
        f.write("concurrency\tarm\tnormalized_ttft\terr_rate\n")
        for c, arm, y, rate in sorted(table):
            f.write(f"{c}\t{arm}\t{y:.4f}\t{rate:.4f}\n")
    print(f"저장: {args.out}.pdf / .png\n수치: {tsv}")

    # 교차점을 말로도 남긴다. 그림에서 눈으로 읽은 값을 본문에 적다가 틀리기 쉽다.
    for arm, label, _, _ in ARMS:
        pts = [(c, y) for c, a, y, _ in sorted(table) if a == arm]
        below = [c for c, y in pts if y < 1.0]
        if below:
            print(f"  {label}: C={min(below)} 부터 recompute 보다 빠르다"
                  f" (최저 {min(y for _, y in pts):.2f}x)")
        elif pts:
            print(f"  {label}: 측정한 C 범위에서 recompute 를 넘지 못했다"
                  f" (최저 {min(y for _, y in pts):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

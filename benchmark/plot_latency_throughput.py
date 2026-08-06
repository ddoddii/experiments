#!/usr/bin/env python3
"""
지연-처리량 곡선: 가로 throughput (tokens/s), 세로 normalized latency (ms/token).

한 점이 하나의 부하 수준(동시성)이고, 선을 따라가면 부하가 올라간다. 이 형태가
normalized-TTFT-vs-C 보다 나은 점은 **같은 처리량에서 얼마나 느린가** 를 직접 읽을 수
있다는 것이다. 부하 축이 서로 다른 시스템을 비교할 때, 동시성을 가로축에 두면 "같은
동시성" 이 서로 다른 처리량을 뜻해서 비교가 어긋난다. 처리량을 가로축에 두면 그 문제가
사라지고, 곡선이 위로 꺾이는 지점이 곧 포화점이 된다.

    normalized latency = 전체 e2e 지연 합 / 전체 생성 토큰 합   (ms/token)
    throughput         = summary.overall_throughput_tok_per_s

TTFT 를 토큰 수로 나눠 분산시킨 값이라, TTFT 와 TPOT 을 하나로 합친 지표다. 디코드가
긴 요청에서는 TTFT 가 희석되므로, TTFT 만 볼 때와 결론이 다를 수 있다 -- 둘 다 그리고
차이가 나면 그 차이 자체를 설명해야 한다.

    python benchmark/plot_latency_throughput.py --dirs 'results/a100/a100_bfcl_c*' \
        --out results/a100/fig_latency_throughput

plot_scaling.py 와 같은 가드를 쓴다: 에러율 5% 초과 점은 빼고, prefill 풀이 다른 점은
같은 곡선에 올리지 않는다.
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
    ("recompute", "Recompute", "#9E9E9E", dict(marker="D", ls="-.")),
    ("hicache", "SGLang (host DRAM)", PALETTE["hicache"], dict(marker="o", ls="--")),
    ("park_gpu", "Ours (idle GPU HBM)", PALETTE["park"], dict(marker="s", ls="-")),
    ("radix", "Radix only", PALETTE["both"], dict(marker="^", ls=":")),
]
MAX_ERROR_RATE = 0.05


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def _prefill_pool(d):
    for f in glob.glob(os.path.join(d, "p1.*.digest.txt")):
        try:
            m = re.search(r"max_total_num_tokens=(\d+)", open(f).read())
            if m:
                return int(m.group(1))
        except OSError:
            pass
    return None


def _point(d, arm):
    """(throughput tok/s, normalized latency ms/tok, 에러율)."""
    j = _load(os.path.join(d, f"bench_{arm}.json"))
    if not j:
        return None
    s = j.get("summary", {})
    n, e = s.get("total_items") or 0, s.get("error_items") or 0
    rate = (e / n) if n else 0.0
    if rate > MAX_ERROR_RATE:
        return None, None, rate

    # 요청 전체 지연을 생성 토큰으로 나눈다. 턴 단위로 합산해야 -- item 평균을 다시
    # 평균내면 턴 수가 다른 item 들이 같은 가중치를 받아 값이 흔들린다.
    lat, tok = 0.0, 0
    for item in j.get("results", []):
        for t in item.get("turns", []):
            if not isinstance(t, dict):
                continue
            ot = t.get("output_tokens") or 0
            el = t.get("e2e_latency_s")
            if ot > 0 and el:
                lat += el
                tok += ot
    if not tok:
        return None, None, rate
    return s.get("overall_throughput_tok_per_s"), lat / tok * 1000.0, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", default="results/a100/fig_latency_throughput")
    args = ap.parse_args()

    points = {}
    for d in sorted(glob.glob(args.dirs)):
        if d.endswith(".prev") or d.endswith("_probe") or not os.path.isdir(d):
            continue
        m = re.search(r"_c(\d+)$", os.path.basename(d.rstrip("/")))
        if m:
            points[int(m.group(1))] = d
    if not points:
        print(f"동시성을 읽을 수 있는 디렉터리가 없다: {args.dirs}")
        return 1

    # 설정이 다른 점은 같은 곡선에 올리지 않는다 (plot_scaling.py 와 같은 이유).
    pools = {c: _prefill_pool(d) for c, d in points.items()}
    known = [v for v in pools.values() if v]
    common = max(set(known), key=known.count) if known else None
    for c, v in sorted(pools.items()):
        if v and common and v != common:
            print(f"[제외] C={c}: prefill pool={v} (이 곡선은 {common}) -> 설정이 달라 뺀다")
            points.pop(c, None)
    cs = sorted(points)
    print(f"찾은 점: C={cs}  (prefill pool={common})")

    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    table, dropped = [], []

    for arm, label, color, style in ARMS:
        xs, ys, ann = [], [], []
        for c in cs:
            r = _point(points[c], arm)
            if not r or r[0] is None:
                if r and r[2] and r[2] > MAX_ERROR_RATE:
                    dropped.append((c, label, r[2]))
                continue
            x, y, rate = r
            xs.append(x); ys.append(y); ann.append(c)
            table.append((c, arm, x, y, rate))
        if not xs:
            continue
        # 처리량 순이 아니라 **부하(C) 순** 으로 잇는다. 처리량으로 정렬하면 포화 후
        # 처리량이 되레 줄어드는 arm 의 선이 스스로를 되짚어 엉킨다. C 순으로 이으면
        # 그 되짚음이 "부하를 올렸는데 처리량이 떨어졌다" 라는 읽을 수 있는 궤적이 된다.
        ax.plot(xs, ys, color=color, label=label, lw=1.5, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.4, **style)

    # 각 점이 어느 동시성인지 표시한다. 처리량 축만 보면 부하 수준을 알 수 없다.
    base = [t for t in table if t[1] == "recompute"] or table
    for c, _, x, y, _ in sorted(base):
        ax.annotate(f"C={c}", (x, y), textcoords="offset points", xytext=(3, -9),
                    fontsize=5.5, color="#666666")

    ax.set_xlabel("Throughput (tokens / s)")
    ax.set_ylabel("Normalized latency\n(ms / token)")
    ax.grid(color=PALETTE["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0)
    fig.tight_layout()
    savefig(fig, args.out)

    for c, label, rate in dropped:
        print(f"[제외] C={c} {label}: {rate*100:.0f}% 실패 -> 점 없음")
    tsv = args.out + "_values.tsv"
    with open(tsv, "w") as f:
        f.write("concurrency\tarm\tthroughput_tok_s\tnorm_latency_ms_per_tok\terr_rate\n")
        for c, arm, x, y, rate in sorted(table):
            f.write(f"{c}\t{arm}\t{x:.2f}\t{y:.2f}\t{rate:.4f}\n")
    print(f"저장: {args.out}.pdf / .png\n수치: {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
컨텍스트 길이 축 2-패널: (a) Host->GPU 트래픽 (GB), (b) p95 TTFT (ms).

KV 캐시 연구에서 가장 자연스러운 축이다. 컨텍스트가 길수록 KV 가 커지고, "그걸 어디에
두느냐" 의 차이가 단조롭게 벌어진다 -- 처리량 축처럼 포화해서 멈추지 않는다.

(a) 가 이 설계의 주장을 가장 직접적으로 말한다. host DRAM 계층은 되읽을 때마다 PCIe 로
KV 를 끌어와야 하고, 그 양은 컨텍스트에 비례해 커진다. peer GPU HBM 에 둔 쪽은 그 트래픽이
아예 발생하지 않는다. 측정 출처:

    hicache   sglang:load_back_tokens_total (L2->L1 되읽기) x 토큰당 바이트
    park_gpu  park telemetry 의 host_bytes  (peer 에서 가져오므로 보통 0)
    recompute 0 (계층이 없다)

(b) 는 평균이 아니라 p95 다. hicache 의 write-through 는 축출 경로를 막아 스톨을 만드는데,
스톨은 꼬리에 몰리므로 평균은 그걸 감춘다.

    python benchmark/plot_context.py --dirs 'results/a100/a100_longctx_w*' \
        --out results/a100/fig_context

가로축 눈금은 PREFIX_WORDS 가 아니라 근사 토큰 수로 적는다 (x1.33). 논문에서 "context
length" 라고 쓰려면 단어가 아니라 토큰이어야 한다.
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
]
MAX_ERROR_RATE = 0.05
BYTES_PER_TOKEN = 131072      # 실측 (park telemetry 의 bytes_per_token)
WORDS_TO_TOKENS = 1.33        # 영문 대략치. 정확한 값은 벤치 로그의 프롬프트 토큰 수.


def _json(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def _host_gb(d, arm):
    """이 arm 이 host->GPU 로 실제로 끌어온 양 (GB)."""
    if arm == "park_gpu":
        # park 은 peer GPU 에서 가져온다. host 로 넘친 분량만 여기 잡힌다.
        for f in glob.glob(os.path.join(d, "parked_gpu*.park_gpu.json")):
            j = _json(f)
            if j:
                return (j.get("host_bytes") or 0) / 2**30
        return 0.0
    m = _json(os.path.join(d, f"metrics_{arm}_delta.json"))
    if not m:
        return None
    tok = (m.get("summary") or {}).get("load_back_tokens (host->GPU)")
    if tok is None:
        return None
    return tok * BYTES_PER_TOKEN / 2**30


def _p95_ms(d, arm):
    j = _json(os.path.join(d, f"bench_{arm}.json"))
    if not j:
        return None, None
    s = j.get("summary", {})
    n, e = s.get("total_items") or 0, s.get("error_items") or 0
    rate = (e / n) if n else 0.0
    if rate > MAX_ERROR_RATE:
        return None, rate
    p = s.get("ttft_p95_s") or s.get("window_ttft_p95_s")
    return (p * 1000.0 if p else None), rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", default="results/a100/fig_context")
    args = ap.parse_args()

    points = {}
    for d in sorted(glob.glob(args.dirs)):
        if d.endswith(".prev") or d.endswith("_probe") or not os.path.isdir(d):
            continue
        m = re.search(r"_w(\d+)$", os.path.basename(d.rstrip("/")))
        if m:
            points[int(m.group(1))] = d
    if not points:
        print(f"PREFIX_WORDS 를 읽을 수 있는 디렉터리가 없다: {args.dirs}")
        return 1
    ws = sorted(points)
    xs_tok = [int(w * WORDS_TO_TOKENS) for w in ws]
    print(f"찾은 점: PREFIX_WORDS={ws}  (약 {xs_tok} tokens)")

    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    table, dropped = [], []

    for arm, label, color, style in ARMS:
        gx, gy, lx, ly = [], [], [], []
        for w, xt in zip(ws, xs_tok):
            gb = _host_gb(points[w], arm)
            p95, rate = _p95_ms(points[w], arm)
            if rate and rate > MAX_ERROR_RATE:
                dropped.append((w, label, rate))
            if gb is not None:
                gx.append(xt); gy.append(gb)
            if p95 is not None:
                lx.append(xt); ly.append(p95)
            table.append((w, arm, gb, p95, rate))
        if gx:
            axes[0].plot(gx, gy, color=color, label=label, lw=1.5, ms=4.5,
                         markeredgecolor="white", markeredgewidth=0.4, **style)
        if lx:
            axes[1].plot(lx, ly, color=color, label=label, lw=1.5, ms=4.5,
                         markeredgecolor="white", markeredgewidth=0.4, **style)

    for ax, ylab in zip(axes, ["Host→GPU traffic (GB)", "p95 TTFT (ms)"]):
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel(ylab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs_tok)
        ax.set_xticklabels([f"{t//1000}k" if t >= 1000 else str(t) for t in xs_tok])
        ax.grid(color=PALETTE["grid"], lw=0.5)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=6.5)
    fig.tight_layout()
    savefig(fig, args.out)

    for w, label, rate in dropped:
        print(f"[제외] PREFIX_WORDS={w} {label}: {rate*100:.0f}% 실패")
    tsv = args.out + "_values.tsv"
    with open(tsv, "w") as f:
        f.write("prefix_words\tctx_tokens_approx\tarm\thost_gpu_gb\tp95_ttft_ms\terr_rate\n")
        for w, arm, gb, p95, rate in sorted(table):
            f.write(f"{w}\t{int(w*WORDS_TO_TOKENS)}\t{arm}\t{gb}\t{p95}\t{rate}\n")
    print(f"저장: {args.out}.pdf / .png\n수치: {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

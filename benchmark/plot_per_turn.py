#!/usr/bin/env python3
"""
턴 번호 축: 대화가 길어질수록 victim cache 가 벌어지는가.

가로축 턴 번호, 세로축 그 턴의 TTFT. 새로 돌릴 필요가 없다 -- 턴별 ttft_s 가 이미
모든 bench_*.json 에 들어 있다.

이 축이 이 설계에 잘 맞는 이유: BFCL 은 멀티턴이고, 턴이 진행될수록 재사용 가능한
prefix 가 길어진다. 1턴에는 되돌려줄 것이 없고, 5턴쯤 되면 앞 턴들의 KV 가 이미 풀에서
밀려나 victim cache 가 잡고 있어야 할 것이 쌓인다. 즉 턴 번호는 "이 요청이 캐시에
기대하는 정도" 의 축이고, arm 간 격차가 턴을 따라 벌어지면 그게 곧 메커니즘의 증거다.
컨텍스트 길이 축과 성격이 같지만, 합성 프롬프트가 아니라 데이터셋 자체에서 나온다.

주의: 턴마다 표본 수가 다르다. BFCL v3 multi_turn_base 200개의 턴 수 분포는

    1턴 3개 / 2턴 38 / 3턴 48 / 4턴 51 / 5턴 44 / 6턴 14 / 7턴 2

라서 뒤로 갈수록 표본이 급감한다 (턴7 은 2개). 표본이 적은 턴은 평균이 요동치므로
--min-n 미만은 그리지 않고, 각 점의 n 을 출력과 표에 남긴다. n 을 적지 않은 꼬리 평균은
그림에서 가장 흔한 거짓말이다.

    python benchmark/plot_per_turn.py --dir results/a100/a100_bfcl_c32_full200 \
        --out results/a100/fig_per_turn

--normalize 를 주면 같은 턴의 recompute 대비 비율로 그린다 (턴마다 절대 TTFT 가
크게 다르므로, 격차를 보려면 이쪽이 읽기 쉽다).
"""
import argparse
import json
import os
import statistics
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


def per_turn(path):
    """{턴번호(1부터): [ttft_s, ...]}. 실패한 arm 이면 None."""
    if not os.path.exists(path):
        return None
    try:
        j = json.load(open(path))
    except Exception:  # noqa: BLE001
        return None
    s = j.get("summary", {})
    n, e = s.get("total_items") or 0, s.get("error_items") or 0
    if n and (e / n) > MAX_ERROR_RATE:
        print(f"[제외] {os.path.basename(path)}: {e}/{n} 실패 -> 이 arm 은 그리지 않는다")
        return None
    out = {}
    for item in j.get("results", []):
        for t in item.get("turns", []):
            if not isinstance(t, dict):
                continue
            v = t.get("ttft_s")
            if v and t.get("turn") is not None:
                # 저장된 turn 은 0부터. 사람이 읽는 축은 1부터.
                out.setdefault(int(t["turn"]) + 1, []).append(float(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results/a100/fig_per_turn")
    ap.add_argument("--min-n", type=int, default=20,
                    help="이 미만의 표본을 가진 턴은 그리지 않는다")
    ap.add_argument("--normalize", action="store_true",
                    help="같은 턴의 recompute 대비 비율로 그린다")
    args = ap.parse_args()

    data = {arm: per_turn(os.path.join(args.dir, f"bench_{arm}.json"))
            for arm, _, _, _ in ARMS}
    if not any(data.values()):
        print(f"쓸 수 있는 bench_*.json 이 없다: {args.dir}")
        return 1

    base = data.get("recompute")
    turns = sorted({t for d in data.values() if d for t in d})
    keep = [t for t in turns
            if all(d is None or len(d.get(t, [])) >= args.min_n for d in data.values())
            and any(d and len(d.get(t, [])) >= args.min_n for d in data.values())]
    thin = [t for t in turns if t not in keep]
    if thin:
        ns = {t: max((len(d.get(t, [])) for d in data.values() if d), default=0) for t in thin}
        print(f"[제외] 표본 부족한 턴 (n<{args.min_n}): " +
              ", ".join(f"턴{t}(n={ns[t]})" for t in thin))
    if not keep:
        print("표본이 충분한 턴이 없다. --min-n 을 낮추거나 더 많은 item 으로 재실행하라.")
        return 1

    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    table = []

    for arm, label, color, style in ARMS:
        d = data.get(arm)
        if not d:
            continue
        xs, ys = [], []
        for t in keep:
            vals = d.get(t, [])
            if len(vals) < args.min_n:
                continue
            y = statistics.mean(vals)
            if args.normalize:
                bv = base.get(t, []) if base else []
                if not bv:
                    continue
                y /= statistics.mean(bv)
            xs.append(t)
            ys.append(y)
            table.append((t, arm, statistics.mean(vals), len(vals)))
        if xs:
            ax.plot(xs, ys, color=color, label=label, lw=1.5, ms=4.5,
                    markeredgecolor="white", markeredgewidth=0.4, **style)

    if args.normalize:
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_ylabel("TTFT / Recompute (same turn)")
    else:
        ax.set_ylabel("TTFT (s)")
    ax.set_xlabel("Turn number")
    ax.set_xticks(keep)
    ax.grid(color=PALETTE["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0)
    fig.tight_layout()
    savefig(fig, args.out)

    tsv = args.out + "_values.tsv"
    with open(tsv, "w") as f:
        f.write("turn\tarm\tmean_ttft_s\tn\n")
        for t, arm, v, n in sorted(table):
            f.write(f"{t}\t{arm}\t{v:.4f}\t{n}\n")
    print(f"저장: {args.out}.pdf / .png\n수치: {tsv}")

    # 격차가 턴을 따라 벌어지는지 말로도 남긴다.
    if base:
        for arm, label, _, _ in ARMS:
            if arm == "recompute" or not data.get(arm):
                continue
            rr = [(t, statistics.mean(data[arm][t]) / statistics.mean(base[t]))
                  for t in keep if data[arm].get(t) and base.get(t)]
            if len(rr) >= 2:
                print(f"  {label}: 턴{rr[0][0]} {rr[0][1]:.2f}x -> 턴{rr[-1][0]} {rr[-1][1]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

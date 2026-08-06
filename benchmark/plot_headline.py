#!/usr/bin/env python3
"""
헤드라인 4-패널 그림: Cache Hit Rate / Normalized TTFT / Normalized TPOT / Host DRAM.

각 arm 을 자기 설정으로 돌린 결과를 그대로 그린다. victim tier 크기를 억지로 맞추지
않는다 -- park pool 이 decode GPU 의 유휴 HBM 만큼이라는 게 설계의 주장이므로, 그걸
hicache 의 host tier 크기에 맞춰 깎으면 주장하는 시스템이 아닌 것을 재게 된다.
대신 자원 사용량을 별도 축(Host DRAM)으로 함께 보고한다. 그래야 "빠른데 host DRAM 도
덜 쓴다" 가 한 그림에서 읽히고, 용량이 다르다는 사실도 숨겨지지 않는다.

    python benchmark/plot_headline.py --dir results/a100/a100_bfcl_c32 \
        --label Llama-3.1-8B --out results/a100/fig_headline

여러 모델을 나란히 놓으려면 --dir 를 여러 번:

    python benchmark/plot_headline.py \
        --dir results/a100/llama8b:Llama-3.1-8B \
        --dir results/a100/qwen14b:Qwen3-14B \
        --out results/a100/fig_headline

그림에 들어가는 수치의 출처 (전부 실측 파일에서 읽는다. 손으로 넣는 값은 없다):
    Cache Hit Rate  metrics_<arm>_delta.json  summary.reuse_ratio
    TTFT / TPOT     bench_<arm>.json          summary.avg_ttft_s / avg_tpot_s
    Host DRAM       mem_<arm>.csv             proc_rss_mb 의 최대값
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paperstyle import PALETTE, savefig, use_paper_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# 왼쪽부터 그림에 놓이는 순서. (디렉터리의 arm 이름, 범례 이름, 색)
ARMS = [
    ("recompute", "Recompute", "#9E9E9E"),
    ("hicache", "SGLang", PALETTE["hicache"]),
    ("park_gpu", "Ours", PALETTE["park"]),
]


def _read_summary(path):
    if not os.path.exists(path):
        return None
    try:
        j = json.load(open(path))
        return j.get("summary", j)
    except Exception:  # noqa: BLE001
        return None


def _host_dram_gb(path):
    """이 arm 이 실제로 쓴 host DRAM 최대값 (GB)."""
    if not os.path.exists(path):
        return None
    best = 0.0
    for r in csv.DictReader(open(path)):
        v = r.get("proc_rss_mb")
        if v in ("", "None", None):
            continue
        try:
            best = max(best, float(v))
        except ValueError:
            pass
    return best / 1024 if best else None


# 이 비율을 넘게 실패한 arm 은 그리지 않는다. 실패한 요청은 평균에서 그냥 빠지는데,
# 느린 요청부터 타임아웃되므로 그 평균은 살아남은 빠른 요청들만의 평균이다 -- 즉 실패한
# arm 에 유리하게 편향된다. 실제로 C=64 hicache 는 200턴 전부 120초 타임아웃에 걸려
# 156/200 item 이 실패했는데, 보고된 TTFT 71s 는 완주한 44개만의 값이었고 TPOT 은
# 오히려 제일 좋아 보였다. 그 막대를 그리면 측정값처럼 읽힌다.
MAX_ERROR_RATE = 0.05


def collect(d):
    """한 디렉터리에서 arm 별 지표를 뽑는다. 없는 arm 은 None 으로 남긴다."""
    out = {}
    for arm, _, _ in ARMS:
        b = _read_summary(os.path.join(d, f"bench_{arm}.json"))
        m = _read_summary(os.path.join(d, f"metrics_{arm}_delta.json"))
        total = (b or {}).get("total_items") or 0
        errs = (b or {}).get("error_items") or 0
        rate = (errs / total) if total else 0.0
        ok = rate <= MAX_ERROR_RATE
        out[arm] = {
            "ttft": b.get("avg_ttft_s") if (b and ok) else None,
            "tpot": b.get("avg_tpot_s") if (b and ok) else None,
            "hit": (m.get("reuse_ratio") * 100) if m and m.get("reuse_ratio") is not None else None,
            "dram": _host_dram_gb(os.path.join(d, f"mem_{arm}.csv")),
            "err_rate": rate,
            "n": total,
            "suppressed": bool(b) and not ok,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True,
                    help="결과 디렉터리. 'path:라벨' 형식으로 라벨 지정 가능")
    ap.add_argument("--out", default="results/a100/fig_headline")
    args = ap.parse_args()

    groups = []
    for spec in args.dir:
        path, _, label = spec.partition(":")
        groups.append((label or os.path.basename(path.rstrip("/")), collect(path)))

    use_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.4))
    panels = [
        ("hit", "Cache Hit Rate (%)", False),
        ("ttft", "Normalized TTFT", True),
        ("tpot", "Normalized TPOT", True),
        ("dram", "Host DRAM (GB)", False),
    ]
    width = 0.26
    missing = []

    for ax, (key, ylabel, normalize) in zip(axes, panels):
        for gi, (glabel, data) in enumerate(groups):
            base = data["recompute"].get(key) if normalize else None
            for ai, (arm, legend, color) in enumerate(ARMS):
                v = data[arm].get(key)
                if v is None:
                    missing.append(f"{glabel}/{arm}/{key}")
                    continue
                # hit rate 는 recompute 가 0 이므로 막대를 그리지 않는다 (참조 그림과 동일).
                if key == "hit" and arm == "recompute":
                    continue
                y = (v / base) if (normalize and base) else v
                ax.bar(gi + (ai - 1) * width, y, width * 0.92, color=color,
                       edgecolor="black", linewidth=0.4)
            # 개선 배수를 위에 적는다: hit/dram 은 SGLang 대비, TTFT/TPOT 은 우리 대비 baseline
            s, o = data["hicache"].get(key), data["park_gpu"].get(key)
            if s and o:
                ratio = (s / o) if key in ("ttft", "tpot", "dram") else (o / s)
                ax.text(gi, ax.get_ylim()[1] * 0.94, f"{ratio:.2f}×",
                        ha="center", fontsize=7.5, fontweight="bold")
        if normalize:
            ax.axhline(1.0, color="black", ls="--", lw=0.7)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups], rotation=20, ha="right")
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.5)
        ax.set_axisbelow(True)

    # 범례는 proxy 로 만든다. 첫 패널에서 긁으면 recompute 가 빠진다 -- 그 패널은
    # hit rate 라 recompute 막대를 아예 그리지 않기 때문이다 (0% 라서).
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c, edgecolor="black", lw=0.4) for _, _, c in ARMS],
               [lab for _, lab, _ in ARMS],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    savefig(fig, args.out)

    # 에러율 때문에 뺀 arm 을 먼저, 크게 알린다. 이건 "데이터가 없다" 가 아니라
    # "이 설정에서 그 arm 은 완주하지 못했다" 는 결과이고, 그대로 보고해야 한다.
    for glabel, data in groups:
        for arm, legend, _ in ARMS:
            d = data[arm]
            if d.get("suppressed"):
                print(f"[제외] {glabel}/{legend}: {d['err_rate']*100:.0f}% 실패"
                      f" ({int(d['err_rate']*d['n'])}/{d['n']} items) -> TTFT/TPOT 막대 없음")
                print("       실패한 요청은 평균에서 빠지는데 느린 것부터 빠지므로,")
                print("       남은 평균은 그 arm 에 유리하게 편향된다. TIMEOUT 을 올려")
                print("       모든 arm 이 완주하게 한 뒤 다시 비교하라 (TIMEOUT=600).")
    if missing:
        # 조용히 빈 막대를 그리면 "그 arm 이 0" 처럼 보인다. 무엇이 없는지 말한다.
        print("[warn] 다음 값이 없어 막대를 그리지 않았다 (arm 이 실패했거나 파일이 없음):")
        for m in sorted(set(missing)):
            print(f"        {m}")
    print(f"저장: {args.out}.pdf / .png")

    # 그림에 들어간 숫자를 표로도 남긴다. 캡션에 설정을 적을 때 필요하고,
    # 그림만 보고 수치를 되읽는 실수를 막는다.
    tsv = args.out + "_values.tsv"
    with open(tsv, "w") as f:
        f.write("group\tarm\thit_pct\tttft_s\ttpot_s\thost_dram_gb\terr_rate\tn_items\n")
        for glabel, data in groups:
            for arm, _, _ in ARMS:
                d = data[arm]
                f.write(f"{glabel}\t{arm}\t{d['hit']}\t{d['ttft']}\t{d['tpot']}"
                        f"\t{d['dram']}\t{d.get('err_rate')}\t{d.get('n')}\n")
    print(f"수치: {tsv}")


if __name__ == "__main__":
    main()

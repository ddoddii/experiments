#!/usr/bin/env python3
"""
이 run 이 애초에 결과를 낼 수 있는 조건이었는지 확인한다.

hicache 도 idle-KV-parking 도 VICTIM CACHE 다. 둘 다 "prefill KV 풀에서 밀려난 것"을
받아서 나중에 돌려주는 구조다. 그러면 전제가 하나 있다: **풀이 실제로 밀어내야 한다.**
풀에 여유가 있으면 밀려나는 게 없고, victim cache 는 잡을 것이 없다. 그 상태에서 남는
것은 비용뿐이다 -- host 풀 할당, write-through 트래픽, park 풀이 먹는 HBM.

측정된 예 (A100 80GB, BFCL C=32):

    prefill pool cap 416,216 tokens,  peak used 158,971  ->  38% 밖에 안 찼다
    L3 prefetched 0,  /tmp/hicache 0 MB     (L2/L3 는 한 번도 쓰이지 않았다)
    reuse 74.6% -- 전부 L1(GPU radix) 히트. 즉 plain radix 로도 똑같이 나온다
    그런데 TTFT: recompute 17.3s  vs  hicache 31.9s   (1.84배 악화)

즉 "hicache 가 졌다"가 아니라 "hicache 가 이길 수 있는 상황이 아니었다"이다. A6000
48GB 에서는 같은 워크로드가 풀을 채웠지만, 카드가 80GB 로 커지면서 워크로드가 그대로면
압박이 사라진다. 하드웨어가 커졌는데 워크로드를 안 키운 것이 원인이다.

그래서 이 스크립트는 arm 별 최대 점유율을 뽑아서, 낮으면 크게 경고한다. 이 숫자를
보지 않고 TTFT 만 보면 "victim cache 는 효과가 없다"는 잘못된 결론에 도달한다.

사용:
    python scripts/slurm/check_pressure.py --dir results/a100/a100_bfcl_c32
"""
import argparse
import csv
import glob
import os

# 이 아래면 "풀이 차지 않았다" 로 본다. 60% 는 관대한 선이다 -- radix 트리는 LRU 로
# 밀어내므로 100% 근처에서만 실제 eviction 이 일어난다.
PRESSURE_FLOOR = 0.60


def col(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if v in ("", "None", None):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "occ_*.csv")))
    if not files:
        print(f"occ_*.csv 가 없다: {args.dir}")
        return 1

    print("=" * 72)
    print(" PREFILL KV POOL 압박 점검 -- victim cache 가 잡을 것이 있었는가")
    print("=" * 72)
    print(f"  {'arm':<22} {'capacity':>10} {'peak used':>10} {'peak util':>10}  {'hit':>6}")
    print("  " + "-" * 64)

    low = []
    peaks = []
    for f in files:
        arm = os.path.basename(f)[len("occ_"):-len(".csv")]
        rows = list(csv.DictReader(open(f)))
        # P0 이 없으면 (2P2D 등) 첫 prefill 컬럼을 찾는다.
        used_key = "P0_used_tok" if rows and "P0_used_tok" in rows[0] else None
        cap_key = "P0_cap_tok" if rows and "P0_cap_tok" in rows[0] else None
        if not used_key or not cap_key:
            print(f"  {arm:<22} <P0 컬럼 없음>")
            continue
        used, cap, hit = col(rows, used_key), col(rows, cap_key), col(rows, "P0_hit")
        if not used or not cap or max(cap) == 0:
            print(f"  {arm:<22} <데이터 없음>")
            continue
        util = max(used) / max(cap)
        flag = "" if util >= PRESSURE_FLOOR else "   <-- 차지 않았다"
        print(f"  {arm:<22} {max(cap):>10.0f} {max(used):>10.0f} {util * 100:>9.1f}%"
              f" {max(hit) if hit else 0:>6.3f}{flag}")
        peaks.append((max(used), arm))
        if util < PRESSURE_FLOOR:
            low.append((arm, util, max(cap), max(used)))

    print()
    if not low:
        print("  모든 arm 에서 풀이 충분히 찼다. victim cache 비교가 성립한다.")
        return 0

    print(f"  경고: {len(low)}개 arm 에서 prefill 풀이 {PRESSURE_FLOOR * 100:.0f}% 미만으로만 찼다.")
    print()
    print("  풀에 여유가 있으면 밀려나는 KV 가 없고, hicache 의 L2/L3 도 park 풀도")
    print("  받을 것이 없다. 남는 것은 비용뿐이므로 이 상태의 TTFT 비교는")
    print("  'victim cache 가 효과 없다' 가 아니라 '측정할 조건이 아니었다' 를 뜻한다.")
    print()
    print("  풀을 워크로드에 맞게 줄여서 다시 돌려라 (모든 arm 에 똑같이 적용해야 한다):")
    # 기준은 "가장 많이 찬 arm" 이다. 도중에 끊긴 arm 을 기준으로 삼으면 워킹셋을
    # 과소평가해서 터무니없이 작은 풀을 제안하게 된다 (실제로 그랬다: 중단된
    # park_host 의 54k 를 보고 27k 를 제안했는데, 완주한 hicache 는 159k 였다).
    peak_used, peak_arm = max(peaks)
    suggest = int(peak_used * 0.5 / 1000) * 1000
    print(f"    ./scripts/slurm/submit.sh MODE=full PREFILL_MAX_TOTAL_TOKENS={suggest} \\")
    print("        MOONCAKE_LD_FIX=system MC_FORCE_TCP=1")
    print()
    print(f"  ({peak_arm} 의 peak used 가 {peak_used:.0f} 로 가장 크므로, 그 절반이면 약 2배")
    print("   oversubscribe 된다. 카드가 A6000 48GB 에서 A100 80GB 로 커지는 동안")
    print("   워크로드가 그대로여서 압박이 사라진 것이 원인이다.)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

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


BYTES_PER_TOKEN = 131072  # 측정값 (park telemetry 의 bytes_per_token). 8B, 32 layers.


def plan(per_session, conc, oversub, card_gb, weights_gb=16.0):
    """압박이 '캐시가 이기는 구간' 에 오도록 knob 을 함께 계산한다.

    구간이 셋이고, 가운데만 쓸모가 있다:

      P >> W   밀려나는 게 없다. victim cache 는 잡을 게 없고 비용만 낸다.
               (A100 80GB 기본값이 여기였다: 점유 38%, hicache 가 recompute 보다 1.84배 나쁨)
      P ~< W   밀려난 것이 곧 '곧 다시 쓸 것' 이다. victim cache 가 그걸 받아서 돌려준다.
               <- 여기를 노려야 한다
      P << W   thrashing. victim tier 에서도 재사용 전에 쫓겨나고, 재계산이 지배한다.
               (C=64 + P=79k 는 4배 oversubscribe 라 여기로 간다)

    그리고 두 번째 조건이 하나 더 있다. 밀려난 양(overflow = W - P)을 victim tier 가
    담을 수 있어야 한다. hicache 의 host tier 는 P x HICACHE_RATIO 로 정해지므로,
    P 를 줄이면 host tier 도 같이 줄어든다 -- 압박을 만들려고 P 를 줄였는데 받아줄
    곳도 같이 줄어드는 함정이 있다. park pool 은 반대로 P 와 무관하게 지정된다.
    그래서 두 tier 를 같은 크기로 맞춰야 '용량' 이 아니라 '매체/소프트웨어' 를 비교하게 된다.
    """
    W = per_session * conc
    P = int(W / oversub / 1000) * 1000
    overflow = W - P
    # tier 는 두 조건을 동시에 만족해야 한다.
    #
    #   (1) 밀려난 양(overflow)을 여유 있게 담아야 한다  -> overflow x 1.3
    #   (2) sglang 은 host tier 가 device pool 보다 크기를 요구한다:
    #         memory_pool_host.py: assert self.size > device_pool.size
    #       즉 HICACHE_RATIO 는 1 미만이 될 수 없다. 이 제약을 무시하면 서버가
    #       assertion 으로 죽는다 (계산상 0.67 이 나올 수 있어서 실제로 걸린다).
    #       park pool 에는 이 제약이 없지만, 두 tier 를 같은 크기로 맞춰야
    #       '어디에 두느냐' 를 비교하게 되므로 park 쪽도 같은 값을 쓴다.
    tier = int(max(overflow * 1.3, P * 1.1) / 1000) * 1000
    ratio = round(tier / P, 2) if P else 0
    tier_gb = tier * BYTES_PER_TOKEN / 2**30
    # decode GPU 에 park pool 이 들어갈 자리가 있는가.
    free_gb = card_gb - weights_gb - tier_gb

    print()
    print("  ─── 캐시가 이기는 구간으로 맞추기 ─────────────────────────────")
    print(f"  가정: 세션당 {per_session:.0f} tokens (이 run 실측), C={conc},"
          f" 목표 oversubscribe {oversub}배")
    print()
    print(f"    워킹셋 W          = {per_session:.0f} x {conc} = {W:.0f} tokens")
    print(f"    prefill pool P    = W / {oversub} = {P} tokens   <- PREFILL_MAX_TOTAL_TOKENS")
    print(f"    밀려나는 양       = W - P = {overflow:.0f} tokens")
    _why = "밀려난 양 x1.3" if overflow * 1.3 >= P * 1.1 else "P x1.1 (sglang 이 host>device 를 요구)"
    print(f"    victim tier 크기  = {tier} tokens ({tier_gb:.1f} GB)   [{_why}]")
    print()
    print("  두 arm 의 victim tier 를 같은 크기로 맞춘다. 안 맞추면 '어디에 두느냐'가")
    print("  아니라 '얼마나 담느냐'를 비교하게 된다:")
    print(f"    park_gpu : PARK_POOL_TOKENS_PER_GPU={tier}")
    print(f"    hicache  : HICACHE_RATIO={ratio}   (host tier = P x ratio = {int(P * ratio)})")
    print()
    print(f"  decode GPU 여유: {card_gb:.0f}GB - 가중치 {weights_gb:.0f}GB"
          f" - park pool {tier_gb:.1f}GB = {free_gb:.1f}GB 가 decode KV 몫")
    if free_gb < 8:
        print("    [경고] 남는 게 너무 적다. C 를 낮추거나 oversubscribe 를 줄여라.")
    print()
    print("  park pool 이 커지면 decode pool 이 그만큼 줄어든다. 그 값은 다시 재야 하므로")
    print("  probe 를 먼저 돌린다:")
    print()
    print(f"    ./scripts/slurm/submit.sh CONCURRENCY={conc} \\")
    print(f"        PREFILL_MAX_TOTAL_TOKENS={P} PARK_POOL_TOKENS_PER_GPU={tier} \\")
    print(f"        HICACHE_RATIO={ratio} MOONCAKE_LD_FIX=system MC_FORCE_TCP=1")
    print()
    print("  probe 가 알려준 DECODE_MAX_TOTAL_TOKENS 를 넣어 본 실험:")
    print()
    print(f"    ./scripts/slurm/submit.sh MODE=full CONCURRENCY={conc} \\")
    print(f"        PREFILL_MAX_TOTAL_TOKENS={P} PARK_POOL_TOKENS_PER_GPU={tier} \\")
    print(f"        HICACHE_RATIO={ratio} DECODE_MAX_TOTAL_TOKENS=<probe 값> \\")
    print('        ARMS="recompute radix hicache park_gpu" \\')
    print("        MOONCAKE_LD_FIX=system MC_FORCE_TCP=1")
    print()
    print("  radix arm 을 꼭 넣어라. 압박이 생기면 hicache 의 L2/L3 가 처음으로 실제로")
    print("  동작하는데, 그때 '캐싱 자체의 이득' 과 'hicache 계층의 비용' 을 분리하는")
    print("  유일한 대조군이다.")
    print()
    print("  다음 run 에서 확인할 것 (이 순서로):")
    print("    1. peak util 이 100% 에 닿는가          -> 안 닿으면 P 를 더 줄인다")
    print("    2. park telemetry 의 fetch_hits > 0 인가 -> 0 이면 tier 가 작거나 재사용 전에 쫓겨난다")
    print("    3. 그 다음에 TTFT 를 본다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--plan-concurrency", type=int, default=None,
                    help="이 동시성에 맞춘 knob 조합을 계산한다")
    ap.add_argument("--oversubscribe", type=float, default=1.5,
                    help="워킹셋 / prefill 풀 비율 목표 (기본 1.5)")
    ap.add_argument("--card-gb", type=float, default=80.0)
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

    # ─── 동시성으로 압박을 만들려면 얼마나 필요한가 ───────────────────────────
    # "캐시 히트가 늘면 점유가 줄지 않나" 는 자주 나오는 오해다. 반대다: 히트는 그
    # prefix 노드를 살려두는 것이므로 KV 가 계속 상주한다. 점유를 줄이는 것은 히트가
    # 아니라 EVICTION 이다. 그래서 동시 세션이 늘면 서로 다른 prefix 가 그만큼 더
    # 상주하고, 점유는 대체로 선형으로 오른다. 실제로 이 run 에서 세션당 몫이
    # 일정하게 나오므로 (아래) 그 선형 가정을 데이터로 확인할 수 있다.
    conc = _concurrency(args.dir)
    if args.plan_concurrency and conc and peak_used:
        plan(peak_used / conc, args.plan_concurrency, args.oversubscribe, args.card_gb)
        return 2
    if conc and peak_used:
        per = peak_used / conc
        cap = max(c for _, _, c, _ in low) if low else 0
        print()
        print("  ─── 동시성을 올려서 압박을 만들려면 ───────────────────────────")
        print(f"  이 run 의 동시 세션당 몫: {peak_used:.0f} / C={conc} = 약 {per:.0f} tokens")
        print("  (BFCL 은 tool 정의가 매 턴 재전송되는데 item 마다 tool 조합이 달라서")
        print("   세션 간 공유가 약하다. 그래서 C 에 거의 비례해 점유가 오른다.)")
        print()
        print(f"  {'C':>6} {'예상 점유':>12} {'capacity 대비':>14}")
        for c2 in (conc, conc * 2, conc * 3, conc * 4):
            est = per * c2
            print(f"  {c2:>6} {est:>12.0f} {est / cap * 100 if cap else 0:>13.0f}%")
        need = int(cap / per) if per else 0
        print()
        print(f"  풀을 100% 채우려면 C≈{need}, 2배 oversubscribe 하려면 C≈{need * 2} 이 필요하다.")
        print("  BFCL 은 item 이 200개뿐이라 C 가 그쯤 되면 데이터셋 대부분이 동시에 떠서")
        print("  TTFT 가 사실상 큐 대기 시간이 된다 -- prefill 효과를 재려는 지표가")
        print("  다른 것을 재게 된다. 그래서 풀을 줄이는 쪽이 변수를 하나만 움직인다.")
    return 2


def _concurrency(d):
    """RUN_INFO.txt 또는 bench_*.json 에서 이 run 의 동시성을 읽는다."""
    import json
    ri = os.path.join(d, "RUN_INFO.txt")
    if os.path.exists(ri):
        for line in open(ri):
            if "concurrency=" in line:
                try:
                    return int(line.split("concurrency=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
    for f in glob.glob(os.path.join(d, "bench_*.json")):
        try:
            j = json.load(open(f))
            s = j.get("summary", j)
            if isinstance(s, dict) and s.get("concurrency"):
                return int(s["concurrency"])
        except Exception:  # noqa: BLE001
            continue
    return None


if __name__ == "__main__":
    raise SystemExit(main())

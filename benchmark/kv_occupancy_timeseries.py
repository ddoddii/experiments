#!/usr/bin/env python3
"""
2P2D GPU별 KV 점유 시계열 샘플러 (Phase 2 slice-2 motivation).

질문: NpNd에서 "같은 순간에 한 GPU는 KV 포화 직전, 다른 GPU는 여유"인 **불균형이 실재하는가?**
있으면 → 라우터 pressure-balancing(유휴 GPU로 KV 이동) 방향에 힘이 실린다.

각 서버(P0/P1/D0/D1)의 /metrics에서 KV 점유를 주기적으로 긁어 CSV로 append(+flush).
백그라운드로 띄워두고 벤치를 돌린 뒤 kill하면 됨 (증분 기록이라 중간에 죽어도 데이터 보존).

읽는 메트릭 (server당 gauge):
  sglang:cache_occupancy    ← **주지표**. (used + evictable) / capacity.
                              prefix cache가 붙잡고 있는 KV까지 포함한 실제 점유율.
  sglang:token_usage        압박(pressure) 지표. _get_token_info()가 evictable_size를
                              **빼기** 때문에, prefix로 가득 찬 prefill 풀도 ~2%로 보인다.
                              캐시에 관한 질문에 이걸 쓰면 조용히 0을 답한다.
  sglang:max_total_num_tokens  풀 용량 (여유 공간을 GB로 환산하는 데 필요)
  sglang:num_used_tokens    활성 참조 토큰 수 (token_usage와 같은 회계)
  sglang:num_running_reqs   현재 실행 중 요청 수

  ⚠ cache_occupancy는 이 fork에서 추가한 gauge다 (metrics/collector.py). 업스트림
    서버에는 없으므로 그 경우 token_usage로 폴백하고, 두 값을 모두 CSV에 기록한다.

사용:
  # 기본 2P2D 레이아웃 (P0:30000 P1:30001 D0:30002 D1:30003)
  python benchmark/kv_occupancy_timeseries.py --out results/kv_ts/2p2d.csv --interval 0.5 &
  ...벤치 실행...
  kill %1     # 또는 SAMPLER_PID

환경/인자:
  --targets "port:label:role ..."  기본 2P2D. role=P|D (플롯 색/그룹용)
  --interval  샘플 주기 초 (기본 0.5)
  --duration  최대 지속 초 (기본 0 = 무한, kill까지)
  --out       CSV 경로
"""
import argparse
import os
import time
import urllib.request

DEFAULT_TARGETS = "30000:P0:P 30001:P1:P 30002:D0:D 30003:D1:D"

# Pool CAPACITY, so free space can be reported in absolute tokens (and thus GB) and not
# only as a saturation fraction. A fraction alone cannot answer "was the headroom big
# enough to be worth borrowing?", which is the question the imbalance analysis exists to
# ask -- and the prefill pool is capped (--max-total-tokens) while decode's is not, so
# equal fractions on the two sides mean very different numbers of free tokens.
#
# The metric's name has moved between SGLang versions, so try several and take the first
# that resolves; the collector falls back to used/usage if none of them exist.
CAPACITY_METRICS = (
    "sglang:max_total_num_tokens",
    "sglang:max_total_tokens",
    "sglang:token_capacity",
)


def scrape_all(url, timeout=3.0):
    """Fetch /metrics ONCE and return {metric_name: float} for every gauge on the page.

    One request per target per tick instead of one per metric: the previous version
    opened a connection for each of the three metrics, which at 4 targets and 0.5 s was
    24 requests/s of self-inflicted load onto the servers being measured."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 (server busy / transient) -> empty this tick
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            out[name] = float(parts[1])
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=0.0)  # 0 = until killed
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="results/kv_ts/2p2d.csv")
    args = ap.parse_args()

    targets = []  # (port, label, role)
    for tok in args.targets.split():
        port, label, role = (tok.split(":") + ["", ""])[:3]
        targets.append((int(port), label, role or "?"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # header: t, then per-target occupancy / used / running / capacity / pressure
    cols = ["t_s"]
    for _, label, _ in targets:
        cols += [f"{label}_use", f"{label}_used_tok", f"{label}_running",
                 f"{label}_cap_tok", f"{label}_pressure"]
    f = open(args.out, "w", buffering=1)  # line-buffered -> survives kill
    f.write(",".join(cols) + "\n")
    # also stash the role map as a comment for the plotter
    f.write("# roles: " + " ".join(f"{lab}:{role}" for _, lab, role in targets) + "\n")

    # Record WHICH occupancy metric each server actually exposes, rather than leaving the
    # reader to infer it. When cache_occupancy is missing the sampler falls back to
    # token_usage and writes identical values into both columns -- which is also what a
    # server with a genuinely empty prefix cache produces. Those two cases have opposite
    # meanings ("the measurement is broken" vs "the cache really is on another tier") and
    # nothing downstream could tell them apart.
    src = {}
    for port, label, _ in targets:
        m = scrape_all(f"http://{args.host}:{port}/metrics")
        if not m:
            src[label] = "unreachable"
        elif "sglang:cache_occupancy" in m:
            src[label] = "cache_occupancy"
        else:
            src[label] = "token_usage"
    f.write("# metric: " + " ".join(f"{k}:{v}" for k, v in src.items()) + "\n")
    bad = sorted(k for k, v in src.items() if v == "token_usage")
    if bad:
        print(f"[kv-ts] WARNING: {bad} do not export sglang:cache_occupancy. Occupancy "
              f"for them will be token_usage, which EXCLUDES prefix-cached KV. Pull and "
              f"restart the SGLang source tree if the prefix cache matters.")

    print(f"[kv-ts] sampling {len(targets)} servers every {args.interval}s -> {args.out} "
          f"(metric: {src}; Ctrl-C / kill to stop)")
    t0 = time.time()
    n = 0
    try:
        while True:
            t = time.time() - t0
            row = [f"{t:.2f}"]
            for port, _, _ in targets:
                m = scrape_all(f"http://{args.host}:{port}/metrics")
                # PRIMARY: cache_occupancy = (used + evictable) / capacity. Prefer it over
                # token_usage, which subtracts the prefix cache and therefore reports a
                # prefill pool packed with retained prefixes as ~2% occupied -- the exact
                # reading that made the first M1 probe come back as a flat zero. Fall
                # back to token_usage only against a server too old to export it, and
                # record both so the difference stays visible in the CSV.
                use = m.get("sglang:cache_occupancy")
                pressure = m.get("sglang:token_usage")
                if use is None:
                    use = pressure
                used = m.get("sglang:num_used_tokens")
                run = m.get("sglang:num_running_reqs")
                cap = next((m[k] for k in CAPACITY_METRICS if k in m), None)
                # Derive capacity when the server exposes no capacity gauge. Only valid
                # while the pool is meaningfully occupied: at usage ~0 the ratio is
                # 0/0-ish and rounding turns it into noise or a divide-by-zero.
                if cap is None and pressure and used is not None and pressure > 0.02:
                    cap = used / pressure   # both come from token_usage's accounting
                row += ["" if use is None else f"{use:.4f}",
                        "" if used is None else f"{used:.0f}",
                        "" if run is None else f"{run:.0f}",
                        "" if cap is None else f"{cap:.0f}",
                        "" if pressure is None else f"{pressure:.4f}"]
            f.write(",".join(row) + "\n")
            n += 1
            if args.duration and t >= args.duration:
                break
            # keep cadence roughly fixed regardless of scrape latency
            time.sleep(max(0.0, args.interval - ((time.time() - t0) - t)))
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        print(f"[kv-ts] wrote {n} samples to {args.out}")


if __name__ == "__main__":
    main()

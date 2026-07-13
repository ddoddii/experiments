#!/usr/bin/env python3
"""
2P2D GPU별 KV 점유 시계열 샘플러 (Phase 2 slice-2 motivation).

질문: NpNd에서 "같은 순간에 한 GPU는 KV 포화 직전, 다른 GPU는 여유"인 **불균형이 실재하는가?**
있으면 → 라우터 pressure-balancing(유휴 GPU로 KV 이동) 방향에 힘이 실린다.

각 서버(P0/P1/D0/D1)의 /metrics에서 KV 점유를 주기적으로 긁어 CSV로 append(+flush).
백그라운드로 띄워두고 벤치를 돌린 뒤 kill하면 됨 (증분 기록이라 중간에 죽어도 데이터 보존).

읽는 메트릭 (server당 gauge):
  sglang:token_usage        GPU KV 풀 사용 비율 (0~1)   ← 주지표
  sglang:num_used_tokens    GPU KV 사용 토큰 수
  sglang:num_running_reqs   현재 실행 중 요청 수 (참고: prefill/decode 활동)

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
METRICS = ("sglang:token_usage", "sglang:num_used_tokens", "sglang:num_running_reqs")


def scrape_value(url, metric, timeout=3.0):
    """Return the float value of a prometheus gauge line `metric{...} value`, or None."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 (server busy / transient) -> None this tick
        return None
    val = None
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(metric):
            continue
        # metric may be `name value` or `name{labels} value`; take the last field.
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            try:
                val = float(parts[1])
            except ValueError:
                continue
    return val


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
    # header: t, then per-target token_usage / used_tokens / running
    cols = ["t_s"]
    for _, label, _ in targets:
        cols += [f"{label}_use", f"{label}_used_tok", f"{label}_running"]
    f = open(args.out, "w", buffering=1)  # line-buffered -> survives kill
    f.write(",".join(cols) + "\n")
    # also stash the role map as a comment for the plotter
    f.write("# roles: " + " ".join(f"{lab}:{role}" for _, lab, role in targets) + "\n")

    print(f"[kv-ts] sampling {len(targets)} servers every {args.interval}s -> {args.out} "
          f"(Ctrl-C / kill to stop)")
    t0 = time.time()
    n = 0
    try:
        while True:
            t = time.time() - t0
            row = [f"{t:.2f}"]
            for port, _, _ in targets:
                base = f"http://{args.host}:{port}/metrics"
                use = scrape_value(base, "sglang:token_usage")
                used = scrape_value(base, "sglang:num_used_tokens")
                run = scrape_value(base, "sglang:num_running_reqs")
                row += ["" if use is None else f"{use:.4f}",
                        "" if used is None else f"{used:.0f}",
                        "" if run is None else f"{run:.0f}"]
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

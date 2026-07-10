#!/usr/bin/env python3
"""
Head-to-head 분석: idle-KV-parking(4a) vs host-DRAM hicache vs radix.

run_head_to_head.sh 가 각 arm 별로 남긴
  results/bfcl_multiturn_results_<TAG>_<arm>.json   (벤치 summary)
  <OUTDIR>/metrics_<arm>_delta.json                  (reuse/recompute delta)
를 병합해 하나의 비교표 + 판정을 만든다.

핵심 질문: park 저장 티어가 host-DRAM hicache 대비 end-to-end 이득이 있는가?
  - avg_ttft_s   낮을수록 좋음
  - reuse_ratio  cached/prompt (fetch 통합이 있으면 높다)
  - throughput   overall tok/s

사용:
  python benchmark/head_to_head_analyze.py --outdir results/head_to_head/<TAG> \
      --tag <TAG> --arms "radix hicache park"
"""
import argparse
import json
import os


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect_arm(tag, arm, outdir):
    """arm 하나의 벤치 summary + reuse delta 를 하나의 dict 로."""
    bench = load_json(f"results/bfcl_multiturn_results_{tag}_{arm}.json")
    delta = load_json(os.path.join(outdir, f"metrics_{arm}_delta.json"))
    row = {"arm": arm}
    if bench and "summary" in bench:
        s = bench["summary"]
        row.update({
            "avg_ttft_s": s.get("avg_ttft_s"),
            "avg_tpot_s": s.get("avg_tpot_s"),
            "overall_tput": s.get("overall_throughput_tok_per_s"),
            "wall_s": s.get("total_wall_time_s"),
            "errors": s.get("error_items"),
        })
    else:
        row["missing_bench"] = True
    if delta and delta.get("summary"):
        ds = delta["summary"]
        row.update({
            "reuse_ratio": ds.get("reuse_ratio"),
            "cached_tokens": ds.get("cached_tokens (reuse/fetch)"),
            "uncached_tokens": ds.get("uncached_tokens (recompute)"),
            "L2_host_used": ds.get("hicache_host_used_tokens (L2)"),
            "ttft_metric_s": ds.get("ttft_avg_s"),
        })
    else:
        row["missing_metrics"] = True
    return row


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct_delta(a, b):
    """a 대비 b 의 상대 변화(%). a=기준."""
    if a in (None, 0) or b is None:
        return None
    return round((b - a) / a * 100.0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="radix hicache park")
    args = ap.parse_args()

    arms = args.arms.split()
    rows = [collect_arm(args.tag, a, args.outdir) for a in arms]
    by = {r["arm"]: r for r in rows}

    # --- 표 출력 ---
    cols = [
        ("arm", "arm", 8, 0),
        ("avg_ttft_s", "TTFT(s)", 9, 4),
        ("avg_tpot_s", "TPOT(s)", 9, 4),
        ("overall_tput", "tok/s", 8, 2),
        ("reuse_ratio", "reuse", 7, 4),
        ("cached_tokens", "cached", 9, 0),
        ("L2_host_used", "L2_used", 9, 0),
        ("wall_s", "wall(s)", 8, 1),
    ]
    header = " ".join(f"{h:>{w}}" for _, h, w, _ in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        line = " ".join(f"{fmt(r.get(k), nd):>{w}}" for k, _, w, nd in cols)
        print(line)

    # --- 판정 (radix 기준으로 hicache/park 의 TTFT 개선율) ---
    print("\n판정 (baseline = radix):")
    base = by.get("radix", {})
    verdict = {"tag": args.tag, "arms": {r["arm"]: r for r in rows}, "comparison": {}}
    for arm in arms:
        if arm == "radix":
            continue
        r = by.get(arm, {})
        d_ttft = pct_delta(base.get("avg_ttft_s"), r.get("avg_ttft_s"))
        d_tput = pct_delta(base.get("overall_tput"), r.get("overall_tput"))
        verdict["comparison"][arm] = {"ttft_pct_vs_radix": d_ttft, "tput_pct_vs_radix": d_tput}
        sign = "개선" if (d_ttft is not None and d_ttft < 0) else "악화/무변화"
        print(f"  {arm:<8} TTFT {fmt(d_ttft,1)}% ({sign}), throughput {fmt(d_tput,1)}% vs radix")

    # --- 핵심 질문 1: park(fetch) 가 radix(GPU prefix cache) 를 이기는가 ---
    rx, pk = by.get("radix", {}), by.get("park", {})
    hi = by.get("hicache", {})
    rr = rx.get("reuse_ratio")
    pr = pk.get("reuse_ratio")
    hr = hi.get("reuse_ratio")
    if rx.get("avg_ttft_s") and pk.get("avg_ttft_s"):
        gap_rx = pct_delta(rx["avg_ttft_s"], pk["avg_ttft_s"])  # -면 park가 더 빠름
        verdict["park_vs_radix_ttft_pct"] = gap_rx
        print("\n핵심 결론 1 — park(fetch) vs radix(GPU prefix cache, 축출분만 recompute):")
        print(f"  park TTFT {fmt(pk['avg_ttft_s'])}s  vs  radix TTFT {fmt(rx['avg_ttft_s'])}s "
              f"→ park는 radix 대비 {fmt(gap_rx,1)}% ({'개선' if (gap_rx or 0)<0 else '무개선'})")
        print(f"  reuse_ratio: radix={fmt(rr)}  park={fmt(pr)}"
              + (f"  hicache={fmt(hr)}" if hr is not None else ""))
        if pr is not None and rr is not None:
            if pr - rr > 0.03:
                print("  → park reuse > radix reuse: fetch-on-hit 이 실제로 prefix-hit 을 "
                      "만들어 recompute 를 대체함. (fetch 동작 확인)")
            elif abs(pr - rr) <= 0.03:
                print("  → park reuse ≈ radix reuse: fetch 가 prefix-hit 을 못 만듦 "
                      "(park 미동작/미적중). p 로그의 GPU-fetch DIAG 확인 필요.")

    # --- 핵심 질문 2: park(fetch) vs hicache(host-DRAM fetch) ---
    if hi.get("avg_ttft_s") and pk.get("avg_ttft_s"):
        gap_hi = pct_delta(hi["avg_ttft_s"], pk["avg_ttft_s"])
        verdict["park_vs_hicache_ttft_pct"] = gap_hi
        print("\n핵심 결론 2 — park(GPU-fetch, PCIe) vs hicache(host-DRAM fetch, PCIe):")
        print(f"  park TTFT {fmt(pk['avg_ttft_s'])}s  vs  hicache TTFT {fmt(hi['avg_ttft_s'])}s "
              f"→ park는 hicache 대비 {fmt(gap_hi,1)}% (+면 더 느림, ≈0이면 동률)")

    # --- 핵심 질문 3: park(GEN=1, GPU) vs decode_offload(생성 KV를 host/disk로) ---
    do = by.get("decode_offload", {})
    if do.get("avg_ttft_s") and pk.get("avg_ttft_s"):
        gap_do = pct_delta(do["avg_ttft_s"], pk["avg_ttft_s"])
        verdict["park_vs_decode_offload_ttft_pct"] = gap_do
        print("\n핵심 결론 3 — park(생성 KV→GPU2) vs decode_offload(생성 KV→host/disk):")
        print(f"  reuse: park={fmt(pk.get('reuse_ratio'))}  decode_offload={fmt(do.get('reuse_ratio'))}"
              f"  (둘 다 생성 KV 잡으면 비슷해야)")
        print(f"  park TTFT {fmt(pk['avg_ttft_s'])}s  vs  decode_offload TTFT {fmt(do['avg_ttft_s'])}s "
              f"→ park는 {fmt(gap_do,1)}% (≈0이면 성능 대등 → park의 이점 = host RAM 0)")

    out = os.path.join(args.outdir, "head_to_head_summary.json")
    json.dump(verdict, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

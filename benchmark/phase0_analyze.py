#!/usr/bin/env python3
"""
Phase 0 baseline 집계기.

각 CACHE_MODE의 산출물을 모아 비교 테이블을 만든다:
  - results/phase0/metrics_<mode>_delta.json  (재계산 vs fetch, from scraper)
  - results/bfcl_multiturn_results_phase0_<mode>.json  (TTFT/TPOT/throughput, from benchmark)

출력:
  - results/phase0/phase0_summary.csv
  - results/phase0/phase0_summary.md
  - stdout에 요약 테이블

핵심 관전 포인트:
  reuse_ratio(캐시 재사용률)가 올라갈수록 avg TTFT가 떨어지면
  = "fetch가 recompute를 대체해 이득"이라는 Phase 0 가설이 확인된다.
"""
import argparse
import csv
import glob
import json
import os
import re

MODE_ORDER = ["none", "radix", "hicache_host", "hicache_file", "hicache_file_decode_offload"]


def load_json(path):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def per_turn_ttft(bench):
    """벤치마크 결과에서 turn 인덱스별 평균 TTFT 집계 (context 성장 → TTFT 성장 관찰용)."""
    acc, cnt = {}, {}
    for item in bench.get("results", []):
        tb = item.get("ttft_by_turn") or {}
        for k, v in tb.items():
            if v is None:
                continue
            acc[k] = acc.get(k, 0.0) + float(v)
            cnt[k] = cnt.get(k, 0) + 1
    return {int(k): round(acc[k] / cnt[k], 4) for k in sorted(acc, key=lambda x: int(x)) if cnt[k]}


def collect(outdir, results_dir):
    rows = []
    delta_files = glob.glob(os.path.join(outdir, "metrics_*_delta.json"))
    modes = []
    for f in delta_files:
        m = re.search(r"metrics_(.+)_delta\.json$", os.path.basename(f))
        if m:
            modes.append(m.group(1))
    # 순서 보정
    modes = [m for m in MODE_ORDER if m in modes] + [m for m in modes if m not in MODE_ORDER]

    for mode in modes:
        delta = load_json(os.path.join(outdir, f"metrics_{mode}_delta.json")) or {}
        summ = delta.get("summary", {})
        bench = load_json(os.path.join(results_dir, f"bfcl_multiturn_results_phase0_{mode}.json")) or {}
        bsum = bench.get("summary", {})
        rows.append({
            "mode": mode,
            "reuse_ratio": summ.get("reuse_ratio"),
            "recompute_ratio": summ.get("recompute_ratio"),
            "cache_hit_rate": summ.get("cache_hit_rate_gauge"),
            "prompt_tokens": summ.get("prompt_tokens"),
            "cached_tokens": summ.get("cached_tokens (reuse/fetch)"),
            "uncached_tokens": summ.get("uncached_tokens (recompute)"),
            "L3_prefetched": summ.get("L3_prefetched_tokens (storage fetch)"),
            "L2_host_used": summ.get("hicache_host_used_tokens (L2)"),
            "avg_ttft_s": bsum.get("avg_ttft_s"),
            "avg_tpot_s": bsum.get("avg_tpot_s"),
            "overall_tput": bsum.get("overall_throughput_tok_per_s"),
            "success": f"{bsum.get('success_items','?')}/{bsum.get('total_items','?')}",
            "per_turn_ttft": per_turn_ttft(bench) if bench else {},
        })
    return rows


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/phase0")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    rows = collect(args.outdir, args.results_dir)
    if not rows:
        print("no phase0 delta files found in", args.outdir)
        return

    cols = ["mode", "reuse_ratio", "recompute_ratio", "cache_hit_rate",
            "L3_prefetched", "L2_host_used", "avg_ttft_s", "avg_tpot_s",
            "overall_tput", "success"]

    # CSV
    csv_path = os.path.join(args.outdir, "phase0_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])

    # Markdown
    md = ["# Phase 0 baseline — recompute vs fetch\n",
          "| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    md.append("\n## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)\n")
    for r in rows:
        if r["per_turn_ttft"]:
            seq = ", ".join(f"t{k}:{v}" for k, v in r["per_turn_ttft"].items())
            md.append(f"- **{r['mode']}**: {seq}")
    md_text = "\n".join(md) + "\n"
    md_path = os.path.join(args.outdir, "phase0_summary.md")
    open(md_path, "w").write(md_text)

    print(md_text)
    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()

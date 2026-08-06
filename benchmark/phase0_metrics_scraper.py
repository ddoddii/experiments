#!/usr/bin/env python3
"""
Phase 0 baseline: SGLang /metrics 스크래퍼.

기존 hierarchical cache 경로가 multi-turn 워크로드에서
"재계산(recompute) vs fetch(reuse)"를 어디까지 커버하는지 정량화한다.

핵심 지표 (sglang이 --enable-metrics 시 노출):
  - sglang:prompt_tokens_total        전체 prompt(=prefill 입력) 토큰
  - sglang:cached_tokens_total        그 중 캐시에서 재사용된 토큰 (= fetch/reuse)
  - uncached = prompt - cached         실제로 재계산(recompute)된 토큰
  - sglang:prefetched_tokens_total    L3(storage)에서 당겨온 토큰 (Tier3 fetch)
  - sglang:cache_hit_rate             prefix cache hit rate (gauge)
  - sglang:hicache_host_used_tokens   L2(CPU DRAM host pool) 사용 토큰
  - sglang:hicache_host_total_tokens  L2 host pool 용량
  - sglang:num_used_tokens            GPU KV 사용 토큰 (gauge)

사용:
  # 벤치마크 직전/직후로 두 번 찍어 delta를 계산
  python benchmark/phase0_metrics_scraper.py --tag before --out results/phase0/metrics_<mode>_before.json
  python benchmark/phase0_metrics_scraper.py --tag after  --out results/phase0/metrics_<mode>_after.json
  # delta 계산
  python benchmark/phase0_metrics_scraper.py --delta \
      results/phase0/metrics_<mode>_before.json results/phase0/metrics_<mode>_after.json \
      --out results/phase0/metrics_<mode>_delta.json
"""
import argparse
import json
import os
import sys
import time
import urllib.request

# 관심 메트릭 이름 -> 종류 ("counter"는 delta, "gauge"는 스냅샷 값)
COUNTERS = [
    "sglang:prompt_tokens_total",
    "sglang:cached_tokens_total",
    "sglang:prefetched_tokens_total",
    "sglang:generation_tokens_total",
]
GAUGES = [
    "sglang:cache_hit_rate",
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    # L2(host DRAM) -> L1(GPU) 로 되읽은 토큰. 이게 곧 Host->GPU 트래픽이고,
    # host 계층에 KV 를 두는 설계가 실제로 PCIe 로 얼마나 옮기는지를 유일하게
    # 직접 말해 준다. park 은 peer GPU 에서 가져오므로 여기 거의 안 잡힌다.
    "sglang:load_back_tokens_total",
    "sglang:num_used_tokens",
    "sglang:token_usage",
]
# histogram: _sum / _count 로 평균을 구함
HISTOGRAMS = [
    "sglang:time_to_first_token_seconds",
    "sglang:uncached_prompt_tokens_histogram",
    "sglang:prompt_tokens_histogram",
]


def parse_prometheus(text):
    """Prometheus text exposition을 {metric_name: aggregated_value}로 축약.

    라벨(model_name 등)은 무시하고 같은 metric 이름끼리 합산한다.
    histogram은 _sum, _count suffix를 별도 키로 보존.
    """
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 형식: name{labels} value   또는   name value
        try:
            if "{" in line:
                name = line[: line.index("{")]
                value = line.rsplit(" ", 1)[1]
            else:
                name, value = line.rsplit(" ", 1)
        except (ValueError, IndexError):
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
    return out


def scrape(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return parse_prometheus(r.read().decode("utf-8", errors="replace"))


def collect(prefill_url, decode_url, tag):
    snap = {"tag": tag, "ts": time.time(), "nodes": {}}
    for role, url in [("prefill", prefill_url), ("decode", decode_url)]:
        try:
            raw = scrape(url)
        except Exception as e:  # noqa: BLE001
            snap["nodes"][role] = {"error": str(e), "url": url}
            continue
        node = {"url": url}
        for m in COUNTERS + GAUGES:
            if m in raw:
                node[m] = raw[m]
        for m in HISTOGRAMS:
            for suf in ("_sum", "_count"):
                if m + suf in raw:
                    node[m + suf] = raw[m + suf]
        snap["nodes"][role] = node
    return snap


def compute_delta(before, after):
    """before/after 스냅샷에서 counter delta와 gauge 최종값을 정리하고
    재계산 vs fetch 요약을 만든다."""
    delta = {"nodes": {}, "summary": {}}
    for role in ("prefill", "decode"):
        b = before["nodes"].get(role, {})
        a = after["nodes"].get(role, {})
        if "error" in b or "error" in a or not a:
            delta["nodes"][role] = {"error": a.get("error") or b.get("error") or "missing"}
            continue
        nd = {}
        # counter: after에만 있으면 before를 0으로 간주 (Prometheus는 값이 0인 카운터를
        # 관측 전까지 노출하지 않을 수 있어, before에 키가 없는 경우가 정상이다).
        for m in COUNTERS:
            if m in a:
                nd[m + "_delta"] = a[m] - b.get(m, 0.0)
        for m in GAUGES:
            if m in a:
                nd[m] = a[m]
        for m in HISTOGRAMS:
            s, c = m + "_sum", m + "_count"
            if s in a and c in a:
                dc = a[c] - b.get(c, 0.0)
                ds = a[s] - b.get(s, 0.0)
                nd[m + "_avg"] = (ds / dc) if dc > 0 else None
        delta["nodes"][role] = nd

    # --- 재계산 vs fetch 요약 ---
    # PD disagg에서 prompt/cached는 prefill·decode 양쪽에 동일하게 찍히고,
    # TTFT는 decode 노드에만, hicache host(L2)는 hierarchical을 켠 노드에만 찍힌다.
    # → 두 노드에서 병합 조회한다.
    def pick(key, default=None):
        for role in ("prefill", "decode"):
            v = delta["nodes"].get(role, {}).get(key)
            if v is not None:
                return v
        return default

    prompt = pick("sglang:prompt_tokens_total_delta")
    if prompt and prompt > 0:
        cached = pick("sglang:cached_tokens_total_delta", 0.0)          # 캐시 없으면 0 = reuse 0
        prefetched = pick("sglang:prefetched_tokens_total_delta", 0.0)  # L3 storage fetch
        uncached = prompt - cached
        delta["summary"] = {
            "prompt_tokens": prompt,
            "cached_tokens (reuse/fetch)": cached,
            "uncached_tokens (recompute)": uncached,
            "reuse_ratio": round(cached / prompt, 4),
            "recompute_ratio": round(uncached / prompt, 4),
            "L3_prefetched_tokens (storage fetch)": prefetched,
            "L3_share_of_reuse": round(prefetched / cached, 4) if cached else 0.0,
            "cache_hit_rate_gauge": pick("sglang:cache_hit_rate"),
            "hicache_host_used_tokens (L2)": pick("sglang:hicache_host_used_tokens"),
            "load_back_tokens (host->GPU)": pick("sglang:load_back_tokens_total_delta", 0.0),
            "ttft_avg_s": pick("sglang:time_to_first_token_seconds_avg"),
        }
    return delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-url", default="http://127.0.0.1:30000/metrics")
    ap.add_argument("--decode-url", default="http://127.0.0.1:30001/metrics")
    ap.add_argument("--tag", default="snapshot")
    ap.add_argument("--out", default=None)
    ap.add_argument("--delta", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="두 스냅샷 파일에서 delta 요약 계산")
    args = ap.parse_args()

    if args.delta:
        before = json.load(open(args.delta[0]))
        after = json.load(open(args.delta[1]))
        result = compute_delta(before, after)
        print(json.dumps(result.get("summary", {}), indent=2, ensure_ascii=False))
    else:
        result = collect(args.prefill_url, args.decode_url, args.tag)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

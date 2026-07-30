#!/usr/bin/env python3
"""
Collapse one arm's four artifacts into ONE row of the evaluation table.

An arm of the placement experiment produces:
  bench json   sglang_BFCL_v3_multi_turn_concurrent.py  -> TTFT distribution, throughput
  mem csv      sys_mem_breakdown.py                     -> host RSS / page cache / per-GPU HBM
  park csv     park_location_sampler.py                 -> KV residency + fetch source split
  delta json   phase0_metrics_scraper.py --delta        -> prefix reuse / recomputed tokens

Comparing arms by opening four files each is how numbers end up mis-transcribed into a
paper table, so this does the reduction once, states the peak-vs-mean choice explicitly
for every metric, and prints the markdown table directly.

Peaks, not means, for memory: the claim is about COMMITMENT -- the capacity another
process could not have used -- and that is set by the high-water mark. Means are printed
alongside so a spiky arm is visible as such.

사용:
  # one arm
  python benchmark/collect_arm_metrics.py --arm radix \
      --bench   results/exp1/bench_radix.json \
      --mem     results/exp1/mem_radix.csv \
      --park    results/exp1/parked_radix.csv \
      --metrics results/exp1/metrics_radix_delta.json

  # all arms of a directory laid out by run_exp1_placement.sh -> the paper table
  python benchmark/collect_arm_metrics.py --dir results/exp1 \
      --arms radix hicache park --out results/exp1/table.json
"""
import argparse
import csv
import json
import os


# ----------------------------------------------------------------- loading helpers
def _rows(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v) if v not in ("", None) else 0.0
                except (TypeError, ValueError):
                    row[k] = 0.0
            out.append(row)
    return out


def _json(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _col(rows, key):
    return [r[key] for r in rows if key in r]


def _peak(rows, key):
    v = _col(rows, key)
    return max(v) if v else None


def _mean(rows, key):
    v = _col(rows, key)
    return sum(v) / len(v) if v else None


def _last(rows, key):
    """Final value of a cumulative counter. Not the peak: these only grow, and a peak
    taken over a series that includes post-run samples would be identical anyway, but
    'last' says what is meant."""
    v = _col(rows, key)
    return v[-1] if v else None


def _gpu_keys(rows, suffix):
    if not rows:
        return []
    return sorted((k for k in rows[0] if k.startswith("gpu") and k.endswith(suffix)
                   and not k.startswith("gpu_")),
                  key=lambda k: int(k[3:-len(suffix)]))


# ----------------------------------------------------------------- one arm
def collect(arm, bench=None, mem=None, park=None, metrics=None):
    bench_all = _json(bench)
    b = bench_all.get("summary", {})
    m = _rows(mem)
    p = _rows(park)
    sm = _json(metrics).get("summary", {})

    hbm_keys = _gpu_keys(m, "_hbm_mb")
    prompt = sm.get("prompt_tokens")
    cached = sm.get("cached_tokens (reuse/fetch)")
    n_turns, n_err, n_empty = _turn_health(bench_all)
    bad = (n_turns > 0 and (n_err + n_empty) / n_turns > 0.05)

    row = {
        "arm": arm,
        # ---------------- validity (read this before anything else)
        "n_turns": n_turns or None,
        "n_errors": n_err if n_turns else None,
        "n_empty": n_empty if n_turns else None,
        "fail_rate": round((n_err + n_empty) / n_turns, 3) if n_turns else None,
        "valid": (not bad) if n_turns else None,
        # ---------------- memory (peaks: commitment is set by the high-water mark)
        "peak_host_rss_gb": _div(_peak(m, "proc_rss_mb"), 1024),
        "mean_host_rss_gb": _div(_mean(m, "proc_rss_mb"), 1024),
        "peak_page_cache_gb": _div(_peak(m, "file_cache_mb"), 1024),
        "peak_host_footprint_gb": _div(_peak(m, "rss_plus_cache_mb"), 1024),
        # AnonPages/MemAvailable are system-wide: the number an adjacent agent stack feels
        "peak_anonpages_gb": _div(_peak(m, "anonpages_mb"), 1024),
        "min_mem_available_gb": _div(_min(m, "mem_avail_mb"), 1024),
        "peak_gpu_hbm_total_gb": _div(_peak(m, "gpu_hbm_mb"), 1024),
        "peak_gpu_hbm_each_gb": {k[:-7]: _div(_peak(m, k), 1024) for k in hbm_keys},
        # ---------------- KV residency (park telemetry; absent for radix/hicache arms)
        "kv_peer_gpu_peak_gb": _peak(p, "park_peer_gb"),
        "kv_local_park_peak_gb": _peak(p, "park_local_gb"),
        "kv_cpu_overflow_peak_gb": _peak(p, "host_gb"),
        "kv_serving_peak_gb": _peak(p, "serving_gb"),
        "kv_dropped_total_gb": _last(p, "dropped_gb"),
        "kv_host_evicted_total_gb": _last(p, "host_evicted_gb"),
        "peer_share_of_parked": _mean(p, "peer_frac_of_parked"),
        "host_share_of_parked": _mean(p, "host_frac"),
        # ---------------- fetch source (read-back counterpart of residency)
        "fetch_hits": _last(p, "fetch_hits"),
        "fetch_local_hits": _last(p, "fetch_local_hits"),
        "fetch_peer_hits": _last(p, "fetch_peer_hits"),
        "fetch_host_hits": _last(p, "fetch_host_hits"),
        "fetch_nospace": _last(p, "fetch_nospace"),
        # ---------------- performance
        "ttft_p50_s": b.get("ttft_p50_s"),
        "ttft_p95_s": b.get("ttft_p95_s"),
        "ttft_p99_s": b.get("ttft_p99_s"),
        "ttft_avg_s": b.get("avg_ttft_s"),
        "throughput_tok_s": b.get("overall_throughput_tok_s") or b.get("throughput_tok_s"),
        "goodput_tok_s": _goodput(_json(bench)),
        "errors": b.get("n_errors") if b.get("n_errors") is not None else b.get("errors"),
        # ---------------- reuse vs recompute (server-side counters, not the client's view)
        "prefix_reuse_ratio": sm.get("reuse_ratio"),
        "recomputed_tokens": sm.get("uncached_tokens (recompute)"),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
    }
    return row


def _div(v, d):
    return round(v / d, 2) if v is not None else None


def _min(rows, key):
    v = _col(rows, key)
    return min(v) if v else None


def _turn_health(bench):
    """(n_turns, n_errors, n_empty). Every other number in the row is meaningless if this
    is bad, so it is computed for every arm and rendered first.

    This exists because a full Exp1+Exp2 sweep once produced complete, plausible-looking
    tables -- TTFT percentiles, reuse ratios, residency splits -- from runs in which
    199 of 220 turns had failed with HTTP 503. The percentiles were over the ~20 turns
    that happened to land after the router finally came up."""
    turns = errors = empty = 0
    for item in (bench.get("results") or []):
        for t in (item.get("turns") or []):
            turns += 1
            if t.get("error"):
                errors += 1
            elif not t.get("output_tokens"):
                empty += 1
    return turns, errors, empty


def _goodput(bench):
    """Output tokens per wall second counting ONLY turns that completed successfully.

    Distinct from overall throughput, which the benchmark computes over all output
    tokens: an arm that degrades into empty 200s would post a flattering throughput
    while producing nothing usable."""
    s = bench.get("summary") or {}
    results = bench.get("results") or []
    wall = s.get("total_wall_time_s") or s.get("wall_time_s")
    if not wall:
        return None
    good = 0
    for item in results:
        for t in (item.get("turns") or []):
            if t.get("error") or not t.get("output_tokens"):
                continue
            good += t["output_tokens"]
    return round(good / wall, 2)


# ----------------------------------------------------------------- table rendering
GROUPS = [
    ("Validity", [
        ("n_turns", "turns"),
        ("n_errors", "errors"),
        ("n_empty", "empty (200, no tokens)"),
        ("fail_rate", "fail rate"),
    ]),
    ("Memory (peak)", [
        ("peak_host_rss_gb", "host RSS (GB)"),
        ("peak_page_cache_gb", "page cache (GB)"),
        ("peak_host_footprint_gb", "host footprint RSS+cache (GB)"),
        ("peak_anonpages_gb", "AnonPages (GB)"),
        ("min_mem_available_gb", "MemAvailable min (GB)"),
        ("peak_gpu_hbm_total_gb", "GPU HBM total (GB)"),
    ]),
    ("KV residency", [
        ("kv_serving_peak_gb", "local GPU serving (GB)"),
        ("kv_local_park_peak_gb", "local GPU park (GB)"),
        ("kv_peer_gpu_peak_gb", "peer GPU (GB)"),
        ("kv_cpu_overflow_peak_gb", "CPU DRAM overflow (GB)"),
        ("kv_dropped_total_gb", "dropped, cumulative (GB)"),
        ("peer_share_of_parked", "peer share of parked"),
        ("host_share_of_parked", "host share of parked"),
    ]),
    ("Fetch source", [
        ("fetch_hits", "fetch hits"),
        ("fetch_local_hits", "  from local park"),
        ("fetch_peer_hits", "  from peer GPU"),
        ("fetch_host_hits", "  from CPU DRAM"),
        ("fetch_nospace", "gave up (no space)"),
    ]),
    ("Performance", [
        ("ttft_p50_s", "TTFT p50 (s)"),
        ("ttft_p95_s", "TTFT p95 (s)"),
        ("ttft_p99_s", "TTFT p99 (s)"),
        ("throughput_tok_s", "throughput (tok/s)"),
        ("goodput_tok_s", "goodput (tok/s)"),
        ("prefix_reuse_ratio", "prefix reuse ratio"),
        ("recomputed_tokens", "recomputed tokens"),
        ("errors", "errors"),
    ]),
]


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def render(rows):
    arms = [r["arm"] for r in rows]
    w = max(34, *(len(a) for a in arms))
    out = ["| metric | " + " | ".join(arms) + " |",
           "|" + "---|" * (len(arms) + 1)]
    for group, fields in GROUPS:
        out.append(f"| **{group}** |" + " |" * len(arms))
        for key, label in fields:
            if all(r.get(key) is None for r in rows):
                continue
            out.append(f"| {label} | " + " | ".join(_fmt(r.get(key)) for r in rows) + " |")
    # per-GPU HBM gets its own block: the column set is discovered, not fixed
    gpus = sorted({g for r in rows for g in (r.get("peak_gpu_hbm_each_gb") or {})})
    for g in gpus:
        vals = [(r.get("peak_gpu_hbm_each_gb") or {}).get(g) for r in rows]
        if all(v is None for v in vals):
            continue
        out.append(f"| peak {g} HBM (GB) | " + " | ".join(_fmt(v) for v in vals) + " |")
    del w
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", help="single-arm mode: label for this arm")
    ap.add_argument("--bench")
    ap.add_argument("--mem")
    ap.add_argument("--park")
    ap.add_argument("--metrics")
    ap.add_argument("--dir", help="multi-arm mode: directory laid out by run_exp1_placement.sh")
    ap.add_argument("--arms", nargs="*", default=["radix", "hicache", "park"])
    ap.add_argument("--out", help="write the collected rows as JSON")
    args = ap.parse_args()

    rows = []
    if args.dir:
        for arm in args.arms:
            d = args.dir
            rows.append(collect(
                arm,
                bench=os.path.join(d, f"bench_{arm}.json"),
                mem=os.path.join(d, f"mem_{arm}.csv"),
                park=os.path.join(d, f"parked_{arm}.csv"),
                metrics=os.path.join(d, f"metrics_{arm}_delta.json")))
    elif args.arm:
        rows.append(collect(args.arm, args.bench, args.mem, args.park, args.metrics))
    else:
        ap.error("give either --dir or --arm")

    missing = [r["arm"] for r in rows if r["ttft_p50_s"] is None]
    if missing:
        print(f"[warn] no bench summary for: {', '.join(missing)} "
              f"-- performance columns will be empty for those arms\n")

    # Refuse to present a broken run as a result. The table is still printed -- it is
    # useful for debugging -- but it must not be readable as an outcome.
    invalid = [r for r in rows if r.get("valid") is False]
    if invalid:
        bar = "!" * 74
        print(bar)
        print("INVALID RUN -- DO NOT READ THESE NUMBERS AS A RESULT")
        for r in invalid:
            print(f"  {r['arm']:<12} {r['n_errors']}/{r['n_turns']} turns failed, "
                  f"{r['n_empty']} empty  (fail rate {r['fail_rate']:.0%})")
        print("  Every metric below is computed over the turns that DID succeed, which")
        print("  are not a random sample -- they are whichever turns ran while the")
        print("  system happened to be healthy. Fix the cause and re-run.")
        print("  Most common cause: the benchmark started before the router could route.")
        print("  Check bench_<arm>.log for 'No available servers' (HTTP 503).")
        print(bar)
        print()
    print(render(rows))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\n[collect] -> {args.out}")


if __name__ == "__main__":
    main()

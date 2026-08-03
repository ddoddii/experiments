#!/usr/bin/env python3
"""
Reduce a KV-occupancy time series to the memory-imbalance metrics (Exp 2).

The claim under test is about MEMORY, not latency: does KV cache end up where there is
free space, and does per-GPU saturation even out? So everything here is computed from
kv_occupancy_timeseries.py's CSV -- per-GPU pool saturation sampled over the run.

The metric that matters most is M1, STRANDED HEADROOM: at each instant, if some pool was
saturating (and therefore evicting), how much free KV capacity existed on other GPUs at
that same moment? Integrated over the run it is the memory that the system owned, paid
for, and could not use. If that number is small, the premise of the experiment is false
and no placement policy can help -- which is why the baseline arm is run first, alone.

Two things this script is deliberate about:

  * Saturation is a FRACTION, but headroom is reported in absolute tokens and GB. The
    prefill pools are capped with --max-total-tokens while decode's are not, so "P is at
    95% and D is at 20%" says nothing about whether D's spare room is worth borrowing.
    Only the absolute number answers that.

  * M1 depends on two thresholds, so it is reported as a CURVE over the saturation
    threshold rather than as one number. A single (0.90, 0.50) pair invites the reading
    that the thresholds were chosen after seeing the data.

사용:
  python benchmark/collect_imbalance.py --csv results/exp2/pd/occ_hicache.csv
  python benchmark/collect_imbalance.py --dir results/exp2/pd --arms hicache park_pd \
      --out results/exp2/pd/imbalance.json
"""
import argparse
import csv
import json
import math
import os
import statistics as st

# Llama-3.1-8B: 32 layers x 8 KV heads x 128 dim x 2 (K,V) x 2 bytes = 128 KiB/token.
# Llama-2-13B (MHA) is 800 and Qwen3-14B is 160, so this must be passed for those.
DEFAULT_KIB_PER_TOKEN = 128

# Thresholds for M1. "Saturating" is deliberately below 1.0: SGLang starts evicting well
# before the pool is literally full, and a pool pinned at exactly 1.0 is rare.
HI_DEFAULT = 0.90
LO_DEFAULT = 0.50
HI_CURVE = (0.70, 0.80, 0.85, 0.90, 0.95)


def load(csv_path):
    """Return (labels, roles, rows) where each row is {label: {use, used, run, cap}}."""
    roles, rows, labels, metric_src = {}, [], [], {}
    with open(csv_path) as fh:
        header = None
        for raw in fh:
            if raw.startswith("# metric:"):
                for tok in raw.split(":", 1)[1].split():
                    lab, _, m = tok.partition(":")
                    metric_src[lab] = m
                continue
            if raw.startswith("# roles:"):
                for tok in raw.split(":", 1)[1].split():
                    lab, _, role = tok.partition(":")
                    roles[lab] = role
                continue
            if header is None:
                header = next(csv.reader([raw]))
                # columns are t_s then 4 per label; recover the label order from the
                # header rather than assuming the default 2P2D layout.
                labels = [c[:-4] for c in header if c.endswith("_use")]
                continue
            vals = next(csv.reader([raw]))
            if len(vals) != len(header):
                continue  # truncated final line from a killed sampler
            d = dict(zip(header, vals))
            rec = {}
            for lab in labels:
                def g(suffix):
                    v = d.get(f"{lab}{suffix}", "")
                    return float(v) if v not in ("", None) else None
                rec[lab] = {"use": g("_use"), "used": g("_used_tok"),
                            "run": g("_running"), "cap": g("_cap_tok"),
                            "pressure": g("_pressure"), "hit": g("_hit")}
            rows.append((float(d["t_s"]), rec))
    return labels, roles, rows, metric_src


def fill_capacity(labels, rows):
    """Capacity is a server constant, but the gauge can be missing on early/idle ticks.
    Carry the run's median forward so free-space arithmetic does not silently drop
    samples -- dropping exactly the idle ticks would bias headroom DOWNWARD, i.e. against
    the result this script is trying to test, but it would also make the number wrong.

    CSVs written before the sampler gained a capacity column have no _cap_tok at all. The
    fallback matters more than it looks: without it every headroom figure would be None,
    stranded would come out as a clean 0.0, and 'the premise is false, abandon Exp 2' is
    exactly the conclusion a missing column would fake. So derive it from used/use on the
    samples where the pool is occupied enough for the ratio to be meaningful."""
    caps, derived = {}, []
    for lab in labels:
        seen = [r[lab]["cap"] for _, r in rows if r[lab]["cap"]]
        if not seen:
            seen = [r[lab]["used"] / r[lab]["use"] for _, r in rows
                    if r[lab]["use"] and r[lab]["use"] > 0.02 and r[lab]["used"]]
            if seen:
                derived.append(lab)
        caps[lab] = st.median(seen) if seen else None
    if derived:
        print(f"[warn] no capacity gauge for {derived}; derived from used/usage. "
              f"Re-run with the current kv_occupancy_timeseries.py to record it directly.")
    missing = [lab for lab, c in caps.items() if not c]
    if missing:
        print(f"[warn] capacity UNKNOWN for {missing} -- these GPUs contribute 0 to "
              f"stranded headroom, so the M1 figure below is a LOWER BOUND, not a result.")
    return caps


def free_tokens(rec, lab, cap):
    u = rec[lab]["use"]
    if u is None or cap is None:
        return None
    return max(0.0, (1.0 - u) * cap)


def stranded(labels, rows, caps, hi, lo, kib):
    """M1: GB-seconds of free KV capacity that sat on a slack GPU while another was
    saturating. Integrated with the real inter-sample gaps, not a nominal interval, so a
    sampler that stalled does not silently inflate the integral."""
    total_gbs, busy_s, span_s = 0.0, 0.0, 0.0
    per_lab = {lab: 0.0 for lab in labels}
    for i, (t, rec) in enumerate(rows):
        dt = (rows[i + 1][0] - t) if i + 1 < len(rows) else 0.0
        if dt <= 0 or dt > 10:      # skip the last sample and any sampler stall
            continue
        span_s += dt
        uses = {lab: rec[lab]["use"] for lab in labels if rec[lab]["use"] is not None}
        if not uses or max(uses.values()) <= hi:
            continue
        busy_s += dt
        for lab, u in uses.items():
            if u >= lo:
                continue
            f = free_tokens(rec, lab, caps[lab])
            if f is None:
                continue
            gb = f * kib / (1024 * 1024)
            total_gbs += gb * dt
            per_lab[lab] += gb * dt
    return {
        "stranded_gb_s": round(total_gbs, 1),
        "saturated_s": round(busy_s, 1),
        "measured_s": round(span_s, 1),
        "saturated_frac": round(busy_s / span_s, 4) if span_s else None,
        # The average amount of memory stranded WHILE saturated -- easier to state in a
        # paper than GB-seconds, and it is the size of the resource being wasted.
        "mean_stranded_gb": round(total_gbs / busy_s, 2) if busy_s else 0.0,
        "by_gpu_gb_s": {k: round(v, 1) for k, v in per_lab.items()},
    }


def imbalance(labels, rows):
    """M2: how uneven saturation is, per sample, then summarised over time.

    CoV for the headline (a bar chart of it is legible), spread for the worst moment, and
    Jain's fairness index because resource-allocation reviewers already know its scale."""
    cov, spread, jain = [], [], []
    for _, rec in rows:
        u = [rec[lab]["use"] for lab in labels if rec[lab]["use"] is not None]
        if len(u) < 2:
            continue
        m = sum(u) / len(u)
        spread.append(max(u) - min(u))
        if m > 1e-6:
            cov.append(st.pstdev(u) / m)
        sq = sum(x * x for x in u)
        if sq > 1e-12:
            jain.append((sum(u) ** 2) / (len(u) * sq))

    def s(xs, f=st.mean):
        return round(f(xs), 4) if xs else None

    return {
        "cov_mean": s(cov),
        "cov_p95": round(pct(cov, 95), 4) if cov else None,
        "spread_mean": s(spread),
        "spread_p95": round(pct(spread, 95), 4) if spread else None,
        "jain_mean": s(jain),
        "samples": len(spread),
    }


def pct(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def per_gpu(labels, roles, rows, caps, kib):
    """Per-GPU saturation and free space, plus the P-vs-D roll-up -- the number that
    states the prefill/decode imbalance directly."""
    out, by_role = {}, {}
    for lab in labels:
        us = [r[lab]["use"] for _, r in rows if r[lab]["use"] is not None]
        ps = [r[lab]["pressure"] for _, r in rows if r[lab]["pressure"] is not None]
        hs = [r[lab]["hit"] for _, r in rows if r[lab]["hit"] is not None]
        fs = [free_tokens(r, lab, caps[lab]) for _, r in rows if r[lab]["use"] is not None]
        fs = [f for f in fs if f is not None]
        if not us:
            continue
        out[lab] = {
            "role": roles.get(lab, "?"),
            "cap_tokens": int(caps[lab]) if caps[lab] else None,
            "cap_gb": round(caps[lab] * kib / (1024 * 1024), 2) if caps[lab] else None,
            "use_mean": round(st.mean(us), 4),
            # Active working set only. occupancy - pressure IS the prefix cache, which is
            # what a prefill node is holding and a decode node is not.
            "pressure_mean": round(st.mean(ps), 4) if ps else None,
            "cached_mean": round(st.mean(us) - st.mean(ps), 4) if ps else None,
            # A cache that is full AND missing is thrashing; a full cache at 100% hit is
            # simply well-sized. Saturation alone cannot tell those apart, so the
            # imbalance claim has to be read together with this number.
            "hit_rate_final": round(hs[-1], 4) if hs else None,
            "use_p95": round(pct(us, 95), 4),
            "free_gb_mean": round(st.mean(fs) * kib / (1024 * 1024), 2) if fs else None,
            "frac_time_over_90": round(sum(1 for u in us if u > 0.90) / len(us), 4),
        }
        by_role.setdefault(roles.get(lab, "?"), []).append(st.mean(us))
    roll = {f"{r}_use_mean": round(st.mean(v), 4) for r, v in by_role.items()}
    if "P" in by_role and "D" in by_role:
        roll["pd_gap"] = round(st.mean(by_role["P"]) - st.mean(by_role["D"]), 4)
    return out, roll


def check_metric(labels, rows, metric_src):
    """Report which occupancy metric the CSV actually holds.

    sglang:token_usage subtracts the prefix cache, so a prefill pool full of retained
    prefixes reads as ~2% occupied; a probe built on it returned a flat 0.0 GB*s and
    looked like a legitimate negative result. The sampler now records the metric it found
    per server in a `# metric:` header.

    Do NOT infer this from the values. Occupancy == pressure has two opposite causes --
    the sampler fell back to token_usage (broken measurement), or the prefix cache really
    holds nothing on that GPU (a finding, and for a HiCache prefill that offloads to host
    DRAM, the expected one). Only the recorded source separates them."""
    if not metric_src:
        print("[WARN] CSV predates the `# metric:` header, so which gauge it holds is "
              "unknown. If occupancy == pressure below, that is ambiguous between a "
              "sampler fallback and an genuinely empty GPU-resident cache -- re-run "
              "rather than interpret it.")
        return "unknown"
    fallback = sorted(k for k, v in metric_src.items() if v != "cache_occupancy")
    if fallback:
        print(f"[WARN] {fallback} reported via {metric_src}. Those servers do not export "
              f"sglang:cache_occupancy, so their occupancy EXCLUDES prefix-cached KV. "
              f"Pull the SGLang source tree and restart before trusting them.")
        return "token_usage (FALLBACK)"
    return "cache_occupancy"


# A6000. Used to turn per-GPU HBM into a free-space figure; override with --gpu-total-gb.
DEFAULT_GPU_TOTAL_GB = 49.0


def gpu_hbm(mem_csv, gpu_total_gb):
    """Per-GPU HBM from sys_mem_breakdown.py's CSV.

    The KV-pool view and the hardware view disagree, and the disagreement is the point.
    A prefill capped at --max-total-tokens 60000 reserves 7.3 GB of pool on a 49 GB card,
    so its pool can read 'full' while 24 GB of the GPU is untouched; a decode reserves
    almost the whole card via --mem-fraction-static and then leaves most of that pool
    empty. Only the hardware numbers say how much HBM is actually available to borrow."""
    if not os.path.exists(mem_csv):
        return None
    cols, rows = None, []
    with open(mem_csv) as fh:
        for raw in csv.reader(fh):
            if cols is None:
                cols = raw
                continue
            if len(raw) == len(cols):
                rows.append(dict(zip(cols, raw)))
    if not rows:
        return None
    out = {}
    for i in range(4):
        key = f"gpu{i}_hbm_mb"
        vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        if not vals:
            continue
        out[f"gpu{i}"] = {
            "used_gb_mean": round(st.mean(vals) / 1024, 2),
            "used_gb_peak": round(max(vals) / 1024, 2),
            # Free at PEAK usage: the conservative figure, i.e. what could have been
            # borrowed for the whole run without ever colliding with the server.
            "free_gb_at_peak": round(gpu_total_gb - max(vals) / 1024, 2),
        }
    if out:
        out["total_free_gb_at_peak"] = round(
            sum(v["free_gb_at_peak"] for k, v in out.items() if k.startswith("gpu")), 2)
    return out


def analyse(csv_path, kib, hi, lo, mem_csv=None, gpu_total_gb=DEFAULT_GPU_TOTAL_GB):
    labels, roles, rows, metric_src = load(csv_path)
    if not rows:
        return {"error": f"no samples in {csv_path}"}
    metric = check_metric(labels, rows, metric_src)
    caps = fill_capacity(labels, rows)
    gpus, roll = per_gpu(labels, roles, rows, caps, kib)
    return {
        "csv": csv_path,
        "metric": metric,
        "samples": len(rows),
        "duration_s": round(rows[-1][0] - rows[0][0], 1),
        "per_gpu": gpus,
        "role_rollup": roll,
        "M1_stranded": stranded(labels, rows, caps, hi, lo, kib),
        "M1_threshold_curve": {
            f"hi={h}": stranded(labels, rows, caps, h, lo, kib)["stranded_gb_s"]
            for h in HI_CURVE
        },
        "M2_imbalance": imbalance(labels, rows),
        "gpu_hbm": gpu_hbm(mem_csv, gpu_total_gb) if mem_csv else None,
    }


def report(tag, a):
    if "error" in a:
        print(f"\n### {tag}: {a['error']}")
        return
    print(f"\n### {tag}   ({a['samples']} samples, {a['duration_s']}s)")
    print(f"  metric: {a['metric']}")
    print(f"{'gpu':>5} {'role':>5} {'cap(GB)':>8} {'occup':>7} {'active':>7} {'cached':>7} "
          f"{'p95':>7} {'free(GB)':>9} {'>90%':>7} {'hit':>6}")
    for lab, g in a["per_gpu"].items():
        pm = "  --  " if g["pressure_mean"] is None else f"{g['pressure_mean']:>6.3f}"
        cm = "  --  " if g["cached_mean"] is None else f"{g['cached_mean']:>6.3f}"
        print(f"{lab:>5} {g['role']:>5} {str(g['cap_gb']):>8} {g['use_mean']:>7.3f} "
              f"{pm:>7} {cm:>7} {g['use_p95']:>7.3f} {str(g['free_gb_mean']):>9} "
              f"{g['frac_time_over_90']*100:>6.1f}% "
              f"{'   -- ' if g['hit_rate_final'] is None else format(g['hit_rate_final'],'>6.3f')}")
    r = a["role_rollup"]
    if "pd_gap" in r:
        print(f"  P mean {r['P_use_mean']:.3f} vs D mean {r['D_use_mean']:.3f}"
              f"   -> P/D gap {r['pd_gap']:+.3f}")
    m1 = a["M1_stranded"]
    print(f"  M1 stranded headroom : {m1['stranded_gb_s']} GB*s   "
          f"({m1['mean_stranded_gb']} GB average while saturated)")
    print(f"     saturated for      : {m1['saturated_s']}s of {m1['measured_s']}s "
          f"({(m1['saturated_frac'] or 0)*100:.1f}%)")
    print(f"     by gpu (GB*s)      : {m1['by_gpu_gb_s']}")
    print(f"     vs threshold       : {a['M1_threshold_curve']}")
    m2 = a["M2_imbalance"]
    print(f"  M2 imbalance         : CoV {m2['cov_mean']}  spread mean {m2['spread_mean']}"
          f"  p95 {m2['spread_p95']}  Jain {m2['jain_mean']}")
    g = a.get("gpu_hbm")
    if g:
        parts = " ".join(f"{k}:{v['used_gb_peak']}/{v['free_gb_at_peak']}free"
                         for k, v in g.items() if k.startswith("gpu"))
        print(f"  GPU HBM peak/free    : {parts}")
        print(f"     idle HBM cluster-wide (at each GPU's peak): "
              f"{g['total_free_gb_at_peak']} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="single occupancy CSV")
    ap.add_argument("--dir", help="directory holding occ_<arm>.csv")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--kib-per-token", type=float, default=DEFAULT_KIB_PER_TOKEN)
    ap.add_argument("--hi", type=float, default=HI_DEFAULT)
    ap.add_argument("--lo", type=float, default=LO_DEFAULT)
    ap.add_argument("--gpu-total-gb", type=float, default=DEFAULT_GPU_TOTAL_GB)
    ap.add_argument("--out")
    args = ap.parse_args()

    jobs = []
    if args.csv:
        jobs.append((os.path.basename(args.csv), args.csv))
    if args.dir:
        arms = args.arms or sorted(
            f[4:-4] for f in os.listdir(args.dir)
            if f.startswith("occ_") and f.endswith(".csv")
        )
        for arm in arms:
            jobs.append((arm, os.path.join(args.dir, f"occ_{arm}.csv")))
    if not jobs:
        ap.error("give --csv or --dir")

    out = {}
    for tag, path in jobs:
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        mem = os.path.join(os.path.dirname(path), f"mem_{tag}.csv")
        out[tag] = analyse(path, args.kib_per_token, args.hi, args.lo,
                           mem_csv=mem, gpu_total_gb=args.gpu_total_gb)
        report(tag, out[tag])

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[collect] -> {args.out}")


if __name__ == "__main__":
    main()

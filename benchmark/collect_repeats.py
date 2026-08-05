#!/usr/bin/env python3
"""
Aggregate repeated Exp 2 runs into PAIRED differences, one per repeat.

The point is not the average across repeats -- it is whether the sign holds on every
independent workload. A mean of +3, +4, -6 and a mean of +0.3, +0.4, +0.3 both look like
"about +0.3 on average" if you only print the mean, and only one of them is a result.
So every repeat's delta is printed, and the verdict is stated in terms of how many
repeats agreed rather than as a confidence interval, which three samples cannot support.

Each repeat compares the two arms on the SAME ShareGPT slice, so the difference within a
repeat is paired; the slice changes between repeats, so a sign flip means the effect is
workload-specific.

사용:
  python benchmark/collect_repeats.py --dirs results/exp2/repeat_nocap_c32_r*
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

BASE, OURS = "park_local", "park_pd"
# (key, label, lower_is_better)
METRICS = [
    ("ttft_p50_s", "TTFT p50 (s)", True),
    ("ttft_p95_s", "TTFT p95 (s)", True),
    ("ttft_p99_s", "TTFT p99 (s)", True),
    ("overall_throughput_tok_per_s", "throughput (t/s)", False),
]


def run_data(d):
    out = {}
    for arm in (BASE, OURS):
        bench = os.path.join(d, f"bench_{arm}.json")
        parked = os.path.join(d, f"parked_{arm}.csv")
        if not (os.path.exists(bench) and os.path.exists(parked)):
            return None
        s = json.load(open(bench))["summary"]
        rows = list(csv.DictReader(open(parked)))
        f = lambda k: float(rows[-1].get(k, "") or 0)
        h, m = f("fetch_hits"), f("fetch_miss")
        rec = {k: s.get(k) for k, _, _ in METRICS}
        # Share of radix-MISSING prefill requests that a parked prefix rescued. Not Exp 1's
        # token-weighted cached/prompt ratio, and not a hit rate over all requests --
        # fetch_already is excluded from the denominator by design.
        rec["park_hit_rate_pct"] = 100 * h / (h + m) if h + m else 0.0
        rec["fetch_already"] = f("fetch_already")
        rec["fetch_miss"] = m
        rec["parked_gb"] = max(
            sum(float(r.get(f"gpu{g}_gb", "") or 0) for g in range(4)) for r in rows)
        rec["items"] = s.get("success_items")
        rec["errors"] = s.get("error_items")
        out[arm] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    runs = []
    for d in sorted(sum((glob.glob(x) for x in args.dirs), [])):
        r = run_data(d)
        if r is None:
            print(f"[skip] {d}: incomplete (both arms need bench_*.json and parked_*.csv)")
            continue
        runs.append((os.path.basename(d), r))
    if not runs:
        raise SystemExit("no complete repeats found")

    bad = [(n, a, r[a]["errors"]) for n, r in runs for a in (BASE, OURS) if r[a]["errors"]]
    if bad:
        print(f"[warn] repeats with request errors: {bad} -- a run that dropped requests "
              f"has a tail statistic computed over the survivors")

    print(f"\n{len(runs)} repeats: " + ", ".join(n for n, _ in runs))
    # parked_gb is CONFIGURED, not measured: PARK_POOL_TOKENS_PER_GPU x candidate count,
    # and both arms fill their pools to capacity. Reporting it as "Ours better in ALL,
    # median |Δ| = 199.9%" dresses a constant up as an experimental win. Kept because
    # "the pools did fill" is worth confirming, flagged so it is never read as evidence.
    rows = [("park-fetch hit rate (%)", "park_hit_rate_pct", False, False),
            ("park-fetch misses", "fetch_miss", True, False),
            ("parked KV (GB)", "parked_gb", False, True)] + \
           [(lab, k, lo, False) for k, lab, lo in METRICS]

    summary = {}
    for lab, key, lower, configured in rows:
        deltas, line = [], []
        for _, r in runs:
            b, o = r[BASE][key], r[OURS][key]
            if b is None or o is None:
                line.append("   --  ")
                continue
            d = o - b
            deltas.append(d if not lower else -d)   # positive = Ours better, always
            line.append(f"{b:7.2f}→{o:<7.2f}")
        if not deltas:
            continue
        # A delta of exactly zero is a TIE, not a loss. Throughput came out identical to
        # the cent in a synthetic check and was reported as "Ours worse in ALL", which is
        # the opposite of what an unchanged metric means. Ties are held to 0.5% of the
        # baseline so that rounding noise does not get read as a direction either.
        bases = [r[BASE][key] for _, r in runs if r[BASE][key] is not None]
        tol = 0.005 * (st.mean([abs(b) for b in bases]) if bases else 0)
        wins = sum(1 for d in deltas if d > tol)
        losses = sum(1 for d in deltas if d < -tol)
        n = len(deltas)
        verdict = ("Ours better in ALL" if wins == n else
                   "Ours worse in ALL" if losses == n else
                   "unchanged (all within 0.5%)" if wins == 0 and losses == 0 else
                   f"SPLIT {wins} better / {losses} worse / {n - wins - losses} tied")
        rel = [abs(d) / abs(r[BASE][key]) * 100 for d, (_, r) in zip(deltas, runs)
               if r[BASE][key]]
        # How much the BASELINE alone moves between repeats. An effect smaller than the
        # baseline's own run-to-run spread is not measurable at this n, however consistent
        # its sign happens to look -- a single run of this comparison reported p95 -70%
        # while the baseline's p95 by itself ranged over 4.5x across runs.
        spread = (100 * (max(bases) - min(bases)) / abs(st.median(bases))
                  if len(bases) > 1 and st.median(bases) else None)
        med = st.median(rel) if rel else None
        drowned = (med is not None and spread is not None and med < spread
                   and not configured)
        summary[key] = {"deltas": [round(d, 4) for d in deltas], "wins": wins,
                        "losses": losses, "n": n, "verdict": verdict,
                        "rel_pct_median": round(med, 1) if med is not None else None,
                        "baseline_spread_pct": round(spread, 1) if spread else None,
                        "below_baseline_spread": drowned, "configured": configured}
        flag = "  <-- SPLIT" if "SPLIT" in verdict else ""
        print(f"\n  {lab}" + ("   [configured, not measured]" if configured else ""))
        print(f"    " + "   ".join(line))
        print(f"    {verdict}" + (f", median |Δ| = {med:.1f}%" if med is not None else "")
              + flag)
        if spread is not None and not configured:
            print(f"    baseline alone varies {spread:.1f}% across repeats"
                  + ("   <-- EFFECT IS SMALLER THAN THIS" if drowned else ""))

    print("\n" + "=" * 64)
    split = [k for k, v in summary.items() if "SPLIT" in v["verdict"]]
    weak = [k for k, v in summary.items() if v["below_baseline_spread"]]
    if split:
        print(f"  NOT reproducible on: {split}")
        print(f"  Do not quote a single-run figure for these; report the range, or run "
              f"more repeats.")
    else:
        print("  Every metric kept its sign across all repeats.")
    if weak:
        print(f"  Effect smaller than the baseline's own run-to-run spread: {weak}")
        print(f"  A consistent sign here is not yet evidence -- n is too small to "
              f"separate the effect from the noise the baseline shows on its own.")

    if args.out:
        json.dump({"runs": {n: r for n, r in runs}, "summary": summary},
                  open(args.out, "w"), indent=2)
        print(f"\n[collect] -> {args.out}")


if __name__ == "__main__":
    main()

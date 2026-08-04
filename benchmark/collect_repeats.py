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
        rec["park_hit_rate_pct"] = 100 * h / (h + m) if h + m else 0.0
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
    rows = [("park hit rate (%)", "park_hit_rate_pct", False),
            ("park-fetch misses", "fetch_miss", True),
            ("parked KV (GB)", "parked_gb", False)] + \
           [(lab, k, lo) for k, lab, lo in METRICS]

    summary = {}
    for lab, key, lower in rows:
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
        summary[key] = {"deltas": [round(d, 4) for d in deltas], "wins": wins,
                        "losses": losses, "n": n, "verdict": verdict,
                        "rel_pct_median": round(st.median(rel), 1) if rel else None}
        flag = "  <-- SPLIT" if "SPLIT" in verdict else ""
        print(f"\n  {lab}")
        print(f"    " + "   ".join(line))
        print(f"    {verdict}"
              + (f", median |Δ| = {st.median(rel):.1f}%" if rel else "") + flag)

    print("\n" + "=" * 64)
    split = [k for k, v in summary.items() if "SPLIT" in v["verdict"]]
    if split:
        print(f"  NOT reproducible on: {split}")
        print(f"  Do not quote a single-run figure for these; report the range, or run "
              f"more repeats.")
    else:
        print("  Every metric kept its sign across all repeats.")

    if args.out:
        json.dump({"runs": {n: r for n, r in runs}, "summary": summary},
                  open(args.out, "w"), indent=2)
        print(f"\n[collect] -> {args.out}")


if __name__ == "__main__":
    main()

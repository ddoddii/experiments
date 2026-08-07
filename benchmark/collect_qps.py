#!/usr/bin/env python3
"""
Reduce an open-loop sweep directory to one curve per arm: rate -> throughput, TTFT.

Reads every bench_<arm>_r<rate>.json the sweep wrote and pulls the OPEN-LOOP fields
(window_throughput_tok_s, window_ttft_*, sessions_unfinished_at_drain). Points whose
summary lacks `mode: open_loop` are skipped rather than silently mixed in -- a
closed-loop file's throughput means something else entirely.

Prints a markdown table and writes json for plot_qps.py.

WHAT MAKES A POINT UNUSABLE, AND WHY IT IS FLAGGED RATHER THAN DROPPED
  past_saturation  the drain timed out on >10% of sessions: the server never finished
                   the offered load, so this point's TTFT is queueing delay and its
                   throughput is a lower bound. Still worth plotting -- the knee is the
                   result -- but it must not be read as a latency measurement.
  wrapped          the corpus cycled, so some sessions replayed a conversation already
                   served. Free cache hits for the cached arms; the point is biased.

사용:
  python benchmark/collect_qps.py --dir results/qps/qps_llama8b --out .../qps.json
"""
import argparse
import glob
import json
import os
import re


SLOS = [0.5, 1.0, 2.0]


def _slo_capacity(rows, slo, key="ttft_p50_s"):
    """Delivered tok/s at the point where TTFT would cross `slo`.

    Linear interpolation between the last point under budget and the first point over
    it. Returns None if even the lightest load already exceeds the budget (the arm never
    meets it) and the top point's throughput if the budget is never crossed (the sweep
    did not push it far enough -- a lower bound, not a measured capacity).
    """
    pts = [r for r in rows if r.get(key) is not None and r.get("throughput_tok_s") is not None]
    pts.sort(key=lambda r: r["throughput_tok_s"])
    under = [r for r in pts if r[key] <= slo]
    if not under:
        return None
    last = under[-1]
    over = [r for r in pts if r["throughput_tok_s"] > last["throughput_tok_s"]
            and r[key] > slo]
    if not over:
        return last["throughput_tok_s"]
    nxt = over[0]
    span = nxt[key] - last[key]
    if span <= 0:
        return last["throughput_tok_s"]
    frac = (slo - last[key]) / span
    return last["throughput_tok_s"] + frac * (nxt["throughput_tok_s"] - last["throughput_tok_s"])


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pat = re.compile(r"bench_(?P<arm>.+)_r(?P<rate>[\d.]+)(?:\.summary)?\.json$")
    curves = {}
    seen = set()
    # Full file first, summary-only as the fallback: when a disk-full killed the big
    # write the summary is all that survived, and it holds every field read below.
    _all = sorted(glob.glob(os.path.join(args.dir, "bench_*_r*.json")))
    paths = ([p for p in _all if not p.endswith(".summary.json")]
             + [p for p in _all if p.endswith(".summary.json")])
    for path in paths:
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        ident = (m.group("arm"), m.group("rate"))
        if ident in seen:
            continue
        b = _load(path)
        s = (b or {}).get("summary") or {}
        if s.get("mode") != "open_loop":
            print(f"[skip] {os.path.basename(path)}: not an open-loop run")
            continue
        unfinished = s.get("sessions_unfinished_at_drain") or 0
        launched = s.get("sessions_launched") or 1
        row = {
            "rate": float(m.group("rate")),
            "throughput_tok_s": s.get("window_throughput_tok_s"),
            "turn_rate_s": s.get("window_turn_rate_s"),
            "ttft_p50_s": s.get("window_ttft_p50_s"),
            "ttft_p95_s": s.get("window_ttft_p95_s"),
            "ttft_p99_s": s.get("window_ttft_p99_s"),
            # Job delay: session arrival -> session completion (all turns + think-time).
            # Present only on scripts ported to the job-delay convention (BFCL, ShareGPT
            # open-loop as of this addition); older result files simply lack the key and
            # these come back None, same as any other missing field here.
            "job_delay_mean_s": s.get("window_job_delay_mean_s"),
            "job_delay_p50_s": s.get("window_job_delay_p50_s"),
            "job_delay_p95_s": s.get("window_job_delay_p95_s"),
            "job_delay_p99_s": s.get("window_job_delay_p99_s"),
            "turns": s.get("window_turns"),
            "peak_inflight": s.get("peak_inflight_sessions"),
            "unfinished": unfinished,
            "launched": launched,
            "past_saturation": unfinished > 0.10 * launched,
            "wrapped": (s.get("sessions_repeated") or 0) > 0,
        }
        seen.add(ident)
        curves.setdefault(m.group("arm"), []).append(row)

    for arm in curves:
        curves[arm].sort(key=lambda r: r["rate"])

    if not curves:
        print(f"[error] no open-loop points in {args.dir}")
        return

    canon = ["recompute", "radix", "hicache", "hicache_memfrac", "park"]
    order = [a for a in canon if a in curves] + [a for a in curves if a not in canon]

    for arm in order:
        rows = curves[arm]
        print(f"\n### {arm}\n")
        print("| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 "
              "| job delay mean | job delay p95 | peak inflight | unfinished | flags |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            flags = " ".join(f for f, on in
                             (("SATURATED", r["past_saturation"]), ("WRAPPED", r["wrapped"]))
                             if on) or ""
            print(f"| {r['rate']} | {r['turn_rate_s']} | {r['throughput_tok_s']} | "
                  f"{r['ttft_p50_s']} | {r['ttft_p95_s']} | {r['job_delay_mean_s']} | "
                  f"{r['job_delay_p95_s']} | {r['peak_inflight']} | "
                  f"{r['unfinished']}/{r['launched']} | {flags} |")

    # SLO capacity: the most load an arm carries while STAYING under a latency budget.
    #
    # This, not peak throughput, is the comparison that survives a decode-bound server.
    # Peak throughput asks "how much can it push if latency is allowed to go anywhere",
    # and when the bottleneck is decode -- which KV placement does not touch -- every arm
    # answers the same and the metric separates nothing. Holding latency fixed asks the
    # question a deployment actually asks, and prefill work does move that answer.
    print("\n### SLO capacity (delivered tok/s while median TTFT stays under budget)\n")
    print("| arm | " + " | ".join(f"<= {s}s" for s in SLOS) + " |")
    print("|---|" + "---|" * len(SLOS))
    slo_cap = {}
    for arm in order:
        cells = []
        for slo in SLOS:
            cap = _slo_capacity(curves[arm], slo)
            slo_cap.setdefault(arm, {})[slo] = cap
            cells.append("—" if cap is None else f"{cap:.0f}")
        print(f"| {arm} | " + " | ".join(cells) + " |")
    for slo in SLOS:
        p = slo_cap.get("park", {}).get(slo)
        if not p:
            continue
        parts = []
        for base in ("recompute", "hicache"):
            b = slo_cap.get(base, {}).get(slo)
            if b:
                parts.append(f"{p / b:.2f}x {base}")
        if parts:
            print(f"\n- **at {slo}s median TTFT, Ours carries " + ", ".join(parts) + "**")

    # Capacity = the highest delivered throughput the arm reached. Reported alongside the
    # rate it needed, because "more tok/s" and "more tok/s at a lower offered rate" are
    # different claims and only the first is what the plateau shows.
    print("\n### capacity (peak delivered throughput)\n")
    print("| arm | peak tok/s | at rate | TTFT p50 there |")
    print("|---|---|---|---|")
    peaks = {}
    for arm in order:
        ok = [r for r in curves[arm] if r["throughput_tok_s"] is not None]
        if not ok:
            continue
        best = max(ok, key=lambda r: r["throughput_tok_s"])
        peaks[arm] = best
        print(f"| {arm} | {best['throughput_tok_s']} | {best['rate']} | "
              f"{best['ttft_p50_s']} |")
    if "park" in peaks:
        for base in ("recompute", "hicache"):
            if base in peaks and peaks[base]["throughput_tok_s"]:
                r = peaks["park"]["throughput_tok_s"] / peaks[base]["throughput_tok_s"]
                print(f"\n- **Ours vs {base}: {r:.2f}x peak throughput**")

    if any(r["wrapped"] for rows in curves.values() for r in rows):
        print("\n> WARNING: at least one point wrapped the corpus. Those points replayed "
              "conversations already served, which is a free cache hit for the cached "
              "arms and no help to recompute. Raise MAX_ITEMS and re-run them.")
    if not any(r["past_saturation"] for rows in curves.values() for r in rows):
        print("\n> NOTE: no point hit saturation. Delivered throughput may still be "
              "client-limited, in which case the plateau is not capacity. Extend RATES "
              "upward until TTFT knees and unfinished sessions appear.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"arms": {a: curves[a] for a in order}}, fh, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

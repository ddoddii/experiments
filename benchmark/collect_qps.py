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

    pat = re.compile(r"bench_(?P<arm>.+)_r(?P<rate>[\d.]+)\.json$")
    curves = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "bench_*_r*.json"))):
        m = pat.search(os.path.basename(path))
        if not m:
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
            "turns": s.get("window_turns"),
            "peak_inflight": s.get("peak_inflight_sessions"),
            "unfinished": unfinished,
            "launched": launched,
            "past_saturation": unfinished > 0.10 * launched,
            "wrapped": (s.get("sessions_repeated") or 0) > 0,
        }
        curves.setdefault(m.group("arm"), []).append(row)

    for arm in curves:
        curves[arm].sort(key=lambda r: r["rate"])

    if not curves:
        print(f"[error] no open-loop points in {args.dir}")
        return

    canon = ["recompute", "radix", "hicache", "park"]
    order = [a for a in canon if a in curves] + [a for a in curves if a not in canon]

    for arm in order:
        rows = curves[arm]
        print(f"\n### {arm}\n")
        print("| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 "
              "| peak inflight | unfinished | flags |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            flags = " ".join(f for f, on in
                             (("SATURATED", r["past_saturation"]), ("WRAPPED", r["wrapped"]))
                             if on) or ""
            print(f"| {r['rate']} | {r['turn_rate_s']} | {r['throughput_tok_s']} | "
                  f"{r['ttft_p50_s']} | {r['ttft_p95_s']} | {r['peak_inflight']} | "
                  f"{r['unfinished']}/{r['launched']} | {flags} |")

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

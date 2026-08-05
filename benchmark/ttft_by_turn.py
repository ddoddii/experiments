#!/usr/bin/env python3
"""
TTFT by turn index across arms -- the view that separates mechanism from overhead.

An aggregate TTFT tells you an arm is slower; it cannot tell you why. Turn 0 has no
reusable prefix and triggers no fetch in ANY arm, so a difference there is a constant
per-request overhead (memory budget, extra scheduler work) rather than the placement
mechanism. Turns >=1 are where reuse can pay. Exp 1 was read wrongly until this split was
made: park looked like it was paying a fetch cost, but it was +0.56 s on turn 0.

Also prints the SIZE OF THE PRIZE: median context in tokens and what a full re-prefill of
that context costs, from results/intro/fig_ttft_ctx_sweep.json. If the prize is smaller
than the overhead, the workload cannot demonstrate the mechanism no matter how well it
works.

사용:
  python benchmark/ttft_by_turn.py --dir results/exp1/p60000_c16_d3 \
      --arms radix hicache park
"""
import argparse
import bisect
import json
import os
import statistics as st

# measured re-prefill cost, results/intro/fig_ttft_ctx_sweep.json
CTX_MS = [(1000, 161), (2000, 293), (4000, 592), (8000, 1203), (16000, 2706), (32000, 6681)]


def reprefill_ms(tokens):
    xs = [c for c, _ in CTX_MS]
    i = bisect.bisect_left(xs, tokens)
    if i == 0:
        return CTX_MS[0][1] * tokens / CTX_MS[0][0]      # linear below the first point
    if i >= len(CTX_MS):
        return CTX_MS[-1][1]
    (x0, y0), (x1, y1) = CTX_MS[i - 1], CTX_MS[i]
    return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)


def load(path, max_turn=3):
    by = {}
    with open(path) as fh:
        d = json.load(fh)
    for item in d.get("results") or []:
        for t in item.get("turns") or []:
            if t.get("ttft_s") is None:
                continue
            # prompt_tokens is exact (ShareGPT runner); context_chars needs the /4
            # estimate (BFCL runner). Prefer the exact one and mark which was used.
            if t.get("prompt_tokens"):
                tok, exact = float(t["prompt_tokens"]), True
            else:
                tok, exact = t.get("context_chars", 0), False
            by.setdefault(min(t["turn"], max_turn), []).append((t["ttft_s"], tok, exact))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["radix", "hicache", "park"])
    ap.add_argument("--baseline", default="radix", help="arm the deltas are measured from")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    args = ap.parse_args()

    arms = {}
    for a in args.arms:
        p = os.path.join(args.dir, f"bench_{a}.json")
        if os.path.exists(p):
            arms[a] = load(p)
        else:
            print(f"[warn] missing {p}")
    if not arms:
        return
    base = args.baseline if args.baseline in arms else list(arms)[0]

    print(f"median TTFT by turn index (baseline = {base})\n")
    hdr = f"{'turn':>5} {'ctx tok':>8} " + "".join(f"{a:>20}" for a in arms)
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(arms[base]):
        vals = arms[base][k]
        exact = vals[0][2] if vals else False
        ctx = st.median(x[1] for x in vals) / (1.0 if exact else args.chars_per_token)
        label = ">=3" if k == 3 else str(k)
        row = f"{label:>5} {ctx:>8.0f} "
        b = st.median(x[0] for x in arms[base][k])
        for a in arms:
            v = arms[a].get(k)
            if not v:
                row += f"{'-':>20}"
                continue
            m = st.median(x[0] for x in v)
            d = "" if a == base else f" ({m - b:+.2f})"
            row += f"{m:>13.2f}s{d:>7}"
        print(row)

    print()
    allctx = [x[1] / (1.0 if x[2] else args.chars_per_token)
              for v in arms[base].values() for x in v]
    med = st.median(allctx)
    prize = reprefill_ms(med)
    t0 = {a: st.median(x[0] for x in arms[a][0]) for a in arms if 0 in arms[a]}
    print(f"median context {med:.0f} tokens -> a FULL re-prefill costs ~{prize:.0f} ms,")
    print(f"which is the most any amount of KV reuse can save on one turn.")
    for a, v in t0.items():
        if a == base:
            continue
        oh = (v - t0[base]) * 1000
        verdict = ("swamps the prize" if oh > prize else "below the prize")
        print(f"  {a:<10} turn-0 overhead {oh:+7.0f} ms  -- {verdict}")
    print("\nTurn-0 differences are NOT the mechanism: no arm can reuse or fetch there.")


if __name__ == "__main__":
    main()

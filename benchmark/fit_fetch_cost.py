#!/usr/bin/env python3
"""
Fit fetch cost per tier from the in-situ per-fetch trace:  ms = a + bytes / B.

THIS IS THE MEASUREMENT THAT SEPARATES THE TWO HYPOTHESES, and they are separated by the
two coefficients of one line:

    a  (intercept, ms)   fixed per-call cost -- index lookup, allocation, pinning,
                         handshake, thread hop, kernel launches. "The framework is slow."
    B  (slope, GB/s)     achieved bandwidth of the medium.  "PCIe is slow."

A microbenchmark can only give B, and gives it under conditions no server ever sees: one
big contiguous copy, warm buffers, nothing else running. The trace is written from the
real serving path, so it carries a as well -- and a is the term that scales with the
NUMBER of transfers rather than their size, which is exactly the "GPU fetches in one lump,
host fetches in dribs" question. A tier that moves the same bytes in more calls pays a
more times, and that shows up as a large intercept, not as low bandwidth.

READ ORDER MATTERS. Fit B first and compare it to results/nvlink_microbench.json: if the
fitted B is close to the microbenchmark, the medium is being driven at hardware speed and
anything left over is a. If the fitted B is far below it, the transfer itself is
inefficient (too many small copies) and a and B are entangled -- the script says so rather
than reporting a clean split.

REFUSES TO FIT ASYNC ROWS. With SGLANG_KV_PARK_SYNC_FETCH=0 (the default) the copy is
enqueued and never awaited, so the recorded ms is scheduler time. Fitting it yields
bandwidths in the TB/s and an intercept near zero -- a result that looks like a discovery
and is an artefact. Rows carry the sync flag; async rows are reported separately as
enqueue cost and excluded from every fit.

사용:
  python benchmark/fit_fetch_cost.py --dir results/why/why_c32_p32000 \
      --out results/why/why_c32_p32000/fig_fetch_cost
"""
import argparse
import csv
import glob
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

INK, MUTED = PALETTE["ink"], PALETTE["muted"]
TIER_COLOR = {"local": "#0072B2", "peer": PALETTE["park"], "host": "#E69F00"}
# results/nvlink_microbench.json -- the hardware ceiling each fitted slope is judged against.
MICRO_GBPS = {"local": 52.7, "peer": 52.7, "host": 26.3}
PREFILL_MS_PER_TOK = 0.131656


def load(d):
    rows = []
    for p in glob.glob(os.path.join(d, "trace_*", "fetch_gpu*.csv")):
        arm = os.path.basename(os.path.dirname(p))[len("trace_"):]
        for r in csv.DictReader(open(p)):
            try:
                rows.append({"arm": arm, "tier": r["tier"],
                             "tokens": int(r["tokens"]), "bytes": int(r["bytes"]),
                             "ms": float(r["ms"]), "sync": int(r["sync"])})
            except (KeyError, ValueError):
                continue
    return rows


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, r2) or None if x has no spread -- a
    single transfer size cannot separate a fixed cost from a per-byte one."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load(args.dir)
    if not rows:
        raise SystemExit(f"no trace_*/fetch_gpu*.csv under {args.dir} -- "
                         f"run with COSTMODEL=1 (SGLANG_KV_PARK_FETCH_TRACE)")

    a_rows = [r for r in rows if not r["sync"]]
    s_rows = [r for r in rows if r["sync"]]
    print(f"{len(rows)} fetches: {len(s_rows)} synchronous (fittable), "
          f"{len(a_rows)} async (enqueue only)")
    if a_rows:
        by = {}
        for r in a_rows:
            by.setdefault(r["tier"], []).append(r["ms"])
        for t, v in sorted(by.items()):
            print(f"  async {t:>5}: {len(v):>5} fetches, enqueue median {st.median(v):.3f} ms "
                  f"-- the transfer overlaps the forward pass and is off the critical path")
    if not s_rows:
        raise SystemExit("nothing to fit: every row is async. Re-run the cost-model arm "
                         "with SGLANG_KV_PARK_SYNC_FETCH=1.")

    fits = {}
    print(f"\n{'tier':>6} {'n':>6} {'MB median':>10} {'intercept ms':>13} "
          f"{'slope GB/s':>11} {'r2':>6} {'micro GB/s':>11}")
    for tier in ("local", "peer", "host"):
        v = [r for r in s_rows if r["tier"] == tier]
        if not v:
            continue
        gb = [r["bytes"] / 2**30 for r in v]
        ms = [r["ms"] for r in v]
        f = ols(gb, ms)
        med_mb = st.median([r["bytes"] / 2**20 for r in v])
        if f is None:
            print(f"{tier:>6} {len(v):>6} {med_mb:>10.1f} "
                  f"{'(single size -- cannot separate a from B)':>44}")
            continue
        a, b, r2 = f
        bw = 1e3 / b if b > 0 else float("nan")   # ms per GiB -> GiB/s
        fits[tier] = (a, bw, r2, len(v), med_mb)
        print(f"{tier:>6} {len(v):>6} {med_mb:>10.1f} {a:>13.3f} {bw:>11.1f} "
              f"{r2:>6.2f} {MICRO_GBPS.get(tier, float('nan')):>11.1f}")

    print("\n=== reading the fit ===")
    for tier, (a, bw, r2, n, med_mb) in fits.items():
        micro = MICRO_GBPS.get(tier)
        eff = 100 * bw / micro if micro else float("nan")
        typ_ms = a + med_mb / 1024 / bw * 1e3 if bw > 0 else float("nan")
        share = 100 * a / typ_ms if typ_ms > 0 else float("nan")
        print(f"  {tier}: at the median {med_mb:.0f} MB fetch, cost ~{typ_ms:.1f} ms, of "
              f"which the fixed per-call term is {a:.2f} ms ({share:.0f}%).")
        print(f"         achieved {bw:.1f} GB/s = {eff:.0f}% of the {micro} GB/s "
              f"microbenchmark ceiling."
              + ("" if eff > 60 else
                 "  LOW: the transfer itself is inefficient, so the\n"
                 "         intercept and slope are entangled and the split above is not "
                 "clean."))
    if "host" in fits and ("peer" in fits or "local" in fits):
        g = fits.get("peer") or fits.get("local")
        h = fits["host"]
        med_mb = st.median([r["bytes"] / 2**20 for r in s_rows])
        cost_g = g[0] + med_mb / 1024 / g[1] * 1e3
        cost_h = h[0] + med_mb / 1024 / h[1] * 1e3
        d_fixed = h[0] - g[0]
        d_bw = (cost_h - cost_g) - d_fixed
        print(f"\n  host vs GPU at a {med_mb:.0f} MB fetch: {cost_h:.1f} vs {cost_g:.1f} ms, "
              f"a gap of {cost_h - cost_g:+.1f} ms")
        print(f"     of which fixed per-call overhead {d_fixed:+.1f} ms  "
              f"({100 * d_fixed / (cost_h - cost_g):.0f}%)"
              if abs(cost_h - cost_g) > 1e-9 else "")
        print(f"                  bandwidth          {d_bw:+.1f} ms  "
              f"({100 * d_bw / (cost_h - cost_g):.0f}%)"
              if abs(cost_h - cost_g) > 1e-9 else "")
        # The comparison that decides whether any of this matters at all.
        tok = st.median([r["tokens"] for r in s_rows])
        print(f"     for reference, re-prefilling those {tok:.0f} tokens costs "
              f"{PREFILL_MS_PER_TOK * tok:.0f} ms -- "
              f"the whole host/GPU gap is {100 * (cost_h - cost_g) / (PREFILL_MS_PER_TOK * tok):.1f}% "
              f"of what a hit saves either way")

    if not args.out:
        return
    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 2.8))
    for tier, (a, bw, r2, n, _m) in fits.items():
        v = [r for r in s_rows if r["tier"] == tier]
        ax.scatter([r["bytes"] / 2**20 for r in v], [r["ms"] for r in v], s=5,
                   color=TIER_COLOR[tier], alpha=0.35, linewidths=0, zorder=3)
        xs = [0, max(r["bytes"] for r in v) / 2**20]
        ax.plot(xs, [a + x / 1024 / bw * 1e3 for x in xs], color=TIER_COLOR[tier],
                lw=1.4, zorder=4,
                label=f"{tier}: {a:.2f} ms + {bw:.0f} GB/s")
    ax.set_xlabel("fetch size (MB)")
    ax.set_ylabel("fetch time (ms)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.5)
    ax.text(0.98, 0.04, "intercept = per-call software cost;  slope = link",
            transform=ax.transAxes, ha="right", fontsize=5.8, color=MUTED, style="italic")
    style_axes(ax)
    fig.tight_layout(pad=0.4)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
How much tool-call gap does the advantage need? -- the break-even, not an assumed value.

WHY THIS EXISTS. The gap in these runs is a time.sleep in the harness; the benchmark
replays recorded turns and does not execute the tools. So any single TOOL_DELAY is a
number somebody chose, and a result quoted at one value is partly a result about that
choice. Swept, it becomes a measurement: the gap is idle time that background work can
hide in, so the question is how much of it the advantage requires.

That break-even is comparable to something real. Tool calls in deployed agents cost
roughly: local file operations a few ms, database queries 10-500 ms, HTTP APIs
100 ms-2 s, sandboxed code execution seconds. A break-even in the tens of milliseconds
means the claim covers essentially every agentic deployment. A break-even of seconds means
it covers few, and the paper should say so.

THE GAP IS NOT A HANDICAP HANDED TO ONE SIDE. hicache's writeback is a background thread
and uses idle time too. Both arms get the same gap; what the sweep shows is which one
converts it better.

READ THE ZERO-GAP POINT FIRST. At gap 0 there is no idle time at all, so any difference
there is not about hiding -- it is the floor the rest of the curve is measured from.

사용:
  GAPS="0 0.25 0.5 1 2 3" ARMS="hicache park_gpu" WORKLOAD=bfcl \
    ./scripts/sglang/run_why_faster.sh
  python benchmark/plot_gap_sweep.py --dir results/why/bfcl_gap --out results/why/fig_gap
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

INK, MUTED = PALETTE["ink"], PALETTE["muted"]
COLOR = {"hicache": "#0072B2", "park_gpu": PALETTE["park"], "park_host": "#E69F00",
         "recompute": "#999999"}
# Rough cost of real tool calls, for judging whether a break-even is reachable in practice.
REAL = [("file op", 0.005), ("DB query", 0.1), ("HTTP API", 0.5), ("sandboxed exec", 2.0)]


def load(d):
    """arm -> {gap: [summaries]}. Filenames are bench_<arm>_g<gap>[_r<rep>].json."""
    out = {}
    for p in glob.glob(os.path.join(d, "bench_*_g*.json")):
        m = re.match(r"bench_(.+?)_g([0-9.]+)(?:_r\d+)?\.json$", os.path.basename(p))
        if not m:
            continue
        arm, gap = m.group(1), float(m.group(2))
        try:
            s = json.load(open(p))["summary"]
        except Exception:  # noqa: BLE001
            continue
        out.setdefault(arm, {}).setdefault(gap, []).append(s)
    return out


def med(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--metric", default="avg_ttft_s")
    ap.add_argument("--base", default="hicache")
    ap.add_argument("--arm", default="park_gpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load(args.dir)
    if args.base not in data or args.arm not in data:
        raise SystemExit(f"need both {args.base} and {args.arm} in {args.dir}; "
                         f"found {sorted(data)}")
    gaps = sorted(set(data[args.base]) & set(data[args.arm]))
    if not gaps:
        raise SystemExit("no gap values shared by both arms")

    print(f"\n{'gap s':>7} {args.base:>10} {args.arm:>10} {'delta':>8} {'n':>4}")
    rel = []
    for g in gaps:
        b = med([s.get(args.metric) for s in data[args.base][g]])
        a = med([s.get(args.metric) for s in data[args.arm][g]])
        n = min(len(data[args.base][g]), len(data[args.arm][g]))
        if b is None or a is None or not b:
            continue
        d = 100 * (a - b) / b
        rel.append((g, d, a, b))
        print(f"{g:>7.2f} {b:>10.3f} {a:>10.3f} {d:>+7.1f}% {n:>4}")

    # Break-even: the smallest gap at which the arm stops losing. Interpolated between the
    # bracketing points rather than snapped to a sampled value, so the answer is not an
    # artefact of which gaps happened to be run.
    be = None
    for i in range(1, len(rel)):
        g0, d0, _, _ = rel[i - 1]
        g1, d1, _, _ = rel[i]
        if d0 > 0 >= d1:
            be = g0 + (g1 - g0) * (d0 / (d0 - d1)) if d0 != d1 else g1
            break
    print(f"\n=== break-even ===")
    if rel and rel[0][1] <= 0:
        print(f"  {args.arm} already wins at gap {rel[0][0]:.2f} s -- no gap is required.")
        print("  The advantage does not depend on idle time, which is the strongest form")
        print("  of this result: it holds even for tools that return instantly.")
    elif be is not None:
        print(f"  {args.arm} overtakes {args.base} at a gap of ~{be * 1000:.0f} ms")
        reach = [n for n, c in REAL if c >= be]
        if reach:
            print(f"  Real tool calls at or above that: {', '.join(reach)}")
            print("  So the claim covers deployments whose tools are at least that slow.")
        else:
            print(f"  No common tool call is that slow ({be:.1f} s). State the requirement")
            print("  explicitly rather than presenting the win as general.")
    else:
        print(f"  {args.arm} does not overtake {args.base} at any gap tested "
              f"({gaps[0]:.2f}-{gaps[-1]:.2f} s).")
        print("  The gap is not the missing ingredient. Do not attribute the loss to it.")

    if not args.out:
        return
    use_paper_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for arm in (args.base, args.arm):
        xs = [g for g in gaps if med([s.get(args.metric) for s in data[arm][g]])]
        ys = [med([s.get(args.metric) for s in data[arm][g]]) for g in xs]
        ax.plot(xs, ys, "o-", ms=3, lw=1.4, color=COLOR.get(arm, INK), label=arm)
    ax.set_xlabel("tool-call gap (s)")
    ax.set_ylabel("TTFT (s)")
    ax.legend(frameon=False, fontsize=6.5, handlelength=1.6)
    style_axes(ax)

    ax2.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax2.plot([r[0] for r in rel], [r[1] for r in rel], "o-", ms=3, lw=1.4,
             color=COLOR.get(args.arm, INK))
    if be is not None:
        ax2.axvline(be, color=INK, lw=0.8, ls=":")
        ax2.annotate(f"break-even\n{be * 1000:.0f} ms", (be, 0), textcoords="offset points",
                     xytext=(6, 12), fontsize=6, color=INK)
    ax2.set_xlabel("tool-call gap (s)")
    ax2.set_ylabel(f"{args.arm} vs {args.base} (%)")
    ax2.text(0.98, 0.04, "below 0 = ours faster", transform=ax2.transAxes, ha="right",
             fontsize=5.8, color=MUTED, style="italic")
    style_axes(ax2)
    fig.tight_layout(pad=0.5)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

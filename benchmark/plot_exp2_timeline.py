#!/usr/bin/env python3
"""
Exp 2 main figure: idle decode HBM over time, what moves into it, and what comes back out.

The bar version of this figure was a single snapshot, and a snapshot cannot show the two
things that make the mechanism legitimate rather than lucky: that the idle capacity is
there for the WHOLE run (not a moment the experiment happened to catch), and that the KV
parked into it keeps being read back until the end.

WHAT THE TIME AXIS HONESTLY SHOWS -- read this before changing the figure:

  Filling is NOT gradual. The park pools reach 50% of their final size at t=29 s and 99%
  at t=55 s, i.e. inside the first 6% of a 950 s run, and are then flat to the second.
  Drawing a slow ramp would be a lie about the data. It is a step, and the step is drawn
  as a step. What limits it is the CONFIGURED per-GPU park size, not the available
  headroom -- roughly 87% of the idle capacity is never claimed, and the figure says so
  rather than cropping the axis to hide it.

  Reuse IS gradual and runs the length of the experiment: 50% of park-fetch hits land by
  t=392 s, 99% by t=843 s. That is the panel that earns a time axis, and it is where the
  two arms separate.

So the story the figure tells is "claimed at once, used throughout", which is what the
data says, not "the empty space gradually fills up", which it does not.

사용:
  python benchmark/plot_exp2_timeline.py --dirs 'results/exp2/repeat_nocap_c32_r*' \
      --out results/exp2/fig_exp2_timeline
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

BASE, OURS = "park_local", "park_pd"
C_PARK = PALETTE["park"]
C_ROOM = "#E3E7EA"          # idle capacity: present, unclaimed
C_BASE = "#9AA3AA"
INK, MUTED = PALETTE["ink"], PALETTE["muted"]
KIB_PER_TOKEN = 128         # Llama-3.1-8B; matches collect_imbalance.py


def _f(r, k):
    try:
        return float(r.get(k, "") or 0)
    except (TypeError, ValueError):
        return 0.0


def parked_series(d, arm, decode_gpus):
    """Parked KV over time: total, and the part sitting on the idle (decode) GPUs."""
    rows = list(csv.DictReader(open(os.path.join(d, f"parked_{arm}.csv"))))
    t = [_f(r, "elapsed_s") for r in rows]
    idle = [sum(_f(r, f"gpu{g}_gb") for g in decode_gpus) for r in rows]
    tot = [sum(_f(r, f"gpu{g}_gb") for g in range(4)) for r in rows]
    hits = [_f(r, "fetch_hits") for r in rows]
    ftok = [_f(r, "fetched_tokens") for r in rows]
    return {"t": t, "idle": idle, "total": tot, "hits": hits, "ftok": ftok}


def free_series(d, arm, labels=("D0", "D1")):
    """Free KV capacity on the decode nodes, in GiB, over time.

    Rows where a node is unreachable are skipped rather than counted as zero free: the
    sampler records a blank until a gauge is first published, and treating that as "no
    idle capacity" would understate exactly the quantity this panel exists to show.
    """
    path = os.path.join(d, f"occ_{arm}.csv")
    rows = [r for r in csv.DictReader(
        (l for l in open(path) if not l.startswith("#")))]
    t, free = [], []
    for r in rows:
        vals = []
        for lab in labels:
            cap, use = r.get(f"{lab}_cap_tok", ""), r.get(f"{lab}_use", "")
            if not cap or not use:
                continue
            vals.append(float(cap) * (1.0 - float(use)))
        if not vals:
            continue
        t.append(_f(r, "t_s"))
        free.append(sum(vals) * KIB_PER_TOKEN / (1024 * 1024))
    return t, free


def rate_series(d, arm):
    """Running park-fetch hit rate, hits/(hits+miss), from the cumulative counters."""
    rows = list(csv.DictReader(open(os.path.join(d, f"parked_{arm}.csv"))))
    t, r = [], []
    for row in rows:
        h, m = _f(row, "fetch_hits"), _f(row, "fetch_miss")
        if h + m < 20:      # rate is meaningless over the first handful of requests
            continue
        t.append(_f(row, "elapsed_s"))
        r.append(100 * h / (h + m))
    return t, r


def median_curve(series, n=200):
    """Pointwise median across repeats on a shared grid.

    Repeats differ in length, so the median is taken only where every repeat still has
    data; extending the shortest run's last value across the others would flatten the
    tail of the curve into a constant that no run actually measured.
    """
    end = min(s[0][-1] for s in series)
    start = max(s[0][0] for s in series)
    grid = [start + (end - start) * i / (n - 1) for i in range(n)]
    out = []
    for x in grid:
        vals = []
        for t, r in series:
            j = next((k for k, tv in enumerate(t) if tv >= x), len(t) - 1)
            vals.append(r[j])
        out.append(st.median(vals))
    return grid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--decode-gpus", nargs="+", type=int, default=[1, 3],
                    help="PD_LAYOUT=b: decode on gpu1/gpu3. Passed, not inferred.")
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=3.5)
    ap.add_argument("--out", default="results/exp2/fig_exp2_timeline")
    args = ap.parse_args()

    dirs = sorted(sum((glob.glob(x) for x in args.dirs), []))
    if not dirs:
        raise SystemExit(f"no directories matched {args.dirs}")
    # One representative run for the traces -- averaging time series across repeats of
    # different lengths would smear the step that is the whole point of panel (a). The
    # repeat whose final fetch count is the median is used, so it is not cherry-picked.
    finals = [(parked_series(d, OURS, args.decode_gpus)["hits"][-1], d) for d in dirs]
    rep = sorted(finals)[len(finals) // 2][1]
    print(f"  {len(dirs)} run(s); traces from {os.path.basename(rep)} (median by fetch hits)")

    P = {a: parked_series(rep, a, args.decode_gpus) for a in (BASE, OURS)}
    tf, free = free_series(rep, OURS)

    # When the pools fill, and how much of the room they take.
    fin = P[OURS]["idle"][-1]
    t99 = next(t for t, v in zip(P[OURS]["t"], P[OURS]["idle"]) if v >= 0.99 * fin)
    room = st.median(free) if free else 0.0
    span = P[OURS]["t"][-1]
    print(f"  idle decode capacity (median)  {room:.1f} GiB")
    print(f"  parked onto idle GPUs          {fin:.2f} GB  = {100*fin/room:.0f}% of it")
    print(f"  99% of that in place by t={t99:.0f}s ({100*t99/span:.1f}% of the run)")
    for a in (BASE, OURS):
        print(f"  {a:11s} final fetch hits {P[a]['hits'][-1]:.0f}, "
              f"fetched tokens {P[a]['ftok'][-1]:,.0f}")

    use_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(args.width, args.height), sharex=True,
                                   gridspec_kw={"height_ratios": [1.45, 1], "hspace": 0.18})

    # (a) the room, and what moved into it.
    ax1.fill_between(tf, 0, free, color=C_ROOM, lw=0, zorder=1,
                     label="idle KV capacity on decode GPUs")
    ax1.plot(tf, free, color=MUTED, lw=0.6, zorder=2)
    ax1.fill_between(P[OURS]["t"], 0, P[OURS]["idle"], color=C_PARK, lw=0, zorder=3,
                     label="parked there (ours)")
    ax1.plot(P[BASE]["t"], P[BASE]["idle"], color=INK, lw=1.1, ls="--", zorder=4,
             label="parked there (local-only) — stays at 0")
    ax1.set_ylabel("KV on idle GPUs (GB)")
    ax1.set_ylim(0, max(free) * 1.18 if free else 1)
    ax1.set_title("Idle decode memory is there all run; borrowing claims it at once", pad=3)
    ax1.legend(loc="upper right", frameon=False, fontsize=5.8, handlelength=1.4,
               labelspacing=0.28, borderaxespad=0.3)

    # The step, called a step. And the ceiling that is never approached -- the limit is the
    # configured park size, so the panel must not read as "the idle memory is now in use".
    ax1.annotate(f"full in {t99:.0f} s\n({100*t99/span:.0f}% of the run)",
                 xy=(t99, fin), xytext=(span * 0.13, max(free) * 0.55),
                 fontsize=5.8, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.7,
                                 connectionstyle="arc3,rad=0.25"))
    ax1.text(span * 0.98, room * 0.62,
             f"{100*(1-fin/room):.0f}% of the idle capacity is still unused\n"
             f"(limit is the configured park size, not the headroom)",
             fontsize=5.6, color=MUTED, ha="right", va="center", style="italic")
    style_axes(ax1)

    # (b) what the borrowed room buys, sustained. Running park-fetch hit rate rather than
    # cumulative tokens: the token totals differ by only 3% because both arms fetch from
    # SOMETHING, and the claim is not "more reuse happens" but "a larger resident cache
    # keeps a higher share of requests off the recompute path". Every repeat is drawn --
    # the bands are separated at the end and for most of the run, but r1 crosses in the
    # middle, and a median-only panel would have concealed the one repeat that disagrees.
    for a, c, lab in ((BASE, C_BASE, "local-only"), (OURS, C_PARK, "ours (GPU-first)")):
        series = [rate_series(d, a) for d in dirs]
        for t, r in series:
            ax2.plot(t, r, color=c, lw=0.55, alpha=0.5, zorder=2)
        grid, med = median_curve(series)
        ax2.plot(grid, med, color=c, lw=1.5, zorder=3,
                 label=f"{lab}  (n={len(series)})")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("park-fetch hit rate\n(running, %)")
    ax2.set_title("…and holds a higher reuse rate for the rest of the run", pad=3)
    # Start after the warm-up transient: the first few requests give a rate of 0 or 100%
    # and an auto-scaled axis spends most of its height on that artefact.
    ax2.set_xlim(30, span)
    # A zoomed y-range. The quantity is a rate whose interesting variation lives in a
    # 15-point window, and a 0-based axis renders both arms as the same flat line; the
    # tick labels carry the absolute level so the zoom cannot be read as a bigger effect
    # than it is.
    ax2.set_ylim(35, 72)
    ax2.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.6,
               borderaxespad=0.3, ncol=2)
    style_axes(ax2)

    fig.tight_layout(pad=0.4)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

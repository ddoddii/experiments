#!/usr/bin/env python3
"""
KV-pool occupancy over time, one axes, four lines: {SGLang, Ours} x {Prefill, Decode}.

THIS IS POOL UTILISATION, NOT nvidia-smi. SGLang reserves its whole KV pool at startup
from --mem-fraction-static, so a card-level memory reading is flat from the first second
no matter how much KV is resident and can never show a cache filling up. The curve that
rises and saturates is occupancy = (max_total - available) / max_total, which only the
server knows -- see kv_usage_sampler.py, whose CSVs this reads.

WHY NOT token_usage, WHICH SOUNDS RIGHT: SGLang defines it as
(max_total - available - evictable) / max_total, so a pool that is 100% full of reusable
prefix cache and a pool that is genuinely empty both report ~0. Using it here would show
the arm with the best cache as the emptiest. The sampler records it alongside occupancy
so the two stay distinguishable; --stat token_usage plots it deliberately.

ENCODING: hue is the system, shade is the role -- one legend entry per line as in the
reference figure, but arms stay recognisable by colour family rather than needing the
reader to memorise four unrelated colours.

사용:
  python benchmark/plot_kv_occupancy.py --dir results/agent_trace --c 4 \
      --out results/agent_trace/fig_kv_occupancy
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

# (arm key, label, prefill colour, decode colour)
ARMS = [
    ("recompute", "Recompute", "#5A5F63", "#A8ADB1"),
    ("hicache", "SGLang", PALETTE["hicache"], "#7FC4E8"),
    ("park_host", "Ours (host DRAM)", "#B87A00", "#F2C266"),
    ("park", "Ours", PALETTE["park"], "#7FD9BE"),
]
ROLES = [("prefill", "Prefill", "-"), ("decode", "Decode", "-")]


def load(path, stat):
    """(t, {role: [percent]}) from a kv_usage_sampler CSV."""
    t, out = [], {r: [] for r, _, _ in ROLES}
    col = {"occupancy": "{r}_occupancy_frac",
           "token_usage": "{r}_token_usage"}[stat]
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], out
    for r in rows:
        try:
            ts = float(r["elapsed_s"])
        except (KeyError, ValueError):
            continue
        vals = {}
        for role, _, _ in ROLES:
            name = col.format(r=role)
            # token_usage has no role aggregate in the CSV (it is per node), so average
            # the per-node columns for that stat rather than silently dropping the role.
            if name in r:
                raw = r[name]
            else:
                per = [r[k] for k in r if k.startswith(role) and k.endswith("_token_usage")]
                per = [x for x in per if x not in ("", None)]
                raw = str(sum(map(float, per)) / len(per)) if per else ""
            if raw in ("", None):
                vals = {}
                break
            vals[role] = float(raw) * 100
        if not vals:
            continue
        t.append(ts)
        for role in vals:
            out[role].append(vals[role])
    return t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/agent_trace")
    ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--arms", default="hicache,park",
                     help="comma separated; four lines per pair of arms is already the "
                          "practical limit for one axes")
    ap.add_argument("--stat", default="occupancy",
                     choices=["occupancy", "token_usage"])
    ap.add_argument("--out", default="results/agent_trace/fig_kv_occupancy")
    ap.add_argument("--xmax", type=float, default=None, help="clip the time axis (s)")
    ap.add_argument("--width", type=float, default=3.6)
    ap.add_argument("--height", type=float, default=2.8)
    args = ap.parse_args()

    want = [a.strip() for a in args.arms.split(",")]
    series = []
    for key, label, cpre, cdec in ARMS:
        if key not in want:
            continue
        p = os.path.join(args.dir, f"kv_{key}_c{args.c}.csv")
        if not os.path.exists(p):
            print(f"[warn] {p} not found -- run the sweep with kv_usage_sampler wired in")
            continue
        t, roles = load(p, args.stat)
        if t:
            series.append((label, t, roles, cpre, cdec))
    if not series:
        raise SystemExit(
            f"no kv_*_c{args.c}.csv in {args.dir}. This figure needs the pool-occupancy "
            f"sampler; the mem_*.csv files hold nvidia-smi readings, which are flat by "
            f"construction and cannot show a cache filling.")

    use_paper_style()
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for label, t, roles, cpre, cdec in series:
        for role, rlabel, ls in ROLES:
            ys = roles[role]
            if not ys:
                continue
            n = min(len(t), len(ys))
            ax.plot(t[:n], ys[:n], color=cpre if role == "prefill" else cdec,
                    ls=ls, lw=1.6, label=f"{label} {rlabel}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(("KV pool occupancy (%)" if args.stat == "occupancy"
                   else "token_usage (%)"))
    ax.set_ylim(0, 100)
    if args.xmax:
        ax.set_xlim(0, args.xmax)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, args.out)

    print(f"\nfinal occupancy (C={args.c}, {args.stat}):")
    for label, t, roles, _, _ in series:
        parts = [f"{r} {roles[r][-1]:5.1f}%" for r, _, _ in ROLES if roles[r]]
        print(f"  {label:20s} " + "   ".join(parts))


if __name__ == "__main__":
    main()

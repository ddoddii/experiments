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


def load(path, stat, unit="pct", bytes_per_token=131072):
    """(t, {role: [value]}) from a kv_usage_sampler CSV.

    unit="pct" divides by each node's own max_total_num_tokens. That is the right
    reading ONLY when the arms being compared were served with the SAME pool size --
    otherwise 90% of a 60k pool and 90% of a 112k pool draw as one line while differing
    by 1.9x in bytes held, which inverts the conclusion the figure exists to support.
    unit="gb" plots what is actually resident and is safe across configurations.
    """
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
            if unit == "gb":
                mx = r.get(f"{role}_max_tokens")
                if mx in ("", None):
                    vals = {}
                    break
                vals[role] = float(raw) * float(mx) * bytes_per_token / 2 ** 30
            else:
                vals[role] = float(raw) * 100
        if not vals:
            continue
        t.append(ts)
        for role in vals:
            out[role].append(vals[role])
    return t, out


def pool_sizes(path):
    """{role: max_tokens} from the first complete row, for the mismatch check."""
    with open(path) as fh:
        for r in csv.DictReader(fh):
            got = {}
            for role, _, _ in ROLES:
                v = r.get(f"{role}_max_tokens")
                if v not in ("", None) and float(v) > 0:
                    got[role] = int(float(v))
            if len(got) == len(ROLES):
                return got
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/agent_trace")
    ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--arms", default="hicache,park",
                     help="comma separated; four lines per pair of arms is already the "
                          "practical limit for one axes")
    ap.add_argument("--stat", default="occupancy",
                     choices=["occupancy", "token_usage"])
    ap.add_argument(
        "--unit", default="pct", choices=["pct", "gb"],
        help="pct is a fraction of each arm's OWN pool and is only comparable when the "
             "arms ran with the same --max-total-tokens; gb plots resident bytes and is "
             "safe when they did not. A mismatch is detected and refused in pct mode.")
    ap.add_argument("--bytes-per-token", type=int, default=131072,
                     help="2 x 32 layers x 8 kv heads x 128 dim x 2 B (Llama-3.1-8B GQA)")
    ap.add_argument("--out", default="results/agent_trace/fig_kv_occupancy")
    ap.add_argument("--xmax", type=float, default=None, help="clip the time axis (s)")
    ap.add_argument(
        "--smooth", type=int, default=0,
        help="centred rolling-mean window in SAMPLES (sampler interval is 2s by "
             "default, so 15 ~= 30s). The decode pool genuinely oscillates request to "
             "request; at a few thousand samples that fills the panel with ink and "
             "hides the level. Smoothing is stated in the axis label rather than done "
             "silently, and the printed final values below stay UNSMOOTHED so a quoted "
             "number is never a filter artefact.")
    ap.add_argument("--width", type=float, default=3.6)
    ap.add_argument("--height", type=float, default=2.8)
    args = ap.parse_args()

    want = [a.strip() for a in args.arms.split(",")]
    series, pools = [], {}
    for key, label, cpre, cdec in ARMS:
        if key not in want:
            continue
        p = os.path.join(args.dir, f"kv_{key}_c{args.c}.csv")
        if not os.path.exists(p):
            print(f"[warn] {p} not found -- run the sweep with kv_usage_sampler wired in")
            continue
        t, roles = load(p, args.stat, args.unit, args.bytes_per_token)
        if t:
            series.append((label, t, roles, cpre, cdec))
            pools[label] = pool_sizes(p)
    if not series:
        raise SystemExit(
            f"no kv_*_c{args.c}.csv in {args.dir}. This figure needs the pool-occupancy "
            f"sampler; the mem_*.csv files hold nvidia-smi readings, which are flat by "
            f"construction and cannot show a cache filling.")

    # A percentage is only a comparison if the denominators agree. These runs are served
    # with --max-total-tokens set per arm, so a sizing change to one arm silently rescales
    # only that arm's axis; the resulting figure looks fine and means nothing.
    for role, _, _ in ROLES:
        sizes = {lbl: p[role] for lbl, p in pools.items() if role in p}
        # 1% tolerance: SGLang derives the pool from free HBM at startup, so two runs of
        # the SAME config land a few tokens apart (209036 vs 209038 here). Flagging that
        # would train the reader to ignore the warning that matters.
        if sizes and (max(sizes.values()) - min(sizes.values())) > 0.01 * max(sizes.values()):
            detail = ", ".join(f"{k} {v}" for k, v in sorted(sizes.items()))
            if args.unit == "pct":
                raise SystemExit(
                    f"{role} pools differ across arms ({detail}), so occupancy % is not "
                    f"comparable: the same percentage is a different number of bytes per "
                    f"arm. Re-run the baselines at the same --max-total-tokens, or pass "
                    f"--unit gb to plot resident bytes instead.")
            print(f"[note] {role} pools differ ({detail}); plotting GB, so the lines are "
                  f"comparable, but the arms did not have equal capacity.")

    use_paper_style()
    def smooth(ys):
        w = args.smooth
        if w < 2 or len(ys) < w:
            return ys
        out, half = [], w // 2
        for i in range(len(ys)):
            lo, hi = max(0, i - half), min(len(ys), i + half + 1)
            out.append(sum(ys[lo:hi]) / (hi - lo))
        return out

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for label, t, roles, cpre, cdec in series:
        for role, rlabel, ls in ROLES:
            ys = roles[role]
            if not ys:
                continue
            ys = smooth(ys)
            n = min(len(t), len(ys))
            ax.plot(t[:n], ys[:n], color=cpre if role == "prefill" else cdec,
                    ls=ls, lw=1.6, label=f"{label} {rlabel}")
    ax.set_xlabel("time (s)")
    if args.unit == "gb":
        base = "Resident KV (GB)" if args.stat == "occupancy" else "token_usage (GB)"
    else:
        base = "KV pool occupancy (%)" if args.stat == "occupancy" else "token_usage (%)"
    ax.set_ylabel(base + (f"\n(rolling mean, {args.smooth} samples)"
                          if args.smooth >= 2 else ""))
    ax.set_ylim(0, 100 if args.unit == "pct" else None)
    if args.xmax:
        ax.set_xlim(0, args.xmax)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, args.out)

    suffix = "GB" if args.unit == "gb" else "%"
    print(f"\nfinal (C={args.c}, {args.stat}, {args.unit}):")
    for label, t, roles, _, _ in series:
        parts = [f"{r} {roles[r][-1]:6.1f}{suffix}" for r, _, _ in ROLES if roles[r]]
        pool = pools.get(label, {})
        cap = "  pool=" + "/".join(f"{pool[r]}" for r, _, _ in ROLES if r in pool)
        print(f"  {label:20s} " + "   ".join(parts) + cap)


if __name__ == "__main__":
    main()

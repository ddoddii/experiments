#!/usr/bin/env python3
"""
2P2D per-GPU KV occupancy imbalance (paper intro / motivation figure).

Reads kv_occupancy_timeseries.py CSV and shows the idle-GPU OPPORTUNITY that
motivates KV victim caching:
  (a) per-GPU KV pool usage over time (P0/P1/D0/D1) + saturation/headroom lines.
  (b) max/min occupancy envelope across the 4 GPUs.
Both panels shade the "opportunity" spans -- instants where AT LEAST ONE GPU is
saturated (usage >= hi) AND AT LEAST ONE GPU has headroom (usage <= lo), i.e. a
busy GPU and an idle GPU coexist so KV could be relocated. In panel (b) the shading
is visually exact: shaded  <=>  max-line above `hi` AND min-line below `lo`.

Note: the criterion is the two-threshold coexistence, NOT "spread > lo". A shaded
instant only needs max>=hi and min<=lo, so its spread (max-min) can be as small as
hi-lo (e.g. 0.3) -- that is expected, not a mislabel.

사용: python benchmark/plot_kv_imbalance.py --csv results/kv_ts/2p2d_p20k.csv \
        --hi 0.8 --lo 0.5 --out results/kv_ts/fig_imbalance
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

# role-grouped, colorblind-safe: prefill = blues, decode = amber/vermilion.
# solid/dashed within a role so the pair is separable in grayscale.
SERIES = {
    "P0": ("#0072B2", "-"),
    "P1": ("#56B4E9", "--"),
    "D0": ("#E69F00", "-"),
    "D1": ("#D55E00", "--"),
}
SHADE = "#9C7A3C"   # muted amber wash for opportunity spans
INK, MUTED = PALETTE["ink"], PALETTE["muted"]


def load(csv_path):
    rows = [ln.rstrip("\n") for ln in open(csv_path) if not ln.startswith("#")]
    reader = csv.DictReader(rows)
    labels = [c[:-4] for c in reader.fieldnames if c.endswith("_use")]
    data = {"t": []} | {l: [] for l in labels}
    for r in reader:
        try:
            data["t"].append(float(r["t_s"]))
        except (ValueError, KeyError):
            continue
        for l in labels:
            v = r.get(f"{l}_use", "")
            data[l].append(float(v) if v not in ("", None) else None)
    return data, labels


def opp_spans(t, opp):
    """Merge consecutive opportunity samples into (t_start, t_end) intervals."""
    spans, i, n = [], 0, len(t)
    while i < n:
        if opp[i]:
            j = i
            while j + 1 < n and opp[j + 1]:
                j += 1
            lo = t[i] - (t[i] - t[i - 1]) / 2 if i > 0 else t[i]
            hi = t[j] + (t[j + 1] - t[j]) / 2 if j + 1 < n else t[j]
            spans.append((lo, hi))
            i = j + 1
        else:
            i += 1
    return spans


def main_role(args, data, labels, t):
    """Role-conditioned variant: D-envelope (max over decode GPUs) vs P-envelope (min
    over prefill GPUs), shading instants where D >= d_hi AND P <= p_lo at the SAME
    instant. This is the defensible version of the "decode saturates while prefill
    idles" claim -- unlike the plain max/min-over-all-4-GPUs mode, it doesn't let a
    P-vs-P or D-vs-D spread masquerade as a P-vs-D asymmetry."""
    p_labels = [l for l in labels if l.upper().startswith("P")]
    d_labels = [l for l in labels if l.upper().startswith("D")]
    if not p_labels or not d_labels:
        raise SystemExit(f"--role needs both P* and D* columns; found labels={labels}")

    d_env, p_env, cond = [], [], []
    for i in range(len(t)):
        dv = [data[l][i] for l in d_labels if data[l][i] is not None]
        pv = [data[l][i] for l in p_labels if data[l][i] is not None]
        if dv and pv:
            d_env.append(max(dv)); p_env.append(min(pv))
            cond.append(1 if (max(dv) >= args.d_hi and min(pv) <= args.p_lo) else 0)
        else:
            d_env.append(None); p_env.append(None); cond.append(0)
    frac = sum(cond) / len(cond) if cond else 0.0
    spans = opp_spans(t, cond)

    print(f"\n=== 2P2D role-conditioned imbalance ({os.path.basename(args.csv)}) ===")
    print(f"  P labels={p_labels}  D labels={d_labels}")
    print(f"  D>={args.d_hi:g} AND P<={args.p_lo:g} (same instant): {frac*100:.1f}% of time")
    d_alone = sum(1 for v in d_env if v is not None and v >= args.d_hi)
    p_alone = sum(1 for v in p_env if v is not None and v <= args.p_lo)
    n = sum(1 for v in d_env if v is not None)
    print(f"  (for reference) D>={args.d_hi:g} alone: {100*d_alone/n:.1f}%   "
          f"P<={args.p_lo:g} alone: {100*p_alone/n:.1f}%")

    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.7))
    for a, b in spans:
        ax.axvspan(a, b, color=SHADE, alpha=0.16, lw=0, zorder=1)
    xs = [t[i] for i in range(len(t)) if d_env[i] is not None]
    dv = [d_env[i] for i in range(len(t)) if d_env[i] is not None]
    pv = [p_env[i] for i in range(len(t)) if p_env[i] is not None]
    ax.fill_between(xs, pv, dv, color="#CFCFCF", alpha=0.5, lw=0, zorder=2)
    ax.plot(xs, dv, color=SERIES["D0"][0], lw=1.3, zorder=3, label="max decode GPU")
    ax.plot(xs, pv, color=SERIES["P0"][0], lw=1.3, ls="--", zorder=3, label="min prefill GPU")
    ax.axhline(args.d_hi, color=MUTED, lw=0.7, ls=(0, (5, 3)))
    ax.axhline(args.p_lo, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    ax.set_ylabel("occupancy")
    ax.set_xlabel("time (s)")
    ax.set_ylim(0, 1.03)
    ax.set_title(f"Decode saturated (≥{args.d_hi:g}) while prefill idles (≤{args.p_lo:g})"
                f"  — {frac*100:.0f}% of the run (shaded)")
    ax.legend(ncol=2, loc="lower right", handlelength=1.6)
    style_axes(ax)

    fig.tight_layout(pad=0.6)
    out = args.out or (os.path.splitext(args.csv)[0] + "_role")
    stem = out[:-4] if out.endswith((".png", ".pdf")) else out
    savefig(fig, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--hi", type=float, default=0.8, help="saturation threshold")
    ap.add_argument("--lo", type=float, default=0.5, help="headroom threshold")
    ap.add_argument("--out", default=None)
    ap.add_argument("--single", action="store_true",
                    help="render only panel (b) (max/min envelope + opportunity) -- "
                    "drops the cluttered per-GPU occupancy panel (a)")
    ap.add_argument("--role", action="store_true",
                    help="role-conditioned mode: shade instants where a DECODE GPU is "
                    "saturated (>=--d-hi) AND a PREFILL GPU has deep headroom (<=--p-lo) "
                    "at the SAME instant -- the exact claim 'D saturates while P idles'. "
                    "Roles inferred from labels starting with P/D.")
    ap.add_argument("--d-hi", type=float, default=0.85, help="decode saturation threshold (--role mode)")
    ap.add_argument("--p-lo", type=float, default=0.30, help="prefill headroom threshold (--role mode)")
    args = ap.parse_args()

    data, labels = load(args.csv)
    t = data["t"]
    if not t:
        raise SystemExit(f"no samples in {args.csv}")

    if args.role:
        main_role(args, data, labels, t)
        return

    mx, mn, opp = [], [], []
    for i in range(len(t)):
        vals = [data[l][i] for l in labels if data[l][i] is not None]
        if len(vals) >= 2:
            mx.append(max(vals)); mn.append(min(vals))
            opp.append(1 if (max(vals) >= args.hi and min(vals) <= args.lo) else 0)
        else:
            mx.append(None); mn.append(None); opp.append(0)
    opp_frac = sum(opp) / len(opp) if opp else 0.0
    spans = opp_spans(t, opp)

    print(f"\n=== 2P2D KV imbalance ({os.path.basename(args.csv)}) ===")
    print(f"  samples={len(t)}  duration={t[-1]:.0f}s")
    for l in labels:
        vv = [x for x in data[l] if x is not None]
        if vv:
            print(f"  {l}: mean={sum(vv)/len(vv):.2f} max={max(vv):.2f} min={min(vv):.2f}")
    print(f"  OPPORTUNITY (some GPU>={args.hi} AND some GPU<={args.lo}): {opp_frac*100:.1f}% of time")

    use_paper_style()

    def shade(ax):
        for a, b in spans:
            ax.axvspan(a, b, color=SHADE, alpha=0.16, lw=0, zorder=1)

    def draw_envelope(ax, title):
        shade(ax)
        xs = [t[i] for i in range(len(t)) if mx[i] is not None]
        mxv = [mx[i] for i in range(len(t)) if mx[i] is not None]
        mnv = [mn[i] for i in range(len(t)) if mn[i] is not None]
        ax.fill_between(xs, mnv, mxv, color="#CFCFCF", alpha=0.5, lw=0, zorder=2)
        ax.plot(xs, mxv, color=INK, lw=1.1, zorder=3, label="max GPU")
        ax.plot(xs, mnv, color=INK, lw=1.1, ls="--", zorder=3, label="min GPU")
        ax.axhline(args.hi, color=MUTED, lw=0.7, ls=(0, (5, 3)))
        ax.axhline(args.lo, color=MUTED, lw=0.7, ls=(0, (1, 2)))
        ax.set_ylabel("occupancy\nenvelope")
        ax.set_xlabel("time (s)")
        ax.set_ylim(0, 1.03)
        ax.set_title(title)
        ax.legend(ncol=2, loc="lower right", handlelength=1.6)
        style_axes(ax)

    if args.single:
        # panel (b) only -- the caption/number belongs in the paper's Fig. N text,
        # so keep the in-figure title terse.
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.6))
        draw_envelope(ax, f"Opportunity: max GPU ≥ {args.hi:g} and min GPU ≤ {args.lo:g}  "
                          f"— {opp_frac*100:.0f}% of the run (shaded)")
        fig.tight_layout(pad=0.6)
        out = args.out or (os.path.splitext(args.csv)[0] + "_imbalance_single")
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 3.9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1.7]})
        # (a) per-GPU occupancy
        shade(ax1)
        for l in labels:
            col, ls = SERIES.get(l, ("#888", "-"))
            xs = [t[i] for i in range(len(t)) if data[l][i] is not None]
            ys = [data[l][i] for i in range(len(t)) if data[l][i] is not None]
            ax1.plot(xs, ys, ls=ls, color=col, lw=1.0, label=l, zorder=3)
        ax1.axhline(args.hi, color=MUTED, lw=0.7, ls=(0, (5, 3)))
        ax1.axhline(args.lo, color=MUTED, lw=0.7, ls=(0, (1, 2)))
        ax1.text(t[-1], args.hi + 0.01, f"saturated {args.hi:g}", color=MUTED, fontsize=6.5,
                 va="bottom", ha="right")
        ax1.text(t[-1], args.lo - 0.01, f"headroom {args.lo:g}", color=MUTED, fontsize=6.5,
                 va="top", ha="right")
        ax1.set_ylabel("KV pool usage")
        ax1.set_ylim(0, 1.03)
        ax1.set_title("(a) Per-GPU KV occupancy (2P2D)")
        ax1.legend(ncol=4, loc="upper center", handlelength=1.6, columnspacing=1.2,
                   borderaxespad=0.2)
        style_axes(ax1)

        # (b) max/min envelope -- shading is exactly verifiable here
        draw_envelope(ax2, f"(b) Opportunity: max GPU ≥ {args.hi:g} and min GPU ≤ {args.lo:g}  "
                          f"— {opp_frac*100:.0f}% of the run (shaded)")
        fig.tight_layout(pad=0.6)
        out = args.out or (os.path.splitext(args.csv)[0] + "_imbalance")

    stem = out[:-4] if out.endswith((".png", ".pdf")) else out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

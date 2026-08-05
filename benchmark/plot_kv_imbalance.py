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
# Role-agnostic palette for --rename. The claim this figure carries is "at any instant
# some GPU is out of room while another has plenty", which is a statement about four
# GPUs, not about prefill versus decode -- and deliberately so: the DIRECTION of the P/D
# imbalance is not fixed. Under a 20k prefill pool at C=24 the decode pools ran hotter
# (0.75 vs 0.44); under a 60k pool at C=16 the prefill pools do (0.89 vs 0.06). A figure
# that names roles commits the paper to one of those, and the other configuration then
# reads as a contradiction. Naming GPUs commits it to neither, and the mechanism does not
# depend on the answer: _select_pool() reads live occupancy and places into whatever is
# idle at that instant.
GPU_SERIES = ["#0072B2", "#E69F00", "#56B4E9", "#D55E00"]
GPU_STYLES = ["-", "-", "--", "--"]
SHADE = "#9C7A3C"   # muted amber/brown wash for opportunity spans
INK, MUTED = PALETTE["ink"], PALETTE["muted"]
MAX_COL = MIN_COL = INK  # both black; separated by linestyle only (solid vs dashed)


def load(csv_path):
    rows = [ln.rstrip("\n") for ln in open(csv_path) if not ln.startswith("#")]
    reader = csv.DictReader(rows)
    labels = [c[:-4] for c in reader.fieldnames if c.endswith("_use")]
    data = {"t": []} | {l: [] for l in labels}
    caps = {l: [] for l in labels}
    for r in reader:
        try:
            data["t"].append(float(r["t_s"]))
        except (ValueError, KeyError):
            continue
        for l in labels:
            v = r.get(f"{l}_use", "")
            data[l].append(float(v) if v not in ("", None) else None)
            c = r.get(f"{l}_cap_tok", "")
            if c not in ("", None):
                caps[l].append(float(c))
    # Pool capacity, so headroom can also be stated in GB. The fraction alone cannot:
    # a prefill capped at 60k tokens holds 7.3 GB and a decode 22.6 GB, so equal-looking
    # occupancies on the two sides are very different amounts of free memory.
    cap = {l: (sorted(v)[len(v) // 2] if v else None) for l, v in caps.items()}
    return data, labels, cap


def rename(data, labels, cap, spec):
    """Apply --rename P0=GPU0,D0=GPU1,... and return the reordered label list."""
    mapping = dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)
    missing = [l for l in labels if l not in mapping]
    if missing:
        raise SystemExit(f"--rename does not cover {missing}; got {sorted(mapping)}")
    for old, new in mapping.items():
        data[new] = data.pop(old)
        cap[new] = cap.pop(old)
    return sorted(mapping.values())


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
    ap.add_argument("--rename", default="",
                    help="relabel series role-agnostically, e.g. "
                         "'P0=GPU0,D0=GPU1,P1=GPU2,D1=GPU3' (PD_LAYOUT=b)")
    ap.add_argument("--kib-per-token", type=float, default=128.0,
                    help="KV bytes/token for the GB figures (Llama-3.1-8B=128)")
    args = ap.parse_args()

    data, labels, cap = load(args.csv)
    if args.rename:
        labels = rename(data, labels, cap, args.rename)
        for i, l in enumerate(labels):
            SERIES[l] = (GPU_SERIES[i % len(GPU_SERIES)], GPU_STYLES[i % len(GPU_STYLES)])
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
    # How much memory the opportunity is worth: free capacity on the slack GPUs at the
    # instants a saturated one coexists with them. A percentage of time says the chance
    # arises; this says whether it is worth taking.
    if all(cap.get(l) for l in labels):
        tot, cnt = 0.0, 0
        for i in range(len(t)):
            if not opp[i]:
                continue
            g = sum((1.0 - data[l][i]) * cap[l] for l in labels
                    if data[l][i] is not None and data[l][i] <= args.lo)
            tot += g * args.kib_per_token / (1024 * 1024)
            cnt += 1
        if cnt:
            print(f"  STRANDED while the opportunity holds: {tot / cnt:.1f} GB on average")
        for l in labels:
            vv = [x for x in data[l] if x is not None]
            print(f"    {l}: pool {cap[l] * args.kib_per_token / (1024*1024):.2f} GB, "
                  f"free {sum((1 - x) for x in vv) / len(vv) * cap[l] * args.kib_per_token / (1024*1024):.2f} GB avg")

    use_paper_style()

    def shade(ax):
        for a, b in spans:
            # zorder above the envelope fill (2) so opportunity spans read as a clear
            # darker overlay, not competing with the lighter gray fill-between.
            ax.axvspan(a, b, color=SHADE, alpha=0.34, lw=0, zorder=2.5)

    def draw_envelope(ax, title):
        xs = [t[i] for i in range(len(t)) if mx[i] is not None]
        mxv = [mx[i] for i in range(len(t)) if mx[i] is not None]
        mnv = [mn[i] for i in range(len(t)) if mn[i] is not None]
        # lighter, recessive envelope fill so the (darker, higher-zorder) opportunity
        # shading from shade() stands out clearly on top of it rather than blending in.
        ax.fill_between(xs, mnv, mxv, color="#E7E7E7", alpha=0.7, lw=0, zorder=1)
        shade(ax)
        ax.plot(xs, mxv, color=MAX_COL, lw=1.2, zorder=3, label="max GPU")
        ax.plot(xs, mnv, color=MIN_COL, lw=1.2, ls="--", zorder=3, label="min GPU")
        ax.axhline(args.hi, color=MUTED, lw=0.7, ls=(0, (5, 3)))
        ax.axhline(args.lo, color=MUTED, lw=0.7, ls=(0, (1, 2)))
        ax.set_ylabel("KV pool occupancy")
        ax.set_xlabel("time (s)")
        ax.set_ylim(0, 1.03)
        ax.set_title(title)
        ax.legend(ncol=2, loc="lower right", handlelength=1.6, frameon=True,
                  fancybox=False, edgecolor=MUTED, facecolor="white")
        style_axes(ax)

    if args.single:
        # panel (b) only, plus a thin binary "opportunity" strip below the envelope
        # so shaded/not-shaded reads as an on/off block instead of a translucent wash
        # that has to be cross-referenced against the wiggly lines by eye.
        fig, (ax, axs) = plt.subplots(2, 1, figsize=(7.0, 3.15), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 0.7]})
        draw_envelope(ax, f"Opportunity: max GPU ≥ {args.hi:g} and min GPU ≤ {args.lo:g}")

        # binary opportunity strip: solid block = condition true, blank = false
        for a, b in spans:
            axs.axvspan(a, b, color=SHADE, alpha=0.9, lw=0)
        axs.set_xlim(ax.get_xlim())
        axs.set_ylim(0, 1)
        axs.set_yticks([])
        axs.set_ylabel("opportunity", fontsize=7, rotation=0, ha="right", va="center")
        axs.set_xlabel("time (s)")
        for s in ("top", "right", "left"):
            axs.spines[s].set_visible(False)
        axs.spines["bottom"].set_color(MUTED)
        axs.tick_params(colors=MUTED, length=2.5, width=0.5)
        ax.set_xlabel("")  # xlabel lives on the shared bottom strip axis instead

        fig.tight_layout(pad=0.6)
        fig.subplots_adjust(hspace=0.12)
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
        ax1.set_title("(a) Per-GPU KV occupancy")
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

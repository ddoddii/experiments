#!/usr/bin/env python3
"""
Fig. 1 (simple form): KV pool occupancy over time, split by ROLE.
  top    = Decode Instance   (D* GPUs)
  bottom = Prefill Instance  (P* GPUs)

Deliberately plain -- one filled band per panel, in the style of the classic
"KV memory utilization vs iteration" motivation figures. Per panel:
  dark fill  : mean occupancy across that role's GPUs
  light fill : min-max spread across that role's GPUs (omitted with --no-band)
  dotted line: that role's time-average, labelled

The message is read off the two panels directly: decode sits near saturation
while prefill runs far below it, so at any instant there is HBM in the prefill
GPUs that no one is using -- the victim-cache opportunity. --headroom makes that
explicit by hatching the unused capacity above the prefill curve.

Reads the same CSV as plot_kv_imbalance.py (kv_occupancy_timeseries.py output):
columns t_s, <label>_use, ... with a "# roles: P0:P ..." comment line.

사용:
  python benchmark/plot_kv_occupancy_2panel.py --csv results/kv_ts/2p2d_p20k.csv \
      --out results/kv_ts/fig1_occupancy_2panel
  # opportunity version (hatched idle HBM in the prefill panel):
  python benchmark/plot_kv_occupancy_2panel.py --csv results/kv_ts/2p2d_p20k.csv \
      --headroom --out results/kv_ts/fig1_occupancy_2panel_headroom
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, savefig

# grayscale by default -- matches the reference figure and prints cleanly.
GRAY = {"fill": "#3B3B3B", "band": "#C9C9C9", "line": "#000000"}
# --color: role-coded (decode amber, prefill blue) to match the other figures.
COLOR = {"D": {"fill": "#E69F00", "band": "#F6DFB0", "line": "#8A6100"},
         "P": {"fill": "#0072B2", "band": "#BBD9EC", "line": "#00456B"}}
MUTED = PALETTE["muted"]


def load(csv_path):
    """-> (t, {label: [use]}) ; ignores the '# roles:' comment line."""
    rows = [ln for ln in open(csv_path) if not ln.startswith("#")]
    reader = csv.DictReader(rows)
    labels = [c[:-4] for c in reader.fieldnames if c.endswith("_use")]
    t, data = [], {l: [] for l in labels}
    for r in reader:
        try:
            t.append(float(r["t_s"]))
        except (ValueError, KeyError):
            continue
        for l in labels:
            v = r.get(f"{l}_use", "")
            data[l].append(float(v) if v not in ("", None) else None)
    return t, data, labels


def decimate(t, series, max_points):
    """Stride-sample to max_points. The BFCL capture is ~1.6M rows; drawing all of
    them makes a black smear and a 20MB PDF. Stride (not averaging) so the peaks
    stay peaks."""
    n = len(t)
    if max_points <= 0 or n <= max_points:
        return t, series
    step = n // max_points + 1
    return t[::step], {k: v[::step] for k, v in series.items()}


def role_envelope(t, data, role_labels):
    """-> (xs, mean, lo, hi) over the GPUs of one role, skipping incomplete samples."""
    xs, mean, lo, hi = [], [], [], []
    for i in range(len(t)):
        vals = [data[l][i] for l in role_labels if data[l][i] is not None]
        if not vals:
            continue
        xs.append(t[i])
        mean.append(100.0 * sum(vals) / len(vals))
        lo.append(100.0 * min(vals))
        hi.append(100.0 * max(vals))
    return xs, mean, lo, hi


def panel(ax, xs, mean, lo, hi, title, cs, args, headroom=False):
    if not args.no_band:
        ax.fill_between(xs, lo, hi, color=cs["band"], lw=0, zorder=2,
                        label="min–max across GPUs")
    ax.fill_between(xs, 0, mean, color=cs["fill"], lw=0, zorder=3, label="mean occupancy")
    ax.plot(xs, mean, color=cs["line"], lw=0.7, zorder=4)

    avg = sum(mean) / len(mean) if mean else 0.0
    ax.axhline(avg, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=5)
    ax.text(xs[-1] if xs else 0, avg, f" avg {avg:.0f}% ", ha="right", va="bottom",
            fontsize=6.5, color=MUTED, zorder=6)

    if headroom:
        # unused capacity above the busiest GPU of this role = the reclaimable HBM.
        # Kept deliberately faint: it is context for the dark occupancy band, not a
        # competing series -- a dense hatch here reads as the subject of the panel.
        matplotlib.rcParams["hatch.linewidth"] = 0.35
        ax.fill_between(xs, hi, 100, facecolor="none", edgecolor="#C4C4C4", lw=0.0,
                        hatch="/", zorder=1)
        ax.text(0.985, 0.93, "idle HBM (reclaimable)", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=MUTED,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    ax.set_ylim(0, 100)
    ax.set_xlim(min(xs) if xs else 0, max(xs) if xs else 1)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, fontsize=8.5, pad=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, length=2.5, width=0.5)
    ax.grid(False)
    return avg


def both_roles(t, data, p_labels, d_labels):
    """-> (xs, decode_mean, prefill_mean) on the samples where BOTH roles report,
    so the two curves are directly comparable instant by instant."""
    xs, dm, pm = [], [], []
    for i in range(len(t)):
        dv = [data[l][i] for l in d_labels if data[l][i] is not None]
        pv = [data[l][i] for l in p_labels if data[l][i] is not None]
        if not dv or not pv:
            continue
        xs.append(t[i])
        dm.append(100.0 * sum(dv) / len(dv))
        pm.append(100.0 * sum(pv) / len(pv))
    return xs, dm, pm


def render_overlay(args, t, data, p_labels, d_labels):
    """Single-axes version (saves ~half the vertical space): decode drawn as the
    LIGHT area behind, prefill as the DARK area in front. Because decode sits above
    prefill nearly everywhere, the light band left visible between the two curves IS
    the imbalance -- the HBM that is busy on decode and idle on prefill at the same
    instant. Same idiom as the classic occupied-vs-demanded utilization figure."""
    xs, dm, pm = both_roles(t, data, p_labels, d_labels)
    if not xs:
        raise SystemExit("no samples where both roles report")

    d_avg = sum(dm) / len(dm)
    p_avg = sum(pm) / len(pm)
    gap = [d - p for d, p in zip(dm, pm)]
    g_avg = sum(gap) / len(gap)
    frac_d_above = 100.0 * sum(1 for g in gap if g > 0) / len(gap)

    c_dec = COLOR["D"]["fill"] if args.color else "#BFBFBF"
    c_pre = COLOR["P"]["fill"] if args.color else "#333333"

    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(args.width, args.height))
    ax.fill_between(xs, 0, dm, color=c_dec, lw=0, zorder=2, label="Decode instance")
    ax.fill_between(xs, 0, pm, color=c_pre, lw=0, zorder=3, label="Prefill instance")
    ax.plot(xs, dm, color="#000000", lw=0.6, zorder=4)
    ax.plot(xs, pm, color="#000000", lw=0.6, zorder=5)

    # the two time-averages, labelled on the right edge -- the whole claim in 2 numbers
    for avg, txt, va in ((d_avg, f"decode avg {d_avg:.0f}%", "bottom"),
                         (p_avg, f"prefill avg {p_avg:.0f}%", "top")):
        ax.axhline(avg, color="#000000", lw=0.7, ls=(0, (4, 2.5)), zorder=6)
        ax.text(xs[-1], avg, f"{txt} ", ha="right", va=va, fontsize=6.4, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    if not args.no_gap_label:
        # name the light band once -- without it the reader has to infer that the gap
        # between the two areas is the point of the figure
        ax.annotate("", xy=(xs[0] + 0.16 * (xs[-1] - xs[0]), d_avg),
                    xytext=(xs[0] + 0.16 * (xs[-1] - xs[0]), p_avg),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="#000000"), zorder=7)
        ax.text(xs[0] + 0.195 * (xs[-1] - xs[0]), (d_avg + p_avg) / 2,
                f"{g_avg:.0f} pts idle on prefill\nwhile decode is busy",
                ha="left", va="center", fontsize=6.4, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    ax.set_ylim(0, 100)
    ax.set_xlim(min(xs), max(xs))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("KV memory\nutilization (%)")
    ax.set_xlabel("time (s)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, length=2.5, width=0.5)
    ax.grid(False)
    ax.legend(loc="lower left", ncol=2, frameon=True, fancybox=False, facecolor="white",
              edgecolor=MUTED, fontsize=6.4, handlelength=1.4, borderpad=0.3,
              columnspacing=1.0).set_zorder(8)

    print(f"\n=== role occupancy, overlay ({os.path.basename(args.csv)}) ===")
    print(f"  decode  {d_labels}  time-avg {d_avg:.1f}%")
    print(f"  prefill {p_labels}  time-avg {p_avg:.1f}%")
    print(f"  mean gap = {g_avg:.1f} points; decode above prefill "
          f"{frac_d_above:.0f}% of samples")
    print(f"  prefill leaves {100 - p_avg:.0f}% of its KV pool unused on average")

    fig.tight_layout(pad=0.4)
    out = args.out or (os.path.splitext(args.csv)[0] + "_overlay")
    savefig(fig, out[:-4] if out.endswith((".png", ".pdf")) else out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--color", action="store_true",
                    help="role-coded colors (decode amber / prefill blue) instead of "
                         "the default grayscale")
    ap.add_argument("--no-band", action="store_true",
                    help="drop the min-max band; plot only the role mean")
    ap.add_argument("--headroom", action="store_true",
                    help="hatch the unused capacity above the prefill curve -- the "
                         "idle HBM the victim cache reclaims")
    ap.add_argument("--max-points", type=int, default=4000,
                    help="decimate to at most this many samples per panel (0 = all)")
    ap.add_argument("--tmax", type=float, default=0.0, help="crop to the first N seconds")
    ap.add_argument("--overlay", action="store_true",
                    help="single axes instead of two stacked panels: decode as the "
                         "light area behind, prefill as the dark area in front, so the "
                         "visible light band is the imbalance. Half the page height.")
    ap.add_argument("--no-gap-label", action="store_true",
                    help="--overlay: drop the arrow/label naming the decode-prefill gap")
    ap.add_argument("--width", type=float, default=6.4, help="figure width (in)")
    ap.add_argument("--height", type=float, default=1.85,
                    help="figure height (in); only used by --overlay")
    args = ap.parse_args()

    t, data, labels = load(args.csv)
    if not t:
        raise SystemExit(f"no samples in {args.csv}")
    if args.tmax > 0:
        keep = [i for i, x in enumerate(t) if x <= args.tmax]
        t = [t[i] for i in keep]
        data = {k: [v[i] for i in keep] for k, v in data.items()}
    t, data = decimate(t, data, args.max_points)

    p_labels = [l for l in labels if l.upper().startswith("P")]
    d_labels = [l for l in labels if l.upper().startswith("D")]
    if not p_labels or not d_labels:
        raise SystemExit(f"need both P* and D* columns; found {labels}")

    if args.overlay:
        render_overlay(args, t, data, p_labels, d_labels)
        return

    dx, dmean, dlo, dhi = role_envelope(t, data, d_labels)
    px, pmean, plo, phi = role_envelope(t, data, p_labels)

    use_paper_style()
    fig, (axD, axP) = plt.subplots(2, 1, figsize=(6.4, 3.4), sharex=True)
    csD = COLOR["D"] if args.color else GRAY
    csP = COLOR["P"] if args.color else GRAY
    d_avg = panel(axD, dx, dmean, dlo, dhi, "Decode Instance", csD, args)
    p_avg = panel(axP, px, pmean, plo, phi, "Prefill Instance", csP, args,
                  headroom=args.headroom)

    axD.set_ylabel("KV memory\nutilization (%)")
    axP.set_ylabel("KV memory\nutilization (%)")
    axP.set_xlabel("time (s)")
    handles, hlabels = axD.get_legend_handles_labels()
    fig.legend(handles, hlabels, loc="upper center", ncol=len(hlabels), frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, 1.02), columnspacing=1.6)

    print(f"\n=== role occupancy ({os.path.basename(args.csv)}) ===")
    print(f"  decode  GPUs={d_labels}  time-avg {d_avg:.1f}%")
    print(f"  prefill GPUs={p_labels}  time-avg {p_avg:.1f}%")
    print(f"  gap = {d_avg - p_avg:.1f} points; prefill leaves "
          f"{100 - p_avg:.0f}% of its KV pool unused on average")
    print(f"  samples plotted: {len(dx)} (of {len(t)} after crop)")

    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.32)
    out = args.out or (os.path.splitext(args.csv)[0] + "_2panel")
    stem = out[:-4] if out.endswith((".png", ".pdf")) else out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

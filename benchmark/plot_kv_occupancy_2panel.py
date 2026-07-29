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

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

The message is read off the two panels directly: one role sits near saturation while
the other runs far below it, so at any instant there is HBM that no one is using --
the victim-cache opportunity. --headroom hatches that unused capacity.

WHICH role is saturated depends on the gauge, and the difference is not cosmetic:
  token_usage       counts only in-flight KV (num_used = max_total - available -
                    evictable), so a prefill pool 80% full of retained prefixes reads
                    ~1%. On this gauge DECODE looks busier.
  cache_occupancy   counts retained prefixes too. On this gauge PREFILL is the busy one
                    (82% vs 11% measured), which is the resource the paper argues about.
An earlier version of this figure was drawn on token_usage and labelled "KV memory
utilization", which inverted the claim. The axis now names the gauge.

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


def read_metric(csv_path):
    """Which gauge the '_use' columns actually hold, from the sampler's '# metric:' line.

    This matters more than it looks. The same column name has carried two different
    quantities: sglang:token_usage, which SUBTRACTS the radix cache (num_used =
    max_total - available - evictable), and sglang:cache_occupancy, which includes it.
    On a prefill node the two differ by ~90x, and the motivation figure was drawn on the
    first without saying so -- reading it as pool occupancy inverted the paper's claim.
    The plotter used to discard this line with the rest of the comments; now it names the
    gauge on the axis so a figure cannot be mistaken for the other measurement.
    """
    # Scan the whole file, not just a header block. The column header is line 1 and the
    # '# metric:' lines come after it, so stopping at the first non-comment line found
    # nothing and reported UNRECORDED on files that do record the gauge. The sampler also
    # re-detects every tick and appends a new line whenever the answer changes (a gauge is
    # absent until first set, so idle nodes start as 'unreachable'), so every line counts.
    names = set()
    for ln in open(csv_path):
        if ln.startswith("# metric:"):
            names.update(p.split(":", 1)[1] for p in ln.split()[2:] if ":" in p)
    names.discard("unreachable")
    return "/".join(sorted(names)) if names else None


def use_axis_label(metric):
    """Axis label that names the gauge, so the two cannot be confused on sight."""
    if metric == "token_usage":
        return "In-flight KV\n(token_usage, %)"
    if metric == "cache_occupancy":
        return "KV pool occupancy\n(cache_occupancy, %)"
    return f"KV memory\nutilization (%)" if not metric else f"KV ({metric}, %)"


def load_pressure(csv_path):
    """-> (t, {label: [token_usage]}). The WORKING SET: KV of requests in the current
    batch, which attention reads on every forward step and which is freed automatically
    when the request finishes. cache_occupancy minus this is the CACHE: finished turns
    nobody is reading, held on the bet that the session returns, freed only by LRU."""
    rows = [ln for ln in open(csv_path) if not ln.startswith("#")]
    reader = csv.DictReader(rows)
    labels = [c[:-9] for c in reader.fieldnames if c.endswith("_pressure")]
    t, data = [], {l: [] for l in labels}
    for r in reader:
        try:
            t.append(float(r["t_s"]))
        except (ValueError, KeyError):
            continue
        for l in labels:
            v = r.get(f"{l}_pressure", "")
            data[l].append(float(v) if v not in ("", None) else None)
    return t, data, labels


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


def both_roles(t, data, p_labels, d_labels, scale=None):
    """-> (xs, decode_sum, prefill_sum) on the samples where BOTH roles report.

    scale=None gives the role MEAN as a percentage (the occupancy view). Passing a
    per-label multiplier instead gives the role SUM in GB: occupancy fractions are
    averaged because they are rates, but bytes are added because a cluster holds the
    sum of what its GPUs hold.
    """
    xs, dm, pm = [], [], []
    for i in range(len(t)):
        dv = [(l, data[l][i]) for l in d_labels if data[l][i] is not None]
        pv = [(l, data[l][i]) for l in p_labels if data[l][i] is not None]
        if not dv or not pv:
            continue
        xs.append(t[i])
        if scale is None:
            dm.append(100.0 * sum(v for _, v in dv) / len(dv))
            pm.append(100.0 * sum(v for _, v in pv) / len(pv))
        else:
            dm.append(sum(v * scale.get(l, 0.0) for l, v in dv))
            pm.append(sum(v * scale.get(l, 0.0) for l, v in pv))
    return xs, dm, pm


def gb_scale(csv_path, labels, kib_per_token):
    """label -> GB per unit of occupancy fraction, from that node's pool capacity."""
    rows = [r for r in csv.DictReader(
        (l for l in open(csv_path) if not l.startswith("#")))]
    out = {}
    for lab in labels:
        caps = [float(r[f"{lab}_cap_tok"]) for r in rows if r.get(f"{lab}_cap_tok")]
        if caps:
            out[lab] = (sum(caps) / len(caps)) * kib_per_token * 1024 / 1024 ** 3
    return out


def render_decomposed(args, t, data, pdata, p_labels, d_labels):
    """Each role's curve split into working set and cache.

    The plain overlay draws prefill at 82% and decode at 11% as if they were the same
    quantity. They are not: 98.9% of the prefill curve is cache and 98.8% of the decode
    curve is working set, so the two lines are nearly disjoint measurements sharing an
    axis. Splitting them makes the figure state that instead of leaving it to the caption
    -- and it explains the shapes, since a cache ramps to an eviction equilibrium and
    holds while a working set breathes with the number of live sequences and never
    accumulates."""
    xs, dm, pm = both_roles(t, data, p_labels, d_labels)
    xw, dw, pw = both_roles(t, pdata, p_labels, d_labels)
    n = min(len(xs), len(xw))
    xs, dm, pm, dw, pw = xs[:n], dm[:n], pm[:n], dw[:n], pw[:n]
    if not xs:
        raise SystemExit("no samples where both roles report both gauges")

    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(args.width, args.height))
    C = {"P": ("#0072B2", "#BBD9EC"), "D": ("#E69F00", "#F6DFB0")}

    order = (("Prefill", pm, pw, C["P"]), ("Decode", dm, dw, C["D"]))
    if sum(pm) / len(pm) < sum(dm) / len(dm):          # busier role behind
        order = order[::-1]
    for z, (name, tot, ws, (c_ws, c_cache)) in enumerate(order):
        ax.fill_between(xs, ws, tot, color=c_cache, lw=0, zorder=2 + 2 * z,
                        label=f"{name}: cache (finished turns, LRU-evictable)")
        ax.fill_between(xs, 0, ws, color=c_ws, lw=0, zorder=3 + 2 * z,
                        label=f"{name}: working set (in the current batch)")
        ax.plot(xs, tot, color="#000000", lw=0.6, zorder=6)

    for name, tot, ws in (("prefill", pm, pw), ("decode", dm, dw)):
        a_t, a_w = sum(tot) / len(tot), sum(ws) / len(ws)
        print(f"  {name:8s} curve {a_t:5.1f}%  =  working set {a_w:5.1f}% "
              f"({100 * a_w / a_t if a_t else 0:4.1f}% of it)  +  cache "
              f"{a_t - a_w:5.1f}% ({100 * (a_t - a_w) / a_t if a_t else 0:4.1f}%)")

    if args.absolute:
        ax.set_ylim(0, max(max(dm), max(pm)) * 1.12)
        ax.set_ylabel("KV actually held (GB)\nreserved-but-empty excluded")
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel(use_axis_label(args.metric))
    ax.set_xlim(min(xs), max(xs))
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper left", frameon=False, fontsize=5.6, labelspacing=0.25,
              handlelength=1.3, borderaxespad=0.3, ncol=2, columnspacing=1.0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("bottom", "left"):
        ax.spines[s_].set_color(MUTED); ax.spines[s_].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, length=2.5, width=0.5)
    ax.grid(False)
    fig.tight_layout(pad=0.4)
    savefig(fig, args.out or "fig_kv_decomposed")


def render_overlay(args, t, data, p_labels, d_labels):
    """Single-axes version (saves ~half the vertical space): the BUSIER role is drawn as
    the area behind, the idler in front, so the band left visible between the two curves
    IS the imbalance -- HBM busy on one role and idle on the other at the same instant.

    Which role is busier is decided by the data, not assumed. It flips with the gauge:
    on token_usage decode reads higher, on cache_occupancy prefill does. See
    read_metric()."""
    scale = (gb_scale(args.csv, p_labels + d_labels, args.kib_per_token)
             if args.absolute else None)
    xs, dm, pm = both_roles(t, data, p_labels, d_labels, scale)
    if not xs:
        raise SystemExit("no samples where both roles report")

    d_avg = sum(dm) / len(dm)
    p_avg = sum(pm) / len(pm)
    gap = [d - p for d, p in zip(dm, pm)]
    g_avg = sum(gap) / len(gap)
    frac_d_above = 100.0 * sum(1 for g in gap if g > 0) / len(gap)

    if args.export_json:
        import json
        # Decimated hard: a native PowerPoint chart embeds its own worksheet, and ~950
        # points x 2 series makes the deck slow to open and painful to edit by hand.
        # Stride, never average -- averaging would round off the saturation plateau that
        # is the point of the figure.
        step = max(1, len(xs) // 220)
        json.dump({"metric": args.metric, "csv": os.path.basename(args.csv),
                   "t": xs[::step], "decode": dm[::step], "prefill": pm[::step],
                   "decode_avg": d_avg, "prefill_avg": p_avg, "gap_avg": g_avg,
                   "frac_decode_above": frac_d_above, "n_full": len(xs)},
                  open(args.export_json, "w"), indent=1)
        print(f"[export] {args.export_json}  ({len(xs[::step])} of {len(xs)} samples)")

    c_dec = COLOR["D"]["fill"] if args.color else "#BFBFBF"
    c_pre = COLOR["P"]["fill"] if args.color else "#333333"

    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(args.width, args.height))
    # thinner outlines at column width -- 0.6pt over a dense series turns the whole
    # band black once the figure is scaled down to ~3.5in
    olw = 0.6 if args.width >= 4.5 else 0.45
    # Both areas start at 0 and overlap, so the BUSIER role has to be drawn behind or it
    # is hidden completely and the visible band stops being the gap. Which role that is
    # depends on the gauge: on token_usage decode reads higher (it holds in-flight KV,
    # while prefill's pool is mostly retained prefixes that token_usage subtracts); on
    # cache_occupancy prefill reads higher. This used to be fixed at decode-behind, so
    # re-running it on cache_occupancy data buried the decode curve under a solid prefill
    # area and still printed "idle on prefill while decode is busy".
    busy_first = ((dm, c_dec, "Decode instance"), (pm, c_pre, "Prefill instance")) \
        if d_avg >= p_avg else ((pm, c_pre, "Prefill instance"), (dm, c_dec, "Decode instance"))
    for z, (series, colour, lab) in enumerate(busy_first):
        ax.fill_between(xs, 0, series, color=colour, lw=0, zorder=2 + z, label=lab)
    ax.plot(xs, dm, color="#000000", lw=olw, zorder=4)
    ax.plot(xs, pm, color="#000000", lw=olw, zorder=5)

    # the two time-averages, labelled on the right edge -- the whole claim in 2 numbers.
    # At single-column width these crowd the plot, so --no-avg drops them and the
    # numbers go in the caption instead (they are printed to stdout either way).
    if not args.no_avg:
        _u = " GB" if args.absolute else "%"
        for avg, txt, va in ((d_avg, f"decode avg {d_avg:.1f}{_u}", "bottom"),
                             (p_avg, f"prefill avg {p_avg:.1f}{_u}", "top")):
            ax.axhline(avg, color="#000000", lw=0.7, ls=(0, (4, 2.5)), zorder=6)
            ax.text(xs[-1], avg, f"{txt} ", ha="right", va=va, fontsize=6.4, zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    if not args.no_gap_label:
        # name the light band once -- without it the reader has to infer that the gap
        # between the two areas is the point of the figure
        ax.annotate("", xy=(xs[0] + 0.16 * (xs[-1] - xs[0]), d_avg),
                    xytext=(xs[0] + 0.16 * (xs[-1] - xs[0]), p_avg),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="#000000"), zorder=7)
        # Direction read off the data, never assumed. With the sign hardcoded this label
        # rendered as "-71 pts idle on prefill while decode is busy" on a run where
        # prefill was the busy one -- wrong sign and wrong roles in the same sentence.
        idle, busy = ("prefill", "decode") if g_avg > 0 else ("decode", "prefill")
        ax.text(xs[0] + 0.195 * (xs[-1] - xs[0]), (d_avg + p_avg) / 2,
                (f"{abs(g_avg):.1f} GB more KV on {busy}\nthan on {idle}"
                 if args.absolute else
                 f"{abs(g_avg):.0f} pts idle on {idle}\nwhile {busy} is busy"),
                ha="left", va="center", fontsize=6.4, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    if args.absolute:
        ax.set_ylim(0, max(max(dm), max(pm)) * 1.12)
        ax.set_ylabel("KV actually held (GB)\nreserved-but-empty excluded")
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel(use_axis_label(args.metric))
    ax.set_xlim(min(xs), max(xs))
    ax.set_xlabel("time (s)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, length=2.5, width=0.5)
    ax.grid(False)
    if args.legend_top:
        # at column width an in-plot legend covers a third of the data; put it above
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  fontsize=6.6, handlelength=1.4, borderpad=0.2, columnspacing=1.4,
                  handletextpad=0.5)
    else:
        ax.legend(loc="lower left", ncol=2, frameon=True, fancybox=False,
                  facecolor="white", edgecolor=MUTED, fontsize=6.4, handlelength=1.4,
                  borderpad=0.3, columnspacing=1.0).set_zorder(8)

    print(f"\n=== role occupancy, overlay ({os.path.basename(args.csv)}) ===")
    _u = " GB (summed over the role's GPUs)" if args.absolute else "%"
    print(f"  decode  {d_labels}  time-avg {d_avg:.1f}{_u}")
    print(f"  prefill {p_labels}  time-avg {p_avg:.1f}{_u}")
    print(f"  ratio prefill/decode = {p_avg / d_avg if d_avg else 0:.1f}x")
    print(f"  mean gap = {g_avg:.1f} "
          f"{'GB' if args.absolute else 'points'}; decode above prefill "
          f"{frac_d_above:.0f}% of samples")
    # Only meaningful on the percentage view: in absolute mode p_avg is GB, not a share
    # of anything, so "leaves 59% unused" would be arithmetic on the wrong units.
    if not args.absolute:
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
    ap.add_argument("--absolute", action="store_true",
                    help="overlay in GB of KV actually holding data, instead of %% of the "
                         "reserved pool. Removes the confound of the two roles having "
                         "different pool sizes and mem-fractions.")
    ap.add_argument("--kib-per-token", type=float, default=128,
                    help="KV bytes per token; Llama-3.1-8B is 128 KiB")
    ap.add_argument("--decompose", action="store_true",
                    help="split each role's curve into working set (token_usage) and "
                         "cache (the rest of cache_occupancy)")
    ap.add_argument("--overlay", action="store_true",
                    help="single axes instead of two stacked panels: decode as the "
                         "light area behind, prefill as the dark area in front, so the "
                         "visible light band is the imbalance. Half the page height.")
    ap.add_argument("--no-gap-label", action="store_true",
                    help="--overlay: drop the arrow/label naming the decode-prefill gap")
    ap.add_argument("--legend-top", action="store_true",
                    help="--overlay: put the legend above the axes instead of inside "
                         "(recommended at --width 3.5, where it would cover the data)")
    ap.add_argument("--no-avg", action="store_true",
                    help="--overlay: drop the dashed time-average lines and their labels")
    # Emitted from inside the renderer, after the same role-averaging and the same metric
    # check the figure uses, so the deck and the PDF cannot drift apart. A separate CSV
    # parser for the PPTX would be a second place for the gauge confusion to reappear.
    ap.add_argument("--export-json", default=None,
                    help="also write the overlay series to JSON (for make_*_pptx.js)")
    ap.add_argument("--width", type=float, default=6.4,
                    help="figure width (in); 3.5 = one IEEE/CAL column")
    ap.add_argument("--height", type=float, default=1.85,
                    help="figure height (in); only used by --overlay")
    args = ap.parse_args()

    args.metric = read_metric(args.csv)
    print(f"  gauge in '_use' columns: {args.metric or 'UNRECORDED'}")
    if args.metric == "token_usage":
        print("  [warn] token_usage EXCLUDES the radix cache (num_used = max_total - "
              "available - evictable), so a prefill pool full of retained prefixes reads "
              "near zero. This is in-flight KV, not pool occupancy -- do not label the "
              "result 'KV memory utilization'.")
    elif args.metric is None:
        print("  [warn] the CSV has no '# metric:' line, so which gauge these numbers "
              "came from is unrecorded. Re-sample with the current "
              "kv_occupancy_timeseries.py before using this in the paper.")

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

    if args.decompose:
        tp, pdata, _ = load_pressure(args.csv)
        render_decomposed(args, t, data, pdata, p_labels, d_labels)
        return
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

    axD.set_ylabel(use_axis_label(args.metric))
    axP.set_ylabel(use_axis_label(args.metric))
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

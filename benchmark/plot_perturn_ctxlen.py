#!/usr/bin/env python3
"""
Two-panel figure: per-turn TTFT and decode throughput vs GROWING CONTEXT LENGTH,
comparing park (session-keyed slab, host-RAM-free) against hicache.

x-axis = context length (prompt tokens the server saw at that turn) -> grows as the
conversation advances turn by turn. Points from all items/turns are binned by context
length; each arm is drawn as mean line + IQR (25-75%) band.
  (left)  TTFT vs context length   -- lower is better
  (right) throughput vs context length -- higher is better

Consumes sglang_perturn_ctxlen.py output ({"points":[{ctx,ttft,tput,turn}...]}).
Accepts multiple files per arm (reps) -- points are pooled.

사용:
  python benchmark/plot_perturn_ctxlen.py \
    --park results/perturn_park_r*.json --hicache results/perturn_hicache_r*.json \
    --out results/perturn/perturn_ctxlen.png
"""
import argparse
import json
import os
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

C_HICACHE = PALETTE["hicache"]
C_PARK = PALETTE["park"]
C_BOTH = PALETTE["both"]
INK, MUTED = PALETTE["ink"], PALETTE["muted"]


def load_points(paths):
    """Build per-turn points from the `results` turns so we get e2e latency and
    completion tokens (the flat `points` list lacks them). Each point:
      ctx  = prompt_tokens (context length the server saw)
      ttft = time to first token
      good = effective throughput = completion_tokens / e2e_latency
             (tok/s from request submit to last token -- includes TTFT, so it is NOT
             confounded by the decode-batch-occupancy effect that inflates per-request
             *decode* rate when an arm is starving/queueing).
    """
    pts = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] skip {p}: {e}")
            continue
        for r in d.get("results", []):
            for t in r.get("turns", []):
                ctx, ttft = t.get("prompt_tokens"), t.get("ttft_s")
                e2e, ntok = t.get("e2e_latency_s"), t.get("completion_tokens")
                if not ctx or not ttft:
                    continue
                good = (ntok / e2e) if (e2e and ntok) else None
                pts.append({"ctx": ctx, "ttft": ttft, "good": good, "turn": t.get("turn")})
    return pts


def sliding(points, key, win, step, min_n):
    """Overlapping sliding-window median over context length: many points AND smooth
    (adjacent points share data). Returns (center, median, q25, q75, n) per window."""
    pts = [(p["ctx"], p[key]) for p in points
           if p.get(key) is not None and p.get("ctx") is not None]
    if not pts:
        return []
    xs = sorted(x for x, _ in pts)
    lo, hi = xs[0], xs[-1]
    rows, c = [], lo + win / 2
    while c <= hi - win / 2 + step:
        vals = sorted(v for x, v in pts if c - win / 2 <= x <= c + win / 2)
        n = len(vals)
        if n >= min_n:
            med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
            rows.append((c, med, vals[int(0.25 * (n - 1))], vals[int(0.75 * (n - 1))], n))
        c += step
    return rows


def binned(points, key, bin_w, min_n):
    """Group points into fixed-width context-length bins; return per-bin
    (center, median, q25, q75, n). Median (not mean) so the long TTFT tail doesn't
    drag the line."""
    buckets = {}
    for pt in points:
        v = pt.get(key)
        c = pt.get("ctx")
        if v is None or c is None:
            continue
        b = int(c // bin_w)
        buckets.setdefault(b, []).append(v)
    rows = []
    for b in sorted(buckets):
        vals = sorted(buckets[b])
        if len(vals) < min_n:
            continue
        n = len(vals)
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        q25 = vals[int(0.25 * (n - 1))]
        q75 = vals[int(0.75 * (n - 1))]
        rows.append(((b + 0.5) * bin_w, med, q25, q75, n))
    return rows


def draw(ax, arms, ylabel, title, better):
    """arms = list of (rows, color, label, key)."""
    for rows, color, label, key in arms:
        if not rows:
            continue
        xs = [r[0] / 1000.0 for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, color=color, lw=1.5, ls=STYLE[key]["ls"], marker=STYLE[key]["marker"],
                ms=4, label=label, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("context length (k tokens), grows each turn")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({better})")
    style_axes(ax)
    # legend goes in a single boxed row below both panels (see main)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park", nargs="+", required=True)
    ap.add_argument("--hicache", nargs="+", required=True)
    ap.add_argument("--both", nargs="+", default=[], help="optional coexistence arm (hicache+victim)")
    ap.add_argument("--bin", type=int, default=800, help="context-length bin width (tokens); "
                    "smaller = more points but noisier (500->7pts, 800->5pts, 1500->3pts)")
    ap.add_argument("--smooth", type=int, default=0, help="if >0, use an overlapping "
                    "sliding-window median of this width (tokens) instead of disjoint bins "
                    "-- many points AND smooth (the clean 'tie' view, e.g. 1400)")
    ap.add_argument("--min-n", type=int, default=3, help="min points per bin to plot")
    ap.add_argument("--out", default="results/perturn/perturn_ctxlen.png")
    args = ap.parse_args()

    park = load_points(args.park)
    hic = load_points(args.hicache)
    both = load_points(args.both) if args.both else []
    print(f"park points={len(park)}  hicache points={len(hic)}  both points={len(both)}")
    if not park or not hic:
        print("[error] one arm has no points -- did both runs succeed with include_usage?")

    def bin_arm(pts, key):
        if not pts:
            return []
        if args.smooth > 0:
            return sliding(pts, key, args.smooth, max(1, args.smooth // 6), args.min_n)
        return binned(pts, key, args.bin, args.min_n)

    # console summary (overall medians)
    def omed(pts, k):
        vs = sorted(p[k] for p in pts if p.get(k) is not None)
        return median(vs) if vs else float("nan")
    print(f"\n=== overall medians ===")
    for name, pts in (("hicache", hic), ("park", park), ("both", both)):
        if pts:
            print(f"  {name:8} TTFT={omed(pts,'ttft'):.3f}s  goodput={omed(pts,'good'):.1f} tok/s")

    def arms(key):
        out = [(bin_arm(hic, key), C_HICACHE, "HiCache (host tier)", "hicache"),
               (bin_arm(park, key), C_PARK, "KV victim cache (GPU, host-free)", "park")]
        if both:
            out.append((bin_arm(both, key), C_BOTH, "HiCache + victim (layered)", "both"))
        return out

    use_paper_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    draw(axL, arms("ttft"), "median TTFT (s)", "(a) TTFT", "lower is better")
    draw(axR, arms("good"), "median effective throughput (tok/s)",
         "(b) Effective throughput", "higher is better")
    # single boxed legend centered below both panels
    handles, labels = axL.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True,
               fancybox=False, edgecolor=MUTED, handlelength=2.6, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.0))
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

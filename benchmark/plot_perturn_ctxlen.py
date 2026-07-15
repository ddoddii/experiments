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

# dataviz palette (light): blue = hicache (incumbent), aqua = park (ours)
C_HICACHE = "#2a78d6"
C_PARK = "#1baf7a"
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


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


def draw(ax, park_rows, hic_rows, ylabel, title, better):
    for rows, color, label in ((hic_rows, C_HICACHE, "hicache"),
                               (park_rows, C_PARK, "park (slab, host-RAM-free)")):
        if not rows:
            continue
        xs = [r[0] / 1000.0 for r in rows]
        ys = [r[1] for r in rows]
        lo = [r[2] for r in rows]
        hi = [r[3] for r in rows]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.14, lw=0, zorder=2)
        ax.plot(xs, ys, "-o", color=color, lw=2, ms=6, label=label, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("context length (k tokens)  —  grows each turn →", color=SECOND, fontsize=10.5)
    ax.set_ylabel(ylabel, color=SECOND, fontsize=10.5)
    ax.set_title(f"{title}   ({better})", fontsize=11, color=INK, fontweight="bold")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park", nargs="+", required=True)
    ap.add_argument("--hicache", nargs="+", required=True)
    ap.add_argument("--bin", type=int, default=1500, help="context-length bin width (tokens)")
    ap.add_argument("--min-n", type=int, default=3, help="min points per bin to plot")
    ap.add_argument("--out", default="results/perturn/perturn_ctxlen.png")
    args = ap.parse_args()

    park = load_points(args.park)
    hic = load_points(args.hicache)
    print(f"park points={len(park)}  hicache points={len(hic)}")
    if not park or not hic:
        print("[error] one arm has no points -- did both runs succeed with include_usage?")

    park_ttft = binned(park, "ttft", args.bin, args.min_n)
    hic_ttft = binned(hic, "ttft", args.bin, args.min_n)
    park_good = binned(park, "good", args.bin, args.min_n)
    hic_good = binned(hic, "good", args.bin, args.min_n)

    # console summary (overall medians)
    def omed(pts, k):
        vs = sorted(p[k] for p in pts if p.get(k) is not None)
        return median(vs) if vs else float("nan")
    print(f"\n=== overall medians ===")
    print(f"  TTFT       park={omed(park,'ttft'):.3f}s  hicache={omed(hic,'ttft'):.3f}s")
    print(f"  goodput    park={omed(park,'good'):.1f}   hicache={omed(hic,'good'):.1f} tok/s  (completion/e2e)")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Per-turn TTFT / effective throughput vs growing context length  (2P2D, BFCL multi-turn)",
                 fontsize=13, fontweight="bold", color=INK)
    draw(axL, park_ttft, hic_ttft, "median TTFT (s)", "TTFT", "lower better")
    draw(axR, park_good, hic_good, "median effective throughput (tok/s)",
         "Effective throughput  (tokens / end-to-end time)", "higher better")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

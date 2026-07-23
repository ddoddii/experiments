#!/usr/bin/env python3
"""
TTFT (and effective throughput) vs CONVERSATION DEPTH (turn index) -- the honest
axis for a recompute-vs-prefix-cache comparison.

Why turn index and NOT context length: the two arms are driven by DIFFERENT variables.
  - recompute (radix off): re-prefills the WHOLE context every turn, so its TTFT is a
    function of ABSOLUTE context length -> binning by ctx is fair for it.
  - cache arms (park/hicache): reuse the cached prefix and only prefill the NEW delta,
    so their TTFT is a function of the per-turn delta, NOT absolute ctx. At turn>=1 the
    delta is tiny, so TTFT is ~flat-low regardless of how long the context is.
Plotting the cache arms' TTFT against absolute ctx mixes expensive turn-0 full-prefills
with cheap cached later-turns at overlapping ctx values and inflates the per-bin median
-- which is exactly how the ctx-binned figure made recompute look "faster". Turn index
separates them cleanly: turn 0 is identical across arms (no reuse possible), then the
cache arms drop while recompute stays flat.

Consumes the same results/{CONFIG}.json (results[].turns[] with turn/ttft_s/e2e_latency_s/
completion_tokens). Accepts multiple files per arm (reps pooled).

사용:
  python benchmark/plot_perturn_byturn.py \
    --park results/perturn_park_r*.json --hicache results/perturn_hicache_r*.json \
    --recompute results/perturn_recompute_r*.json \
    --out results/perturn/fig_perturn_byturn.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

C_HICACHE = PALETTE["hicache"]
C_PARK = PALETTE["park"]
C_BOTH = PALETTE["both"]
C_RECOMPUTE = PALETTE["recompute"]
INK, MUTED = PALETTE["ink"], PALETTE["muted"]


def load_points(paths):
    """Per-turn points keyed by turn index. good = completion_tokens / e2e_latency."""
    pts = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] skip {p}: {e}")
            continue
        for r in d.get("results", []):
            for t in r.get("turns", []):
                turn, ttft = t.get("turn"), t.get("ttft_s")
                e2e, ntok = t.get("e2e_latency_s"), t.get("completion_tokens")
                if turn is None or ttft is None:
                    continue
                good = (ntok / e2e) if (e2e and ntok) else None
                pts.append({"turn": turn, "ttft": ttft, "good": good,
                            "ctx": t.get("prompt_tokens")})
    return pts


def by_turn(points, key, min_n):
    """Group by turn index; return per-turn (turn, median, q25, q75, n)."""
    buckets = {}
    for pt in points:
        v = pt.get(key)
        k = pt.get("turn")
        if v is None or k is None:
            continue
        buckets.setdefault(k, []).append(v)
    rows = []
    for k in sorted(buckets):
        vals = sorted(buckets[k])
        if len(vals) < min_n:
            continue
        n = len(vals)
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        rows.append((k, med, vals[int(0.25 * (n - 1))], vals[int(0.75 * (n - 1))], n))
    return rows


def draw(ax, arms, ylabel, title, better):
    """arms = list of (rows, color, label, key)."""
    for rows, color, label, key in arms:
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, color=color, lw=1.5, ls=STYLE[key]["ls"], marker=STYLE[key]["marker"],
                ms=4, label=label, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("conversation depth (turn index)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({better})")
    # integer turn ticks only
    ax.xaxis.get_major_locator().set_params(integer=True)
    style_axes(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park", nargs="+", required=True)
    ap.add_argument("--hicache", nargs="+", required=True)
    ap.add_argument("--both", nargs="+", default=[])
    ap.add_argument("--recompute", nargs="+", default=[])
    ap.add_argument("--min-n", type=int, default=3, help="min points per turn to plot")
    ap.add_argument("--max-turn", type=int, default=0, help="if >0, clip to this turn index "
                    "(later turns have few conversations left -> noisy)")
    ap.add_argument("--out", default="results/perturn/fig_perturn_byturn.png")
    args = ap.parse_args()

    park = load_points(args.park)
    hic = load_points(args.hicache)
    both = load_points(args.both) if args.both else []
    recomp = load_points(args.recompute) if args.recompute else []
    print(f"park pts={len(park)}  hicache pts={len(hic)}  both pts={len(both)}  "
          f"recompute pts={len(recomp)}")

    def rows(pts, key):
        r = by_turn(pts, key, args.min_n) if pts else []
        if args.max_turn > 0:
            r = [x for x in r if x[0] <= args.max_turn]
        return r

    # console table
    print("\n=== median TTFT by turn ===")
    hdr = {p["turn"] for p in hic + park + recomp}
    print(f"  {'turn':>4}  {'hicache':>8}  {'park':>8}  {'recompute':>9}")
    hi_m = {r[0]: r[1] for r in rows(hic, 'ttft')}
    pk_m = {r[0]: r[1] for r in rows(park, 'ttft')}
    rc_m = {r[0]: r[1] for r in rows(recomp, 'ttft')}
    for k in sorted(hdr):
        def fmt(m):
            return f"{m[k]:.3f}s" if k in m else "   -   "
        print(f"  {k:>4}  {fmt(hi_m):>8}  {fmt(pk_m):>8}  {fmt(rc_m):>9}")

    def arms(key):
        out = [(rows(hic, key), C_HICACHE, "SGLang", "hicache"),
               (rows(park, key), C_PARK, "KV Victim Cache", "park")]
        if both:
            out.append((rows(both, key), C_BOTH, "SGLang + KV Victim Cache (layered)", "both"))
        if recomp:
            out.append((rows(recomp, key), C_RECOMPUTE, "Recompute (no cache)", "recompute"))
        return out

    use_paper_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    draw(axL, arms("ttft"), "median TTFT (s)", "(a) TTFT", "lower is better")
    draw(axR, arms("good"), "median effective throughput (tok/s)",
         "(b) Effective throughput", "higher is better")
    handles, labels = axL.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True,
               fancybox=False, edgecolor=MUTED, handlelength=2.6, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.0))
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

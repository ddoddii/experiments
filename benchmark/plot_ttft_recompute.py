#!/usr/bin/env python3
"""
Single-panel TTFT figure: 'cached' vs 'recompute' only (2 lines).

x-axis is selectable:
  --x turn : conversation depth (turn index). turn 0 is identical across arms (no
             reuse possible); then 'cached' drops to the delta-prefill cost while
             'recompute' stays flat (re-prefills the whole context every turn).
  --x ctx  : absolute context length (prompt tokens), binned. This is the axis for
             the 'recompute cost grows with context' story -- but note it only shows
             super-linear growth if the workload's context actually spans a wide range
             (BFCL barely grows across turns: ~4.8k->5.1k, so recompute looks flat;
             use ShareGPT or a long-context sweep to see the blow-up).

  --cached and --recompute each take one or more result JSONs (results[].turns[] with
  turn / prompt_tokens / ttft_s). Reps are pooled.

사용:
  # conversation-depth view (cached drops, recompute flat)
  python benchmark/plot_ttft_recompute.py \
    --cached results/perturn_hicache_r1.json \
    --recompute results/perturn_recompute_r1.json \
    --x turn --max-turn 4 --out results/perturn/fig_ttft_byturn.png

  # context-length view (recompute cost vs context) -- best on ShareGPT
  python benchmark/plot_ttft_recompute.py \
    --cached results/perturn_sharegpt_hicache_r1.json \
    --recompute results/perturn_sharegpt_recompute_r1.json \
    --x ctx --bin 1000 --out results/perturn_sharegpt/fig_ttft_ctx.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

C_CACHED = PALETTE["park"]        # green solid square
C_RECOMPUTE = PALETTE["recompute"]  # red dash-dot diamond
MUTED = PALETTE["muted"]


def load(paths):
    pts = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] skip {p}: {e}")
            continue
        for r in d.get("results", []):
            for t in r.get("turns", []):
                if t.get("ttft_s") is None:
                    continue
                pts.append({"turn": t.get("turn"), "ctx": t.get("prompt_tokens"),
                            "ttft": t.get("ttft_s")})
    return pts


def agg_turn(points, min_n, max_turn):
    buckets = {}
    for pt in points:
        k = pt.get("turn")
        if k is None:
            continue
        buckets.setdefault(k, []).append(pt["ttft"])
    rows = []
    for k in sorted(buckets):
        if max_turn and k > max_turn:
            continue
        vals = sorted(buckets[k])
        if len(vals) < min_n:
            continue
        n = len(vals)
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        rows.append((k, med, n))
    return rows


def agg_ctx(points, bin_w, min_n):
    buckets = {}
    for pt in points:
        c = pt.get("ctx")
        if c is None:
            continue
        buckets.setdefault(int(c // bin_w), []).append(pt["ttft"])
    rows = []
    for b in sorted(buckets):
        vals = sorted(buckets[b])
        if len(vals) < min_n:
            continue
        n = len(vals)
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        rows.append(((b + 0.5) * bin_w / 1000.0, med, n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", nargs="+", required=True)
    ap.add_argument("--recompute", nargs="+", required=True)
    ap.add_argument("--x", choices=["turn", "ctx"], default="turn")
    ap.add_argument("--bin", type=int, default=1000, help="ctx bin width (tokens), --x ctx")
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--max-turn", type=int, default=0, help="clip turns above this (--x turn)")
    ap.add_argument("--out", default="results/perturn/fig_ttft.png")
    args = ap.parse_args()

    cached = load(args.cached)
    recomp = load(args.recompute)
    print(f"cached pts={len(cached)}  recompute pts={len(recomp)}")

    if args.x == "turn":
        cr = agg_turn(cached, args.min_n, args.max_turn)
        rr = agg_turn(recomp, args.min_n, args.max_turn)
        xlabel = "conversation depth (turn index)"
    else:
        cr = agg_ctx(cached, args.bin, args.min_n)
        rr = agg_ctx(recomp, args.bin, args.min_n)
        xlabel = "context length (k tokens)"

    # console table
    print(f"\n=== median TTFT ({args.x}) ===")
    print(f"  {'x':>6}  {'cached':>8}  {'recompute':>9}")
    cm = {r[0]: r[1] for r in cr}
    rm = {r[0]: r[1] for r in rr}
    for k in sorted(set(cm) | set(rm)):
        cs = f"{cm[k]:.3f}s" if k in cm else "   -   "
        rs = f"{rm[k]:.3f}s" if k in rm else "   -   "
        print(f"  {k:>6}  {cs:>8}  {rs:>9}")

    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for rows, color, label, key in (
        (rr, C_RECOMPUTE, "recompute", "recompute"),
        (cr, C_CACHED, "cached", "park"),
    ):
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, color=color, lw=1.6, ls=STYLE[key]["ls"], marker=STYLE[key]["marker"],
                ms=4.5, label=label, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median TTFT (s)")
    ax.set_title("TTFT (lower is better)")
    if args.x == "turn":
        ax.xaxis.get_major_locator().set_params(integer=True)
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor=MUTED, facecolor="white",
              handlelength=2.6, loc="best")
    fig.tight_layout()
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

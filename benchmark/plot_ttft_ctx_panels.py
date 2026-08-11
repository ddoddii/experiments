#!/usr/bin/env python3
"""
Four-panel summary of the TTFT-vs-context-length sweep (recompute / hicache /
park_host / park).

The panels are chosen so the figure cannot be read as claiming more than the data
supports. In particular panel (d) exists because the medians alone are misleading at
64k: park_host's median is 24.0s but its BEST rep is 5.5s, and hicache's median is
19.4s but its best rep is 5.1s. Both arms restore sometimes and fall through to a full
re-prefill otherwise, so the median gap between park and those arms is mostly a
DIFFERENCE IN HOW OFTEN THE RESTORE HAPPENS, not in how fast the transfer is. Plotting
only medians would show a 5.2x "medium" effect that the per-rep spread does not support.

  (a) TTFT vs context length, log-log -- the headline curve.
  (b) Speedup over recompute -- same data, normalised; 1.0 means "no better than
      recomputing from scratch".
  (c) Where the restored KV came from, in tokens: GPU park pool vs host DRAM. This is
      the placement evidence -- park is GPU-only, park_host is host-only by
      construction (SGLANG_KV_PARK_FORCE_HOST=1), and the two baselines restore
      nothing through this path at all.
  (d) 64k per-rep spread: best rep vs median. Separates "the restore was fast" from
      "the restore happened at all".

사용:
  python benchmark/plot_ttft_ctx_panels.py --dir results/ttft_ctx \
      --out results/ttft_ctx/fig_ttft_ctx_panels
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

ARMS = [
    ("recompute", "Recompute", PALETTE["recompute"], "recompute"),
    ("hicache", "SGLang", PALETTE["hicache"], "hicache"),
    ("park_host", "Ours (host DRAM)", PALETTE["both"], "both"),
    ("park", "Ours", PALETTE["park"], "park"),
]


def fmt_tokens(n):
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 1024:
        return f"{n / 1024:.1f}k"
    return str(n)


def load(dirpath):
    out = {}
    for key, *_ in ARMS:
        path = os.path.join(dirpath, f"{key}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            rows = json.load(fh)["rows"]
        out[key] = {r["context_len"]: r for r in rows}
    if not out:
        raise SystemExit(f"no <arm>.json files found in {dirpath}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ttft_ctx")
    ap.add_argument("--out", default="results/ttft_ctx/fig_ttft_ctx_panels")
    ap.add_argument("--detail-len", type=int, default=65536,
                     help="context length broken out in panel (d)")
    ap.add_argument("--restore-units", default="gb", choices=["gb", "tokens"],
                     help="y unit for panel (c)")
    ap.add_argument(
        "--restore-basis", default="rep", choices=["rep", "total"],
        help="panel (c): 'rep' divides by the repetition count, 'total' shows the raw "
             "sweep total. Default is per-rep because the total scales with REPS, an "
             "arbitrary experimental choice -- reporting 25.7GB restored at 64k invites "
             "the reader to compare it against a per-request budget it was never "
             "measured against. Per rep it is 8.58GB, which is one document's KV and "
             "can be checked against 65536 x 128 KiB = 8.59GB directly.",
    )
    ap.add_argument(
        "--bytes-per-token", type=int, default=131072,
        help="fallback KV bytes/token for the GB conversion, used only when the result "
             "rows predate the probe recording it. 131072 = 128 KiB = Llama-3.1-8B "
             "(GQA, bf16); an MHA model is several times that, so this default would "
             "understate a different model badly while still looking plausible.",
    )
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=5.0)
    args = ap.parse_args()

    data = load(args.dir)
    xs = sorted({L for arm in data.values() for L in arm})

    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(args.width, args.height))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---------------------------------------------------------------- (a) TTFT
    for key, label, color, skey in ARMS:
        if key not in data:
            continue
        pts = [(L, data[key][L]["ttft_median_s"]) for L in xs
               if L in data[key] and data[key][L].get("ttft_median_s") is not None]
        if pts:
            ax_a.plot([p[0] for p in pts], [p[1] for p in pts],
                      color=color, label=label, **STYLE[skey])
    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([fmt_tokens(v) for v in xs])
    ax_a.xaxis.set_minor_locator(mticker.NullLocator())
    ax_a.set_xlabel("Context length (tokens)")
    ax_a.set_ylabel("TTFT (median, s)")
    ax_a.set_title("(a)  Time to first token")
    ax_a.legend(loc="upper left")
    style_axes(ax_a)

    # ---------------------------------------------------------------- (b) speedup
    base = data.get("recompute", {})
    for key, label, color, skey in ARMS:
        if key not in data or key == "recompute":
            continue
        pts = []
        for L in xs:
            b = base.get(L, {}).get("ttft_median_s")
            v = data[key].get(L, {}).get("ttft_median_s")
            if b and v:
                pts.append((L, b / v))
        if pts:
            ax_b.plot([p[0] for p in pts], [p[1] for p in pts],
                      color=color, label=label, **STYLE[skey])
    ax_b.axhline(1.0, color=PALETTE["recompute"], lw=0.8, ls="-.", zorder=1)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([fmt_tokens(v) for v in xs])
    ax_b.xaxis.set_minor_locator(mticker.NullLocator())
    ax_b.set_xlabel("Context length (tokens)")
    ax_b.set_ylabel("Speedup over recompute")
    ax_b.set_title("(b)  Speedup vs. no cache")
    style_axes(ax_b)

    # ---------------------------------------------------------------- (c) restore source
    width = 0.38
    idx = range(len(xs))
    def restored(arm, field):
        vals = []
        for L in xs:
            row = data.get(arm, {}).get(L, {})
            v = row.get(field, 0) or 0
            if args.restore_basis == "rep":
                v /= max(1, row.get("reps") or 1)
            vals.append(v)
        return vals

    gpu = restored("park", "park_gpu_tokens")
    hst = restored("park_host", "park_host_tokens")
    if args.restore_units == "gb":
        # Prefer the bytes/token the probe recorded for the model actually measured;
        # only fall back to the CLI default for rows written before it did, and say so.
        bpts = [r.get("bytes_per_token") for arm in data.values() for r in arm.values()
                if r.get("bytes_per_token")]
        bpt = bpts[0] if bpts else args.bytes_per_token
        if not bpts:
            print(f"[warn] no bytes_per_token in these results -- converting with the "
                  f"--bytes-per-token default ({args.bytes_per_token} = "
                  f"{args.bytes_per_token / 1024:.0f} KiB/token). Correct for "
                  f"Llama-3.1-8B; wrong for a model with a different KV shape.")
        scale, unit = bpt / 1e9, "GB"
    else:
        scale, unit = 1 / 1000, "k tokens"
    ax_c.bar([i - width / 2 for i in idx], [g * scale for g in gpu], width,
             color=PALETTE["park"], label="Ours: GPU park pool")
    ax_c.bar([i + width / 2 for i in idx], [h * scale for h in hst], width,
             color=PALETTE["both"], label="Ours (host DRAM): host")
    ax_c.set_xticks(list(idx))
    ax_c.set_xticklabels([fmt_tokens(v) for v in xs])
    ax_c.set_xlabel("Context length (tokens)")
    ax_c.set_ylabel(f"KV restored ({unit})")
    ax_c.set_title("(c)  Where the restored KV came from"
                   + (", per request" if args.restore_basis == "rep" else ", sweep total"))
    ax_c.legend(loc="upper left")
    style_axes(ax_c)

    # ---------------------------------------------------------------- (d) per-rep spread
    L = args.detail_len
    present = [(k, lab, c) for k, lab, c, _ in ARMS if L in data.get(k, {})]
    pos = range(len(present))
    best = [data[k][L]["ttft_min_s"] for k, _, _ in present]
    med = [data[k][L]["ttft_median_s"] for k, _, _ in present]
    ax_d.bar([i - width / 2 for i in pos], best, width,
             color=[c for _, _, c in present], alpha=0.55)
    ax_d.bar([i + width / 2 for i in pos], med, width,
             color=[c for _, _, c in present])
    ax_d.set_xticks(list(pos))
    ax_d.set_xticklabels([lab.replace(" (host DRAM)", "\n(host)") for _, lab, _ in present])
    ax_d.set_ylabel("TTFT (s)")
    ax_d.set_title(f"(d)  {fmt_tokens(L)}: best rep vs. median")
    # Neutral proxy handles. Passing label= to the bar() calls above would make the
    # legend adopt the FIRST bar's colour (recompute's red) for both entries, implying
    # the pair encodes an arm rather than best-vs-median within every arm.
    from matplotlib.patches import Patch
    ax_d.legend(handles=[Patch(facecolor=PALETTE["muted"], alpha=0.55, label="best rep"),
                         Patch(facecolor=PALETTE["muted"], label="median")],
                loc="upper left")
    style_axes(ax_d)

    fig.tight_layout()
    savefig(fig, args.out)


if __name__ == "__main__":
    main()

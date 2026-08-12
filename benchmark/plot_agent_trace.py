#!/usr/bin/env python3
"""
Agent-trace replay summary in the reference paper's normalised-bar style: one panel per
metric, x = concurrency, grouped bars per arm, everything divided by the recompute bar
so 1.0 means "no better than having no cache at all".

TPOT IS SHOWN EVEN THOUGH IT IS FLAT, and that is the point of including it. Prefix
reuse is a PREFILL optimisation: it changes what the server has to compute before the
first token and does essentially nothing for the per-token decode rate afterwards. A
normalised TPOT panel sitting at ~1.0 (or slightly above) is the honest statement that
the gain is where the mechanism acts and that it does not come from somewhere else.
Omitting the panel would let a reader assume the TTFT win carries into decode.

The TTFT panel defaults to the MEDIAN. --ttft-stat mean is available and is the fairer
summary when the tail matters: on the C=4 data park's p50 is 0.41x recompute but its
mean/turn is 0.66x, because park misses fall back to a full re-prefill and land in the
tail. Both are real; reporting only p50 would overstate the effect.

사용:
  python benchmark/plot_agent_trace.py --dir results/agent_trace \
      --out results/agent_trace/fig_agent_trace
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

ARMS = [
    ("recompute", "Recompute", "#B8BCC0"),
    ("hicache", "SGLang", PALETTE["hicache"]),
    ("park_host", "Ours (host DRAM)", PALETTE["both"]),
    ("park", "Ours", PALETTE["park"]),
]


def _flat(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
    return out


def load_hit_rates(dirpath, node="prefill"):
    """Server-side prefix hit rate per (arm, concurrency), from the prometheus
    before/after scrapes: cached_tokens / prompt_tokens on the PREFILL nodes.

    Prefill rather than decode because that is where a prefix hit removes work that
    would otherwise land in TTFT; the decode counter tracks the same reuse one hop
    later and would double-report it.

    This is the SERVER's own accounting, independent of the park telemetry, so it says
    whether a mechanism actually reduced recomputation rather than whether its own
    counters incremented."""
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "metrics_*_after.json"))):
        tag = re.match(r"metrics_(.+)_after\.json$", os.path.basename(f))
        b = f.replace("_after", "_before")
        if not tag or not os.path.exists(b):
            continue
        m = re.match(r"(.+)_c(\d+)$", tag.group(1))
        if not m:
            continue
        try:
            A, B = _flat(json.load(open(f))), _flat(json.load(open(b)))
        except Exception:  # noqa: BLE001
            continue
        ck = f"nodes.{node}.sglang:cached_tokens_total"
        pk = f"nodes.{node}.sglang:prompt_tokens_total"
        if ck not in A or pk not in A:
            continue
        c = float(A[ck]) - float(B.get(ck, 0))
        p = float(A[pk]) - float(B.get(pk, 0))
        if p > 0:
            out.setdefault(m.group(1), {})[int(m.group(2))] = 100 * c / p
    return out


def load(dirpath):
    out = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "bench_*_c*.json"))):
        m = re.match(r"bench_(.+)_c(\d+)\.json$", os.path.basename(p))
        if not m:
            continue
        arm, c = m.group(1), int(m.group(2))
        try:
            with open(p) as fh:
                out.setdefault(arm, {})[c] = json.load(fh)["summary"]
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {p}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/agent_trace")
    ap.add_argument("--out", default="results/agent_trace/fig_agent_trace")
    ap.add_argument("--ttft-stat", default="ttft_p50_s",
                     choices=["ttft_p50_s", "ttft_mean_per_turn_s", "ttft_p95_s"])
    ap.add_argument("--width", type=float, default=9.5)
    ap.add_argument("--height", type=float, default=2.5)
    args = ap.parse_args()

    data = load(args.dir)
    if "recompute" not in data:
        raise SystemExit("no recompute arm found -- it is the normalisation baseline, "
                          "so without it every bar would be relative to nothing")
    # Only concurrencies that HAVE the baseline. Taking the union across arms leaves a
    # labelled but empty group wherever recompute has not been run yet (a half-finished
    # sweep always has some), and an empty group reads as "measured, and it was zero"
    # rather than "not measured".
    cs = sorted(data["recompute"])
    skipped = sorted({c for arm in data.values() for c in arm} - set(cs))
    if skipped:
        print(f"[note] concurrency {skipped} present for some arms but not recompute, "
              f"so it cannot be normalised -- omitted from the figure.")
    present = [a for a in ARMS if a[0] in data]

    use_paper_style()
    hits = load_hit_rates(args.dir)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(args.width, args.height))

    PANELS = [
        (ax_a, args.ttft_stat, "Normalized TTFT", "(a)  Time to first token"),
        (ax_b, "avg_tpot_s", "Normalized TPOT", "(b)  Time per output token"),
    ]
    width = 0.8 / max(1, len(present))
    for ax, key, ylab, title in PANELS:
        for ai, (arm, label, color) in enumerate(present):
            xs, ys = [], []
            for ci, c in enumerate(cs):
                base = data["recompute"].get(c, {}).get(key)
                v = data[arm].get(c, {}).get(key)
                if base and v:
                    xs.append(ci + (ai - (len(present) - 1) / 2) * width)
                    ys.append(v / base)
            if xs:
                ax.bar(xs, ys, width, color=color, label=label,
                       edgecolor="black", linewidth=0.4)
        ax.axhline(1.0, color="black", lw=0.9, ls="--", zorder=3)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([f"C={c}" for c in cs])
        ax.set_xlabel("Concurrent sessions")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, 1.25)
        style_axes(ax)
    ax_a.legend(loc="lower left", ncol=1)

    # (c) Cache hit rate -- NOT normalised: it is already a ratio, and dividing one hit
    # rate by another would be meaningless. recompute is omitted because it runs with
    # DISABLE_RADIX_CACHE=1, so its hit rate is 0 by construction and a zero-height bar
    # would suggest a measurement rather than an absent mechanism.
    cache_arms = [a for a in present if a[0] != "recompute" and a[0] in hits]
    w2 = 0.8 / max(1, len(cache_arms))
    for ai, (arm, label, color) in enumerate(cache_arms):
        xs, ys = [], []
        for ci, c in enumerate(cs):
            v = hits[arm].get(c)
            if v is not None:
                xs.append(ci + (ai - (len(cache_arms) - 1) / 2) * w2)
                ys.append(v)
        if xs:
            ax_c.bar(xs, ys, w2, color=color, label=label,
                     edgecolor="black", linewidth=0.4)
    ax_c.set_xticks(range(len(cs)))
    ax_c.set_xticklabels([f"C={c}" for c in cs])
    ax_c.set_xlabel("Concurrent sessions")
    ax_c.set_ylabel("Cache Hit Rate (%)")
    ax_c.set_title("(c)  Server-side prefix reuse")
    ax_c.set_ylim(0, 100)
    style_axes(ax_c)

    fig.tight_layout()
    savefig(fig, args.out)

    # The normalised bars hide the absolute scale, and at these context lengths the
    # absolute numbers are the reason the experiment exists.
    print(f"\nabsolute values ({args.ttft_stat} / avg_tpot_s):")
    for c in cs:
        print(f"  C={c}")
        for arm, label, _ in present:
            s = data[arm].get(c)
            if s:
                hr = hits.get(arm, {}).get(c)
                print(f"    {label:20s} TTFT {s.get(args.ttft_stat)}s   "
                      f"TPOT {s.get('avg_tpot_s')}s   "
                      f"p95 {s.get('ttft_p95_s')}s   "
                      f"hit {hr:.1f}%" if hr is not None else
                      f"    {label:20s} TTFT {s.get(args.ttft_stat)}s   "
                      f"TPOT {s.get('avg_tpot_s')}s   p95 {s.get('ttft_p95_s')}s")


if __name__ == "__main__":
    main()

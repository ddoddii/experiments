#!/usr/bin/env python3
"""
Motivation figure: "host DRAM spent on KV offloading is host DRAM the agent stack
cannot have." Two panels from agent_host_pressure.py --mode ramp outputs:

  (a) host capacity left to the agent stack -- bars of MemAvailable during steady
      LLM serving, annotated with the admitted agent concurrency K_max (how many
      concurrent tool executions of --tool-mem-mb each still fit).
  (b) tool-execution p99 latency vs agent concurrency K, one curve per config,
      with a marker at each config's admission cliff (last sustainable K).

사용:
  python benchmark/plot_agent_pressure.py \
    --ramp write_through=results/agent/ramp_wt.json \
           write_back=results/agent/ramp_wb.json \
           L2_only=results/agent/ramp_l2.json \
           park=results/agent/ramp_park.json \
    --out results/agent/fig_agent_host_capacity.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

# config label -> (palette key, STYLE key). park (victim cache) green, HiCache blue,
# extra HiCache variants fall back to amber/red so a 4-config figure stays legible.
FALLBACK = ["hicache", "both", "recompute", "hicache"]


def style_for(label, i):
    low = label.lower()
    if "park" in low or "victim" in low:
        return "park", "park"
    key = FALLBACK[min(i, len(FALLBACK) - 1)]
    return key, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramp", nargs="+", required=True, help="label=ramp.json ...")
    ap.add_argument("--out", default="results/agent/fig_agent_host_capacity.png")
    args = ap.parse_args()

    runs = []
    for spec in args.ramp:
        label, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            print(f"[warn] skip {spec}")
            continue
        runs.append((label, json.load(open(path))))
    if not runs:
        print("[error] no ramp JSONs loaded")
        return

    # baseline MemAvailable = the K=1 sample (agent stack essentially idle)
    print(f"\n=== agent-stack host capacity ===")
    print(f"  {'config':>22}  {'avail@K=1(GB)':>13}  {'K_max':>6}  "
          f"{'admitted(GB)':>12}  {'p99@K=1(ms)':>11}")
    for label, d in runs:
        r0 = d["rows"][0] if d.get("rows") else {}
        print(f"  {label:>22}  {r0.get('mem_avail_gb', 0):>13.1f}  {d['kmax']:>6}  "
              f"{d['admitted_gb']:>12.1f}  {str(r0.get('p99_ms')):>11}")

    use_paper_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # ---- (a) capacity bars
    labels = [l for l, _ in runs]
    avail = [d["rows"][0]["mem_avail_gb"] if d.get("rows") else 0.0 for _, d in runs]
    x = np.arange(len(runs))
    colors = [PALETTE[style_for(l, i)[0]] for i, l in enumerate(labels)]
    axA.bar(x, avail, width=0.62, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    mt = next((d.get("mem_total_gb") for _, d in runs if d.get("mem_total_gb")), None)
    if mt:
        axA.axhline(mt, color=PALETTE["muted"], lw=0.8, ls=":", zorder=2)
        axA.text(len(runs) - 0.5, mt, f" MemTotal {mt:.0f}GB", ha="right", va="bottom",
                 fontsize=6, color=PALETTE["muted"])
    for xi, (a, (_, d)) in enumerate(zip(avail, runs)):
        axA.text(xi, a + 0.01 * max(avail), f"{a:.0f}\n({d['kmax']} tools)",
                 ha="center", va="bottom", fontsize=6.2)
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=20, ha="right")
    axA.set_ylabel("host DRAM left for agent stack (GB)")
    axA.set_title("(a) Capacity available to the agent stack")
    axA.set_ylim(0, (mt or max(avail)) * 1.18)
    style_axes(axA)

    # ---- (b) tool p99 vs concurrency
    for i, (label, d) in enumerate(runs):
        pkey, skey = style_for(label, i)
        rows = [r for r in d.get("rows", []) if r.get("p99_ms") is not None]
        xs = [r["k"] for r in rows]
        ys = [r["p99_ms"] for r in rows]
        # thin the markers: a 95-step ramp would otherwise be a solid band of markers
        every = max(1, len(xs) // 12)
        axB.plot(xs, ys, color=PALETTE[pkey], lw=1.5, ls=STYLE[skey]["ls"],
                 marker=STYLE[skey]["marker"], ms=3.8, markevery=every, label=label,
                 zorder=3, markeredgecolor="white", markeredgewidth=0.5)
        if d["kmax"] and rows:                # admission cliff
            last = [r for r in rows if r["k"] == d["kmax"]]
            if last:
                axB.plot([d["kmax"]], [last[0]["p99_ms"]], marker="x", ms=7,
                         color=PALETTE[pkey], mew=1.6, zorder=4)
    axB.set_xlabel("concurrent tool executions (agent concurrency $K$)")
    axB.set_ylabel("tool-call p99 latency (ms)")
    axB.set_title("(b) Tool execution under host pressure")
    style_axes(axB)
    axB.legend(frameon=True, fancybox=False, edgecolor=PALETTE["muted"],
               facecolor="white", loc="upper left", fontsize=6.4)
    axB.text(0.98, 0.03, "$\\times$ = admission cliff (last sustainable $K$)",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=6,
             color=PALETTE["muted"])

    fig.tight_layout()
    savefig(fig, args.out.rsplit(".", 1)[0])


if __name__ == "__main__":
    main()

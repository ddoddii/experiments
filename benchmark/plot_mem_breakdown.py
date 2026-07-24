#!/usr/bin/env python3
"""
Per-config host/GPU memory decomposition -- the figure that answers the reviewer:
"is that 94GB a CPU DRAM KV pool or just reclaimable Linux page cache?"

Each config is one stacked bar of the PEAK (or --stat mean/last) host footprint,
split into:
  - proc_anon    : SGLang anonymous host RSS  -> NON-reclaimable (L2 pool + heap/pinned)
  - file_cache   : file-backed page cache     -> RECLAIMABLE (HiCache L3 on /tmp/hicache)
with GPU HBM drawn as a separate adjacent bar (different resource, not host RAM).

Feed one breakdown CSV (from sys_mem_breakdown.py) per config, labelled:
  --csv write_through=results/mem/bd_wt.csv \
        write_through_selective=results/mem/bd_wts.csv \
        write_back=results/mem/bd_wb.csv \
        L2_only=results/mem/bd_l2.csv \
        park=results/mem/bd_park.csv

사용:
  python benchmark/plot_mem_breakdown.py --csv <label>=<csv> [...] \
    --stat peak --out results/mem/fig_mem_breakdown.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

GB = 1024.0  # MB -> GB


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else 0.0)
                         for k, v in r.items()})
    return rows


def stat(rows, col, how):
    xs = [r.get(col, 0.0) for r in rows if r.get(col) is not None]
    if not xs:
        return 0.0
    if how == "peak":
        return max(xs)
    if how == "last":
        return xs[-1]
    return sum(xs) / len(xs)          # mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="label=path.csv ...")
    ap.add_argument("--stat", choices=["peak", "mean", "last"], default="peak")
    ap.add_argument("--anon-source", choices=["proc", "global"], default="proc",
                    help="proc = SGLang per-process anon RSS (needs the fixed sampler that "
                    "walks child procs); global = /proc/meminfo AnonPages (use with "
                    "--baseline to subtract non-SGLang tenants; correct for old CSVs whose "
                    "per-process anon undercounted the host pool)")
    ap.add_argument("--baseline", default="", help="config label whose anon+file_cache is "
                    "subtracted from every config -- isolates the HiCache-attributable host "
                    "pool (non-recl) and L3 page cache (recl) above the SGLang+weights floor. "
                    "Use the GPU-only 'park' run as baseline.")
    ap.add_argument("--out", default="results/mem/fig_mem_breakdown.png")
    args = ap.parse_args()

    anon_col = "anonpages_mb" if args.anon_source == "global" else "proc_anon_mb"
    raw = {}
    order = []
    for spec in args.csv:
        label, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            print(f"[warn] skip {spec}")
            continue
        rows = load(path)
        raw[label] = {
            "anon": stat(rows, anon_col, args.stat) / GB,
            "file": stat(rows, "file_cache_mb", args.stat) / GB,
            "gpu": stat(rows, "gpu_hbm_mb", args.stat) / GB,
            "dir": stat(rows, "hicache_dir_mb", args.stat) / GB,
        }
        order.append(label)

    base_anon = base_file = 0.0
    if args.baseline and args.baseline in raw:
        base_anon = raw[args.baseline]["anon"]
        base_file = raw[args.baseline]["file"]
        print(f"[baseline] subtracting '{args.baseline}': anon={base_anon:.1f}GB "
              f"file_cache={base_file:.1f}GB (SGLang runtime + model-weight page cache)")

    configs = []
    for label in order:
        r = raw[label]
        configs.append({
            "label": label,
            "anon": max(0.0, r["anon"] - base_anon),   # non-reclaimable host KV pool
            "file": max(0.0, r["file"] - base_file),   # reclaimable L3 file page cache
            "gpu": r["gpu"],
            "dir": r["dir"],                            # on-disk L3 volume
        })
    if not configs:
        print("[error] no CSVs loaded"); return

    labels = [c["label"] for c in configs]
    anon = [c["anon"] for c in configs]
    file_ = [c["file"] for c in configs]
    gpu = [c["gpu"] for c in configs]

    # console table
    print(f"\n=== host/GPU memory ({args.stat}), GB "
          f"{'[baseline-subtracted]' if args.baseline else ''} ===")
    print(f"  {'config':>24}  {'pool(NR)':>9}  {'pagecache(R)':>12}  {'host_tot':>9}  "
          f"{'GPU_HBM':>8}  {'L3_disk':>8}")
    for c in configs:
        print(f"  {c['label']:>24}  {c['anon']:>9.1f}  {c['file']:>12.1f}  "
              f"{c['anon']+c['file']:>9.1f}  {c['gpu']:>8.1f}  {c['dir']:>8.1f}")

    use_paper_style()
    fig, (axH, axG) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [3, 1]})

    import numpy as np
    x = np.arange(len(labels))
    c_nr = PALETTE["hicache"]     # non-reclaimable (blue)
    c_r = PALETTE["both"]         # reclaimable page cache (amber)
    # stacked host bars: anon (non-reclaimable) + file cache (reclaimable)
    axH.bar(x, anon, width=0.62, color=c_nr, edgecolor="white", linewidth=0.5,
            label="host KV pool (non-reclaimable anon)", zorder=3)
    axH.bar(x, file_, width=0.62, bottom=anon, color=c_r, edgecolor="white", linewidth=0.5,
            label="L3 file page cache (reclaimable)", zorder=3)
    dirs = [c["dir"] for c in configs]
    tops = [a + fv for a, fv in zip(anon, file_)]
    ymax = max(tops) if tops else 1.0
    for xi, (a, fv, dv) in enumerate(zip(anon, file_, dirs)):
        if a + fv > 0:
            axH.text(xi, a + fv + 0.01 * ymax, f"{a+fv:.0f}", ha="center", va="bottom",
                     fontsize=6.5)
        if dv > 1 and fv > 0:   # on-disk L3 volume (the disk-filling write amplification)
            axH.text(xi, a + fv / 2, f"{dv:.0f}GB\non disk", ha="center", va="center",
                     fontsize=5.8, color="white", fontweight="bold")
    axH.set_xticks(x); axH.set_xticklabels(labels, rotation=20, ha="right")
    axH.set_ylabel("host memory (GB)")
    axH.set_title("(a) Host RAM: real allocation vs reclaimable cache")
    style_axes(axH)
    axH.legend(frameon=True, fancybox=False, edgecolor=PALETTE["muted"],
               facecolor="white", loc="upper right", fontsize=6.2)

    # GPU HBM as a separate resource
    axG.bar(x, gpu, width=0.62, color=PALETTE["park"], edgecolor="white",
            linewidth=0.5, zorder=3)
    for xi, g in enumerate(gpu):
        if g > 0:
            axG.text(xi, g, f"{g:.0f}", ha="center", va="bottom", fontsize=6.5)
    axG.set_xticks(x); axG.set_xticklabels(labels, rotation=20, ha="right")
    axG.set_ylabel("GPU HBM (GB)")
    axG.set_title("(b) GPU HBM")
    style_axes(axG)

    fig.tight_layout()
    savefig(fig, args.out.rsplit(".", 1)[0])


if __name__ == "__main__":
    main()

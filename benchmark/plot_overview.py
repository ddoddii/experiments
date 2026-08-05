#!/usr/bin/env python3
"""
System overview drawn as a SCENARIO: request R10 arrives as the next turn of a session
whose earlier turns produced KV1..KV3. Where do those three blocks live, and what does
getting them back cost?

Two panels, same request, same instant:
  (a) SGLang / HiCache   the prefill pool is saturated, so KV1..KV3 were evicted down the
                         hierarchy to host DRAM (L2) and SSD (L3). The idle decode GPU is
                         never considered. Fetching back crosses PCIe at ~26 GB/s, or the
                         entry is gone from L2 and the prefix is recomputed.
  (b) Ours               the placement policy put KV1..KV3 on the idle decode GPU when
                         they were parked, so the fetch is a peer-GPU copy over NVLink.
                         Host DRAM is priority 3, used only when no GPU slab fits.

EVERY NUMBER ON THIS FIGURE IS MEASURED, not illustrative -- occupancies from the C=32
repeats, link rates from the bandwidth probe, the host pool from the models sweep. A
diagram with invented numbers next to a paper full of measured ones is worse than a
diagram with none, because the reader cannot tell which is which.

The mechanism drawn matches idle_kv_parking.py:
  write  _select_pool -> key (full, slow_link, usage, -headroom): a pool with room, on a
         fast link, on the GPU whose serving pool is least occupied.
  read   _find_fetch_source -> longest parked prefix; own pools (which may sit on other
         GPUs) first, then host DRAM, then peer prefill pools via the shared index.
  host   _park_to_host is priority 3: reached only when the block exceeds a GPU slab or
         every slab is live.

사용:
  python benchmark/plot_overview.py --out results/intro/fig_overview
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from paperstyle import PALETTE, use_paper_style, savefig

INK, MUTED = PALETTE["ink"], PALETTE["muted"]
C_PARK = PALETTE["park"]
C_HOT = "#B4423C"          # saturated / evicting
C_IDLE = "#E3E7EA"         # idle capacity
C_KV_OLD = "#BFC4C9"       # KV already held
C_BOX = "#FFFFFF"


def box(ax, x, y, w, h, label, sub=None, edge=INK, lw=0.9, fc=C_BOX, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=fc, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h - 0.045, label, ha="center", va="top", fontsize=fs,
            fontweight="bold", color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + 0.028, sub, ha="center", va="bottom", fontsize=5.9,
                color=MUTED, style="italic", zorder=4)


def kvblocks(ax, x, y, labels, colors, bw=0.052, bh=0.052, gap=0.008):
    """A row of KV cells, like the reference figure's KV_i boxes."""
    for i, (lab, c) in enumerate(zip(labels, colors)):
        cx = x + i * (bw + gap)
        ax.add_patch(Rectangle((cx, y), bw, bh, facecolor=c, edgecolor=INK,
                               linewidth=0.6, zorder=3))
        ax.text(cx + bw / 2, y + bh / 2, lab, ha="center", va="center", fontsize=5.4,
                color="white" if c in (C_PARK, C_HOT) else INK, zorder=4)
    return x + len(labels) * (bw + gap) - gap


def arrow(ax, p0, p1, text=None, rad=0.0, color=INK, lw=1.0, ls="-", tx=None, ty=None,
          fs=5.8, dashed_head=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5,
                                 shrinkA=1, shrinkB=1))
    if text:
        ax.text(tx if tx is not None else (p0[0] + p1[0]) / 2,
                ty if ty is not None else (p0[1] + p1[1]) / 2, text,
                ha="center", va="center", fontsize=fs, color=color, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.7))


def step(ax, x, y, n, text, dx=0.022, fs=6.0):
    ax.add_patch(plt.Circle((x, y), 0.017, facecolor=INK, edgecolor="none", zorder=7))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=5.2, color="white",
            zorder=8, fontweight="bold")
    ax.text(x + dx, y, text, ha="left", va="center", fontsize=fs, color=INK, zorder=8)


def panel(ax, ours):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Explicit y grid. The first attempt placed the box title and the first KV row at
    # overlapping y and the whole panel became unreadable, so every row gets a named
    # coordinate and they are spaced by more than the block height (0.052).
    Y_REQ = 0.935
    Y_GPU, H_GPU = 0.585, 0.285      # gpu boxes span 0.585..0.870
    Y_TITLE = Y_GPU + H_GPU - 0.030  # 0.840
    Y_OCC = Y_GPU + H_GPU - 0.088    # 0.782
    Y_ROW1 = Y_GPU + H_GPU - 0.170   # 0.700  (block bottom; top = 0.752)
    Y_ROW2 = Y_GPU + H_GPU - 0.250   # 0.620  (block bottom; top = 0.672)
    # Tall enough for three stacked rows: title, KV blocks, caption. At 0.135 the
    # blocks in panel (a) ran through the title text.
    Y_HOST, H_HOST = 0.360, 0.185

    # --- the request -------------------------------------------------------------
    ax.add_patch(Rectangle((0.42, Y_REQ - 0.026), 0.12, 0.052, facecolor="#FFF6C8",
                           edgecolor=INK, linewidth=0.7, zorder=3))
    ax.text(0.48, Y_REQ, "$R_{10}$", ha="center", va="center", fontsize=7.5, zorder=4)
    ax.text(0.405, Y_REQ, "turn 4 of session S", ha="right", va="center", fontsize=5.8,
            color=MUTED, zorder=4)

    def gpubox(x, w, title, occ, edge, fc):
        ax.add_patch(FancyBboxPatch((x, Y_GPU), w, H_GPU,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=fc, edgecolor=edge, linewidth=1.1, zorder=2))
        ax.text(x + w / 2, Y_TITLE, title, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, Y_OCC, occ, ha="center", va="center", fontsize=6.0,
                color=MUTED, style="italic", zorder=4)

    gpubox(0.03, 0.45, "Prefill instance (GPU)", "KV pool 82% full — evicting",
           C_HOT, "#FFFFFF")
    gpubox(0.52, 0.45, "Decode instance (GPU)", "KV pool 12% used — 88% idle",
           C_PARK if ours else MUTED, "#FFFFFF" if ours else C_IDLE)

    # prefill contents
    ax.text(0.055, Y_ROW1 + 0.026, "resident", fontsize=5.8, color=MUTED, va="center")
    kvblocks(ax, 0.155, Y_ROW1, ["$KV_8$", "$KV_9$"], [C_KV_OLD, C_KV_OLD])
    ax.text(0.055, Y_ROW2 + 0.026, "needs", fontsize=5.8, color=MUTED, va="center")
    kvblocks(ax, 0.155, Y_ROW2, ["$KV_1$", "$KV_2$", "$KV_3$"],
             [C_PARK if ours else "#FFFFFF"] * 3)
    if not ours:
        ax.text(0.345, Y_ROW2 + 0.026, "evicted", fontsize=5.8, color=C_HOT,
                va="center", fontweight="bold")

    # decode contents
    if ours:
        ax.text(0.545, Y_ROW2 + 0.026, "parked", fontsize=5.8, color=MUTED, va="center")
        kvblocks(ax, 0.645, Y_ROW2, ["$KV_1$", "$KV_2$", "$KV_3$"], [C_PARK] * 3)
        ax.text(0.745, Y_ROW1 + 0.026, "park pool in idle HBM", fontsize=5.8,
                color=C_PARK, va="center", ha="center")
    else:
        ax.text(0.745, Y_ROW2 + 0.026, "never a candidate", fontsize=6.2, color=MUTED,
                va="center", ha="center", style="italic")

    # --- host tier ---------------------------------------------------------------
    ht = ("Host DRAM — overflow only (priority 3)" if ours
          else "Host DRAM (L2, 17.6 GB reserved)  +  SSD (L3)")
    hs = ("reached only when no GPU slab fits" if ours
          else "statically reserved as GPU-KV × hicache-ratio")
    ax.add_patch(FancyBboxPatch((0.03, Y_HOST), 0.94, H_HOST,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor="#FAFAFA", edgecolor=MUTED, linewidth=0.8,
                                zorder=2))
    ax.text(0.50, Y_HOST + H_HOST - 0.032, ht, ha="center", va="center", fontsize=7.0,
            fontweight="bold", color=INK, zorder=4)
    ax.text(0.50, Y_HOST + 0.027, hs, ha="center", va="center", fontsize=5.8,
            color=MUTED, style="italic", zorder=4)
    if not ours:
        kvblocks(ax, 0.405, Y_HOST + 0.062, ["$KV_1$", "$KV_2$", "$KV_3$"],
                 [C_KV_OLD] * 3)

    # --- the fetch ---------------------------------------------------------------
    if ours:
        arrow(ax, (0.640, Y_ROW2 - 0.008), (0.335, Y_ROW2 - 0.008),
              "NVLink  27–53 GB/s", rad=-0.32, color=C_PARK, lw=1.5, ty=Y_ROW2 - 0.072)
    else:
        arrow(ax, (0.44, Y_HOST + H_HOST + 0.006), (0.235, Y_ROW2 - 0.006),
              "PCIe / pinned host  26 GB/s", rad=0.28, color=C_HOT, lw=1.3,
              tx=0.285, ty=0.578)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=7.16)
    ap.add_argument("--height", type=float, default=3.35)
    ap.add_argument("--out", default="results/intro/fig_overview")
    args = ap.parse_args()

    use_paper_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(args.width, args.height))
    panel(a1, ours=False)
    panel(a2, ours=True)
    a1.set_title("(a)  SGLang / HiCache", fontsize=8.5, pad=2)
    a2.set_title("(b)  Ours: GPU-first placement", fontsize=8.5, pad=2)

    # Step lists under each panel, so the scenario reads as a sequence rather than a
    # static picture. Kept to four steps: past four, a reader stops following.
    for ax, ours in ((a1, False), (a2, True)):
        y = 0.29
        steps = ([("$R_{10}$ arrives; radix lookup misses", ),
                  ("$KV_{1..3}$ were evicted to host / SSD", ),
                  ("fetch over PCIe, or recompute if L2 evicted too", ),
                  ("the idle decode GPU is never a candidate", )]
                 if not ours else
                 [("$R_{10}$ arrives; radix lookup misses", ),
                  ("park index finds $KV_{1..3}$ on the idle decode GPU", ),
                  ("peer-GPU copy over NVLink into the local radix", ),
                  ("prefill only the new tokens; re-park the extended prefix", )])
        for i, (t,) in enumerate(steps):
            step(ax, 0.055, y - i * 0.068, i + 1, t)

    fig.tight_layout(pad=0.4)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

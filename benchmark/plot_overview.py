#!/usr/bin/env python3
"""
System overview drawn as a SCENARIO: R10 arrives as the next turn of session S, whose
earlier turns produced KV1..KV3. Where do those blocks live, and what does getting them
back cost?

WHICH LABELS ARE MEASURED AND WHICH ARE THE SCENARIO -- the distinction matters because
the rest of the paper is measurements:

  MEASURED, from the runs:
    82% / 12% pool occupancy      C=32 repeats (cache_occupancy, 3 runs)
    NVLink 27-53 GB/s             the bandwidth probe
    priority-3 host tier          idle_kv_parking.py _park_to_host

  SCENARIO, i.e. one illustrative request and its blocks:
    KV1..KV11 and their states.

  serving   is a real, measured state: sglang:token_usage counts exactly the KV of
            requests in the current batch (max_total - available - evictable).
  waiting / completed  are NOT distinguished by the implementation. The radix cache holds
            both as `evictable` entries ordered by last_access_time; a TreeNode carries
            lock_ref and last_access_time and nothing that says "this session will come
            back". They are split here because THE DIFFERENCE IS THE POINT -- in a
            multi-turn agent workload a session pausing for a tool call will reuse its
            prefix while a finished session never will, and LRU is the only signal the
            cache has to tell them apart. Do not put a percentage on this split.

The mechanism drawn matches idle_kv_parking.py:
  write  _select_pool -> key (full, slow_link, usage, -headroom)
  read   _find_fetch_source -> longest parked prefix; own pools, then host, then peers
  host   _park_to_host is priority 3, reached only when no GPU slab fits
The park pool is drawn as a SEPARATE box inside the decode GPU because that is what it
is: an allocation from unallocated HBM beside the serving pool, not free slots inside it.

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
C_HOT = "#B4423C"
C_SERV = "#4C78A8"      # serving: in the current batch
C_WAIT = C_PARK         # waiting for the next turn: what parking exists to keep
C_DONE = "#E8EAEC"      # completed: still resident, will never be reused
C_ABSENT = "#FFFFFF"

BW, BH, GAP = 0.044, 0.045, 0.008


def kvblocks(ax, x, y, labels, colors, dashed=False):
    for i, (lab, c) in enumerate(zip(labels, colors)):
        cx = x + i * (BW + GAP)
        ax.add_patch(Rectangle((cx, y), BW, BH, facecolor=c, edgecolor=INK,
                               linewidth=0.6, zorder=3,
                               linestyle=(0, (2, 1.6)) if dashed else "-"))
        ax.text(cx + BW / 2, y + BH / 2, lab, ha="center", va="center", fontsize=4.9,
                color="white" if c in (C_SERV, C_PARK) else INK, zorder=4)
    return x + len(labels) * (BW + GAP) - GAP


def step(ax, x, y, n, r=0.017, fs=5.2):
    ax.add_patch(plt.Circle((x, y), r, facecolor=INK, edgecolor="white", lw=0.8, zorder=9))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=fs, color="white",
            zorder=10, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=5.6)
    ap.add_argument("--height", type=float, default=3.7)
    ap.add_argument("--out", default="results/intro/fig_overview")
    args = ap.parse_args()

    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(args.width, args.height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Named y grid; rows are spaced by more than the block height so a title can never be
    # drawn through a row of cells (which is what the first draft did).
    Y_REQ = 0.960
    Y_GPU, H_GPU = 0.560, 0.345
    Y_TITLE = Y_GPU + H_GPU - 0.032
    Y_OCC = Y_GPU + H_GPU - 0.080
    Y_SERV = Y_GPU + H_GPU - 0.165
    Y_PARK = Y_SERV - 0.135                      # park sub-box bottom
    Y_HOST, H_HOST = 0.395, 0.100
    XP, XD, W = 0.020, 0.515, 0.465

    # --- request ------------------------------------------------------------------
    ax.add_patch(Rectangle((0.085, Y_REQ - 0.023), 0.10, 0.046, facecolor="#FFF6C8",
                           edgecolor=INK, linewidth=0.7, zorder=3))
    ax.text(0.135, Y_REQ, "$R_{10}$", ha="center", va="center", fontsize=7, zorder=4)
    ax.text(0.198, Y_REQ, "next turn of session S", ha="left", va="center", fontsize=5.8,
            color=MUTED, zorder=4)
    ax.add_patch(FancyArrowPatch((0.135, Y_REQ - 0.025), (0.135, Y_GPU + H_GPU + 0.004),
                                 arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.9,
                                 zorder=5))
    step(ax, 0.175, (Y_REQ - 0.025 + Y_GPU + H_GPU) / 2, 1)

    def gpubox(x, title, occ, edge):
        ax.add_patch(FancyBboxPatch((x, Y_GPU), W, H_GPU,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor="#FFFFFF", edgecolor=edge, linewidth=1.1,
                                    zorder=2))
        ax.text(x + W / 2, Y_TITLE, title, ha="center", va="center", fontsize=7,
                fontweight="bold", color=INK, zorder=4)
        ax.text(x + W / 2, Y_OCC, occ, ha="center", va="center", fontsize=5.6,
                color=MUTED, style="italic", zorder=4)

    gpubox(XP, "Prefill instance (GPU)", "serving KV pool 82% full — evicting", C_HOT)
    gpubox(XD, "Decode instance (GPU)", "serving KV pool 12% used — 88% idle", C_PARK)

    # --- prefill contents ---------------------------------------------------------
    ax.text(XP + 0.020, Y_SERV + 0.022, "serving", fontsize=5.6, color=MUTED, va="center")
    kvblocks(ax, XP + 0.105, Y_SERV, ["$KV_8$", "$KV_9$"], [C_SERV] * 2)
    ax.text(XP + 0.020, Y_SERV - 0.078, "needs", fontsize=5.6, color=MUTED, va="center")
    xe = kvblocks(ax, XP + 0.105, Y_SERV - 0.100, ["$KV_1$", "$KV_2$", "$KV_3$"],
                  [C_ABSENT] * 3, dashed=True)
    ax.text(xe + 0.012, Y_SERV - 0.078, "evicted", fontsize=5.6, color=C_HOT,
            va="center", fontweight="bold")

    # --- decode contents ----------------------------------------------------------
    ax.text(XD + 0.020, Y_SERV + 0.022, "serving", fontsize=5.6, color=MUTED, va="center")
    kvblocks(ax, XD + 0.105, Y_SERV, ["$KV_{11}$"], [C_SERV])
    # The park pool is its own allocation from unallocated HBM, beside the serving pool --
    # not free slots inside it. Nested box for exactly that reason.
    ax.add_patch(FancyBboxPatch((XD + 0.014, Y_PARK), W - 0.028, 0.115,
                                boxstyle="round,pad=0.003,rounding_size=0.008",
                                facecolor="#F2FAF7", edgecolor=C_PARK, linewidth=0.8,
                                linestyle=(0, (3, 2)), zorder=2))
    ax.text(XD + 0.024, Y_PARK + 0.092, "park pool (idle HBM)", fontsize=5.6,
            color=C_PARK, va="center", fontweight="bold", zorder=4)
    kvblocks(ax, XD + 0.024, Y_PARK + 0.020, ["$KV_1$", "$KV_2$", "$KV_3$"], [C_WAIT] * 3)
    kvblocks(ax, XD + 0.024 + 3 * (BW + GAP) + 0.022, Y_PARK + 0.020,
             ["$KV_4$", "$KV_7$"], [C_DONE] * 2)

    # --- host tier -----------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((XP, Y_HOST), 0.96, H_HOST,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor="#FAFAFA", edgecolor=MUTED, linewidth=0.8,
                                zorder=2))
    ax.text(0.50, Y_HOST + H_HOST - 0.030, "Host DRAM — overflow only (priority 3)",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK, zorder=4)
    ax.text(0.50, Y_HOST + 0.024, "reached only when a block exceeds a GPU slab, or "
            "every slab is live", ha="center", va="center", fontsize=5.5,
            color=MUTED, style="italic", zorder=4)

    # --- the fetch ------------------------------------------------------------------
    ax.add_patch(FancyArrowPatch((XD + 0.020, Y_PARK + 0.042),
                                 (XP + 0.105 + 3 * (BW + GAP) - GAP + 0.004,
                                  Y_SERV - 0.078),
                                 arrowstyle="-|>", mutation_scale=8, color=C_PARK,
                                 lw=1.5, connectionstyle="arc3,rad=-0.22", zorder=6))
    ax.text(0.505, Y_PARK - 0.030, "peer-GPU copy over NVLink, 27–53 GB/s",
            ha="center", va="center", fontsize=5.8, color=C_PARK, zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    step(ax, XD + 0.006, Y_PARK + 0.042, 2)
    step(ax, 0.400, Y_PARK + 0.006, 3)

    # --- legend ----------------------------------------------------------------------
    ly = 0.318
    for i, (lab, c, dashed) in enumerate((
            ("serving", C_SERV, False),
            ("waiting — session paused between turns", C_WAIT, False),
            ("completed", C_DONE, False),
            ("evicted / absent", C_ABSENT, True))):
        lx = [0.025, 0.145, 0.560, 0.700][i]
        ax.add_patch(Rectangle((lx, ly - 0.013), 0.028, 0.026, facecolor=c,
                               edgecolor=INK, linewidth=0.6,
                               linestyle=(0, (2, 1.6)) if dashed else "-", zorder=3))
        ax.text(lx + 0.036, ly, lab, fontsize=5.7, va="center", color=INK)
    ax.text(0.025, ly - 0.040,
            "the cache cannot tell waiting from completed — both are LRU-evictable; "
            "that is what parking is for",
            fontsize=5.3, va="center", color=MUTED, style="italic")

    # --- steps -------------------------------------------------------------------------
    for i, t in enumerate([
            "$R_{10}$ arrives; the local radix misses — its prefix was evicted under pressure",
            "the park index finds $KV_{1..3}$ on the idle decode GPU",
            "peer-GPU copy back into the prefill KV pool and radix",
            "prefill only the new tokens; re-park the extended prefix"]):
        step(ax, 0.033, 0.195 - i * 0.053, i + 1)
        ax.text(0.060, 0.195 - i * 0.053, t, fontsize=6.0, va="center", color=INK)

    fig.tight_layout(pad=0.3)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Actual HBM occupancy per role, from nvidia-smi -- "how full is the card", regardless of
whether the bytes hold live KV, cached KV, or nothing.

READ THIS BEFORE USING IT AS THE MOTIVATION FIGURE. HBM occupancy is nearly a constant:
prefill sits at 92.8% and decode at 84.2%, and over a 950 s run the decode GPUs move by
0.09 GiB. That is not a measurement artefact -- --mem-fraction-static reserves the KV
pool at startup, so the card is full from the first second whether the pool holds
anything or not. The line is flat because the allocation is static.

So the two views answer different questions and neither replaces the other:

  --mode hbm        how much of the card is committed.        92.8% vs 84.2%, flat.
                    Answers "can I fit another model here": no.
                    Does NOT show the imbalance, because a reserved-but-empty pool
                    looks exactly like a full one.

  --mode breakdown  what the committed bytes are DOING.       weights / KV holding live
                    data / KV reserved but empty / free. This is the same 92.8% split
                    into the part that is working and the part that is only reserved --
                    which is where the imbalance actually lives.

The KV-pool-occupancy figure (plot_kv_occupancy_2panel.py) is the third view: the
fraction of the reserved pool that holds KV, 82% against 11%.

사용:
  python benchmark/plot_hbm_occupancy.py --csv results/exp2/repeat_nocap_c32_r2/mem_park_local.csv \
      --occ results/exp2/repeat_nocap_c32_r2/occ_park_local.csv --mode breakdown \
      --out results/fig0/fig_hbm
"""
import argparse
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paperstyle import PALETTE, use_paper_style, style_axes, savefig

INK, MUTED = PALETTE["ink"], PALETTE["muted"]
C_P, C_D = "#0072B2", "#E69F00"
C_W = "#7A7F85"        # model weights
C_LIVE = PALETTE["park"]
C_RESV = "#E3E7EA"     # reserved KV pool holding nothing
C_FREE = "#FFFFFF"


def hbm_series(path, gpus):
    rows = list(csv.DictReader(open(path)))
    t = [float(r.get("elapsed_s", "") or 0) for r in rows]
    per = {g: [float(r.get(f"gpu{g}_hbm_mb", "") or 0) / 1024.0 for r in rows]
           for g in gpus}
    return t, per


def occ_mean(path, labels):
    rows = [r for r in csv.DictReader(
        (l for l in open(path) if not l.startswith("#")))]
    out = {}
    for lab in labels:
        u = [float(r[f"{lab}_use"]) for r in rows if r.get(f"{lab}_use")]
        c = [float(r[f"{lab}_cap_tok"]) for r in rows if r.get(f"{lab}_cap_tok")]
        if u and c:
            out[lab] = (st.mean(u), st.mean(c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="mem_*.csv from sys_mem_breakdown.py")
    ap.add_argument("--occ", help="occ_*.csv, needed for --mode breakdown")
    ap.add_argument("--mode", choices=("hbm", "breakdown"), default="hbm")
    ap.add_argument("--prefill-gpus", nargs="+", type=int, default=[0, 2])
    ap.add_argument("--decode-gpus", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--card-gb", type=float, default=48.0)
    ap.add_argument("--weights-gb", type=float, default=15.06,
                    help="model weights per GPU, from the server log's 'Load weight end'")
    ap.add_argument("--kib-per-token", type=float, default=128)
    ap.add_argument("--width", type=float, default=6.0)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--out", default="results/fig0/fig_hbm")
    args = ap.parse_args()

    gpus = args.prefill_gpus + args.decode_gpus
    t, per = hbm_series(args.csv, gpus)
    role = {g: ("P" if g in args.prefill_gpus else "D") for g in gpus}
    for g in gpus:
        v = per[g]
        print(f"  GPU{g} ({role[g]})  HBM mean {st.mean(v):5.1f} GiB "
              f"= {100 * st.mean(v) / args.card_gb:4.1f}% of the card   "
              f"range {max(v) - min(v):.2f} GiB over the run")

    use_paper_style()

    if args.mode == "hbm":
        fig, ax = plt.subplots(1, 1, figsize=(args.width, args.height))
        for label, gs, c in (("Prefill", args.prefill_gpus, C_P),
                             ("Decode", args.decode_gpus, C_D)):
            mean = [100 * sum(per[g][i] for g in gs) / len(gs) / args.card_gb
                    for i in range(len(t))]
            ax.plot(t, mean, color=c, lw=1.6, label=f"{label}  "
                    f"(avg {st.mean(mean):.1f}%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(min(t), max(t))
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"GPU HBM in use\n(% of {args.card_gb:.0f} GB card)")
        ax.legend(loc="lower right", frameon=False, fontsize=6.5, handlelength=1.6)
        # State the flatness on the figure. A reader seeing two near-constant lines will
        # otherwise assume the sampler broke.
        ax.text(0.5, 0.06, "reserved at startup by --mem-fraction-static: "
                "the card is full whether the pool holds KV or not",
                transform=ax.transAxes, ha="center", fontsize=5.8, color=MUTED,
                style="italic")
        style_axes(ax)
    else:
        if not args.occ:
            raise SystemExit("--mode breakdown needs --occ")
        occ = occ_mean(args.occ, ["P0", "P1", "D0", "D1"])
        lab_of = {args.prefill_gpus[0]: "P0", args.prefill_gpus[1]: "P1",
                  args.decode_gpus[0]: "D0", args.decode_gpus[1]: "D1"}
        fig, ax = plt.subplots(1, 1, figsize=(args.width * 0.75, args.height * 1.15))
        order = sorted(gpus)
        for i, g in enumerate(order):
            hbm = st.mean(per[g])
            u, cap = occ.get(lab_of[g], (0.0, 0.0))
            pool = cap * args.kib_per_token * 1024 / 1024 ** 3
            live = pool * u
            resv = max(0.0, pool - live)
            other = max(0.0, hbm - args.weights_gb - pool)   # activations, ctx, park
            free = max(0.0, args.card_gb - hbm)
            b = 0
            for h, c in ((args.weights_gb, C_W), (live, C_LIVE), (resv, C_RESV),
                         (other, "#CFD4D8"), (free, C_FREE)):
                ax.bar(i, h, bottom=b, width=0.6, color=c, edgecolor=INK, lw=0.4,
                       zorder=3)
                b += h
            ax.text(i, args.weights_gb + live + resv / 2,
                    f"{resv:.0f}", ha="center", va="center", fontsize=6, color=MUTED,
                    zorder=4)
            print(f"  GPU{g}: weights {args.weights_gb:.1f} + KV live {live:.1f} + "
                  f"KV reserved-empty {resv:.1f} + other {other:.1f} + free {free:.1f}")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"GPU{g}\n({'Prefill' if role[g] == 'P' else 'Decode'})"
                            for g in order], fontsize=7)
        ax.set_ylabel("GPU HBM (GB)")
        ax.set_ylim(0, args.card_gb)
        ax.legend(handles=[Patch(facecolor=C_W, edgecolor=INK, lw=0.4, label="model weights"),
                           Patch(facecolor=C_LIVE, edgecolor=INK, lw=0.4, label="KV pool: holding KV"),
                           Patch(facecolor=C_RESV, edgecolor=INK, lw=0.4, label="KV pool: reserved, empty"),
                           Patch(facecolor="#CFD4D8", edgecolor=INK, lw=0.4, label="activations, CUDA ctx"),
                           Patch(facecolor=C_FREE, edgecolor=INK, lw=0.4, label="unallocated")],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                  fontsize=6, handlelength=1.2, columnspacing=1.1, borderaxespad=0.3)
        style_axes(ax)

    fig.tight_layout(pad=0.4)
    stem = args.out[:-4] if args.out.endswith((".png", ".pdf")) else args.out
    savefig(fig, stem)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
System overview (architecture) — paper-quality vector schematic, SVG -> PDF + PNG.

Three panels, read left to right:

  (a) Today (HiCache)   the same session prefix KV_1..KV_3 ends up in host DRAM,
                        because there is no runtime placement at all: the host tier
                        is a boot-time reservation (--hicache-ratio) that does not
                        move with load, and the idle decode HBM is unreachable.
  (b) Ours              the SAME blocks, parked in transiently-idle peer HBM. The
                        host tier is drawn EMPTY on purpose -- that emptiness is the
                        result (MemAvailable 105 GB vs 44 GB), not an unfinished box.
  (c) Placement policy  the priority ladder with the MEASURED bandwidth beside each
                        rung, including the rung that is excluded: a non-NVLink peer
                        at 3.3 GB/s loses to host DRAM at 26.3 GB/s. That single row
                        is the argument for ranking by measurement instead of by a
                        fixed "GPU before CPU" tier order.

(a) and (b) use identical geometry and identical block identities so the eye reads
one difference: which box KV_1..KV_3 sit in. Everything else is held constant.

WHICH LABELS ARE MEASURED AND WHICH ARE THE SCENARIO -- the distinction matters
because the rest of the paper is measurements:

  MEASURED, from the runs and probes:
    prefill 83% = 1% live + 81% cache   exp2/final_nocap_c32 per_gpu, C=32
    decode  14% = 14% live +  0% cache  same. Decode holds no evictable cache AT ALL,
                                        which is why it is the natural victim store --
                                        do not restate its 14% as "serving pool full".
    52.8 GB/s peer, 26.3 GB/s host      nvlink_xproc_microbench.json (537 MB, IPC
                                        checksum MATCH) and nvlink_microbench.json.
                                        Host costs 4.0x end to end, not 2x: the round
                                        trip crosses PCIe twice (park + fetch), and
                                        speedup_vs_host is 4.0 at every size measured.
                                        Do NOT quote "27-53 GB/s" -- the 27.2 in the
                                        design doc is the OTHER DIRECTION on the same
                                        bridge in the VMM probe, unexplained and 2x off
                                        the dedicated microbench on the real park path.
    3.3 GB/s non-NVLink peer            vmm_probe.py --all-pairs-bw
    61 GB host reservation              results/mem/bd_*.csv, invariant across policy
    MemAvailable 44 -> 105 GB           same
    1203 ms re-prefill @ 8k             fig_ttft_ctx_sweep.json

  NOTE, before reusing these occupancy numbers elsewhere: paper/evaluation.md quotes
  the OPPOSITE asymmetry (decode 74.6% vs prefill 44.6%) from results/kv_ts/2p2d_p20k,
  where BOTH pools are capped at 20k tokens. Which role is the idle one is set by pool
  sizing, not by the workload. This figure follows the uncapped C=32 runs.
  SCENARIO, i.e. one illustrative request and its blocks:
    KV_1..KV_11 and their states.

  serving   is a real, measured state: sglang:token_usage counts exactly the KV of
            requests in the current batch (max_total - available - evictable).
  waiting / completed  are NOT distinguished by the implementation -- the radix cache
            holds both as evictable entries ordered by last_access_time. They are
            split here because THE DIFFERENCE IS THE POINT: a session paused for a
            tool call will reuse its prefix, a finished one never will, and LRU is
            the only signal the cache has. Do not put a percentage on this split.

사용:
  python benchmark/make_fig_overview_arch.py --out results/intro/fig_overview_arch
"""
import argparse
import os

FONT = "Times New Roman, Times, serif"

# palette — matches make_fig_design.py / paperstyle.py
BLUE, BLUE_F = "#2F6DB0", "#DCE7F5"      # serving (in the current batch)
GREEN, GREEN_F = "#009E73", "#D5F0E6"    # parked / waiting — this work
RED, RED_F = "#B4423C", "#F6DEDE"        # host tier / slow / excluded
AMBER = "#E69F00"
DONE_F = "#E8EAEC"                        # completed, resident, never reused
INK, MUTED, HAIR = "#111111", "#555555", "#9A9A9A"

W, H = 1180, 556

# --- layout grid -------------------------------------------------------------
GUT = 92                                  # band labels are right-aligned here
PA_X, PA_W = 104, 296                     # panel (a)
PB_X, PB_W = 424, 360                     # panel (b)
PC_X, PC_W = 816, 340                     # panel (c)

CB_Y, CB_H = 40, 78                       # control band
DIV_Y = 129                               # control / hardware divider
T1_Y, T2_Y, T3_Y = 140, 252, 364          # tier rows; 34 px gaps carry the arrows
TH, TH3 = 78, 64                          # tier heights (host row is shorter)
STEP_Y = 452                              # numbered walk-through
LEG_Y = 528


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=10, col=INK, w="normal", anchor="middle", ital=False):
    it = 'font-style="italic"' if ital else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{col}" '
            f'font-weight="{w}" text-anchor="{anchor}" {it}>{esc(s)}</text>')


def sub(x, y, base, idx, size=10, col=INK, w="normal", anchor="middle"):
    """KV with a real subscript. tspan dy is portable; Unicode subscript glyphs are
    not present in Times and would render as boxes."""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{col}" '
            f'font-weight="{w}" text-anchor="{anchor}">{esc(base)}'
            f'<tspan font-size="{size*0.70:.1f}" dy="{size*0.22:.1f}">{esc(idx)}</tspan></text>')


def rrect(x, y, w, h, fill, stroke, sw=1.3, rx=6, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {d}/>')


# --- KV block ----------------------------------------------------------------
CW, CH, CG = 32, 22, 5                    # cell width / height / gap
KIND = {                                  # fill, stroke, text, dash
    "serving":  (BLUE,   BLUE,  "white", ""),
    "waiting":  (GREEN,  GREEN, "white", ""),
    "done":     (DONE_F, HAIR,  INK,     ""),
    "absent":   ("white", INK,  MUTED,   "2.4 1.8"),
}


def kvrow(x, y, items):
    """items: list of (index string, kind). Returns (svg, right edge)."""
    out = []
    for i, (idx, kind) in enumerate(items):
        f, st, tc, dash = KIND[kind]
        cx = x + i * (CW + CG)
        d = f'stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<rect x="{cx}" y="{y}" width="{CW}" height="{CH}" rx="2.5" fill="{f}" '
                   f'stroke="{st}" stroke-width="1" {d}/>')
        out.append(sub(cx + CW / 2, y + CH / 2 + 3.4, "KV", idx, 9, tc))
    return "".join(out), x + len(items) * (CW + CG) - CG


def hatch(x, y, w, h, label=None, lcol=MUTED):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2.5" fill="url(#hatch)" '
           f'stroke="{HAIR}" stroke-width="0.9" stroke-dasharray="2.4 1.8"/>']
    if label:
        out.append(f'<rect x="{x+3}" y="{y+h/2-6}" width="{w-6}" height="12" fill="white" '
                   f'opacity="0.86"/>')
        out.append(txt(x + w / 2, y + h / 2 + 3.2, label, 7.6, lcol, "normal", "middle", True))
    return "".join(out)


def circnum(x, y, n, col=INK, r=8.2, fs=9.5):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}" stroke="white" stroke-width="1"/>'
            + txt(x, y + 3.4, str(n), fs, "white", "bold"))


def build():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="white"/>', '<defs>']
    for name, col in (("aMut", MUTED), ("aGrn", GREEN), ("aRed", RED), ("aInk", INK)):
        s.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="8.4" refY="5" '
                 f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 f'<path d="M0 0 L10 5 L0 10 z" fill="{col}"/></marker>')
    s.append('<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
             'patternTransform="rotate(45)">'
             '<line x1="0" y1="0" x2="0" y2="6" stroke="#CFCFCF" stroke-width="1.5"/></pattern>')
    s.append('</defs>')

    def arrow(x1, y1, x2, y2, col, mk, sw=1.8, dash=""):
        d = f'stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{col}" '
                 f'stroke-width="{sw}" {d} marker-end="url(#{mk})"/>')

    def wtext(x, y, t, size, col, w="normal", anchor="end", ital=False):
        """text on a white plate — used where a label sits on a rule or an arrow lane."""
        wid = len(t) * size * 0.46
        x0 = x - wid if anchor == "end" else (x - wid / 2 if anchor == "middle" else x)
        s.append(f'<rect x="{x0-3}" y="{y-size*0.82}" width="{wid+6}" height="{size*1.15}" '
                 f'fill="white"/>')
        s.append(txt(x, y, t, size, col, w, anchor, ital))

    # ============================ panel titles ================================
    s.append(txt(PA_X + PA_W / 2, 25, "(a) Today — HiCache", 12.5, INK, "bold"))
    s.append(txt(PB_X + PB_W / 2, 25, "(b) Ours — KV-aware unified memory", 12.5, INK, "bold"))
    s.append(txt(PC_X + PC_W / 2, 25, "(c) Placement policy", 12.5, INK, "bold"))

    # ============================ band labels =================================
    s.append(f'<line x1="20" y1="{DIV_Y}" x2="800" y2="{DIV_Y}" stroke="{HAIR}" '
             f'stroke-width="1.1" stroke-dasharray="4 3"/>')
    s.append(txt(GUT, CB_Y + CB_H / 2 - 4, "Control", 10.5, MUTED, "bold", "end", True))
    s.append(txt(GUT, CB_Y + CB_H / 2 + 9, "plane", 10.5, MUTED, "bold", "end", True))
    s.append(txt(GUT, T2_Y + 30, "Hardware", 10.5, MUTED, "bold", "end", True))
    s.append(txt(GUT, T2_Y + 45, "unified KV", 8.6, GREEN, "bold", "end", True))
    s.append(txt(GUT, T2_Y + 56, "memory space", 8.6, GREEN, "bold", "end", True))

    # ============================ (a) control band ============================
    s.append(rrect(PA_X, CB_Y, PA_W, CB_H, "#FAFAFA", HAIR, 1.1, 6, "4 3"))
    s.append(txt(PA_X + PA_W / 2, CB_Y + 20, "no runtime placement", 11, MUTED, "bold"))
    for i, t in enumerate(["host tier sized once at boot (--hicache-ratio)",
                           "61 GB held whether the cache is 5% or 95% full;",
                           "write policy and storage backend do not change it"]):
        s.append(txt(PA_X + PA_W / 2, CB_Y + 36 + i * 12, t, 8.4, MUTED, "normal", "middle", True))

    # ============================ (b) control band ============================
    s.append(rrect(PB_X, CB_Y, PB_W, CB_H, "#F6FBF9", GREEN, 1.5))
    s.append(circnum(PB_X + 16, CB_Y + 16, 1))
    s.append(rrect(PB_X + 28, CB_Y + 7, 34, 17, "#FFF6C8", INK, 0.9, 3))
    s.append(sub(PB_X + 45, CB_Y + 20, "R", "10", 10.5, INK))
    s.append(txt(PB_X + 68, CB_Y + 20, "next turn of session S — local radix misses",
                 8.6, INK, "normal", "start"))

    s.append(txt(PB_X + 10, CB_Y + 46, "KV Manager", 9.4, GREEN, "bold", "start"))
    s.append(rrect(PB_X + 96, CB_Y + 28, 254, 14, "white", HAIR, 0.8, 2))
    s.append(txt(PB_X + 102, CB_Y + 38, "Resource Monitor — per-GPU pressure and headroom",
                 8.0, INK, "normal", "start"))
    s.append(rrect(PB_X + 96, CB_Y + 45, 254, 14, "white", HAIR, 0.8, 2))
    s.append(txt(PB_X + 102, CB_Y + 55, "Destination Selection — ranked by measured link BW",
                 8.0, INK, "normal", "start"))
    s.append(txt(PB_X + PB_W / 2, CB_Y + 70,
                 "/dev/shm + flock — shared by all four serving processes, no daemon",
                 8.2, MUTED, "normal", "middle", True))

    # control <-> hardware coupling (crosses the divider, as in CachedAttention Fig. 5)
    arrow(PB_X + 60, CB_Y + CB_H + 2, PB_X + 60, T1_Y - 2, GREEN, "aGrn", 1.6, "4 2.5")
    wtext(PB_X + 84, DIV_Y + 3, "place() / fetch()", 8.4, GREEN, "bold", "start")
    s.append(circnum(PB_X + 72, DIV_Y, 2, GREEN, 7.4, 8.6))
    arrow(PB_X + 300, T1_Y - 2, PB_X + 300, CB_Y + CB_H + 2, MUTED, "aMut", 1.4, "4 2.5")
    wtext(PB_X + 293, DIV_Y + 3, "pressure, residency", 8.4, MUTED, "normal", "end")

    # ============================ tier boxes ==================================
    def tier(px, pw, y, h, name, occ, edge, fill, hero=False, right=None, rcol=INK):
        s.append(rrect(px, y, pw, h, fill, edge, 1.9 if hero else 1.3))
        s.append(txt(px + 11, y + 17, name, 10.8, INK, "bold", "start"))
        s.append(txt(px + 11, y + 29, occ, 8.4, MUTED, "normal", "start", True))
        if right:                          # headline number, kept out of the block rows
            s.append(txt(px + pw - 11, y + 29, right, 9.2, rcol, "bold", "end"))

    BY = lambda y: y + 38                 # first block row inside a tier

    # ---- (a) prefill / decode / host ----
    tier(PA_X, PA_W, T1_Y, TH, "Prefill instance (GPU)",
         "KV pool 83% full — but only 1% live; 81% is evictable cache", RED, "white")
    g, _ = kvrow(PA_X + 11, BY(T1_Y), [("8", "serving"), ("9", "serving")])
    s.append(g)
    g, xe = kvrow(PA_X + 11 + 2 * (CW + CG) + 10, BY(T1_Y),
                  [("1", "absent"), ("2", "absent"), ("3", "absent")])
    s.append(g)
    s.append(txt(PA_X + 11, BY(T1_Y) + CH + 12, "session S prefix — evicted under pressure",
                 8.0, RED, "normal", "start", True))

    tier(PA_X, PA_W, T2_Y, TH, "Decode instance (GPU)",
         "14% used, all of it live — holds no cache, 86% free", HAIR, "white")
    g, _ = kvrow(PA_X + 11, BY(T2_Y), [("11", "serving")])
    s.append(g)
    s.append(hatch(PA_X + 11 + (CW + CG) + 10, BY(T2_Y), 200, CH,
                   "idle HBM — no interface reaches it"))
    s.append(txt(PA_X + 11, BY(T2_Y) + CH + 12, "capacity already paid for, and left unused",
                 8.0, MUTED, "normal", "start", True))

    tier(PA_X, PA_W, T3_Y, TH3, "Host DRAM — HiCache", "61 GB reserved at boot", RED, RED_F,
         right="MemAvailable 44 GB", rcol=RED)
    g, _ = kvrow(PA_X + 11, BY(T3_Y) - 4,
                 [("1", "waiting"), ("2", "waiting"), ("3", "waiting"), ("4", "done")])
    s.append(g)
    s.append(hatch(PA_X + 11 + 4 * (CW + CG) + 6, BY(T3_Y) - 4, 118, CH, "reserved, unused"))

    # ---- (b) prefill / decode / host ----
    tier(PB_X, PB_W, T1_Y, TH, "Prefill instance (GPU)",
         "KV pool 83% full — but only 1% live; 81% is evictable cache", RED, "white")
    g, _ = kvrow(PB_X + 11, BY(T1_Y), [("8", "serving"), ("9", "serving")])
    s.append(g)
    g, _ = kvrow(PB_X + 11 + 2 * (CW + CG) + 10, BY(T1_Y),
                 [("1", "absent"), ("2", "absent"), ("3", "absent")])
    s.append(g)
    s.append(txt(PB_X + 11, BY(T1_Y) + CH + 12,
                 "prefill only the new tokens, then re-park the extended prefix",
                 8.0, GREEN, "normal", "start", True))

    tier(PB_X, PB_W, T2_Y, TH, "Decode instance (GPU)",
         "14% used, all of it live — holds no cache, 86% free", GREEN, "white", hero=True)
    g, _ = kvrow(PB_X + 11, BY(T2_Y), [("11", "serving")])
    s.append(g)
    px0 = PB_X + 11 + (CW + CG) + 10
    s.append(rrect(px0, T2_Y + 30, PB_W - (px0 - PB_X) - 11, TH - 38, GREEN_F, GREEN, 1.1, 4,
                   "3.5 2.5"))
    s.append(txt(px0 + 7, T2_Y + 41, "Victim Cache Pool — transiently-idle HBM", 8.2, GREEN,
                 "bold", "start"))
    g, _ = kvrow(px0 + 7, T2_Y + 47,
                 [("1", "waiting"), ("2", "waiting"), ("3", "waiting"), ("4", "done")])
    s.append(g)

    tier(PB_X, PB_W, T3_Y, TH3, "Host DRAM", "empty — nothing parked here", MUTED, "white",
         right="MemAvailable 105 GB", rcol=GREEN)
    s.append(hatch(PB_X + 11, BY(T3_Y) - 4, 178, CH, "left to the agent stack", GREEN))

    MID_A = (T1_Y + TH + T2_Y) / 2         # upper gap centre
    MID_B = (T2_Y + TH + T3_Y) / 2         # lower gap centre

    # ---- inter-tier arrows: (a) ----
    ax = PA_X + PA_W - 26
    s.append(f'<line x1="{ax}" y1="{T1_Y+TH+5}" x2="{ax}" y2="{T2_Y-5}" stroke="{HAIR}" '
             f'stroke-width="1.4" stroke-dasharray="3 2.5"/>')
    s.append(f'<path d="M {ax-7} {MID_A-7} L {ax+7} {MID_A+7} M {ax+7} {MID_A-7} '
             f'L {ax-7} {MID_A+7}" stroke="{RED}" stroke-width="2.4" stroke-linecap="round"/>')
    wtext(ax - 14, MID_A + 3, "no path to peer HBM", 8.2, RED, "normal")
    arrow(ax, T2_Y + TH + 5, ax, T3_Y - 5, RED, "aRed", 1.9)
    wtext(ax - 14, MID_B + 3, "evict → host, always", 8.2, RED, "bold")

    # ---- inter-tier arrows: (b) ----
    bx = PB_X + PB_W - 54                  # fetch lane; re-park lane sits 22 px right
    arrow(bx, T2_Y - 5, bx, T1_Y + TH + 5, GREEN, "aGrn", 2.2)
    arrow(bx + 22, T1_Y + TH + 5, bx + 22, T2_Y - 5, GREEN, "aGrn", 1.5, "3.5 2.5")
    s.append(circnum(bx - 12, MID_A, 3, GREEN, 7.4, 8.6))
    s.append(circnum(bx + 34, MID_A, 4, GREEN, 7.4, 8.6))
    wtext(bx - 24, MID_A + 3, "fetch / re-park — peer copy, 52.8 GB/s = 4.0× host", 8.2, GREEN, "bold")
    arrow(bx + 11, T2_Y + TH + 5, bx + 11, T3_Y - 5, RED, "aRed", 1.4, "3 2.5")
    wtext(bx - 3, MID_B + 3, "priority 3 — only when no GPU slab fits", 8.2, RED, "normal")

    # ============================ (c) the ladder ==============================
    s.append(txt(PC_X + PC_W / 2, 42, "ranked by bandwidth measured at boot — not by tier",
                 8.6, MUTED, "normal", "middle", True))
    rungs = [
        ("1", "idlest NVLink peer GPU HBM", "headroom ≥ size; 4.0× the host round trip",
         "52.8 GB/s", GREEN, GREEN_F, GREEN, False),
        ("2", "any GPU whose link beats host", "topology-ranked, re-measured per node",
         "", GREEN, "white", GREEN, False),
        ("✗", "non-NVLink peer GPU", "excluded — slower than host DRAM",
         "3.3 GB/s", RED, "#FAFAFA", HAIR, True),
        ("3", "host DRAM overflow", "last resort — and the host path pays PCIe twice",
         "26.3 GB/s ×2", AMBER, RED_F, AMBER, False),
        ("4", "drop → recompute", "victim cache: correctness never depends on it",
         "1203 ms @ 8k", MUTED, "white", HAIR, False),
    ]
    ry, rh, rg = 56, 46, 8
    for n, name, why, val, ncol, fill, edge, dim in rungs:
        s.append(rrect(PC_X, ry, PC_W, rh, fill, edge, 1.6 if n == "1" else 1.1))
        if n == "✗":
            s.append(txt(PC_X + 20, ry + rh / 2 + 5, "✗", 14, RED, "bold"))
        else:
            s.append(circnum(PC_X + 20, ry + rh / 2, n, ncol))
        s.append(txt(PC_X + 38, ry + 19, name, 10.4, MUTED if dim else INK, "bold", "start"))
        s.append(txt(PC_X + 38, ry + 33, why, 8.3, MUTED, "normal", "start", True))
        if val:
            s.append(txt(PC_X + PC_W - 11, ry + 19, val, 9.6, ncol, "bold", "end"))
        ry += rh + rg

    s.append(rrect(PC_X, ry + 4, PC_W, 74, "#FAFAFA", HAIR, 1.0, 6, "4 3"))
    s.append(txt(PC_X + 11, ry + 21, "serving always wins", 9.6, INK, "bold", "start"))
    for i, t in enumerate([
            "on pressure, the lowest-reuse parked KV moves GPU→GPU,",
            "then →host, then is dropped (§3.3). Parked KV is a victim",
            "cache: the consistency fallback is recompute, so borrowed",
            "memory can always be handed back."]):
        s.append(txt(PC_X + 11, ry + 34 + i * 11, t, 8.3, MUTED, "normal", "start"))

    # the question every reader of (c) asks next -- answered in the space under the ladder
    uy = ry + 86
    s.append(rrect(PC_X, uy, PC_W, 96, "white", HAIR, 1.0))
    s.append(txt(PC_X + 11, uy + 17, "Why not CUDA Unified Memory? (§2.2)", 9.6, INK,
                 "bold", "start"))
    # leading spaces collapse in SVG, so continuation lines carry an explicit x offset
    for i, (ind, t) in enumerate([
            (0, "• eviction has exactly one destination — host. UM grows"),
            (7, "the very commitment this design sets out to reduce."),
            (0, "• it cannot cross process boundaries, and 2P+2D is four"),
            (7, "separate processes."),
            (0, "• it cannot discard: managed pages must be preserved, but"),
            (7, "KV may be dropped because recompute is always correct.")]):
        s.append(txt(PC_X + 11 + ind, uy + 31 + i * 11, t, 8.3, MUTED, "normal", "start"))

    # ============================ walk-through ================================
    for i, (n, t) in enumerate([
            (1, "The next turn of session S arrives; the prefill node's local radix misses "
                "— its prefix was evicted under pressure while the session was in a tool call."),
            (2, "The control plane resolves the prefix hash: the park index says loc = GPU2, "
                "and the link table ranks that copy above host DRAM."),
            (3, "Bulk peer-GPU copy GPU2 → GPU0, inserted into the radix as an ordinary "
                "prefix hit. No page faults, no host round trip."),
            (4, "Prefill computes only the new tokens and re-parks the extended prefix on "
                "whichever GPU is idlest then — host DRAM is never touched.")]):
        y = STEP_Y + i * 17
        s.append(circnum(PA_X + 8, y - 3, n, GREEN, 7.4, 8.6))
        s.append(txt(PA_X + 22, y, t, 9.2, INK, "normal", "start"))

    # ============================ legend ======================================
    lx = PA_X
    for lab, kind in (("serving (current batch)", "serving"),
                      ("waiting — session paused between turns", "waiting"),
                      ("completed", "done"),
                      ("evicted / absent", "absent")):
        f, st, _, dash = KIND[kind]
        d = f'stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<rect x="{lx}" y="{LEG_Y-9}" width="18" height="12" rx="2" fill="{f}" '
                 f'stroke="{st}" stroke-width="1" {d}/>')
        s.append(txt(lx + 24, LEG_Y, lab, 8.6, INK, "normal", "start"))
        lx += 30 + len(lab) * 4.3
    s.append(f'<rect x="{lx}" y="{LEG_Y-9}" width="18" height="12" rx="2" fill="url(#hatch)" '
             f'stroke="{HAIR}" stroke-width="0.9" stroke-dasharray="2.4 1.8"/>')
    s.append(txt(lx + 24, LEG_Y, "capacity held but unused", 8.6, INK, "normal", "start"))
    s.append(txt(PA_X, LEG_Y + 14,
                 "The cache cannot tell waiting from completed — both are LRU-evictable "
                 "entries ordered by last access. That is what parking is for.",
                 8.4, MUTED, "normal", "start", True))

    s.append('</svg>')
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/intro/fig_overview_arch")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg = build()
    open(args.out + ".svg", "w").write(svg)
    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=args.out + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=args.out + ".png", scale=2.0,
                     background_color="white")
    print(f"[saved] {args.out}.svg + .pdf + .png")


if __name__ == "__main__":
    main()

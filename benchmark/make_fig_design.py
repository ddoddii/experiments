#!/usr/bin/env python3
"""
Fig 2 (design) — "KV Victim Cache" overview, paper-quality vector schematic
(SVG -> PDF + PNG), Times New Roman.

Left:  the classic CPU victim-cache analogy (L1 cache -> small victim buffer ->
       main memory) -- our conceptual anchor.
Right: KV Victim Cache (ours). Three KV tiers with K/V block grids:
       L1 GPU serving pool (radix) -> L2 GPU victim cache (transiently-idle HBM,
       this work) -> L3 host DRAM (HiCache). Park (evict) / fetch (reuse) move
       blocks between L1 and L2; host is demoted to a cold overflow across PCIe.
       Right-hand bullets give each tier's speed / role. An L2 callout states the
       novel placement (idlest peer GPU, pressure-aware, cross-node).

사용: python benchmark/make_fig_design.py --out results/perturn/fig_design
"""
import argparse
import math

FONT = "Times New Roman, Times, serif"
# palette (matches the other paper figures)
BLUE, BLUE_F = "#2F6DB0", "#DCE7F5"     # serving / L1 / fast
GREEN, GREEN_F = "#009E73", "#D5F0E6"   # victim (this work)
RED, RED_F = "#B4423C", "#F6DEDE"       # host / HiCache / slow
AMBER = "#E69F00"
INK, MUTED, HAIR = "#111111", "#555555", "#9A9A9A"
W, H = 1080, 360


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=12.5, col=INK, w="normal", anchor="middle", ital=False, style=""):
    it = 'font-style="italic"' if ital else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{col}" '
            f'font-weight="{w}" text-anchor="{anchor}" {it} {style}>{esc(s)}</text>')


def rrect(x, y, w, h, fill, stroke, sw=1.3, rx=6, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {d}/>')


def grid(x, y, n, cell, gap, filled, fillcol, rows=("K", "V"), rowfill=(None, None)):
    """K/V block grid: len(rows) rows of n cells; `filled` = #filled from left."""
    out = []
    for ri, rlab in enumerate(rows):
        ry = y + ri * (cell + gap)
        out.append(txt(x - 9, ry + cell - 3.5, rlab, 9.5, MUTED, "bold", "middle", True))
        fc = rowfill[ri] if rowfill[ri] else fillcol
        for c in range(n):
            cx = x + c * (cell + gap)
            f = fc if c < filled else "white"
            out.append(f'<rect x="{cx}" y="{ry}" width="{cell}" height="{cell}" rx="2" '
                       f'fill="{f}" stroke="{HAIR}" stroke-width="0.8"/>')
    return "".join(out)


def bullet(x, y, s, col=INK):
    return (f'<rect x="{x}" y="{y-7}" width="4.5" height="9" rx="1" fill="{col}"/>'
            + txt(x + 10, y, s, 10.5, INK, "normal", "start"))


def star(cx, cy, r, col):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append(f"{cx+rr*math.cos(a):.1f},{cy+rr*math.sin(a):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{col}"/>'


def build():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append('<defs>')
    for name, col in (("aMut", MUTED), ("aGrn", GREEN), ("aRed", RED), ("aAmb", AMBER)):
        s.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="7.5" '
                 f'markerHeight="7.5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" '
                 f'fill="{col}"/></marker>')
    s.append('</defs>')

    def arrow(x1, y1, x2, y2, col, mk, sw=2.0, dash="", curve=None):
        d = f'stroke-dasharray="{dash}"' if dash else ""
        p = (f'M {x1} {y1} L {x2} {y2}' if curve is None
             else f'M {x1} {y1} Q {curve[0]} {curve[1]} {x2} {y2}')
        s.append(f'<path d="{p}" fill="none" stroke="{col}" stroke-width="{sw}" {d} '
                 f'marker-end="url(#{mk})"/>')

    # ============ LEFT: CPU victim-cache analogy ============
    s.append(txt(150, 26, "Victim cache (CPU architecture)", 12.5, INK, "bold"))
    lx, lw = 74, 152
    an = [(46, BLUE_F, BLUE, "L1 cache", "fast, small"),
          (140, GREEN_F, GREEN, "victim cache", "evicted lines"),
          (238, RED_F, RED, "main memory", "slow, large")]
    for y, f, st, a, b in an:
        thick = 2.2 if st == GREEN else 1.3
        s.append(rrect(lx, y, lw, 56, f, st, thick))
        s.append(txt(lx + lw / 2, y + 24, a, 12, INK, "bold"))
        s.append(txt(lx + lw / 2, y + 41, b, 10, MUTED, "normal", "middle", True))
    # evict (down) / refill (up) between L1 and victim
    arrow(lx - 12, 102, lx - 12, 140, MUTED, "aMut", 1.8)
    arrow(lx + lw + 12, 140, lx + lw + 12, 102, GREEN, "aGrn", 1.8)
    s.append(txt(lx - 16, 124, "evict", 9.5, MUTED, "normal", "end"))
    s.append(txt(lx + lw + 16, 124, "refill", 9.5, GREEN, "bold", "start"))
    arrow(lx + lw / 2, 196, lx + lw / 2, 238, HAIR, "aMut", 1.4, "4 3")
    s.append(txt(lx + lw / 2 + 6, 220, "miss", 9, MUTED, "normal", "start"))

    # analogy connector (chevrons)
    for k in range(3):
        cx = 268 + k * 15
        s.append(f'<path d="M {cx} 150 L {cx+11} 158 L {cx} 166" fill="none" stroke="{AMBER}" '
                 f'stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>')
    s.append(txt(286, 140, "maps to", 10, AMBER, "bold"))
    s.append(f'<line x1="330" y1="34" x2="330" y2="352" stroke="{HAIR}" stroke-width="1" stroke-dasharray="2 3"/>')

    # ============ RIGHT: KV Victim Cache (ours) ============
    s.append(txt(700, 26, "KV Victim Cache (ours)", 13, INK, "bold"))
    tx, tw = 486, 316          # tier box
    gx = tx + 58               # grid origin x
    tiers = [
        (42, 76, BLUE_F, BLUE, "GPU serving pool", "L1 · radix", 6, BLUE, False),
        (158, 80, GREEN_F, GREEN, "GPU victim cache", "L2 · transiently-idle HBM", 4, GREEN, True),
        (262, 76, RED_F, RED, "host DRAM — HiCache", "L3 · file tier", 8, RED, False),
    ]
    for y, hb, f, st, name, sub, filled, gc, hero in tiers:
        s.append(rrect(tx, y, tw, hb, f, st, 2.4 if hero else 1.4))
        s.append(txt(tx + 12, y + 19, name, 12.5, INK, "bold", "start"))
        s.append(txt(tx + 12, y + 33, sub, 10, MUTED, "normal", "start", True))
        if hero:
            s.append(star(tx + tw - 76, y + 14, 6.5, GREEN))
            s.append(txt(tx + tw - 64, y + 18, "this work", 10, GREEN, "bold", "start"))
        s.append(grid(gx, y + 42, 8, 13, 3, filled, gc))

    # park (evict, down) / fetch (reuse, up) between L1 (ends 118) and L2 (starts 158)
    arrow(tx + 150, 120, tx + 150, 156, GREEN, "aGrn", 2.4)     # park down
    arrow(tx + 190, 156, tx + 190, 120, GREEN, "aGrn", 2.4)     # fetch up
    s.append(f'<circle cx="{tx+150}" cy="138" r="9" fill="{AMBER}"/>' + txt(tx + 150, 142, "2", 11, "white", "bold"))
    s.append(txt(tx + 140, 142, "park (evict)", 10, GREEN, "bold", "end"))
    s.append(f'<circle cx="{tx+190}" cy="138" r="9" fill="{AMBER}"/>' + txt(tx + 190, 142, "3", 11, "white", "bold"))
    s.append(txt(tx + 200, 142, "fetch (reuse)", 10, GREEN, "bold", "start"))

    # GPU|host boundary (PCIe) + cold-overflow spill between L2 (238) and L3 (262)
    s.append(f'<line x1="{tx-36}" y1="250" x2="{tx+tw+264}" y2="250" stroke="{HAIR}" '
             f'stroke-width="1" stroke-dasharray="3 3"/>')
    s.append(txt(tx - 36, 246, "GPU HBM", 9, MUTED, "bold", "start"))
    s.append(txt(tx - 36, 261, "host (PCIe)", 9, MUTED, "bold", "start"))
    arrow(tx + tw / 2, 238, tx + tw / 2, 262, RED, "aRed", 1.8, "5 3")
    s.append(txt(tx + tw / 2 + 12, 246, "cold overflow", 9, RED, "normal", "start"))
    s.append(txt(tx + tw / 2 + 12, 258, "(only when GPU full)", 8.5, MUTED, "normal", "start"))

    # right-hand property bullets, aligned to tier centres
    bxx = tx + tw + 22
    s.append(bullet(bxx, 80, "fast · active KV · small"))
    s.append(bullet(bxx, 188, "GPU↔GPU (fast) · evicted KV"))
    s.append(bullet(bxx, 206, "idlest peer GPU: pressure-aware,", GREEN))
    s.append(txt(bxx + 10, 221, "cross-node via shared index", 10.5, GREEN, "normal", "start"))
    s.append(bullet(bxx, 300, "PCIe (slow) · CPU/RAM-contended"))

    s.append('</svg>')
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/perturn/fig_design")
    args = ap.parse_args()
    import os
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

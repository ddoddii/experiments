#!/usr/bin/env python3
"""
Fig 2 (design) — "KV Victim Cache: memory hierarchy + Detect/Park/Fetch mechanism".
Hand-authored vector schematic (SVG -> PDF + PNG), Times New Roman, matching the
paper palette. Two panels:
  (a) KV memory hierarchy: L1 GPU serving -> L2 GPU victim (this work) -> L3 host
      DRAM (HiCache, CPU-contended) -> L4 disk.
  (b) 2P2D dataflow: (1) detect idlest prefill via usage telemetry, (2) park evicted
      session KV onto it over CUDA IPC, (3) next turn fetches from the victim pool
      (local or peer via the shared index) into the serving radix.

사용: python benchmark/make_fig_design.py --out results/perturn/fig_design
"""
import argparse
import math

FONT = "Times New Roman, Times, serif"
# paper palette (matches paperstyle.py)
GREEN, GREEN_L = "#009E73", "#D7F0E7"      # victim (this work)
BLUE, BLUE_L = "#0072B2", "#D6E6F3"        # host / HiCache
AMBER = "#E69F00"                          # step badges
INK, MUTED = "#111111", "#555555"
NODE, NODE_L, SERV_L = "#444444", "#F4F4F4", "#E7E7E7"
GRID = "#CFCFCF"

W, H = 1024, 512


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=13, col=INK, w="normal", anchor="middle", style=""):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{col}" font-weight="{w}" text-anchor="{anchor}" {style}>{esc(s)}</text>')


def box(x, y, w, h, fill, stroke, sw=1.2, rx=7, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {d}/>')


def star(cx, cy, r, col):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append(f"{cx + rr * math.cos(ang):.1f},{cy + rr * math.sin(ang):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{col}"/>'


def badge(cx, cy, n, col=AMBER):
    return (f'<circle cx="{cx}" cy="{cy}" r="11" fill="{col}" stroke="white" stroke-width="1.5"/>'
            + txt(cx, cy + 4.5, n, 13, "white", "bold"))


def arrow(x1, y1, x2, y2, col, sw=2.0, dash="", marker="arrow", curve=None):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    if curve is None:
        path = f'M {x1} {y1} L {x2} {y2}'
    else:
        cx, cy = curve
        path = f'M {x1} {y1} Q {cx} {cy} {x2} {y2}'
    return (f'<path d="{path}" fill="none" stroke="{col}" stroke-width="{sw}" '
            f'{d} marker-end="url(#{marker})"/>')


def build():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    # arrowhead markers
    for name, col in (("arrow", MUTED), ("arrowG", GREEN), ("arrowA", AMBER), ("arrowB", BLUE)):
        s.append(f'<defs><marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                 f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker></defs>')

    # =============== Panel (a): memory hierarchy ===============
    s.append(txt(185, 30, "(a) KV memory hierarchy", 15, INK, "bold"))
    bx, bw = 78, 214
    tiers = [
        (62,  SERV_L, NODE, "L1   GPU serving pool (radix)", None, INK),
        (120, GREEN_L, GREEN, "L2   GPU victim cache", "transiently-idle HBM", INK),
        (178, BLUE_L, BLUE, "L3   host DRAM (HiCache)", "CPU / RAM contended", INK),
        (236, "#F0F0F0", MUTED, "L4   disk (file backend)", None, MUTED),
    ]
    for y, fill, stroke, label, sub, col in tiers:
        thick = 2.4 if stroke == GREEN else 1.2
        s.append(box(bx, y, bw, 50, fill, stroke, thick))
        if sub:
            s.append(txt(bx + bw / 2, y + 21, label, 12.5, col, "bold"))
            s.append(txt(bx + bw / 2, y + 38, sub, 10.5, MUTED))
        else:
            s.append(txt(bx + bw / 2, y + 30, label, 12.5, col, "bold"))
    # "this work" star tag on L2
    s.append(star(bx + bw + 12, 137, 7.5, GREEN))
    s.append(txt(bx + bw + 24, 133, "this", 10.5, GREEN, "bold", "start"))
    s.append(txt(bx + bw + 24, 145, "work", 10.5, GREEN, "bold", "start"))
    # fast/slow speed arrow on the left
    s.append(arrow(46, 66, 46, 282, MUTED, 1.6))
    s.append(txt(30, 74, "fast", 10, MUTED, "normal", "middle", 'transform="rotate(-90 30 74)"'))
    s.append(txt(30, 275, "slow", 10, MUTED, "normal", "middle", 'transform="rotate(-90 30 275)"'))
    # GPU vs host brace labels
    s.append(f'<path d="M {bx+bw+44} 62 L {bx+bw+52} 62 L {bx+bw+52} 170 L {bx+bw+44} 170" '
             f'fill="none" stroke="{MUTED}" stroke-width="1"/>')
    s.append(txt(bx + bw + 74, 118, "GPU", 10.5, MUTED, "bold", "middle",
                 f'transform="rotate(90 {bx+bw+74} 118)"'))
    s.append(f'<path d="M {bx+bw+44} 178 L {bx+bw+52} 178 L {bx+bw+52} 286 L {bx+bw+44} 286" '
             f'fill="none" stroke="{MUTED}" stroke-width="1"/>')
    s.append(txt(bx + bw + 74, 232, "host", 10.5, MUTED, "bold", "middle",
                 f'transform="rotate(90 {bx+bw+74} 232)"'))
    s.append(txt(185, 320, "victim = evicted KV kept on idle GPU", 11, INK, "bold"))
    s.append(txt(185, 336, "instead of spilled to CPU/host", 11, MUTED))

    # vertical divider
    s.append(f'<line x1="352" y1="46" x2="352" y2="470" stroke="{GRID}" stroke-width="1"/>')

    # =============== Panel (b): mechanism ===============
    s.append(txt(690, 30, "(b) Detect → Park → Fetch   (2P2D)", 15, INK, "bold"))

    # decode node D0
    dx, dy, dw, dh = 392, 150, 116, 66
    s.append(box(dx, dy, dw, dh, NODE_L, NODE, 1.2))
    s.append(txt(dx + dw / 2, dy + 25, "D0  (GPU2)", 12, INK, "bold"))
    s.append(txt(dx + dw / 2, dy + 43, "decode; holds", 10, MUTED))
    s.append(txt(dx + dw / 2, dy + 56, "finished KV", 10, MUTED))
    s.append(txt(dx + dw / 2, dy + dh + 15, "(D1 GPU3 symmetric)", 9.5, MUTED))

    # prefill nodes P0 (busy) and P1 (idle) with serving + victim sub-pools
    def prefill(px, py, name, gpu, usage, tag, tagcol):
        pw, ph = 208, 116
        s.append(box(px, py, pw, ph, "white", NODE, 1.3))
        s.append(txt(px + 12, py + 22, f"{name}  ({gpu})", 12, INK, "bold", "start"))
        s.append(txt(px + pw - 12, py + 22, f"usage {usage}", 11, tagcol, "bold", "end"))
        if tag:
            s.append(txt(px + pw - 12, py + 36, tag, 10, tagcol, "bold", "end"))
        # serving sub-box (L1)
        s.append(box(px + 14, py + 44, pw - 28, 28, SERV_L, NODE, 1.0, 5))
        s.append(txt(px + pw / 2, py + 62, "serving KV pool (L1)", 11, INK))
        # victim sub-box (L2)
        s.append(box(px + 14, py + 78, pw - 28, 28, GREEN_L, GREEN, 1.8, 5))
        s.append(txt(px + pw / 2, py + 96, "victim cache (L2, idle HBM)", 11, GREEN, "bold"))
        return px, py, pw, ph

    p0 = prefill(768, 66, "P0", "GPU0", "82%", "busy", BLUE)
    p1 = prefill(768, 262, "P1", "GPU1", "23%", "← idlest", GREEN)

    # shared index / telemetry box
    six, siy, siw, sih = 512, 430, 402, 44
    s.append(box(six, siy, siw, sih, "#FBF6EC", AMBER, 1.3))
    s.append(txt(six + siw / 2, siy + 19, "Shared Park Index  +  usage telemetry", 11.5, INK, "bold"))
    s.append(txt(six + siw / 2, siy + 35, "(/dev/shm, cross-node)", 10, MUTED))

    # --- (1) DETECT: P0/P1 publish usage, D reads telemetry ---
    s.append(arrow(792, 182, 720, siy, MUTED, 1.4, "4 3"))           # P0 -> index
    s.append(arrow(792, 378, 760, siy, MUTED, 1.4, "4 3"))           # P1 -> index
    s.append(arrow(six, siy + 12, 470, 210, MUTED, 1.4, "4 3", curve=(430, 430)))  # index -> D0 (read)
    s.append(badge(548, 408, "1"))
    s.append(txt(548, 393, "publish KV usage / read", 9.5, MUTED))

    # --- (2) PARK: D0 -> P1 victim (idlest), CUDA IPC ---
    s.append(arrow(508, 183, 782, 352, GREEN, 2.2, "", "arrowG", curve=(650, 300)))
    s.append(badge(628, 250, "2"))
    s.append(txt(632, 232, "park evicted KV → idlest P", 10.5, GREEN, "bold", "middle"))
    s.append(txt(632, 219, "(CUDA IPC, GPU→GPU)", 9.5, MUTED, "normal", "middle"))

    # --- (3) FETCH: next turn at P0 misses -> peer P1 victim via shared index -> P0 serving ---
    # incoming turn N into P0
    s.append(arrow(690, 78, 766, 92, INK, 1.6))
    s.append(txt(676, 74, "turn N", 10, INK, "bold", "end"))
    # cross-node fetch: P1 victim -> P0 serving (curved on the right)
    s.append(arrow(978, 356, 978, 124, GREEN, 2.2, "", "arrowG", curve=(1016, 240)))
    s.append(badge(998, 240, "3"))
    s.append(txt(958, 210, "cross-node fetch", 10.5, GREEN, "bold", "end"))
    s.append(txt(958, 198, "via shared index → radix hit", 9.5, MUTED, "normal", "end"))

    s.append('</svg>')
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/perturn/fig_design")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg = build()
    with open(args.out + ".svg", "w") as f:
        f.write(svg)
    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=args.out + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=args.out + ".png", scale=2.0, background_color="white")
    print(f"[saved] {args.out}.svg + .pdf + .png")


if __name__ == "__main__":
    main()

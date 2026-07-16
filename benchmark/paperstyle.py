"""
Publication figure style (IEEE/CAL). Times New Roman if present, else Liberation
Serif -- metric-compatible with Times New Roman (identical metrics/appearance),
the standard open substitute accepted by IEEE/ACM. Vector PDF with embedded
editable text (fonttype 42) + PNG preview.

    from paperstyle import use_paper_style, PALETTE, savefig
    use_paper_style();  ...;  savefig(fig, "results/perturn/fig3")  # -> .pdf + .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colorblind-safe, print-legible (Okabe-Ito-ish). Pair with marker+linestyle so the
# figure survives grayscale.
PALETTE = {
    "hicache": "#0072B2",   # blue
    "park":    "#009E73",   # green
    "both":    "#E69F00",   # amber (only if a 3rd arm is explicitly requested)
    "ink":     "#000000",
    "muted":   "#555555",
    "grid":    "#D9D9D9",
}
# marker + linestyle per arm (redundant encoding for grayscale)
STYLE = {
    "hicache": dict(marker="o", ls="--"),
    "park":    dict(marker="s", ls="-"),
    "both":    dict(marker="^", ls=":"),
}


def use_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType -> selectable/editable text in the PDF
        "ps.fonttype": 42,
    })


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.tick_params(colors=PALETTE["muted"], length=2.5, width=0.5)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(PALETTE["muted"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])


def savefig(fig, stem):
    """Save both a vector PDF (camera-ready) and a PNG (preview) at `stem`.{pdf,png}."""
    import os
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", facecolor="white")
    print(f"[saved] {stem}.pdf + {stem}.png")

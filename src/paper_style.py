"""Shared print style for the paper's figures (IJMPC, World Scientific).

Every figure is drawn at its printed size: 5.0 in (12.7 cm) wide, the text
width of ws-ijmpc, so the manuscript's \\includegraphics[width=12.7cm] places
it at 1:1 and no lettering is reduced. Fonts are 7-8 pt at that size (the
captions are 8 pt).

Nothing here uses transparency, because PostScript has none and the EPS
files would otherwise be flattened or rendered opaque; light fills are given
as solid tints instead.

save(fig, name) writes
    FIGS_DIR/<name>.png       600 dpi raster (what pdflatex includes)
    FIGS_EPS_DIR/<name>.eps   vector EPS with Type 42 (embedded) fonts
Both directories default to ../figures; set the environment variables to
write elsewhere (e.g. straight into the manuscript's figs/ and figs_eps/).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.environ.get("FIGS_DIR", os.path.join(HERE, "..", "figures"))
FIGS_EPS = os.environ.get("FIGS_EPS_DIR", FIGS)
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))

WIDTH = 5.0          # inches = 12.7 cm = ws-ijmpc text width

# Okabe-Ito (colour-vision-deficiency safe), fixed assignment, never cycled
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "black": "#000000", "grey": "#6B7280"}

# the two generators, used identically in every figure
GEN = {"GB": dict(label=r"$G^{B}$", color=OI["blue"], marker="o", ls="-"),
       "GA": dict(label=r"$G^{A}$", color=OI["red"], marker="D", ls="--")}


def tint(hex_colour, f):
    """Solid colour a fraction f of the way from hex_colour to white."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c + (255 - c) * f))
    return "#{:02x}{:02x}{:02x}".format(mix(r), mix(g), mix(b))


def use():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.borderaxespad": .3, "legend.labelspacing": .3,
        "axes.linewidth": .6, "lines.linewidth": 1.1, "lines.markersize": 3.5,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.minor.width": .4, "ytick.minor.width": .4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": .5,
        "axes.axisbelow": True, "errorbar.capsize": 1.5,
        "axes.titlepad": 3.5, "axes.labelpad": 2.5,
        "mathtext.fontset": "dejavusans",
        "ps.fonttype": 42, "pdf.fonttype": 42,
    })


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(FIGS_EPS, exist_ok=True)
    png = os.path.join(FIGS, name + ".png")
    eps = os.path.join(FIGS_EPS, name + ".eps")
    # fixed canvas (no bbox_inches="tight") so the file is exactly WIDTH wide
    fig.savefig(png, dpi=600, facecolor="white")
    try:                                   # flatten to RGB: no alpha channel
        from PIL import Image
        Image.open(png).convert("RGB").save(png, dpi=(600, 600))
    except ImportError:
        pass
    fig.savefig(eps, format="eps", facecolor="white")
    plt.close(fig)
    print("wrote", png, "and", eps)

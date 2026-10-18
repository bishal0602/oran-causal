"""Specimen: the legend and signs key of the design system (DESIGN s.6), synthetic positions. (a) every
method of the five gallery figures in its fixed marker / colour / line style, hero first; (b) the
signs on a data axes: good and bad zones, threshold line, reference line, CI band, whisker, a
not-scored cross and an auto-placed callout with its leader line; (c) the map tiles with their sign
triangles; (d) a diagram connector with its plain-text label.
"""

import matplotlib.pyplot as plt
import numpy as np

from oran_causal.figures import diagram as G
from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import fig_maps as FM
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_key", width="full", paper=False, sources=["synthetic"])

SERIES = [
    T.HERO,
    "pmrt_eq",
    "pcorr_eq",
    "granger_eq",
    "rcot2_eq",
    "corr",
    "pcorr_native",
    "granger_native",
]
POINTS = [
    ("MG:PMRT", True, False, "PMRT map, no gate (open)"),
    ("CS:PMRT", True, True, "PMRT map, gate (filled)"),
    ("MG:GT", False, False, "Oracle map, no gate"),
    ("CS:GT", False, True, "Oracle map, gate"),
]
TILE_ROWS = [
    ("truth", 1, "true edge (+)"),
    ("truth", -1, "true edge (−)"),
    ("right", 1, "right sign"),
    ("wrong", -1, "wrong sign"),
    ("false", 1, "false edge"),
    ("missed", 1, "missed"),
    ("indet", 0, "not scored"),
    (None, 0, "null / not declared"),
]
HEIGHT_PT = 250
TILE_PT = 12  # one map tile cell, in points


def _methods(ax) -> None:
    ax.set_xlim(0, 200)
    ax.set_ylim(HEIGHT_PT, 0)
    ax.axis("off")
    ax.text(
        0,
        8,
        "Methods (fixed in every figure)",
        fontsize=S.LABEL_PT,
        fontweight="bold",
        va="center",
        color=P.INK,
    )
    y = 24
    for code in SERIES:
        st = M.style(code, code == T.HERO)
        st.update(gid=st["gid"], label=None)
        ax.plot([2, 16, 30], [y, y, y], markevery=[1], **st)
        ax.text(36, y, M.name(code), fontsize=S.ANNOT_PT, va="center", color=P.INK)
        y += 15
    y += 4
    ax.text(
        0, y, "Decision points", fontsize=S.LABEL_PT, fontweight="bold", va="center", color=P.INK
    )
    y += 15
    for code, hero, fill, label in POINTS:
        M.row_point(ax, 16, y, None, code, hero=hero, fill=fill)
        ax.text(36, y, label, fontsize=S.ANNOT_PT, va="center", color=P.INK)
        y += 15


def _signs(ax) -> None:
    x = np.linspace(0, 10, 11)
    ax.set(xlim=(0, 10), ylim=(0, 1), xlabel="Synthetic x", ylabel="Synthetic y")
    plots.zone(ax, 0, 5, "good", axis="x")
    plots.zone(ax, 5, 10, "bad", axis="x")
    plots.ref_line(ax, 5, axis="x")
    L.label_line(ax, 5, 0.95, "threshold", where="left")
    L.label_line(ax, 5, 0.95, "bad zone", where="right")
    plots.ref_line(ax, 0.2)
    L.label_line(ax, 9.8, 0.2, "nominal level", where="above", ha="right")
    ref = 0.35 + 0.04 * x
    M.reference_line(ax, x, ref)
    L.label_line(ax, 0.2, ref[0], "reference", where="above")
    y = 0.55 + 0.025 * x
    M.band(ax, x, y - 0.08, y + 0.08, T.HERO, hero=True)
    M.plot(ax, x, y, T.HERO, hero=True, markevery=[2, 8])
    plots.errorbars(ax, [8], [y[8]], [[0.08], [0.08]])
    L.invalid_marks(ax, [7], P.INK)
    L.callout(ax, 2, y[2], "callout\nwith leader", prefer=("s", "sse", "se"))
    plots.finish(ax)


def _tiles(ax) -> None:
    w, h = (
        ax.get_position().width * S.FULL * 72 / TILE_PT,
        ax.get_position().height * HEIGHT_PT / TILE_PT,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h - 1.2, -1.2)
    ax.axis("off")
    ax.text(0, -0.6, "Map tiles", fontsize=S.LABEL_PT, fontweight="bold", va="center", color=P.INK)
    for i, (cat, sign, label) in enumerate(TILE_ROWS):
        col, row = divmod(i, (len(TILE_ROWS) + 1) // 2)
        x0, y0 = col * w / 2, row * 1.35
        FM._draw_tile(ax, x0, y0, cat, sign)
        ax.text(x0 + 1.3, y0 + 0.5, label, fontsize=S.ANNOT_PT, va="center", color=P.INK)


def _connector(ax) -> None:
    ax.set_xlim(0, 200)
    ax.set_ylim(36, 0)
    ax.axis("off")
    ax.text(
        0, 6, "Diagram connector", fontsize=S.LABEL_PT, fontweight="bold", va="center", color=P.INK
    )
    G.arrow(ax, [(6, 26), (190, 26)], label="label")


def build():
    S.apply()
    fig = plt.figure(figsize=(S.FULL, HEIGHT_PT / 72))
    _methods(fig.add_axes((0.01, 0.02, 0.43, 0.96)))
    sig = fig.add_axes((0.56, 0.60, 0.42, 0.34))
    _signs(sig)
    L.panel_header(sig, "b", "Signs")
    _tiles(fig.add_axes((0.49, 0.13, 0.50, 0.33)))
    _connector(fig.add_axes((0.49, 0.0, 0.50, 0.12)))
    return L.finalize(fig)

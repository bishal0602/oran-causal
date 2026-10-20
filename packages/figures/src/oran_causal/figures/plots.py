"""Plot helpers that encode the design rules. Figure scripts should use these
instead of styling marks by hand, so every figure stays consistent.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from oran_causal.figures import palette as P
from oran_causal.figures import style as S

WIDTHS = {"full": S.FULL, "half": S.HALF}


# --- figure creation -----------------------------------------------------------
def figure(
    width: str = "full", aspect: float = S.ASPECT, nrows: int = 1, ncols: int = 1, **kw
) -> tuple[Figure, np.ndarray | Axes]:
    """New figure at final print width. width: 'full' (5.15 in) or 'half' (2.5 in)."""
    S.apply()
    w = WIDTHS[width]
    h = min(w * aspect, S.MAX_HEIGHT)
    kw.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def finish(ax: Axes, grid: bool | str = True) -> None:
    """Soft-modern frame (DESIGN s.4): no spines and no tick marks; the gray panel field and its white grid
    carry the frame. ``grid``: ``False`` removes the grid (e.g. categorical panels), ``"x"`` / ``"y"``
    keeps only the lines across that axis (a row-wise panel keeps its x grid).
    """
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(which="both", length=0)
    if grid in ("x", "y"):
        ax.grid(False)
        ax.grid(axis=grid, visible=True)
    elif not grid:
        ax.grid(False)


def bare(ax: Axes) -> None:
    """An axes that only carries marks (a strip of dots): its field stays, with no spines, ticks or grid;
    labels and titles are still drawn.
    """
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def band(ax: Axes, x, lo, hi, color: str, hero: bool = False, **kw):
    """95% CI band: fill only, no edge."""
    return ax.fill_between(
        x, lo, hi, color=color, alpha=0.18 if hero else 0.14, lw=0, zorder=2, **kw
    )


def errorbars(ax: Axes, x, y, err, **kw):
    """CI whiskers in ink, cap 2 pt, LW_WHISKER."""
    return ax.errorbar(
        x,
        y,
        yerr=err,
        fmt="none",
        ecolor=P.INK,
        elinewidth=S.LW_WHISKER,
        capsize=2,
        capthick=S.LW_WHISKER,
        zorder=5,
        **kw,
    )


# --- reference lines, legends, annotations -------------------------------------
REF_ALPHA = 0.8
REF_DASH = (0, (4, 2))


def ref_line(ax: Axes, y: float, label: str | None = None, axis: str = "y"):
    """Threshold/target line: ink at 80 %, dash (4, 2), LW_REF, labelled directly (not in the legend)."""
    kw = {"color": P.INK, "alpha": REF_ALPHA, "lw": S.LW_REF, "ls": REF_DASH, "zorder": 2}
    if axis == "y":
        ax.axhline(y, **kw)
        if label:
            ax.annotate(
                label,
                xy=(0, y),
                xycoords=("axes fraction", "data"),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=S.ANNOT_PT,
                color=P.INK,
                va="bottom",
            )
    else:
        ax.axvline(y, **kw)
        if label:
            ax.annotate(
                label,
                xy=(y, 1),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                fontsize=S.ANNOT_PT,
                color=P.INK,
                va="top",
            )


def span(
    ax: Axes,
    lo: float,
    hi: float,
    role: str,
    axis: str = "y",
    tint: float | None = None,
    gid: str | None = None,
    zorder: float = 0,
):
    """A tinted region (role 'good' | 'bad' | 'ours') between lo and hi on ``axis``: the role hue at
    ``tint`` (default ``P.ZONE_TINT``), no edge.
    """
    color = {"good": P.GOOD, "bad": P.BAD, "ours": P.OURS}[role]
    draw = ax.axhspan if axis == "y" else ax.axvspan
    return draw(
        lo,
        hi,
        fc=P.tint(color, P.ZONE_TINT if tint is None else tint),
        ec="none",
        lw=0,
        zorder=zorder,
        gid=gid or f"sem:{role}-zone",
    )


def row_band(ax: Axes, y: float, height: float = 1.0):
    """The hero's row in a row-wise panel: a light blue band behind the row at ``y``."""
    return span(
        ax, y - height / 2, y + height / 2, "ours", tint=0.16, gid="sem:ours-row", zorder=0.4
    )


def zero_line(ax: Axes, axis: str = "x"):
    """A plain axis line at 0 (a lollipop baseline), under the marks."""
    line = ax.axvline if axis == "x" else ax.axhline
    return line(0, color=P.AXIS, lw=S.LW_AXIS, zorder=2, gid="zero-axis")


def arrow(
    target,
    p: tuple[float, float],
    q: tuple[float, float],
    color: str = P.INK,
    lw: float = 0.8,
    transform=None,
    gid: str = "arrow",
    zorder: float = 2.5,
    shrink: tuple[float, float] = (0, 0),
):
    """A plain straight arrow from ``p`` to ``q`` (axes or figure ``target``; same head as
    ``diagram.arrow``), e.g. the shift a gate causes or the direction of time. ``shrink``: points
    kept clear at the tail and the head (markers).
    """
    from matplotlib.patches import FancyArrowPatch

    patch = FancyArrowPatch(
        p,
        q,
        arrowstyle="-|>,head_length=4,head_width=2.2",
        mutation_scale=1,
        color=color,
        lw=lw,
        shrinkA=shrink[0],
        shrinkB=shrink[1],
        zorder=zorder,
        gid=gid,
        **({"transform": transform} if transform is not None else {}),
    )
    if hasattr(target, "add_patch"):
        target.add_patch(patch)
    else:
        target.add_artist(patch)
    return patch


def zone(ax: Axes, lo: float, hi: float, role: str, axis: str = "y", gid: str | None = None):
    """A semantic region (role 'good' | 'bad') between lo and hi on ``axis`` (DESIGN s.6): a light tint
    of the role hue, bordered by a full-strength line on the limit (the region's inner edge).
    """
    color = P.GOOD if role == "good" else P.BAD
    gid = gid or f"sem:{role}-zone"
    z = span(ax, lo, hi, role, axis, gid=gid)
    edge = lo if role == "bad" else hi
    line = ax.axhline if axis == "y" else ax.axvline
    line(edge, color=color, lw=S.LW_REF, zorder=0.6, gid=gid + "-edge")
    return z


TEXT_CONTRAST = 4.5  # WCAG AA for normal-size text


def _text_on(color) -> str:
    """Text colour on a filled cell: ink unless ink is under WCAG 4.5:1 against the cell, then the
    colour with the higher contrast (white on dark cells). Chosen by WCAG relative luminance.
    """
    if P.contrast(color, P.INK) >= TEXT_CONTRAST:
        return P.INK
    return max((P.INK, P.WHITE), key=lambda t: P.contrast(color, t))

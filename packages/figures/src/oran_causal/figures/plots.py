"""Plot helpers that encode the design rules. Figure scripts should use these
instead of styling marks by hand, so every figure stays consistent.
"""

from collections.abc import Sequence

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


def panel_label(ax: Axes, letter: str, x: float = -0.02, y: float = 1.0) -> None:
    """Bold lowercase panel letter ('a', 'b', ...) at the top-left of the axes area."""
    ax.annotate(
        letter,
        xy=(x, y),
        xycoords="axes fraction",
        xytext=(-14, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=S.PANEL_PT,
        fontweight="bold",
        color=P.INK,
    )


def finish(ax: Axes, grid: bool = True) -> None:
    """Soft-modern frame (DESIGN s.4): no spines and no tick marks; the gray panel field and its white grid
    carry the frame. ``grid=False`` removes the grid (e.g. categorical panels).
    """
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(which="both", length=0)
    if not grid:
        ax.grid(False)


# --- methods (hero + baselines) ------------------------------------------------
def plot_method(ax: Axes, x, y, k: int, label: str | None = None, markevery=None, **kw):
    """K = 0 proposed (filled marker, heavy line, on top); k = 1..3 baselines A..C (open marker,
    light line).
    """
    hero = k == 0
    c = P.method(k)
    style = {
        "color": c,
        "marker": P.METHOD_MARKERS[k],
        "label": label,
        "markevery": markevery,
        "lw": S.LW_PROPOSED if hero else S.LW_BASELINE,
        "zorder": 4 if hero else 3,
        "mfc": c if hero else P.WHITE,
        "mec": P.WHITE if hero else c,
        "mew": 0.5 if hero else 0.8,
        "ms": S.MARKER_SIZE + (0.4 if hero else 0),
    }
    style.update(kw)
    return ax.plot(x, y, **style)


def band(ax: Axes, x, lo, hi, color: str, hero: bool = False, **kw):
    """95% CI band: fill only, no edge."""
    return ax.fill_between(
        x, lo, hi, color=color, alpha=0.18 if hero else 0.14, lw=0, zorder=2, **kw
    )


def bar_method(ax: Axes, x, h, k: int, width: float, label: str | None = None, **kw):
    """Proposed bar solid; baseline bars = light tint + colored edge (the bar version of an open marker)."""
    c = P.method(k)
    if k == 0:
        return ax.bar(x, h, width, color=c, label=label, zorder=3, **kw)
    return ax.bar(
        x, h, width, color=P.tint(c, 0.28), edgecolor=c, linewidth=0.7, label=label, zorder=3, **kw
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


def grouped_method_bars(
    ax: Axes,
    groups: Sequence[str],
    values: np.ndarray,
    errors: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    group_width: float = 0.74,
    gap: float = 0.02,
):
    """values[g, k]: group g, method slot k. Draws bars with fixed slot colors and optional CI whiskers."""
    values = np.asarray(values)
    n_groups, n_methods = values.shape
    bw = group_width / n_methods
    xs = np.arange(n_groups)
    for k in range(n_methods):
        xk = xs - group_width / 2 + (k + 0.5) * bw
        bar_method(ax, xk, values[:, k], k, bw - gap, label=labels[k] if labels else None)
        if errors is not None:
            errorbars(ax, xk, values[:, k], np.asarray(errors)[:, k])
    ax.set_xticks(xs, groups)
    ax.tick_params(axis="x", length=0)


# --- entities (peers) ----------------------------------------------------------
def plot_entity(ax: Axes, x, y, k: int, label: str | None = None, markevery=None, **kw):
    """Entity slot k: filled marker, equal weight."""
    style = {
        "color": P.entity(k),
        "marker": P.ENTITY_MARKERS[k],
        "label": label,
        "markevery": markevery,
        "lw": S.LW_ENTITY,
        "mfc": P.entity(k),
        "mec": P.WHITE,
        "mew": 0.5,
        "zorder": 3,
    }
    style.update(kw)
    return ax.plot(x, y, **style)


def stacked_shares(
    ax: Axes,
    rows: Sequence[str],
    shares: np.ndarray,
    labels: Sequence[str],
    annotate_min: float = 0.15,
    height: float = 0.66,
):
    """Horizontal 100% stacked bars; shares[r, k] in [0, 1]; 1 pt white gaps between segments."""
    shares = np.asarray(shares)
    left = np.zeros(len(rows))
    for k in range(shares.shape[1]):
        ax.barh(
            rows,
            shares[:, k],
            left=left,
            height=height,
            color=P.entity(k),
            edgecolor=P.WHITE,
            linewidth=1.0,
            label=labels[k],
            zorder=3,
        )
        for r, v in enumerate(shares[:, k]):
            if v >= annotate_min:
                ax.text(
                    left[r] + v / 2,
                    r,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=S.ANNOT_PT,
                    color=_text_on(P.entity(k)),
                )
        left += shares[:, k]
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


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


def legend_right(ax: Axes, handles=None, labels=None, **kw):
    """Legend column outside the axes, top-aligned (framed by the rc legend box)."""
    args = [a for a in (handles, labels) if a is not None]
    return ax.legend(*args, loc="upper left", bbox_to_anchor=(1.02, 1.0), **kw)


def legend_top(ax: Axes, ncol: int = 2, handles=None, labels=None, **kw):
    """Legend row above the axes, left-aligned (framed by the rc legend box)."""
    args = [a for a in (handles, labels) if a is not None]
    return ax.legend(*args, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol, **kw)


def heatmap(
    ax: Axes,
    data: np.ndarray,
    rows: Sequence[str],
    cols: Sequence[str],
    diverging: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.2f}",
    outline_mask=None,
    cbar_label: str | None = None,
    annotate: bool = True,
):
    """Annotated heatmap. Sequential (magnitude) by default; diverging (signed) is symmetric about 0."""
    data = np.asarray(data, dtype=float)
    cmap = P.DIV_CMAP if diverging else P.SEQ_CMAP
    if diverging:
        lim = np.nanmax(np.abs(data)) if vmax is None else vmax
        vmin, vmax = -lim, lim
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=P.WHITE, linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                txt = fmt.format(v).replace("-", "−")
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=S.ANNOT_PT,
                    color=_text_on(cmap(im.norm(v))),
                )
    if outline_mask is not None:
        from matplotlib.patches import Rectangle

        for i, j in zip(*np.nonzero(outline_mask), strict=False):
            ax.add_patch(
                Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec=P.INK, lw=1.0, zorder=4)
            )
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=S.LW_AXIS)
    if cbar_label:
        cb.set_label(cbar_label)
    return im, cb


def zone(ax: Axes, lo: float, hi: float, role: str, axis: str = "y", gid: str | None = None):
    """A semantic region (role 'good' | 'bad') between lo and hi on ``axis`` (DESIGN s.6): a light tint
    of the role hue, bordered by a full-strength line on the limit (the region's inner edge).
    """
    color = P.GOOD if role == "good" else P.BAD
    span = ax.axhspan if axis == "y" else ax.axvspan
    gid = gid or f"sem:{role}-zone"
    z = span(lo, hi, fc=P.tint(color, P.ZONE_TINT), ec="none", lw=0, zorder=0, gid=gid)
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


def heat_colorbar(
    fig, ax, norm, ticks, ticklabels, label, drop: float = 0.62, thick: float = 0.055
):
    """Shared colour bar for heat-cell panels (fig_study4_validity (b), referee value grids): a thin
    horizontal bar under the panel, as an inset so the constrained layout reserves room for it.
    """
    from matplotlib.cm import ScalarMappable

    from oran_causal.figures import palette as _P

    fig.draw_without_rendering()
    h_ax = ax.get_window_extent().height / fig.dpi
    cax = ax.inset_axes((0.08, -(drop + thick) / h_ax, 0.84, thick / h_ax))
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=_P.DIV_CMAP), cax=cax, orientation="horizontal", ticks=ticks
    )
    cb.ax.set_xticklabels(ticklabels, fontsize=S.ANNOT_PT)
    cb.ax.tick_params(length=2, width=0.5, pad=1.5)
    cb.outline.set_visible(False)
    cb.set_label(label, fontsize=S.ANNOT_PT, labelpad=2)
    return cb

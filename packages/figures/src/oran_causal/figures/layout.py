"""Figure-composition helpers: the layout rules every multi-panel figure shares (DESIGN.md s.4, s.6, s.7 and the
[NEW 2026-10-05] rules there). Figure modules call these instead of placing legends, panel headers, INVALID marks,
dodges or reference labels by hand, and end ``build()`` with ``finalize(fig)``.

    panel_header(ax, "a", "E1, R2")      letter + tag with a fixed hierarchy and gap (DESIGN s.3)
    fig_legend(fig, handles, labels)     top row, centred, balanced columns; or the right column (DESIGN s.6)
    tidy_grid(axs)                       shared-axis tick hygiene for a grid of panels (DESIGN s.4)
    recall_axis(ax) / invalid_marks()    recall axis with an in-axes INVALID strip + automatic legend entry (DESIGN s.7)
    dodge(ax, i, n)                      horizontal dodge (a transform) so identical series stay visible (DESIGN s.7)
    ecdf_line(p, grid, phase)            ECDF that starts at the first positive p, never NaN-masked (DESIGN s.7)
    rate_axis(ax, "y")                   rejection-rate axis without an artificial floor: symlog, 0 shown (DESIGN s.4)
    label_line(ax, x, y, text)           data-anchored label next to a line, offset <= 6 pt (DESIGN s.4)
    banner(fig, ax, y, h, text)          category header as a full-width banner row (DESIGN s.7)
    method_rows(rows) / method_labels()  row-wise (forest / grid) layout with a gap between methods and the bold
                                         method name on its first sub-row (DESIGN s.7 [NEW 2026-10-05])
    finalize(fig)                        resolves letters and legend columns after one layout pass
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory, offset_copy

from oran_causal.figures import palette as P
from oran_causal.figures import style as S

TAG_PT = S.LEGEND_PT  # panel tag: regular 8 pt under the 10 pt bold letter ([G5] was 7.5 / 9.5)
LETTER_GAP_PT = 14  # minimum gap between a panel letter's right edge and its tag / axes edge
HEADER_RISE_PT = 5  # baseline of letter and tag above the axes top
MAX_LABEL_OFFSET_PT = 6  # data-anchored labels sit at most this far from their anchor
INVALID_Y = 1.15  # INVALID strip of a recall axis (data units, above the 0-1 scale)
RECALL_YLIM = (-0.04, 1.24)
RATE_TICKS = [0, 0.01, 0.05, 0.2, 1]
RATE_LIM = (
    -0.003,
    1.35,
)  # [2026-10-06] headroom so a marker at rate 0 or 1 is never half clipped (checks)
INVALID_MS = S.MARKER_SIZE + 0.4
INVALID_STEP_PT = (
    INVALID_MS + 1.2
)  # [2026-10-06] crosses sharing an x are spaced this far apart (never touch)


def _root(fig) -> Figure:
    while isinstance(fig, SubFigure):
        fig = fig.figure
    return fig


def _reg(fig, key: str) -> list:
    root = _root(fig)
    if hasattr(root, "_cdd"):
        root._cdd.setdefault("invalid_pts", [])
    if not hasattr(root, "_cdd"):
        root._cdd = {"headers": [], "legends": [], "invalid": False, "invalid_pts": []}
    return root._cdd[key] if key else root._cdd


# --- panel headers (DESIGN s.3) -----------------------------------------------------------------------------------------------
def panel_header(ax: Axes, letter: str, tag: str | None = None, inset: bool = False) -> None:
    """Bold letter at the top-left, left of the axes' tick-label / y-label column; optional tag at
    the axes' left edge on the same baseline. ``finalize`` moves the letter so the gap to the tag is
    at least LETTER_GAP_PT. ``inset`` [NEW 2026-10-05]: for a panel without a label column (a graph
    or diagram canvas at the page edge), the letter starts at the axes' left edge and the tag
    follows LETTER_GAP_PT after it.
    """
    x_let, ha = (0, "left") if inset else (-LETTER_GAP_PT, "right")
    let = ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(x_let, HEADER_RISE_PT),
        textcoords="offset points",
        ha=ha,
        va="baseline",
        fontsize=S.PANEL_PT,
        fontweight="bold",
        color=P.INK,
    )
    let.set_gid("panel-letter")
    t = None
    if tag:
        t = ax.annotate(
            tag,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(LETTER_GAP_PT + 7 if inset else 0, HEADER_RISE_PT),
            textcoords="offset points",
            ha="left",
            va="baseline",
            fontsize=TAG_PT,
            color=P.INK,
        )
        t.set_gid("panel-tag")
    if not inset:
        _reg(ax.figure, "headers").append((ax, let, t))


def _place_letters(fig: Figure, r) -> None:
    for ax, let, _ in _reg(fig, "headers"):
        x0 = ax.get_window_extent(r).x0
        parts = [
            a.get_window_extent(r) for a in (ax.yaxis.label,) if a.get_text() and a.get_visible()
        ]
        parts += [
            t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_text() and t.get_visible()
        ]
        parts += [
            t.get_window_extent(r) for t in ax.texts if t.get_gid() == "row-label"
        ]  # outer row labels
        left = min([p.x0 for p in parts], default=x0)
        px = fig.dpi / 72
        w = let.get_window_extent(r).width
        x = min(left, x0 - LETTER_GAP_PT * px - w)  # above the label column, or the minimum gap
        let.set_horizontalalignment("left")
        let.xyann = ((x - x0) / px, HEADER_RISE_PT)


# --- legends (DESIGN s.6) -----------------------------------------------------------------------------------------------------
def invalid_handle() -> Line2D:
    return Line2D(
        [], [], ls="", marker="x", ms=S.MARKER_SIZE + 0.4, mew=1.0, color=P.INK, mec=P.INK
    )


INVALID_LABEL = "INVALID condition"  # [G3 2026-10-12] was "INVALID cell (no recall)"; "INVALID condition (no recall)"
# is 20 pt wider and pushed the E4 page legend into one column (letter b then hit the y label)


def fig_legend(fig, handles: Sequence, labels: Sequence[str], where: str = "top", **kw):
    """Figure-level legend. ``top``: one centred row above every panel header, wrapped into balanced
    rows when it does not fit the width (finalize). ``right``: a top-aligned column right of the
    panels. ``auto`` [DECIDED 2026-10-08, user; agreed style page]: a right column on a full-width
    figure, unless it would squeeze a data panel below MIN_PANEL_IN, then the top row; always the
    top row on a half-width figure (finalize decides). Never below or inside the axes. An ``INVALID
    condition`` entry is appended automatically when the figure drew INVALID marks.
    """
    handles, labels = list(handles), list(labels)
    if _reg(fig, "invalid") and INVALID_LABEL not in labels:
        handles.append(invalid_handle())
        labels.append(INVALID_LABEL)
    if where == "auto" and _root(fig).get_figwidth() < (S.FULL + S.HALF) / 2:
        where = "top"  # half width: top row (agreed style page)
    loc = {
        "top": "outside upper center",
        "right": "outside right upper",
        "auto": "outside right upper",
    }[where]
    ncol = len(labels) if where == "top" else 1
    leg = fig.legend(handles, labels, loc=loc, ncol=ncol, **kw)
    _reg(fig, "legends").append((fig, leg, handles, labels, where, kw))
    return leg


MIN_PANEL_IN = 1.0  # = checks.MIN_AXES_IN: no data panel narrower than this (DESIGN s.2)


def _data_axes(fig: Figure) -> list:
    out, seen = [], set()
    stack = [fig]
    while stack:
        f = stack.pop()
        stack.extend(getattr(f, "subfigs", []))
        for ax in f.axes:
            if (
                id(ax) not in seen
                and ax.get_visible()
                and ax.axison
                and ax.get_label() != "<colorbar>"
            ):
                seen.add(id(ax))
                out.append(ax)
    return out


def _resolve_auto(fig: Figure, r) -> bool:
    """``auto`` legends: keep the right column if every data panel stays >= MIN_PANEL_IN wide and
    the legend lies on the canvas; otherwise replace it by the top row.
    """
    changed = False
    reg = _reg(fig, "legends")
    for i, (owner, leg, handles, labels, where, kw) in enumerate(reg):
        if where != "auto":
            continue
        narrow = any(
            ax.get_window_extent(r).width / fig.dpi < MIN_PANEL_IN - 1e-6 for ax in _data_axes(fig)
        )
        off = leg.get_window_extent(r).x1 > fig.bbox.width + 1
        if narrow or off:
            leg.remove()
            leg = owner.legend(handles, labels, loc="outside upper center", ncol=len(labels), **kw)
            reg[i] = (owner, leg, handles, labels, "top", kw)
            changed = True
        else:
            reg[i] = (owner, leg, handles, labels, "right", kw)
    return changed


def _fit_legends(fig: Figure, r) -> bool:
    changed = False
    reg = _reg(fig, "legends")
    for i, (owner, leg, handles, labels, where, kw) in enumerate(reg):
        if where != "top":
            continue
        avail = owner.bbox.width * 0.98
        ncol = leg._ncols
        if leg.get_window_extent(r).width <= avail or ncol == 1:
            continue
        n = len(labels)
        for rows in range(2, n + 1):
            ncol = int(np.ceil(n / rows))
            leg.remove()
            leg = owner.legend(handles, labels, loc="outside upper center", ncol=ncol, **kw)
            if leg.get_window_extent(r).width <= avail or ncol == 1:
                break
        reg[i] = (owner, leg, handles, labels, where, kw)
        changed = True
    return changed


# --- shared-axis hygiene (DESIGN s.4) ---------------------------------------------------------------------------------------
def _same_x(a: Axes, b: Axes) -> bool:
    return (
        a.get_xscale() == b.get_xscale()
        and np.allclose(a.get_xlim(), b.get_xlim())
        and list(a.get_xticks()) == list(b.get_xticks())
    )


def _same_y(a: Axes, b: Axes) -> bool:
    return (
        a.get_yscale() == b.get_yscale()
        and np.allclose(a.get_ylim(), b.get_ylim())
        and list(a.get_yticks()) == list(b.get_yticks())
    )


def hide_x(ax: Axes) -> None:
    """An unlabeled shared x axis: no tick marks, tick labels, x label or bottom spine (no dangling ticks)."""
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlabel("")
    ax.spines["bottom"].set_visible(False)


def hide_y_labels(ax: Axes) -> None:
    """An inner column with the same y scale as its row's left panel: keep the axis line (and the
    grid), drop the repeated tick marks, tick labels and the y label.
    """
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_ylabel("")


def tidy_grid(axs) -> None:
    """Call after every panel's limits and ticks are set (and after ``plots.finish``). In each
    column, a panel whose x axis equals the bottom panel's loses its x axis; in each row, a panel
    whose y axis equals the left panel's loses its y tick labels.
    """
    a = np.atleast_2d(np.asarray(axs, dtype=object))
    nr, nc = a.shape
    for j in range(nc):
        col = [a[i, j] for i in range(nr) if a[i, j] is not None and a[i, j].axison]
        for ax in col[:-1]:
            if _same_x(ax, col[-1]):
                hide_x(ax)
    for i in range(nr):
        row = [a[i, j] for j in range(nc) if a[i, j] is not None and a[i, j].axison]
        for ax in row[1:]:
            if _same_y(ax, row[0]):
                hide_y_labels(ax)


# --- recall axis with INVALID strip (DESIGN s.7) ---------------------------------------------------------------------------
def recall_axis(ax: Axes, label: str | None = "Recall") -> None:
    """Recall on 0-1 with a reserved strip above the scale (inside the axes, above the trimmed
    spine) where INVALID cells are marked; the strip is the same in every recall panel so rows
    align.
    """
    ax.set_ylim(*RECALL_YLIM)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    if label:
        ax.set_ylabel(label)


def invalid_marks(ax: Axes, xs, color: str, **kw) -> None:
    """INVALID cells (no recall to report): a cross in the method's color in the strip of a
    ``recall_axis``. The figure legend then gets an ``INVALID condition`` entry automatically.
    ``finalize`` spaces the crosses that share an x (in call order, i.e. legend order)
    INVALID_STEP_PT apart, centred on the x, so they never touch; a series' dodge ``transform`` is
    not used for them.
    """
    xs = list(xs)
    if not xs:
        return
    kw.pop("transform", None)
    for x in xs:
        (ln,) = ax.plot(
            [x],
            [INVALID_Y],
            ls="",
            marker="x",
            ms=INVALID_MS,
            mew=1.0,
            color=color,
            mec=color,
            zorder=5,
            gid="invalid",
            **kw,
        )
        _reg(ax.figure, "invalid_pts").append((ax, x, ln))
    _reg(ax.figure, "")["invalid"] = True


def _space_invalid(fig) -> None:
    groups = {}
    for ax, x, ln in _reg(fig, "invalid_pts"):
        groups.setdefault((id(ax), float(x)), []).append((ax, ln))
    for members in groups.values():
        m = len(members)
        for k, (ax, ln) in enumerate(members):
            ln.set_transform(
                offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=(k - (m - 1) / 2) * INVALID_STEP_PT,
                    y=0,
                    units="points",
                )
            )


# --- dodge (DESIGN s.7) -----------------------------------------------------------------------------------------------------
DODGE_PT = 3.4
DODGE_NARROW_PT = (
    2.6  # [NEW 2026-10-05] narrow log-x panels: neighbouring x positions < 20 pt apart
)


def dodge(ax: Axes, i: int, n: int, step_pt: float = DODGE_PT):
    """Transform for series i of n, shifted horizontally by a fixed number of points so equal values
    at the same x stay visible side by side, whatever the axis scale. Pass it as ``transform=`` to
    the series and its band; state the dodge in the caption.
    """
    return offset_copy(
        ax.transData,
        fig=ax.figure,
        x=(i - (n - 1) / 2) * step_pt,
        y=0,
        units="points",
    )


# --- ECDF (DESIGN s.7) ------------------------------------------------------------------------------------------------------
def ecdf_line(p: np.ndarray, grid: np.ndarray, phase: int = 0, every: int = 10):
    """(x, y, markevery) of the empirical CDF of p on a log grid. The line starts at the smallest
    positive p (where the ECDF first leaves zero) and is connected from there; markers sit on the
    global grid every ``every`` points from ``phase`` so different series never share marker
    positions. No NaN, no isolated markers.
    """
    s = np.sort(np.asarray(p, float))
    s = s[s > 0]
    y = np.searchsorted(s, grid, side="right") / len(p)
    keep = grid > s[0]
    x = np.concatenate([[s[0]], grid[keep]])
    y = np.concatenate([[1 / len(p)], y[keep]])
    idx = np.nonzero(keep)[0]
    mark = np.concatenate([[False], (idx - phase) % every == 0])
    return x, y, list(mark)


# --- rate axis (DESIGN s.4) -------------------------------------------------------------------------------------------------
def rate_axis(ax: Axes, axis: str = "y", label: str | None = None) -> None:
    """Rejection-rate axis 0-1 with no artificial floor: symmetric-log (linear below 0.01), so a
    rate of 0 is drawn at 0 and small rates stay resolved; ticks 0, 0.01, 0.05, 0.2, 1.
    """
    kw = {"linthresh": 0.01, "linscale": 0.35}
    labels = ["0", "0.01", "0.05", "0.2", "1"]
    if axis == "y":
        ax.set_yscale("symlog", **kw)
        ax.set_ylim(*RATE_LIM)
        ax.yaxis.set_major_locator(FixedLocator(RATE_TICKS))
        ax.set_yticks(RATE_TICKS, labels)
        ax.yaxis.set_minor_formatter(NullFormatter())
        if label:
            ax.set_ylabel(label)
    else:
        ax.set_xscale("symlog", **kw)
        ax.set_xlim(*RATE_LIM)
        ax.set_xticks(RATE_TICKS, labels)
        ax.xaxis.set_minor_formatter(NullFormatter())
        if label:
            ax.set_xlabel(label)
    ax.minorticks_off()


# --- data-anchored labels (DESIGN s.4) ----------------------------------------------------------------------------------------
def label_line(ax: Axes, x: float, y: float, text: str, where: str = "above", **kw):
    """A direct label for a reference line or annotation, anchored to the data point (x, y) it names
    and offset at most MAX_LABEL_OFFSET_PT. ``where``: above / below / right / left.
    """
    d = MAX_LABEL_OFFSET_PT - 2
    off, ha, va = {
        "above": ((0, d), "left", "bottom"),
        "below": ((0, -d), "left", "top"),
        "right": ((d, 0), "left", "center"),
        "left": ((-d, 0), "right", "center"),
    }[where]
    return ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=off,
        textcoords="offset points",
        ha=kw.pop("ha", ha),
        va=kw.pop("va", va),
        fontsize=kw.pop("fontsize", S.ANNOT_PT),
        color=P.INK,
        **kw,
    )


# --- category banners (DESIGN s.7) --------------------------------------------------------------------------------------------
def banner(fig, ax: Axes, y: float, h: float, text: str):
    """Category header as a banner: a light band across the full figure width at data height y (of
    ``ax``), with a bold label at its left. Axes drawn over it must have a transparent background
    (``ax.patch.set_visible(False)``).
    """
    root = _root(fig)
    tr = blended_transform_factory(root.transFigure, ax.transData)
    band = Rectangle(
        (0, y - h / 2), 1, h, transform=tr, fc=P.tint(P.NEUTRAL, 0.14), ec="none", zorder=-5
    )
    band.set_in_layout(False)
    root.add_artist(band)
    t = ax.annotate(
        text,
        xy=(0, y),
        xycoords=("figure fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=S.ANNOT_PT,
        fontweight="bold",
        color=P.INK,
    )
    t.set_gid("banner")
    t.set_in_layout(False)
    return t


# --- method separation in row-wise figures (DESIGN s.7) ------------------------------------------------------------------
ROW_PITCH = 0.92  # sub-row pitch (data units of the shared row axis)
METHOD_GAP = 0.95  # extra space between two methods (about one sub-row)
BANNER_H = 1.25  # height of a category banner row


def method_rows(
    rows: Sequence, pitch: float = ROW_PITCH, gap: float = METHOD_GAP, banner_h: float = BANNER_H
):
    """Row positions of a row-wise figure, top-down. ``rows``: (name, None) for a category banner,
    or (name, subs) with the method's sub-row keys (e.g. regimes). Sub-rows of one method sit
    ``pitch`` apart; consecutive methods are separated by an extra ``gap``. Returns banners [(y,
    text)], groups [(name, [y of each sub-row])], subrows [(y, name, key)] and the y limits.
    """
    banners, groups, subrows = [], [], []
    y = 0.0
    for name, subs in rows:
        if subs is None:
            y -= banner_h / 2
            banners.append((y, name))
            y -= banner_h / 2 + gap * 0.6
            continue
        ys = [y - j * pitch for j in range(len(subs))]
        subrows += [(yy, name, k) for yy, k in zip(ys, subs, strict=False)]
        groups.append((name, ys))
        y = ys[-1] - pitch / 2 - gap
    return banners, groups, subrows, (y + gap, 0.05)


def method_labels(ax: Axes, groups: Sequence, offset_pt: float) -> None:
    """Bold method name right-aligned ``offset_pt`` left of the axes, on the method's first sub-row
    (a group header in the table sense), so each method reads as one block below its name.
    """
    for name, ys in groups:
        ax.annotate(
            name,
            xy=(0, ys[0]),
            xycoords=("axes fraction", "data"),
            xytext=(-offset_pt, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=S.ANNOT_PT,
            fontweight="bold",
            color=P.INK,
            linespacing=1.0,
            gid="row-label",
        )


# --- finalize ---------------------------------------------------------------------------------------------------------
def finalize(fig: Figure) -> Figure:
    """Resolve the registered headers and legends after a layout pass (letters left of the
    tick-label column, legend rows balanced to the width). Call last in ``build()``.
    """
    _space_invalid(fig)
    fig.draw_without_rendering()
    _resolve_auto(fig, fig.canvas.get_renderer())
    for _ in range(2):
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()
        _fit_legends(fig, r)
        _place_letters(fig, r)
    fig.draw_without_rendering()
    return fig

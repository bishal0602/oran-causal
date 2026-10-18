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
from matplotlib.transforms import Bbox, blended_transform_factory, offset_copy

from oran_causal.figures import palette as P
from oran_causal.figures import style as S

TAG_PT = S.TAG_PT  # panel tag: bold, next to the bold panel letter (DESIGN s.3)
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
        root._cdd.setdefault("callouts", [])
    if not hasattr(root, "_cdd"):
        root._cdd = {
            "headers": [],
            "legends": [],
            "invalid": False,
            "invalid_pts": [],
            "callouts": [],
        }
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
            fontweight="bold",
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


LEGEND_TOP_LOC = "outside upper center"
INVALID_LABEL = "INVALID condition"  # [G3 2026-10-12] was "INVALID cell (no recall)"; "INVALID condition (no recall)"
# is 20 pt wider and pushed the E4 page legend into one column (letter b then hit the y label)


def fig_legend(fig, handles: Sequence, labels: Sequence[str], where: str = "top", **kw):
    """Figure-level legend (DESIGN s.6). ``top`` (the standard): one
    centred, framed row above every panel header, wrapped into balanced rows when it does not fit the
    width (finalize). ``right``: a top-aligned column right of the panels. Never below or inside the
    axes. An ``INVALID condition`` entry is appended automatically when the figure drew INVALID marks.
    """
    handles, labels = list(handles), list(labels)
    if _reg(fig, "invalid") and INVALID_LABEL not in labels:
        handles.append(invalid_handle())
        labels.append(INVALID_LABEL)
    loc = {"top": LEGEND_TOP_LOC, "right": "outside right upper"}[where]
    ncol = len(labels) if where == "top" else 1
    leg = fig.legend(handles, labels, loc=loc, ncol=ncol, **kw)
    _reg(fig, "legends").append((fig, leg, handles, labels, where, kw))
    return leg


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
            leg = owner.legend(handles, labels, loc=LEGEND_TOP_LOC, ncol=ncol, **kw)
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
    """An inner column with the same y scale as its row's left panel: no ticks, tick labels or y label (the
    grid carries the scale).
    """
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_ylabel("")


def tidy_grid(axs) -> None:
    """Call after every panel's limits and ticks are set (and after ``plots.finish``). In each
    column, a panel whose x axis equals the bottom panel's loses its x title but keeps its tick labels
    (every facet row is readable on its own, DESIGN s.4); in each row, a panel whose y axis equals the
    left panel's loses its y tick labels.
    """
    a = np.atleast_2d(np.asarray(axs, dtype=object))
    nr, nc = a.shape
    for j in range(nc):
        col = [a[i, j] for i in range(nr) if a[i, j] is not None and a[i, j].axison]
        for ax in col[:-1]:
            if _same_x(ax, col[-1]):
                ax.set_xlabel("")
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
    3.0  # narrow log-x panels (neighbouring x < 20 pt apart); > checks.COINCIDE_FRAC x hero marker
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


# --- callouts (direction rule: auto-placed, leader line beyond MAX_LABEL_OFFSET_PT) --------------------------------------
CALLOUT_DIST_PT = (8, 12, 17, 23, 30, 38, 48, 60)
CALLOUT_DIRS = {
    name: (round(float(np.cos(np.radians(deg))), 3), round(float(np.sin(np.radians(deg))), 3))
    for name, deg in (
        ("e", 0),
        ("ene", 22.5),
        ("ne", 45),
        ("nne", 67.5),
        ("n", 90),
        ("nnw", 112.5),
        ("nw", 135),
        ("wnw", 157.5),
        ("w", 180),
        ("wsw", 202.5),
        ("sw", 225),
        ("ssw", 247.5),
        ("s", 270),
        ("sse", 292.5),
        ("se", 315),
        ("ese", 337.5),
    )
}
CALLOUT_PAD_PT = 1.5


def callout(
    ax: Axes, x: float, y: float, text: str, prefer: Sequence[str] = ("n", "e", "w", "s"), **kw
):
    """A label for the data point (x, y), placed by ``finalize`` at the first candidate offset (the
    preferred directions first, then the others, nearest first) whose box touches no other text,
    marker, data line, whisker or arrow and stays inside the axes. Beyond MAX_LABEL_OFFSET_PT it gets
    a thin ink leader line to the point.
    """
    t = ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=kw.pop("fontsize", S.ANNOT_PT),
        color=P.INK,
        linespacing=kw.pop("linespacing", 1.0),
        **kw,
    )
    t.set_gid("callout")
    order = list(prefer) + [d for d in CALLOUT_DIRS if d not in prefer]
    _reg(ax.figure, "callouts").append((ax, t, order))
    return t


def _obstacles(fig, ax, r, skip) -> tuple[list, list]:
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    from matplotlib.text import Text

    boxes, paths = [], []
    px = fig.dpi / 72
    for t in fig.findobj(Text):
        if t is skip or not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_gid() == "callout" and not getattr(t, "_placed", False):
            continue
        boxes.append(Text.get_window_extent(t, r))
    for ln in ax.get_lines():
        if not ln.get_visible() or ln.get_gid() in ("halo",):
            continue
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if ln.get_marker() not in (None, "", "None", " ") and ln.get_markersize() > 0:
            h = (ln.get_markersize() / 2 + 0.8) * px
            boxes += [Bbox.from_extents(a - h, b - h, a + h, b + h) for a, b in xy]
        if ln.get_linestyle() not in ("None", "", " ") and len(xy) > 1:
            paths.append(Path(xy))
    for c in ax.collections:
        if isinstance(c, LineCollection) and c.get_visible():
            tr = c.get_transform()
            paths += [Path(tr.transform(seg)) for seg in c.get_segments() if len(seg) > 1]
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch) and p.get_visible():
            paths.append(p.get_transform().transform_path(p.get_path()))
    return boxes, paths


def _callout_cost(bb, leader, boxes, paths, inner, own) -> float:
    hits = sum(bb.overlaps(b) for b in boxes) + sum(
        p.intersects_bbox(bb, filled=False) for p in paths
    )
    inside = inner.contains(bb.x0, bb.y0) and inner.contains(bb.x1, bb.y1)
    cross = 0
    if leader is not None:
        cross = sum(leader.intersects_bbox(b, filled=True) for b in boxes if not b.overlaps(own))
    return 10 * hits + 10 * (not inside) + 4 * cross


def _place_callouts(fig: Figure, r) -> None:
    """Each registered callout goes to the candidate (preferred directions first, distances
    nearest first) with the lowest cost: other texts, markers, data lines, whiskers and arrows it
    touches, leaving the axes, and marks its leader line crosses; ties go to the nearer candidate.
    The callout's own marker (and halo) is not an obstacle; beyond MAX_LABEL_OFFSET_PT it gets a
    thin ink leader line to the point.
    """
    from matplotlib.path import Path

    px = fig.dpi / 72
    pad = CALLOUT_PAD_PT * px
    for ax, t, order in _reg(fig, "callouts"):
        boxes, paths = _obstacles(fig, ax, r, t)
        ax_xy = ax.transData.transform(t.xy)
        own = Bbox.from_extents(
            ax_xy[0] - 4 * px, ax_xy[1] - 4 * px, ax_xy[0] + 4 * px, ax_xy[1] + 4 * px
        )
        boxes = [b for b in boxes if not b.contains(*ax_xy)]
        inner = ax.get_window_extent(r).padded(-1 * px)
        best = None
        for dist in CALLOUT_DIST_PT:
            for rank, d in enumerate(order):
                ux, uy = CALLOUT_DIRS[d]
                ha = "center" if abs(ux) < 0.3 else ("left" if ux > 0 else "right")
                va = "center" if abs(uy) < 0.3 else ("bottom" if uy > 0 else "top")
                t.set_horizontalalignment(ha)
                t.set_verticalalignment(va)
                t.xyann = (ux * dist, uy * dist)
                bb = t.get_window_extent(r).padded(pad)
                leader = None
                if dist > MAX_LABEL_OFFSET_PT:
                    end = (ax_xy[0] + ux * dist * px, ax_xy[1] + uy * dist * px)
                    start = (ax_xy[0] + ux * 4 * px, ax_xy[1] + uy * 4 * px)
                    leader = Path([start, end])
                cost = _callout_cost(bb, leader, boxes, paths, inner, own) + dist / 60 + rank / 100
                if best is None or cost < best[0]:
                    best = (cost, ha, va, t.xyann, dist)
        _, ha, va, off, dist = best
        t.set_horizontalalignment(ha)
        t.set_verticalalignment(va)
        t.xyann = off
        if dist > MAX_LABEL_OFFSET_PT:
            ux, uy = (np.sign(round(off[0], 3)), np.sign(round(off[1], 3)))
            rel = (
                0.5 if ha == "center" else (0.0 if ha == "left" else 1.0),
                0.5 if va == "center" else (0.0 if va == "bottom" else 1.0),
            )
            props = {
                "fontsize": t.get_fontsize(),
                "fontweight": t.get_fontweight(),
                "linespacing": t._linespacing,
            }
            t.remove()
            t = ax.annotate(
                t.get_text(),
                xy=t.xy,
                xycoords="data",
                xytext=off,
                textcoords="offset points",
                ha=ha,
                va=va,
                color=P.INK,
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.6,
                    "color": P.INK,
                    "shrinkA": 1.0,
                    "shrinkB": 4.0,
                    "relpos": rel,
                },
                **props,
            )
            t.set_gid("callout")
            t.arrow_patch.set_gid("leader")
        t._placed = True
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()


# --- finalize ---------------------------------------------------------------------------------------------------------
def finalize(fig: Figure) -> Figure:
    """Resolve the registered headers and legends after a layout pass (letters left of the
    tick-label column, legend rows balanced to the width). Call last in ``build()``.
    """
    _space_invalid(fig)
    for _ in range(2):
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()
        _fit_legends(fig, r)
        _place_letters(fig, r)
    if _reg(fig, "callouts"):
        fig.draw_without_rendering()
        _place_callouts(fig, fig.canvas.get_renderer())
    fig.draw_without_rendering()
    return fig

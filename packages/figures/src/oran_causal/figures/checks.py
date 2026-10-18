"""Layout checks run on every registered figure (tests/test_layout.py). Each returns a list of
problems (strings).

The classes they catch come from the 2026-10-05 figure review: overlapping text (letters, tags,
labels, legends), legends below or inside the axes, text intruding into another panel, labels
floating far from their data anchor, isolated markers from NaN-masked lines, coincident markers of
different series (a hidden series), bare tick marks on unlabeled shared axes, repeated tick labels
across a row, cramped facets, and artists outside the canvas. [2026-10-06] Two more from the Q8
check: markers half clipped at the axes edge, and INVALID crosses that touch.
"""

from itertools import combinations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text

MIN_AXES_IN = 1.0  # minimum width of a data panel (in)
TEXT_PAD_PT = 0.5  # text boxes may touch but not overlap by more than this
MAX_ANCHOR_PT = 8  # a data-anchored label sits within this offset of its anchor
COINCIDE_FRAC = 0.6  # markers of two series closer than this x marker size coincide


def _figs(fig):
    yield fig
    for sf in getattr(fig, "subfigs", []):
        yield from _figs(sf)


def _axes(fig):
    seen = set()  # the root figure also lists its subfigures' axes
    for f in _figs(fig):
        for ax in f.axes:
            if id(ax) not in seen:
                seen.add(id(ax))
                yield ax


def data_axes(fig) -> list:
    """Visible Cartesian panels (graphs and diagrams turn their axis off)."""
    return [a for a in _axes(fig) if a.get_visible() and a.axison and a.get_label() != "<colorbar>"]


def _visible(t: Text, r) -> bool:
    if not t.get_visible() or not t.get_text().strip():
        return False
    if isinstance(t, Annotation):
        try:
            return t._check_xy(r)
        except Exception:
            return True
    return True


def _drawn_ticklabels(axis) -> list[Text]:
    """Tick labels matplotlib draws: those whose tick lies inside the view interval (a locator also
    lists ticks beyond the limits, which are never drawn).
    """
    axis.get_ticklabels()
    lo, hi = sorted(axis.get_view_interval())
    eps = 1e-9 * max(1.0, abs(hi - lo))
    return [
        lab
        for tick in axis.get_major_ticks()
        if lo - eps <= tick.get_loc() <= hi + eps
        for lab in (tick.label1, tick.label2)
        if lab.get_visible()
    ]


def _pt(fig) -> float:
    return fig.dpi / 72


def texts(fig, r) -> list[tuple]:
    """(description, bbox, owner axes or None, rotated tick label?) for every visible text box; a
    legend counts as one box.
    """
    out = []
    for ax in _axes(fig):
        if not ax.get_visible():
            continue
        name = ax.get_label() or f"axes@{id(ax) % 997}"
        for t in ax.texts:
            if _visible(t, r):
                out.append(
                    (f"{name}: text '{t.get_text()[:30]}'", Text.get_window_extent(t, r), ax, False)
                )
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
            if _visible(lab, r):
                out.append(
                    (f"{name}: label '{lab.get_text()[:30]}'", lab.get_window_extent(r), ax, False)
                )
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                for t in _drawn_ticklabels(axis):
                    if _visible(t, r):
                        out.append(
                            (
                                f"{name}: tick '{t.get_text()}'",
                                t.get_window_extent(r),
                                ax,
                                t.get_rotation() % 90 != 0,
                            )
                        )
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            out.append((f"{name}: legend", leg.get_window_extent(r), ax, False))
    for f in _figs(fig):
        for t in f.texts:
            if _visible(t, r):
                out.append(
                    (f"figure text '{t.get_text()[:30]}'", t.get_window_extent(r), None, False)
                )
        for leg in f.legends:
            if leg.get_visible():
                out.append(("figure legend", leg.get_window_extent(r), None, False))
    return out


def _overlap(a, b, pad: float) -> bool:
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > pad) and (min(a.y1, b.y1) - max(a.y0, b.y0) > pad)


def text_overlaps(fig, r) -> list[str]:
    """Pairs of overlapping text boxes. Two rotated tick labels are skipped: their axis-aligned
    boxes overlap while the slanted glyphs do not.
    """
    items = texts(fig, r)
    pad = TEXT_PAD_PT * _pt(fig)
    return [
        f"text overlap: {a[0]} x {b[0]}"
        for a, b in combinations(items, 2)
        if _overlap(a[1], b[1], pad) and not (a[3] and b[3])
    ]


def text_intrusions(fig, r) -> list[str]:
    """No text box (except a figure legend's own row) reaches into another panel's data area."""
    pad = TEXT_PAD_PT * _pt(fig)
    out = []
    axes = data_axes(fig)
    for desc, bb, owner, _ in texts(fig, r):
        for ax in axes:
            if ax is owner:
                continue
            if _overlap(bb, ax.get_window_extent(r), pad):
                out.append(
                    f"text intrudes into another panel: {desc} -> {ax.get_label() or 'axes'}"
                )
    return out


def legend_placement(fig, r) -> list[str]:
    """Legends sit above the panels they span or in a column to their right; never inside a data
    area or below it.
    """
    out = []
    axes = data_axes(fig)
    pad = 2 * _pt(fig)
    legends = [(leg, leg.get_window_extent(r)) for f in _figs(fig) for leg in f.legends]
    legends += [
        (a.get_legend(), a.get_legend().get_window_extent(r))
        for a in _axes(fig)
        if a.get_legend() is not None and a.axison
    ]
    for leg, bb in legends:
        for ax in axes:
            ab = ax.get_window_extent(r)
            if min(bb.x1, ab.x1) - max(bb.x0, ab.x0) <= pad:  # no horizontal overlap: a side column
                continue
            if bb.y0 < ab.y1 - pad:
                where = "inside" if bb.y1 > ab.y0 + pad else "below"
                out.append(f"legend {where} a panel ({ax.get_label() or 'axes'})")
    return out


def anchored_labels(fig) -> list[str]:
    """Annotations anchored in data coordinates stay within MAX_ANCHOR_PT of their anchor."""
    out = []
    for ax in _axes(fig):
        for t in ax.texts:
            if (
                not isinstance(t, Annotation)
                or t.anncoords != "offset points"
                or t.xycoords != "data"
            ):
                continue
            if t.arrow_patch is not None:
                continue
            d = float(np.hypot(*t.xyann))
            if d > MAX_ANCHOR_PT and t.get_text().strip():
                out.append(f"label '{t.get_text()[:30]}' floats {d:.0f} pt from its data anchor")
    return out


def _finite(ax, xy: np.ndarray) -> np.ndarray:
    ok = np.isfinite(xy).all(axis=1)
    if ax.get_xscale() == "log":
        ok &= xy[:, 0] > 0
    if ax.get_yscale() == "log":
        ok &= xy[:, 1] > 0
    return ok


def _has_line(ln: Line2D) -> bool:
    return ln.get_linestyle() not in ("None", "", " ", "none") and ln.get_linewidth() > 0


def _has_marker(ln: Line2D) -> bool:
    return ln.get_marker() not in ("None", "", " ", "none", None)


def lone_markers(fig) -> list[str]:
    """A connected series never shows a point with no drawn neighbour (NaN / log masking)."""
    out = []
    for ax in data_axes(fig):
        for ln in ax.lines:
            if not (ln.get_visible() and _has_line(ln) and _has_marker(ln)):
                continue
            ok = _finite(ax, np.asarray(ln.get_xydata(), float))
            if ok.sum() == 0:
                continue
            nb = np.zeros_like(ok)
            nb[1:] |= ok[:-1]
            nb[:-1] |= ok[1:]
            lone = ok & ~nb
            if lone.any():
                out.append(f"isolated marker(s) in series '{ln.get_label()}' ({int(lone.sum())})")
    return out


def _marker_points(ax, ln: Line2D) -> np.ndarray:
    xy = np.asarray(ln.get_xydata(), float)
    idx = np.arange(len(xy))
    me = ln.get_markevery()
    if me is None:
        sel = idx
    elif isinstance(me, (int, np.integer)):
        sel = idx[::me]
    elif (
        isinstance(me, tuple) and len(me) == 2 and all(isinstance(v, (int, np.integer)) for v in me)
    ):
        sel = idx[me[0] :: me[1]]
    elif isinstance(me, (list, np.ndarray)) and len(me) == len(xy) and np.asarray(me).dtype == bool:
        sel = idx[np.asarray(me)]
    elif isinstance(me, (list, np.ndarray)):
        sel = np.asarray(me, int)
    else:
        sel = idx
    xy = xy[sel]
    xy = xy[_finite(ax, xy)]
    return ln.get_transform().transform(xy) if len(xy) else xy


def coincident_series(fig) -> list[str]:
    """Two series never put markers on the same spot (one would hide the other): dodge them."""
    out = []
    for ax in data_axes(fig):
        lines = [
            ln
            for ln in ax.lines
            if ln.get_visible()
            and _has_marker(ln)
            and ln.get_gid() not in ("invalid", "halo", "gate-ring")
            and ln.get_marker() not in ("_", "|", 0, 1, 2, 3)
        ]  # error-bar caps are not series
        pts = [(ln, _marker_points(ax, ln)) for ln in lines]
        for (a, pa), (b, pb) in combinations(pts, 2):
            if not len(pa) or not len(pb):
                continue
            tol = COINCIDE_FRAC * max(a.get_markersize(), b.get_markersize()) * _pt(fig)
            d = np.hypot(*(pa[:, None, :] - pb[None, :, :]).transpose(2, 0, 1))
            k = int((d < tol).sum())
            if k:
                out.append(
                    f"series '{a.get_label()}' and '{b.get_label()}' share {k} marker position(s)"
                )
    return out


def clipped_markers(fig) -> list[str]:
    """No marker is cut by its axes edge: a clipped series marker sits at least one marker radius
    inside the axes, or wholly outside it (deliberately out of view). Error-bar caps and the CS gate
    ring are not series.
    """
    out = []
    pt = _pt(fig)
    for ax in data_axes(fig):
        bb = ax.bbox
        for ln in ax.lines:
            if (
                not (ln.get_visible() and _has_marker(ln) and ln.get_clip_on())
                or ln.get_gid() == "gate-ring"
                or ln.get_marker() in ("_", "|", 0, 1, 2, 3)
            ):
                continue
            xy = _marker_points(ax, ln)
            if not len(xy):
                continue
            rad = ln.get_markersize() / 2 * pt - 0.5 * pt  # half a point of tolerance
            inside = (
                (xy[:, 0] >= bb.x0 + rad)
                & (xy[:, 0] <= bb.x1 - rad)
                & (xy[:, 1] >= bb.y0 + rad)
                & (xy[:, 1] <= bb.y1 - rad)
            )
            outside = (
                (xy[:, 0] < bb.x0 - rad)
                | (xy[:, 0] > bb.x1 + rad)
                | (xy[:, 1] < bb.y0 - rad)
                | (xy[:, 1] > bb.y1 + rad)
            )
            k = int((~inside & ~outside).sum())
            if k:
                out.append(
                    f"{k} marker(s) of '{ln.get_label()}' cut by the axes edge (add headroom)"
                )
    return out


def touching_crosses(fig) -> list[str]:
    """INVALID crosses never touch: centres at least one cross size apart."""
    out = []
    pt = _pt(fig)
    for ax in data_axes(fig):
        pts = [
            (ln, _marker_points(ax, ln))
            for ln in ax.lines
            if ln.get_gid() == "invalid" and ln.get_visible()
        ]
        xy = np.concatenate([p for _, p in pts if len(p)]) if pts else np.zeros((0, 2))
        if len(xy) < 2:
            continue
        size = max(ln.get_markersize() for ln, _ in pts) * pt
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        k = int((d[np.triu_indices(len(xy), 1)] < size).sum())
        if k:
            out.append(
                f"{k} pair(s) of INVALID crosses touch in {ax.get_label() or 'a panel'} (space them)"
            )
    return out


def tick_hygiene(fig, r) -> list[str]:
    """No bare tick marks on an unlabeled axis; no tick labels repeated across a row of identical y axes."""
    out = []
    axes = data_axes(fig)
    for ax in axes:
        for axis, side in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            ticks = [
                t
                for t in axis.get_major_ticks()
                if t.tick1line.get_visible() and t.tick1line.get_markersize()
            ]
            labels = [t for t in _drawn_ticklabels(axis) if _visible(t, r)]
            if ticks and not labels:
                out.append(
                    f"{ax.get_label() or 'axes'}: {side} tick marks without labels (hide the unlabeled axis)"
                )
    for a, b in combinations(axes, 2):
        ra, rb = a.get_window_extent(r), b.get_window_extent(r)
        if abs(ra.y0 - rb.y0) > 1 or abs(ra.y1 - rb.y1) > 1:
            continue
        la = [t.get_text() for t in _drawn_ticklabels(a.yaxis) if _visible(t, r)]
        lb = [t.get_text() for t in _drawn_ticklabels(b.yaxis) if _visible(t, r)]
        if la and la == lb and np.allclose(a.get_ylim(), b.get_ylim()):
            out.append(
                f"repeated y tick labels {la} across a row (show them on the left panel only)"
            )
    return out


def facet_width(fig, r) -> list[str]:
    out = []
    for ax in data_axes(fig):
        w = ax.get_window_extent(r).width / fig.dpi
        if w < MIN_AXES_IN - 1e-6:
            out.append(f"panel {ax.get_label() or 'axes'} is {w:.2f} in wide (< {MIN_AXES_IN} in)")
    return out


def on_canvas(fig, r) -> list[str]:
    """Every text box lies inside the figure canvas (the PDF is built at print width; nothing may
    spill over).
    """
    W, H = fig.bbox.width, fig.bbox.height
    pad = 1 * _pt(fig)
    return [
        f"outside the canvas: {d}"
        for d, bb, _, _ in texts(fig, r)
        if bb.x0 < -pad or bb.y0 < -pad or bb.x1 > W + pad or bb.y1 > H + pad
    ]


def audit(fig: Figure) -> dict[str, list[str]]:
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    return {
        "text overlap": text_overlaps(fig, r),
        "text intrusion": text_intrusions(fig, r),
        "legend placement": legend_placement(fig, r),
        "anchored labels": anchored_labels(fig),
        "lone markers": lone_markers(fig),
        "coincident series": coincident_series(fig),
        "tick hygiene": tick_hygiene(fig, r),
        "facet width": facet_width(fig, r),
        "on canvas": on_canvas(fig, r),
        "clipped markers": clipped_markers(fig),
        "touching crosses": touching_crosses(fig),
    }

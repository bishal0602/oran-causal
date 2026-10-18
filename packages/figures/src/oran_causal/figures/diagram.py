"""Primitives for architecture / pipeline diagrams drawn in matplotlib.

Work in a points-like coordinate system: `canvas(width_pt, height_pt)` returns an
axes whose data units are points at print size, so box sizes and font sizes match.

Roles (see DESIGN.md, "Diagrams"):
  * neutral component  -> white fill, NEUTRAL outline LW_BOX
  * entity component   -> entity tint 12 %, entity outline (e.g. one color per xApp)
  * proposed component -> HERO tint 18 %, HERO outline 1.2 pt, bold label
  * container (e.g. near-RT RIC) -> a filled FIELD card, no outline, bold title top-left
Connectors are plain ink arrows (straight or single elbow) with the label as plain text beside them.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from oran_causal.figures import palette as P
from oran_causal.figures import style as S

EDGE_PT = 2.0
RADIUS = 5.0  # one corner radius for every box (containers: RADIUS + 2)
LW_BOX = 0.8
LW_ARROW = 0.7
ENTITY_TINT = 0.12
HERO_TINT = 0.18


def canvas(width: str = "full", height_pt: float = 180) -> tuple[Figure, Axes]:
    import matplotlib.pyplot as plt

    S.apply()
    w_in = {"full": S.FULL, "half": S.HALF}[width]
    fig = plt.figure(figsize=(w_in, height_pt / 72))
    # [2026-10-12] the axes stops EDGE_PT short of each side edge (data units stay figure points): a
    # full-width axes plus the savefig pad came out 373.7 bp wide, wider than \textwidth (370.6 bp),
    # so LaTeX shrank 7 pt text to 6.94
    ax = fig.add_axes((EDGE_PT / (w_in * 72), 0, 1 - 2 * EDGE_PT / (w_in * 72), 1))
    ax.set_xlim(EDGE_PT, w_in * 72 - EDGE_PT)
    ax.set_ylim(height_pt, 0)  # y grows downward, like a page
    ax.axis("off")
    return fig, ax


def box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    sub: str | None = None,
    role: str = "neutral",
    color: str | None = None,
    radius: float | None = None,
    lw: float | None = None,
    ls: object = "-",
    gid: str | None = None,
) -> FancyBboxPatch:
    """role: 'neutral' | 'entity' (pass color=P.entity(k)) | 'proposed'. ``lw`` / ``ls`` override
    the outline (xApp boxes: 1.0 pt, DESIGN s.12.2); labels are always ink.
    """
    if radius is None:
        radius = RADIUS
    if role == "proposed":
        fc, ec, lw0, weight = P.tint(P.HERO, HERO_TINT), P.HERO, LW_BOX + 0.4, "bold"
    elif role == "entity":
        if color is None:
            raise ValueError("entity boxes need color=P.entity(k)")
        fc, ec, lw0, weight = P.tint(color, ENTITY_TINT), color, LW_BOX, "normal"
    else:
        fc, ec, lw0, weight = P.WHITE, P.NEUTRAL, LW_BOX, "normal"
    patch = ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            fc=fc,
            ec=ec,
            lw=lw0 if lw is None else lw,
            ls=ls,
            zorder=3,
            gid=gid,
        )
    )
    cy = y + h / 2
    if sub:
        ax.text(
            x + w / 2,
            cy - 3.5,
            label,
            ha="center",
            va="center",
            fontsize=S.TICK_PT,
            fontweight=weight,
            zorder=4,
        )
        ax.text(x + w / 2, cy + 5, sub, ha="center", va="center", fontsize=S.ANNOT_PT, zorder=4)
    else:
        ax.text(
            x + w / 2,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=S.TICK_PT,
            fontweight=weight,
            zorder=4,
        )
    return patch


def container(ax: Axes, x: float, y: float, w: float, h: float, title: str) -> None:
    """A grouping card (e.g. the near-RT RIC): FIELD fill, no outline, bold title top-left."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={RADIUS + 2}",
            fc=P.FIELD,
            ec="none",
            lw=0,
            zorder=1,
            gid="container",
        )
    )
    ax.text(
        x + 8,
        y + 10,
        title,
        ha="left",
        va="center",
        fontsize=S.ANNOT_PT,
        fontweight="bold",
        color=P.INK,
        zorder=4,
    )


def arrow(
    ax: Axes,
    points: list[tuple[float, float]],
    label: str | None = None,
    dashed: bool = False,
    label_at: tuple[float, float] | None = None,
) -> None:
    """Polyline arrow through `points` (orthogonal routing by listing corners). Head on the last
    point. Always ink, never an xApp's hue (DESIGN s.9, s.12.2).
    """
    ls = (0, (3, 1.6)) if dashed else "-"
    for a, b in zip(points[:-2], points[1:-1], strict=False):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=P.INK,
            lw=LW_ARROW,
            ls=ls,
            marker="",
            solid_capstyle="butt",
            zorder=2,
            gid="arrow",
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>,head_length=4,head_width=2.2",
            mutation_scale=1,
            color=P.INK,
            lw=LW_ARROW,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
            gid="arrow",
        )
    )
    if label:
        lx, ly = label_at or (
            (points[0][0] + points[1][0]) / 2,
            (points[0][1] + points[1][1]) / 2 - 4,
        )
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=S.ANNOT_PT, zorder=4)

"""Causal-graph drawing: NCP nodes on the left (circles), KPI nodes on the right
(rounded squares), edges flow left to right. Edge status encodes recovery vs
ground truth: correct = ink solid, spurious = SPURIOUS dashed, missed = gray dotted.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from oran_causal.figures import palette as P
from oran_causal.figures import style as S

EDGE_STYLE = {
    "correct": {"color": P.INK, "ls": "-", "lw": 0.8},
    "spurious": {"color": P.SPURIOUS, "ls": (0, (3, 1.6)), "lw": 0.9},
    "missed": {"color": P.MISSED, "ls": (0, (1, 1.6)), "lw": 0.9},
    "plain": {"color": P.INK, "ls": "-", "lw": 0.8},
}
NCP_COLOR = P.ENTITIES[0]
KPI_COLOR = P.ENTITIES[2]


@dataclass
class Edge:
    src: str
    dst: str
    status: str = "plain"  # correct | spurious | missed | plain


def bipartite_layout(
    ncps: Sequence[str], kpis: Sequence[str], x_gap: float = 3.0, y_gap: float = 1.0
) -> dict[str, tuple[float, float]]:
    pos = {}
    for col, names in ((0.0, ncps), (x_gap, kpis)):
        offset = (len(names) - 1) * y_gap / 2
        for i, n in enumerate(names):
            pos[n] = (col, offset - i * y_gap)
    return pos


def draw_causal_graph(
    ax: Axes,
    ncps: Sequence[str],
    kpis: Sequence[str],
    edges: Sequence[Edge],
    pos: dict[str, tuple[float, float]] | None = None,
    radius: float = 0.3,
    labels: dict[str, str] | None = None,
    legend: bool = True,
) -> None:
    pos = pos or bipartite_layout(ncps, kpis)
    labels = labels or {}
    ax.set_gid(
        "node-roles"
    )  # graph node roles (DESIGN s.8 / s.12), not the KPI identity slots of s.12.2
    kpi_set = set(kpis)
    for e in edges:
        st = EDGE_STYLE[e.status]
        (x1, y1), (x2, y2) = pos[e.src], pos[e.dst]
        bend = (
            0.5 if (e.src in kpi_set and e.dst in kpi_set) else 0.0
        )  # implicit KPI->KPI edges curve outward
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={-bend}",
                arrowstyle="-|>,head_length=3.2,head_width=1.8",
                shrinkA=radius * 30,
                shrinkB=radius * 30 + 1,
                mutation_scale=1,
                color=st["color"],
                lw=st["lw"],
                ls=st["ls"],
                zorder=2,
            )
        )
    for n in ncps:
        x, y = pos[n]
        ax.add_patch(
            Circle((x, y), radius, fc=P.tint(NCP_COLOR, 0.16), ec=NCP_COLOR, lw=0.8, zorder=3)
        )
        ax.text(x, y, labels.get(n, n), ha="center", va="center", fontsize=S.ANNOT_PT, zorder=4)
    for n in kpis:
        x, y = pos[n]
        ax.add_patch(
            FancyBboxPatch(
                (x - radius, y - radius),
                2 * radius,
                2 * radius,
                boxstyle=f"round,pad=0,rounding_size={radius * 0.33}",
                fc=P.tint(KPI_COLOR, 0.16),
                ec=KPI_COLOR,
                lw=0.8,
                zorder=3,
            )
        )
        ax.text(x, y, labels.get(n, n), ha="center", va="center", fontsize=S.ANNOT_PT, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    xs, ys = zip(*pos.values(), strict=False)
    ax.set_xlim(min(xs) - 1.2 * radius - 0.1, max(xs) + 1.2 * radius + 0.9)
    ax.set_ylim(min(ys) - 1.2 * radius, max(ys) + 1.2 * radius)
    if legend:
        statuses = [
            s for s in ("correct", "spurious", "missed") if any(e.status == s for e in edges)
        ]
        handles = [
            Line2D(
                [],
                [],
                color=EDGE_STYLE[s]["color"],
                ls=EDGE_STYLE[s]["ls"],
                lw=EDGE_STYLE[s]["lw"],
                label=s.capitalize(),
            )
            for s in statuses
        ]
        handles += [
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=6,
                mfc=P.tint(NCP_COLOR, 0.16),
                mec=NCP_COLOR,
                label="Control parameter (NCP)",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="s",
                ms=6,
                mfc=P.tint(KPI_COLOR, 0.16),
                mec=KPI_COLOR,
                label="KPI",
            ),
        ]
        ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, handlelength=1.6
        )

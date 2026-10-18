"""Environment truth-graph panel (DESIGN.md s.8 [NEW 2026-10-05], design G-A): a layered bipartite drawing of a
ground-truth causal graph from actions (knobs) to KPIs, sized for a narrow page column.

    g = EnvGraph(acts, kpis, edges, placebos=["Pl"])      edges: (src, dst, kind), kind in KINDS
    order = g.order()                                    barycentric node order (fewest crossings, natural on ties)
    draw_env_graph(ax, g, width_pt, height_pt)           draws on a canvas whose data units are points
    crossings(g, acts, kpis)                             crossing count of a node order (used by the tests)

Layout rules: actions in a left column (context Z on top, placebo columns at the bottom), KPIs in a right column.
Both columns are ordered barycentrically, which minimises crossings and is kept only when it beats the natural
order. A source with several children leaves through one short trunk and then splits. All plain edges into a KPI
merge into one arrow stub (bundled by target). Edges of another kind (gated, context) keep their own port and arrow,
so a crowded KPI stays legible. Lagged KPI -> KPI edges are arcs on the right of the KPI column, nested by span; the
outermost arc of each nest is labelled t−1 (the legend states it for all). The R3 behaviour policy on Z (E4) is a dotted purple curve on the left of the action column. Node
roles use the fixed entity slots of DESIGN s.12: action blue, KPI teal, context Z purple; the placebo action is
neutral gray (2026-10-06: the gold entity slot was ΔE 4 from the design-blind series then on the same page).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import permutations

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

from oran_causal.figures import palette as P
from oran_causal.figures import style as S

ACTION = P.ENTITIES[0]  # entity slot 1
KPI = P.ENTITIES[2]  # entity slot 3
PLACEBO = (
    P.NEUTRAL
)  # [2026-10-06] neutral gray (was entity slot 4 gold: ΔE 4 to the then design-blind series)
CONTEXT = P.ENTITIES[4]  # entity slot 5
KINDS = ("true", "gated", "context", "policy")
EDGE = {
    "true": {"color": P.INK, "ls": "-", "lw": 0.8},
    "gated": {"color": P.INK, "ls": (0, (4, 1.4, 1, 1.4)), "lw": 0.8},
    "context": {"color": CONTEXT, "ls": (0, (3, 1.6)), "lw": 0.8},
    "policy": {"color": CONTEXT, "ls": (0, (1.2, 1.4)), "lw": 0.9},
}
EDGE_LABEL = {
    "true": "Edge (one step later)",  # [G5] shorter: 8 pt Arial in the 1.37 in column
    "gated": "Gated edge",
    "context": "Z → KPI",
    "policy": "R3 policy on Z",
}
R = 6.3  # node radius (pt)
LAG = "t−1"
HEAD = "-|>,head_length=3.0,head_width=1.6"


@dataclass
class EnvGraph:
    acts: list
    kpis: list
    edges: list  # (src, dst, kind)
    placebos: list = field(default_factory=lambda: ["Pl"])

    @property
    def z(self) -> bool:
        return any(s == "Z" for s, _, _ in self.edges)

    @property
    def z_ordered(self) -> bool:
        """Z drives KPIs only (no R3 policy edges): it is ordered with the actions; otherwise it sits on top."""
        return self.z and not any(s == "Z" and k == "policy" for s, _, k in self.edges)

    def sources(self) -> list:
        return list(self.acts) + (["Z"] if self.z_ordered else [])

    def kinds(self) -> list:
        return [k for k in KINDS if any(e[2] == k for e in self.edges)]

    def order(self, sweeps: int = 6, exhaustive_max: int = 7) -> tuple[list, list]:
        """Node order with the fewest crossings found (the placebo and Z keep their places). Up to
        ``exhaustive_max`` KPIs every KPI order is tried, each with the barycentric action order
        plus adjacent-swap refinement; above it, alternating barycentric sweeps. Ties keep the order
        closest to the natural one (first found).
        """
        if len(self.kpis) <= exhaustive_max:
            best = None
            for kp in permutations(self.kpis):
                kp = list(kp)
                acts = _bary(self.sources(), self._children_pos(kp, self.sources()))
                acts, kp = self._transpose(crossings(self, acts, kp), acts, kp)
                c = crossings(self, acts, kp)
                if best is None or c < best[0]:
                    best = (c, acts, kp)
                if c == 0:
                    break
            return best[1], best[2]
        acts, kpis = self.sources(), list(self.kpis)
        best = (crossings(self, acts, kpis), acts, kpis)
        for _ in range(sweeps):
            kpis = _bary(kpis, self._parents_pos(acts, kpis))
            acts = _bary(acts, self._children_pos(kpis, acts))
            c = crossings(self, acts, kpis)
            if c < best[0]:
                best = (c, acts, kpis)
        return self._transpose(*best)

    def _transpose(self, c: int, acts: list, kpis: list) -> tuple[list, list]:
        """Greedy adjacent swaps on either column while they remove crossings (the 'transpose' refinement)."""
        improved = True
        while improved and c:
            improved = False
            for layer in (acts, kpis):
                for i in range(len(layer) - 1):
                    layer[i], layer[i + 1] = layer[i + 1], layer[i]
                    n = crossings(self, acts, kpis)
                    if n < c:
                        c, improved = n, True
                    else:
                        layer[i], layer[i + 1] = layer[i + 1], layer[i]
        return acts, kpis

    def _parents_pos(self, acts, kpis):
        ia = {a: i for i, a in enumerate(acts)}
        ik = {k: i for i, k in enumerate(kpis)}
        out = {}
        for k in kpis:
            ps = [ia[s] for s, d, _ in self.edges if d == k and s in ia]
            ks = [
                ik[s] + 0.5 for s, d, _ in self.edges if d == k and s in ik
            ]  # KPI-only parent: just below it
            v = ps or ks
            out[k] = sum(v) / len(v) if v else ik[k]
        return out

    def _children_pos(self, kpis, acts):
        ik = {k: i for i, k in enumerate(kpis)}
        out = {}
        for a in acts:
            cs = [ik[d] for s, d, _ in self.edges if s == a and d in ik]
            out[a] = sum(cs) / len(cs) if cs else None
        return out


def _bary(nodes: list, key: dict) -> list:
    """Stable sort by barycentre; a node without one keeps its current index as its key."""
    return [
        n
        for _, _, n in sorted(
            (key.get(n) if key.get(n) is not None else i, i, n) for i, n in enumerate(nodes)
        )
    ]


def crossings(g: EnvGraph, acts: Sequence, kpis: Sequence) -> int:
    """Crossings of a node order: pairs of action -> KPI edges whose end points interleave, plus
    pairs of KPI -> KPI arcs (all drawn on the same side) whose spans interleave.
    """
    ia = {a: i for i, a in enumerate(acts)}
    ik = {k: i for i, k in enumerate(kpis)}
    ak = [(ia[s], ik[d]) for s, d, _ in g.edges if s in ia and d in ik]
    n = sum(
        1 for i, (a1, k1) in enumerate(ak) for a2, k2 in ak[i + 1 :] if (a1 - a2) * (k1 - k2) < 0
    )
    arcs = [tuple(sorted((ik[s], ik[d]))) for s, d, _ in g.edges if s in ik and d in ik]
    n += sum(
        1
        for i, (a, b) in enumerate(arcs)
        for c, d in arcs[i + 1 :]
        if a < c < b < d or c < a < d < b
    )
    return n


def _column(n: int, top: float, bottom: float, step_max: float) -> list:
    if n == 1:
        return [(top + bottom) / 2]
    step = min(step_max, (top - bottom) / (n - 1))
    mid = (top + bottom) / 2
    return [mid + (n - 1) / 2 * step - i * step for i in range(n)]


def _color(n: str) -> str:
    if n == "Z":
        return CONTEXT
    if n in ("Pl", "Plc"):
        return PLACEBO
    return KPI if n.startswith("K") else ACTION


def node(
    ax: Axes,
    x: float,
    y: float,
    n: str,
    label_pt: float | None = None,
    r: float = R,
    gid: str | None = None,
) -> None:
    """One graph node of role ``n``: action blue circle, KPI teal rounded square, confounder dashed
    circle, placebo neutral; ``label_pt`` writes the name inside (None: unlabelled).
    """
    c = _color(n)
    if n.startswith("K"):
        patch = FancyBboxPatch(
            (x - r * 0.92, y - r * 0.92),
            1.84 * r,
            1.84 * r,
            boxstyle=f"round,pad=0,rounding_size={r * 0.35}",
            fc=P.tint(c, 0.16),
            ec=c,
            lw=0.7,
            zorder=4,
        )
    else:
        patch = Circle(
            (x, y),
            r,
            fc=P.tint(c, 0.16 if label_pt or n != "Z" else 0.28),
            ec=c,
            lw=0.7,
            zorder=4,
            ls=(0, (2, 1) if label_pt else (1.6, 0.9)) if n == "Z" else "-",
        )
    patch.set_gid(gid)
    ax.add_patch(patch)
    if label_pt:
        ax.text(x, y, n, ha="center", va="center", fontsize=label_pt, color=P.INK, zorder=5)


def edge(
    ax: Axes,
    pts,
    kind: str,
    head: bool,
    curve: bool = True,
    gid: str | None = None,
    style: dict | None = None,
    head_style: str = HEAD,
    capstyle: str | None = None,
):
    """One bundled edge of ``kind`` through the control points ``pts`` (a cubic chain, or straight
    when ``curve`` is false); ``style`` overrides the kind's colour / dash / width.
    """
    st = style or EDGE[kind]
    codes = [Path.MOVETO] + [Path.CURVE4 if curve else Path.LINETO] * (len(pts) - 1)
    patch = FancyArrowPatch(
        path=Path(pts, codes),
        arrowstyle=head_style if head else "-",
        mutation_scale=1,
        color=st["color"],
        lw=st["lw"],
        ls=st["ls"],
        zorder=2,
        shrinkA=0,
        shrinkB=0,
    )
    if capstyle:
        patch.set_capstyle(capstyle)
    patch.set_gid(gid)
    ax.add_patch(patch)
    return patch


def legend_handles(g: EnvGraph) -> tuple[list, list]:
    hs, ls = [], []
    nodes = [
        ("P0", "Action"),
        ("K0", "KPI"),
        ("Pl", "Placebo (Plc: R3)" if "Plc" in g.placebos else "Placebo action"),
    ]
    if g.z:
        nodes.append(("Z", "Confounder Z"))
    for n, lab in nodes:
        c = _color(n)
        hs.append(
            Line2D(
                [],
                [],
                ls="",
                marker="s" if n.startswith("K") else "o",
                ms=5,
                mfc=P.tint(c, 0.16),
                mec=c,
                mew=0.7,
            )
        )
        ls.append(lab)
    for k in g.kinds():
        hs.append(Line2D([], [], **EDGE[k]))
        ls.append(EDGE_LABEL[k])
    if any(s.startswith("K") for s, _, _ in g.edges):
        hs.append(Line2D([], [], ls="", marker=""))
        ls.append(f"{LAG}: previous step")
    return hs, ls


def draw_env_graph(
    ax: Axes,
    g: EnvGraph,
    width_pt: float,
    height_pt: float,
    legend: bool = True,
    *,
    xa: float = 17.0,
    xk: float | None = None,
    act_step: float = 21,
    kpi_step: float = 26,
    kpi_inset: float = 8,
    label_pt: float | None = None,
    lag_pt: float | None = None,
) -> dict:
    """Draw ``g`` on ``ax`` (axis off; data units = points, origin bottom left). Returns the node
    positions. Keywords (defaults = the Appendix A page): ``xa`` / ``xk`` the action and KPI column
    x (``xk`` default ``width_pt - 37``), ``act_step`` / ``kpi_step`` the largest node pitch of each
    column, ``kpi_inset`` how far the KPI column stays inside the action column's top and bottom,
    ``label_pt`` / ``lag_pt`` the node-label and t−1 sizes.
    """
    label_pt = S.MIN_PT if label_pt is None else label_pt  # [G5] 8 pt floor (was ANNOT_PT - 0.5)
    lag_pt = S.MIN_PT if lag_pt is None else lag_pt  # [G5] 8 pt floor (was ANNOT_PT - 1)
    ax.set_xlim(0, width_pt)
    ax.set_ylim(0, height_pt)
    ax.axis("off")
    ax.set_gid(
        "node-roles"
    )  # graph node roles: own canvas + legend, exempt from the entities audit
    acts, kpis = g.order()
    hs, ls = legend_handles(g) if legend else ([], [])
    floor = 10 + 9.2 * len(ls) + (8 if legend else 0)  # legend band at the bottom
    xk = width_pt - 37.0 if xk is None else xk
    left = (["Z"] if g.z and not g.z_ordered else []) + acts + list(g.placebos)
    ya = dict(zip(left, _column(len(left), height_pt - 10, floor, act_step), strict=False))
    yk = dict(
        zip(
            kpis,
            _column(len(kpis), height_pt - 10 - kpi_inset, floor + kpi_inset, kpi_step),
            strict=False,
        )
    )
    pos = {**{n: (xa, y) for n, y in ya.items()}, **{k: (xk, y) for k, y in yk.items()}}
    xs, xm = xa + R + 5, xk - R - 8  # split (source) and merge (target) points
    lw_curve = (xm - xs) * 0.45
    # source trunks for fan-out
    for s in left:
        if sum(1 for e in g.edges if e[0] == s and e[1] in yk) > 1:
            kind = "context" if s == "Z" else "true"
            edge(ax, [(xa + R, ya[s]), (xs, ya[s])], kind, head=False, curve=False)
    for k in kpis:
        ins = [(s, kind) for s, d, kind in g.edges if d == k and s in ya]
        plain = [s for s, kind in ins if kind == "true"]
        others = [(s, kind) for s, kind in ins if kind != "true"]
        ports = (["true"] if plain else []) + [s for s, _ in others]
        off = {p: (len(ports) - 1) / 2 * 3.4 - i * 3.4 for i, p in enumerate(ports)}
        for s in plain:
            x0 = xs if sum(1 for e in g.edges if e[0] == s and e[1] in yk) > 1 else xa + R
            y1 = yk[k] + off["true"]
            edge(
                ax,
                [(x0, ya[s]), (x0 + lw_curve, ya[s]), (xm - lw_curve, y1), (xm, y1)],
                "true",
                head=False,
            )
        if plain:
            edge(
                ax,
                [(xm, yk[k] + off["true"]), (xk - R - 0.5, yk[k] + off["true"])],
                "true",
                head=True,
                curve=False,
            )
        for s, kind in others:
            x0 = xs if sum(1 for e in g.edges if e[0] == s and e[1] in yk) > 1 else xa + R
            y1 = yk[k] + off[s]
            edge(
                ax,
                [(x0, ya[s]), (x0 + lw_curve, ya[s]), (xm - lw_curve, y1), (xk - R - 0.5, y1)],
                kind,
                head=True,
            )
    # lagged KPI -> KPI arcs, nested by span
    step = abs(next(iter(yk.values())) - list(yk.values())[1]) if len(yk) > 1 else 20
    arcs = sorted(
        (e for e in g.edges if e[0] in yk and e[1] in yk), key=lambda e: abs(yk[e[0]] - yk[e[1]])
    )
    spans = [tuple(sorted((yk[s], yk[d]))) for s, d, _ in arcs]
    for (s, d, kind), (lo, hi) in zip(arcs, spans, strict=False):
        y0, y1 = yk[s], yk[d]
        outer = not any(
            a <= lo and hi <= b and (a, b) != (lo, hi) for a, b in spans
        )  # label the nest's outer arc
        span = abs(y0 - y1) / step
        bulge = 6 + 6 * span
        ys0, ys1 = y0 + (-1.6 if y1 < y0 else 1.6), y1 + (1.6 if y1 < y0 else -1.6)
        edge(
            ax,
            [
                (xk + R, ys0),
                (xk + R + bulge * 1.33, ys0),
                (xk + R + bulge * 1.33, ys1),
                (xk + R + 0.5, ys1),
            ],
            kind,
            head=True,
        )
        if outer:
            ax.text(
                xk + R + bulge + 2,
                (y0 + y1) / 2,
                LAG,
                ha="left",
                va="center",
                fontsize=lag_pt,
                color=P.INK,
            )
    # R3 behaviour policy on Z: left of the action column
    for s, d, kind in g.edges:
        if kind == "policy":
            y0, y1 = ya[s], ya[d]
            b = 6 + 2.5 * abs(y0 - y1) / 21
            edge(
                ax,
                [(xa - R, y0), (xa - R - b, y0), (xa - R - b, y1), (xa - R - 0.5, y1)],
                kind,
                head=True,
            )
    for n, (x, y) in pos.items():
        node(ax, x, y, n, label_pt)
    if legend:
        leg = ax.legend(
            hs,
            ls,
            loc="lower left",
            bbox_to_anchor=(0, 0),
            ncol=1,
            fontsize=S.ANNOT_PT,
            handlelength=1.7,
            handletextpad=0.4,
            labelspacing=0.35,
            borderaxespad=0,
            frameon=False,
        )
        leg.set_in_layout(False)
    return pos

"""Section 5.4 (sec:envs): what the controlled environments are. One message: each environment
isolates one mechanism in a known causal graph, and the same graph is logged under three designs.

(a) The five ground-truth causal graphs as a row of small multiples (``_env_page.GRAPHS``, copied
read-only from the simulator's generator), actions left, KPIs right (DESIGN s.10). Node order is
``EnvGraph.order()`` (fewest crossings, as the appendix pages); every true edge is drawn, plain
edges into a KPI merge into one arrow stub and a source with several children leaves through one
trunk, so the rare-regime graph (16 action edges) stays a fan, not a tangle. No node labels: the
roles are the legend's (action, KPI, placebo action gray and childless, confounder purple dashed).
Lagged KPI -> KPI edges are arcs right of the KPI column, the outermost one marked t−1. E4's
confounded policy on Z (confounded logging only) is the dotted purple curve left of its action
column; it also feeds the diagnostic placebo of that design (the second gray node), the confounded
null edge that ``fig_logging_validity`` scores.

(b) The three logging designs as action-over-time strips of one action (E4's, range [0, 1], the one
environment logged under all three). Draws follow the generator's rules (``generate.py``
``_draw_iid`` / ``_draw_dither``, ``e4_logged`` behaviour policy at lam 1); the constants are
mirrored here and asserted against the simulator's constants at build time. The strips are an
illustration of the rule, not a logged corpus: no axis numbers. Strip titles use the words of
``fig_logging_validity``'s x axis (independent, dithered setpoint, confounded policy).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

from oran_causal.figures import envgraph as EG
from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures import style as S
from oran_causal.figures.figs import _env_page as E
from oran_causal.figures.figs.fig_logging_validity import DESIGN_LABEL
from oran_causal.figures.registry import FigSpec
from oran_causal.sim.controlled.datasets import e4_logged, generate
from oran_causal.sim.controlled.e4 import E4Env

WORLDS = ["E1", "E2", "E3", "E4", "E5"]
SRC_E4 = "packages/sim/src/oran_causal/sim/controlled/datasets/e4_logged.py"
SPEC = FigSpec(
    name="fig_envs",
    width="full",
    paper=True,
    sources=[E.SRC_ENV.format(w.lower()) for w in WORLDS] + [E.SRC_GEN, SRC_E4],
    caption_hint="Controlled environments and logging designs.",
)

# plain names (STORY, s05 text: "a clean linear graph, an edge active only in a rare regime, delayed
# harm through lagged KPI cascades, latent confounding, and the three mechanisms composed"); E4
# "Confounding" fits its 62 pt column
TITLE = {
    "E1": "Linear graph",
    "E2": "Rare regime",
    "E3": "Delayed harm",
    "E4": "Confounding",
    "E5": "Composed",
}
ACTION_EDGES = {
    "E1": 4,
    "E2": 16,
    "E3": 4,
    "E4": 1,
    "E5": 6,
}  # action -> KPI edges (true + gated), as the pages
LAG_EDGES = {"E1": 2, "E2": 0, "E3": 3, "E4": 0, "E5": 1}
STRIP_TITLE = [
    s.replace("\n", " ").capitalize() for s in DESIGN_LABEL
]  # same words as fig_logging_validity

# generator constants (mirrored; _check_constants asserts them against the simulator)
DITHER_BLOCKS, DITHER_ENVELOPE, DITHER_DELTA = 20, 0.25, 0.10
ETA_SD, LAM, GRID_N = 0.5, 1.0, 101
ROWS = 40  # rows drawn per strip (2 per setpoint block)
SEED = 7800

# geometry (pt). Canvas = data units in points.
W_PT, H_PT = S.FULL * 72, 2.6 * 72
LEG_H = 15.0  # legend band
HEAD = 14.0  # header band above each row (letter and titles, baseline HEADER_RISE_PT up)
G_H = 96.0  # graph canvas height
GAP = 6.0
STRIP_H = 27.0
R = 3.6  # node radius
PAD = 5.0  # node centre to canvas top / bottom
ACT_STEP, KPI_STEP = 14.0, 17.0
LINE = {"lw": 0.6}
HEAD_STYLE = "-|>,head_length=2.4,head_width=1.3"
# per graph: canvas x and width, action / KPI column x (canvas pt)
COLS = {
    "E1": (2.0, 64.0, 5.0, 38.0),
    "E2": (70.0, 88.0, 5.0, 82.0),
    "E3": (162.0, 72.0, 5.0, 40.0),
    "E4": (238.0, 54.0, 13.0, 46.0),
    "E5": (296.0, 72.0, 5.0, 48.0),
}
# strips: three equal widths, two equal gutters; a gutter holds the direct label of the next strip's
# line start
STRIP_X0, STRIP_W, STRIP_GUTTER = 14.0, 86.0, 48.0
STRIPS = [(STRIP_X0 + j * (STRIP_W + STRIP_GUTTER), STRIP_W) for j in range(3)]
TIME_DROP = 5.0  # shared time arrow below the strips (pt)
# [ENV2] the confounded behaviour policy is the logging design of that environment: gold (DESIGN s.5
# semantic role), round dots, so it cannot be read as the purple dashed confounder -> KPI edge
POLICY = {"color": P.DESIGN_ROLE, "ls": (0, (0.1, 1.7)), "lw": 1.1}


# --- (a) graphs -----------------------------------------------------------------------------------------------------
def _path(
    ax, pts, kind: str, head: bool, curve: bool = True, gid: str | None = None
) -> FancyArrowPatch:
    st = EG.EDGE[kind]
    codes = [Path.MOVETO] + [Path.CURVE4 if curve else Path.LINETO] * (len(pts) - 1)
    if kind == "policy":
        st = POLICY
    p = FancyArrowPatch(
        path=Path(pts, codes),
        arrowstyle=HEAD_STYLE if head else "-",
        mutation_scale=1,
        color=st["color"],
        lw=st["lw"] if kind == "policy" else LINE["lw"],
        ls=st["ls"],
        zorder=2,
        shrinkA=0,
        shrinkB=0,
        capstyle="round" if kind == "policy" else "butt",
    )
    if gid:
        p.set_gid(gid)
    ax.add_patch(p)
    return p


def _node(ax, x: float, y: float, n: str) -> None:
    c = EG._color(n)
    if n.startswith("K"):
        p = FancyBboxPatch(
            (x - R * 0.92, y - R * 0.92),
            1.84 * R,
            1.84 * R,
            boxstyle=f"round,pad=0,rounding_size={R * 0.35}",
            fc=P.tint(c, 0.16),
            ec=c,
            lw=0.7,
            zorder=4,
        )
    else:
        p = Circle(
            (x, y),
            R,
            fc=P.tint(c, 0.16 if n != "Z" else 0.28),
            ec=c,
            lw=0.7,
            zorder=4,
            ls=(0, (1.6, 0.9)) if n == "Z" else "-",
        )
    p.set_gid(f"node:{n}")
    ax.add_patch(p)


def draw_graph(ax, g: EG.EnvGraph, w: float, h: float, xa: float, xk: float) -> dict:
    """Compact layered drawing of ``g`` (no node labels); every edge patch carries gid
    ``edge:<src>-><dst>`` (a bundled stub carries ``stub:<dst>``). Returns the node positions.
    """
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_gid(
        "node-roles"
    )  # graph node roles: own canvas + legend, exempt from the entities audit
    acts, kpis = g.order()
    left = (["Z"] if g.z and not g.z_ordered else []) + acts + list(g.placebos)
    ya = dict(zip(left, EG._column(len(left), h - PAD, PAD, ACT_STEP), strict=False))
    yk = dict(zip(kpis, EG._column(len(kpis), h - PAD - 4, PAD + 4, KPI_STEP), strict=False))
    fan = {s: sum(1 for e in g.edges if e[0] == s and e[1] in yk) for s in left}
    xs, xm = xa + R + 3.5, xk - R - 5.5  # split (source) and merge (target) points
    bend = (xm - xs) * 0.45
    for s in left:
        if fan[s] > 1:
            _path(
                ax,
                [(xa + R, ya[s]), (xs, ya[s])],
                "context" if s == "Z" else "true",
                head=False,
                curve=False,
            )
    for k in kpis:
        ins = [(s, kind) for s, d, kind in g.edges if d == k and s in ya]
        plain = [s for s, kind in ins if kind == "true"]
        ports = (["true"] if plain else []) + [s for s, kind in ins if kind != "true"]
        off = {p: (len(ports) - 1) / 2 * 2.6 - i * 2.6 for i, p in enumerate(ports)}
        for s, kind in ins:
            x0 = xs if fan[s] > 1 else xa + R
            y1 = yk[k] + off["true" if kind == "true" else s]
            end = xm if kind == "true" else xk - R - 0.4
            _path(
                ax,
                [(x0, ya[s]), (x0 + bend, ya[s]), (end - bend, y1), (end, y1)],
                kind,
                head=kind != "true",
                gid=f"edge:{s}->{k}",
            )
        if plain:
            _path(
                ax,
                [(xm, yk[k] + off["true"]), (xk - R - 0.4, yk[k] + off["true"])],
                "true",
                head=True,
                curve=False,
                gid=f"stub:{k}",
            )
    step = abs(np.diff(sorted(yk.values()))).min() if len(yk) > 1 else KPI_STEP
    arcs = sorted(
        (e for e in g.edges if e[0] in yk and e[1] in yk), key=lambda e: abs(yk[e[0]] - yk[e[1]])
    )
    widest = None
    for s, d, kind in arcs:
        y0, y1 = yk[s], yk[d]
        bulge = 2.5 + 3.0 * abs(y0 - y1) / step
        sg = -1 if y1 < y0 else 1
        _path(
            ax,
            [
                (xk + R * 0.92, y0 + sg * 1.2),
                (xk + R + bulge * 1.33, y0 + sg * 1.2),
                (xk + R + bulge * 1.33, y1 - sg * 1.2),
                (xk + R * 0.92 + 0.4, y1 - sg * 1.2),
            ],
            kind,
            head=True,
            gid=f"edge:{s}->{d}",
        )
        if widest is None or bulge >= widest[0]:
            widest = (bulge, (y0 + y1) / 2)
    if widest:
        ax.text(
            xk + R + widest[0] + 1.5,
            widest[1],
            EG.LAG,
            ha="left",
            va="center",
            fontsize=S.MIN_PT,
            color=P.INK,
            gid="lag",
        )
    for s, d, kind in g.edges:
        if kind == "policy":
            y0, y1 = ya[s], ya[d]
            b = 4 + 2.0 * abs(y0 - y1) / ACT_STEP
            _path(
                ax,
                [(xa - R, y0), (xa - R - b, y0), (xa - R - b, y1), (xa - R - 0.4, y1)],
                kind,
                head=True,
                gid=f"edge:{s}->{d}",
            )
    pos = {**{n: (xa, y) for n, y in ya.items()}, **{k: (xk, y) for k, y in yk.items()}}
    for n, (x, y) in pos.items():
        _node(ax, x, y, n)
    return pos


def _check_graphs() -> None:
    for w in WORLDS:
        g = E.GRAPHS[w]
        act = [e for e in g.edges if e[0] in g.acts and e[1] in g.kpis]
        lag = [e for e in g.edges if e[0] in g.kpis and e[1] in g.kpis]
        assert len(act) == ACTION_EDGES[w] and all(k in ("true", "gated") for *_, k in act), (
            w,
            len(act),
        )
        assert len(lag) == LAG_EDGES[w], (w, len(lag))
        assert not any(s in g.placebos for s, _, _ in g.edges), f"{w}: a placebo action has a child"
    assert [w for w in WORLDS if any(k == "gated" for *_, k in E.GRAPHS[w].edges)] == ["E5"]
    assert [w for w in WORLDS if E.GRAPHS[w].z] == ["E4", "E5"]


# --- (b) logging designs --------------------------------------------------------------------------------------------
def _check_constants() -> None:
    assert generate.DITHER_DELTA == DITHER_DELTA
    assert generate.DITHER_BLOCKS == DITHER_BLOCKS
    assert generate.DITHER_ENVELOPE == DITHER_ENVELOPE
    assert e4_logged.ETA_SD == ETA_SD
    assert np.array_equal(e4_logged.GRID, np.arange(GRID_N, dtype=float) / 100.0)
    assert E4Env.id_ranges == [(0.0, 1.0)]
    assert ROWS % DITHER_BLOCKS == 0


def draws() -> dict:
    """One action (range [0, 1], E4's grid V) over ROWS rows under each design, by the generator's rules."""
    rng = np.random.default_rng(SEED)
    grid = np.arange(GRID_N) / 100.0
    iid = grid[rng.integers(0, GRID_N, ROWS)]
    sp = rng.uniform(0.5 - DITHER_ENVELOPE, 0.5 + DITHER_ENVELOPE, DITHER_BLOCKS)
    block = np.arange(ROWS) * DITHER_BLOCKS // ROWS
    dith = sp[block] + rng.uniform(-DITHER_DELTA, DITHER_DELTA, ROWS)
    assert dith.min() >= 0 and dith.max() <= 1, "dither left the range"
    z = rng.standard_normal(ROWS)
    wq = 0.5 + LAM * z + rng.normal(0, ETA_SD, ROWS)
    conf = grid[np.floor(100 * np.clip(wq, 0, 1) + 0.5).astype(int)]
    assert np.corrcoef(conf, z)[0, 1] > 0.6, "the confounded action must follow the confounder"
    assert np.corrcoef(iid, z)[0, 1] ** 2 < 0.1
    return {
        "iid": iid,
        "dither": dith,
        "setpoint": sp,
        "block": block,
        "conf": conf,
        "z": np.clip(0.5 + LAM * z, 0, 1),
    }


def _strip(ax, y, x) -> None:
    ax.plot(x, y, ls="", marker="o", ms=2.0, mfc=EG.ACTION, mec="none", zorder=3)
    ax.set_xlim(-1.5, ROWS + 0.5)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ax.spines:  # no per-strip axes: one shared time arrow under the row
        ax.spines[side].set_visible(False)


# --- figure ---------------------------------------------------------------------------------------------------------
def _ax(fig, x, y_top, w, h):
    return fig.add_axes((x / W_PT, 1 - (y_top + h) / H_PT, w / W_PT, h / H_PT))


def _title(ax, text: str, dx: float = 0.0):
    t = ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx, L.HEADER_RISE_PT),
        textcoords="offset points",
        ha="left",
        va="baseline",
        fontsize=L.TAG_PT,
        color=P.INK,
    )
    t.set_gid("panel-tag")
    return t


def _letter(ax, letter: str, dx: float):
    t = ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx, L.HEADER_RISE_PT),
        textcoords="offset points",
        ha="left",
        va="baseline",
        fontsize=S.PANEL_PT,
        fontweight="bold",
        color=P.INK,
    )
    t.set_gid("panel-letter")


def legend_items() -> tuple[list, list]:
    def node(n):
        c = EG._color(n)
        return Line2D(
            [],
            [],
            ls="",
            marker="s" if n.startswith("K") else "o",
            ms=5,
            mfc=P.tint(c, 0.16),
            mec=c,
            mew=0.7,
        )

    hs = [
        node("P0"),
        node("K0"),
        node("Pl"),
        node("Z"),
        Line2D([], [], color=P.INK, lw=0.8, ls=EG.EDGE["gated"]["ls"]),
        Line2D(
            [],
            [],
            color=POLICY["color"],
            lw=POLICY["lw"],
            ls=POLICY["ls"],
            dash_capstyle="round",
        ),
    ]
    return hs, ["Action", "KPI", "Placebo action", "Confounder", "Gated edge", "Confounded policy"]


def build():
    _check_graphs()
    _check_constants()
    S.apply()
    fig = plt.figure(figsize=(S.FULL, H_PT / 72))
    top_a = LEG_H + HEAD
    for i, w in enumerate(WORLDS):
        x, cw, xa, xk = COLS[w]
        ax = _ax(fig, x, top_a, cw, G_H)
        ax.set_label(f"graph {w}")
        draw_graph(ax, E.GRAPHS[w], cw, G_H, xa, xk)
        if i == 0:
            _letter(ax, "a", -x + 1)
        _title(ax, TITLE[w], dx=11.0 if i == 0 else 0.0)
    d = draws()
    top_b = top_a + G_H + GAP + HEAD
    rows = np.arange(ROWS)
    strips = []
    for j, ((x, sw), key) in enumerate(zip(STRIPS, ("iid", "dither", "conf"), strict=False)):
        ax = _ax(fig, x, top_b, sw, STRIP_H)
        ax.set_label(f"strip {key}")
        strips.append(ax)
        # [ENV2] what the action is set from, as a thin step line under the action dots: the gold
        # setpoint (one step per block) and the purple confounder (one step per row); its direct
        # label sits in the gutter at its start
        line = {
            "dither": (
                d["setpoint"],
                np.r_[np.searchsorted(d["block"], np.arange(DITHER_BLOCKS)), ROWS],
                P.DESIGN_ROLE,
                "sem:design",
                "setpoint",
            ),
            "conf": (d["z"], np.arange(ROWS + 1), P.CONFOUNDER, "sem:confounder", "confounder"),
        }.get(key)
        if line:
            v, edges, c, gid, text = line
            ax.stairs(v, edges - 0.5, baseline=None, color=c, lw=0.8, zorder=2, gid=gid)
            L.label_line(ax, -0.5, v[0], text, where="left")
        _strip(ax, d[key], rows)
        _title(ax, STRIP_TITLE[j], dx=11.0 if j == 0 else 0.0)
    _letter(strips[0], "b", -STRIPS[0][0] + 1)
    strips[0].set_ylabel("Action", fontsize=S.MIN_PT, labelpad=1)
    y = 1 - (top_b + STRIP_H + TIME_DROP) / H_PT
    x0, x1 = STRIPS[0][0] / W_PT, (STRIPS[-1][0] + STRIP_W) / W_PT
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle=HEAD_STYLE,
            mutation_scale=1.4,
            color=P.AXIS,
            lw=S.LW_AXIS,
            shrinkA=0,
            shrinkB=0,
        )
    )
    fig.text(
        (x0 + x1) / 2,
        y - 2 / H_PT,
        "Time (logged rows)",
        ha="center",
        va="top",
        fontsize=S.MIN_PT,
        color=P.INK,
    )
    hs, ls = legend_items()
    fig.legend(
        hs,
        ls,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(ls),
        fontsize=S.LEGEND_PT,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.1,
    )
    return L.finalize(fig)

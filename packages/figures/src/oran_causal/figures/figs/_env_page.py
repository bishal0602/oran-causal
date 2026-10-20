"""Appendix A template: one page per Study 4 environment: (a) the environment's ground-truth causal graph in a narrow left column
(oran_causal.figures.envgraph), right of it a 2 x 2 grid of (b, c) the truth-null
rejection rate vs n and (d, e) recall vs n in R1 and R2, for the design-based tests, design-adjusted
partial correlation and two design-blind methods. The logging designs R1 / R2 are defined in the
caption. E4 has no truth-null candidate in R1 / R2 (REPORT V0), so its rate panels show the placebo
edge (lam 1 cells, the only E4 R1 / R2 cells). The cell table for every method is ``tab_study4_env``
(LaTeX).

Graphs are the generator's ground truth, copied here read-only (figures read no simulator state):
``oran_causal/sim/controlled/e1.py`` ... ``e5.py`` adjacency lists ((kpi, source) pairs; a source >=
num_params is a lagged KPI) and ``oran_causal/sim/controlled/datasets/generate.py`` (truth_for,
candidates_for, the placebo columns, E4 R3 / E5 notes). Every edge acts one step later (row t ->
KPIs at t + 1). The placebo action Pl never has a child. E4: the confounder Z drives K0 in every
regime; in R3 it is the observed context on which the logged behaviour policy draws P0 and the
diagnostic placebo Plc. E5 (fully composed, mode "do"): Z is latent and drives K1 only; P0 -> K1 and
K2 -> K1 act only inside the gate.
"""

import matplotlib.pyplot as plt

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.data import load_csv
from oran_causal.figures.envgraph import EnvGraph, draw_env_graph
from oran_causal.figures.figs import _study4 as T

SRC_ENV = "packages/sim/src/oran_causal/sim/controlled/{}.py"
SRC_GEN = "packages/sim/src/oran_causal/sim/controlled/datasets/generate.py"


def _adj(p: int, pairs) -> list:
    """(kpi, source) adjacency pairs of an env -> (source, target) names (source >= p is the lagged
    KPI source - p).
    """
    return [(f"P{s}" if s < p else f"K{s - p}", f"K{k}") for k, s in pairs]


def _graph(p: int, k: int, pairs, gated=(), extra=(), placebos=("Pl",)) -> EnvGraph:
    e = [(s, d, "gated" if (s, d) in gated else "true") for s, d in _adj(p, pairs)]
    return EnvGraph(
        [f"P{i}" for i in range(p)], [f"K{j}" for j in range(k)], e + list(extra), list(placebos)
    )


GRAPHS = {
    "E1": _graph(4, 4, [(0, 0), (1, 1), (2, 2), (3, 3), (2, 4), (3, 5)]),
    "E2": _graph(
        8,
        6,
        [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 3),
            (2, 0),
            (3, 1),
            (3, 4),
            (3, 5),
            (4, 1),
            (4, 4),
            (4, 5),
            (5, 0),
            (5, 6),
            (5, 7),
        ],
    ),
    "E3": _graph(4, 5, [(0, 0), (1, 4), (2, 1), (2, 5), (3, 2), (3, 5), (4, 3)]),
    "E4": _graph(
        1,
        1,
        [(0, 0)],
        extra=[("Z", "K0", "context"), ("Z", "P0", "policy"), ("Z", "Plc", "policy")],
        placebos=("Pl", "Plc"),
    ),
    "E5": _graph(
        4,
        4,
        [(0, 0), (2, 0), (1, 1), (1, 2), (3, 3), (1, 0), (1, 6)],
        gated={("P0", "K1"), ("K2", "K1")},
        extra=[("Z", "K1", "context")],
    ),
}
# one-line mechanism of each environment (caption hints); E2 without "gated" to
# match the paper (App. A title; no gated edge is drawn)
BLURB = {
    "E1": "clean identifiable linear graph",
    "E2": "edge active only in a rare regime",
    "E3": "delayed harm through lagged KPI cascades",
    "E4": "latent confounding",
    "E5": "E2-E4 mechanisms composed",
}
ARMS = [
    "pmrt_nl_eq",
    "pmrt_eq",
    "pcorr_eq",
    "corr",
    "pcorr_native",
]  # legend order (oran_causal.figures.methods)
COLOUR_NOTE = " Colour identifies the method (the same in every figure); it does not mean validity"
PAGE_H = 3.6  # in
GRAPH_W, GRAPH_X = (
    1.30,
    0.03,
)  # graph canvas width and left margin (in); the left subfigure is 1.37 in wide
GRAPH_TOP, GRAPH_BOT = 0.22, 0.04


def sources(world: str) -> list[str]:
    return [SRC_ENV.format(world.lower()), SRC_GEN, T.SRC_CELLS, T.SRC_RECALL]


def _xaxis(ax):
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(T.NS, ["0.5", "1", "4", "8", "24"])
    ax.set(xlim=(330, 36000), xlabel="n (×10³)")


def build_page(world: str):
    S.apply()
    fig = plt.figure(figsize=(S.FULL, PAGE_H), layout="constrained")
    _, right = fig.subfigures(1, 2, width_ratios=[1, 2.75])
    g = fig.add_axes(
        (
            GRAPH_X / S.FULL,
            GRAPH_BOT / PAGE_H,
            GRAPH_W / S.FULL,
            (PAGE_H - GRAPH_TOP - GRAPH_BOT) / PAGE_H,
        )
    )  # fixed size: not managed by the layout
    draw_env_graph(g, GRAPHS[world], GRAPH_W * 72, (PAGE_H - GRAPH_TOP - GRAPH_BOT) * 72)
    axs = right.subplots(2, 2)
    c = T.cells()
    c = c[c.world == world]
    r = load_csv(*T.EVAL, "csv", "recall_vs_n.csv")
    r = r[r.world == world]
    col, ylab = "null_raw", "Truth-null rate"
    if world == "E4":  # R1 / R2 cells are lam 1; placebo edge only
        c, r = c[c.lam == 1.0], r[r.lam == 1.0]
        col, ylab = "plac_raw", "Placebo-edge rate"
    for j, reg in enumerate(("R1", "R2")):
        va, ra = axs[0, j], axs[1, j]
        for arm in ARMS:  # 1.5 in panels: n 500 / 1000 ~15 pt apart -> narrow dodge
            q = c[(c.arm == arm) & (c.regime == reg)].sort_values("n")
            T.series(
                va,
                q.n,
                q[f"{col}_rate"],
                q[f"{col}_lo"],
                q[f"{col}_hi"],
                arm,
                ARMS,
                L.DODGE_NARROW_PT,
            )
            q = r[(r.arm == arm) & (r.regime == reg)].sort_values("n")
            ok = q[q.state != "INVALID"]
            T.series(
                ra,
                ok.n,
                ok.recall_mean,
                ok.recall_lo,
                ok.recall_hi,
                arm,
                ARMS,
                L.DODGE_NARROW_PT,
                invalid_x=q[q.state == "INVALID"].n,
            )
        plots.ref_line(va, 0.05)
        _xaxis(va)
        L.rate_axis(va, "y", ylab if j == 0 else None)
        plots.finish(va)
        _xaxis(ra)
        L.recall_axis(ra, "Recall" if j == 0 else None)
        plots.finish(ra)
    L.tidy_grid(axs)
    L.panel_header(g, "a", "Causal graph", inset=True)
    L.panel_header(axs[0, 0], "b", f"{world}, R1")
    L.panel_header(axs[0, 1], "c", f"{world}, R2")
    L.panel_header(axs[1, 0], "d")
    L.panel_header(axs[1, 1], "e")
    M.legend(right, ARMS, hero=T.HERO)
    return L.finalize(fig)

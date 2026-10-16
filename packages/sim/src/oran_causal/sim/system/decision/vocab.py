"""Hypothesis space of the system-level discovery studies.

A hypothesis (f, r, k) pairs a knob family f, a relation r of the affected cells to the unit's
cell (own, neighbours, far) and a KPI k: 4 x 3 x 5 = 60 hypotheses in the "dir" orientation (the
effect of a knob-value increase). The gate DAG puts the load hypothesis of each (f, r) group
above the other four KPIs of the group. Importing this module never loads the plant.
"""

import numpy as np

from oran_causal.core.declare import Structure

Hypothesis = tuple[str, str, str]

FAMILIES = ("carrier", "sleep", "ptx", "prot_min")
RELATIONS = ("own", "nbr", "far")
KPIS = ("pv", "v", "e", "rlf", "load")
CELL_KPIS = ("pv", "e", "v", "rlf", "load", "prb")
ORIENTS = ("dir", "act")
PRIMARY_ORIENT = "dir"
HYPOTHESES: tuple[Hypothesis, ...] = tuple(
    (f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS
)

MODES = ("accept", "half", "reject", "accept+rb")
LEVEL = {"accept": 1.0, "half": 0.5, "reject": 0.0, "accept+rb": 1.0}
LEVEL_ARR = np.array([LEVEL[m] for m in MODES])

PHYSICS_PRIOR: dict[Hypothesis, int] = {
    ("sleep", "nbr", "load"): 1,
    ("sleep", "nbr", "pv"): 1,
    ("sleep", "nbr", "v"): 1,
    ("sleep", "own", "load"): -1,
    ("sleep", "own", "e"): -1,
    ("carrier", "own", "e"): 1,
    ("carrier", "own", "pv"): -1,
    ("carrier", "own", "v"): -1,
    ("ptx", "own", "e"): 1,
    ("ptx", "own", "load"): 1,
    ("ptx", "nbr", "load"): -1,
    ("prot_min", "own", "pv"): -1,
}

KPI_GATE = "load"
GATE_DELTA = 0.2


def gate_parents(hyps: tuple[Hypothesis, ...] = HYPOTHESES) -> list[list[int]]:
    """Returns the mediator-first DAG: (f, r, load) is the parent of (f, r, k) for k != load."""
    idx = {h: i for i, h in enumerate(hyps)}
    return [[] if h[2] == KPI_GATE else [idx[(h[0], h[1], KPI_GATE)]] for h in hyps]


def gate_graph(
    hyps: tuple[Hypothesis, ...] = HYPOTHESES, delta: float = GATE_DELTA
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the graphical Holm design (initial weights, transition matrix).

    The initial weight is spread evenly over the load roots. A rejected load passes its weight to
    the other KPIs of its group; a rejected child passes 1 - delta to its siblings and delta to the
    loads of the other groups.
    """
    n = len(hyps)
    loads = [i for i, h in enumerate(hyps) if h[2] == KPI_GATE]
    w0 = np.zeros(n)
    w0[loads] = 1.0 / len(loads)
    graph = np.zeros((n, n))
    for i, h in enumerate(hyps):
        grp = [j for j, g in enumerate(hyps) if g[:2] == h[:2] and j != i]
        if h[2] == KPI_GATE:
            graph[i, grp] = 1.0 / len(grp)
        else:
            sib = [j for j in grp if hyps[j][2] != KPI_GATE]
            other = [j for j in loads if hyps[j][:2] != h[:2]]
            graph[i, sib] = (1.0 - delta) / len(sib)
            graph[i, other] = delta / len(other)
    return w0, graph


def pfilter_groups(hyps: tuple[Hypothesis, ...] = HYPOTHESES) -> list[str]:
    """Returns the p-filter group label "f|r" of every hypothesis."""
    return ["|".join(h[:2]) for h in hyps]


def structure(hyps: tuple[Hypothesis, ...] = HYPOTHESES, delta: float = GATE_DELTA) -> Structure:
    """Returns the declaration structure (gate DAG, graphical Holm design, p-filter groups)."""
    w0, graph = gate_graph(hyps, delta)
    return Structure(
        parents=gate_parents(hyps), holm_weights=w0, holm_graph=graph, groups=pfilter_groups(hyps)
    )

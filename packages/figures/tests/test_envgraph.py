"""Environment truth-graph panels (oran_causal.figures.envgraph, DESIGN s.8): node
order and crossing counts.
"""

import pytest

from oran_causal.figures.envgraph import EnvGraph, crossings
from oran_causal.figures.figs._env_page import GRAPHS

# fewest crossings of the drawn order (action -> KPI pairs + interleaved KPI arcs); E2's fan-out
# cannot reach 0
MAX_CROSSINGS = {"E1": 0, "E2": 6, "E3": 0, "E4": 0, "E5": 0}


@pytest.mark.parametrize("world", list(GRAPHS))
def test_graph_order_crossings(world):
    g = GRAPHS[world]
    acts, kpis = g.order()
    assert sorted(acts) == sorted(g.sources())
    assert sorted(kpis) == sorted(g.kpis)
    drawn = crossings(g, acts, kpis)
    assert drawn <= crossings(g, g.sources(), g.kpis), "ordering must never add crossings"
    assert drawn <= MAX_CROSSINGS[world], f"{world}: {drawn} crossings"


@pytest.mark.parametrize("world", list(GRAPHS))
def test_graph_edges_have_nodes(world):
    g = GRAPHS[world]
    nodes = set(g.acts) | set(g.kpis) | set(g.placebos) | ({"Z"} if g.z else set())
    for s, d, kind in g.edges:
        assert s in nodes
        assert d in nodes
        assert kind in ("true", "gated", "context", "policy")
    assert not any(s in g.placebos for s, _, _ in g.edges), "a placebo action never has a child"


def test_crossings_counts_interleaving():
    g = EnvGraph(["P0", "P1"], ["K0", "K1"], [("P0", "K1", "true"), ("P1", "K0", "true")])
    assert crossings(g, ["P0", "P1"], ["K0", "K1"]) == 1
    acts, kpis = g.order()
    assert crossings(g, acts, kpis) == 0

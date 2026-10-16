import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oran_causal.sim.system.decision import vocab

DATA = Path(__file__).parent / "data"

pytestmark = pytest.mark.claim("DEF-SYS-HYP")


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads((DATA / "decision_vocab.json").read_text())


def test_sixty_hypotheses_in_family_relation_kpi_order(golden: dict) -> None:
    assert len(vocab.HYPOTHESES) == 60 == len(set(vocab.HYPOTHESES))
    assert ["|".join(h) for h in vocab.HYPOTHESES] == golden["hypotheses"]
    assert list(vocab.FAMILIES) == golden["families"]
    assert list(vocab.RELATIONS) == golden["relations"]
    assert list(vocab.KPIS) == golden["kpis"] == golden["gt_kpis"] == golden["cu_kpis"]
    assert list(vocab.CELL_KPIS) == golden["cell_kpis"]
    assert golden["primary_orient"] == vocab.PRIMARY_ORIENT
    assert list(vocab.ORIENTS) == golden["orients"]


def test_modes_levels_and_physics_prior(golden: dict) -> None:
    assert list(vocab.MODES) == golden["modes"]
    assert golden["level"] == vocab.LEVEL
    assert np.array_equal(vocab.LEVEL_ARR, [golden["level"][m] for m in golden["modes"]])
    prior = {"|".join(h): s for h, s in vocab.PHYSICS_PRIOR.items()}
    assert prior == golden["physics_prior"]
    assert set(vocab.PHYSICS_PRIOR) <= set(vocab.HYPOTHESES)


def test_gate_structure_equals_the_old_layer(golden: dict) -> None:
    st = vocab.structure()
    assert st.parents is not None
    assert st.holm_weights is not None
    assert st.holm_graph is not None
    assert st.groups is not None
    assert [list(p) for p in st.parents] == golden["parents"]
    assert np.array_equal(st.holm_weights, np.array(golden["holm_weights"]))
    assert np.array_equal(st.holm_graph, np.array(golden["holm_graph"]))
    assert list(st.groups) == golden["groups"]


def test_gate_dag_roots_are_the_twelve_loads() -> None:
    parents = vocab.gate_parents()
    roots = [vocab.HYPOTHESES[i] for i, p in enumerate(parents) if not p]
    assert len(roots) == 12
    assert all(h[2] == vocab.KPI_GATE for h in roots)
    for i, p in enumerate(parents):
        if p:
            assert vocab.HYPOTHESES[p[0]][:2] == vocab.HYPOTHESES[i][:2]
    w0, graph = vocab.gate_graph()
    assert w0.sum() == pytest.approx(1.0)
    assert np.allclose(graph.sum(1), 1.0)
    assert np.all(np.diag(graph) == 0)


def test_vocab_gt_and_records_do_not_load_the_plant() -> None:
    code = (
        "import sys; import oran_causal.sim.system.decision.records; "
        "print(any(m.startswith('oran_causal.sim.system.') and 'decision' not in m "
        "for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

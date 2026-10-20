import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oran_causal.sim.system.decision import vocab as dv

DATA = Path(__file__).parent / "data"

pytestmark = pytest.mark.claim("DEF-SYS-HYP")


@pytest.fixture(scope="module")
def vocab() -> dict:
    return json.loads((DATA / "decision_vocab.json").read_text())


def test_sixty_hypotheses_in_family_relation_kpi_order(vocab: dict) -> None:
    assert len(dv.HYPOTHESES) == 60 == len(set(dv.HYPOTHESES))
    assert ["|".join(h) for h in dv.HYPOTHESES] == vocab["hypotheses"]
    assert list(dv.FAMILIES) == vocab["families"]
    assert list(dv.RELATIONS) == vocab["relations"]
    assert list(dv.KPIS) == vocab["kpis"] == vocab["gt_kpis"] == vocab["cu_kpis"]
    assert list(dv.CELL_KPIS) == vocab["cell_kpis"]
    assert vocab["primary_orient"] == dv.PRIMARY_ORIENT
    assert list(dv.ORIENTS) == vocab["orients"]


def test_modes_levels_and_physics_prior(vocab: dict) -> None:
    assert list(dv.MODES) == vocab["modes"]
    assert vocab["level"] == dv.LEVEL
    assert np.array_equal(dv.LEVEL_ARR, [vocab["level"][m] for m in vocab["modes"]])
    prior = {"|".join(h): s for h, s in dv.PHYSICS_PRIOR.items()}
    assert prior == vocab["physics_prior"]
    assert set(dv.PHYSICS_PRIOR) <= set(dv.HYPOTHESES)


def test_gate_dag_roots_are_the_twelve_loads() -> None:
    parents = dv.gate_parents()
    roots = [dv.HYPOTHESES[i] for i, p in enumerate(parents) if not p]
    assert len(roots) == 12
    assert all(h[2] == dv.KPI_GATE for h in roots)
    for i, p in enumerate(parents):
        if p:
            assert dv.HYPOTHESES[p[0]][:2] == dv.HYPOTHESES[i][:2]
    w0, graph = dv.gate_graph()
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

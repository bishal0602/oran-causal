from typing import Any

import numpy as np
import pytest

from oran_causal.core.data import UnitTable
from oran_causal.methods.system import maps as mp
from oran_causal.sim.system.decision.records import build_unit_data
from oran_causal.sim.system.decision.synthetic import SyntheticSpec, synthetic_records

CELLS = mp.gt_cells(
    [
        ("x", "own", "pv", "TRUE", 2.0),
        ("x", "own", "e", "TRUE", -3.0),
        ("x", "nbr", "pv", "NULL", 0.1),
        ("x", "far", "pv", "INDET", 0.4),
    ]
)


@pytest.fixture(scope="module")
def data() -> UnitTable:
    spec = SyntheticSpec(n_ep=6, units_per_ep=60, effect=120.0)
    return build_unit_data(synthetic_records(np.random.default_rng(3), spec))


def test_gt_cells_and_map() -> None:
    cells = mp.gt_cells()
    assert len(cells) == 60
    M = mp.gt_map(cells)
    assert len(M) == sum(c["status"] == "TRUE" for c in cells)
    assert M[("sleep", "nbr", "pv")] > 0


def test_edge_quality_classes() -> None:
    M = {
        ("x", "own", "pv"): 1.0,
        ("x", "own", "e"): 1.0,
        ("x", "nbr", "pv"): 1.0,
        ("x", "far", "pv"): 1.0,
        ("y", "own", "pv"): 1.0,
    }
    q = mp.edge_quality(M, CELLS)
    assert q["edges"]["true_right"] == [("x", "own", "pv")]
    assert q["edges"]["true_wrong"] == [("x", "own", "e")]
    assert q["edges"]["null"] == [("x", "nbr", "pv")]
    assert q["edges"]["indet"] == [("x", "far", "pv")]
    assert q["edges"]["unknown"] == [("y", "own", "pv")]
    assert q["precision_true_right"] == 1 / 5
    assert q["recall_true_right"] == 1 / 2
    assert np.isnan(mp.edge_quality({}, CELLS)["precision_true_right"])


def test_clean_map_and_inject() -> None:
    M = {("x", "own", "pv"): 5.0, ("x", "own", "e"): 1.0, ("x", "nbr", "pv"): 1.0}
    assert mp.clean_map(M, CELLS) == {("x", "own", "pv"): 5.0}
    assert mp.inject(
        {("a", "own", "pv"): 1.0}, {("a", "own", "pv"): 2.0, ("b", "own", "e"): 3.0}
    ) == {
        ("a", "own", "pv"): 2.0,
        ("b", "own", "e"): 3.0,
    }


def test_placebo_tau_and_tuple_scores() -> None:
    sc = mp.tuple_scores(
        {
            "x|far|pv": {"score": 3.0},
            "x|far|v": {"score": 2.0},
            "x|far|e": {"score": float("nan")},
            "x|own|pv": {"score": 9.0},
        }
    )
    assert ("x", "far", "pv") in sc
    assert mp.placebo_tau(sc, ("far",))["tau"] == 2.0
    assert mp.placebo_tau(sc, ("far", "own"), max_fp=0)["tau"] == 9.0
    few = mp.placebo_tau(sc, ("far",), max_fp=2)
    assert few["tau"] == float("-inf")
    assert few["n_scorable"] == 2


def test_naive_slope_and_map_from_declared(data: UnitTable) -> None:
    assert mp.naive_slope(data, "sleep", "nbr", "pv") > 0
    decl: dict[Any, dict[str, Any]] = {
        "sleep|nbr|pv": {"declared": True, "sign": -1},
        ("ptx", "own", "e"): {"declared": True, "sign": 0},
        ("ptx", "own", "pv"): {"declared": False, "sign": 1},
    }
    got, drop = mp.map_from_declared(decl, data)
    assert drop == 1
    assert got == {("sleep", "nbr", "pv"): -abs(mp.naive_slope(data, "sleep", "nbr", "pv"))}


def _unit(knob: str, step: float, t0: float, own: float = 0.0, c: int = 0) -> dict[str, Any]:
    ctx = {
        "knob": knob,
        "step": step,
        "own_prot_below_frac": own,
        "nbr_max_prot_below_frac": 0.0,
    }
    return {"knob": knob, "c": c, "t0": t0, "ctx": ctx}


def test_replay_counts_flips_against_the_reference() -> None:
    maps = {
        "GT": {("sleep", "nbr", "pv"): 1.0, ("sleep", "nbr", "e"): 1.0},
        "none": {},
    }
    ep = {
        "units": [
            _unit("sleep", 1.0, 0.0, c=0),
            _unit("sleep", -1.0, 200.0, c=1),
            _unit("sleep", 1.0, 300.0, c=2),
            _unit("cio", 1.0, 300.0),
        ]
    }
    out = mp.replay([ep, ep], maps)
    assert out["n_units"] == [3, 3]
    assert out["decisions"] == {"GT": [2, 2], "none": [0, 0]}
    assert out["defer_rate"]["GT"] == pytest.approx(4 / 6)
    f = out["flips"]["none"]
    assert f["per_episode"] == [2, 2]
    assert f["pre_warm"] == [1, 1]
    assert f["post_warm"] == [1, 1]
    assert f["by"] == {"sleep+:refdefer->accept": [2, 2]}
    with pytest.raises(ValueError, match="reference map"):
        mp.replay([ep], {"a": {}})
    with pytest.raises(ValueError, match="missing map"):
        mp.replay([ep], maps, pairs={"p": ("GT", "zz")})

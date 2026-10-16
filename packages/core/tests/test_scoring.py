import math

import numpy as np
import pytest

from oran_causal.core.data import EdgeResult, Result, Truth
from oran_causal.core.scoring import (
    apply_threshold,
    confusion,
    not_testable_edges,
    placebo_tau,
    score,
    score_method,
)

FAMILIES = ("carrier", "sleep", "ptx", "prot_min")
RELATIONS = ("own", "nbr", "far")
KPIS = ("pv", "v", "e", "rlf", "load")
HYPS = [(f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS]


def reference() -> dict:
    ref = {h: {"status": "NULL", "sign": 0} for h in HYPS}
    for h in (
        ("sleep", "nbr", "load"),
        ("sleep", "nbr", "pv"),
        ("sleep", "nbr", "v"),
        ("ptx", "own", "e"),
    ):
        ref[h] = {"status": "TRUE", "sign": 1}
    for h in (("carrier", "own", "e"), ("prot_min", "far", "pv")):
        ref[h] = {"status": "INDET", "sign": 0}
    return ref


def declared(on: set, sign: dict | None = None) -> dict:
    sign = sign or {}
    return {h: {"declared": h in on, "sign": sign.get(h, 1)} for h in HYPS}


@pytest.mark.claim("DEF-MET-MAP")
def test_confusion_counts_indet_and_unmapped() -> None:
    on = {
        ("sleep", "nbr", "load"),
        ("sleep", "nbr", "pv"),
        ("ptx", "own", "e"),
        ("carrier", "own", "e"),
        ("ptx", "nbr", "load"),
        ("sleep", "far", "load"),
    }
    d = declared(on, {("sleep", "nbr", "pv"): -1})
    d[("carrier", "own", "pv")] = {"declared": None, "sign": 0}
    m = score_method(d, reference(), HYPS)
    ov, ind = m["overall"], m["indirect"]
    assert (ov["tp"], ov["fp"], ov["fn"]) == (3, 2, 1)
    assert ov["n_indet"] == 2
    assert ov["n_unmapped"] == 1
    assert ov["precision"] == 3 / 5
    assert ov["recall"] == 3 / 4
    assert np.isclose(ov["f1"], 2 * 0.6 * 0.75 / 1.35)
    assert np.isclose(ov["sign_acc"], 2 / 3)
    assert (ind["tp"], ind["fp"], ind["fn"]) == (2, 1, 1)
    assert m["far_declared"] == 1
    assert m["far_fp"] == 1


def test_empty_declarations_and_nan_rules() -> None:
    m = score_method(declared(set()), reference(), HYPS)
    assert np.isnan(m["overall"]["precision"])
    assert m["overall"]["recall"] == 0
    assert m["overall"]["f1"] == 0
    assert np.isnan(m["overall"]["sign_acc"])
    none_true = {h: {"status": "NULL", "sign": 0} for h in HYPS}
    assert np.isnan(confusion(declared(set()), none_true, HYPS)["recall"])


def result(edges: list[tuple[str, str, float, bool]], notes: dict | None = None) -> Result:
    rows = tuple(EdgeResult(s, t, sc, None, 1, d) for s, t, sc, d in edges)
    return Result("m", "1", rows, 0.0, {}, notes or {})


def test_score_primary_secondary_and_placebo() -> None:
    truth = Truth(
        "E1",
        "R1",
        frozenset({("P0", "K0"), ("K0", "K1")}),
        {("P0", "K0"): 1, ("K0", "K1"): -1},
        frozenset({("P1", "K0"), ("P_placebo", "K0"), ("K1", "K0")}),
    )
    res = result(
        [
            ("P0", "K0", 3.0, True),
            ("P1", "K0", 2.0, True),
            ("P_placebo", "K0", 1.0, True),
            ("K0", "K1", 1.0, True),
            ("K1", "K0", 0.1, False),
        ],
        {"not_testable": ["P1->K0"]},
    )
    out = score(res, truth)
    assert (out["tp"], out["fp"], out["n_declared"]) == (1, 0, 1)
    assert out["sign_acc"] == 1.0
    assert out["placebo_declared"] == 1
    assert out["n_not_testable_overridden"] == 1
    assert out["secondary"]["tp"] == 1
    assert out["secondary"]["sign_acc"] == 0.0
    assert out["declared_edges"] == ["K0->K1", "P0->K0", "P_placebo->K0"]
    with pytest.raises(ValueError, match="not a candidate"):
        score(result([("X", "K0", 1.0, True)]), truth)


def test_not_testable_parsing() -> None:
    assert not_testable_edges(result([], {"not_testable_edges": {"A->B": 1}})) == {("A", "B")}
    with pytest.raises(ValueError, match="bad not-testable edge"):
        not_testable_edges(result([], {"not_testable": ["AB"]}))
    with pytest.raises(ValueError, match="must be a list"):
        not_testable_edges(result([], {"not_testable": "A->B"}))


def test_placebo_tau_conformal_rule() -> None:
    runs = [
        result([("P_placebo", "K0", float(s), False) for s in range(k, k + 10)]) for k in (0, 10)
    ]
    assert placebo_tau(runs) == 19.0
    assert placebo_tau(runs, max_declarations=2) == 17.0
    assert placebo_tau([]) == math.inf
    big = [result([("P_placebo", "K0", float(s), False) for s in range(100)])]
    assert placebo_tau(big) == 95.0


def test_apply_threshold() -> None:
    res = result([("A", "B", 2.0, False), ("A", "C", float("nan"), True)])
    out = apply_threshold(res, 1.0)
    assert [e.declared for e in out.edges] == [True, False]
    assert out.config["tau"] == 1.0

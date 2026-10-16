import math

import numpy as np
import pytest

from oran_causal.core.gate import (
    CERTIFIED,
    DECLARED,
    RHO,
    UNRESOLVED,
    Z90,
    GateSpace,
    bound_ub,
    certify,
    classify_edge,
    design_slope_se,
    tau_table,
)

SPACE = GateSpace(
    families=("sleep", "ptx"), relations=("own", "nbr", "far"), guard_kpis=("v", "rlf")
)


@pytest.mark.claim("DEF-GATE", "THM-GATE-BUDGET", "DEF-SIM-CONST")
def test_tau_budget_bounds_the_guard_ratio() -> None:
    g = {"v": 19953.65, "rlf": 35.025}
    n = {("sleep", 1): 40.0, ("ptx", -1): 0.2}
    tau = tau_table(g, n, SPACE)
    for (f, d, k), t in tau.items():
        units = max(n.get((f, d), 0.0), 1.0)
        assert units * len(SPACE.relations) * t == pytest.approx((RHO - 1) * g[k])
    assert tau[("ptx", -1, "rlf")] == pytest.approx(0.05 * 35.025 / 3)


@pytest.mark.claim("DEF-GATE", "ASM-GATE-COV")
def test_bound_ub_records() -> None:
    assert bound_ub(None, 1) == math.inf
    assert bound_ub({"kind": "slope", "beta": 2.0, "se": 1.0}, -1) == pytest.approx(-2.0 + Z90)
    assert bound_ub({"kind": "slope", "beta": 2.0, "se": None}, 1) == math.inf
    assert bound_ub({"kind": "ci", "lo": -1.0, "hi": 3.0}, -1) == 1.0
    with pytest.raises(ValueError, match="unknown bound kind"):
        bound_ub({"kind": "x"}, 1)


@pytest.mark.claim("DEF-GATE")
def test_classify_and_certify() -> None:
    m = {("sleep", "own", "v"): {"sign": 1}}
    bounds = {("sleep", "nbr", "v"): {"kind": "slope", "beta": -0.1, "se": 0.01}}
    assert classify_edge(m, bounds, ("sleep", "own", "v"), 1, 0.0)[0] == DECLARED
    status, ub = classify_edge(m, bounds, ("sleep", "nbr", "v"), 1, 1.0)
    assert status == CERTIFIED
    assert ub == pytest.approx(0.1 + Z90 * 0.01)
    assert classify_edge(m, None, ("sleep", "far", "v"), 1, 1.0)[0] == UNRESOLVED
    tau = tau_table({"v": 1e9, "rlf": 1e9}, {}, SPACE)
    every = {
        (f, r, k): {"kind": "ci", "lo": 0.0, "hi": 0.0}
        for f in SPACE.families
        for r in SPACE.relations
        for k in SPACE.guard_kpis
    }
    out = certify(m, every, tau, lambda f, d: f == "sleep", SPACE)
    assert out["uncertified"] == []
    assert out["classes"]["ptx|1"] == {"can_defer": False, "edges": [], "certified": True}
    assert len(out["classes"]["sleep|1"]["edges"]) == 6
    unc = certify(m, None, tau, lambda f, d: True, SPACE)
    assert unc["uncertified"] == [["sleep", 1], ["sleep", -1], ["ptx", 1], ["ptx", -1]]


@pytest.mark.claim("DEF-SLOPE", "THM-SLOPE")
def test_design_slope_recovers_the_effect() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    v = rng.choice([-0.5, 0.5], n)
    episode = np.repeat(np.arange(40), n // 40)
    sgn = np.ones(n)
    y = 3.0 * v + rng.normal(size=n)
    out = design_slope_se(v, y, None, episode, sgn)
    assert out["kind"] == "slope"
    assert out["n"] == n
    assert abs(out["beta"] - 3.0) < 4 * out["se"]
    assert out["z"] > 20
    assert design_slope_se(v[:1], y[:1], None, episode[:1], sgn[:1])["se"] is None
    assert design_slope_se(np.zeros(5), y[:5], None, episode[:5], sgn[:5])["beta"] == 0.0

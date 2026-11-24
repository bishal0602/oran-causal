import math

import numpy as np
import pytest

from oran_causal.core.gate import (
    CERTIFIED,
    DECLARED,
    FROZEN_WINDOW_S,
    RHO,
    UNRESOLVED,
    WINDOW_S,
    Z90,
    GateSpace,
    bound_ub,
    certify,
    classify_edge,
    design_slope_se,
    tau_table,
)
from oran_causal.core.ridge import ResidualConfig, predictable_residuals

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
    t0 = np.tile(np.arange(n // 40) * 15.0, 40)
    out = design_slope_se(v, y, None, episode, sgn, t0)
    assert out["kind"] == "slope"
    assert out["n"] == n
    assert abs(out["beta"] - 3.0) < 4 * out["se"]
    assert out["z"] > 20
    assert design_slope_se(v[:1], y[:1], None, episode[:1], sgn[:1], t0[:1])["se"] is None
    zero = design_slope_se(np.zeros(5), y[:5], None, episode[:5], sgn[:5], t0[:5])
    assert zero["beta"] == 0.0


def _units(n: int = 80, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    episode = np.repeat([0, 1], n // 2)
    t0 = np.concatenate([np.sort(rng.uniform(0, 600, n // 2)) for _ in range(2)])
    sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    y = rng.normal(size=n) + 0.5 * sgn
    pre = rng.normal(size=n)
    v = rng.choice([-0.5, 0.5], n)
    return v, y, pre, episode, sgn, t0


def test_strict_centre_with_a_zero_window_is_the_frozen_centre() -> None:
    _, y, pre, episode, sgn, t0 = _units()
    cfg = ResidualConfig()
    r0, s0 = predictable_residuals(y, pre, episode, sgn, cfg)
    r1, s1 = predictable_residuals(y, pre, episode, sgn, cfg, t0=t0, window_s=0.0)
    assert np.array_equal(r0, r1)
    assert np.array_equal(s0, s1, equal_nan=True)


@pytest.mark.claim("ASM-SLOPE-CENTRE")
def test_strict_centre_never_uses_an_open_window() -> None:
    _, y, pre, episode, sgn, t0 = _units()
    r, _ = predictable_residuals(y, pre, episode, sgn, t0=t0, window_s=WINDOW_S)
    u = 30
    open_units = [j for j in range(u) if episode[j] == episode[u] and t0[j] + WINDOW_S > t0[u]]
    assert open_units
    y2 = y.copy()
    y2[open_units] += 1000.0
    r2, _ = predictable_residuals(y2, pre, episode, sgn, t0=t0, window_s=WINDOW_S)
    assert r2[u] == r[u]
    assert (
        predictable_residuals(y2, pre, episode, sgn)[0][u]
        != predictable_residuals(y, pre, episode, sgn)[0][u]
    )


def test_strict_centre_rejects_a_negative_window() -> None:
    _, y, pre, episode, sgn, t0 = _units()
    with pytest.raises(ValueError, match="window_s"):
        predictable_residuals(y, pre, episode, sgn, t0=t0, window_s=-1.0)


def test_design_slope_defaults_to_the_strict_centre() -> None:
    v, y, pre, episode, sgn, t0 = _units()
    assert WINDOW_S == 90.0
    assert FROZEN_WINDOW_S == 0.0
    strict = design_slope_se(v, y, pre, episode, sgn, t0)
    explicit = design_slope_se(v, y, pre, episode, sgn, t0, window_s=WINDOW_S)
    frozen = design_slope_se(v, y, pre, episode, sgn, t0, window_s=FROZEN_WINDOW_S)
    r_frozen, _ = predictable_residuals(y, pre, episode, sgn)
    assert strict == explicit
    assert frozen["beta"] == float(v @ r_frozen / (v @ v))
    assert strict["beta"] != frozen["beta"]

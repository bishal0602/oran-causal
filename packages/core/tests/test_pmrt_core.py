import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from oran_causal.core.covariates import info_order
from oran_causal.core.data import Dataset, Design, EdgeResult, Result
from oran_causal.core.design import assignment, categorical_law
from oran_causal.core.pmrt import nonlinear
from oran_causal.core.pmrt.candidates import NOT_APPLICABLE
from oran_causal.core.pmrt.engine import Redraw, RedrawConfig, crt, crt_stat
from oran_causal.core.pmrt.rows import (
    PMRT_CORE_V2_VERSION,
    PMRT_CORE_VERSION,
    PmrtCore,
    PmrtCoreConfig,
)
from oran_causal.core.ridge import AdjustConfig, predictable_weights
from oran_causal.core.scoring import placebo_tau

FAST = PmrtCoreConfig(B=499)
MakeRows = Callable[..., Dataset]


def edge(res: Result, s: str, t: str) -> EdgeResult:
    return next(e for e in res.edges if e.source == s and e.target == t)


def p_of(res: Result, s: str, t: str) -> float:
    p = edge(res, s, t).p
    assert p is not None
    return p


def uniform_redraw(v: np.ndarray) -> Redraw:
    n = len(v)

    def draw(r: np.random.Generator, b: int) -> np.ndarray:
        return r.random((b, n)) - 0.5

    return Redraw(v, np.full(n, 1 / 12), draw)


def test_interface_and_not_applicable(make_rows: MakeRows) -> None:
    d = make_rows(beta=0.3)
    res = PmrtCore(FAST).run(d)
    assert res.method == "pmrt_core"
    assert res.version == PMRT_CORE_VERSION
    assert len(res.edges) == len(d.candidates)
    for s, t in (("K0", "K1"), ("K1", "K0")):
        e = edge(res, s, t)
        assert np.isnan(e.score)
        assert e.p is None
        assert not e.declared
        assert f"{s}->{t}" in res.notes[NOT_APPLICABLE]
    e = edge(res, "P0", "K0")
    assert e.declared
    assert e.sign == 1
    assert p_of(res, "P0", "K0") <= 1.0 / 500 + 1e-12


@pytest.mark.claim("DEF-ENV-ARMS")
def test_none_design_never_falls_back(make_rows: MakeRows) -> None:
    d = make_rows(beta=0.5)
    d2 = dataclasses.replace(d, designs=(Design("none"), *d.designs[1:]))
    res = PmrtCore(FAST).run(d2)
    for t in ("K0", "K1"):
        e = edge(res, "P0", t)
        assert e.p is None
        assert np.isnan(e.score)
        assert not e.declared
        assert "none" in res.notes[NOT_APPLICABLE][f"P0->{t}"]


def test_deterministic_and_chunk_invariant(make_rows: MakeRows) -> None:
    d = make_rows(beta=0.05)
    r1, r2 = PmrtCore(FAST).run(d), PmrtCore(FAST).run(d)
    assert [e.p for e in r1.edges] == [e.p for e in r2.edges]
    a = PmrtCore(PmrtCoreConfig(B=299, chunk=1000)).run(d)
    b = PmrtCore(PmrtCoreConfig(B=299, chunk=7)).run(d)
    assert [e.p for e in a.edges] == [e.p for e in b.edges]


@pytest.mark.claim("DEF-PMRT-SCORE")
def test_dither_redraws_only_random_part(make_rows: MakeRows) -> None:
    d = make_rows(regime="R2", beta=0.5)
    asg = assignment(d.designs[0], d.X_action[:, 0])
    assert not isinstance(asg, str)
    assert d.designs[0].random_part is not None
    np.testing.assert_allclose(asg.v, d.designs[0].random_part)
    v = asg.draw(np.random.default_rng(1), 2000)
    assert abs(v.mean()) < 0.01
    assert v.min() >= -0.1
    assert v.max() <= 0.1
    np.testing.assert_allclose(asg.var, (0.2**2) / 12)


@pytest.mark.claim("DEF-PMRT-SCORE")
def test_logged_categorical_propensity_centring() -> None:
    n = 400
    rng = np.random.default_rng(3)
    p1 = 1 / (1 + np.exp(-rng.normal(size=n)))
    a = (rng.random(n) < p1).astype(float)
    des = Design(
        "logged",
        {"name": "categorical", "values": [0.0, 1.0]},
        propensity=np.column_stack([1 - p1, p1]),
    )
    asg = assignment(des, a)
    assert not isinstance(asg, str)
    np.testing.assert_allclose(asg.v, a - p1)
    np.testing.assert_allclose(asg.var, p1 * (1 - p1))
    np.testing.assert_allclose(asg.draw(np.random.default_rng(0), 4000).mean(0), 0.0, atol=0.06)


@pytest.mark.parametrize("reg", ["R1", "R2"])
def test_power_planted_edge(make_rows: MakeRows, reg: str) -> None:
    res = PmrtCore(FAST).run(make_rows(n=1000, regime=reg, beta=0.3, seed=5))
    e = edge(res, "P0", "K0")
    assert e.declared
    assert e.sign == 1
    assert not edge(res, "P_placebo", "K0").declared


def test_lagged_effect_is_not_a_one_step_effect(make_rows: MakeRows) -> None:
    ps = [
        p_of(PmrtCore(FAST).run(make_rows(n=800, lag_effect=0.5, seed=s)), "P0", "K0")
        for s in range(8)
    ]
    assert np.mean(np.array(ps) <= 0.05) <= 0.25


@pytest.mark.claim("THM-PMRT-ASY")
@pytest.mark.parametrize("reg", ["R1", "R2"])
def test_null_level_small(make_rows: MakeRows, reg: str) -> None:
    ps: list[float] = []
    for s in range(12):
        res = PmrtCore(PmrtCoreConfig(B=199)).run(make_rows(n=400, regime=reg, seed=100 + s))
        ps += [e.p for e in res.edges if e.p is not None]
    arr = np.array(ps)
    assert 0.0 < np.mean(arr <= 0.05) <= 0.15
    assert 0.35 < arr.mean() < 0.65


def test_tune_placebo_threshold(make_rows: MakeRows) -> None:
    dev = [make_rows(n=300, beta=0.3, seed=s) for s in range(4)]
    m = PmrtCore(PmrtCoreConfig(B=99))
    cfg = m.tune(dev)
    runs = [m.run(d, dataclasses.asdict(m.cfg)) for d in dev]
    pl = sorted(
        e.score for r in runs for e in r.edges if e.source == "P_placebo" and np.isfinite(e.score)
    )
    assert cfg["tune_info"]["n_placebo_scores"] == len(pl) == 8
    assert cfg["tau"] == placebo_tau(runs) == pl[-1]
    assert cfg["tau_rule"] == "placebo_conformal_0.05"
    over = [
        e
        for d in dev
        for e in PmrtCore().run(d, cfg).edges
        if e.source == "P_placebo" and e.score > cfg["tau"]
    ]
    assert not over
    assert PmrtCore().run(dev[0], cfg).notes["declared_tau"] is not None


@pytest.mark.claim("THM-PMRT-ASY", "DEF-PMRT-PVAL")
def test_crt_engine_matches_normal_reference() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    redraw = uniform_redraw(rng.uniform(-0.5, 0.5, n))
    w = rng.normal(size=(n, 3))
    out = crt(redraw, w, np.random.default_rng(1), RedrawConfig(b_max=4999, seq_h=None))
    np.testing.assert_allclose(out["p2"], 2 * norm.sf(np.abs(out["z"])), atol=0.03)


def test_concurrent_actions_and_by_family(make_rows: MakeRows) -> None:
    d = make_rows(beta=0.3)
    res = PmrtCore(FAST).run(d)
    assert res.notes["adjust"]["P0"]["concurrent"] == ["P1", "P_placebo"]
    assert res.notes["by_family_m"] == 6
    off = PmrtCore(PmrtCoreConfig(B=499, concurrent_actions=False)).run(d)
    assert off.notes["adjust"]["P0"]["concurrent"] == []
    r2 = PmrtCore(FAST).run(dataclasses.replace(d, designs=(Design("none"), *d.designs[1:])))
    assert r2.notes["adjust"]["P1"]["concurrent"] == ["P_placebo"]
    assert r2.notes["by_family_m"] == 6
    assert r2.notes["n_tested"] == 4


@pytest.mark.claim("DEF-PMRT-PVAL")
def test_besag_clifford_matches_naive_and_is_chunk_invariant() -> None:
    rng = np.random.default_rng(7)
    n, b_max, h = 300, 999, 20
    v = rng.uniform(-0.5, 0.5, n)
    w = np.column_stack(
        [rng.normal(size=n), 0.3 * v + rng.normal(size=n), v + 0.1 * rng.normal(size=n)]
    )
    redraw = uniform_redraw(v)
    out = crt(redraw, w, np.random.default_rng(3), RedrawConfig(b_max=b_max, chunk=64, seq_h=h))
    out1 = crt(redraw, w, np.random.default_rng(3), RedrawConfig(b_max=b_max, chunk=1, seq_h=h))
    np.testing.assert_array_equal(out["p2"], out1["p2"])
    np.testing.assert_array_equal(out["p_plus"], out1["p_plus"])
    r = np.random.default_rng(3)
    sd = np.sqrt(redraw.var @ (w * w))
    z = (v @ w) / sd
    cnt, p = np.zeros(3, int), [0.0, 0.0, 0.0]
    stopped = [False, False, False]
    for k in range(1, b_max + 1):
        zb = (redraw.draw(r, 1) @ w)[0] / sd
        cnt += np.abs(zb) >= np.abs(z) - 1e-9 * (1 + np.abs(z))
        for j in range(3):
            if not stopped[j] and cnt[j] >= h:
                p[j], stopped[j] = h / k, True
    p = [pj if stopped[j] else (1 + cnt[j]) / (b_max + 1) for j, pj in enumerate(p)]
    np.testing.assert_allclose(out["p2"], p, rtol=0, atol=0)
    assert out["p2"][2] == 1 / (b_max + 1)
    assert out["draws"][2] == b_max
    assert out["draws"][0] < b_max
    fixed = crt(redraw, w, np.random.default_rng(3), RedrawConfig(b_max=b_max, seq_h=None))
    assert fixed["draws_total"] == b_max
    assert np.all(fixed["p2"] >= 1 / (b_max + 1))


def test_clipped_normal_and_categorical_rows_designs() -> None:
    n = 200
    des = Design("iid", {"name": "clipped_normal", "mean": 0.5, "sd": 1.1, "lo": 0.0, "hi": 1.0})
    x = np.clip(np.random.default_rng(0).normal(0.5, 1.1, 400_000), 0, 1)
    asg = assignment(des, np.full(n, 0.5))
    assert not isinstance(asg, str)
    np.testing.assert_allclose(0.5 - asg.v, x.mean(), atol=2e-3)
    np.testing.assert_allclose(asg.var, x.var(), rtol=1e-2)
    draws = asg.draw(np.random.default_rng(1), 4000)
    assert abs(draws.mean()) < 0.01
    assert abs(draws.var() - x.var()) < 0.01
    probs = np.random.default_rng(2).dirichlet(np.ones(3), n)
    rows = Design(
        "logged", {"name": "categorical_rows", "values": [0.0, 0.5, 1.0]}, propensity=probs
    )
    a = np.full(n, 1.0)
    rows_asg = assignment(rows, a)
    assert not isinstance(rows_asg, str)
    np.testing.assert_allclose(rows_asg.v, a - probs @ [0.0, 0.5, 1.0])
    assert isinstance(assignment(Design("iid", {"name": "weird"}), a), str)


def test_categorical_bisection_equals_comparison_count() -> None:
    n, m = 50, 101
    vals = np.linspace(0, 1, m)
    rngp = np.random.default_rng(4)
    for probs in (
        np.broadcast_to(rngp.dirichlet(np.ones(m)), (n, m)).copy(),
        rngp.dirichlet(np.ones(m), n),
    ):
        _, _, draw = categorical_law(vals, probs, n)
        v = draw(np.random.default_rng(9), 300)
        u = np.random.default_rng(9).random((300, n))
        cum = np.cumsum(probs, axis=1)
        cum[:, -1] = np.inf
        idx = sum((u >= cum[None, :, k]).astype(np.int64) for k in range(m - 1))
        np.testing.assert_array_equal(v, vals[idx] - probs @ vals)


def test_default_weights_are_unclipped_plain_residual() -> None:
    assert PmrtCoreConfig().huber_c is None
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(300, 3)), rng.standard_t(2, size=(300, 2))
    w0, _ = predictable_weights(x, y, AdjustConfig())
    wc, _ = predictable_weights(x, y, AdjustConfig(huber_c=2.5))
    assert np.abs(w0).max() > np.abs(wc).max()
    assert np.any(np.abs(wc) < np.abs(w0))


@pytest.mark.claim("ASM-A3-PRED")
def test_row_weights_ignore_the_future() -> None:
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(400, 4)), rng.normal(size=(400, 2))
    w, info = predictable_weights(x, y, AdjustConfig())
    for cut in (info["burn"] + 5, 200, 399):
        x2, y2 = x.copy(), y.copy()
        x2[cut + 1 :] += 1e6
        y2[cut + 1 :] -= 1e6
        w2, _ = predictable_weights(x2, y2, AdjustConfig())
        assert np.array_equal(w2[: cut + 1], w[: cut + 1])


def run_nl(ds: Dataset, **kw: Any) -> Result:
    cfg = dataclasses.replace(nonlinear.PmrtNlConfig(B=299, gbm_trees=40), **kw)
    return nonlinear.PmrtNl(cfg).run(ds)


@pytest.mark.parametrize(
    ("st", "adj"), [("poly", "ridge"), ("rff", "ridge"), ("gbm", "gbm"), ("poly", "gbm")]
)
def test_nonlinear_power_on_centred_bump(make_bump: MakeRows, st: str, adj: str) -> None:
    res = run_nl(make_bump(), statistic=st, adjust=adj)
    assert res.method == "pmrt_nl"
    assert res.version == nonlinear.PMRT_NL_VERSION
    assert len(res.edges) == 8
    assert p_of(res, "P0", "K0") <= 0.01
    assert all(e.p is None and np.isnan(e.score) for e in res.edges if e.source in ("K0", "K1"))
    assert res.notes["statistic"] == st
    assert res.notes["adjust"] == adj


def test_nonlinear_deterministic_and_chunk_invariant(make_bump: MakeRows) -> None:
    ds = make_bump(seed=3)
    a, b, c = (
        run_nl(ds, statistic="rff"),
        run_nl(ds, statistic="rff"),
        run_nl(ds, statistic="rff", chunk=7),
    )
    for x, y, z in zip(a.edges, b.edges, c.edges, strict=True):
        assert x.p == y.p == z.p
        assert x.score == y.score == z.score or np.isnan(x.score)


@pytest.mark.claim("THM-EXACT")
def test_nonlinear_null_level_exact_case(make_rows: MakeRows) -> None:
    ps: list[float] = []
    for s in range(30):
        ds = make_rows(n=300, seed=100 + s, regime="R1")
        for st in ("poly", "rff"):
            ps += [e.p for e in run_nl(ds, statistic=st, covariates="r3").edges if e.p is not None]
    arr = np.array(ps)
    assert 0.01 <= np.mean(arr <= 0.05) <= 0.10
    assert 0.4 <= arr.mean() <= 0.6


def test_nonlinear_dither_redraws_only_random_part(make_bump: MakeRows) -> None:
    ds = make_bump(regime="R2")
    law = nonlinear.design_law(ds.designs[0], ds.X_action[:, 0], info_order(ds), 48)
    assert not isinstance(law, str)
    assert law.draw is not None
    assert np.allclose(law.off + law.v, ds.X_action[info_order(ds), 0])
    assert np.all(np.abs(law.draw(np.random.default_rng(0), 50)) <= 0.1 + 1e-12)
    assert np.isfinite(edge(run_nl(ds, statistic="poly"), "P0", "K0").score)


def test_uniform_grid_moments_exact_for_piecewise_linear() -> None:
    rng = np.random.default_rng(1)
    law = nonlinear.Law(
        np.zeros(3),
        np.full(3, 1 / 3),
        None,
        np.zeros(3),
        np.zeros((1, 1)),
        np.ones((1, 1)),
        "uniform",
        glo=np.full(3, -1.0),
        ghi=np.full(3, 1.0),
    )
    h = rng.normal(size=(3, 9))
    m, v = nonlinear.grid_moments(law, h)
    x = np.linspace(-1, 1, 400_001)
    for t in range(3):
        f = np.interp(x, np.linspace(-1, 1, 9), h[t])
        assert abs(m[t] - f.mean()) < 1e-4
        assert abs(v[t] - f.var()) < 1e-4


def test_crt_stat_counts_and_observed_value() -> None:
    v = np.array([0.5, -0.2, 0.1])

    def stat(x: np.ndarray) -> np.ndarray:
        return (x**2).sum(1, keepdims=True)

    def draw(rng: np.random.Generator, b: int) -> np.ndarray:
        return rng.uniform(-1, 1, (b, 3))

    r = crt_stat(
        stat, v, draw, np.random.default_rng(0), RedrawConfig(b_max=999, chunk=64, seq_h=None)
    )
    vb = draw(np.random.default_rng(0), 999)
    assert r["t"][0] == pytest.approx(0.30)
    assert r["p"][0] == pytest.approx((1 + np.sum(stat(vb)[:, 0] >= 0.30 - 1e-9 * 1.3)) / 1000)


@pytest.mark.claim("DEF-ENV-ARMS")
def test_statistic_option_delegates_to_nonlinear(make_bump: MakeRows) -> None:
    ds = make_bump(seed=6)
    core = PmrtCore(PmrtCoreConfig(statistic="gbm", B=299)).run(ds)
    nl = run_nl(ds, statistic="gbm", gbm_trees=nonlinear.PmrtNlConfig().gbm_trees)
    assert core.method == "pmrt_core"
    assert core.version == PMRT_CORE_V2_VERSION
    for x, y in zip(core.edges, nl.edges, strict=True):
        assert (x.source, x.target, x.p, x.sign, x.declared) == (
            y.source,
            y.target,
            y.p,
            y.sign,
            y.declared,
        )
    assert PmrtCore().run(ds).version == PMRT_CORE_VERSION
    for bad in (
        PmrtCoreConfig(adjust="gbm"),
        PmrtCoreConfig(statistic="gbm", huber_c=2.5),
        PmrtCoreConfig(statistic="x"),
    ):
        with pytest.raises(ValueError, match=r"adjust|huber_c|statistic"):
            PmrtCore(bad).run(ds)


@pytest.mark.claim("THM-COUNTEREX")
def test_reproposal_makes_the_redraw_p_value_anti_conservative() -> None:
    a = 2.0
    cfg = RedrawConfig(b_max=4999, seq_h=None)

    def p_plus(v: list[float], w: list[float], seed: int) -> float:
        def draw(rng: np.random.Generator, b: int) -> np.ndarray:
            return rng.choice([-0.5, 0.5], size=(b, len(v)))

        redraw = Redraw(np.array(v), np.full(len(v), 0.25), draw)
        out = crt(redraw, np.array(w)[:, None], np.random.default_rng(seed), cfg)
        return float(out["p_plus"][0])

    accepted = p_plus([0.5], [1.0], 1)
    rejected_then_accepted = p_plus([-0.5, 0.5], [1.0, a], 2)
    rejected_twice = p_plus([-0.5, -0.5], [1.0, a], 3)
    assert accepted == pytest.approx(0.5, abs=0.03)
    assert rejected_then_accepted == pytest.approx(0.5, abs=0.03)
    assert rejected_twice == 1.0
    levels = (accepted, rejected_then_accepted, rejected_twice)
    prob = (0.5, 0.25, 0.25)
    pr_le_half = sum(q for q, p in zip(prob, levels, strict=True) if p <= 0.5 + 0.03)
    assert pr_le_half == pytest.approx(0.75)
    assert pr_le_half > 0.5


@pytest.mark.claim("ASM-S4-KNOWN")
def test_null_validity_needs_the_logged_design_to_be_told_correctly() -> None:
    n, runs = 100, 400
    cfg = RedrawConfig(b_max=199, seq_h=None)

    def rejection_rate(told_width: float) -> float:
        rng = np.random.default_rng(5)
        hits = 0
        for _ in range(runs):
            v = rng.uniform(-0.5, 0.5, n)
            w = rng.normal(size=(n, 1))

            def draw(r: np.random.Generator, b: int) -> np.ndarray:
                return told_width * (r.random((b, n)) - 0.5)

            redraw = Redraw(v, np.full(n, told_width**2 / 12), draw)
            hits += crt(redraw, w, rng, cfg)["p2"][0] <= 0.05
        return hits / runs

    assert 0.02 <= rejection_rate(1.0) <= 0.085
    assert rejection_rate(0.5) > 0.2
    assert rejection_rate(2.0) < 0.02

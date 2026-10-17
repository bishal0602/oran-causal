import dataclasses

import numpy as np
import pytest

from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.methods.controlled._rcot import (
    FROZEN_BLOCK_PERM_REPS_V2,
    RCoTDiscoveryConfig,
    _gamma_sf,
    frozen_config_v2,
    rcot_pvalue_analytic,
    standardize,
)
from oran_causal.methods.controlled.pcorr import (
    EPS_COND,
    FROZEN_FLOOR,
    largest_gap,
    partial_correlation_scores,
)
from oran_causal.methods.controlled.rcot2 import bh_per_target, block_perm_sequential


def _ks_uniform(p: np.ndarray) -> float:
    p = np.sort(np.asarray(p))
    return float(np.max(np.abs(np.arange(1, p.size + 1) / p.size - p)))


def _zstd(a):
    return (a - a.mean(0)) / a.std(0)


def test_largest_gap_cuts_between_zero_and_signal():
    assert largest_gap(np.array([0.0, 0.0, 0.0, 0.8, 0.9]), floor=FROZEN_FLOOR) == pytest.approx(
        0.4
    )
    assert largest_gap(np.array([0.0, 0.0001, 0.5, 0.52]), floor=1e-3) == pytest.approx(0.51)
    assert largest_gap(np.array([0.0, 0.0001, 0.0004])) == pytest.approx(0.00025)
    assert largest_gap(np.array([0.3])) == 0.3
    assert largest_gap(np.array([])) == 0.0
    assert FROZEN_FLOOR == 0.0
    assert EPS_COND == 1e-8


def test_partial_correlation_true_edges_near_one_and_non_edges_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((800, 5))
    y = np.column_stack([x[:, 0] + 0.5 * x[:, 3], -x[:, 1]])
    part = partial_correlation_scores(_zstd(x), _zstd(y))
    assert part.shape == (2, 5)
    assert part[0, 0] == pytest.approx(1.0)
    assert part[1, 1] == pytest.approx(-1.0)
    assert np.all(part[0, [1, 2, 4]] == 0.0)


def test_near_collinear_input_is_a_stop_not_an_absent_edge():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((600, 8))
    x[:, 7] = x[:, 6] + 3e-5 * rng.standard_normal(600)
    y = (x[:, 7] - x[:, 6])[:, None] + 0.1 * rng.standard_normal((600, 1))
    with pytest.raises(ValueError, match="recorded STOP"):
        partial_correlation_scores(_zstd(x), _zstd(y))


@pytest.fixture(scope="module")
def rcot_selfcheck():
    cfg = RCoTDiscoveryConfig()
    reps, n = 120, 200
    rng = np.random.default_rng(123)
    out: dict[str, list[float]] = {"A": [], "B": [], "C": []}
    for r in range(reps):
        z = rng.normal(size=(n, 3))
        out["A"].append(rcot_pvalue_analytic(rng.normal(size=n), rng.normal(size=n), z, cfg, r)[1])
        z2 = rng.normal(size=(n, 3))
        bx, by = rng.normal(size=3), rng.normal(size=3)
        xb, yb = z2 @ bx + rng.normal(size=n), z2 @ by + rng.normal(size=n)
        out["B"].append(rcot_pvalue_analytic(xb, yb, z2, cfg, 1000 + r)[1])
        z3 = rng.normal(size=(n, 3))
        xc = rng.normal(size=n)
        out["C"].append(
            rcot_pvalue_analytic(xc, 0.7 * xc + rng.normal(size=n), z3, cfg, 2000 + r)[1]
        )
    return {k: np.asarray(v) for k, v in out.items()}


@pytest.mark.parametrize("case", ["A", "B"])
def test_rcot_null_is_calibrated(rcot_selfcheck, case):
    p = rcot_selfcheck[case]
    assert float(np.mean(p < 0.05)) <= 0.15
    assert 0.40 <= float(p.mean()) <= 0.60
    assert _ks_uniform(p) <= 0.15


def test_rcot_power_direct_edge(rcot_selfcheck):
    assert float(np.mean(rcot_selfcheck["C"] < 0.05)) >= 0.90


def test_both_null_variants_give_valid_pvalues():
    cfg_hbe = RCoTDiscoveryConfig(null_method="analytic_hbe")
    cfg_perm = RCoTDiscoveryConfig(null_method="block_perm", block_perm_reps=49, block_size=15)
    rng = np.random.default_rng(5)
    for r in range(6):
        z = rng.normal(size=(120, 3))
        x, y = rng.normal(size=120), rng.normal(size=120)
        _, p_h, g_h = rcot_pvalue_analytic(x, y, z, cfg_hbe, r)
        _, p_p, g_p, _ = block_perm_sequential(x, y, z, cfg_perm, r, h=None)
        assert 0.0 <= p_h <= 1.0
        assert not g_h
        assert 0.0 <= p_p <= 1.0
        assert not g_p


def test_v2_only_changes_block_perm_reps():
    v1, v2 = RCoTDiscoveryConfig(), frozen_config_v2()
    assert v1.block_perm_reps == 99
    assert v2.block_perm_reps == 299 == FROZEN_BLOCK_PERM_REPS_V2
    a, b = dataclasses.asdict(v1), dataclasses.asdict(v2)
    assert {k for k in a if a[k] != b[k]} == {"block_perm_reps"}
    assert (v2.dz, v2.dxy, v2.ridge, v2.q) == (25, 5, 1e-6, 0.05)
    assert (v2.null_method, v2.block_size, v2.residual_epsilon, v2.rng_seed) == (
        "block_perm",
        25,
        1e-12,
        0,
    )


def test_config_validation_and_standardize():
    with pytest.raises(ValueError, match="null_method"):
        RCoTDiscoveryConfig(null_method="bogus").validate()
    with pytest.raises(ValueError, match="dz and dxy"):
        RCoTDiscoveryConfig(dz=0).validate()
    with pytest.raises(ValueError, match="zero-variance"):
        standardize(np.ones((10, 2)))


def test_gamma_sf_matches_scipy():
    from scipy import stats

    for x, k, theta in ((0.5, 2.0, 1.0), (3.0, 0.7, 2.0), (40.0, 5.0, 3.0), (1e-3, 1.5, 1.0)):
        assert _gamma_sf(x, k, theta) == pytest.approx(
            stats.gamma.sf(x, k, scale=theta), rel=1e-9, abs=1e-14
        )
    assert _gamma_sf(0.0, 2.0, 1.0) == 1.0


def _bh_threshold_rule(p, q):
    m = len(p)
    sp = np.sort(p, kind="stable")
    passing = sp <= q * np.arange(1, m + 1) / m
    if not passing.any():
        return np.zeros(m, dtype=bool)
    return p <= sp[int(np.nonzero(passing)[0].max())]


@pytest.mark.claim("THM-BY-FIN")
def test_bh_per_target_equals_frozen_threshold_rule():
    q = 0.05
    assert weighted_step_up(np.array([0.001, 0.2, 0.5, 0.9]), None, q, reshape=False).tolist() == [
        1,
        0,
        0,
        0,
    ]
    assert weighted_step_up(np.array([0.01, 0.02, 0.03, 0.9]), None, q, reshape=False).tolist() == [
        1,
        1,
        1,
        0,
    ]
    assert not weighted_step_up(np.ones(4), None, q, reshape=False).any()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        m = int(rng.integers(1, 20))
        p = np.round(rng.uniform(0, 0.2, m) ** 2, 4)
        pairs = tuple((i, 0) for i in range(m))
        assert np.array_equal(bh_per_target(p, pairs, q), _bh_threshold_rule(p, q))

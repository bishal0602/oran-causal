from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from oran_causal.core.design import v_design, v_var
from oran_causal.core.pmrt.units import demean
from oran_causal.core.ridge import ResidualConfig, predictable_residuals, ridge_fit, ridge_gcv

MakeUnits = Callable[..., dict[str, Any]]


def crt_z(d: dict[str, np.ndarray], r: np.ndarray, design: tuple[np.ndarray, np.ndarray]) -> float:
    policy, levels = design
    probs = np.broadcast_to(policy, (len(r), 3))
    v = v_design(d["mode"], probs, d["sgn"], levels)
    var = v_var(probs, levels)
    return float(v @ r / np.sqrt(var[0] * (r @ r)))


@pytest.mark.claim("ASM-A3-PRED")
def test_predictable_residuals_ignore_later_units(make_units: MakeUnits) -> None:
    d = make_units(np.random.default_rng(5), 6)
    r, s = predictable_residuals(d["y"], d["pre"], d["episode"], d["sgn"])
    n = len(r)
    for k in (n // 4, n // 2, n - 2):
        rk, sk = predictable_residuals(d["y"][:k], d["pre"][:k], d["episode"][:k], d["sgn"][:k])
        assert np.array_equal(rk, r[:k])
        assert np.array_equal(sk, s[:k], equal_nan=True)
    y2 = d["y"].copy()
    y2[n // 2 + 1 :] += 1e6
    r2, _ = predictable_residuals(y2, d["pre"], d["episode"], d["sgn"])
    assert np.array_equal(r2[: n // 2 + 1], r[: n // 2 + 1])


@pytest.mark.claim("ASM-A3-PRED", "THM-PMRT-ASY")
def test_predictable_residual_is_unbiased_where_fe_demeaning_is_not(
    make_units: MakeUnits, unit_design: tuple[np.ndarray, np.ndarray]
) -> None:
    rng = np.random.default_rng(7)
    z_fe, z_pred = [], []
    for _ in range(120):
        d = make_units(rng, 80)
        r_fe = demean(d["y"], d["episode"] * 2 + (d["sgn"] > 0))
        r_pr, _ = predictable_residuals(
            d["y"], d["pre"], d["episode"], d["sgn"], ResidualConfig(burn=0)
        )
        z_fe.append(crt_z(d, r_fe, unit_design))
        z_pred.append(crt_z(d, r_pr, unit_design))
    se = 1.0 / np.sqrt(len(z_fe))
    assert np.mean(z_fe) < -4 * se
    assert abs(np.mean(z_pred)) < 3 * se


def test_ridge_fit_handles_constant_covariates() -> None:
    x = np.ones((20, 2))
    y = np.arange(40, dtype=float).reshape(20, 2)
    predict, res = ridge_fit(x, y)
    assert np.allclose(predict(np.ones((3, 2))), y.mean(0))
    assert np.allclose(res, y - y.mean(0))


def test_ridge_gcv_recovers_a_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    x -= x.mean(0)
    b = np.array([[1.0, 0.0], [0.5, -2.0], [0.0, 1.0]])
    coef, lam = ridge_gcv(x, x @ b + 0.01 * rng.normal(size=(500, 2)))
    assert lam == pytest.approx(1e-3)
    assert np.allclose(coef, b, atol=0.01)

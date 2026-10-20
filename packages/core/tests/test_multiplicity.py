from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oran_causal.core.declare import (
    HypStats,
    declare,
)
from oran_causal.core.multiplicity import c_harm, e_bh, weighted_step_up

DATA = Path(__file__).parent / "data"
GRID = 4000.0
NAN_CODE = 65535


def p_of(codes: np.ndarray) -> np.ndarray:
    return np.where(codes == NAN_CODE, np.nan, codes.astype(float) / GRID)


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"fdr_{name}.npz") as z:
        return dict(z)


def reference_step_up(p: list[float], w: list[float], q: float, reshape: bool) -> list[bool]:
    m = len(p)
    cm = sum(Fraction(1, k) for k in range(1, m + 1)) if reshape else Fraction(1)
    r = [
        Fraction(pj) / Fraction(wj) if np.isfinite(pj) and wj > 0 else None
        for pj, wj in zip(p, w, strict=True)
    ]

    def threshold(k: int) -> Fraction:
        return Fraction(q) * k / (m * cm)

    big_r = max(
        (k for k in range(1, m + 1) if sum(x is not None and x <= threshold(k) for x in r) >= k),
        default=0,
    )
    return [bool(big_r) and x is not None and x <= threshold(big_r) for x in r]


def near_threshold(p: list[float], w: list[float], q: float, reshape: bool) -> bool:
    m = len(p)
    cm = c_harm(m) if reshape else 1.0
    thresholds = q * np.arange(1, m + 1) / (m * cm)
    for pj, wj in zip(p, w, strict=True):
        if np.isfinite(pj) and wj > 0 and np.any(np.abs(pj / wj - thresholds) <= 1e-9 * thresholds):
            return True
    return False


grid_p = st.integers(0, 400).map(lambda k: k / 4000.0)
p_value = st.one_of(grid_p, st.floats(0.0, 1.0), st.just(float("nan")))


@st.composite
def step_up_case(draw: st.DrawFn) -> tuple[list[float], list[float], bool]:
    m = draw(st.integers(1, 40))
    p = draw(st.lists(p_value, min_size=m, max_size=m))
    k = draw(st.lists(st.integers(0, 5), min_size=m, max_size=m))
    if sum(k) == 0:
        k[0] = 1
    w = (m * np.array(k, float) / sum(k)).tolist()
    return p, w, draw(st.booleans())


@pytest.mark.claim("THM-BY-FIN", "THM-FDR-ASY")
@settings(max_examples=200, deadline=None)
@given(step_up_case())
def test_step_up_matches_exact_reference(case: tuple[list[float], list[float], bool]) -> None:
    p, w, reshape = case
    got = weighted_step_up(np.array(p), np.array(w), 0.05, reshape=reshape).tolist()
    if got != reference_step_up(p, w, 0.05, reshape):
        assert near_threshold(p, w, 0.05, reshape)


def test_step_up_never_rejects_untested() -> None:
    p = np.array([np.nan, 0.0, 0.0])
    assert weighted_step_up(p).tolist() == [False, True, True]


def test_step_up_rejects_bad_weights() -> None:
    with pytest.raises(ValueError, match="sum m"):
        weighted_step_up(np.array([0.01, 0.02]), np.array([1.0, 0.5]))


def test_by_is_subset_of_bh() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = rng.random(30) ** 3
        assert not np.any(weighted_step_up(p) & ~weighted_step_up(p, reshape=False))


def test_e_bh_counts_nan_as_zero() -> None:
    assert e_bh(np.array([np.nan, 1e6, 1e6]), 0.05).tolist() == [False, True, True]


def test_declare_needs_prior_for_weighted() -> None:
    with pytest.raises(ValueError, match="Prior"):
        declare({}, "wby", [("a",)])


def test_declare_needs_parents_for_dagger() -> None:
    with pytest.raises(ValueError, match="parents"):
        declare({("a",): HypStats(p2=0.01)}, "dagger", [("a",)])

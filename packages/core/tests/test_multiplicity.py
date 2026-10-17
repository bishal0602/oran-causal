import json
import sys
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oran_causal.core.declare import (
    PROCEDURES,
    DeclareConfig,
    HypStats,
    Prior,
    Structure,
    WeightConfig,
    declare,
    prior_weights,
)
from oran_causal.core.multiplicity import c_harm, dagger, e_bh, weighted_step_up

DATA = Path(__file__).parent / "data"
GRID = 4000.0
NAN_CODE = 65535
N_HYPS = 60
DECLARE_STRIDE = 50
PRIOR_WEIGHTS = {"win32": "declare_p", "linux": "prior_weights_linux"}


def p_of(codes: np.ndarray) -> np.ndarray:
    return np.where(codes == NAN_CODE, np.nan, codes.astype(float) / GRID)


def e_of(codes: np.ndarray) -> np.ndarray:
    return np.where(codes < 0, np.nan, codes.astype(float) / 100)


def w_of(k: np.ndarray) -> np.ndarray:
    k = k.astype(float)
    return len(k) * k / k.sum()


def bits(packed: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(packed)[:n].astype(bool)


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"fdr_{name}.npz") as z:
        return dict(z)


@pytest.mark.claim("DEF-FDR-WBY", "THM-BY-FIN")
def test_step_up_matches_fdr_layer() -> None:
    z = fixture("step_up")
    ms = z["su_m"].astype(int)
    n = int(ms.sum())
    golden = {k: bits(z[f"su_{k}"], n) for k in ("by", "bh", "wby")}
    start = 0
    for m in ms:
        sl = slice(start, start + m)
        p, w = p_of(z["su_p"][sl]), w_of(z["su_k"][sl])
        assert np.array_equal(weighted_step_up(p, None, 0.05, reshape=True), golden["by"][sl])
        assert np.array_equal(weighted_step_up(p, None, 0.05, reshape=False), golden["bh"][sl])
        assert np.array_equal(weighted_step_up(p, w, 0.05, reshape=True), golden["wby"][sl])
        start += m
    assert len(ms) == 10_000


def test_dagger_matches_fdr_layer() -> None:
    z = fixture("dagger")
    dags = json.loads(z["dg_parents"].tobytes())
    n = sum(len(d) for d in dags)
    golden = bits(z["dg_dec"], n)
    start = 0
    for parents in dags:
        sl = slice(start, start + len(parents))
        assert np.array_equal(dagger(p_of(z["dg_p"][sl]), parents, 0.05), golden[sl])
        start += len(parents)


def declare_inputs(z: dict[str, np.ndarray], e: dict[str, np.ndarray], i: int):
    hyps = [tuple(h.split("|")) for h in z["dc_hyps"]]
    p2, pp, pm = (p_of(x) for x in z["dc_p"][i])
    sign = z["dc_sign"][i].astype(int)
    ec = e["dc_e"][i] if i < len(e["dc_e"]) else np.full((3, N_HYPS), -1)
    e2, ep, em = (e_of(x) for x in ec)
    stats = {
        h: HypStats(p2[j], int(sign[j]), pp[j], pm[j], e2[j], ep[j], em[j], int(sign[j]))
        for j, h in enumerate(hyps)
    }
    prior = Prior(z={h: float(z["dc_z"][i][j]) / 100 for j, h in enumerate(hyps)}, n_prior=480)
    return hyps, stats, prior


def declare_structure(z: dict[str, np.ndarray]) -> Structure:
    return Structure(
        parents=json.loads(z["dc_parents"].tobytes()),
        holm_weights=z["gh_w0"],
        holm_graph=z["gh_G"],
        groups=list(z["dc_groups"]),
    )


def check_declare(rows: range, procedures: tuple[str, ...]) -> None:
    z, e = fixture("declare_p"), fixture("declare_e")
    structure = declare_structure(z)
    n_sets = len(z["dc_p"])
    for proc in procedures:
        n_rows = len(e["dc_e"]) if proc in ("ebh", "ebh1s", "webh1s") else n_sets
        packed = z[f"dc_{proc}"] if f"dc_{proc}" in z else e[f"dc_{proc}"]
        golden = bits(packed, n_rows * N_HYPS)
        golden = golden.reshape(n_rows, N_HYPS)
        for i in rows:
            if i >= n_rows:
                continue
            hyps, stats, prior = declare_inputs(z, e, i)
            config = DeclareConfig(prior=prior, n_target=600, structure=structure)
            out = declare(stats, proc, hyps, config)
            assert [out[h]["declared"] for h in hyps] == golden[i].tolist(), (proc, i)


@pytest.mark.claim("DEF-FDR-WBY")
def test_declare_matches_fdr_layer_subset() -> None:
    check_declare(range(0, 2000, DECLARE_STRIDE), PROCEDURES)


@pytest.mark.slow
def test_declare_matches_fdr_layer_full() -> None:
    check_declare(range(2000), PROCEDURES)


@pytest.mark.claim("DEF-FDR-WBY", "ASM-FDR-FIXED")
def test_prior_weights_match_fdr_layer() -> None:
    if sys.platform not in PRIOR_WEIGHTS:
        pytest.skip(f"no reference weights for platform {sys.platform}")
    z, e = fixture("declare_p"), fixture("declare_e")
    weights = fixture(PRIOR_WEIGHTS[sys.platform])["dc_weights"]
    for i in range(0, len(weights), 3):
        golden = weights[i]
        hyps, _, prior = declare_inputs(z, e, i)
        w = prior_weights(prior, hyps, WeightConfig(n_target=600))
        assert np.array_equal(np.array([w[h] for h in hyps]), golden)
        assert np.isclose(golden.sum(), N_HYPS)


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

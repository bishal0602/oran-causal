import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from oran_causal.core.declare import (
    PROCEDURES,
    DeclareConfig,
    HypStats,
    Prior,
    Structure,
    WeightConfig,
    declare,
    prior_directions,
    prior_weights,
)
from oran_causal.core.multiplicity import (
    W_CAP,
    cap_weights,
    dag_effective,
    dagger,
    e_bh,
    graphical_holm,
    p_filter,
    rw_weights,
    weighted_step_up,
)

DATA = Path(__file__).parent / "data"
Hyp = tuple[str, ...]


@cache
def space() -> tuple[list[Hyp], Structure]:
    with np.load(DATA / "fdr_declare_p.npz") as z:
        hyps = [tuple(h.split("|")) for h in z["dc_hyps"]]
        structure = Structure(
            parents=json.loads(z["dc_parents"].tobytes()),
            holm_weights=z["gh_w0"],
            holm_graph=z["gh_G"],
            groups=list(z["dc_groups"]),
        )
    return hyps, structure


def by_reference(p: np.ndarray, q: float) -> np.ndarray:
    m = len(p)
    cm = sum(1.0 / k for k in range(1, m + 1))
    o = np.argsort(p, kind="stable")
    ok = p[o] <= q * np.arange(1, m + 1) / (m * cm)
    out = np.zeros(m, bool)
    if ok.any():
        out[o[: np.nonzero(ok)[0].max() + 1]] = True
    return out


def bh_reference(p: np.ndarray, q: float) -> np.ndarray:
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    out = np.zeros(m, bool)
    if ok.any():
        out[o[: np.nonzero(ok)[0].max() + 1]] = True
    return out


def holm_reference(p: np.ndarray, a: float) -> np.ndarray:
    o = np.argsort(p)
    m = len(p)
    out = np.zeros(m, bool)
    for i, j in enumerate(o):
        if p[j] > a / (m - i):
            break
        out[j] = True
    return out


def e_bh_reference(e: np.ndarray, q: float) -> np.ndarray:
    m = len(e)
    order = np.argsort(-e, kind="stable")
    ok = e[order] >= m / (q * np.arange(1, m + 1))
    out = np.zeros(m, bool)
    if ok.any():
        out[order[: int(np.nonzero(ok)[0].max()) + 1]] = True
    return out


@pytest.fixture
def pvals() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [
        np.concatenate([rng.uniform(size=40), rng.uniform(0, 0.004, size=20)]) for _ in range(30)
    ]


@pytest.mark.claim("DEF-FDR-WBY")
def test_weighted_step_up_reduces_to_by_and_bh(pvals: list[np.ndarray]) -> None:
    for p in pvals:
        assert np.array_equal(weighted_step_up(p, None, 0.05, True), by_reference(p, 0.05))
        assert np.array_equal(weighted_step_up(p, None, 0.05, False), bh_reference(p, 0.05))


def test_weights_must_sum_to_m() -> None:
    with pytest.raises(ValueError, match="sum m"):
        weighted_step_up(np.full(4, 0.01), np.array([1, 1, 1, 2.0]))
    with pytest.raises(ValueError, match="sum m"):
        e_bh(np.ones(4), 0.05, np.array([1, 1, 1, 2.0]))


def test_dagger_flat_is_by_and_bh(pvals: list[np.ndarray]) -> None:
    par: list[list[int]] = [[] for _ in range(60)]
    for p in pvals:
        assert np.array_equal(dagger(p, par, 0.05, True), by_reference(p, 0.05))
        assert np.array_equal(dagger(p, par, 0.05, False), bh_reference(p, 0.05))


def test_dagger_effective_counts_and_gating() -> None:
    hyps, structure = space()
    par = structure.parents
    assert par is not None
    ell, mm = dag_effective(par)
    loads = [i for i, h in enumerate(hyps) if h[2] == "load"]
    assert len(loads) == 12
    assert set(ell[loads]) == {4.0}
    assert set(mm[loads]) == {5.0}
    assert ell.sum() - ell[loads].sum() == 48
    p = np.full(60, 1e-6)
    p[loads] = 0.9
    assert not dagger(p, par, 0.05).any()
    p[loads[0]] = 1e-6
    rej = dagger(p, par, 0.05)
    kids = [i for i, h in enumerate(hyps) if h[:2] == hyps[loads[0]][:2]]
    assert set(np.nonzero(rej)[0]) == set(kids)


def test_graphical_holm_complete_graph_is_holm(pvals: list[np.ndarray]) -> None:
    n = 60
    w0 = np.full(n, 1.0 / n)
    g = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    for p in pvals[:10]:
        assert np.array_equal(graphical_holm(p, w0, g, 0.05), holm_reference(p, 0.05))


def test_gate_graph_is_proper() -> None:
    _, structure = space()
    w0, g = structure.holm_weights, structure.holm_graph
    assert w0 is not None
    assert g is not None
    assert abs(w0.sum() - 1) < 1e-12
    assert np.all(np.diag(g) == 0)
    assert np.all(g.sum(1) <= 1 + 1e-12)


def test_p_filter_singletons_is_by(pvals: list[np.ndarray]) -> None:
    for p in pvals:
        assert np.array_equal(p_filter(p, list(range(60)), None, 0.05), by_reference(p, 0.05))


def test_e_bh_matches_reference() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        e = np.exp(rng.normal(0, 3, 60))
        assert np.array_equal(e_bh(e, 0.05), e_bh_reference(e, 0.05))


@pytest.mark.claim("DEF-FDR-WBY", "ASM-FDR-FIXED", "CIT-RW")
def test_prior_weights_floor_mean_and_directions() -> None:
    hyps, _ = space()
    rng = np.random.default_rng(0)
    z = {h: float(rng.normal(0, 3)) for h in hyps}
    pr = Prior(z=z, n_prior=480)
    for n in (60, 120, 300):
        w = np.array(list(prior_weights(pr, hyps, WeightConfig(n_target=n, floor=0.2)).values()))
        assert abs(w.mean() - 1) < 1e-9
        assert w.min() >= 0.2 - 1e-12
        assert w.max() <= W_CAP + 1e-9
    d = prior_directions(pr, hyps, 3.0)
    assert all(d[h] == (np.sign(z[h]) if abs(z[h]) >= 3 else 0) for h in hyps)
    assert np.allclose(rw_weights(np.zeros(5), 0.05), 1.0)
    c = cap_weights(np.array([10.0, 1, 1, 0.5, 0.5, 0, 0, 0, 0, 0]), 3.0)
    assert abs(c.sum() - 13) < 1e-9
    assert c.max() <= 3 + 1e-9


def calibrated(x: float) -> float:
    return 0.5 / np.sqrt(max(x, 1e-12))


def make_stats(
    hyps: list[Hyp], p2: np.ndarray, pp: np.ndarray, pm: np.ndarray, sign: np.ndarray
) -> dict[Hyp, HypStats]:
    return {
        h: HypStats(
            p2=p2[i],
            p_plus=pp[i],
            p_minus=pm[i],
            sign=int(sign[i]),
            e2=calibrated(p2[i]),
            e_plus=calibrated(pp[i]),
            e_minus=calibrated(pm[i]),
            e_sign=int(sign[i]),
        )
        for i, h in enumerate(hyps)
    }


@pytest.mark.claim("DEF-FDR-WBY", "ASM-ORIENT")
def test_declare_one_sided_sign_and_untested() -> None:
    hyps, structure = space()
    h0 = hyps[0]
    z = dict.fromkeys(hyps, 0.0)
    z[h0] = -5.0
    config = DeclareConfig(prior=Prior(z=z, n_prior=480), n_target=60, structure=structure)
    p2, pp, pm = np.full(60, 0.5), np.full(60, 0.5), np.full(60, 0.5)
    pm[0], p2[0] = 1e-5, 2e-5
    st = make_stats(hyps, p2, pp, pm, np.ones(60))
    d = declare(st, "by1s", hyps, config)
    assert d[h0]["declared"]
    assert d[h0]["sign"] == -1
    assert d[h0]["one_sided"]
    st[h0] = HypStats()
    for proc in PROCEDURES:
        assert not declare(st, proc, hyps, config)[h0]["declared"]
    with pytest.raises(ValueError, match="Prior"):
        declare(st, "wby", hyps, DeclareConfig(n_target=60))


def correlated_z(rng: np.random.Generator, mu: np.ndarray, grp: np.ndarray) -> np.ndarray:
    f = rng.normal(size=4)
    sgn = np.where(np.arange(60) % 2, 1.0, -1.0)
    return mu + 0.8 * sgn * f[grp] + 0.6 * rng.normal(size=60)


@pytest.mark.claim("THM-BY-FIN", "THM-FDR-ASY")
def test_fdr_under_dependence_simulation() -> None:
    hyps, structure = space()
    rng = np.random.default_rng(7)
    q, reps = 0.1, 150
    loads = [i for i, h in enumerate(hyps) if h[2] == "load"]
    alt_groups = {hyps[j][:2] for j in loads[:2]}
    alt = set(loads[:3]) | {i for i, h in enumerate(hyps) if h[:2] in alt_groups}
    mu = np.array([3.0 if i in alt else 0.0 for i in range(60)])
    fams = sorted({h[0] for h in hyps})
    grp = np.array([fams.index(h[0]) for h in hyps])
    zp = {h: (rng.normal(mu[i], 1.0) * 1.5) for i, h in enumerate(hyps)}
    config = DeclareConfig(q=q, prior=Prior(z=zp, n_prior=120), n_target=120, structure=structure)
    res: dict[str, list[float]] = {p: [] for p in PROCEDURES if p != "bh_invalid"}
    for _ in range(reps):
        z = correlated_z(rng, mu, grp)
        st = make_stats(hyps, 2 * norm.sf(np.abs(z)), norm.sf(z), norm.cdf(z), np.sign(z))
        for p, out in res.items():
            d = declare(st, p, hyps, config)
            r = [i for i, h in enumerate(hyps) if d[h]["declared"]]
            v = sum(1 for i in r if i not in alt)
            out.append(float(v > 0) if p in ("gholm", "gholm1s") else v / max(len(r), 1))
    for p, v in res.items():
        assert np.mean(v) <= q + 0.04, (p, np.mean(v))

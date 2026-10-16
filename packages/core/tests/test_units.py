from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from oran_causal.core.design import v_design, v_var
from oran_causal.core.pmrt.units import (
    PmrtConfig,
    Shrinkage,
    TrainSet,
    demean,
    fit_core,
    h_weights,
    lin_z,
    pos_profile,
    rank1_pos,
    running_center,
    slot_matrix,
    split_arm,
    variant_scores,
)

MakeUnits = Callable[..., dict[str, Any]]
Design = tuple[np.ndarray, np.ndarray]


def naive_center(k: np.ndarray, ep: np.ndarray, sg: np.ndarray, shrink: Shrinkage) -> np.ndarray:
    out = np.zeros(len(k))
    for u in range(len(k)):
        g = 1 if sg[u] > 0 else -1
        pp = [k[j] for j in range(u) if (1 if sg[j] > 0 else -1) == g]
        pm = (sum(pp) + shrink.n1 * shrink.prior[g][0]) / (len(pp) + shrink.n1)
        ss = [k[j] for j in range(u) if ep[j] == ep[u] and (1 if sg[j] > 0 else -1) == g]
        out[u] = k[u] - (sum(ss) + shrink.n0 * pm) / (len(ss) + shrink.n0)
    return out


def test_slot_matrix_ranks_relation_cells() -> None:
    rel = np.array([[True, True, True, False], [False, True, True, True]])
    score = np.array([[0, 1, 5, 9], [0, 3, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]], float)
    a = slot_matrix(np.array([0, 1]), rel, score, 3)
    assert a[0].tolist() == [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    assert a[1].tolist() == [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert np.array_equal(a.sum(1), rel.astype(float))


@pytest.mark.claim("ASM-A3-PRED", "DEF-PMRT-WEIGHT")
def test_running_center_matches_naive_and_is_predictable() -> None:
    rng = np.random.default_rng(1)
    n = 200
    k, ep, sg = rng.normal(size=n), np.sort(rng.integers(0, 6, n)), rng.choice([-1.0, 1.0], n)
    shrink = Shrinkage(7.0, 13.0, {1: [0.3], -1: [-0.2]})
    r = running_center(k, ep, sg, shrink)
    assert np.allclose(r, naive_center(k, ep, sg, shrink))
    k2 = k.copy()
    k2[120:] += 1e6
    assert np.array_equal(running_center(k2, ep, sg, shrink)[:120], r[:120])
    assert np.array_equal(running_center(k[:90], ep[:90], sg[:90], shrink), r[:90])


@pytest.mark.claim("DEF-PMRT-WEIGHT", "THM-PMRT-ASY")
def test_running_centre_unbiased_where_realised_fe_is_not(
    make_units: MakeUnits, unit_design: Design
) -> None:
    policy, levels = unit_design
    rng = np.random.default_rng(7)
    z_fe, z_rc = [], []
    for _ in range(300):
        d = make_units(rng, 20)
        probs = np.broadcast_to(policy, (len(d["y"]), 3))
        v = v_design(d["mode"], probs, d["sgn"], levels)
        var = float(v_var(probs, levels)[0])
        r_fe = demean(d["y"], d["episode"] * 2 + (d["sgn"] > 0))
        r_rc = running_center(
            d["y"], d["episode"], d["sgn"], Shrinkage(20.0, 50.0, {1: [0.0], -1: [0.0]})
        )
        z_fe.append(v @ r_fe / np.sqrt(var * (r_fe @ r_fe)))
        z_rc.append(v @ r_rc / np.sqrt(var * (r_rc @ r_rc)))
    se = 1.0 / np.sqrt(len(z_rc))
    assert abs(np.mean(z_rc)) < 3 * se * 1.2
    assert abs(np.mean(z_fe)) > abs(np.mean(z_rc)) + 2 * se


@pytest.mark.claim("DEF-PMRT-SCORE")
def test_v_design_has_mean_zero_and_design_variance(unit_design: Design) -> None:
    policy, levels = unit_design
    probs = np.random.default_rng(0).dirichlet(np.ones(3), 50)
    sgn = np.where(np.arange(50) % 2, 1.0, -1.0)
    mean = sum(probs[:, j] * v_design(np.full(50, j), probs, sgn, levels) for j in range(3))
    assert np.allclose(mean, 0.0)
    second = sum(probs[:, j] * v_design(np.full(50, j), probs, sgn, levels) ** 2 for j in range(3))
    assert np.allclose(second, v_var(probs, levels))
    assert np.allclose(v_var(np.broadcast_to(policy, (1, 3)), levels), 0.7 * 0.3)


@pytest.mark.claim("DEF-PMRT-STAT")
def test_lin_z_is_zero_without_design_variance() -> None:
    v, w = np.array([0.5, -0.5]), np.array([1.0, 2.0])
    assert lin_z(v, w, np.zeros(2)) == 0.0
    assert lin_z(v, w, np.full(2, 0.25)) == pytest.approx(-0.5 / np.sqrt(0.25 * 5))


def test_split_arm_and_profiles() -> None:
    assert split_arm("rank1_h") == ("rank1", "_h")
    assert split_arm("plain") == ("plain", "")
    assert np.allclose(pos_profile(np.array([-1.0, -3.0, 0.5])), [0.25, 0.75, 0.0])
    assert np.allclose(pos_profile(np.zeros(4)), 0.25)
    m = np.outer([1.0, 2.0], [3.0, 1.0, 0.0, 0.0])
    assert np.allclose(rank1_pos(m), m)


def train_set(rng: np.random.Generator, n: int = 400, s: int = 2, effect: float = 2.0) -> TrainSet:
    v = rng.choice([-0.5, 0.5], n)
    x = rng.normal(size=(n, 3))
    p = (
        rng.normal(size=(n, s * 4))
        + 0.5 * x[:, :1]
        + effect * v[:, None] * np.tile([1, 1, 1, 0], s)
    )
    st = rng.integers(0, 6, n) * 2 + (rng.random(n) < 0.5)
    okp = np.ones((n, 4), bool)
    okp[:40, 3] = False
    p[:40, 3::4] = np.nan
    return TrainSet(p, x, okp, v, st)


@pytest.mark.claim("DEF-PMRT-WEIGHT")
def test_fit_core_learns_the_planted_matched_filter() -> None:
    train = train_set(np.random.default_rng(3))
    core = fit_core(train, 2, PmrtConfig())
    assert core["z_plain"] > 10
    assert core["sdir"]["rank1"] == 1.0
    k = variant_scores(core, train.P, train.X, train.okp)
    assert set(k) == {"plain", "plain4", "gls1", "full", "rank1", "loadsp", "prof"}
    assert np.corrcoef(k["plain"], train.v)[0, 1] > 0.5
    assert set(core["prior"]["plain"]) == {1, -1}


@pytest.mark.claim("DEF-PMRT-WEIGHT")
def test_h_weights_are_clipped() -> None:
    train = train_set(np.random.default_rng(4))
    cfg = PmrtConfig(h_clip=2.0)
    core = fit_core(train, 2, cfg)
    hm = {
        "w": np.full(np.sum(core["keep"]), 50.0),
        "xm": np.zeros(np.sum(core["keep"])),
        "ym": 0.0,
        "med": 1.0,
    }
    h = h_weights(hm, core, train.X, cfg)
    assert h.min() >= 0.5
    assert h.max() <= 2.0

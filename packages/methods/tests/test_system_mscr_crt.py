import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import kstest

from oran_causal.core.data import UnitTable
from oran_causal.methods.system import mscr_crt as v2
from oran_causal.sim.system.decision.records import PRIVILEGED_KEYS, build_unit_data
from oran_causal.sim.system.decision.synthetic import SyntheticSpec, synthetic_records
from oran_causal.sim.system.decision.vocab import FAMILIES, KPIS, MODES

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _recs(seed: int, **kw: Any) -> list[dict]:
    return synthetic_records(np.random.default_rng(seed), SyntheticSpec(**kw))


@pytest.fixture(scope="module")
def planted() -> UnitTable:
    return build_unit_data(_recs(11, n_ep=8, units_per_ep=60, effect=200.0))


@pytest.fixture(scope="module")
def null_data() -> UnitTable:
    return build_unit_data(_recs(31, n_ep=8, units_per_ep=60))


def _redraw_modes(d: UnitTable, rng: np.random.Generator) -> UnitTable:
    m = v2.PiAssignment(d).mode_draws(rng, np.arange(d.n), 1)[0]
    return dataclasses.replace(d, mode=m)


def _v1_beta(x: np.ndarray, y: np.ndarray, ep: np.ndarray) -> float:
    xc, yc = x.astype(float).copy(), y.astype(float).copy()
    for e in np.unique(ep):
        m = ep == e
        xc[m] -= xc[m].mean()
        yc[m] -= yc[m].mean()
    den = float(np.dot(xc, xc))
    return float(np.dot(xc, yc) / den) if den > 0 else 0.0


def test_regressor_is_design_centred_and_uses_the_v2_dose(null_data: UnitTable) -> None:
    d = null_data
    rows = d.rows_of("sleep")
    v = v2.design_regressor(d, rows)
    mu = d.probs[rows] @ v2.LEVEL_V2_ARR
    assert np.allclose(v, d.sgn[rows] * (v2.LEVEL_V2_ARR[d.mode[rows]] - mu))
    half = d.mode[rows] == MODES.index("half")
    assert half.any()
    assert np.allclose(v[half], (d.sgn[rows] * (1.0 - mu))[half])
    m = v2.PiAssignment(d).mode_draws(np.random.default_rng(0), rows, 20000)
    vd = d.sgn[rows][None, :] * (v2.LEVEL_V2_ARR[m] - mu[None, :])
    assert np.abs(vd.mean(0)).max() < 0.03


def test_residualise_fe_plus_pre_and_degenerate_pre_guard() -> None:
    rng = np.random.default_rng(0)
    st = np.repeat(np.arange(6), 20)
    pre = rng.normal(size=120)
    y = 3.0 * pre + st * 10.0 + rng.normal(size=120)
    r, info = v2.residualise(y, pre, st)
    x = np.column_stack([(st[:, None] == np.arange(6)[None, :]).astype(float), pre])
    want = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    assert info["resid"] == "fe+pre"
    assert np.allclose(r, want)
    r2, info2 = v2.residualise(y, np.ones(120), st)
    assert info2["resid"] == "fe"
    assert np.allclose(r2, y - np.bincount(st, y)[st] / 20.0)
    r3, info3 = v2.residualise(y, st * 1.0, st)
    assert info3["resid"] == "fe"
    assert np.allclose(r3, r2)


def test_p_values_are_valid_under_a_sharp_null(null_data: UnitTable) -> None:
    cfg = v2.UnitCRTConfigV2(B=199)
    ps = []
    for i in range(200):
        d = _redraw_modes(null_data, np.random.default_rng([77, i]))
        t = v2.crt_unit_test_v2(d, ("sleep", "nbr", "load"), cfg, v2.CrtKey(0, i))
        if t["status"] == "tested":
            ps.append(t["p"])
    ps = np.array(ps)
    assert len(ps) >= 190
    assert 0.42 < ps.mean() < 0.58
    assert (ps <= 0.05).mean() <= 0.09
    assert (ps <= 0.10).mean() <= 0.16
    assert kstest(ps, "uniform").pvalue > 0.01


def test_planted_additive_effect_is_detected_with_its_sign(planted: UnitTable) -> None:
    run = v2.run_crt_units_v2(planted, v2.UnitCRTConfigV2(B=1999), families=("sleep",))
    res = {(r["relation"], r["kpi"]): r for r in run["results"]}
    nl = res[("nbr", "load")]
    assert (nl["status"], nl["sign"]) == ("declared", 1)
    assert nl["z_approx"] > 3
    assert (res[("nbr", "pv")]["status"], res[("nbr", "pv")]["sign"]) == ("declared", 1)
    assert res[("own", "load")]["sign"] == -1
    assert res[("own", "load")]["p"] <= 0.01
    for k in KPIS:
        assert res[("far", k)]["status"] != "declared"
    e = v2.edges_from_crt_v2(run)
    assert e[("sleep", "nbr", "load")]["declared"]
    assert e[("sleep", "nbr", "load")]["sign"] == 1
    assert not e[("sleep", "far", "load")]["declared"]
    assert run["power_floor"]["rank1_ok"]


def test_sign_fix_when_sgn_is_confounded_with_the_outcome_trend(null_data: UnitTable) -> None:
    d = null_data
    rng = np.random.default_rng(5)
    rows = d.rows_of("sleep")
    key = ("nbr", "load")
    lv2 = v2.LEVEL_V2_ARR[d.mode]
    y = 50.0 * d.sgn - 10.0 * lv2 * d.sgn + rng.normal(0, 2.0, d.n)
    dd = dataclasses.replace(d, y={**d.y, key: y})
    assert _v1_beta(dd.x[rows], y[rows], dd.episode[rows]) > 0
    r = v2.crt_unit_test_v2(dd, ("sleep", *key), v2.UnitCRTConfigV2(B=999))
    assert r["sign"] == -1
    assert r["p"] <= 0.01
    assert r["beta"] == pytest.approx(-10.0, rel=0.25)


def test_chunk_invariance(planted: UnitTable) -> None:
    a = v2.UnitCRTConfigV2(B=1500, chunk=1500)
    b = dataclasses.replace(a, chunk=400)
    c = dataclasses.replace(a, max_chunk_bytes=8 * 150 * 37)
    for h in (("nbr", "load"), ("far", "pv"), ("own", "e")):
        ra, rb, rc = (v2.crt_unit_test_v2(planted, ("sleep", *h), x) for x in (a, b, c))
        assert rc["chunk"] < rb["chunk"] < ra["chunk"]
        assert ra["p"] == rb["p"] == rc["p"]
        assert ra["z_approx"] == pytest.approx(rc["z_approx"], rel=1e-9)
    run = v2.run_crt_units_v2(planted, dataclasses.replace(a, B=199), families=("ptx",))
    single = v2.crt_unit_test_v2(planted, ("ptx", "own", "e"), dataclasses.replace(a, B=199))
    batched = next(r for r in run["results"] if (r["relation"], r["kpi"]) == ("own", "e"))
    assert batched["p"] == single["p"]
    assert batched["s_obs"] == pytest.approx(single["s_obs"], rel=1e-12)


def test_streams_carry_the_version_component() -> None:
    fi = FAMILIES.index("sleep")
    u1 = np.random.default_rng([0, v2.CRT_TAG, fi, 0, 0]).random(5)
    u2 = v2._stream(v2.UnitCRTConfigV2(), fi, v2.KEY0).random(5)
    assert np.array_equal(u2, np.random.default_rng([0, 6616, 2, fi, 0, 0]).random(5))
    assert not np.allclose(u1, u2)


def test_placebo_k0_rule() -> None:
    d = build_unit_data(_recs(21, n_ep=8, units_per_ep=60, effect=60.0, placebo=True))
    assert np.all(d.applied == MODES.index("accept"))
    assert np.any(d.mode != d.applied)
    k0 = v2.placebo_rejection_units_v2(d, v2.UnitCRTConfigV2(B=199))
    assert k0["n_tested"] > 0
    assert k0["n_by_declared"] <= 1
    assert k0["pass"]
    assert k0["k_max"] == 7 if k0["n_tested"] == 60 else k0["k_max"] >= 1
    sl = next(
        h
        for h in k0["per_hypothesis"]
        if (h["family"], h["relation"], h["kpi"]) == ("sleep", "nbr", "load")
    )
    assert sl["p"] > 0.01
    with pytest.raises(ValueError, match="placebo"):
        v2.placebo_rejection_units_v2(build_unit_data(_recs(1, n_ep=2)))


def test_support_rule(planted: UnitTable) -> None:
    one = planted.subset(planted.episode < 2)
    r = v2.crt_unit_test_v2(one, ("sleep", "nbr", "load"), v2.UnitCRTConfigV2(B=99))
    assert (r["status"], r["reason"]) == ("undetermined", "support")


class _Poison(dict):
    def __getitem__(self, k: Any) -> Any:
        raise AssertionError("privileged field read")

    def get(self, *a: Any, **k: Any) -> Any:
        raise AssertionError("privileged field read")

    def __iter__(self) -> Any:
        raise AssertionError("privileged field read")

    def items(self) -> Any:
        raise AssertionError("privileged field read")


def test_privileged_fields_are_never_read() -> None:
    recs = _recs(8, n_ep=4, units_per_ep=40)
    cfg = v2.UnitCRTConfigV2(B=99, min_episodes=1)
    clean = v2.run_crt_units_v2(build_unit_data(recs), cfg, families=("sleep",))
    for r in recs:
        for k in PRIVILEGED_KEYS:
            r[k] = _Poison(cand={"1": 2})
    pois = v2.run_crt_units_v2(build_unit_data(recs), cfg, families=("sleep",))
    assert [r["p"] for r in pois["results"]] == [r["p"] for r in clean["results"]]
    body = Path(v2.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for k in PRIVILEGED_KEYS:
        assert k not in body


def test_carryover_audit_is_optional_and_descriptive(planted: UnitTable) -> None:
    run = v2.run_crt_units_v2(planted, v2.UnitCRTConfigV2(B=99), families=("sleep",))
    assert "audit" not in run
    a = v2.carryover_audit_v2(
        planted, v2.UnitCRTConfigV2(B=99, audit_B=99), families=("sleep", "ptx")
    )
    assert a["B"] == 99
    assert all(r["lag"] == 1 for r in a["results"])


def test_draws_match_the_logged_pi0_propensities() -> None:
    recs = _recs(4, n_ep=4, units_per_ep=50)
    tab = {"accept": 0.7, "half": 0.1, "reject": 0.2}
    recs[0] = dict(recs[0], pi0_table={**recs[0]["pi0_table"], "SliceGuarantee": tab})
    recs[0]["units"] = [
        dict(u, p=tab[u["mode"]]) if u["x"] == "SliceGuarantee" else u for u in recs[0]["units"]
    ]
    d = build_unit_data(recs)
    assert d.meta["p_mismatch"] == 0
    pa = v2.PiAssignment(d)
    rows = d.rows_of("prot_min")
    m = pa.conditional_draws(np.random.default_rng([0, v2.CRT_TAG]), "prot_min", 20000, what="mode")
    for sel, want in (
        (d.episode[rows] == 0, (0.7, 0.1, 0.2)),
        (d.episode[rows] > 0, (0.5, 0.2, 0.3)),
    ):
        freq = [(m[:, sel] == MODES.index(k)).mean() for k in ("accept", "half", "reject")]
        assert np.allclose(freq, want, atol=0.01)
    x = pa.conditional_draws(np.random.default_rng([0, v2.CRT_TAG]), "prot_min", 50)
    lv = pa.conditional_draws(np.random.default_rng([0, v2.CRT_TAG]), "prot_min", 50, what="level")
    assert np.array_equal(x, lv * d.sgn[rows][None, :])
    with pytest.raises(ValueError, match="rows outside"):
        pa.conditional_draws(np.random.default_rng(0), "prot_min", 5, rows=d.rows_of("sleep"))


def test_power_floor() -> None:
    pf = v2.power_floor(999, 60)
    assert not pf["rank1_ok"]
    assert pf["min_hypotheses_at_floor_to_declare"] == 6
    big = v2.power_floor(9999, 60)
    assert big["rank1_ok"]
    assert big["min_B_rank1"] <= 9999

import copy
from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system import config as C
from oran_causal.sim.system.decision import collect as CP
from oran_causal.sim.system.decision import units as U
from oran_causal.sim.system.env import E6Env

FEASIBLE = CP.Collection(open_rule="feasible")


def cfg_p3(seed: int = 6, warm: float = 60.0, scored: float = 90.0) -> C.E6Config:
    return C.E6Config(
        seed=seed,
        mix="ES",
        scenario="surge",
        load_factor=1.0,
        n_ue=300,
        n_pico=3,
        mobility="ped",
        warmup_s=warm,
        scored_s=scored,
        e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")),
    )


def accept_all_reference(cfg: C.E6Config) -> E6Env:
    ref = E6Env(cfg, log=False, wg3=True)
    while ref.sec < ref.total_s:
        ref.step(None)
    return ref


@pytest.mark.claim("DEF-SYS-LOG")
def test_pi0_tables_match_the_design_and_the_propensity_floor() -> None:
    assert CP.PI0_HIGH["ES"] == {"accept": 0.5, "half": 0.2, "reject": 0.3}
    sg = {"accept": 0.5, "half": 0.15, "reject": 0.25, "accept+rb": 0.10}
    assert CP.PI0_HIGH["SliceGuarantee"] == sg
    lo = CP.PI0_LOW
    assert lo["ES"] == pytest.approx({"accept": 0.78, "half": 0.10, "reject": 0.12})
    assert lo["PowerES"] == pytest.approx(lo["ES"])
    sg_lo = {"accept": 0.70, "half": 0.10, "reject": 0.10, "accept+rb": 0.10}
    assert lo["SliceGuarantee"] == pytest.approx(sg_lo)
    for reg in ("high", "low"):
        for t in CP.PI0[reg].values():
            assert sum(t.values()) == pytest.approx(1.0)
            assert min(t.values()) >= CP.P_MIN - 1e-12
    want = {"accept": 0.8, "half": 0.08, "reject": 0.12}
    assert CP.low_table(CP.PI0_HIGH["ES"], p_min=None) == pytest.approx(want)
    assert [CP.regime_for(j) for j in range(4)] == ["high", "low", "high", "low"]
    for x in ("ES", "PowerES", "SliceGuarantee"):
        assert CP.PI0_HIGH_NO_RB[x] == {"accept": 0.5, "half": 0.2, "reject": 0.3}
        assert CP.PI0_V4[x] == {"accept": 0.5, "reject": 0.5}
    assert CP.RandomizedUnitPolicy(5, "high_no_rb").tables is CP.PI0_HIGH_NO_RB


@pytest.mark.claim("DEF-SYS-LOG")
def test_draws_are_keyed_by_seed_cell_xapp_t0_and_follow_the_table() -> None:
    pol = CP.RandomizedUnitPolicy(123, "high")
    u = {"c": 4, "x": "SliceGuarantee", "x_idx": 2, "t0": 300.0}
    first = pol(dict(u))
    assert all(pol(dict(u)) == first for _ in range(5))
    want = float(np.random.default_rng([123, 6612, 4, 2, 300]).random())
    assert CP.RandomizedUnitPolicy(123, "high").draw_u(u) == want
    n, cnt = 0, {}
    for c in range(24):
        for t0 in range(120, 720):
            m, p = pol({"c": c, "x": "SliceGuarantee", "x_idx": 2, "t0": float(t0)})
            cnt[m] = cnt.get(m, 0) + 1
            n += 1
            assert p == CP.PI0_HIGH["SliceGuarantee"][m]
    for m, p in CP.PI0_HIGH["SliceGuarantee"].items():
        assert cnt[m] / n == pytest.approx(p, abs=0.01)
    with pytest.raises(ValueError, match="regime"):
        CP.RandomizedUnitPolicy(1, "mid")


def _ctx(knob: str, step: float, own: float, nbr: float, below: float) -> dict:
    return {
        "knob": knob,
        "step": step,
        "own_prb_util": own,
        "nbr_max_prb_util": nbr,
        "own_prot_below_frac": below,
    }


@pytest.mark.claim("DEF-SYS-LOG")
@pytest.mark.claim("ASM-A1-DESIGN")
def test_incumbent_propensity_depends_on_context_only_and_is_bounded() -> None:
    pol = CP.IncumbentPolicy(7)
    cases: list[tuple[dict[str, Any], str, float]] = [
        (_ctx("carrier", -1.0, 0.3, 0.2, 0.0), "saving", 0.85),
        (_ctx("sleep", 1.0, 0.7, np.nan, 0.0), "saving", 0.15),
        (_ctx("ptx", 1.0, np.nan, np.nan, 0.0), "restore", 0.35),
        (_ctx("carrier", 1.0, 0.1, 0.9, 0.0), "restore", 0.85),
        (_ctx("prot_min", 0.05, 0.1, 0.1, 0.10), "sg_raise", 0.85),
        (_ctx("prot_min", 0.05, 0.1, 0.1, np.nan), "sg_raise", 0.30),
        (_ctx("prot_min", -0.05, 0.1, 0.1, 0.0), "sg_lower", 0.85),
        (_ctx("prot_min", -0.05, 0.1, 0.1, 0.02), "sg_lower", 0.20),
    ]
    for ctx, cls, pa in cases:
        p, info = pol.accept_prob(ctx)
        assert info["cls"] == cls
        assert p == pytest.approx(pa)
        assert CP.INC_FLOOR <= p <= 1 - CP.INC_FLOOR
    tight = CP.IncumbentPolicy(7, table={**CP.INCUMBENT_TABLE, "saving": {"lo": 1.0, "hi": 0.0}})
    assert tight.accept_prob(cases[0][0])[0] == pytest.approx(0.85)
    assert tight.accept_prob(cases[1][0])[0] == pytest.approx(0.15)
    unit: dict[str, Any] = {"c": 3, "x_idx": 0, "t0": 200.0, "ctx": cases[0][0]}
    m, p = pol(unit)
    assert unit["probs"] == {"accept": 0.85, "reject": pytest.approx(0.15)}
    assert p == unit["probs"][m]
    key = [7, CP.INC_TAG, 3, 0, 200]
    assert m == ("accept" if np.random.default_rng(key).random() < 0.85 else "reject")
    with pytest.raises(ValueError, match="no rule"):
        CP.incumbent_class({"knob": "cio", "step": 1.0})
    with pytest.raises(ValueError, match="floor"):
        CP.IncumbentPolicy(1, floor=0.6)
    consts = CP.incumbent_consts()
    assert consts["floor"] == CP.INC_FLOOR
    assert consts["tag"] == CP.INC_TAG


@pytest.mark.claim("DEF-SYS-LOG")
def test_placebo_incumbent_logs_the_draw_and_applies_accept() -> None:
    unit: dict[str, Any] = {
        "c": 3,
        "x_idx": 0,
        "t0": 200.0,
        "ctx": _ctx("carrier", -1.0, 0.3, 0.2, 0.0),
    }
    want = CP.IncumbentPolicy(7)(dict(unit))
    unit["mode"], unit["p"] = CP.PlaceboIncumbent(7)(unit)
    assert (unit["mode"], unit["p"]) == ("accept", 1.0)
    assert (unit["pi0_mode"], unit["pi0_p"]) == want
    units = [unit, {"mode": "reject", "p": 0.5}]
    CP.finalize_units(units)
    assert (units[0]["mode"], units[0]["p"], units[0]["applied_mode"]) == (*want, "accept")
    assert "applied_mode" not in units[1]


def test_all_accept_pi0_is_bit_identical_to_accept_all_and_the_tap_matches_sla() -> None:
    cfg = cfg_p3()
    ref = accept_all_reference(cfg)
    tables = {x: {"accept": 1.0} for x in CP.PI0_HIGH}
    res = CP.run_collection(cfg, CP.RandomizedUnitPolicy(cfg.seed, tables=tables))
    env = res["env"]
    assert env.plant.sla == ref.plant.sla
    assert env.stats == ref.stats
    assert np.array_equal(env.plant.q, ref.plant.q)
    assert env.config() == ref.config()
    assert res["units"]
    assert all(u["mode"] == "accept" and u["p"] == 1.0 for u in res["units"])
    tap, sla = res["tap"], env.plant.sla
    assert tap.pv.sum() == sla["prot_viol"]
    assert tap.v.sum() == sla["viol_ue_s"]
    assert tap.e.sum() == pytest.approx(sla["energy_j"], rel=1e-12)
    assert sla["prot_viol"] > 0
    assert sla["energy_j"] > 0


def test_unit_log_context_propensity_and_window_kpis() -> None:
    cfg = cfg_p3(seed=7)
    res = CP.run_collection(cfg, CP.RandomizedUnitPolicy(cfg.seed, "high"))
    units = res["units"]
    assert {u["x"] for u in units} >= {"ES", "SliceGuarantee"}
    assert {u["mode"] for u in units} >= {"accept", "half", "reject"}
    for u in units:
        assert u["t0"] >= cfg.warmup_s
        assert u["p"] == CP.PI0_HIGH[u["x"]][u["mode"]]
        assert u["c"] in u["exp"]
        k = u["kpi"]
        assert k["trunc"] or k["secs"] == U.T_UNIT
        assert k["e"] >= k["e_own"] > 0
        assert k["pv"] >= k["pv_own"] >= 0
        assert k["v"] >= k["v_own"] >= 0
        assert "own_prot_below_frac" in u["ctx"]
        assert "nbr_max_prb_util" in u["ctx"]
        assert set(u["ctx"]) == set(units[0]["ctx"])
    ok = [u for u in units if not u["kpi"]["trunc"]]
    assert ok
    assert any(u["kpi"]["pv"] > 0 for u in units)
    env = E6Env(cfg, log=False, wg3=True)
    tap = CP.install_tap(env)
    arb = U.UnitArbiter(CP.RandomizedUnitPolicy(cfg.seed, "high"), warmup_s=cfg.warmup_s)
    cum = {}
    while env.sec < env.total_s:
        o = env.step_propose()
        cum[int(o["t"])] = tap.arrays()
        env.step_apply(arb(o))
    for u in ok:
        t0, t1, ex = int(u["t0"]), int(u["t0"] + U.T_UNIT), u["exp"]
        want = [float(cum[t1][i][ex].sum() - cum[t0][i][ex].sum()) for i in range(3)]
        got = [u["kpi"]["pv"], u["kpi"]["e"], u["kpi"]["v"]]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_collection_reproduces_inside_env_copy() -> None:
    cfg = cfg_p3(seed=8, scored=60.0)
    pol = CP.RandomizedUnitPolicy(cfg.seed, "high")
    env = E6Env(cfg, log=False, wg3=True)
    CP.install_tap(env)
    arb = U.UnitArbiter(pol, warmup_s=cfg.warmup_s)
    while env.sec < 90:
        env.step_apply(arb(env.step_propose()))
    env2, arb2 = env.copy(), copy.deepcopy(arb)
    assert CP.get_tap(env2) is not CP.get_tap(env)
    assert CP.get_tap(env2).plant is env2.plant
    n0 = len(arb.units)
    for e, a in ((env, arb), (env2, arb2)):
        while e.sec < e.total_s:
            e.step_apply(a(e.step_propose()))

    def key(units: list) -> list:
        return [(u["c"], u["x"], u["t0"], u["mode"], u["p"]) for u in units[n0:]]

    assert key(arb.units)
    assert key(arb.units) == key(arb2.units)
    assert env.plant.sla == env2.plant.sla
    assert np.array_equal(CP.get_tap(env).pv, CP.get_tap(env2).pv)
    with pytest.raises(RuntimeError, match="no CellKPITap"):
        CP.get_tap(E6Env(cfg, log=False, wg3=True))


@pytest.mark.claim("DEF-SYS-LOG")
def test_placebo_logs_pi0_draws_applies_accept_and_is_bit_identical_to_accept_all() -> None:
    cfg = cfg_p3(seed=6, scored=60.0)
    ref = accept_all_reference(cfg)
    tab = CP.PI0_HIGH_NO_RB
    res = CP.run_collection(cfg, CP.PlaceboPolicy(cfg.seed, tables=tab), FEASIBLE)
    env = res["env"]
    assert env.plant.sla == ref.plant.sla
    assert env.stats == ref.stats
    assert np.array_equal(env.plant.q, ref.plant.q)
    assert env.config() == ref.config()
    pi0 = CP.RandomizedUnitPolicy(cfg.seed, tables=tab)
    for u in res["units"]:
        assert u["applied_mode"] == "accept"
        assert "pi0_mode" not in u
        assert (u["mode"], u["p"]) == pi0(dict(u))
        assert u["p"] == tab[u["x"]][u["mode"]]
    assert {u["mode"] for u in res["units"]} >= {"accept", "half", "reject"}


def test_tap_series_matches_the_cumulative_arrays_and_sla() -> None:
    cfg = cfg_p3(seed=7, scored=60.0)
    pol = CP.RandomizedUnitPolicy(cfg.seed, tables=CP.PI0_HIGH_NO_RB)
    res = CP.run_collection(cfg, pol, FEASIBLE)
    env, tap = res["env"], res["tap"]
    s = tap.series()
    T = env.total_s
    assert s["fields"] == list(CP.SERIES_FIELDS)
    assert s["data"].shape == (T, len(CP.SERIES_FIELDS), env.plant.nc)
    assert np.array_equal(s["t"], np.arange(1, T + 1))
    assert s["scored"].sum() == T - int(cfg.warmup_s)
    d, sc = s["data"].astype(float), s["scored"]
    f = {k: i for i, k in enumerate(CP.SERIES_FIELDS)}
    ext = dict(zip(("pv", "e", "v", *CP.EXT_FIELDS), tap.arrays(ext=True), strict=True))
    assert np.array_equal(d[sc, f["pv"]].sum(0), ext["pv"])
    assert np.array_equal(d[sc, f["v"]].sum(0), ext["v"])
    assert np.allclose(d[sc, f["e"]].sum(0), ext["e"], rtol=1e-5)
    assert np.array_equal(d[sc, f["rlf"]].sum(0), ext["rlf"])
    assert np.array_equal(d[sc, f["ue"]].sum(0), ext["load"])
    assert np.allclose(d[sc, f["prb_used"]].sum(0), ext["prb"], rtol=1e-5)
    assert d[:, f["rlf"]].sum() == env.plant.sla["rlf"]
    assert np.all(d[:, f["ue"]].sum(1) == env.plant.n)
    assert np.all(d[:, f["prot_ue"]].sum(1) == env.plant.prot.sum())
    assert np.all(d[:, f["prb_used"]] <= d[:, f["prb_cap"]] + 1e-3)
    assert np.all(d[:, f["pv"]] <= d[:, f["prot_ue"]])
    assert len(tap.arrays()) == 3


def test_count_all_tap_counts_warmup_seconds() -> None:
    cfg = cfg_p3(seed=7, warm=60.0, scored=30.0)
    pol = CP.RandomizedUnitPolicy(cfg.seed, tables=CP.PI0_V4)
    opts = CP.Collection(open_rule="feasible", arb_warmup_s=0.0, count_all=True)
    res = CP.run_collection(cfg, pol, opts)
    s, tap = res["tap"].series(), res["tap"]
    d = s["data"].astype(float)
    assert np.array_equal(d[:, 0].sum(0), tap.pv)
    assert np.array_equal(d[:, 6].sum(0), tap.load)
    assert d[~s["scored"], 6].sum() > 0

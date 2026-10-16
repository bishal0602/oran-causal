import dataclasses
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system import config as C
from oran_causal.sim.system import sim
from oran_causal.sim.system.baselines import SMORestore, freeze
from oran_causal.sim.system.env import Arbiter, E6Env
from oran_causal.sim.system.ric import MOB_KEYS
from oran_causal.sim.system.sim import MistuneScenario, Plant, SurgeScenario


def cfg(**kw: Any) -> C.E6Config:
    base: dict[str, Any] = {
        "seed": 5,
        "load": "medium",
        "mobility": "mixed",
        "mix": "none",
        "warmup_s": 0.0,
        "scored_s": 100.0,
    }
    return C.E6Config(**{**base, **kw})


def ticks(p: Plant, n: int) -> None:
    for _ in range(n):
        p.tick()


def seconds(s: float) -> int:
    return round(s / C.TICK_S)


def surge(p: Plant) -> SurgeScenario:
    assert isinstance(p.scn, SurgeScenario)
    return p.scn


def mistune(p: Plant) -> MistuneScenario:
    assert isinstance(p.scn, MistuneScenario)
    return p.scn


def test_base_ignores_scenario_parameters() -> None:
    a = sim.Plant(cfg())
    b = sim.Plant(
        cfg(scenario="base", surge_mult=9.0, corr_frac_ue=0.5, mis_hys_db=5.0, mis_ttt_ms=640)
    )
    assert a.scn is None
    assert b.scn is None
    ticks(a, 100)
    ticks(b, 100)
    assert a.sla == b.sla
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.serv, b.serv)


def test_unknown_scenario_rejected() -> None:
    with pytest.raises(ValueError, match="scenario"):
        sim.Plant(cfg(scenario="outage"))


def test_holdout_variants_are_single_factor_config_overrides() -> None:
    fields = {f.name for f in dataclasses.fields(C.E6Config)}
    dev = C.E6Config()
    for scn, variants in C.SCENARIO_HOLDOUT.items():
        assert scn in sim.SCENARIOS
        for name, over in variants.items():
            assert set(over) <= fields, name
            assert any(getattr(dev, k) != v for k, v in over.items()), name
            dataclasses.replace(dev, scenario=scn, **over).validate_scenario()


def test_surge_multiplier_inside_region_during_window_only() -> None:
    c = cfg(scenario="surge", warmup_s=50.0, scored_s=100.0)
    p = sim.Plant(c)
    s = surge(p)
    t0, ramp, t_end = s.t0, s.ramp, s.t_end
    assert t0 == pytest.approx(50.0 + c.surge_onset_frac * 100.0)
    assert t_end < 150.0
    pos = np.vstack([s.centre(t0), s.centre(t0) + np.array([c.surge_radius_m + 60.0, 0.0])])
    pos = p.lay.wrap(pos)
    hold = t0 + ramp + 1.0
    np.testing.assert_allclose(s.traffic_mult(hold, pos), [c.surge_mult, 1.0])
    mid = [1 + (c.surge_mult - 1) / 2, 1.0]
    np.testing.assert_allclose(s.traffic_mult(t0 + ramp / 2, pos), mid)
    for t in (0.0, t0 - 1.0, t_end + 1.0):
        assert np.all(s.traffic_mult(t, pos) == 1.0)


class _FileRecorder:
    def __init__(self, g: np.random.Generator, rec: dict[str, np.ndarray]) -> None:
        self.g, self.rec, self.calls = g, rec, 0

    def poisson(self, lam: np.ndarray) -> np.ndarray:
        out = self.g.poisson(lam)
        self.calls += 1
        if self.calls == 2:
            self.rec["files"] = out
        return out


def test_surge_realised_offered_load_in_region(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = sim.tape_rng
    rec: dict[str, np.ndarray] = {}

    def rng(seed: int, stream: str, t: int = 0, salt: int | None = None) -> Any:
        g = orig(seed, stream, t, salt)
        return g if stream != "traffic" else _FileRecorder(g, rec)

    monkeypatch.setattr(sim, "tape_rng", rng)
    kw: dict[str, Any] = {
        "surge_onset_frac": 0.0,
        "surge_ramp_frac": 0.0,
        "surge_hold_frac": 1.0,
        "load": "high",
    }
    tot = {}
    for arm in ("base", "surge"):
        p = sim.Plant(cfg(scenario=arm, **kw))
        geo = surge(sim.Plant(cfg(scenario="surge", **kw)))
        acc = np.zeros(2)
        for _ in range(seconds(60)):
            now = p.t * C.TICK_S
            p.tick()
            files = np.where(p.sl != sim.LL, rec["files"], 0)
            ins = geo.inside(now, p.pos)
            acc += [files[ins].sum(), files[~ins].sum()]
        tot[arm] = acc
    r_in = tot["surge"][0] / tot["base"][0]
    r_out = tot["surge"][1] / tot["base"][1]
    assert tot["base"][0] >= 50
    assert 0.8 * C.E6Config().surge_mult < r_in < 1.2 * C.E6Config().surge_mult
    assert 0.85 < r_out < 1.15


def test_moving_surge_drifts_at_declared_speed() -> None:
    v = float(C.SCENARIO_HOLDOUT["surge"]["H-move"]["surge_speed_mps"])
    s = surge(sim.Plant(cfg(scenario="surge", surge_speed_mps=v)))
    d = s.dist(s.centre(s.t0 + 20.0)[None], s.centre(s.t0))[0]
    assert s.t_end > s.t0 + 20.0
    assert d == pytest.approx(20.0 * v, rel=1e-6)
    assert np.allclose(s.centre(s.t_end + 50.0), s.centre(s.t_end))
    assert np.allclose(s.centre(0.0), s.centre(s.t0))


def test_surge_location_modes() -> None:
    for seed in range(4):
        band = surge(sim.Plant(cfg(seed=seed, scenario="surge"))).c0
        vert = surge(sim.Plant(cfg(seed=seed, scenario="surge", surge_loc="vertex"))).c0
        lay = sim.Plant(cfg(seed=seed)).lay
        db = sim.wrap_dist(lay, lay.sites, band).min()
        dv = np.sort(sim.wrap_dist(lay, lay.sites, vert))
        assert 150.0 - 1e-6 <= db <= 250.0 + 1e-6
        np.testing.assert_allclose(dv[:3], C.ISD_M / np.sqrt(3.0), rtol=1e-6)


def _on_segment(p: Plant) -> np.ndarray:
    s = mistune(p)
    rel = p.pos[s.idx][:, None, :] - (s.centre + p.lay.cands())[None]
    along = rel @ s.u
    perp = np.abs(rel @ np.array([-s.u[1], s.u[0]]))
    ok = (perp < 1e-6) & (np.abs(along) <= s.half + 1e-6)
    return ok.any(1)


def test_mistune_corridor_vehicles_from_t0() -> None:
    c = cfg(scenario="mistune")
    p = sim.Plant(c)
    s = mistune(p)
    assert len(s.idx) == round(c.corr_frac_ue * p.n)
    np.testing.assert_allclose(p.speed[s.idx], c.corr_speed_kmh / 3.6)
    assert not p.indoor[s.idx].any()
    assert _on_segment(p).all()
    assert not np.isin(s.idx, p.mobile_idx).any()
    s0 = s.s.copy()
    ticks(p, seconds(5))
    assert _on_segment(p).all()
    moved = np.abs(s.s - s0)
    far = np.abs(s0) < s.half - 5 * s.v
    np.testing.assert_allclose(moved[far], 5 * s.v, rtol=1e-6)


def test_mistune_rollout_sets_cluster_parameters_at_onset_only() -> None:
    c = cfg(scenario="mistune", warmup_s=2.0, scored_s=20.0, mis_onset_frac=0.2)
    p = sim.Plant(c)
    s = mistune(p)
    hys0, ttt0 = p.hys.copy(), p.ttt.copy()
    assert len(s.cluster) >= 2
    ticks(p, seconds(5.9))
    np.testing.assert_array_equal(p.hys, hys0)
    np.testing.assert_array_equal(p.ttt, ttt0)
    ticks(p, seconds(0.2))
    out = np.setdiff1d(np.arange(p.nc), s.cluster)
    assert np.all(p.hys[s.cluster] == c.mis_hys_db)
    assert np.all(p.ttt[s.cluster] == c.mis_ttt_ms)
    np.testing.assert_array_equal(p.hys[out], hys0[out])
    np.testing.assert_array_equal(p.ttt[out], ttt0[out])
    p.hys[s.cluster] = 2.0
    ticks(p, 5)
    assert np.all(p.hys[s.cluster] == 2.0)


def test_mistune_cluster_covers_the_corridor() -> None:
    p = sim.Plant(cfg(scenario="mistune"))
    s = mistune(p)
    pts = p.lay.wrap(s.centre + np.linspace(-s.half, s.half, 41)[:, None] * s.u)
    best = np.argmax(p.gm.lookup(pts), 1)
    assert np.isin(best, s.cluster).all()


@pytest.mark.parametrize("scn", ["surge", "mistune"])
def test_scenario_tape_is_action_independent(scn: str) -> None:
    c = cfg(scenario=scn, surge_onset_frac=0.0, surge_ramp_frac=0.0)
    a, b = sim.Plant(c), sim.Plant(c)
    b.cio[:, :] = 3.0
    b.ll_ratio[:] = 0.3
    for _ in range(seconds(4)):
        a.tick()
        b.tick()
    np.testing.assert_array_equal(a.pos, b.pos)
    now = a.t * C.TICK_S
    assert a.scn is not None
    assert b.scn is not None
    np.testing.assert_array_equal(a.scn.traffic_mult(now, a.pos), b.scn.traffic_mult(now, b.pos))


@pytest.mark.parametrize(
    ("scn", "bad"),
    [
        ("surge", {"surge_radius_m": -1.0}),
        ("surge", {"surge_radius_m": 0.0}),
        ("surge", {"surge_speed_mps": -0.1}),
        ("surge", {"surge_mult": 0.5}),
        ("surge", {"surge_onset_frac": 1.2}),
        ("surge", {"surge_ramp_frac": -0.1}),
        ("surge", {"surge_onset_frac": 0.5, "surge_ramp_frac": 0.2, "surge_hold_frac": 0.3}),
        ("surge", {"surge_loc": "nowhere"}),
        ("surge", {"scored_s": 0.0}),
        ("mistune", {"corr_frac_ue": 0.0}),
        ("mistune", {"corr_frac_ue": 1.5}),
        ("mistune", {"corr_frac_ue": 0.001}),
        ("mistune", {"corr_len_m": 0.0}),
        ("mistune", {"corr_len_m": 1e5}),
        ("mistune", {"corr_speed_kmh": 0.0}),
        ("mistune", {"corr_loc": "nowhere"}),
        ("mistune", {"mis_onset_frac": 1.0}),
        ("mistune", {"mis_onset_frac": -0.1}),
        ("mistune", {"mis_hys_db": 5.5}),
        ("mistune", {"mis_hys_db": 2.25}),
        ("mistune", {"mis_hys_db": -0.5}),
        ("mistune", {"mis_ttt_ms": 500}),
        ("mistune", {"mis_ttt_ms": 1280}),
    ],
)
def test_invalid_scenario_parameters_rejected(scn: str, bad: dict[str, Any]) -> None:
    c = cfg(scenario=scn, **bad)
    with pytest.raises(ValueError, match=r"\w"):
        c.validate_scenario()
    with pytest.raises(ValueError, match=r"\w"):
        sim.make_scenario(c, None)


def test_invalid_parameters_rejected_by_plant_constructor() -> None:
    with pytest.raises(ValueError, match="surge_radius_m"):
        sim.Plant(cfg(scenario="surge", surge_radius_m=-5.0))


def test_default_length_event_timing() -> None:
    d = C.E6Config()
    end = d.warmup_s + d.scored_s
    s = surge(sim.Plant(dataclasses.replace(d, scenario="surge")))
    assert s.t0 == pytest.approx(d.warmup_s + d.surge_onset_frac * d.scored_s)
    span = (2 * d.surge_ramp_frac + d.surge_hold_frac) * d.scored_s
    assert s.t_end == pytest.approx(s.t0 + span)
    assert d.warmup_s < s.t0 < s.t_end <= end
    m = mistune(sim.Plant(dataclasses.replace(d, scenario="mistune")))
    assert m.t_mis == pytest.approx(d.warmup_s + d.mis_onset_frac * d.scored_s)
    assert d.warmup_s < m.t_mis < end


def run_to(env: E6Env, t: int, arbiter: Arbiter | None = None) -> None:
    while env.sec < t:
        env.step(arbiter)


def rollback(env: E6Env, knobs: Iterable[tuple]) -> None:
    obs = env.step_propose()
    env.step_apply(
        {"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": list(knobs)}
    )


def test_oam_push_once_persists_and_is_not_rollbackable() -> None:
    c = cfg(scenario="mistune", warmup_s=10.0, scored_s=1800.0)
    env = E6Env(c, log=False)
    p = env.plant
    scn = mistune(p)
    c0 = int(scn.cluster[0])
    run_to(env, 50)
    assert env.apply(("hys", c0), 2.5, float(env.sec))[0] == "ok"
    assert env.prev_val[("hys", c0)] == 2.0
    run_to(env, int(np.ceil(scn.t_mis)) + 1)
    assert scn.push_t == pytest.approx(scn.t_mis, abs=C.TICK_S + 1e-9)
    assert [tag for _, tag in scn.oam_log] == ["push"]
    assert np.all(p.hys[scn.cluster] == c.mis_hys_db)
    assert np.all(p.ttt[scn.cluster] == c.mis_ttt_ms)
    assert scn.pre_hys is not None
    assert scn.pre_hys[0] == 2.5
    for x in scn.cluster:
        assert env.prev_val[("hys", int(x))] == c.mis_hys_db
        assert env.prev_val[("ttt", int(x))] == c.mis_ttt_ms
    rollback(env, [("hys", c0)])
    assert p.hys[c0] == c.mis_hys_db
    assert env.stats["rollbacks"] == 0
    now = float(env.sec)
    assert env.apply(("ttt", c0), 320, now)[0] == "ok"
    assert p.ttt[c0] == 320
    assert env.prev_val[("ttt", c0)] == c.mis_ttt_ms
    nb = p.lay.neighbours[c0][0]
    assert env.apply(("cio", c0, nb), 1.0, now)[0] == "ok"
    run_to(env, env.sec + 11)
    assert np.all(p.hys[scn.cluster] == c.mis_hys_db)
    assert [tag for _, tag in scn.oam_log] == ["push"]
    rollback(env, [("ttt", c0)])
    assert p.ttt[c0] == c.mis_ttt_ms
    assert env.stats["rollbacks"] == 1


def test_tape_is_arbiter_independent_through_the_rollout() -> None:
    c = cfg(scenario="mistune", mix="M4", warmup_s=10.0, scored_s=1800.0)
    a, b = E6Env(c, log=False), E6Env(c, log=False)
    sa, sb = mistune(a.plant), mistune(b.plant)
    t = int(np.ceil(sa.t_mis)) + 5
    run_to(a, t)
    run_to(b, t, freeze)
    assert a.stats["changes"] > 0
    assert b.stats["changes"] == 0
    assert sa.push_t is not None
    assert sa.push_t == sb.push_t
    np.testing.assert_array_equal(a.plant.pos, b.plant.pos)
    np.testing.assert_array_equal(sa.s, sb.s)
    np.testing.assert_array_equal(sa.cluster, sb.cluster)
    assert np.all(b.plant.hys[sb.cluster] == c.mis_hys_db)
    assert np.all(b.plant.ttt[sb.cluster] == c.mis_ttt_ms)


def mob_report(p: Plant, t0: float, tl: float, att: float) -> dict[str, Any]:
    nc = p.nc
    rep: dict[str, Any] = {"gran": "mob", "t0": t0, "t1": t0 + 30}
    for k in MOB_KEYS:
        rep[k] = np.zeros((nc, nc))
    c0 = int(mistune(p).cluster[0])
    c1 = int(p.lay.neighbours[c0][0])
    rep["too_late"][c0, c1], rep["ho_att"][c0, c1] = tl, att
    rep["rlf"] = np.zeros(nc)
    return rep


def test_smo_alarm_rule() -> None:
    env = E6Env(cfg(scenario="mistune", warmup_s=2.0, scored_s=100.0), log=False)
    p = env.plant
    scn = mistune(p)
    bad, good = (5, 100), (1, 100)
    smo = SMORestore(env)
    assert not smo.alarm(mob_report(p, 0, *bad), scn)
    assert smo.alarm(mob_report(p, 30, *bad), scn)
    smo = SMORestore(env)
    fired = [smo.alarm(mob_report(p, 0, *r), scn) for r in (bad, good, bad)]
    assert fired == [False, False, False]
    smo = SMORestore(env)
    fired = [smo.alarm(mob_report(p, 0, 2, 200), scn) for _ in range(3)]
    assert fired == [False, False, False]


def test_smo_restore_fires_restores_and_is_not_rollbackable() -> None:
    c = cfg(scenario="mistune", warmup_s=10.0, scored_s=600.0)
    env = E6Env(c, log=False)
    p = env.plant
    scn = mistune(p)
    smo = SMORestore(env)
    run_to(env, int(np.ceil(scn.t_mis)) + 1, smo)
    assert smo.fired_t is None
    assert np.all(p.hys[scn.cluster] == c.mis_hys_db)
    assert scn.pre_hys is not None
    assert scn.pre_ttt is not None
    pre_h, pre_t = scn.pre_hys.copy(), scn.pre_ttt.copy()
    obs = env.step_propose()
    assert scn.push_t is not None
    t0 = float(np.ceil(scn.push_t))
    obs["new_reports"] = [mob_report(p, t0, 5, 100), mob_report(p, t0 + 30, 5, 100)]
    env.step_apply(smo(obs))
    assert smo.fired_t == float(env.sec)
    assert [tag for _, tag in scn.oam_log] == ["push", "restore"]
    np.testing.assert_array_equal(p.hys[scn.cluster], pre_h)
    np.testing.assert_array_equal(p.ttt[scn.cluster], pre_t)
    c0 = int(scn.cluster[0])
    rollback(env, [("hys", c0), ("ttt", c0)])
    assert p.hys[c0] == pre_h[0]
    assert p.ttt[c0] == pre_t[0]
    run_to(env, env.sec + 5, smo)
    assert [tag for _, tag in scn.oam_log] == ["push", "restore"]


def test_env_copy_rebinds_the_oam_hook() -> None:
    env = E6Env(cfg(scenario="mistune", warmup_s=2.0, scored_s=100.0), log=False)
    c = env.copy()
    assert getattr(c.plant.oam_hook, "__self__", None) is c
    assert getattr(env.plant.oam_hook, "__self__", None) is env

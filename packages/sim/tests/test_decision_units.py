from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system import config as C
from oran_causal.sim.system.decision import units as U
from oran_causal.sim.system.env import E6Env
from oran_causal.sim.system.xapps_p import ESPico

DATA = Path(__file__).parent / "data"
STATIC = {
    "cells": 3,
    "neighbours": [[1], [0], [1]],
    "is_macro": np.array([True, True, False]),
    "knobs": [
        ("carrier", 0),
        ("carrier", 1),
        ("sleep", 2),
        ("prot_min", 0),
        ("prot_min", 1),
        ("ptx", 0),
    ],
    "xapps": {"ES": [], "SliceGuarantee": [], "PowerES": []},
}


def req(x: str, k: tuple, cur: float, prop: float) -> dict[str, Any]:
    return {"xapp": x, "ver": 1, "knob": k, "cur": cur, "prop": prop, "t": 0.0}


def obs(t: float, reqs: list[dict], config: dict) -> dict[str, Any]:
    return {
        "t": float(t),
        "requests": reqs,
        "config": dict(config),
        "new_reports": [],
        "static": STATIC,
        "locked": {},
        "changes": 0,
        "churn_cap": None,
    }


class Const:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[tuple] = []

    def __call__(self, unit: dict) -> tuple[str, float]:
        self.calls.append((unit["c"], unit["x"], unit["t0"]))
        return self.mode, 0.5


def cfg_p(pair: str, seed: int = 5, warm: float = 60.0, scored: float = 90.0) -> C.E6Config:
    px = {"P1": ("SliceGuarantee",), "P3": ("PowerES", "SliceGuarantee")}[pair]
    return C.E6Config(
        seed=seed,
        mix="ES",
        scenario="base",
        load_factor=0.79,
        n_ue=300,
        n_pico=3,
        mobility="ped",
        warmup_s=warm,
        scored_s=scored,
        e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=px),
    )


@pytest.mark.claim("DEF-SYS-UNIT")
def test_unit_opens_at_first_request_and_mode_holds_for_T() -> None:
    pol = Const("reject")
    arb = U.UnitArbiter(pol, T=60.0)
    cfg = {("carrier", 0): 2.0, ("prot_min", 0): 0.0}
    reqs = [req("ES", ("carrier", 0), 2.0, 1.0), req("SliceGuarantee", ("prot_min", 0), 0.0, 0.05)]
    assert arb(obs(100, reqs, cfg))["decisions"] == ["reject", "reject"]
    assert len(pol.calls) == 2
    for t in (110, 159):
        d = arb(obs(t, [req("ES", ("carrier", 0), 2.0, 1.0)], cfg))
        assert d["decisions"] == ["reject"]
    assert len(pol.calls) == 2
    arb(obs(160, [req("ES", ("carrier", 0), 2.0, 1.0)], cfg))
    assert pol.calls[-1] == (0, "ES", 160.0)
    assert len(pol.calls) == 3
    u = arb.units[0]
    assert (u["c"], u["x"], u["t0"], u["mode"], u["p"], u["n_req"]) == (
        0,
        "ES",
        100.0,
        "reject",
        0.5,
        3,
    )
    assert u["x_idx"] == U.XAPPS_P.index("ES")
    assert u["exp"] == [0, 1]
    assert U.T_UNIT == 60.0


@pytest.mark.claim("DEF-SYS-UNIT")
def test_hold_policy_holds_its_mode_inside_the_window_only() -> None:
    pol = U.HoldPolicy({(3, "ES")}, "reject", 100.0, 101.0)
    assert pol({"c": 3, "x": "ES", "t0": 100.0}) == ("reject", 1.0)
    assert pol({"c": 3, "x": "ES", "t0": 101.0}) == ("accept", 1.0)
    assert pol({"c": 4, "x": "ES", "t0": 100.0}) == ("accept", 1.0)
    assert U.accept_all_policy({}) == ("accept", 1.0)


def test_no_units_during_warmup_everything_accepted() -> None:
    pol = Const("reject")
    arb = U.UnitArbiter(pol, warmup_s=120.0)
    d = arb(obs(119, [req("ES", ("carrier", 0), 2.0, 1.0)], {("carrier", 0): 2.0}))
    assert d["decisions"] == ["accept"]
    assert not pol.calls
    assert not arb.units
    d = arb(obs(120, [req("ES", ("carrier", 0), 2.0, 1.0)], {("carrier", 0): 2.0}))
    assert d["decisions"] == ["reject"]


def test_half_uses_the_slew_rule_with_e6p_quanta() -> None:
    arb = U.UnitArbiter(Const("half"))
    cfg = {("ptx", 0): 0.0, ("prot_min", 0): 0.0, ("carrier", 0): 2.0}
    d = arb(obs(10, [req("PowerES", ("ptx", 0), 0.0, -3.0)], cfg))
    assert d["decisions"] == [("modify", -2.0)]
    d1 = arb(obs(11, [req("SliceGuarantee", ("prot_min", 0), 0.0, 0.05)], cfg))["decisions"]
    d2 = arb(obs(16, [req("SliceGuarantee", ("prot_min", 0), 0.0, 0.05)], cfg))["decisions"]
    assert sorted([d1[0], d2[0]]) == ["accept", "reject"]


def test_rollback_flag_rolls_back_a_recent_change_once_and_never_our_own() -> None:
    arb = U.UnitArbiter(Const("accept+rb"), T=10.0)
    k = ("prot_min", 0)
    arb(obs(99, [], {k: 0.0}))
    arb(obs(100, [], {k: 0.05}))
    d = arb(obs(120, [req("SliceGuarantee", k, 0.05, 0.10)], {k: 0.05}))
    assert d["rollback"] == [k]
    assert d["decisions"] == ["accept"]
    arb(obs(121, [], {k: 0.0}))
    d = arb(obs(131, [req("SliceGuarantee", k, 0.0, 0.05)], {k: 0.0}))
    assert d["rollback"] == []
    arb2 = U.UnitArbiter(Const("accept+rb"))
    arb2(obs(1, [], {k: 0.0}))
    arb2(obs(2, [], {k: 0.05}))
    assert arb2(obs(200, [req("SliceGuarantee", k, 0.05, 0.1)], {k: 0.05}))["rollback"] == []
    with pytest.raises(ValueError, match="rollback flag"):
        U.UnitArbiter(Const("accept+rb"))(
            obs(5, [req("ES", ("carrier", 0), 2.0, 1.0)], {("carrier", 0): 2.0})
        )


def test_feasible_open_rule_skips_dwell_blocked_requests() -> None:
    pol = Const("reject")
    arb = U.UnitArbiter(pol, open_rule="feasible")
    k = ("carrier", 0)
    arb(obs(49, [], {k: 2.0}))
    arb(obs(50, [], {k: 1.0}))
    assert arb(obs(100, [req("ES", k, 1.0, 2.0)], {k: 1.0}))["decisions"] == ["accept"]
    assert not pol.calls
    assert arb(obs(169, [req("ES", k, 1.0, 2.0)], {k: 1.0}))["decisions"] == ["reject"]
    assert len(pol.calls) == 1


def test_exposure_set_is_symmetrised_and_contains_every_pico_candidate() -> None:
    assert U.exposure_sets({"cells": 3, "neighbours": [[1], [], [0]]}) == {
        0: (0, 1, 2),
        1: (0, 1),
        2: (0, 2),
    }
    env = E6Env(cfg_p("P3", scored=10.0, warm=0.0), log=False, wg3=True)
    ex = U.exposure_sets(env.static())
    es = next(x for x in env.xapps if x.name == "ES")
    assert isinstance(es, ESPico)
    for pc, m in es.cand.items():
        assert m in ex[pc]
        assert pc in ex[m]
    for c, s in ex.items():
        assert c in s
        assert all(c in ex[v] for v in s)


def test_split_mode_and_lock_equivalence_is_declared_for_p1_p3() -> None:
    assert U.split_mode("half+rb") == ("half", True)
    assert U.split_mode("reject") == ("reject", False)
    with pytest.raises(ValueError, match="unknown mode"):
        U.split_mode("maybe")
    assert U.LOCK_EQUIV == {"P1": "reject", "P3": "reject"}
    assert U.knob_region([0, 0, 1], ("sleep", 2)) == 1


def _run(cfg: C.E6Config, mode_of: Callable[[dict], str]) -> tuple[E6Env, dict]:
    env = E6Env(cfg, log=False, wg3=True)
    arb = U.UnitArbiter(lambda u: (mode_of(u), 1.0), T=60.0, warmup_s=cfg.warmup_s)
    writers: dict = {}
    while env.sec < env.total_s:
        o = env.step_propose()
        for r in o["requests"]:
            writers.setdefault(r["knob"], set()).add(r["xapp"])
        env.step_apply(arb(o))
    return env, writers


@pytest.mark.slow
@pytest.mark.parametrize("pair", ["P1", "P3"])
def test_lock_is_plant_identical_to_reject_in_p1_p3(pair: str) -> None:
    cfg = cfg_p(pair)

    def mode(which: str) -> Callable[[dict], str]:
        return lambda u: which if (u["c"] + int(u["t0"])) % 2 == 0 else "accept"

    e_lock, writers = _run(cfg, mode("lock"))
    e_rej, _ = _run(cfg, mode("reject"))
    assert all(len(v) == 1 for v in writers.values())
    assert e_lock.stats["locks"] > 0
    assert e_rej.stats["locks"] == 0
    assert e_lock.stats["changes"] > 0
    assert e_lock.plant.sla == e_rej.plant.sla
    assert np.array_equal(e_lock.plant.q, e_rej.plant.q)
    assert e_lock.config() == e_rej.config()
    assert e_lock.stats["changes"] == e_rej.stats["changes"]

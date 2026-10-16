from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system import config as C
from oran_causal.sim.system.baselines import freeze, priority
from oran_causal.sim.system.env import Arbiter, Decision, E6Env, Obs
from oran_causal.sim.system.ric import knob_get


def _env(**kw: Any) -> E6Env:
    cfg = C.E6Config(seed=11, load="medium", mobility="mixed", mix="M4", warmup_s=10, scored_s=20)
    return E6Env(cfg, log=False, **kw)


def _accept_all(obs: Obs) -> Decision:
    return {"decisions": ["accept"] * len(obs["requests"]), "writes": []}


def _advance_to_requests(env: E6Env, arbiter: Arbiter = _accept_all, max_s: int = 60) -> Obs:
    for _ in range(max_s):
        obs = env.step_propose()
        if obs["requests"]:
            return obs
        env.step_apply(arbiter(obs))
    raise AssertionError("no requests proposed")


def _same_sla(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert a.keys() == b.keys()
    for k, v in a.items():
        assert np.array_equal(np.asarray(v), np.asarray(b[k])), k


def _mixed_decision(i: int) -> Any:
    if i == 0:
        return ("lock", 7.0)
    return "accept" if i % 2 else "defer"


def test_copy_between_phases_replays_pending_requests() -> None:
    env = _env()
    for _ in range(5):
        env.step(priority)
    obs = _advance_to_requests(env, priority)
    n = len(obs["requests"])
    dec = {
        "decisions": [_mixed_decision(i) for i in range(n)],
        "rollback": [obs["requests"][0]["knob"]],
    }
    c = env.copy()
    env.step_apply(dec)
    c.step_apply(dec)
    for _ in range(15):
        env.step(priority)
        c.step(priority)
    _same_sla(env.plant.sla, c.plant.sla)
    assert env.stats == c.stats
    assert env.config() == c.config()


def test_lock_blocks_knob_for_duration_then_releases() -> None:
    env = _env()
    obs = _advance_to_requests(env)
    k, secs = obs["requests"][0]["knob"], 25.0
    t_lock, v0 = obs["t"], knob_get(env.plant, k)
    env.step_apply({"decisions": [("lock", secs)] + ["accept"] * (len(obs["requests"]) - 1)})
    assert env.stats["locks"] == 1
    seen_locked, released = 0, False
    for _ in range(int(secs) + 20):
        obs = env.step_propose()
        on_k = sum(r["knob"] == k for r in obs["requests"])
        before = env.stats["lock_blocked"]
        env.step_apply(_accept_all(obs))
        if obs["t"] < t_lock + secs:
            assert k in obs["locked"]
            assert knob_get(env.plant, k) == v0
            assert env.stats["lock_blocked"] - before == on_k
            seen_locked += on_k
        else:
            assert k not in obs["locked"]
            assert env.stats["lock_blocked"] == before
            released = True
    assert seen_locked > 0
    assert released


def test_rollback_restores_pre_change_value() -> None:
    env = _env()
    changed: list[Any] = []
    before: dict[Any, float] = {}
    for _ in range(60):
        obs = env.step_propose()
        before = env.config()
        env.step_apply(_accept_all(obs))
        changed = [k for k, v in env.config().items() if v != before[k]]
        if changed:
            break
    k = changed[0]
    v_prev, v_new = before[k], knob_get(env.plant, k)
    never = next(kk for kk in env.knobs if kk not in env.prev_val)
    for _ in range(130):
        env.step(freeze)
    assert knob_get(env.plant, k) == v_new
    obs = env.step_propose()
    cfg0 = env.config()
    env.step_apply({"decisions": ["reject"] * len(obs["requests"]), "rollback": [k, never]})
    assert knob_get(env.plant, k) == v_prev
    assert env.config()[never] == cfg0[never]
    assert env.stats["rollbacks"] == 1


def test_churn_cap_blocks_changes_beyond_cap() -> None:
    free = _env()
    free.run()
    assert free.stats["changes"] > 5
    assert free.stats["churn_blocked"] == 0
    env = _env(churn_cap=5)
    s = env.run()
    assert s["changes"] == 5
    assert s["churn_blocked"] > 0
    assert s["rej"] >= s["churn_blocked"]


def test_wg3_rejects_free_writes_but_allows_actions() -> None:
    env = _env(wg3=True)
    obs = env.step_propose()
    k = env.knobs[0]
    with pytest.raises(ValueError, match="wg3"):
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [(k, 3.0)]})
    env2 = _env(wg3=True)
    obs = env2.step_propose()
    env2.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": [k]})
    loose = _env()
    obs = loose.step_propose()
    loose.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [(k, 3.0)]})
    assert loose.stats["writes"] == 1


def test_phase_order_enforced() -> None:
    env = _env()
    with pytest.raises(RuntimeError, match=r"\w"):
        env.step_apply({"decisions": []})
    env.step_propose()
    with pytest.raises(RuntimeError, match=r"\w"):
        env.step_propose()


def test_pending_requests_are_the_offered_requests() -> None:
    env = _env()
    assert env.pending_requests == []
    obs = _advance_to_requests(env)
    assert env.pending_requests == obs["requests"]
    env.step_apply(_accept_all(obs))
    assert env.pending_requests == []

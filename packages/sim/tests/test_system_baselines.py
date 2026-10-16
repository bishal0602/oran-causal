from typing import Any

import pytest

from oran_causal.sim.system import config as C
from oran_causal.sim.system.baselines import (
    STATIC_DEFAULT,
    apply_static,
    freeze,
    static_policy,
    subset,
    tuned_static,
    tuned_static_grid,
)
from oran_causal.sim.system.env import Arbiter, E6Env
from oran_causal.sim.system.ric import LIMITS, knob_get

SLA = (
    "svr",
    "ll_viol",
    "embb_viol",
    "outage_viol",
    "severe",
    "energy_kwh",
    "rlf_per_ue_h",
    "ho_per_ue_h",
    "pingpong",
)


def cfg(**kw: Any) -> C.E6Config:
    base: dict[str, Any] = {
        "seed": 3,
        "load": "high",
        "mobility": "mixed",
        "mix": "M4",
        "warmup_s": 10,
        "scored_s": 40,
    }
    return C.E6Config(**{**base, **kw})


def run(
    arb: Arbiter | None, spec: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any], E6Env]:
    env = E6Env(cfg(), log=False)
    if spec is not None:
        arb = static_policy(env, spec)
    s = env.run(arb)
    return {k: s[k] for k in SLA}, s, env


def test_subset_all_equals_no_arbiter() -> None:
    a, sa, _ = run(None)
    b, sb, _ = run(subset(("MRO", "TS", "ES", "SLICE")))
    assert sa["acc"] > 0
    assert sb["rej"] == 0
    assert a == b


def test_empty_subset_equals_freeze() -> None:
    a, _, ea = run(freeze)
    b, _, eb = run(subset(()))
    assert a == b
    assert ea.config() == eb.config()


def test_subset_only_changes_kept_xapps_knobs() -> None:
    _, s, env = run(subset(("SLICE",)))
    init = E6Env(cfg(), log=False).config()
    changed = {k[0] for k, v in env.config().items() if v != init[k]}
    assert s["rej"] > 0
    assert changed <= {"ll_ratio"}


def test_apply_static_sets_knobs_within_limits() -> None:
    env = E6Env(cfg(), log=False)
    spec = {"cio_to_pico": 3.0, "hys": 1.0, "ttt": 160, "ll_ratio": 0.1, "carriers": "one"}
    set_ = apply_static(env, spec)
    p, lay = env.plant, env.plant.lay
    assert set_
    for k, v in set_:
        lo, hi = LIMITS[k[0]][:2]
        assert lo <= knob_get(p, k) == v <= hi
    macro = [c for c in range(lay.n_cells) if lay.is_macro[c]]
    pico = [c for c in range(lay.n_cells) if not lay.is_macro[c]]
    pairs = [(m, q) for m in macro for q in lay.neighbours[m] if q in pico]
    assert pairs
    assert all(p.cio[m, q] == 3.0 and p.cio[q, m] == -3.0 for m, q in pairs)
    assert all(p.cio[a, b] == 0.0 for a in macro for b in lay.neighbours[a] if b in macro)
    assert (p.hys == 1.0).all()
    assert (p.ttt == 160).all()
    assert (p.ll_ratio == 0.1).all()
    assert (p.n_car[macro] == 1).all()
    assert (p.n_car[pico] == 1).all()


def test_static_policy_holds_config_and_default_equals_freeze() -> None:
    spec = {"cio_to_pico": 6.0, "ttt": 640}
    env = E6Env(cfg(), log=False)
    static_policy(env, spec)
    before = env.config()
    _, _, env2 = run(None, spec=spec)
    assert env2.config() == before
    assert run(freeze)[0] == run(None, spec=STATIC_DEFAULT)[0]


@pytest.mark.parametrize(
    "spec",
    [
        {"hys": 9.0},
        {"ttt": 300},
        {"cio_to_pico": 2.5},
        {"ll_ratio": 0.9},
        {"carriers": 5},
        {"tilt": 1.0},
    ],
)
def test_apply_static_rejects_invalid(spec: dict[str, Any]) -> None:
    env = E6Env(cfg(), log=False)
    with pytest.raises((ValueError, KeyError)):
        apply_static(env, spec)


def test_tuned_grid_inside_limits_and_starts_at_default() -> None:
    g = tuned_static_grid()
    assert {k: v[0] for k, v in g.items()} == STATIC_DEFAULT
    env = E6Env(cfg(), log=False)
    for k, vals in g.items():
        for v in vals:
            apply_static(env, {k: v})


def test_tuned_static_coordinate_descent_finds_separable_minimum() -> None:
    g = tuned_static_grid()
    target = {"cio_to_pico": 3.0, "hys": 4.0, "ttt": 160, "ll_ratio": 0.1, "carriers": "all"}
    calls = []

    def f(spec: dict[str, Any]) -> float:
        calls.append(spec)
        return sum(0 if spec[k] == target[k] else 1 + g[k].index(spec[k]) for k in g)

    res = tuned_static(f)
    assert res["best"] == target
    assert res["best_val"] == 0
    assert res["n_eval"] == len(calls) == len(res["path"])
    assert res["path"][0]["spec"] == STATIC_DEFAULT

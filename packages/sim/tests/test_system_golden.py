import copy
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system import baselines as B
from oran_causal.sim.system import config as C
from oran_causal.sim.system.env import Arbiter, Decision, E6Env, Obs

DATA = Path(__file__).parent / "data"
SHORT = json.loads((DATA / "system_short_episodes.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((DATA / "system_golden_records.json").read_text(encoding="utf-8"))
LOAD_FACTOR = GOLDEN["load_factor"]
PLANT_ATTRS = (
    "pos",
    "q",
    "serv",
    "sinr_ewma",
    "l3",
    "rho",
    "occ",
    "m",
    "ptx_off",
    "prot_min",
    "hys",
    "ttt",
    "cio",
    "ll_ratio",
    "asleep",
    "waking_until",
    "n_car",
    "car_on_at",
    "ttt_cand",
    "ttt_acc",
    "int_until",
    "t310",
    "last_ho_t",
    "last_ho_src",
    "heading",
    "ll_arr",
    "ll_cum_a",
    "ll_cum_s",
    "cell_viol_run",
)
SLA_KEYS = (
    "viol_ue_s",
    "ue_s",
    "ll_viol",
    "embb_viol",
    "outage_viol",
    "severe",
    "energy_j",
    "rlf",
    "ho",
    "pingpong",
    "prot_viol",
    "prot_ue_s",
    "lowsinr_ue_s",
)


def _upd(h: Any, x: Any) -> None:
    if isinstance(x, np.ndarray):
        h.update(str((x.dtype.str, x.shape)).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, dict):
        for k in sorted(x, key=repr):
            h.update(repr(k).encode())
            _upd(h, x[k])
    elif isinstance(x, list | tuple):
        h.update(f"[{len(x)}".encode())
        for v in x:
            _upd(h, v)
    elif isinstance(x, float | np.floating):
        h.update(float(x).hex().encode())
    else:
        h.update(repr(x).encode())


def _hash(x: Any) -> str:
    h = hashlib.sha256()
    _upd(h, x)
    return h.hexdigest()


def _digest(env: E6Env, obs: Obs | None, dec: Decision | None) -> str:
    p = env.plant
    parts: list[Any] = [getattr(p, a) for a in PLANT_ATTRS]
    parts += [p.sla, env.stats, env.config(), env.prev_val, env.last_change, env.lock_until]
    parts.append([d["age"] for d in env.deferred])
    if obs is not None:
        parts += [obs["new_reports"], obs["requests"]]
    parts.append(dec)
    parts += [
        (x.name, x.scale, x.behaviour, x.next_t, x.ver, x.pending_target, x.hold_until)
        for x in env.xapps
    ]
    if p.scn is not None:
        parts.append(getattr(p.scn, "oam_log", None))
    h = hashlib.sha256()
    for x in parts:
        _upd(h, x)
    return h.hexdigest()


class Mixed:
    def __init__(self, writes: bool) -> None:
        self.writes = writes

    def _decision(self, t: int, i: int, r: dict[str, Any]) -> Any:
        j = (t + i) % 7
        if j in {0, 1, 2}:
            return "accept"
        if j == 3:
            return "reject"
        if j == 4:
            return ("modify", r["cur"] + 0.6 * (r["prop"] - r["cur"]))
        if j == 5:
            return "defer"
        return ("lock", 5.0 + t % 11) if t % 3 else "accept"

    def __call__(self, obs: Obs) -> Decision:
        t = int(obs["t"])
        reqs = obs["requests"]
        out: Decision = {
            "decisions": [self._decision(t, i, r) for i, r in enumerate(reqs)],
            "writes": [],
        }
        knobs = obs["static"]["knobs"]
        if t % 13 == 0 and reqs:
            out["rollback"] = [reqs[0]["knob"], knobs[t % len(knobs)]]
        if self.writes and t % 17 == 0:
            k = knobs[(3 * t) % len(knobs)]
            out["writes"] = [(k, obs["config"][k] + 1.0)]
        return out


def golden_cfg(seed: int, arm: str, warmup_s: float = 120.0, scored_s: float = 600.0) -> C.E6Config:
    xapps = () if arm == "freeze" else ("PowerES", "SliceGuarantee")
    return C.E6Config(
        seed=seed,
        mix="none" if arm == "freeze" else "ES",
        scenario="surge",
        load_factor=LOAD_FACTOR,
        n_ue=300,
        n_pico=3,
        mobility="ped",
        kpm="nominal",
        update=False,
        warmup_s=warmup_s,
        scored_s=scored_s,
        e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=xapps),
    )


def golden_arbiter(arm: str) -> Arbiter | None:
    if arm == "noarb":
        return None
    if arm == "freeze":
        return B.freeze
    return B.subset(arm.removeprefix("sub:").split("+"))


Case = tuple[C.E6Config, dict[str, Any], Callable[[E6Env], Arbiter | None], int]


def short_case(name: str) -> Case:
    if name == "golden_short_noarb":
        return golden_cfg(191100, "noarb", 20.0, 60.0), {"wg3": True}, lambda env: None, 80
    if name == "golden_short_sub":
        sub = golden_arbiter("sub:ES+PowerES")
        return golden_cfg(191101, "sub", 20.0, 60.0), {"wg3": True}, lambda env: sub, 80
    p = C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "Coverage", "SliceGuarantee"))
    cfg = C.E6Config(
        seed=9,
        mix="M4",
        scenario="mistune",
        warmup_s=10,
        scored_s=110,
        mis_onset_frac=0.1,
        e6p=p,
    )
    return cfg, {"trace": True, "trace_snapshot_s": 7}, lambda env: Mixed(True), 120


def _accept_all(obs: Obs) -> Decision:
    return {"decisions": ["accept"] * len(obs["requests"]), "writes": []}


def _fork_digest(env: E6Env, arb: Arbiter | None, dec: Decision) -> str:
    fork = env.copy(reseed=3)
    fork.step_apply(dec)
    farb = None if arb is None or hasattr(arb, "env") else copy.deepcopy(arb)
    for _ in range(5):
        fork.step(farb)
    return "reseed:" + _digest(fork, None, None)


def run_short(name: str) -> dict[str, Any]:
    cfg, kw, mk, secs = short_case(name)
    env = E6Env(cfg, log=True, **kw)
    arb = mk(env)
    digs = []
    for s in range(secs):
        obs = env.step_propose()
        dec = _accept_all(obs) if arb is None else arb(obs)
        if s == secs // 2:
            digs.append(_fork_digest(env, arb, dec))
        env.step_apply(dec)
        digs.append(_digest(env, obs, dec))
    score = {k: (v if isinstance(v, int) else float(v)) for k, v in env.score().items()}
    out = {"digests": digs, "score": score, "log": _hash(env.log)}
    if kw.get("trace"):
        tr = env.get_trace()
        out["trace"] = _hash(tr.arrays)
        out["trace_meta"] = json.dumps(tr.meta, sort_keys=True)
    return out


def _short_digest(d: str) -> str:
    head, sep, tail = d.partition(":")
    n = SHORT["digest_hex"]
    return f"{head}:{tail[:n]}" if sep else d[:n]


@pytest.mark.parametrize("name", ["golden_short_noarb", "golden_short_sub", "e6p_mixed_trace"])
def test_short_episode_is_bit_identical_to_the_old_plant(name: str) -> None:
    ref = SHORT["platforms"].get(sys.platform, {}).get(name)
    if ref is None:
        pytest.skip(f"no reference digests for platform {sys.platform}")
    got = run_short(name)
    digs = [_short_digest(d) for d in got["digests"]]
    first = next(
        (i for i, (a, b) in enumerate(zip(digs, ref["digests"], strict=True)) if a != b), None
    )
    assert first is None, f"{name}: first differing step {first}"
    assert got["score"] == ref["score"]
    assert got["log"] == ref["log"]
    if "trace" in ref:
        assert got["trace"] == ref["trace"]
        assert got["trace_meta"] == ref["trace_meta"]


@pytest.mark.claim("DEF-SIM-CONST")
def test_golden_episode_constants() -> None:
    cfg = golden_cfg(191100, "noarb")
    env = E6Env(cfg, log=False, wg3=True)
    assert env.plant.nc == 24
    assert env.plant.n == 300
    assert pytest.approx(0.04) == C.TICK_S
    assert (cfg.warmup_s, cfg.scored_s) == (120.0, 600.0)
    assert env.total_s == 720


def outcome(env: E6Env) -> dict[str, Any]:
    sla = env.plant.sla
    return {**{k: float(sla[k]) for k in SLA_KEYS}, **{"st_" + k: v for k, v in env.stats.items()}}


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(GOLDEN["episodes"]))
def test_golden_episode_is_bit_identical_to_the_stored_record(key: str) -> None:
    seed, arm = key.split("|")
    env = E6Env(golden_cfg(int(seed), arm), log=False, wg3=True)
    arb = golden_arbiter(arm)
    while env.sec < env.total_s:
        env.step(arb)
    assert outcome(env) == GOLDEN["episodes"][key]["outcome"]

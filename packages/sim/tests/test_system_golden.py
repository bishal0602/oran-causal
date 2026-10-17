import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from oran_causal.sim.system import baselines as B
from oran_causal.sim.system import config as C
from oran_causal.sim.system.env import Arbiter, Decision, E6Env, Obs

DATA = Path(__file__).parent / "data"
SHORT = json.loads((DATA / "system_short_episodes.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((DATA / "system_golden_records.json").read_text(encoding="utf-8"))
LOAD_FACTOR = GOLDEN["load_factor"]
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


def run_short(name: str) -> dict[str, float]:
    cfg, kw, mk, secs = short_case(name)
    env = E6Env(cfg, log=True, **kw)
    arb = mk(env)
    for _ in range(secs):
        obs = env.step_propose()
        env.step_apply(_accept_all(obs) if arb is None else arb(obs))
    return {k: (v if isinstance(v, int) else float(v)) for k, v in env.score().items()}


ENERGY_RTOL = 1e-3
COUNT_RTOL = 0.03
COUNT_ATOL = 2
EPISODE_RTOL = 0.05


@pytest.mark.parametrize("name", ["golden_short_noarb", "golden_short_sub", "e6p_mixed_trace"])
def test_short_episode_scores_match_the_old_plant(name: str) -> None:
    got, want = run_short(name), SHORT["episodes"][name]
    assert got.keys() == want.keys()
    for k, w in want.items():
        rtol, atol = (ENERGY_RTOL, 0.0) if k == "energy_kwh" else (COUNT_RTOL, COUNT_ATOL)
        assert got[k] == pytest.approx(w, rel=rtol, abs=atol), k


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
def test_golden_episode_matches_the_stored_record(key: str) -> None:
    seed, arm = key.split("|")
    env = E6Env(golden_cfg(int(seed), arm), log=False, wg3=True)
    arb = golden_arbiter(arm)
    while env.sec < env.total_s:
        env.step(arb)
    got, want = outcome(env), GOLDEN["episodes"][key]["outcome"]
    assert got.keys() == want.keys()
    for k, w in want.items():
        rtol, atol = (ENERGY_RTOL, 0.0) if k == "energy_j" else (EPISODE_RTOL, COUNT_ATOL)
        assert got[k] == pytest.approx(w, rel=rtol, abs=atol), k

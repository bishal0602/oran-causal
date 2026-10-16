from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system.config import E6Config
from oran_causal.sim.system.env import Decision, E6Env, Obs
from oran_causal.sim.system.trace import OUT_CODES, RB_CODES, WR_CODES, Trace, region_labels

OC = {c: i for i, c in enumerate(OUT_CODES)}


def _cfg(seed: int = 16) -> E6Config:
    return E6Config(seed=seed, load="medium", mobility="mixed", mix="M4", warmup_s=30, scored_s=120)


class Exerciser:
    def __call__(self, obs: Obs) -> Decision:
        t = int(obs["t"])
        reqs = obs["requests"]
        dec: list[Any] = []
        for i, r in enumerate(reqs):
            j = (t + i) % 6
            if j == 0:
                dec.append("reject")
            elif j == 1:
                dec.append(("modify", r["cur"] + 0.5 * (r["prop"] - r["cur"])))
            elif j == 2:
                dec.append("defer")
            elif j == 3:
                dec.append(("lock", 4.0))
            else:
                dec.append("accept")
        out: Decision = {"decisions": dec, "writes": []}
        knobs = obs["static"]["knobs"]
        if t % 7 == 0 and reqs:
            out["rollback"] = [reqs[0]["knob"]]
        if t % 11 == 0:
            k = knobs[t % len(knobs)]
            out["writes"] = [(k, obs["config"][k] + 1.0)]
        return out


def _exercised(trace: bool) -> E6Env:
    env = E6Env(_cfg(), log=False, churn_cap=60, trace=trace)
    env.run(Exerciser())
    return env


@pytest.fixture(scope="module")
def noarb() -> tuple[E6Env, dict[str, Any], Trace]:
    env = E6Env(_cfg(), log=True, trace=True)
    s = env.run()
    return env, s, env.get_trace()


@pytest.fixture(scope="module")
def exercised() -> tuple[E6Env, Trace]:
    env = _exercised(trace=True)
    return env, env.get_trace()


def test_trace_off_changes_nothing(
    noarb: tuple[E6Env, dict[str, Any], Trace], exercised: tuple[E6Env, Trace]
) -> None:
    _, s_on, _ = noarb
    off = E6Env(_cfg(), log=True)
    assert off.run() == s_on
    assert _exercised(trace=False).score() == exercised[0].score()
    with pytest.raises(RuntimeError, match="trace=False"):
        off.get_trace()


def test_copy_is_untraced() -> None:
    env = E6Env(_cfg(), log=False, trace=True)
    env.step_propose()
    c = env.copy()
    with pytest.raises(RuntimeError, match="trace=False"):
        c.get_trace()
    env.step_apply({"decisions": ["accept"] * len(env.pending_requests), "writes": []})
    c.step_apply({"decisions": ["accept"] * len(c.pending_requests), "writes": []})
    assert len(env.get_trace()["t"]) == 1


def test_one_row_per_second_and_config_rebuild(
    noarb: tuple[E6Env, dict[str, Any], Trace],
) -> None:
    env, _, tr = noarb
    a = tr.arrays
    assert np.array_equal(a["t"], np.arange(1, env.total_s + 1))
    assert a["scored"].sum() == env.cfg.scored_s
    for k, v in tr.labels().items():
        assert len(v) == len(a["t"]), k
    assert not set(tr.features()) & set(tr.labels())
    for t in (5, 31, 60, 61, 100, env.total_s - 1):
        logged = np.array(list(env.log[t - 1]["config"].values()))
        assert np.allclose(tr.config_at(t + 1), logged)
    with pytest.raises(ValueError, match="snapshot"):
        tr.config_at(0)


def test_labels_sum_to_plant_counters(noarb: tuple[E6Env, dict[str, Any], Trace]) -> None:
    env, _, tr = noarb
    a, sla = tr.arrays, env.plant.sla
    sc = a["scored"]
    assert a["lab_viol"][sc].sum() == sla["viol_ue_s"]
    assert a["lab_viol"][sc][..., 0].sum() == sla["ll_viol"]
    assert a["lab_viol"][sc][..., 1].sum() == sla["embb_viol"]
    assert a["lab_outage"][sc].sum() == sla["outage_viol"]
    assert a["lab_ue"][sc].sum() == sla["ue_s"]
    assert a["lab_severe"].sum() == sla["severe"]
    assert a["lab_rlf"].sum() == sla["rlf"]
    assert np.isclose(a["lab_energy_j"][sc].astype(float).sum(), sla["energy_j"], rtol=1e-5)
    y = region_labels(tr, env.plant.lay.cell_site, env.cfg.warmup_s, env.total_s)
    assert len(y["region_ids"]) == 10
    assert y["viol"].sum() == sla["viol_ue_s"]
    assert np.isclose(y["energy_j"].sum(), sla["energy_j"], rtol=1e-5)


def test_ack_nack_and_churn_match_stats(exercised: tuple[E6Env, Trace]) -> None:
    _, tr = exercised
    a, m = tr.arrays, tr.meta
    st = dict(zip(m["stats"], a["stats"][-1], strict=True))
    out, dec = a["rq_out"], a["rq_dec"]
    ack = np.isin(out, [OC["ok"], OC["noop"]])
    rejected = ("reject", "locked", "lock_set", "expired", "churn")
    assert st["req"] == len(out)
    assert st["acc"] == (ack & (dec == m["dec_codes"].index("accept"))).sum()
    assert st["mod"] == (ack & (dec == m["dec_codes"].index("modify"))).sum()
    assert st["rej"] == np.isin(out, [OC[c] for c in rejected]).sum()
    assert st["locks"] == (out == OC["lock_set"]).sum() == len(a["lk_t"]) > 0
    assert st["lock_blocked"] == (out == OC["locked"]).sum()
    assert (out == OC["actuator"]).any()
    assert st["rollbacks"] == (a["rb_out"] == RB_CODES.index("applied")).sum() > 0
    churn = (
        (out == OC["churn"]).sum()
        + (a["rb_out"] == RB_CODES.index("churn")).sum()
        + (a["wr_out"] == WR_CODES.index("churn")).sum()
    )
    assert st["churn_blocked"] == churn > 0
    writes_ok = (a["wr_out"] == WR_CODES.index("ok")).sum()
    assert st["writes"] == writes_ok > 0
    rb_ok = (a["rb_out"] == RB_CODES.index("applied")).sum()
    assert st["changes"] == (out == OC["ok"]).sum() + rb_ok + writes_ok
    assert a["changes"][-1] == st["changes"]
    assert np.all(np.diff(a["changes"]) >= 0)
    capped = a["churn_cap"] >= 0
    assert capped.all()
    assert np.all(a["changes"][capped] <= a["churn_cap"][capped])
    assert np.all(np.diff(a["stats"], axis=0) >= 0)


def test_npz_round_trip(exercised: tuple[E6Env, Trace], tmp_path: Path) -> None:
    _, tr = exercised
    p = tmp_path / "ep.npz"
    tr.to_npz(p)
    back = Trace.from_npz(p)
    assert back.meta == tr.meta
    assert back.meta["schema"] == "e6-trace/1"
    assert set(back.arrays) == set(tr.arrays)
    for k, v in tr.arrays.items():
        assert np.array_equal(back[k], v, equal_nan=v.dtype.kind == "f"), k
        assert back[k].dtype == v.dtype, k
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, __meta__=np.array('{"schema": "other/1"}'))
    with pytest.raises(ValueError, match="not an E6 trace"):
        Trace.from_npz(bad)

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from oran_causal.methods.system.published import (
    MANIFEST_P,
    QACM,
    QOS_P,
    KPIPredictor,
    int_distance,
    kpi_of,
    one_step_values,
    run_logged,
    transitions,
)
from oran_causal.sim.system import config as C
from oran_causal.sim.system.env import E6Env

NC = 4
ZS = {
    "ll_delay": (0.05, 0.05),
    "embb_thp": (3e6, 1e6),
    "ho_fail": (0.0, 0.05),
    "energy": (0, 1),
    "prot_below": (0.02, 0.02),
    "edge_sinr": (0.0, 3.0),
}
STATIC = {"xapps": {"MRO": [], "TS": [], "ES": [], "SLICE": []}}


def fast(t: float, delay: float, util: float = 0.5) -> dict[str, Any]:
    return {
        "gran": "fast",
        "t0": t - 1,
        "t1": t,
        "ll_delay_p95": np.full(NC, delay, float),
        "prb_util": np.full(NC, util, float),
    }


def thp(t: float, p5: float) -> dict[str, Any]:
    return {"gran": "thp", "t0": t - 5, "t1": t, "embb_thp_p5": np.full(NC, p5, float)}


def obs(t: float, reqs: list[dict], reports: Sequence[dict] = ()) -> dict[str, Any]:
    return {
        "t": float(t),
        "new_reports": list(reports),
        "requests": reqs,
        "static": STATIC,
        "config": {("cio", 0, 1): 0.0, ("ll_ratio", 0): 0.1, ("carrier", 0): 2.0},
    }


def obs_p(
    t: float,
    reqs: list[dict],
    config: dict,
    reports: Sequence[dict] = (),
    xapps: Sequence[str] = ("ES", "SliceGuarantee"),
) -> dict[str, Any]:
    return {
        "t": float(t),
        "requests": reqs,
        "config": dict(config),
        "new_reports": list(reports),
        "static": {"xapps": {x: [] for x in xapps}},
    }


def req(x: str, k: tuple, cur: float, prop: float, t: float = 1.0) -> dict[str, Any]:
    return {"xapp": x, "ver": 1, "knob": k, "cur": cur, "prop": prop, "t": t}


class LinearPredictor:
    """Stub predictor: z-KPI moves linearly with the pending change (per knob type and KPI)."""

    def __init__(self, slope: dict[tuple[str, str], float]) -> None:
        self.slope = slope

    def z(self, name: str, v: Any) -> np.ndarray:
        mu, sd = ZS[name]
        return (np.asarray(v, float) - mu) / sd

    def predict_z(self, typ: str, name: str, X: Any) -> np.ndarray:
        X = np.asarray(X, float)
        return self.z(name, X[:, 5]) + self.slope.get((typ, name), 0.0) * X[:, 3]


def test_qacm_rejects_request_that_degrades_a_violating_xapp() -> None:
    q = QACM(LinearPredictor({("cio", "ll_delay"): 1.0}))
    out = q(obs(5, [req("TS", ("cio", 0, 1), 0.0, 1.0)], [fast(5, 0.2), thp(5, 3e6)]))
    assert out["decisions"] == ["reject"]
    assert q.stats["conflicts"] == 1


def test_qacm_accepts_when_no_other_xapp_violates() -> None:
    q = QACM(LinearPredictor({("cio", "ll_delay"): 1.0}))
    out = q(obs(5, [req("TS", ("cio", 0, 1), 0.0, 1.0)], [fast(5, 0.01), thp(5, 3e6)]))
    assert out["decisions"] == ["accept"]
    assert q.stats["conflicts"] == 0


def test_qacm_accepts_request_that_helps_the_violating_xapp() -> None:
    q = QACM(LinearPredictor({("cio", "ll_delay"): -1.0}))
    out = q(obs(5, [req("TS", ("cio", 0, 1), 0.0, 1.0)], [fast(5, 0.2), thp(5, 3e6)]))
    assert out["decisions"] == ["accept"]
    assert q.stats["conflicts"] == 1


def test_qacm_modifies_to_an_interior_compromise() -> None:
    pred = LinearPredictor({("ll_ratio", "ll_delay"): -10.0, ("ll_ratio", "embb_thp"): -1.0})
    q = QACM(pred)
    out = q(obs(5, [req("SLICE", ("ll_ratio", 0), 0.1, 0.2)], [fast(5, 0.12), thp(5, 1.9e6)]))
    d = out["decisions"][0]
    assert isinstance(d, tuple)
    assert d[0] == "modify"
    assert 0.1 < d[1] < 0.2


def test_qacm_direct_same_knob_conflict_keeps_one_request() -> None:
    q = QACM(LinearPredictor({}))
    reqs = [req("MRO", ("cio", 0, 1), 0.0, -1.0), req("TS", ("cio", 0, 1), 0.0, 1.0)]
    out = q(obs(5, reqs, [fast(5, 0.01), thp(5, 3e6)]))
    assert sorted(out["decisions"]) == ["accept", "reject"]


def test_one_step_values_respect_actuator_step() -> None:
    assert one_step_values(("cio", 0, 1), 0.0) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert one_step_values(("ttt", 0), 320.0) == [256.0, 320.0, 480.0]
    assert one_step_values(("carrier", 0), 2.0) == [1.0, 2.0]


def test_one_step_values_cover_e6p_knob_types() -> None:
    assert one_step_values(("ptx", 0), 0.0) == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert one_step_values(("ptx", 0), -8.0) == [-9.0, -8.0, -7.0, -6.0, -5.0]
    assert one_step_values(("prot_min", 0), 0.0) == [0.0, 0.05, 0.1]
    assert one_step_values(("sleep", 5), 0.0) == [0.0, 1.0]


def fast_p(t: float, below: float, edge: float = 0.0) -> dict[str, Any]:
    return {
        "gran": "fast",
        "t0": t - 1,
        "t1": t,
        "prb_util": np.full(NC, 0.3),
        "prot_below_frac": np.full(NC, below),
        "edge_sinr_p": np.full(NC, edge),
    }


def test_kpi_of_reads_e6p_reports() -> None:
    rep = fast_p(5, 0.1, -7.0)
    assert np.all(kpi_of(rep, "prot_below") == 0.1)
    assert np.all(kpi_of(rep, "edge_sinr") == -7.0)


def test_qacm_e6p_rejects_carrier_off_only_when_protected_floors_fail() -> None:
    pred = LinearPredictor({("carrier", "prot_below"): -2.0})
    cfg = {("carrier", 1): 2.0}
    r = [req("ES", ("carrier", 1), 2.0, 1.0)]
    q = QACM(pred, manifest=MANIFEST_P, qos=QOS_P)
    assert q(obs_p(5, r, cfg, [fast_p(5, 0.2)]))["decisions"] == ["reject"]
    assert q.stats["conflicts"] == 1
    q2 = QACM(pred, manifest=MANIFEST_P, qos=QOS_P)
    assert q2(obs_p(5, r, cfg, [fast_p(5, 0.0)]))["decisions"] == ["accept"]
    assert q2.stats["conflicts"] == 0


def test_qacm_e6p_direct_ptx_conflict_between_power_xapps() -> None:
    q = QACM(LinearPredictor({("ptx", "edge_sinr"): 1.0}), manifest=MANIFEST_P, qos=QOS_P)
    reqs = [req("PowerES", ("ptx", 1), 0.0, -3.0), req("Coverage", ("ptx", 1), 0.0, 3.0)]
    o = obs_p(5, reqs, {("ptx", 1): 0.0}, [fast_p(5, np.nan, -8.0)], ("PowerES", "Coverage"))
    assert q(o)["decisions"] == ["reject", "accept"]


def test_int_distance_zero_for_identical_and_grows_with_shift() -> None:
    a = np.random.default_rng(0).normal(size=500)
    assert int_distance(a, a) == 0.0
    assert 0 < int_distance(a, a + 0.5) < int_distance(a, a + 2.0) <= 1.0


@pytest.fixture(scope="module")
def dev_log() -> list[dict]:
    return run_logged(C.E6Config(seed=11, mix="M4", load="high", warmup_s=10, scored_s=50))[1]


def test_transitions_and_predictor_fit_from_logs(dev_log: list[dict]) -> None:
    data = transitions([dev_log], ("ll_delay", "embb_thp"))
    assert ("ll_ratio", "ll_delay") in data
    X, y = data[("ll_ratio", "ll_delay")]
    assert X.shape[1] == 7
    assert len(X) == len(y) > 0
    assert np.all(np.isfinite(X))
    pred = KPIPredictor(kpis=("ll_delay", "embb_thp")).fit([dev_log])
    assert pred.report[("ll_ratio", "ll_delay")]["model"] in ("ANN", "PR")


def test_qacm_runs_closed_loop(dev_log: list[dict]) -> None:
    cfg = C.E6Config(seed=12, mix="M4", load="high", warmup_s=10, scored_s=20)
    q = QACM(KPIPredictor(kpis=("ll_delay", "embb_thp", "ho_fail")).fit([dev_log]))
    s = E6Env(cfg, log=False).run(q)
    assert s["req"] > 0
    assert s["acc"] + s["mod"] <= s["req"]
    assert q.stats["accept"] + q.stats["reject"] + q.stats["modify"] > 0

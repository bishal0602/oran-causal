from typing import Any

import numpy as np
import pytest

from oran_causal.sim.system import config as C
from oran_causal.sim.system.decision import collect as CP
from oran_causal.sim.system.decision import labels as LP
from oran_causal.sim.system.decision import units as U
from oran_causal.sim.system.env import E6Env

H = 20


def cfg_p3(seed: int = 7) -> C.E6Config:
    return C.E6Config(
        seed=seed,
        mix="ES",
        scenario="surge",
        load_factor=1.0,
        n_ue=300,
        n_pico=3,
        mobility="ped",
        warmup_s=60.0,
        scored_s=60.0,
        e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")),
    )


def _kind(u: dict[str, Any]) -> str | None:
    if u["x"] == "ES" and u["knob"] == "carrier":
        return "es"
    if u["x"] == "SliceGuarantee":
        return "sg"
    if u["knob"] == "sleep":
        return "pico"
    return None


def held_decisions(at: LP.Paused, u: dict[str, Any]) -> list[tuple[float, str]]:
    sim = at.env.copy(reseed=1)
    arb = at.snap.fork(U.HoldPolicy({(u["c"], u["x"])}, "reject", u["t0"], u["t0"] + 1))
    seen, o = [], at.obs
    for _ in range(int(U.T_UNIT) + 15):
        d = arb(o)
        for r, dd in zip(o["requests"], d["decisions"], strict=True):
            if r["xapp"] == u["x"] and r["knob"][1] == u["c"]:
                seen.append((o["t"], dd))
            else:
                assert dd == "accept"
        sim.step_apply(d)
        if sim.sec >= sim.total_s:
            break
        o = sim.step_propose()
    return seen


def _record(lab: LP.Labeller, at: LP.Paused, u: dict[str, Any], env: E6Env) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "unit": u,
        "label": lab.label(at, u),
        "aa": [LP.aa_rollout(env, k, H) for k in LP.KS],
        "cache": {k: lab.cache[(u["t0"], k)] for k in LP.KS},
    }
    rec["probe"] = lab.region_probe(at, u, env.plant.lay.cell_site, "reject", rec["label"])
    if _kind(u) == "es":
        pol = U.HoldPolicy({(u["c"], u["x"])}, "reject", u["t0"], u["t0"] + 1)
        rec["rep"] = [LP.rollout(at, pol, 1, H) for _ in range(2)]
        rec["rep2"] = LP.rollout(at, pol, 2, H)
        rec["held"] = held_decisions(at, u)
    return rec


@pytest.fixture(scope="module")
def labelled() -> tuple[dict[str, Any], dict[str, Any]]:
    lab = LP.Labeller(H=H)
    out: dict[str, Any] = {}

    def hook(
        env: E6Env, obs: Any, snap: U.UnitArbiter | None, opened: list[dict[str, Any]]
    ) -> None:
        assert snap is not None
        for u in opened:
            want = _kind(u)
            if want is None or want in out:
                continue
            out[want] = _record(lab, LP.Paused(env, obs, snap), u, env)

    res = CP.run_collection(cfg_p3(), CP.RandomizedUnitPolicy(7, "high"), labeller=hook)
    assert {"es", "sg"} <= set(out)
    return out, res


@pytest.mark.slow
@pytest.mark.claim("DEF-EST-TAUAA")
def test_accept_label_equals_a_direct_accept_all_rollout(labelled: tuple) -> None:
    out, _ = labelled
    for rec in out.values():
        u, lab = rec["unit"], rec["label"]
        for i, k in enumerate(LP.KS):
            aa, cached = rec["aa"][i], rec["cache"][k]
            assert np.array_equal(cached["net"], aa["net"])
            assert np.array_equal(cached["cell"], aa["cell"])
            assert lab["raw"]["accept"][i]["net"] == [float(x) for x in aa["net"]]
            assert lab["raw"]["accept"][i]["exp"] == [
                float(aa["cell"][j, u["exp"]].sum()) for j in range(3)
            ]
            assert aa["secs"] == H
        assert all(v == [0.0, 0.0] for sc in lab["d"]["accept"].values() for v in sc.values())


@pytest.mark.slow
@pytest.mark.claim("DEF-EST-TAUAA")
def test_contrasts_are_mode_minus_accept_per_reseed_and_mean_over_k(labelled: tuple) -> None:
    out, _ = labelled
    for rec in out.values():
        lab = rec["label"]
        assert lab["modes"] == list(LP.label_modes(rec["unit"]["x"]))
        assert lab["ks"] == [1, 2]
        for m in lab["modes"]:
            for sc in ("exp", "net"):
                for j, kpi in enumerate(LP.KPIS):
                    per_k = [
                        lab["raw"][m][i][sc][j] - lab["raw"]["accept"][i][sc][j] for i in range(2)
                    ]
                    assert lab["d"][m][sc][kpi] == pytest.approx(per_k)
                    assert lab["mean"][m][sc][kpi] == pytest.approx(np.mean(per_k))
        assert lab["cpu_s"] > 0


@pytest.mark.slow
def test_same_reseed_reproduces_and_different_reseeds_redraw(labelled: tuple) -> None:
    out, _ = labelled
    a, b, c = out["es"]["rep"][0], out["es"]["rep"][1], out["es"]["rep2"]
    assert np.array_equal(a["net"], b["net"])
    assert np.array_equal(a["cell"], b["cell"])
    assert not np.array_equal(a["cell"], c["cell"])
    assert out["es"]["label"]["raw"]["reject"][0]["net"] == [float(x) for x in a["net"]]


@pytest.mark.slow
@pytest.mark.claim("DEF-EST-TAUAA")
def test_held_mode_governs_the_unit_for_t_then_accept(labelled: tuple) -> None:
    out, _ = labelled
    u, seen = out["es"]["unit"], out["es"]["held"]
    assert seen
    assert seen[0][0] == u["t0"]
    for t, d in seen:
        assert d == ("reject" if t < u["t0"] + U.T_UNIT else "accept")


def test_label_sampling_rates_and_tag() -> None:
    for x, rate in (("ES", 0.3), ("PowerES", 0.3), ("SliceGuarantee", 0.05)):
        assert LP.LABEL_RATE[x] == rate
        hits = [
            LP.sampled(11, {"c": c, "x": x, "x_idx": U.x_index(x), "t0": float(t)})
            for c in range(24)
            for t in range(120, 720)
        ]
        assert np.mean(hits) == pytest.approx(rate, abs=0.01)
    u = {"c": 3, "x": "ES", "x_idx": 0, "t0": 200.0}
    assert LP.sampled(11, u) == (np.random.default_rng([11, 6613, 3, 0, 200]).random() < 0.3)
    assert LP.label_modes("ES") == ("accept", "half", "reject") == LP.label_modes("PowerES")
    assert LP.label_modes("SliceGuarantee") == ("accept", "half", "reject", "accept+rb")


@pytest.mark.slow
def test_region_probe_covers_the_units_site_and_sums_cell_contrasts(labelled: tuple) -> None:
    out, res = labelled
    site = res["env"].plant.lay.cell_site
    pr, u = out["es"]["probe"], out["es"]["unit"]
    assert pr is not None
    assert pr["cells"] == [q for q in range(len(site)) if site[q] == site[u["c"]]]
    assert len(pr["cells"]) == 3
    assert len(pr["dR"]) == 2
    for i in range(2):
        want = np.sum([pr["dC"][str(q)][i] for q in pr["cells"]], 0).tolist()
        assert pr["sum_dC"][i] == pytest.approx(want)
    assert pr["dC"][str(u["c"])][0] == [
        out["es"]["label"]["d"]["reject"]["net"][k][0] for k in LP.KPIS
    ]
    if "pico" in out:
        assert out["pico"]["probe"] is None

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from oran_causal.core.data import UnitTable
from oran_causal.experiments.system import study1_discovery as d1
from oran_causal.experiments.system import study2_discovery as d2
from oran_causal.sim.system.decision.vocab import HYPOTHESES


def test_declared_list() -> None:
    decl: dict = {h: {"declared": False, "sign": 0} for h in HYPOTHESES}
    decl[("sleep", "nbr", "pv")] = {"declared": True, "sign": 1}
    decl[("ptx", "nbr", "load")] = {"declared": True, "sign": -1}
    assert d1.declared_list(decl) == [("ptx|nbr|load", -1), ("sleep|nbr|pv", 1)]
    rec = d1.decl_record({("a", "b", "c"): {"declared": True, "sign": 1, "p": 0.1, "x": 9}})
    assert rec == {"a|b|c": {"declared": True, "sign": 1, "p": 0.1}}


def test_slice_plan_full() -> None:
    plan = d1.slice_plan(range(188100, 188700))
    kinds = [s.kind for s in plan]
    assert kinds.count("60") == 10
    assert kinds.count("120") == 5
    assert kinds.count("300") == 2
    s = {p.name: p for p in plan}
    assert (s["pooled"].split, s["pooled"].n_target) == (0, 600)
    assert (s["s60_3"].split, s["s60_3"].seeds[0]) == (23, 188280)
    assert (s["s120_4"].split, s["s300_1"].split) == (44, 61)
    assert len(s["s300_1"].seeds) == 300


def test_slice_plan_pseudo_uses_ranks() -> None:
    plan = d1.slice_plan([5, 9, 7, 1] * 1 + list(range(100, 120)), pseudo=True, sizes=(4, 8, 16))
    assert [p.name for p in plan if p.kind == "60"][:2] == ["s60_0", "s60_1"]
    assert plan[1].seeds == (1, 5, 7, 9)


def test_null_groups_and_shifts() -> None:
    seeds = list(range(188100, 188700))
    groups = d1.null_groups(seeds, 60, pseudo=False)
    assert len(groups) == 10
    assert groups[1] == list(range(60, 120))
    assert d1.null_groups(list(range(7)), 3, pseudo=True) == [[0, 1, 2], [3, 4, 5]]
    recs = [{"seed": i, "lab_series": f"L{i}"} for i in range(3)]
    v = d1.shift_variants(recs, 4)
    assert [n for n, _ in v] == ["shift1", "shift2", "shift4"]
    assert [r["lab_series"] for r in v[1][1]] == ["L2", "L0", "L1"]
    assert [r["seed"] for r in v[1][1]] == [0, 1, 2]


class _Data:
    n = 7


def test_placebo_check() -> None:
    def method(data: object, split: int) -> dict:
        assert split == 9
        return {
            ("a", "own", "pv"): {"declared": True, "p": 0.01},
            ("b", "own", "pv"): {"declared": False, "p": 0.2},
            ("c", "own", "pv"): {"declared": False, "p": float("nan")},
            ("d", "own", "pv"): {},
        }

    out = d1.placebo_check(method, cast(UnitTable, _Data()))
    assert out["declared"] == ["a|own|pv"]
    assert (out["n_p"], out["n_reject"], out["rate"], out["min_p"]) == (2, 1, 0.5, 0.01)
    assert out["units"] == 7


def test_keys_to_str() -> None:
    assert d1.keys_to_str({("a", "b"): [{("c",): 1}]}) == {"a|b": [{"c": 1}]}


@pytest.mark.parametrize(
    ("u", "why"),
    [
        ({"mode": "accept", "p": 0.6, "probs": {"accept": 0.6, "reject": 0.4}}, None),
        ({"mode": "accept", "p": 0.6}, "missing"),
        ({"mode": "accept", "p": 0.6, "probs": {"accept": 0.6, "half": 0.4}}, "keys"),
        ({"mode": "accept", "p": 0.6, "probs": {"accept": 0.6, "reject": 0.5}}, "sum"),
        ({"mode": "accept", "p": 0.9, "probs": {"accept": 0.9, "reject": 0.1}}, "floor"),
        ({"mode": "reject", "p": 0.6, "probs": {"accept": 0.6, "reject": 0.4}}, "p"),
    ],
)
def test_check_unit_row(u: dict, why: str | None) -> None:
    assert d2.check_unit_row(u) == why


def test_check_episode() -> None:
    rec = {
        "pi0_table": None,
        "directional": True,
        "units": [
            {"knob": "sleep", "mode": "accept", "p": 0.6, "probs": {"accept": 0.6, "reject": 0.4}},
            {"knob": "cio", "mode": "accept"},
            {"knob": "ptx", "mode": "accept", "p": 0.6},
        ],
    }
    assert d2.check_episode(rec) == {
        "units": 2,
        "bad": {"missing": 1},
        "pi0_table_null": True,
        "directional": True,
        "ok": False,
    }


def test_repro_check() -> None:
    out = dict.fromkeys(d2.study2.REPRO_SEEDS, 0)
    base = dict.fromkeys(("prot_viol", "prot_ue_s", "energy_j", "viol_ue_s", "ue_s"), 1.0)
    base |= {"nonprot_embb_viol": 2.0, "ll_viol": 0.0, "rlf": 3.0}
    coll = {s: {"lab_outcome": base} for s in out}
    arms = {s: {"incumbent": base, "MG:allaccept": base, "noarb": base} for s in out}
    r = d2.repro_check(coll, arms)
    assert r["noarb_alias"]
    assert r["incumbent_vs_logging"]["ok"]
    arms[186081]["noarb"] = base | {"rlf": 4.0}
    r = d2.repro_check(coll, arms)
    assert not r["noarb_alias"]
    assert r["allaccept_vs_noarb"]["per_seed"][186081] == 1.0


def test_granger_plc_transfer() -> None:
    plc = {
        ("sleep", "nbr", "pv"): {"r2": 0.3, "sign": 1},
        ("ptx", "own", "v"): {"r2": 0.2, "sign": -1},
        ("carrier", "far", "e"): {"r2": 0.9, "sign": 1},
    }
    disc = {
        ("sleep", "nbr", "pv"): {"r2": 0.25, "sign": 1},
        ("ptx", "own", "v"): {"r2": 0.1, "sign": -1},
    }
    decl, tau = d2.granger_plc_transfer(plc, disc, ("own", "nbr"))
    assert tau == {"tau_r2": 0.2, "max_fp": 1, "n_scorable": 2}
    assert decl[("sleep", "nbr", "pv")]["declared"]
    assert not decl[("ptx", "own", "v")]["declared"]
    assert decl[("carrier", "own", "e")] == {
        "declared": False,
        "sign": 0,
        "score": decl[("carrier", "own", "e")]["score"],
    }
    _, tau1 = d2.granger_plc_transfer({("ptx", "own", "v"): {"r2": 0.2, "sign": 1}}, {}, ("own",))
    assert tau1["tau_r2"] == float("-inf")


def test_by_matches_step_up() -> None:
    p = [0.001, 0.04, 0.2, 0.0005, 0.03]
    m, c = len(p), sum(1 / k for k in range(1, len(p) + 1))
    order = np.argsort(p, kind="stable")
    ok = np.asarray(p)[order] <= np.arange(1, m + 1) * 0.05 / (m * c)
    want = np.zeros(m, bool)
    want[order[: np.nonzero(ok)[0].max() + 1]] = True
    assert d2._by(p) == want.tolist()
    assert d2._by([]) == []


def test_flip_summary_is_seeded() -> None:
    rep = {
        "flips": {
            "b": {
                "per_episode": [1, 2, 3, 4],
                "pre_warm": [0, 1],
                "post_warm": [1],
                "by": {"x": [1, 0]},
            },
            "a": {"per_episode": [0], "pre_warm": [], "post_warm": [], "by": {}},
        }
    }
    s1, s2 = d2.flip_summary(rep, 200), d2.flip_summary(rep, 200)
    assert s1["b"] == s2["b"]
    assert s1["b"]["mean_per_episode"] == 2.5
    assert s1["b"]["by_mean"] == {"x": 0.5}
    assert np.isnan(s1["a"]["ci90"][0])
    assert s1["b"]["ci90"][0] <= 2.5 <= s1["b"]["ci90"][1]


def test_code_sha256_names_the_pinned_modules() -> None:
    sha = d2.code_sha256()
    assert set(sha) == {
        "oran_causal.experiments.system.study2_discovery",
        "oran_causal.experiments.system.study2",
        "oran_causal.methods.system.referees",
    }
    assert all(len(v) == 64 for v in sha.values())


def _fake_analysis(monkeypatch: pytest.MonkeyPatch, module: Any, tests: dict) -> None:
    monkeypatch.setattr(module, "analyze", lambda inputs, settings: tests)


def test_study2_run_writes_tests_and_disc_conf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_analysis(monkeypatch, d2, {"schema": "tests", "n": (1, 2)})
    seen = []

    def disc_from_tests(t: dict) -> dict:
        seen.append(t)
        return {"label": {"label": "DISC-PASS"}}

    monkeypatch.setattr(d2.crit2, "disc_from_tests", disc_from_tests)
    path = d2.run(cast(d2.Inputs, None), tmp_path)
    assert path == tmp_path / "disc_conf.json"
    assert seen == [{"schema": "tests", "n": [1, 2]}]
    assert json.loads((tmp_path / "tests_conf.json").read_text()) == seen[0]
    assert json.loads(path.read_text())["label"]["label"] == "DISC-PASS"


def test_study1_run_writes_tests_analysis_and_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_analysis(monkeypatch, d1, {"schema": "tests"})
    monkeypatch.setattr(d1.crit, "from_tests", lambda t: t | {"verdict": {"verdict": "V"}})
    monkeypatch.setattr(d1.crit, "verdict_record", lambda rep: {"verdict": rep["verdict"]})
    path = d1.run(cast(d1.Inputs, None), tmp_path)
    assert path == tmp_path / "analysis_v4.json"
    assert json.loads((tmp_path / "tests_v4.json").read_text()) == {"schema": "tests"}
    assert json.loads(path.read_text())["verdict"]["verdict"] == "V"
    assert json.loads((tmp_path / "verdict_v4.json").read_text()) == {"verdict": {"verdict": "V"}}

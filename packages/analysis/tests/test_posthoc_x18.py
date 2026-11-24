import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from oran_causal.analysis.posthoc import x18
from oran_causal.analysis.posthoc.refs import RefereeTools
from oran_causal.analysis.system import stats

SEEDS = list(range(191100, 191112))
VIOL = {
    "noarb": 90.0,
    "freeze": 60.0,
    "sub:ES+PowerES": 40.0,
    "sub:ES": 50.0,
    "sub:PowerES": 30.0,
    "sub:SliceGuarantee": 70.0,
    "never_sleep": 80.0,
    "CS:PMRT": 80.0,
    "MG:PMRT": 60.0,
}
ENERGY = {"freeze": 3.5e6, "sub:ES+PowerES": 2.0e6, "noarb": 2.0e6}
RLF = {"MG:PMRT": 45.0, "sub:PowerES": 47.0}


def rec(arm: str, s: int, viol: float | None = None) -> dict:
    rng = np.random.default_rng([s, len(arm)])
    return {
        "arm": arm,
        "prot_viol": (VIOL[arm] if viol is None else viol) + float(rng.normal(0, 1)),
        "prot_ue_s": 900.0,
        "energy_j": ENERGY.get(arm, 2.1e6),
        "viol_ue_s": 2e4,
        "ue_s": 1.2e5,
        "nonprot_embb_viol": 1.8e4,
        "ll_viol": 10.0,
        "rlf": RLF.get(arm, 40.0),
    }


def records() -> stats.Records:
    return {s: {a: rec(a, s) for a in VIOL} for s in SEEDS}


ARMS = ["never_sleep", "CS:PMRT", "MG:PMRT"]


def test_grids_follow_the_declaration():
    assert x18.rho_of(1.10) == pytest.approx(1.05)
    assert x18.rho_of(1.30) == pytest.approx(1.15)
    assert pytest.approx((1.025, 1.05, 1.075, 1.10, 1.125, 1.15)) == x18.RHO_GRID
    assert x18.key(1.1) == "1.100"
    assert x18.cell_key(1.05, 0.8) == "1.050|0.800"
    assert len(x18.sig_id("x")) == 12


def test_eligibility_rule_equals_the_frozen_gate_a():
    R = records()
    st = stats.gate_a(R, ARMS, n_boot=0)
    P = x18.pooled_all(R, st.seeds, ARMS)
    assert x18.check_frozen_eligibility(st, P, [*stats.ANCHORS, *ARMS]) == []
    assert not st.point["MG:PMRT"]["eligible"]


def test_a_looser_limit_admits_the_rlf_breach_and_a_floor_can_exclude():
    R = records()
    st = stats.gate_a(R, ARMS, n_boot=0)
    P = x18.pooled_all(R, st.seeds, ARMS)
    assert x18.eligibility(P, "MG:PMRT", 1.10, 0.90)["guard_fail"] == ["rlf"]
    assert x18.eligibility(P, "MG:PMRT", 1.15, 0.90)["eligible"]
    ret = st.point["never_sleep"]["retention"]
    assert x18.eligibility(P, "never_sleep", 1.10, ret - 0.01)["matched"]
    assert not x18.eligibility(P, "never_sleep", 1.10, ret + 0.01)["matched"]


def test_cell_picks_the_best_eligible_static_rule_and_scores_contrasts():
    R = records()
    st = stats.gate_a(R, ARMS, n_boot=500)
    P = x18.pooled_all(R, st.seeds, ARMS)
    tight = x18.cell(st, P, [*stats.ANCHORS, *ARMS], 1.10, 0.0)
    loose = x18.cell(st, P, [*stats.ANCHORS, *ARMS], 1.20, 0.0)
    assert tight["best_static"] == "sub:ES+PowerES"
    assert loose["best_static"] == "sub:PowerES"
    assert loose["arms"]["MG:PMRT"]["eligible"]
    assert tight["arms"]["MG:PMRT"]["Rstar"] <= 0.0
    row = tight["arms"]["never_sleep"]
    assert row["dRstar"] < 0
    assert not row["beats"]
    assert len(row["dRstar_ci90"]) == 2


def _cert(reuse_hi: str | None) -> dict:
    def arm(reuse: str | None, sid: str | None, unc: list) -> dict:
        return {
            "uncertified": unc,
            "uncertified_frozen": [["sleep", 1]],
            "changed_from_frozen": unc != [["sleep", 1]],
            "reuse": reuse,
            "sig_id": sid,
        }

    return {
        "by_rho": {
            "1.050": {"arms": {"CS:PMRT": arm("CS:PMRT", None, [["sleep", 1]])}},
            "1.150": {"arms": {"CS:PMRT": arm(reuse_hi, "abc", [])}},
        }
    }


def test_gated_arms_take_their_target_or_stay_pending():
    R = records()
    cert = _cert(None)
    virt, target, pending = x18.add_gated(R, cert, SEEDS, [1.05, 1.15])
    assert virt == {("CS:PMRT", "1.050"): "CS:PMRT@rho1.050"}
    assert target[("CS:PMRT", "1.150")] == "X18:abc"
    assert pending == [["CS:PMRT", "1.150", "X18:abc"]]
    for s in SEEDS:
        R[s]["X18:abc"] = rec("CS:PMRT", s, viol=55.0)
    virt, _, pending = x18.add_gated(R, cert, SEEDS, [1.05, 1.15])
    assert not pending
    assert R[SEEDS[0]]["CS:PMRT@rho1.150"]["x18_of"] == "X18:abc"


def test_gated_section_renames_rows_and_marks_pending():
    R = records()
    cert = _cert(None)
    virt, target, _ = x18.add_gated(R, cert, SEEDS, [1.05, 1.15])
    st = stats.gate_a(R, [*ARMS, *virt.values()], n_boot=200)
    P = x18.pooled_all(R, st.seeds, [*ARMS, *virt.values()])
    base = [*stats.ANCHORS, "never_sleep", "MG:PMRT"]
    sec = x18.gated_section(st, P, base, cert, virt, target, [(1.10, 0.9, 1.05), (1.3, 0.9, 1.15)])
    lo, hi = sec["1.100|0.900|1.050"], sec["1.300|0.900|1.150"]
    assert lo["arms"]["CS:PMRT"]["target"] == "CS:PMRT"
    assert lo["arms"]["CS:PMRT"]["virtual"] == "CS:PMRT@rho1.050"
    assert hi["pending"] == ["CS:PMRT"]
    assert "CS:PMRT" not in hi["arms"]


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {"kind": "header", "driver": x18.DRIVER, "cs_stage": "eval"}
    path.write_text("\n".join(json.dumps(r) for r in [head, *rows]) + "\n", encoding="utf-8")


def _job(arm: str, s: int, **kw) -> dict:
    return {
        "kind": "job",
        "schema": "e6p-optaka2-rec/1",
        "sub": x18.SUB,
        "cs_stage": "eval",
        "seed": s,
        "key": [s, arm],
        **rec("CS:PMRT", s),
        "arm": arm,
        "policy_counts": {"defer": {"sleep": 3}},
        **kw,
    }


def _plan() -> dict:
    return {
        "seeds": [191100, 191101],
        "jobs": [{"arm": "X18:abc"}],
        "repro": {"arm": x18.REPRO_ARM, "source_arm": "CS:PMRT", "seeds": [191100]},
    }


def test_part_c_reproduction_and_completeness(tmp_path: Path):
    stored = {191100: {"CS:PMRT": dict(_job("CS:PMRT", 191100), sub="certsafe")}}
    rows = [
        _job(x18.REPRO_ARM, 191100, cpu_s=9.0),
        _job("X18:abc", 191100),
        _job("X18:abc", 191101),
    ]
    _write_shard(tmp_path / "a" / "res_0.jsonl", rows)
    Rc, chk = x18.part_c_records(x18.shard_files(tmp_path), _plan(), stored)
    assert chk["repro_ok"]
    assert chk["complete"] == {"X18:abc": 2}
    assert set(Rc[191101]) == {"X18:abc"}


def test_part_c_reproduction_mismatch_stops(tmp_path: Path):
    stored = {191100: {"CS:PMRT": dict(_job("CS:PMRT", 191100), rlf=1.0)}}
    _write_shard(tmp_path / "a" / "res_0.jsonl", [_job(x18.REPRO_ARM, 191100)])
    with pytest.raises(x18.ReproductionError):
        x18.part_c_records(x18.shard_files(tmp_path), _plan(), stored)


def test_summary_finds_the_first_limit_or_pending():
    def c(beats: bool | None, pending: bool = False) -> dict:
        arms = (
            {}
            if pending
            else {
                "CS:PMRT": {
                    "beats": beats,
                    "dRstar": 0.1,
                    "dRstar_lb90": 0.05,
                    "eligible": True,
                    "uncertified": [],
                    "uncertified_frozen": [["sleep", 1]],
                },
                "CS:GT": {"beats": False, "eligible": True, "uncertified": []},
            }
        )
        out = {"arms": arms, "best_static": "never_sleep"}
        if pending:
            out["pending"] = ["CS:PMRT"]
        return out

    coupled = {}
    for i, limit in enumerate(x18.L_GRID):
        for e in x18.E_GRID:
            k = f"{x18.cell_key(limit, e)}|{x18.key(x18.rho_of(limit))}"
            coupled[k] = c(beats=i >= 3)
    stored3 = {
        "grid": {
            x18.cell_key(x, e): {"best_static": "never_sleep" if e < 0.9 else "sub:ES"}
            for x in x18.L_GRID
            for e in x18.E_GRID
        }
    }
    sm = x18.summary(stored3, coupled)
    first = sm["first_L_gated_map_beats_best_static"]["0.900"]
    assert first["status"] == "found"
    assert first["L"] == x18.L_GRID[3]
    assert first["arm"] == "CS:PMRT"
    assert sm["oracle_vs_pmrt"]["1.300|0.900"]["differ"]
    assert sm["never_sleep_best_static_floors"]["1.100"] == [0.80, 0.85]
    k = f"{x18.cell_key(1.05, 0.9)}|{x18.key(x18.rho_of(1.05))}"
    coupled[k] = c(None, pending=True)
    assert (
        x18.summary(stored3, coupled)["first_L_gated_map_beats_best_static"]["0.900"]["status"]
        == "pending"
    )


def _frozen(st: stats.GateA) -> dict:
    arms = {}
    for a, p in st.point.items():
        b = st.boot[a]
        arms[a] = {
            **{k: p[k] for k in ("V", "R", "Rstar", "retention", "eligible", "guard_ratio")},
            "R_ci90": stats.ci(b["R"]),
            "Rstar_ci90": stats.ci(b["Rstar"]),
            "retention_ci90": stats.ci(b["retention"]),
            "guard_ratio_ci90": {g: stats.ci(v) for g, v in b["guard"].items()},
        }
    return {"arms": arms, "gate_a": st.summary(), "n_seeds": st.n}


def test_stored_grid_reproduces_then_scores_every_cell():
    R = {s: {a: rec(a, s) for a in VIOL} for s in range(191100, 191260)}
    frozen = _frozen(stats.gate_a(R, ARMS, stats.N_BOOT))
    b = x18.Block("study3", R, [*stats.ANCHORS, *ARMS], frozen, {})
    _, _, out = x18.stored_grid(b)
    assert out["repro"]["ok"]
    assert len(out["grid"]) == len(x18.L_GRID) * len(x18.E_GRID)
    assert out["grid"]["1.200|0.800"]["arms"]["MG:PMRT"]["eligible"]
    frozen["arms"]["MG:PMRT"]["eligible"] = True
    with pytest.raises(x18.ReproductionError):
        x18.stored_grid(x18.Block("study3", R, b.arms, frozen, {}))


def test_only_the_pmrt_map_takes_the_strict_coefficients():
    doc = {"arms": {"MG:PMRT": {"map": [["sleep", "own", "pv", 2.0]]}, "MG:corr@dev": {"map": []}}}
    assert x18.maps_of(doc) == {"MG:PMRT": [["sleep", "own", "pv", 2.0]]}
    assert x18.maps_of(None) is None


def _json_bounds() -> RefereeTools:
    tools = SimpleNamespace(bounds_from_json=lambda rows: {tuple(r[:3]): r[3] for r in rows})
    return cast(RefereeTools, tools)


def test_strict_bounds_without_ols_take_the_frozen_ols_bounds():
    art = {"bounds": {"gt": [["sleep", "own", "rlf", {"kind": "ci", "lo": 0.0, "hi": 1.0}]]}}
    pmrt = [["sleep", "own", "rlf", {"kind": "slope", "beta": 1.0, "se": 0.5}]]
    ols = [["sleep", "own", "rlf", {"kind": "slope", "beta": 2.0, "se": 0.1}]]
    got = x18.bound_set(art, {"pmrt": pmrt}, _json_bounds(), {"pmrt": [], "ols": ols})
    assert got["ols"] == {("sleep", "own", "rlf"): ols[0][3]}
    assert got["pmrt"] == {("sleep", "own", "rlf"): pmrt[0][3]}
    with pytest.raises(KeyError):
        x18.bound_set(art, {"pmrt": pmrt}, _json_bounds())

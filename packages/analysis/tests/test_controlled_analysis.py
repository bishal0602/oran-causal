import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.analysis.controlled import fmax, power
from oran_causal.analysis.controlled.cells import (
    build_cells,
    conformal_tau,
    fixed_declare,
    screen,
)
from oran_causal.analysis.controlled.integrity import IntegrityInputs, by_declare, integrity
from oran_causal.analysis.controlled.protocol import S_CAP, S_FLOOR
from oran_causal.analysis.controlled.report import relabel_tune
from oran_causal.analysis.controlled.spec import (
    analysis_block,
    arm_kind,
    cost_field,
    expected_units,
    focal_arm,
    not_in_grid,
    planned_units,
    power_not_applicable,
)
from oran_causal.analysis.controlled.tables import clean, markdown
from oran_causal.analysis.controlled.validity import classify, cluster_ci, combine, rate
from oran_causal.analysis.records import cell_key, file_sha256, load_records, to_result
from oran_causal.core.data import EdgeResult, Result
from oran_causal.core.scoring import PLACEBO, PLACEBO_CONF
from oran_causal.sim.controlled.datasets.generate import generate_dataset
from oran_causal.sim.controlled.datasets.truth import truth_for

DATA = Path(__file__).parent / "data"
C1 = pytest.mark.claim("EMP-S4-C1")
C2C3 = pytest.mark.claim("EMP-S4-C2C3")
E4 = pytest.mark.claim("EMP-S4-E4")


def placebo_result(scores: list[float]) -> Result:
    edges = tuple(
        EdgeResult(PLACEBO, f"K{i}", s, None, 0, declared=False) for i, s in enumerate(scores)
    )
    return Result("m", "v", edges, 0.0, {})


@pytest.fixture(scope="module")
def eval_spec() -> dict[str, Any]:
    return json.loads((DATA / "eval_full_spec.json").read_text(encoding="utf-8"))


def reference_by(ps: list[float], q: float, m: int) -> list[bool]:
    cm = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    k = 0
    for rank, i in enumerate(order, 1):
        if ps[i] <= rank * q / (m * cm):
            k = rank
    out = [False] * len(ps)
    for i in order[:k]:
        out[i] = True
    return out


def test_by_declare_matches_the_reference_step_up() -> None:
    rng = np.random.default_rng(0)
    for m in (1, 5, 20, 54):
        ps = list(np.concatenate([rng.uniform(0, 1e-3, m // 4), rng.uniform(0, 1, m - m // 4)]))
        assert by_declare(ps) == reference_by(ps, 0.05, m)
    wide = by_declare([0.001, 0.004, 0.02, 0.5], m=40)
    assert wide == reference_by([0.001, 0.004, 0.02, 0.5], 0.05, 40)
    assert sum(wide) <= sum(by_declare([0.001, 0.004, 0.02, 0.5]))
    assert by_declare([], m=3) == []


def test_conformal_tau_rule_and_level() -> None:
    assert conformal_tau([placebo_result(list(range(20)))]) == 19.0
    assert conformal_tau([placebo_result(list(range(80)))]) == 76.0
    assert conformal_tau([placebo_result(list(range(19)))]) == 18.0
    assert conformal_tau([placebo_result([math.nan])]) == math.inf
    assert conformal_tau([]) == math.inf
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(4000):
        s = rng.uniform(size=41)
        hits += s[40] > conformal_tau([placebo_result(s[:40].tolist())])
    assert hits / 4000 <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 4000)


@C1
def test_three_way_validity() -> None:
    assert classify([0.051, 0.07]) == "INVALID"
    assert classify([0.03, 0.075]) == "VALID"
    assert classify([0.03, 0.08]) == "INCONCLUSIVE"
    assert classify(None) is None
    assert combine(["VALID", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert combine(["VALID", None, "INVALID"]) == "INVALID"
    assert combine([None]) == "NO_READ"
    r = rate([1, 0, 2], [10, 10, 10])
    assert r is not None
    assert r["hits"] == 3
    assert r["n"] == 30
    assert r["ci"][0] <= 0.1 <= r["ci"][1]
    assert r["validity"] == "INCONCLUSIVE"
    assert rate([0], [0]) is None


def test_simulation_replicates_the_analysis_bootstrap() -> None:
    rng = np.random.default_rng(0)
    for k in (12, 40):
        h = rng.integers(0, 4, k).astype(float)
        bs = (h @ fmax.boot_mult(k).T) / (16 * k)
        want = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        assert cluster_ci(h, np.full(k, 16.0)) == want


def test_fmax_simulation_properties() -> None:
    cells = [(w, "R1", 1.0 if w == "E4" else None, n) for w in ("E1", "E4") for n in (500, 1000)]
    keys = ("null_raw", "plac_raw")
    a = fmax.fmax_simulate(cells, 12, keys, None, nsim=400)
    b = fmax.fmax_simulate(list(reversed(cells)), 12, keys, None, nsim=400)
    assert a == b
    assert a["n_cells"] == 4
    assert a["p_count_le_fmax"] >= 0.95
    assert sum(a["count_dist"].values()) == 400
    assert 0 <= a["p_cell_invalid"] < 0.15
    big = fmax.fmax_simulate(
        [("E1", "R1", None, n) for n in (500, 1000, 4000, 8000, 24000)], 40, keys, None, nsim=400
    )
    assert big["f_max"] >= 0
    assert 0.5 < big["p_pooled_valid"] <= 1.0


def test_dependence_table_is_stamped_with_its_lf_hash(tmp_path: Path) -> None:
    path = tmp_path / "dep.json"
    path.write_bytes(b'{"worlds": {}}\r\n')
    table, again = fmax.load_dependence(path), fmax.load_dependence(path)
    assert table is not None
    assert again is not None
    assert table["_sha"] == again["_sha"] == file_sha256(path)
    assert fmax.load_dependence(tmp_path / "missing.json") is None
    assert fmax.load_dependence(None) is None


@C1
def test_verdicts_per_arm_and_wording(study: tuple[list, dict]) -> None:
    _, out = study
    val = {r["key"]: r["validity"] for r in out["V1_validity"]}
    liberal = ("corr|", "ci_native|")
    assert all(v == "INVALID" for k, v in val.items() if k.startswith(liberal) and "|R2|" in k)
    valid = ("pmrt_eq|", "ci_eq|", "pcorr_hac|")
    assert all(v == "VALID" for k, v in val.items() if k.startswith(valid))
    v = out["V4_verdicts"]
    c1 = v["C1"]
    assert c1["D"] == ["ci_native", "corr", "pcorr_hac"]
    assert c1["verdict"] == "SUPPORTED"
    assert c1["arms"]["corr"]["verdict"] == "FAILURE"
    assert c1["arms"]["pcorr_hac"]["verdict"] == "NOT A FAILURE"
    assert "i.i.d." in c1["wording"]
    assert "pcorr_hac" in c1["wording"]
    assert v["C2a"]["verdict"] == "SUPPORTED"
    assert v["C2a"]["n_cells"] == 4
    assert list(v["C2b"]["arms"]) == ["ci_eq"]
    assert v["C2b"]["verdict"] == "SUPPORTED"
    assert v["C2_wording"].startswith("using the design restores validity")
    assert v["C3"]["verdict"] == "SUPPORTED"
    assert set(v["C3"]["pooled"]) == {"plac_raw", "conf_raw"}
    assert set(v["C3_eq_arms"]) == {"ci_eq", "pc_eq"}
    assert "not specific to PMRT" in v["C3"]["wording"]
    assert v["claim"] == "SUPPORTED"
    v11 = out["V11_integrity"]
    assert v11["label"] == "FINAL", v11["failed"]
    assert v11["by_recheck"]["agree"] == v11["by_recheck"]["records"]


@C1
def test_c1_needs_three_arms_and_fallback_wordings(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    del spec["arms"]["pcorr_hac"]
    out = synth.run(synth.records(spec), spec)
    assert out["V4_verdicts"]["C1"]["verdict"] == "NOT EVALUABLE"
    records = synth.records()
    for r in records:
        if r["arm"] == "ci_eq" and r["job"]["regime"] == "R2" and r["role"] == "measure":
            j = r["job"]
            r.update(
                synth.record(
                    "ci_eq", j["world"], "R2", j["lam"], j["n"], j["seed"], "measure", liberal=True
                )
            )
    v = synth.run(records)["V4_verdicts"]
    assert v["C2b"]["verdict"] == "NOT SUPPORTED"
    assert v["C2a"]["verdict"] == "SUPPORTED"
    assert v["C2_wording"].startswith(
        "design-based inference restores validity; adding design covariates"
    )
    assert v["claim"] == "NOT SUPPORTED"


def test_same_rule_power_comparison(study: tuple[list, dict]) -> None:
    _, out = study
    rules = {(d["arm"], d["rule"]) for d in out["V3_paired"]}
    assert not any(d["arm"] == "corr" and d["regime"] == "R2" for d in out["V3_paired"])
    assert ("corr", "BY") in rules
    assert ("pc_eq", "tau") in rules
    assert ("ci_eq", "BY") in rules
    assert all(
        d["rule"] == ("tau" if d["arm"].startswith("pc_") else "BY") for d in out["V3_paired"]
    )
    assert len(out["V3_paired"]) >= len(out["V3_paired_valid_only"])


def test_tau_from_tune_records_only(synth: Any, study: tuple[list, dict]) -> None:
    records, _ = study
    used, _ = screen(records, synth.spec)
    cells, _ = build_cells(used, synth.spec)
    key = "pc_eq|" + cell_key("E1", "R2", None, 1000, 0.25)
    tune = [to_result(r) for r in records if r["role"] == "tune" and r["key"].startswith(key + "|")]
    assert len(tune) == len(synth.tune)
    assert cells[key]["tau"] == conformal_tau(tune)
    assert cells[key]["n_measure"] == len(synth.meas)


@C2C3
def test_missing_infeasible_and_under_seeded_cells_cap_verdicts(synth: Any) -> None:
    records = [
        r
        for r in synth.records()
        if not (r["arm"] == "pc_native" and r["role"] == "tune" and r["job"]["regime"] == "R1")
    ]
    bad = next(
        r
        for r in records
        if r["arm"] == "corr" and r["role"] == "measure" and r["job"]["regime"] == "R2"
    )
    bad.update(status="infeasible", edges=[], cpu_s=15000.0, reason="exceeded the safety cap")
    drop = next(
        r
        for r in records
        if r["arm"] == "pmrt_eq" and r["role"] == "measure" and r["job"]["regime"] == "R1"
    )
    records.remove(drop)
    out = synth.run(records)
    status = {r["key"]: r["status"] for r in out["V1_validity"]}
    assert all(
        s == "untuned" for k, s in status.items() if k.startswith("pc_native|") and "|R1|" in k
    )
    assert status[bad["key"].rsplit("|", 1)[0]] == "infeasible"
    assert status[drop["key"].rsplit("|", 1)[0]] == "under_seeded"
    v = out["V4_verdicts"]
    c = v["C1"]["arms"]["corr"]
    assert (c["r2_counted"], c["r2_planned"], c["verdict"]) == (1, 2, "NOT EVALUABLE")
    assert v["C1"]["D_counted"] == ["ci_native", "pcorr_hac"]
    assert v["C1"]["verdict"] == "NOT EVALUABLE"
    assert v["C2a"]["verdict"] == "PARTIAL"
    assert v["C2a"]["under_seeded"]
    assert out["V11_integrity"]["label"] == "PROVISIONAL"
    records = synth.records()
    records.remove(
        next(
            r
            for r in records
            if r["arm"] == "corr" and r["role"] == "measure" and r["job"]["regime"] == "R1"
        )
    )
    v = synth.run(records)["V4_verdicts"]
    assert v["C1"]["arms"]["corr"]["verdict"] == "FAILURE"
    assert not v["C1"]["arms"]["corr"]["complete"]
    assert v["C1"]["verdict"] == "PARTIAL"
    assert v["claim"] == "NOT SUPPORTED"


def test_t3_cells_are_shown_and_not_planned(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    spec["arms"]["ci_native"].update(max_n=500, t3_cost_cpu_s={"1000": 9100.0})
    records = synth.records(spec)
    assert not any(r["arm"] == "ci_native" and r["job"]["n"] == 1000 for r in records)
    out = synth.run(records, spec)
    t3 = [r for r in out["V1_validity"] if r["status"] == "infeasible_t3"]
    assert t3
    assert all(
        r["arm"] == "ci_native" and r["n"] == 1000 and r["infeasible_cost_cpu_s"] == 9100.0
        for r in t3
    )
    assert out["V4_verdicts"]["C1"]["arms"]["ci_native"]["r2_planned"] == 1
    assert any(r["t3"] for r in out["V10_cost"])
    assert out["V11_integrity"]["label"] == "FINAL"


def mutate(base: list, index: int, change: Any) -> list:
    records = copy.deepcopy(base)
    change(records[index])
    return records


def test_integrity_holes_make_output_provisional(synth: Any) -> None:
    base = synth.records()
    extra = synth.record("corr", "E1", "R1", None, 500, 3_000_150, "measure", liberal=False)
    cases = {
        "unexpected": [*copy.deepcopy(base), extra],
        "role": mutate(
            base,
            base.index(next(r for r in base if r["role"] == "measure")),
            lambda r: r.update(role="tune"),
        ),
        "candidates_complete": mutate(base, 5, lambda r: r.update(edges=r["edges"][1:])),
        "clean": mutate(base, 7, lambda r: r["code"].update(dirty=None)),
        "duplicates": [*copy.deepcopy(base), copy.deepcopy(base[3])],
        "stamps": mutate(base, 9, lambda r: r["run_mode"].update(protocol_sha256="q" * 64)),
        "one_platform_per_dataset": mutate(base, 11, lambda r: r["host"].update(platform="colab")),
        "pkgs_uniform": mutate(base, 13, lambda r: r.update(pkgs={"numpy": "2.1.0"})),
        "errors_listed": mutate(
            base, 15, lambda r: r.update(status="error", edges=[], error="boom")
        ),
        "commits": mutate(base, 17, lambda r: r["code"].update(commit="def")),
    }
    for check, records in cases.items():
        v11 = synth.run(records)["V11_integrity"]
        assert v11["label"] == "PROVISIONAL"
        assert check in v11["failed"], (check, v11["failed"])
    stray = synth.record("corr", "E1", "R1", None, 500, 3_000_150, "measure", liberal=True)
    out = synth.run([*copy.deepcopy(base), stray])
    reference = synth.run(base)
    assert out == reference | {"V11_integrity": out["V11_integrity"]}
    amendments = {
        "amendments": [{"id": "A1", "commit": "def", "keys": [cases["commits"][17]["key"]]}],
        "persistent_errors": [{"key": cases["errors_listed"][15]["key"], "why": "OOM"}],
    }
    assert (
        "commits"
        not in synth.run(cases["commits"], amendments=amendments)["V11_integrity"]["failed"]
    )
    listed = synth.run(cases["errors_listed"], amendments=amendments)["V11_integrity"]
    assert "errors_listed" not in listed["failed"]


def test_deterministic_and_order_free(synth: Any, study: tuple[list, dict]) -> None:
    records, out = study
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    first = json.dumps(clean(out), sort_keys=True)
    assert first == json.dumps(clean(synth.run(shuffled)), sort_keys=True)


def test_placeholder_seeds_are_reported_not_crashing(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    spec["blocks"][1]["seeds"] = "TBD"
    with pytest.raises(ValueError, match="placeholder"):
        expected_units(spec)
    out = synth.run(synth.records(), spec)
    assert out["V11_integrity"]["label"] == "PROVISIONAL"
    assert "placeholder" in out["V11_integrity"]["missing"][0]


def test_t1_seed_count_on_records(synth: Any) -> None:
    res = power.t1_seed_count(synth.records(), synth.spec)
    assert res["n_pairs"] > 0
    assert res["S"] == S_FLOOR
    assert not res["cap_binds"]
    assert all(p["cell"].split("|")[0] != "E4" for p in res["largest"])
    assert power.paired_seeds(0.3) > power.paired_seeds(0.1)
    assert power.paired_seeds(0.0) == 2


def test_pilot_subset_matches_the_campaign_aggregate(synth: Any) -> None:
    records = load_records(DATA / "pilot_subset.jsonl.gz")
    arms = synth.spec["arms"]
    spec = {
        "name": "dev_pilot",
        "arms": {
            "pmrt_eq": arms["pmrt_eq"],
            "corr": arms["corr"],
            "granger_native": {
                "ref": "granger",
                "config": {"arm": "native"},
                "declare": "by",
                "worlds": ["E3"],
            },
        },
        "blocks": [],
    }
    cells, _ = build_cells(records, spec)
    want = {
        "pmrt_eq|E2|R2|k0.25|n1000": ((0, 64), (0, 12), (3, 64), (0, 12), 0.3125, 0.0),
        "corr|E3|R2|k0.25|n1000": ((26, 32), (7, 10), (30, 32), (7, 10), 1.0, 0.7647058823529411),
        "granger_native|E3|R2|k0.25|n1000": (
            (18, 32),
            (4, 10),
            (21, 32),
            (5, 10),
            1.0,
            0.6923076923076923,
        ),
        "pmrt_eq|E4|R3|lam1|k0.25|n1000": (None, (0, 2), None, (0, 2), 1.0, 0.0),
    }
    for key, (nd, pd, nr, pr, recall, fdp) in want.items():
        p = cells[key]["primary"]
        for name, expected in (
            ("null_decl", nd),
            ("plac_decl", pd),
            ("null_raw", nr),
            ("plac_raw", pr),
        ):
            got = None if p[name] is None else (p[name]["hits"], p[name]["n"])
            assert got == expected, (key, name)
        assert p["recall"]["mean"] == pytest.approx(recall)
        assert p["fdp"]["mean"] == pytest.approx(fdp)
        assert p["validity"] == "few_seeds"
        assert cells[key]["status"] == "few_seeds"
    assert cells["pmrt_eq|E4|R3|lam1|k0.25|n1000"]["primary"]["conf_raw"]["n"] == 2
    screened = {
        "expected": None,
        "expected_error": "pilot",
        "duplicates": [],
        "unexpected": [],
        "role_mismatch": [],
    }

    v11 = integrity(records, screened, spec, IntegrityInputs())
    assert v11["by_recheck"]["agree"] == len(records)
    assert v11["n_candidates_incomplete"] == 0


def test_as_tune_relabels_records_and_spec_blocks(synth: Any) -> None:
    records = synth.records()
    relabelled, spec = relabel_tune(records, {synth.meas[0]}, synth.spec)
    used, report = screen(relabelled, spec)
    assert not report["role_mismatch"]
    assert not report["unexpected"]
    assert len(used) == len(records)
    assert all(r["role"] == "tune" for r in relabelled if r["job"]["seed"] == synth.meas[0])
    with pytest.raises(SystemExit):
        relabel_tune(records, {synth.meas[0]}, {**synth.spec, "name": "eval_full"})


def test_set_d_flag_excludes_the_unchosen_hac_variant(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    spec["arms"]["pcorr_hac"]["set_D"] = "TBD-T8"
    v = synth.run(synth.records(spec), spec)["V4_verdicts"]["C1"]
    assert v["D"] == ["ci_native", "corr"]
    assert v["verdict"] == "NOT EVALUABLE"
    assert list(v["descriptive"]) == ["pcorr_hac"]
    assert v["descriptive"]["pcorr_hac"]["verdict"] == "NOT A FAILURE"


def test_arm_out_of_d_is_descriptive_and_drops_its_eq_partner(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    spec["arms"]["ci_native"]["set_D"] = False
    out = synth.run(synth.records(spec), spec)
    v = out["V4_verdicts"]
    assert v["C1"]["D"] == ["corr", "pcorr_hac"]
    assert v["C1"]["verdict"] == "NOT EVALUABLE"
    assert v["C1"]["descriptive"]["ci_native"]["verdict"] == "FAILURE"
    assert "ci_eq" not in v["C2b"]["arms"]
    assert "ci_native (not in D, descriptive): **FAILURE**" in markdown(out)


def test_tau_pos_inf_declares_nothing(synth: Any) -> None:
    records = synth.records()
    for r in records:
        j = r["job"]
        if (
            r["arm"] == "pc_eq"
            and r["role"] == "tune"
            and (j["regime"], j["n"], j["world"]) == ("R1", 500, "E1")
        ):
            for e in r["edges"]:
                if e["source"] == PLACEBO:
                    e["score"] = None
    key = "pc_eq|" + cell_key("E1", "R1", None, 500, 0.25)
    row = next(x for x in synth.run(records)["V1_validity"] if x["key"] == key)
    assert row["tau_is_pos_inf"]
    assert row["tau"] is None
    assert row["status"] == "ok"
    assert row["null_decl"]["rate"] == 0.0
    assert row["plac_decl"]["rate"] == 0.0


@C2C3
def test_r56_eq_table_c2b_sensitivity_and_named_arm(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    spec["arms"]["ci_native"]["c1_sensitivity_drop"] = True
    liberal = {
        ("corr", "R2"),
        ("ci_native", "R2"),
        ("ci_native", "R1"),
        ("ci_eq", "R1"),
        ("ci_eq", "R2"),
    }
    records = [
        synth.record(
            u["arm"],
            u["world"],
            u["regime"],
            u["lam"],
            u["n"],
            u["seed"],
            u["role"],
            liberal=(u["arm"], u["regime"]) in liberal,
        )
        for u in expected_units(spec).values()
    ]
    out = synth.run(records, spec)
    v = out["V4_verdicts"]
    native = v["C1"]["arms"]["ci_native"]
    assert native["verdict"] == "INVALID IN R1"
    assert native["r2_legs"]
    assert v["C2b"]["arms"] == {}
    assert v["C2b"]["verdict"] == "NOT EVALUABLE"
    w = v["C2b"]["with_r1_failing_partners"]
    assert w["added"] == ["ci_eq"]
    assert w["verdict"] == "NOT SUPPORTED"
    table = v["eq_arms_table"]
    assert set(table) == {"ci_eq", "pc_eq"}
    assert table["pc_eq"]["membership"] == "not a member (declare tau)"
    assert table["ci_eq"]["membership"] == "not a member (native ci_native: INVALID IN R1)"
    assert table["ci_eq"]["verdict"] == "NOT SUPPORTED"
    assert table["ci_eq"]["r2_invalid"] == table["ci_eq"]["r2_planned"]
    assert [x["arm"] for x in v["C2_named"]] == ["ci_eq"]
    assert "ci_eq INVALID in R2 as already in R1" in v["C2_wording"]
    assert "miscalibrated without the design" in v["C2_named"][0]["sentence"]
    sw = v["C1"]["sensitivity_without"]
    assert sw["dropped"] == ["ci_native"]
    assert sw["D_counted"] == ["corr", "pcorr_hac"]
    assert sw["verdict"] == "NOT EVALUABLE"
    md = markdown(out)
    assert "C1 sensitivity without ci_native (R-56" in md
    assert "C2b with the R1-failing partners (R-56" in md
    assert "| ci_eq | not a member (native ci_native: INVALID IN R1) |" in md
    assert "unfiltered (R-56)" in md


def test_r56_default_study_eq_table_and_no_sensitivity(study: tuple[list, dict]) -> None:
    v = study[1]["V4_verdicts"]
    assert "sensitivity_without" not in v["C1"]
    assert v["C2_named"] == []
    assert v["eq_arms_table"]["ci_eq"]["membership"] == "member"
    assert v["C2b"]["with_r1_failing_partners"]["added"] == []
    assert v["C2b"]["with_r1_failing_partners"]["verdict"] == v["C2b"]["verdict"]


def test_t6_cap_60_and_achieved_power() -> None:
    assert (S_CAP, S_FLOOR) == (60, 40)
    for sd in (0.1, 0.18, 0.3):
        s = power.paired_seeds(sd)
        assert power.achieved_power(sd, s) >= 0.8 > power.achieved_power(sd, max(3, s - 3))
    assert power.achieved_power(0.18, 60) > 0.99
    assert power.achieved_power(0.0, 40) == 1.0


@C2C3
def test_c2b_exclusion_and_labels(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    label = "single-conditioner max statistic; cannot condition on the joint design set"
    spec["arms"]["ci_eq"].update({"c2b": False, "label": label})
    spec["arms"]["ci_native"]["label"] = "dependence test"
    out = synth.run(synth.records(spec), spec)
    v = out["V4_verdicts"]
    assert v["C2b"]["arms"] == {}
    assert v["C2b"]["verdict"] == "NOT EVALUABLE"
    x = v["C2b"]["excluded"]["ci_eq"]
    assert x["label"] == label
    assert x["native"] == "ci_native"
    assert x["verdict"] == "SUPPORTED"
    assert v["claim"] == "NOT SUPPORTED"
    assert "ci_eq" in v["C3_eq_arms"]
    assert v["arm_labels"] == {"ci_eq": label, "ci_native": "dependence test"}
    md = markdown(out)
    assert "ci_eq EXCLUDED from C2b (R-40) [" + label in md
    assert "ci_native [dependence test]: **FAILURE**" in md


def test_eval_spec_r40_and_set_d(eval_spec: dict[str, Any]) -> None:
    arms = eval_spec["arms"]
    assert [a for a, d in arms.items() if d.get("c2b", True) is not True] == ["mscr_eq"]
    assert arms["mscr_eq"]["label"].startswith("single-conditioner max statistic")
    assert "mscr_eq_min" not in arms
    blind = sorted(
        a
        for a, d in arms.items()
        if arm_kind(d) == "native" and d["declare"] == "by" and d.get("set_D", True) is True
    )
    assert blind == [
        "corr",
        "granger_native",
        "mscr_native",
        "pcorr_hac_fb",
        "pcorr_native",
        "rcot2_native",
    ]
    assert [a for a, d in arms.items() if d.get("set_D", True) is not True] == ["pcorr_hac"]
    assert len(arms) == 24
    assert not [a for a, d in arms.items() if d.get("budget_wall_s")]


def test_eval_spec_cdl_and_focal_arm(eval_spec: dict[str, Any]) -> None:
    arms = eval_spec["arms"]
    cdl = arms["cdl"]
    assert cdl["declare"] == "tau"
    assert cdl["analysis"] == "primary"
    assert cdl["fixed_threshold"] == 0.16
    assert arm_kind(cdl) == "native"
    assert analysis_block("cdl", eval_spec) == "primary"
    assert [a for a, x in arms.items() if "fixed_threshold" in x] == ["cdl"]
    assert eval_spec["focal"] == "pmrt_nl_eq" == focal_arm(eval_spec)
    assert arm_kind(arms["pmrt_nl_eq"]) == "pmrt"
    assert analysis_block("pmrt_eq", eval_spec) == "secondary"
    assert sorted(a for a, d in arms.items() if arm_kind(d) == "pmrt") == [
        "pmrt_eq",
        "pmrt_nl_eq",
        "pmrt_r3",
    ]
    for arm in ("pmrt_eq", "pmrt_nl_eq", "pmrt_r3"):
        assert power_not_applicable(arms[arm], "R4")
        assert not power_not_applicable(arms[arm], "R3")
    assert not power_not_applicable(arms["pcorr_eq"], "R4")


def test_eval_spec_unit_counts(eval_spec: dict[str, Any]) -> None:
    mscr = {a for a, d in eval_spec["arms"].items() if "mscr" in d["ref"]}
    assert mscr == {"mscr_eq", "mscr_native"}
    assert [a for a, d in eval_spec["arms"].items() if d.get("c1_sensitivity_drop")] == [
        "mscr_native"
    ]
    units = planned_units(eval_spec)
    assert len(units) == 150_960
    assert not any(u["t3"] for u in units.values())
    assert max(u["n"] for u in units.values() if u["arm"] in mscr) == 1000
    assert sum(u["role"] == "tune" for u in units.values()) == 40_520
    datasets = {
        (u["world"], u["regime"], u["lam"], u["n"], u["kappa"], u["seed"]) for u in units.values()
    }
    assert len(datasets) == 11_000


def test_eval_spec_grid_gaps_are_all_ruled(eval_spec: dict[str, Any]) -> None:
    grid = not_in_grid(eval_spec)
    assert grid["has_rules"]
    assert grid["unexplained"] == []
    assert grid["n_cells"] == 70 + 128
    assert sum(r["n_cells"] for r in grid["by_rule"]) == grid["n_cells"]
    assert {c["arm"] for c in grid["cells"]} == {"cdl", "mscr_eq", "mscr_native"}


def test_not_in_grid_cells_never_cap_verdicts(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    others = [a for a in spec["arms"] if a != "ci_native"]
    spec["blocks"] = [{**b, "arms": others} for b in synth.spec["blocks"]] + [
        {**b, "ns": [500], "arms": ["ci_native"]} for b in synth.spec["blocks"]
    ]
    records = synth.records(spec)
    out = synth.run(records, spec)
    assert out["V11_integrity"]["checks"].get("grid_gaps_ruled") is None
    assert out["not_in_grid"]["unexplained"]
    assert {c["arm"] for c in out["not_in_grid"]["cells"]} == {"ci_native"}
    spec["not_in_grid"] = [
        {
            "ruling": "R-x",
            "arms": ["ci_native"],
            "regimes": None,
            "ns": [1000],
            "kappas": None,
            "reason": "test",
        }
    ]
    out = synth.run(records, spec)
    c1 = out["V4_verdicts"]["C1"]["arms"]["ci_native"]
    assert out["V11_integrity"]["checks"]["grid_gaps_ruled"]
    assert out["not_in_grid"]["unexplained"] == []
    assert c1["r2_planned"] == c1["r2_counted"]
    assert c1["complete"]
    assert c1["verdict"] != "NOT EVALUABLE"
    assert not [r for r in out["V1_validity"] if r["arm"] == "ci_native" and r["n"] == 1000]
    md = markdown(out)
    assert "| ci_native | raw_p | " in md
    assert "nig" in md.split("### E1 R1")[1].split("###")[0]
    assert "- R-x: ci_native (ns [1000]): " in md
    spec["not_in_grid"][0]["ns"] = [4000]
    assert not synth.run(records, spec)["V11_integrity"]["checks"]["grid_gaps_ruled"]


def test_fixed_threshold_secondary_scoring(synth: Any) -> None:
    scores = [0.16, 0.159, math.nan, 0.5]
    result = placebo_result(scores)
    assert [e.declared for e in fixed_declare(result, 0.16).edges] == [True, False, False, True]
    spec = copy.deepcopy(synth.spec)
    spec["arms"]["pc_native"]["fixed_threshold"] = 0.995
    out = synth.run(synth.records(spec), spec)
    v0 = out["V0_like_for_like"]
    assert {r["rule"] for r in v0 if r["arm"] == "pc_native"} == {"tau", "fixed"}
    assert {r["rule"] for r in v0 if r["arm"] == "pc_eq"} == {"tau"}
    fixed = [
        r for r in v0 if r["arm"] == "pc_native" and r["rule"] == "fixed" and r["world"] == "E1"
    ]
    assert fixed
    assert all(not r["placebo_is_tuning_column"] for r in fixed)
    assert all(
        r["recall"]["mean"] == 1.0 and r["null"]["rate"] == 0.0 and r["placebo"]["rate"] == 0.0
        for r in fixed
    )
    assert all(r["state"] in ("VALID", "INCONCLUSIVE") for r in fixed)
    md = markdown(out)
    assert "| pc_native | fixed |" in md
    assert "fixed threshold, fixed)" in md


def test_gpu_arm_cost_is_wall_seconds(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    gpu = {"budget_cpu_s": None, "budget_wall_s": 7200}
    spec["arms"]["ci_native"].update(gpu)
    spec["arms"]["ci_eq"].update(gpu, max_n=500, t3_cost_wall_s={"1000": 8000.0})
    assert cost_field(spec["arms"]["ci_native"]) == "wall_s"
    assert cost_field(spec["arms"]["corr"]) == "cpu_s"
    records = synth.records(spec)
    for r in records:
        if r["arm"] == "ci_native":
            r.update(wall_s=50.0, child_wall_s=60.0)
    bad = next(
        r
        for r in records
        if r["arm"] == "ci_native" and r["role"] == "measure" and r["job"]["regime"] == "R1"
    )
    bad.update(
        status="infeasible",
        edges=[],
        cpu_s=300.0,
        wall_s=9000.0,
        cost=9000.0,
        cost_kind="wall-s",
        reason="exceeded the budget 7200 wall-s (measured 9000)",
    )
    out = synth.run(records, spec)
    v1 = {r["key"]: r for r in out["V1_validity"]}
    row = v1[bad["key"].rsplit("|", 1)[0]]
    assert row["status"] == "infeasible"
    assert row["infeasible_cost_wall_s"] == 9000.0
    assert "infeasible_cost_cpu_s" not in row
    t3 = [r for r in out["V1_validity"] if r["status"] == "infeasible_t3"]
    assert t3
    assert all(r["arm"] == "ci_eq" and r["infeasible_cost_wall_s"] == 8000.0 for r in t3)
    cost = out["V10_cost"]
    gpu_rows = [r for r in cost if r["arm"] == "ci_native" and not r["t3"]]
    assert gpu_rows
    assert all(r["budget"] == "wall_s" for r in gpu_rows)
    assert all(r["wall_s_mean"] == 60.0 for r in gpu_rows if r["n_runs"])
    assert any(r["t3"] and r["t3_dev_cost_wall_s"] == 8000.0 for r in cost if r["arm"] == "ci_eq")
    assert all(r["budget"] == "cpu_s" for r in cost if r["arm"] == "corr")
    md = markdown(out)
    assert "(cost 9000 GPU wall-s)" in md
    assert "## V10 cost (GPU arms (R-41)" in md


def test_dataset_hash_equal_across_arms_and_shards(synth: Any) -> None:
    base = synth.records()
    for r in base:
        if r["arm"] == "ci_native":
            r["host"] = {"platform": "kaggle", "node": "gpu-shard", "gpu": ["Tesla T4"]}
    v11 = synth.run(base)["V11_integrity"]
    assert v11["label"] == "FINAL"
    assert v11["checks"]["dataset_hash"]
    assert v11["checks"]["dataset_hash_stamped"]
    recs = copy.deepcopy(base)
    next(r for r in recs if r["arm"] == "ci_native")["dataset_sha256"] = "other"
    v11 = synth.run(recs)["V11_integrity"]
    assert "dataset_hash" in v11["failed"]
    assert v11["n_dataset_hash_mismatch"] == 1
    recs = copy.deepcopy(base)
    bad = next(r for r in recs if r["arm"] == "ci_native" and r["role"] == "measure")
    bad.update(
        status="infeasible", edges=[], cost=9000.0, cost_kind="wall-s", dataset_sha256="other"
    )
    assert "dataset_hash" in synth.run(recs)["V11_integrity"]["failed"]
    recs = copy.deepcopy(base)
    del next(r for r in recs if r["arm"] == "ci_native")["dataset_sha256"]
    v11 = synth.run(recs)["V11_integrity"]
    assert "dataset_hash_stamped" in v11["failed"]
    assert v11["n_dataset_hash_unstamped"] == 1
    recs = copy.deepcopy(base)
    bad = next(r for r in recs if r["arm"] == "ci_native" and r["role"] == "measure")
    bad.update(status="infeasible", edges=[], reason="infeasible at this n (T3)")
    del bad["dataset_sha256"]
    out = synth.run(recs)
    assert out["V11_integrity"]["checks"]["dataset_hash_stamped"]
    assert out["V11_integrity"]["n_dataset_hash_unstamped_not_ok"] == 1
    assert "ok records without a dataset hash 0 (not-ok 1)" in markdown(out)


def r4_spec(base: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(base)
    for b in spec["blocks"]:
        b["regimes"] = [*b["regimes"], "R4"]
        b["e4_lams"]["R4"] = [1.0]
    return spec


def r4_records(synth: Any, spec: dict[str, Any]) -> list[dict[str, Any]]:
    records = synth.records(spec)
    for r in records:
        if r["arm"] == "pmrt_eq" and r["job"]["regime"] == "R4":
            not_applicable = {}
            for e in r["edges"]:
                if e["source"] != PLACEBO and not e["source"].startswith("K"):
                    e.update(p=None, score=None, declared=False)
                    not_applicable[f"{e['source']}->{e['target']}"] = "no known design (kind none)"
            r["notes"] = {"not_applicable": not_applicable, "by_family_m": 2}
    return records


@E4
def test_r4_pmrt_power_not_applicable(synth: Any) -> None:
    spec = r4_spec(synth.spec)
    out = synth.run(r4_records(synth, spec), spec)
    key = cell_key("E4", "R4", 1.0, 1000, 0.25)
    v2 = {r["key"]: r for r in out["V2_recall"]}
    pmrt = v2[f"pmrt_eq|{key}"]
    assert pmrt["power_not_applicable"]
    assert pmrt["recall"] is None
    assert pmrt["state"] == "VALID"
    assert not v2[f"ci_eq|{key}"]["power_not_applicable"]
    assert v2[f"ci_eq|{key}"]["recall"]["mean"] == 1.0
    r3 = v2[f"pmrt_eq|{cell_key('E4', 'R3', 1.0, 1000, 0.25)}"]
    assert not r3["power_not_applicable"]
    assert r3["recall"]["mean"] == 1.0
    v6 = {r["key"]: r for r in out["V6_E4"]}
    assert v6[f"pmrt_eq|{key}"]["recall"] is None
    assert v6[f"pmrt_eq|{key}"]["wrong_sign_rate"] is None
    v0 = [r for r in out["V0_like_for_like"] if r["key"] == f"pmrt_eq|{key}"]
    assert {r["rule"] for r in v0} == {"raw_p", "tau"}
    assert all(r["recall"] is None for r in v0)
    assert all(r["placebo"]["n"] > 0 for r in v0)
    assert not any(d["regime"] == "R4" for d in out["V3_paired"])
    md = markdown(out)
    v2_md = md[md.index("## V2 recall") : md.index("## V3 paired")]
    line = next(x for x in v2_md.split("\n") if x.startswith("| E4 R4 l1 |"))
    assert line.split(" | ")[1] == "NA/NA"
    assert "| pmrt_eq | raw_p | NA / " in md


def test_like_for_like_headline(synth: Any, study: tuple[list, dict]) -> None:
    records, out = study
    v0 = out["V0_like_for_like"]
    keys = {r["key"] for r in out["V1_validity"]}
    assert {r["key"] for r in v0} == keys
    for k in keys:
        rules = sorted(r["rule"] for r in v0 if r["key"] == k)
        assert rules == (["tau"] if k.startswith("pc_") else ["raw_p", "tau"])
    cell = "corr|" + cell_key("E1", "R2", None, 1000, 0.25)
    row = next(r for r in v0 if r["key"] == cell and r["rule"] == "raw_p")
    truth = truth_for("E1", "R2")
    nulls = {f"{s}->{t}" for s, t in truth.null_edges if not s.startswith("K") and s != PLACEBO}
    measured = [r for r in records if r["key"].startswith(cell + "|") and r["role"] == "measure"]
    hits = sum(
        float(e["p"]) <= 0.05
        for r in measured
        for e in r["edges"]
        if f"{e['source']}->{e['target']}" in nulls
    )
    total = sum(f"{e['source']}->{e['target']}" in nulls for r in measured for e in r["edges"])
    assert row["null"]["rate"] == pytest.approx(hits / total)
    assert row["null"]["validity"] == "INVALID"
    assert row["state"] == "INVALID"
    assert row["recall"]["mean"] == 1.0
    n_placebo = sum(s == PLACEBO for s, _ in truth.null_edges)
    assert row["placebo"]["n"] == n_placebo * len(measured)
    assert not row["placebo_is_tuning_column"]
    tau = next(r for r in v0 if r["key"] == cell and r["rule"] == "tau")
    assert tau["placebo_is_tuning_column"]
    assert tau["recall"] is not None
    pc = next(r for r in v0 if r["key"].startswith("pc_eq|") and r["rule"] == "tau")
    assert pc["declare"] == "tau"
    assert pc["null"] is not None
    md = markdown(out)
    assert md.index("## V0 like-for-like (headline, R-42)") < md.index("## V4 claim verdicts")
    assert "| pmrt_eq | raw_p | 1.00 / " in md
    assert "| pc_eq | tau | " in md
    untuned = [r for r in records if not (r["arm"] == "ci_eq" and r["role"] == "tune")]
    rows = [r for r in synth.run(untuned)["V0_like_for_like"] if r["arm"] == "ci_eq"]
    assert all(r["state"] == "untuned" and r["recall"] is None for r in rows if r["rule"] == "tau")
    assert all(r["recall"] is not None for r in rows if r["rule"] == "raw_p")


@C2C3
def test_focal_pmrt_arm_from_spec(synth: Any) -> None:
    spec = copy.deepcopy(synth.spec)
    spec["arms"] = {
        "pmrt_nl_eq": {**synth.spec["arms"]["pmrt_eq"], "analysis": "primary"},
        **spec["arms"],
    }
    spec["arms"]["pmrt_eq"] = {
        **spec["arms"]["pmrt_eq"],
        "analysis": "secondary",
        "label": "linear statistic",
    }
    spec["focal"] = "pmrt_nl_eq"
    assert focal_arm(spec) == "pmrt_nl_eq"
    assert focal_arm(synth.spec) == "pmrt_eq"
    records = synth.records(spec)
    for r in records:
        if r["arm"] == "pmrt_eq" and r["job"]["regime"] == "R2":
            j = r["job"]
            r.update(
                synth.record(
                    "pmrt_eq",
                    j["world"],
                    "R2",
                    j["lam"],
                    j["n"],
                    j["seed"],
                    r["role"],
                    liberal=True,
                )
            )
    out = synth.run(records, spec)
    v = out["V4_verdicts"]
    assert out["focal"] == v["focal"] == "pmrt_nl_eq"
    assert v["C2a"]["verdict"] == "SUPPORTED"
    assert v["C3"]["verdict"] == "SUPPORTED"
    assert v["claim"] == "SUPPORTED"
    assert v["pmrt_secondary"]["pmrt_eq"]["C2a"]["verdict"] == "NOT SUPPORTED"
    assert "pmrt_eq" not in v["C3_eq_arms"]
    assert "pmrt_eq" not in v["C3"]["wording"]
    assert {d["arm"] for d in out["V3_paired"]} >= {"pmrt_eq", "corr"}
    md = markdown(out)
    assert "## V3 paired recall difference pmrt_nl_eq - arm" in md
    assert "C2a pmrt_nl_eq valid in R1 / R2" in md
    assert "pmrt_eq [linear statistic] (secondary PMRT arm" in md
    assert power.t1_seed_count(records, spec)["focal"] == "pmrt_nl_eq"
    assert power.t1_seed_count(records, spec, focal="pmrt_eq")["focal"] == "pmrt_eq"


def test_r4_real_actions_have_no_design() -> None:
    ds, _ = generate_dataset("E4", "R4", 500, 3_000_000, lam=1.0)
    designs = zip(ds.action_names, ds.designs, strict=True)
    real = [d.kind for a, d in designs if a not in (PLACEBO, PLACEBO_CONF)]
    assert real
    assert set(real) == {"none"}
    for regime in ("R1", "R2", "R3"):
        ds, _ = generate_dataset("E4", regime, 500, 3_000_000, lam=1.0)
        kinds = {d.kind for a, d in zip(ds.action_names, ds.designs, strict=True) if a == "P0"}
        assert "none" not in kinds


def test_numpy_scalars_are_cleaned() -> None:
    tree = {"a": np.float64(1.5), "b": [np.int64(2), math.nan, math.inf], "c": np.bool_(True)}
    assert clean(tree) == {"a": 1.5, "b": [2, None, None], "c": True}

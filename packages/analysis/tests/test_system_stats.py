import json
import math
from pathlib import Path

import numpy as np
import pytest

from oran_causal.analysis.system import stats, study2, study3
from oran_causal.analysis.system.stats import (
    JobFilter,
    alias_fill,
    arm_stats,
    check_gt_map,
    ci,
    config_match,
    dumps,
    gate_a,
    load_jobs,
    platform_key,
    ratio,
    sum_counts,
)

GOLDEN = Path(__file__).parent / "data" / "system_stats_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def records(g: dict) -> stats.Records:
    return {int(s): arms for s, arms in g["records"].items()}


@pytest.mark.claim("DEF-MET-DEC")
def test_gate_a_matches_old_arm_stats_multi(golden: dict) -> None:
    st = gate_a(
        records(golden), ["MG:PMRT", "MG:a", "MG:b", "MG:c", "never_sleep"], golden["n_boot"]
    )
    assert st.seeds == golden["gate_a"]["seeds"]
    assert (st.V_AA, st.V_ref, st.ref_arm, st.den) == tuple(
        golden["gate_a"][k] for k in ("V_AA", "V_ref", "ref_arm", "den")
    )
    assert as_json(st.point) == golden["point"]
    for a, c in golden["ci"].items():
        b = st.boot[a]
        assert [ci(b["R"]), ci(b["Rstar"]), ci(b["retention"])] == [
            c["R"],
            c["Rstar"],
            c["retention"],
        ]
        assert {k: ci(v) for k, v in b["guard"].items()} == c["guard"]


def as_json(o: object) -> object:
    return json.loads(json.dumps(o))


def test_d_checks_match_old_code(golden: dict) -> None:
    st = gate_a(
        records(golden), ["MG:PMRT", "MG:a", "MG:b", "MG:c", "never_sleep"], golden["n_boot"]
    )
    assoc = ["MG:a", "MG:b", "MG:c"]
    sig = {"MG:PMRT": "p", "MG:a": "x", "MG:b": "y", "MG:c": "x"}
    assert as_json(study2.d1_check(st.point, st.boot, "MG:PMRT", assoc)) == golden["d1"]
    assert as_json(study2.d2_check(st.boot, "MG:PMRT", assoc, sig)) == golden["d2"]
    assert as_json(study3.d3_check(st.point, st.boot, "MG:PMRT")) == golden["d3"]
    assert as_json(study3.s1_check(st.boot, "MG:PMRT")) == golden["s1"]
    assert sum_counts(records(golden), st.seeds, "MG:PMRT", "defer") == golden["counts"]


def test_arm_stats_matches_old_kill_test(golden: dict) -> None:
    out = arm_stats(records(golden), "MG:a", ("MG:PMRT",), n_boot=300)
    assert as_json(out) == golden["ka"]


def test_arm_stats_none_without_common_seed() -> None:
    assert arm_stats({1: {"noarb": {}}}, "MG:a") is None
    assert gate_a({1: {"noarb": {}}}, ["MG:a"]).n == 0


@pytest.mark.claim("DEF-MET-DEC")
def test_reference_arms_anchor_r(golden: dict) -> None:
    st = gate_a(records(golden), [], n_boot=0)
    assert st.point["noarb"]["R"] == 0.0
    assert st.point[st.ref_arm]["R"] == pytest.approx(1.0)
    assert st.point["freeze"]["retention"] == 0.0
    assert st.point["sub:ES+PowerES"]["retention"] == 1.0
    assert not st.boot


def test_ratio_conventions() -> None:
    assert ratio(2.0, 4.0) == 0.5
    assert ratio(3.0, 0.0) == math.inf
    assert ratio(0.0, 0.0) == 1.0
    np.testing.assert_array_equal(
        ratio(np.array([1.0, 0.0]), np.array([0.0, 0.0])), [math.inf, 1.0]
    )


def test_ci_skips_non_finite() -> None:
    assert ci([np.nan, np.inf]) == [None, None]
    lo, hi = ci([*range(101), np.nan])
    assert (lo, hi) == (5.0, 95.0)


def test_sum_counts_ignores_non_numeric() -> None:
    R = {
        1: {"a": {"policy_counts": {"defer": {"x": 2, "y": "n/a"}}}},
        2: {"a": {"policy_counts": {"defer": {"x": 3, "w": 1}}}},
    }
    assert sum_counts(R, [1, 2], "a", "defer") == {"w": 1, "x": 5}
    assert sum_counts(R, [1], "a", "policy") == {}


def test_alias_fill_copies_target() -> None:
    R = {1: {"MG:a": {"arm": "MG:a", "v": 1}}}
    alias_fill(R, {"MG:b": "MG:a", "MG:a": "MG:a", "MG:c": "MG:z"})
    assert R[1]["MG:b"] == {"arm": "MG:b", "v": 1, "alias_of": "MG:a"}
    assert "MG:c" not in R[1]


def test_load_jobs_filters_stage_schema_smoke(tmp_path: Path) -> None:
    rows = [
        {"kind": "header", "driver": "e6p_conf", "conf_stage": "eval"},
        {"kind": "header", "driver": "other", "conf_stage": "eval"},
        {
            "kind": "job",
            "schema": study2.SCHEMA_EVAL,
            "sub": "conf",
            "conf_stage": "eval",
            "seed": 3,
            "arm": "a",
        },
        {
            "kind": "job",
            "schema": study2.SCHEMA_EVAL,
            "sub": "conf",
            "conf_stage": "dev",
            "seed": 3,
            "arm": "b",
        },
        {
            "kind": "job",
            "schema": study2.SCHEMA_EVAL,
            "sub": "conf",
            "conf_stage": "eval",
            "seed": 4,
            "arm": "a",
            "smoke": True,
        },
    ]
    path = tmp_path / "res_0.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    R, heads = load_jobs([path], study2.EVAL_FILTER)
    assert list(R) == [3]
    assert list(R[3]) == ["a"]
    assert len(heads) == 1
    R, heads = load_jobs(
        [path], JobFilter(study2.SCHEMA_EVAL, "conf", "conf_stage", "eval", allow_smoke=True)
    )
    assert sorted(R) == [3, 4]
    assert len(heads) == 2
    assert study3.job_filter("eval").stage_key == "cs_stage"


def test_config_match_flags_repro_and_jobs() -> None:
    head = {
        "consts": {"load_factor": 1.75, "pair": "P3", "stratum": 3},
        "scenario_version": 1,
        "e6p_spec_commit": "c",
        "frozen_sha256": "f",
        "state": {"load_factor": 1.75},
        "numeric_env": {"numpy": "2", "scipy": "1", "platform": "Linux"},
    }
    job = dict.fromkeys(stats.SUM_FIELDS, 1.0) | {
        "load_factor": 1.75,
        "warmup_s": 120,
        "episode_s": 720,
    }
    R = {7: {"noarb": job, "noarb_repro": dict(job, rlf=2.0)}}
    out = config_match(R, [head], [head], (7,))
    assert all(out[k]["ok"] for k in stats._HEADER_FIELDS)
    assert out["noarb_repro_max_abs_diff"] == {"per_seed": {7: 1.0}, "ok": False}
    assert out["jobs"]["ok"]
    assert config_match(R, [], [head], ())["load_factor"]["ok"] is None


def test_check_gt_map_and_platform_key() -> None:
    cells = [{"family": "sleep", "relation": "nbr", "kpi": "pv", "status": "TRUE", "mean": 9.0}]
    assert check_gt_map(cells, [("sleep", "nbr", "pv", "TRUE", 9.0)])
    assert not check_gt_map(cells, [("sleep", "nbr", "pv", "NULL", 9.0)])
    env = {"numpy": "2.4.2", "platform": "Linux-6.12-x86_64-with-glibc2.35", "simd": ["AVX2"]}
    assert platform_key(env) == ("2.4.2", None, "Linux", "glibc2.35", None, ("AVX2",))
    assert platform_key(None) == (None, None, "", None, None, ())


def test_dumps_joins_tuple_keys_and_keeps_nan() -> None:
    text = dumps({("a", "b"): (1, np.float64(0.5)), "n": float("nan"), "z": np.int64(2)})
    assert text == '{\n "a|b": [\n  1,\n  0.5\n ],\n "n": NaN,\n "z": 2\n}'


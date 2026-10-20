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

VIOLATIONS = {"noarb": 90.0, "freeze": 60.0, "sub:ES+PowerES": 40.0, "sub:ES": 50.0}
VIOLATIONS |= {"sub:PowerES": 30.0, "sub:SliceGuarantee": 70.0}
ENERGY = {"freeze": 3.5e6, "sub:ES+PowerES": 2.0e6}


def records(seeds: int = 4) -> stats.Records:
    """Anchor records whose protected violations and energy differ by arm."""
    return {
        s: {
            a: {
                "prot_viol": v + s,
                "prot_ue_s": 900.0,
                "energy_j": ENERGY.get(a, 2.5e6),
                "viol_ue_s": 2e4,
                "ue_s": 1.2e5,
                "nonprot_embb_viol": 1.8e4,
                "ll_viol": 0.0,
                "rlf": 40.0,
            }
            for a, v in VIOLATIONS.items()
        }
        for s in range(seeds)
    }


def test_arm_stats_none_without_common_seed() -> None:
    assert arm_stats({1: {"noarb": {}}}, "MG:a") is None
    assert gate_a({1: {"noarb": {}}}, ["MG:a"]).n == 0


@pytest.mark.claim("DEF-MET-DEC")
def test_reference_arms_anchor_r() -> None:
    st = gate_a(records(), [], n_boot=0)
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


def _pooled_arms(**arm: float) -> dict[str, dict[str, float]]:
    guards = dict.fromkeys(stats.GUARD_KEYS, 10.0)
    return {
        "freeze": {"V": 5.0, "E": 100.0, **guards},
        "noarb": {"V": 10.0, "E": 100.0, **guards},
        stats.A_ARM: {"V": 5.0, "E": 0.0, **guards},
        "arm": {"V": 4.0, "E": 10.0, **guards, **arm},
    }


@pytest.mark.claim("DEF-SYS-MITIG")
def test_eligibility_needs_retention_and_every_guard_within_the_ratio() -> None:
    def eligible(**arm: float) -> dict:
        P = _pooled_arms(**arm)
        return stats._eligibility(P["arm"], P)

    assert (stats.X_MATCH, stats.GUARD) == (0.90, 1.10)
    assert eligible()["eligible"]
    assert eligible(E=10.0)["matched"]
    assert not eligible(E=10.5)["matched"]
    for key in stats.GUARD_KEYS:
        assert eligible(**{key: 11.0})["eligible"]
        out = eligible(**{key: 11.5})
        assert out["matched"]
        assert not out["guard_ok"]
        assert not out["eligible"]
        assert out["guard_ratio"][key] == pytest.approx(1.15)


@pytest.mark.claim("DEF-SYS-MITIG")
def test_an_ineligible_arm_cannot_score_above_zero_recovery() -> None:
    P = _pooled_arms()
    den = 5.0
    good = stats._point(P["arm"], P, den, 1)
    assert good["eligible"]
    assert good["R"] == good["Rstar"] == pytest.approx(1.2)
    P = _pooled_arms(rlf=12.0)
    bad = stats._point(P["arm"], P, den, 1)
    assert not bad["eligible"]
    assert bad["R"] == pytest.approx(1.2)
    assert bad["Rstar"] == 0.0

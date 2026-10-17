import json
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.system import report_data, study1, study2, study3
from oran_causal.analysis.system.stats import ANCHORS, alias_fill, dumps, load_jobs

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "data" / "records"
ARTIFACTS = ROOT / "packages" / "experiments" / "artifacts"
S1 = RECORDS / "study1_raw"
S2_DISC = RECORDS / "study2_raw" / "e6p-conf-discan-2" / "out" / "disc_conf.json"
RUNS = RECORDS / "study2_3_decision" / "runs"
FD = RECORDS / "system_artifacts" / "report_2026-10-02" / "figures_data.json"
LEGACY_TIES = (
    "count for MSCR+; undefined MSCR+ F1",
    "count for PMRT; undefined PMRT F1",
)

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (S1, S2_DISC, RUNS, FD)),
    reason="frozen Studies 1-3 records not fetched",
)


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def study1_reports() -> dict[str, tuple[dict, str, str]]:
    out = {}
    for run in ("e6p-disc-v4an-1", "e6p-disc-v4an-pmrt-1"):
        d = S1 / run / "out"
        raw = text(d / "analysis_v4.json")
        out[run] = (study1.recompute(json.loads(raw)), raw, text(d / "verdict_v4.json"))
    return out


@pytest.fixture(scope="module")
def disc() -> tuple[dict, str]:
    raw = text(S2_DISC)
    return study2.recompute_disc(json.loads(raw)), raw


def evaluate(study: Any, run: str, artifact: str, out: str) -> tuple[dict, str, str]:
    stored_dir = RUNS / run.replace("eval", "evalan") / "out"
    raw = text(stored_dir / f"eval_{out}.json")
    stored = json.loads(raw)
    art = json.loads(text(ARTIFACTS / artifact))
    files = sorted((RUNS / run).glob("*/res_*.jsonl"), key=lambda p: p.as_posix())
    filt = study2.EVAL_FILTER if study is study2 else study3.job_filter("eval")
    R, heads = load_jobs(files, filt)
    alias_fill(R, art["alias_of"])
    disc_rep = json.loads(text(S2_DISC))
    if study is study2:
        inp = study2.EvalInputs(disc_rep, art, stored["maps_verify"], True, stored["argv"])
    else:
        inp = study3.EvalInputs(disc_rep, art, stored["artifact_verify"], True, stored["argv"])
    rep = study.evaluate(R, heads, inp) | {"timing_s": stored["timing_s"]}
    return rep, raw, text(stored_dir / f"verdict_{out}.json")


@pytest.fixture(scope="module")
def conf_eval() -> tuple[dict, str, str]:
    return evaluate(study2, "e6p-conf-eval-1", "E6P_CONF_MAPS.json", "conf")


@pytest.fixture(scope="module")
def cs_eval() -> tuple[dict, str, str]:
    return evaluate(study3, "e6p-cs-eval-1", "E6P_CERTSAFE.json", "certsafe")


@pytest.fixture(scope="module")
def figures() -> tuple[dict, str]:
    data = report_data.build()
    return data, report_data.figures_json(data)


@pytest.fixture(scope="module")
def fd(figures: tuple[dict, str]) -> dict:
    return json.loads(figures[1])


def test_study1_pmrt_analysis_and_verdict_bit_identical(study1_reports: dict) -> None:
    rep, raw, verdict = study1_reports["e6p-disc-v4an-pmrt-1"]
    assert dumps(rep) == raw
    assert dumps(study1.verdict_record(rep)) == verdict


def test_study1_frozen_label_run_identical_up_to_rename(study1_reports: dict) -> None:
    rep, raw, verdict = study1_reports["e6p-disc-v4an-1"]
    assert raw.count("MSCR+") == 2
    assert dumps(rep) == raw.replace(*LEGACY_TIES)
    assert dumps(study1.verdict_record(rep)) == verdict.replace(*LEGACY_TIES)


def test_study2_discovery_criteria_bit_identical(disc: tuple[dict, str]) -> None:
    rep, raw = disc
    assert dumps(rep) == raw


def test_study2_eval_and_verdict_bit_identical(conf_eval: tuple[dict, str, str]) -> None:
    rep, raw, verdict = conf_eval
    assert dumps(rep) == raw
    assert dumps(study2.verdict_record(rep)) == verdict


def test_study3_eval_and_verdict_bit_identical(cs_eval: tuple[dict, str, str]) -> None:
    rep, raw, verdict = cs_eval
    assert dumps(rep) == raw
    assert dumps(study3.verdict_record(rep)) == verdict


def test_figures_data_bit_identical(figures: tuple[dict, str]) -> None:
    data, txt = figures
    assert txt == text(FD)
    assert len(data["assertions"]) == 55


def test_figures_data_written_bytes_equal_record(tmp_path: Path) -> None:
    out = tmp_path / "figures_data.json"
    report_data.write(out)
    assert out.read_bytes() == FD.read_bytes()
    report_data.write(out, newline="\n")
    assert out.read_bytes() == FD.read_bytes().replace(b"\r\n", b"\n")


def test_study1_partial_fails_only_p2_against_granger_by(study1_reports: dict) -> None:
    rep = study1_reports["e6p-disc-v4an-pmrt-1"][0]
    assert rep["verdict"]["verdict"] == "PARTIAL"
    assert rep["mode"] == "full"
    passed = {k: rep[k]["pass"] for k in ("K0", "K0n", "G", "K1", "P1", "P2", "P3", "S")}
    assert passed == dict.fromkeys(passed, True) | {"P2": False}
    per = rep["P2"]["per_baseline"]
    assert [b for b, v in per.items() if not v["ok"]] == ["granger_by"]
    assert (per["granger_by"]["wins60"], per["granger_by"]["wins120"]) == (2, 4)


def test_study2_discovery_pass_and_referee_not_eligible(
    disc: tuple[dict, str], conf_eval: tuple[dict, str, str]
) -> None:
    assert disc[0]["label"]["label"] == "DISC-PASS"
    rep = conf_eval[0]
    assert rep["verdict"]["verdict"] == "NOT ELIGIBLE"
    assert rep["verdict"]["complete"]
    pm = rep["arms"]["MG:PMRT"]
    assert not pm["eligible"]
    assert pm["matched"]
    assert pm["guard_ratio"]["rlf"] > 1.10


def test_study3_partial_and_gated_pmrt_is_never_sleep(
    cs_eval: tuple[dict, str, str], fd: dict
) -> None:
    rep = cs_eval[0]
    assert rep["verdict"]["verdict"] == "PARTIAL"
    assert rep["E"]["pass"]
    assert [rep[k]["pass"] for k in ("D1", "D2", "D3")] == [True, True, False]
    assert rep["D3"]["delta3"] == 0.0
    assert rep["D3"]["ci90"] == [0.0, 0.0]
    e = fd["certsafe"]["eval"]
    assert e["identical_seeds"] == e["n"] == 160


@pytest.mark.claim("EMP-S1-PLACEBO")
@pytest.mark.claim("EMP-S1-NULLOUT")
@pytest.mark.claim("EMP-GT-LABELS")
def test_claims_study1_validity(fd: dict) -> None:
    v = fd["v4"]
    assert v["K0"] == {
        "m": 60,
        "n_reject": 7,
        "binom_tail": 0.0296941,
        "n_declared": 0,
        "pass": True,
    }
    assert (v["K0n"]["pass"], v["K0n"]["rate05"], v["K0n"]["n_variants_with_decl"]) == (
        True,
        0.0525,
        0,
    )
    assert v["gt"]["counts"] == {"TRUE": 29, "NULL": 21, "INDET": 10}


@pytest.mark.claim("EMP-S1-MAP")
@pytest.mark.claim("EMP-S1-CHAIN")
@pytest.mark.claim("EMP-S1-P2")
def test_claims_study1_map(fd: dict) -> None:
    v = fd["v4"]
    p = v["pooled"]["loadsp_c+wby1s"]
    assert (p["n_declared"], p["ov_precision"], p["sign_acc"]) == (28, 0.92, 1.0)
    assert sorted(v["chain_z_pooled"]) == sorted("|".join(c[:3]) for c in study1.CHAIN)
    assert v["P3_values"]["chain_hits"] == 4
    assert v["verdict"] == "PARTIAL"


@pytest.mark.claim("EMP-S2-PLACEBO")
@pytest.mark.claim("EMP-S2-MAP")
@pytest.mark.claim("EMP-BASE-MAPS")
def test_claims_study2_maps(fd: dict) -> None:
    a = fd["option_a"]
    assert a["disc"]["K0"] == {
        "pass": True,
        "n_reject": 0,
        "m": 60,
        "n_declared": 0,
        "min_p": 0.0639,
    }
    assert (a["disc"]["K0n"]["rate05"], a["disc"]["K0n"]["n_variants_with_decl"]) == (0.05, 0)
    mq = a["map_quality"]
    assert mq["MG:PMRT"] == {
        "edges": 22,
        "true_right": 18,
        "true_wrong": 0,
        "null": 3,
        "indet": 1,
        "unknown": 0,
        "placebo_edges": 0,
    }
    assert (mq["MG:granger_by"]["edges"], mq["MG:granger_by"]["placebo_edges"]) == (47, 29)
    assert max(fd["v4"]["placebo_declarations"].values()) == 13


@pytest.mark.claim("EMP-S2-REFEREE")
@pytest.mark.claim("EMP-S2-MECH")
@pytest.mark.claim("EMP-S2-ASSOC")
def test_claims_study2_referee(fd: dict) -> None:
    e = fd["option_a"]["eval"]
    pm = e["arms"]["MG:PMRT"]
    assert round(pm["R"], 2) == 0.47
    assert round(pm["guard_ratio"]["rlf"], 2) == 1.31
    assert [round(x, 2) for x in pm["guard_ratio_ci90"]["rlf"]] == [1.10, 1.56]
    assert e["never_sleep"] == {"dR": 0.2485038, "ci90": [0.1385655, 0.3618438]}
    referees = {a: v["R"] for a, v in e["arms"].items() if a not in ANCHORS}
    assert max(referees, key=referees.__getitem__) == "MG:PMRT"
    assert fd["option_a"]["ptx_nbr_rlf"]["primary_p"] == 0.4872
    assert e["ptx_up_defer_units"] == {"MG:PMRT": 8674, "MG:GT": 2370}
    ratios = []
    for study, prefix in (("option_a", "MG:"), ("certsafe", "CS:")):
        arms = fd[study]["eval"]["arms"]
        v_aa = fd[study]["eval"]["V_AA"]
        ratios += [
            (round(arms[f"{prefix}{b}"]["V"] / v_aa, 1), arms[f"{prefix}{b}"]["eligible"])
            for b in study2.ASSOC
            if arms[f"{prefix}{b}"]["V"] > 2 * v_aa
        ]
    assert (min(r for r, _ in ratios), max(r for r, _ in ratios)) == (4.5, 5.4)
    assert sum(el for _, el in ratios) >= 2


@pytest.mark.claim("EMP-S3-GATED")
@pytest.mark.claim("EMP-S3-ORACLE")
def test_claims_study3(fd: dict) -> None:
    e = fd["certsafe"]["eval"]
    assert e["D3"] == {"pass": False, "delta3": 0.0, "lb90": 0.0, "ci90": [0.0, 0.0]}
    pm = e["arms"]["CS:PMRT"]
    assert (round(pm["R"], 2), round(pm["guard_ratio"]["rlf"], 2), pm["eligible"]) == (
        0.26,
        0.96,
        True,
    )
    gt = e["secondary"]["CS:GT"]
    assert round(-gt["dR"], 2) == -0.04
    assert [round(-x, 2) for x in reversed(gt["ci90"])] == [-0.10, 0.03]
    retention = (e["arms"]["CS:GT"]["retention"], e["arms"]["never_sleep"]["retention"])
    assert tuple(round(x, 2) for x in retention) == (1.29, 1.15)

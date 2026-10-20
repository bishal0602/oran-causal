from pathlib import Path

import pytest

from oran_causal.analysis import paper_values as pv

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "data" / "records"
NEEDED = (
    "system_artifacts/decision/v4_analysis.json",
    "posthoc_x9_x16/revision/x14_existence.json",
    "study4_eval/eval/report.json",
    "study4_eval/eval/csv/recall_vs_n.csv",
    "bridging/exp_b/exp_b_tables.json",
    "posthoc_x8/x8/summary.json",
    "posthoc_x4/x4/x4_tables.json",
    "posthoc_x2_x3/extras_tables.json",
    "study1_raw/e6p-disc-v4an-pmrt-1/out/analysis_v4.json",
    "study2_raw/e6p-conf-discan-2/out/disc_conf.json",
)
REVISION = "shap_rerun/overlay/posthoc_x9_x16/revision"

pytestmark = pytest.mark.skipif(
    not all((RECORDS / name).exists() for name in NEEDED), reason="frozen records not fetched"
)


@pytest.fixture(scope="module")
def records():
    return pv.Records(ROOT)


@pytest.fixture(scope="module")
def resolve(records):
    specs = {spec.key: spec for spec in pv.load_spec(ROOT / "spec" / "values.toml")}
    return lambda key: pv.resolve(specs[key], records)


@pytest.fixture(scope="module")
def value(resolve):
    return lambda key: resolve(key).value


@pytest.fixture(scope="module")
def text(resolve):
    return lambda key: resolve(key).text


@pytest.mark.claim("EMP-S1-SMALLNULL")
def test_small_effect_null_rejection_rates(value) -> None:
    assert (value("s1.null.count"), value("s1.null.open.count")) == (21, 12)
    assert value("s1.null.rate") == pytest.approx(32 / 210)
    assert value("s1.null.open.rate") == pytest.approx(0.075)


@pytest.mark.claim("EMP-S4-CDD")
def test_cdl_placebo_check_passes_but_nine_conditions_exceed_the_null_rate(value) -> None:
    report = pv.Records(ROOT).document("study4_eval/eval/report.json")
    cdl = [row for row in report["tables"]["V1_validity"] if row["arm"] == "cdl"]
    assert (value("s4.cdl.invalid"), value("s4.cdl.conditions")) == (9, 30)
    assert not [
        row for row in cdl if row["plac_decl"] and row["plac_decl"]["validity"] == "INVALID"
    ]
    invalid = [row for row in cdl if row["validity"] == "INVALID"]
    assert all(row["null_decl"]["validity"] == "INVALID" for row in invalid)
    assert all(row["null_decl"]["ci"][0] > 0.05 for row in invalid)


@pytest.mark.claim("EMP-POWER")
def test_latent_confounding_recall_and_randomised_pooled_recall(value) -> None:
    assert value("s4.e4.pmrtgbm.recall_max") <= 0.075
    assert value("s4.e4.pcorr_eq.recall_24k") == pytest.approx(0.90)
    assert value("br.pcorr_eq.recall") == pytest.approx(21 / 29)
    assert value("br.pcorr_native.recall") == pytest.approx(20 / 29)
    assert value("s1.map.pmrt.rec") == pytest.approx(23 / 29)


@pytest.mark.claim("EMP-S1-STRICT")
def test_strict_null_rerun_declares_in_two_variants_within_the_limit(value, text) -> None:
    counts = (value("x8.redraws"), value("x8.n_variants"), value("x8.variants_with_decl"))
    assert counts == (9999, 590, 2)
    assert value("x8.variants_with_decl") / value("x8.n_variants") <= value("x8.tol_pct")
    assert text("x8.share.1e3") == r"7.9 \times 10^{-4}"
    assert text("x8.share.2e4") == r"1.1 \times 10^{-4}"
    assert value("x8.share.1e3") <= value("x8.lim.1e3")
    assert value("x8.share.2e4") <= value("x8.lim.2e4")


@pytest.mark.claim("EMP-GT-SENS")
def test_label_sensitivity_moves_pmrt_precision_and_few_labels(records, value) -> None:
    x11 = records.document(f"{REVISION}/x11_gt_sensitivity.json")["studies"]
    precision = {
        (study, variant): round(x11[study]["variants"][variant]["methods"][method]["precision"], 2)
        for study, method in (("S1", "loadsp_c+wby1s"), ("S2", "MG:PMRT"))
        for variant in ("delta_x0.5", "delta_x2")
    }
    assert precision == {
        ("S1", "delta_x0.5"): 1.0,
        ("S2", "delta_x0.5"): 1.0,
        ("S1", "delta_x2"): 0.59,
        ("S2", "delta_x2"): 0.73,
    }
    assert (value("x11.delta_half.changed"), value("x11.delta_dbl.changed")) == (10, 16)
    assert value("x11.other.changed_max") == 2


@pytest.mark.claim("EMP-GT-EXIST")
def test_pmrt_declarations_exist_with_the_declared_sign(records) -> None:
    x14 = records.document(f"{REVISION}/x14_existence.json")["studies"]
    pooled = [
        x14[study]["levels"]["0.95"]["methods"][method]["pooled"]
        for study, method in (("S1", "loadsp_c+wby1s"), ("S2", "MG:PMRT"))
    ]
    assert [(p["ex_confirmed"], p["mat_n_declared"]) for p in pooled] == [(26, 28), (22, 22)]
    assert all(p["ex_wrong_sign"] == 0 for p in pooled)
    s1 = records.document("study1_raw/e6p-disc-v4an-pmrt-1/out/analysis_v4.json")
    s2 = records.document("study2_raw/e6p-conf-discan-2/out/disc_conf.json")
    declared = {
        "S1": [
            h for h, _ in s1["slices_result"]["pooled"]["metrics"]["loadsp_c+wby1s"]["declared"]
        ],
        "S2": ["|".join(edge[:3]) for edge in s2["maps"]["MG:PMRT"]],
    }
    remote = {}
    for study, hypotheses in declared.items():
        cells = x14[study]["levels"]["0.95"]["cells"]
        labels = [cells[h]["material"] for h in hypotheses if h.split("|")[1] in ("nbr", "far")]
        remote[study] = (labels.count("TRUE"), len(labels))
    assert remote == {"S1": (12, 13), "S2": (7, 7)}


@pytest.mark.claim("EMP-S3-GATEDIAG")
def test_gate_withholds_transmit_power_up_on_true_harmful_guard_edges(records) -> None:
    x10 = records.document("posthoc_x9_x16/revision/x10_gate_diag.json")
    classes = {row["class"]: row for row in x10["classes"]}
    assert [name for name, row in classes.items() if row["certified"]] == ["sleep|1"]
    assert classes["sleep|1"]["deferred_units_eval"] == 899
    assert classes["ptx|1"]["withheld_units_eval"] == 2579
    guards = [e for e in x10["edges"] if e["class"] == "ptx|1" and e["status"] == "UNRESOLVED"]
    assert len(guards) == 2
    assert all(e["gt_label"] == "TRUE" and e["gt_harm_beyond_tau"] for e in guards)
    figures = records.document(
        "shap_rerun/overlay/system_artifacts/report_2026-10-02/figures_data.json"
    )
    bound = figures["certsafe"]["pmrt_bound_ptx_nbr_rlf"]
    assert (round(bound["ub90_defer_up"], 3), round(bound["tau"], 3)) == (0.148, 0.036)
    assert bound["ub90_defer_up"] > bound["tau"]


@pytest.mark.claim("EMP-X15")
def test_ungated_pmrt_referee_gain_shrinks_on_the_gated_seeds(records, value) -> None:
    x15 = records.document("posthoc_x9_x16/revision/x15_replication.json")
    gated = x15["frozen_summary"]["study3"]["arms"]["MG:PMRT"]
    assert (round(gated["dR_vs_never_sleep"], 2), round(gated["rlf_ratio"], 2)) == (-0.06, 1.05)
    assert (round(value("x15.pooled.dr"), 2), round(value("x15.pooled.rlf"), 2)) == (0.10, 1.18)
    assert x15["Q4"]["dR_MG:PMRT-never_sleep"]["ci90"][0] > 0


@pytest.mark.claim("EMP-X16")
def test_strict_centre_keeps_every_map_sign_and_every_certification(records, text) -> None:
    x16 = records.document("posthoc_x9_x16/revision/x16_centre.json")
    assert len(x16["maps"]) == 22
    assert all(row["sign_agree_strict"] for row in x16["maps"])
    assert (text("x16.map.rel_min"), text("x16.map.rel_max")) == ("$-2.9$", "$+6.5$")
    assert text("x16.gate.median_pct") == "3.5"
    assert x16["certification"]["status_changes"] == []


@pytest.mark.claim("EMP-X4")
def test_random_referee_has_negative_recovery(records) -> None:
    rows = records.document("posthoc_x4/x4/x4_tables.json")["rows"]
    assert rows["X4:rand@0.352"]["R"] < 0
    assert all(row["R"] < 0 for arm, row in rows.items() if arm.startswith("X4:rand"))


@pytest.mark.claim("EMP-S4-C1")
def test_design_blind_tests_lose_validity_under_a_dithered_setpoint(records, text) -> None:
    c1 = records.document("study4_eval/eval/report.json")["tables"]["V4_verdicts"]["C1"]
    assert c1["verdict"] == "SUPPORTED"
    failed = [arm for arm, row in c1["arms"].items() if row["verdict"] == "FAILURE"]
    assert len(failed) == c1["n_failures"] == 5
    for arm in failed:
        assert c1["arms"][arm]["pooled_r1"]["null_raw"]["validity"] == "VALID"
        assert c1["arms"][arm]["pooled_r2"]["null_raw"]["validity"] == "INVALID"
    assert (text("s4.c1.r2_min"), text("s4.c1.r2_max")) == ("0.33", "0.94")


@pytest.mark.claim("EMP-S4-C2C3")
def test_design_based_tests_keep_validity_with_named_exceptions(records) -> None:
    tables = records.document("study4_eval/eval/report.json")["tables"]
    verdicts = tables["V4_verdicts"]
    assert verdicts["claim"] == "SUPPORTED"
    assert verdicts["components"] == dict.fromkeys(("C1", "C2a", "C2b", "C3"), "SUPPORTED")
    assert verdicts["C2a"]["pooled_validity"] == verdicts["C3"]["pooled_validity"] == "VALID"
    assert [row["arm"] for row in verdicts["C2_named"]] == ["rcot2_eq"]
    lin = [
        row["conf_raw"]["validity"]
        for row in tables["V6_E4"]
        if row["arm"] == "pmrt_eq" and row["regime"] == "R3" and row["conf_raw"]
    ]
    assert "INVALID" in lin


@pytest.mark.claim("EMP-S4-E4")
def test_correlation_fails_under_logged_confounding_while_pmrt_gbm_holds(text) -> None:
    assert text("s4.conf.corr_min") == "1.00"
    assert (text("s4.conf.gbm_min"), text("s4.conf.gbm_max")) == ("0.02", "0.06")
    assert text("s4.conf.pcorr_all") == "0.056"


@pytest.mark.claim("EMP-X3")
def test_misspecified_design_breaks_pmrt_validity(records) -> None:
    curves = records.document("posthoc_x2_x3/extras_tables.json")["x3_curve"]
    first = {(row["base_arm"], row["family"]): row["first_invalid"] for row in curves}
    assert first["pmrt_eq", "width narrower"] == "width x0.8"
    assert first["pmrt_nl_eq", "width narrower"] == "width x0.8"
    pmrt = {arm for arm, _ in first if arm.startswith("pmrt")}
    assert pmrt == {"pmrt_eq", "pmrt_nl_eq", "pmrt_r3"}
    assert all(first[arm, "lambda higher"] == "lam x2" for arm in pmrt)

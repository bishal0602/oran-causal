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
)

pytestmark = pytest.mark.skipif(
    not all((RECORDS / name).exists() for name in NEEDED), reason="frozen records not fetched"
)


@pytest.fixture(scope="module")
def value():
    specs = {spec.key: spec for spec in pv.load_spec(ROOT / "spec" / "values.toml")}
    records = pv.Records(ROOT)
    return lambda key: pv.resolve(specs[key], records).value


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

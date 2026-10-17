from pathlib import Path

import pytest

from oran_causal.analysis.system import report_data, shap_splice
from oran_causal.experiments.system import shap_overlay as so

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "data" / "records"
OVERLAY = so.overlay_root(RECORDS)
HAVE_RECORDS = (RECORDS / "shap_rerun" / "study1" / "study1.json").exists() and (
    RECORDS / "study2_3_decision" / "runs"
).exists()


def test_map_quality_text_layout() -> None:
    mq = {
        "MG:a@dev": {
            "edges": 12,
            "true_right": 8,
            "true_wrong": 1,
            "null": 2,
            "indet": 1,
            "placebo_edges": 3,
        },
        "MG:PMRT": {
            "edges": 5,
            "true_right": 5,
            "true_wrong": 0,
            "null": 0,
            "indet": 0,
            "placebo_edges": None,
        },
    }
    lines = so.map_quality_text(mq).splitlines()
    assert lines[0].startswith("arm")
    assert lines[0].endswith("| placebo-map edges")
    assert lines[1] == "MG:a@dev                12     8     1    2     1 | 3"
    assert lines[2] == "MG:PMRT                  5     5     0    0     0 | -"


@pytest.mark.skipif(not HAVE_RECORDS, reason="shap_rerun and Studies 1-3 records not fetched")
def test_map_quality_text_reproduces_the_recorded_listing() -> None:
    fd = report_data.build()
    recorded = (RECORDS / so.DECISION / "conf_map_quality.txt").read_text(encoding="utf-8")
    assert so.map_quality_text(fd["option_a"]["map_quality"]) == recorded


@pytest.mark.slow
@pytest.mark.skipif(
    not HAVE_RECORDS or not OVERLAY.exists(), reason="shap_rerun overlay records not fetched"
)
def test_write_reproduces_the_stored_overlay(tmp_path: Path) -> None:
    files = so.write(RECORDS, out=tmp_path)
    assert {p.relative_to(tmp_path).as_posix() for p in files} == shap_splice.OVERLAY_FILES
    for path in files:
        rel = path.relative_to(tmp_path)
        assert path.read_bytes() == (OVERLAY / rel).read_bytes(), rel.as_posix()

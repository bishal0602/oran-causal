from pathlib import Path

from oran_causal.experiments.system import shap_overlay as so

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "data" / "records"
OVERLAY = so.overlay_root(RECORDS)


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

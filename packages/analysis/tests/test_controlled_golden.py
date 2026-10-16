import gzip
import json
from pathlib import Path

import pytest

from oran_causal.analysis.controlled.fmax import load_dependence
from oran_causal.analysis.controlled.integrity import IntegrityInputs
from oran_causal.analysis.controlled.report import analyse
from oran_causal.analysis.controlled.tables import markdown, to_json
from oran_causal.analysis.records import load_records

DATA = Path(__file__).parent / "data"
FREEZE_COMMIT = "93856c230d84abda2ff3b894218dd6cb0ecd47f9"

pytestmark = [
    pytest.mark.claim("EMP-S4-C1"),
    pytest.mark.claim("EMP-S4-C2C3"),
    pytest.mark.claim("EMP-S4-E4"),
]


@pytest.fixture(scope="module")
def tables() -> dict:
    spec = json.loads((DATA / "eval_subset_spec.json").read_text(encoding="utf-8"))
    inputs = IntegrityInputs(
        freeze_commit=FREEZE_COMMIT,
        protocol={"ok": True},
        dep=load_dependence(DATA / "dependence.json"),
    )
    return analyse(load_records(DATA / "eval_subset_records.jsonl.gz"), spec, inputs)


def test_tables_equal_the_frozen_analysis_output_on_real_records(tables: dict) -> None:
    expected = gzip.decompress((DATA / "expected_tables.json.gz").read_bytes()).decode("utf-8")
    assert to_json(tables) == expected


def test_markdown_equals_the_frozen_analysis_output_on_real_records(tables: dict) -> None:
    expected = gzip.decompress((DATA / "expected_tables.md.gz").read_bytes()).decode("utf-8")
    assert markdown(tables) == expected

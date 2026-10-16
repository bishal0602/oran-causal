import json
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.controlled.report import ReportRequest, run_report
from oran_causal.analysis.records import file_sha256

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "data" / "records"
EVAL = RECORDS / "study4_eval" / "eval"
DEV = RECORDS / "study4_dev"
SPECS = ROOT / "packages" / "experiments" / "specs"
PROTOCOL = ROOT / "packages" / "experiments" / "protocols" / "study4" / "PROTOCOL_A.md"
FREEZE_COMMIT = "93856c230d84abda2ff3b894218dd6cb0ecd47f9"
FROZEN_UTC = "2026-10-05T07:33:00Z"
SPEC_LABEL = "scratchpad/xmethod/specs/eval/full.json"
MERGED_LABEL = "scratchpad/xmethod/results/eval/merged.jsonl.gz"
MAX_DIFFS = 20

pytestmark = [
    pytest.mark.slow,
    pytest.mark.claim("EMP-S4-C1"),
    pytest.mark.claim("EMP-S4-C2C3"),
    pytest.mark.claim("EMP-S4-E4"),
    pytest.mark.skipif(
        not (EVAL / "merged.jsonl.gz").exists(), reason="frozen Study 4 EVAL records not fetched"
    ),
]


def json_diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Lists up to `MAX_DIFFS` paths where two JSON trees differ."""
    if type(a) is not type(b):
        return [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key}: missing on one side")
            else:
                out += json_diff(a[key], b[key], f"{path}/{key}")
            if len(out) >= MAX_DIFFS:
                break
        return out[:MAX_DIFFS]
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += json_diff(x, y, f"{path}[{i}]")
            if len(out) >= MAX_DIFFS:
                break
        return out[:MAX_DIFFS]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def regenerate(out: Path) -> int:
    dev_merged = [
        DEV / "dev" / name / "merged.jsonl.gz" for name in ("full", "ci_c", "pmrt_nl", "cdl")
    ]
    request = ReportRequest(
        spec=SPECS / "eval" / "full.json",
        merged=[EVAL / "merged.jsonl.gz"],
        out_dir=out,
        freeze_commit=FREEZE_COMMIT,
        amendments=SPECS / "eval" / "amendments.json",
        protocol=PROTOCOL,
        dependence=DEV / "fmax_sim" / "dependence.json",
        dev_merged=dev_merged,
        generated_utc=FROZEN_UTC,
        input_labels={
            str(SPECS / "eval" / "full.json"): SPEC_LABEL,
            str(EVAL / "merged.jsonl.gz"): MERGED_LABEL,
        },
    )
    return run_report(request, lambda _: None)


def compare_report_json(regenerated: Path) -> list[str]:
    got = json.loads((regenerated / "report.json").read_text(encoding="utf-8"))
    frozen = json.loads((EVAL / "report.json").read_text(encoding="utf-8"))
    problems = [f"report.json {d}" for d in json_diff(got, frozen)]
    if (regenerated / "report.json").read_bytes() != (EVAL / "report.json").read_bytes():
        problems.append("report.json differs byte-wise")
    verdicts, expected = got["tables"]["V4_verdicts"], frozen["tables"]["V4_verdicts"]
    for component in ("C1", "C2a", "C2b", "C3", "components", "claim"):
        if verdicts[component] != expected[component]:
            problems.append(f"verdict {component} differs")
    return problems


def compare_files(regenerated: Path) -> list[str]:
    problems = []
    frozen_csv = sorted(p.name for p in (EVAL / "csv").iterdir())
    if sorted(p.name for p in (regenerated / "csv").iterdir()) != frozen_csv:
        problems.append("csv file set differs")
    for name in frozen_csv:
        if (regenerated / "csv" / name).read_bytes() != (EVAL / "csv" / name).read_bytes():
            problems.append(f"csv/{name} differs")
    if (regenerated / "REPORT.md").read_bytes() != (EVAL / "REPORT.md").read_bytes():
        problems.append("REPORT.md differs")
    return problems


def test_the_frozen_inputs_are_the_ones_the_report_recorded() -> None:
    frozen = json.loads((EVAL / "report.json").read_text(encoding="utf-8"))["meta"]
    shas = dict(frozen["inputs"])
    assert file_sha256(SPECS / "eval" / "full.json") == shas[SPEC_LABEL]
    assert file_sha256(EVAL / "merged.jsonl.gz") == shas[MERGED_LABEL]


def test_regenerated_study4_eval_report_equals_the_frozen_one(tmp_path: Path) -> None:
    assert regenerate(tmp_path) == 0
    problems = [*compare_report_json(tmp_path), *compare_files(tmp_path)]
    assert problems == []

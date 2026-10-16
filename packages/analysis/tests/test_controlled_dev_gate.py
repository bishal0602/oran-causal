import json
from pathlib import Path

import pytest

from oran_causal.analysis.controlled.report import ReportRequest, run_report

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "data" / "records" / "study4_dev"
FROZEN = DEV / "eval_dev"
INPUTS = Path(__file__).parent / "data" / "dev_gate"
FROZEN_UTC = "2026-10-04T06:15:24Z"
WORKTREE = "D:/academia/major-project/CDD-ORAN-wt"
LABELS = {
    "eval_full_old.json": f"{WORKTREE}/xm-harness/scratchpad/xmethod/specs/eval/full.json",
    "dev_full.json": f"{WORKTREE}/xm-pmrt/scratchpad/xmethod/specs/dev/full.json",
    "dev_ci_c.json": f"{WORKTREE}/xm-pmrt/scratchpad/xmethod/specs/dev/ci_c.json",
    "dev_pmrt_nl.json": f"{WORKTREE}/xm-pmrt/scratchpad/xmethod/specs/dev/pmrt_nl.json",
}
RUNS = ("full", "ci_c", "pmrt_nl")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (FROZEN / "report.json").exists(), reason="frozen Study 4 DEV records not fetched"
    ),
]


def labels() -> dict[str, str]:
    out = {str(INPUTS / name): label for name, label in LABELS.items()}
    for run in RUNS:
        merged = DEV / "dev" / run / "merged.jsonl.gz"
        out[str(merged)] = (
            f"{WORKTREE}/xm-pmrt/scratchpad/xmethod/results/dev/{run}/merged.jsonl.gz"
        )
    return out


def test_regenerated_dev_report_equals_the_frozen_rehearsal(tmp_path: Path) -> None:
    request = ReportRequest(
        spec=INPUTS / "eval_full_old.json",
        merged=[DEV / "dev" / run / "merged.jsonl.gz" for run in RUNS],
        out_dir=tmp_path,
        dev=True,
        dev_specs=[INPUTS / f"dev_{run}.json" for run in RUNS],
        protocol=INPUTS / "PROTOCOL_A_old.md",
        dependence=DEV / "fmax_sim" / "dependence.json",
        generated_utc=FROZEN_UTC,
        input_labels=labels(),
    )
    assert run_report(request, lambda _: None) == 2
    problems = []
    got = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    frozen = json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))
    if got["tables"] != frozen["tables"]:
        problems.append("report.json tables differ")
    if (tmp_path / "report.json").read_bytes() != (FROZEN / "report.json").read_bytes():
        problems.append("report.json differs byte-wise")
    if (tmp_path / "REPORT.md").read_bytes() != (FROZEN / "REPORT.md").read_bytes():
        problems.append("REPORT.md differs")
    for path in sorted((FROZEN / "csv").iterdir()):
        if (tmp_path / "csv" / path.name).read_bytes() != path.read_bytes():
            problems.append(f"csv/{path.name} differs")
    assert problems == []

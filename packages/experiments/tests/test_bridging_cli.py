import json
from pathlib import Path

import pytest

from oran_causal.analysis.bridging import costs, tables
from oran_causal.experiments.bridging import __main__ as B
from oran_causal.experiments.paths import SPECS, WORKSPACE_ROOT

SPEC = SPECS / "exp_b" / "exp_b.json"
EXP_B = WORKSPACE_ROOT / "data" / "records" / "bridging" / "exp_b"
V4 = WORKSPACE_ROOT / "data" / "records" / "study1_raw" / "e6p-disc-v4an-pmrt-1" / "out"
needs_records = pytest.mark.skipif(
    not (EXP_B / "merged.jsonl.gz").exists(), reason="bridging records not present"
)


def test_list_counts_the_spec_units() -> None:
    lines: list[str] = []
    assert B.main(["list", "--spec", str(SPEC)], lines.append) == 0
    summary = json.loads(lines[0])
    assert summary["n_units"] == 1328
    assert sum(summary["per_arm"].values()) == 1328


@pytest.mark.skipif(not (V4 / "analysis_v4.json").exists(), reason="Study 1 records not present")
def test_counts_from_v4_match_the_frozen_counts(tmp_path: Path) -> None:
    out = tmp_path / "counts.json"
    argv = ["counts", "--spec", str(SPEC), "--v4-analysis", str(V4 / "analysis_v4.json")]
    argv += ["--check", str(SPECS / "exp_b" / "counts_v4.json"), "--out", str(out)]
    lines: list[str] = []
    assert B.main(argv, lines.append) == 0
    assert json.loads(lines[0])["mismatches"] == {}


def test_counts_mismatch_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(B.run, "counts_from_v4", lambda spec, analysis: {"60": {"sleep": 1}})
    v4, ref, out = tmp_path / "v4.json", tmp_path / "ref.json", tmp_path / "c.json"
    v4.write_text("{}", encoding="utf-8")
    ref.write_text(json.dumps({"counts": {"60": {"sleep": 9}}}), encoding="utf-8")
    argv = ["counts", "--spec", str(SPEC), "--v4-analysis", str(v4), "--check", str(ref)]
    assert B.main([*argv, "--out", str(out)], lambda _: None) == B.COUNTS_MISMATCH
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["check"]["mismatches"] == {"60": [{"sleep": 9}, {"sleep": 1}]}


@needs_records
def test_table_and_costs_commands_rebuild_the_stored_documents(tmp_path: Path) -> None:
    argv = ["table", "--spec", str(SPEC), "--merged", str(EXP_B / "merged.jsonl.gz")]
    argv += ["--v4-analysis", str(V4 / "analysis_v4.json"), "--out", str(tmp_path)]
    assert B.main(argv) == 0
    factors = "factors_vps_py3.12.13.json"
    argv = ["costs", "--spec", str(SPEC), "--merged", str(EXP_B / "merged.jsonl.gz")]
    argv += ["--factors", str(EXP_B / factors), "--out", str(tmp_path)]
    argv += ["--label", f"scratchpad/xmethod/results/exp_b/{factors}"]
    assert B.main(argv, lambda _: None) == 0
    names = (tables.TABLES_JSON, tables.TABLES_MD, costs.COSTS_JSON, costs.COSTS_MD)
    for name in names:
        assert (tmp_path / name).read_bytes() == (EXP_B / name).read_bytes(), name

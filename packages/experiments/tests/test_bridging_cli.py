import json
from pathlib import Path

import pytest

from oran_causal.experiments.bridging import __main__ as B
from oran_causal.experiments.paths import SPECS

SPEC = SPECS / "bridging" / "exp_b.json"


def test_list_counts_the_spec_units() -> None:
    lines: list[str] = []
    assert B.main(["list", "--spec", str(SPEC)], lines.append) == 0
    summary = json.loads(lines[0])
    assert summary["n_units"] == 1328
    assert sum(summary["per_arm"].values()) == 1328


def test_counts_mismatch_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(B.run, "counts_from_v4", lambda spec, analysis: {"60": {"sleep": 1}})
    v4, ref, out = tmp_path / "v4.json", tmp_path / "ref.json", tmp_path / "c.json"
    v4.write_text("{}", encoding="utf-8")
    ref.write_text(json.dumps({"counts": {"60": {"sleep": 9}}}), encoding="utf-8")
    argv = ["counts", "--spec", str(SPEC), "--v4-analysis", str(v4), "--check", str(ref)]
    assert B.main([*argv, "--out", str(out)], lambda _: None) == B.COUNTS_MISMATCH
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["check"]["mismatches"] == {"60": [{"sleep": 9}, {"sleep": 1}]}

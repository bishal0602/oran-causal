"""T0 gates of the post hoc re-analyses that need the referee functions of the methods member.

X10 and X12 rebuild their stored outputs exactly. X16 rebuilds the DISC unit table from the raw
records and recomputes every slope under both centres; the stored file was made with Python 3.12
and numpy 2.4, so its floats are compared to a relative 1e-9 and everything else exactly.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.posthoc import x10, x12, x16
from oran_causal.experiments.posthoc.tools import referee_tools

RECORDS = Path(__file__).resolve().parents[3] / "data" / "records"
REV = RECORDS / "posthoc_x9_x16" / "revision"

pytestmark = pytest.mark.skipif(not RECORDS.exists(), reason="frozen record sets not present")


def lf(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def same_file(mine: Path, stored: Path) -> bool:
    return lf(mine) == lf(stored)


def test_referee_tools_are_the_methods_functions():
    t = referee_tools()
    assert t.guard_kpis == ("v", "rlf")
    assert t.relations == ("own", "nbr", "far")
    assert t.levels.tolist() == [1.0, 1.0, 0.0, 1.0]
    assert t.map_from_json([["f", "own", "k", None], ["g", "nbr", "k", 2]]) == {
        ("f", "own", "k"): None,
        ("g", "nbr", "k"): 2.0,
    }


@pytest.mark.claim("EMP-S3-GATEDIAG")
def test_x10_gate_diagnosis_is_rebuilt_exactly(tmp_path):
    out = x10.analyse(RECORDS, x10.ARTIFACTS, referee_tools())
    x10.write(out, tmp_path)
    for name in ("x10_gate_diag.json", "x10_gate_diag.csv"):
        assert same_file(tmp_path / name, REV / name)
    assert out["result"]["repro"]["ok"]
    assert {c["class"] for c in out["result"]["classes"] if c["certified"]} == {"sleep|1"}


def test_x10_stops_when_the_certification_is_not_reproduced(tmp_path):
    tools = referee_tools()

    def broken(m, bounds, tau):
        res = tools.certify_map(m, bounds, tau)
        res["uncertified"] = []
        return res

    with pytest.raises(x10.ReproductionError, match="not reproduced"):
        x10.analyse(RECORDS, x10.ARTIFACTS, tools, certify=broken)


def test_x12_thresholds_are_rebuilt_exactly(tmp_path):
    out = x12.run(RECORDS, tools=referee_tools())
    x12.write(out, tmp_path)
    for name in ("x12_threshold.json", "x12_threshold.csv"):
        assert same_file(tmp_path / name, REV / name)
    assert out["repro"]["ok"]
    assert len(out["rows"]) == 14


def _close(a: Any, b: Any, path: str = "", rtol: float = 1e-9) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in _close(a[k], b[k], f"{path}/{k}", rtol)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _close(x, y, f"{path}[{i}]", rtol)
        ]
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return []
        return [] if abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.mark.slow
@pytest.mark.claim("EMP-X16")
def test_x16_centre_is_rebuilt(tmp_path):
    out = x16.run(RECORDS, tools=referee_tools(), cache_dir=tmp_path / "cache")
    x16.write(out, tmp_path)
    mine = json.loads(lf(tmp_path / "x16_centre.json"))
    stored = json.loads(lf(REV / "x16_centre.json"))
    skip = ("sha256", "env", "timing_s", "data")
    diff = _close(
        {k: v for k, v in mine.items() if k not in skip},
        {k: v for k, v in stored.items() if k not in skip},
    )
    assert diff == []
    assert mine["check"]["ok"]
    assert mine["check"]["n_units"] == 79499
    assert mine["certification"]["status_changes"] == []
    assert mine["certification"]["signature_equal"]
    assert mine["data"]["files"][0] == "study2_raw/e6p-conf-disc-1-a/res_0.jsonl"

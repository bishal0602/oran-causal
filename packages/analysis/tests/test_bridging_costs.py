import json
from pathlib import Path

import pytest

from oran_causal.analysis.bridging import costs as C

EXP_B = Path(__file__).resolve().parents[3] / "data" / "records" / "bridging" / "exp_b"
FACTORS = "factors_vps_py3.12.13.json"
LABEL = f"scratchpad/xmethod/results/exp_b/{FACTORS}"


def _rec(arm: str, platform: str, cpu: float, status: str = "ok") -> dict:
    key = f"{arm}|E6|sleep|k0|n10|s1"
    return {
        "key": key,
        "arm": arm,
        "status": status,
        "cpu_s": cpu,
        "host": {"platform": platform, "cpu_model": "m"},
    }


def test_factors_and_budget(tmp_path: Path) -> None:
    f = tmp_path / "factors.json"
    f.write_text(json.dumps({"factors": {"b": {"vps|m": {"f": 2.0}}, "*": {"vps|m": {"f": 3.0}}}}))
    recs = [
        _rec("b", "kaggle", 10.0),
        _rec("b", "vps", 10.0),
        _rec("a", "vps", 10.0),
        _rec("a", "vps", 5000.0, "infeasible"),
    ]
    res = C.build(recs, f, 100)
    assert list(res["arms"]) == ["a", "b"]
    assert res["arms"]["b"]["ref_cpu_s"] == 30.0
    assert res["arms"]["b"]["factor"] == {"vps|m": {"from": "b", "f": 2.0}}
    assert res["arms"]["a"]["ref_cpu_s"] == 130.0
    assert res["arms"]["a"]["max_ref_cpu_s"] == 100.0
    assert res["arms"]["a"]["infeasible"] == 1
    assert res["total_ref_core_h"] == 160.0 / 3600


@pytest.mark.skipif(not (EXP_B / "merged.jsonl.gz").exists(), reason="bridging records not present")
def test_costs_rebuild_the_stored_costs(tmp_path: Path) -> None:
    C.costs(EXP_B / "merged.jsonl.gz", EXP_B / FACTORS, 7200, tmp_path, label=LABEL)
    for name in (C.COSTS_JSON, C.COSTS_MD):
        assert (tmp_path / name).read_bytes() == (EXP_B / name).read_bytes(), name

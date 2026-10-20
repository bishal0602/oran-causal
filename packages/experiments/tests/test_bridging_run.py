import json
from pathlib import Path

import pytest

from oran_causal.analysis.bridging.plan import spec_slices
from oran_causal.experiments.bridging import run as R
from oran_causal.experiments.paths import SPECS


def _spec(name: str) -> dict:
    return R.load_spec(SPECS / "bridging" / f"{name}.json")


def test_sub_spec_slices() -> None:
    pilot = _spec("exp_b_pilot")
    assert pilot["base_spec"] == "packages/experiments/specs/bridging/exp_b.json"
    assert [s.name for s in spec_slices(pilot)] == ["s60_0", "placebo"]
    rest = {s.name for s in spec_slices(_spec("exp_b_rest"))}
    assert len(rest) == 17
    assert not rest & {"s60_0", "placebo"}


@pytest.mark.parametrize(
    ("arms", "families", "match"),
    [
        ({"a|b": {"ref": "x", "declare": "by"}}, None, "needs 'ref'"),
        ({"a": {"ref": "x", "declare": "max"}}, None, "needs 'ref'"),
        ({"a": {"ref": "x", "declare": "tau"}}, ["sleep", "x"], "unknown families"),
    ],
)
def test_load_spec_rejects(tmp_path: Path, arms: dict, families: list | None, match: str) -> None:
    spec = {"arms": arms} | ({"families": families} if families else {})
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        R.load_spec(p)


def test_counts_from_v4_reports_missing_slices() -> None:
    spec = {"slices": ["60"], "arms": {}}
    with pytest.raises(ValueError, match="no unit counts"):
        R.counts_from_v4(spec, {"slices_result": {}})


def test_project_interpolates_log_log(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = {
        "slices": ["placebo"],
        "budget_cpu_s": 50,
        "counts": "unused",
        "arms": {"a": {"ref": "x", "declare": "by", "size_factor": 2}},
    }
    counts = {"placebo": {"carrier": 100, "sleep": 400, "ptx": 1600, "prot_min": 0}}
    dev = {"a": {"100": {"E1": [1.0]}, "400": {"E1": [4.0], "E2": [2.0]}}}
    monkeypatch.setattr(R, "spec_counts", lambda _spec: counts)
    res = R.project(spec, dev)
    t = res["cost_table"]
    assert t["a|E6|carrier|n100"] == pytest.approx(2.0)
    assert t["a|E6|sleep|n400"] == pytest.approx(8.0)
    assert t["a|E6|ptx|n1600"] == pytest.approx(32.0)
    assert res["n_units"] == 3
    assert res["projected_over_budget"] == []

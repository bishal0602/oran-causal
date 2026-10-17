import json
from pathlib import Path

import pytest

from oran_causal.analysis.bridging import plan
from oran_causal.analysis.bridging import tables as T

RECORDS = Path(__file__).resolve().parents[3] / "data" / "records"
EXP_B = RECORDS / "bridging" / "exp_b"
V4 = RECORDS / "study1_raw"
SPEC = Path(__file__).resolve().parents[2] / "experiments" / "specs" / "exp_b" / "exp_b.json"


def test_conformal_tau() -> None:
    assert T.conformal_tau([]) is None
    assert T.conformal_tau([None, float("nan")]) is None
    scores = list(range(1, 20))
    assert T.conformal_tau(scores) == 19.0
    assert T.conformal_tau(scores, level=0.5) == 10.0


def test_auroc() -> None:
    assert T.auroc([], [1.0]) is None
    assert T.auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert T.auroc([1.0], [1.0]) == 0.5
    assert T.auroc([2.0, None], [1.0, 3.0, float("nan")]) == 0.5


def test_sharp_proxy_and_hypotheses() -> None:
    gt = {
        ("sleep", "nbr", "pv"): {"status": "NULL", "ci": [-0.1, 0.2]},
        ("sleep", "nbr", "v"): {"status": "NULL", "ci": [0.1, 0.2]},
        ("sleep", "own", "pv"): {"status": "TRUE", "ci": [-1.0, 1.0]},
    }
    assert T.sharp_proxy(gt) == {("sleep", "nbr", "pv")}
    assert plan.hyp_of("A_sleep", "K_nbr_pv") == ("sleep", "nbr", "pv")
    assert plan.hyp_of(plan.PLACEBO, "K_far_load") is None


def test_slice_plan_and_seeds() -> None:
    names = [s.name for s in plan.slice_plan(("60", "pooled", "placebo"))]
    assert names[0] == "pooled"
    assert names[-1] == "placebo"
    assert len(names) == 12
    s = {x.name: x for x in plan.slice_plan(plan.DEFAULT_KINDS)}
    assert (s["s120_4"].split, s["s120_4"].seeds[0], len(s["s120_4"].seeds)) == (44, 188580, 120)
    assert plan.dataset_seed(s["s300_1"].split, "prot_min") == 60_000_613


@pytest.mark.skipif(not (EXP_B / "merged.jsonl.gz").exists(), reason="bridging records not present")
@pytest.mark.parametrize("v4", ["e6p-disc-v4an-pmrt-1", "e6p-disc-v4an-1"])
def test_tables_rebuild_the_stored_tables(tmp_path: Path, v4: str) -> None:
    analysis = V4 / v4 / "out" / "analysis_v4.json"
    if not analysis.exists():
        pytest.skip(f"{analysis} not present")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    v4a = json.loads(analysis.read_text(encoding="utf-8"))
    T.tables(spec, EXP_B / "merged.jsonl.gz", EXP_B / "gt_v4.json", tmp_path, v4a)
    for name in (T.TABLES_JSON, T.TABLES_MD):
        assert (tmp_path / name).read_bytes() == (EXP_B / name).read_bytes(), name

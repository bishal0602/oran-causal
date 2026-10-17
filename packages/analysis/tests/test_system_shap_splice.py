import copy
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.system import report_data, shap_splice
from oran_causal.analysis.system.stats import JobFilter

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "data" / "records"
FILT = JobFilter(schema="j/1", sub="t", stage_key="st", stage="eval")
SPLICE_PREFIXES = (
    "assertions",
    "option_a/maps_sha256",
    "option_a/disc_json_sha256",
    "option_a/eval/D2/hypotheses",
    "option_a/eval/run/cpu_h",
    "certsafe/artifact_sha256",
    "certsafe/cert/CS:qacm@",
    "certsafe/jobs",
    "certsafe/eval/D2/hypotheses",
    "certsafe/eval/run/cpu_h",
    "certsafe/eval/cpu_h_eval_stored",
)


def test_changed_jobs() -> None:
    old = {"jobs": ["A", "B"], "arms": {"A": {"signature": "s"}, "B": {"signature": "t"}}}
    new = {
        "jobs": ["A", "B", "C"],
        "arms": {"A": {"signature": "s"}, "B": {"signature": "u"}, "C": {"signature": "v"}},
    }
    assert shap_splice.changed_jobs(old, new) == ["B", "C"]
    assert shap_splice.changed_jobs(old, old) == []
    assert shap_splice.frozen_twins(old, new) == {}


def test_frozen_twins_take_the_frozen_job() -> None:
    old = {
        "jobs": ["A", "Q"],
        "arms": {"A": {"signature": ["s"]}, "Q": {"signature": ["q"]}, "S": {"signature": ["t"]}},
    }
    new = {
        "jobs": ["A", "S"],
        "arms": {"A": {"signature": ["s"]}, "Q": {"signature": ["q"]}, "S": {"signature": ["q"]}},
    }
    assert shap_splice.frozen_twins(old, new) == {"S": "Q"}
    assert shap_splice.changed_jobs(old, new) == []


def _analysis() -> dict[str, Any]:
    m = {"shap_gbdt": {"n_declared": 1}, "corr": {"n_declared": 3}}
    return {
        "slices_result": {
            "pooled": {"metrics": copy.deepcopy(m)},
            "s60_0": {"metrics": {"corr": {"n_declared": 2}}},
        },
        "placebo_descriptive": {"shap_gbdt": {"n_declared": 2}, "corr": {"n_declared": 3}},
        "desc_tau": {"shap_gbdt": {"tau": 0.02}, "corr": {"tau": 0.5}},
    }


def test_spliced_v4_replaces_only_the_shap_entries() -> None:
    A = _analysis()
    new = {
        "slices": {"pooled": {"n_declared": 17}},
        "placebo": {"n_declared": 8},
        "tau": {"tau": 1},
    }
    out = shap_splice.spliced_v4(A, {"methods": {"shap_gbdt": new}})
    assert out["slices_result"]["pooled"]["metrics"] == {
        "shap_gbdt": {"n_declared": 17},
        "corr": {"n_declared": 3},
    }
    assert out["slices_result"]["s60_0"] == A["slices_result"]["s60_0"]
    assert out["placebo_descriptive"]["shap_gbdt"] == {"n_declared": 8}
    assert out["desc_tau"] == {"shap_gbdt": {"tau": 1}, "corr": {"tau": 0.5}}
    assert _analysis() == A


def test_spliced_v4_refuses_other_slices() -> None:
    new = {"slices": {"s60_0": {}}, "placebo": {}, "tau": {}}
    with pytest.raises(ValueError, match="slices differ"):
        shap_splice.spliced_v4(_analysis(), {"methods": {"shap_gbdt": new}})


def _write(path: Path, seeds: list[int], arms: list[str], tag: str) -> Path:
    head = {
        "kind": "header",
        "st": "eval",
        "utc": f"2026-01-0{len(tag)}",
        "code": {"git_head": f"{tag}0000000", "e6_dirty": False},
        "platform": {"platform": tag},
    }
    jobs = [
        {
            "kind": "job",
            "schema": "j/1",
            "sub": "t",
            "st": "eval",
            "seed": s,
            "arm": a,
            "cpu_s": 360.0,
            "src": tag,
        }
        for s in seeds
        for a in arms
    ]
    path.write_text("\n".join(json.dumps(r) for r in [head, *jobs]) + "\n", encoding="utf-8")
    return path


def test_merged_records_replace_changed_jobs_and_reuse_twins(tmp_path: Path) -> None:
    sig = {"A": ["a"], "Q": ["q"], "S": ["s"], "B": ["a"]}
    old = {
        "jobs": ["A", "Q", "S"],
        "arms": {a: {"signature": v} for a, v in sig.items()},
        "alias_of": {"A": "A", "Q": "Q", "S": "S", "B": "A"},
    }
    new = {
        "jobs": ["A", "S", "T"],
        "arms": {a: {"signature": v} for a, v in (sig | {"S": ["s2"], "T": ["q"]}).items()},
        "alias_of": {"A": "A", "Q": "T", "S": "S", "T": "T", "B": "A"},
    }
    frozen = _write(tmp_path / "f.jsonl", [1, 2], ["P", "A", "Q", "S"], "frozen")
    rerun = _write(tmp_path / "n.jsonl", [1, 2], ["S", "A"], "new")
    m = shap_splice.merged_records([frozen], [rerun], FILT, old, new)
    assert (m.redo, m.twins) == (["S"], {"T": "Q"})
    for s in (1, 2):
        R = m.R[s]
        assert {a: R[a]["src"] for a in ("P", "A", "S", "T")} == {
            "P": "frozen",
            "A": "frozen",
            "S": "new",
            "T": "frozen",
        }
        assert R["T"]["arm"] == "Q"
        assert (R["Q"]["alias_of"], R["B"]["alias_of"]) == ("T", "A")
    assert m.ledger["n_jobs"] == 8
    assert m.ledger["cpu_h"] == pytest.approx(0.8)
    assert m.ledger["platform"] == ["frozen", "new"]
    assert len(m.heads) == 2


@pytest.mark.skipif(
    not (RECORDS / "shap_rerun" / "study1" / "study1.json").exists()
    or not (RECORDS / "study2_3_decision" / "runs").exists(),
    reason="shap_rerun and Studies 1-3 records not fetched",
)
def test_splice_fd_changes_only_shap_entries_and_keeps_the_verdicts() -> None:
    data = report_data.build()
    fd = shap_splice.splice_fd(data)
    old, new = (json.loads(report_data.figures_json(d)) for d in (data, fd))
    paths = ["/".join(map(str, p)) for p, _, _ in report_data.tree_diff(old, new)]
    assert paths
    assert all("shap_gbdt" in p or p.startswith(SPLICE_PREFIXES) for p in paths)
    assert new["option_a"]["eval"]["verdict"] == "NOT ELIGIBLE"
    assert new["certsafe"]["eval"]["verdict"] == "PARTIAL"
    assert new["v4"]["pooled"]["shap_gbdt"]["n_declared"] == 17
    assert new["v4"]["placebo_declarations"]["shap_gbdt"] == 8
    mq = new["option_a"]["map_quality"]["MG:shap_gbdt@plc"]
    assert (mq["edges"], mq["true_right"], mq["true_wrong"]) == (19, 14, 2)
    assert new["certsafe"]["eval"]["run"]["n_jobs"] == old["certsafe"]["eval"]["run"]["n_jobs"]
    assert len(new["assertions"]) == len(set(new["assertions"]))
    assert set(old["assertions"]) <= set(new["assertions"])


def test_overlaid_reads_the_overlay_copy_of_listed_files_only() -> None:
    rel = "system_artifacts/decision/cs_eval.json"
    assert shap_splice.overlaid(rel) == f"shap_rerun/overlay/{rel}"
    assert shap_splice.overlaid("posthoc_x8/x8/summary.json") == "posthoc_x8/x8/summary.json"


def test_values_aliases_equal_overlaid_recorded_paths() -> None:
    aliases = tomllib.loads((ROOT / "spec" / "values.toml").read_text(encoding="utf-8"))["records"]
    prefix = f"{shap_splice.OVERLAY}/"
    assert any(rel.startswith(prefix) for rel in aliases.values())
    for name, rel in aliases.items():
        assert rel == shap_splice.overlaid(rel.removeprefix(prefix)), name

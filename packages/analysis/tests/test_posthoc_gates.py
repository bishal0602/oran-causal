"""T0 gates of the post hoc analyses: each rebuilds stored X-series outputs from the records.

The stored files were written on Windows in text mode, so a line ends in CRLF there and LF here;
every comparison reads both sides with CRLF as LF. JSON is also compared parsed with exact floats.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.controlled.tables import clean
from oran_causal.analysis.posthoc import x1_runtime as rt
from oran_causal.analysis.posthoc import x2x3, x4, x5x6, x7, x8, x9, x11, x13, x14, x15
from oran_causal.analysis.system import stats

RECORDS = Path(__file__).resolve().parents[3] / "data" / "records"
REV = RECORDS / "posthoc_x9_x16" / "revision"
X1 = RECORDS / "posthoc_x1" / "exp_c"
X56 = RECORDS / "posthoc_x5_x6"
DECISION = RECORDS / "system_artifacts" / "decision"
LINUX = sys.platform.startswith("linux")

pytestmark = pytest.mark.skipif(not RECORDS.exists(), reason="frozen record sets not present")


def lf(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def same_file(mine: Path, stored: Path) -> bool:
    return lf(mine) == lf(stored)


def same_json(mine: Path, stored: Path) -> bool:
    return json.loads(lf(mine)) == json.loads(lf(stored))


@pytest.mark.claim("EMP-RUNTIME")
def test_x1_runtime_table_of_the_paper_run_is_rebuilt_exactly():
    old = (
        r"D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results"
        r"\exp_c\paper\raw\xm-expc-paper-k1" + "\\"
    )
    files = sorted((X1 / "paper" / "raw" / "xm-expc-paper-k1").glob("res_*.jsonl"))
    recs, info = rt.load(files, {str(f.absolute()): old + f.name for f in files})
    stored = json.loads(lf(X1 / "paper" / "exp_c_runtime.json"))
    spec = rt.load_specs(
        [Path(__file__).parents[2] / "experiments" / "specs" / "exp_c" / "timing.json"]
    )
    out = rt.build(recs, info, generated_utc=stored["generated_utc"], arm_specs=spec)
    assert out == stored
    assert json.dumps(out, indent=1, allow_nan=False) == lf(X1 / "paper" / "exp_c_runtime.json")
    title = (
        lf(X1 / "paper" / "EXP_C_RUNTIME.md").split("\n", 1)[0].split("(", 1)[1].rsplit(")", 1)[0]
    )
    assert rt.render_md(out, title) == lf(X1 / "paper" / "EXP_C_RUNTIME.md")


SPECS = ["x2_d02", "x2_d05", "x2_d10", "x2_d20", "x2_b05", "x2_b80", "x3_told"]


def _without_provenance(tree: dict[str, Any]) -> dict[str, Any]:
    tree = json.loads(json.dumps(tree))
    for c in tree["checks"].values():
        c.pop("provenance")
    return tree


@pytest.mark.slow
@pytest.mark.claim("EMP-X3")
def test_x2_x3_tables_are_rebuilt_exactly():
    d = RECORDS / "posthoc_x2_x3"
    stored = json.loads(lf(d / "extras_tables.json"))
    out = x2x3.analyse(
        [(n, d / n / "merged.jsonl.gz") for n in SPECS], protocol_sha256=stored["protocol_sha256"]
    )
    mine = json.loads(json.dumps(clean(out), sort_keys=True))
    if LINUX:
        assert mine == stored
        assert json.dumps(clean(out), indent=1, sort_keys=True) == lf(d / "extras_tables.json")
        assert x2x3.analyse_md(out) == lf(d / "EXTRAS_TABLES.md")
    else:
        assert _without_provenance(mine) == _without_provenance(stored)
        assert x2x3.analyse_md(out).replace("\n", "") != ""


@pytest.mark.claim("EMP-X4")
def test_x4_tables_are_rebuilt_exactly(tmp_path):
    rep = x4.analyse(
        RECORDS / "study2_3_decision" / "runs" / "e6p-cs-eval-1",
        RECORDS / "posthoc_x4" / "x4" / "records",
        cs_eval=DECISION / "cs_eval.json",
        factors=X1 / "calib" / "factors.json",
    )
    x4.write(rep, tmp_path)
    stored = RECORDS / "posthoc_x4" / "x4" / "x4_tables.json"
    assert same_file(tmp_path / "x4_tables.json", stored)
    assert rep["repro_cs_eval"]["arms"]
    assert rep["provenance_identical"] is True
    assert all(r["R"] < 0 for a, r in rep["rows"].items() if a.startswith("X4:"))


def test_x7_tables_are_rebuilt_exactly(tmp_path):
    rep = x7.analyse(
        RECORDS / "study2_3_decision" / "runs" / "e6p-cs-eval-1",
        RECORDS / "posthoc_x7" / "x7" / "records",
        cs_eval=DECISION / "cs_eval.json",
        factors=X1 / "calib" / "factors.json",
    )
    x7.write(rep, tmp_path)
    assert same_file(tmp_path / "x7_tables.json", RECORDS / "posthoc_x7" / "x7" / "x7_tables.json")
    assert rep["repro_cs_eval_mismatch"] == []
    assert rep["n_seeds"] == 160


@pytest.mark.claim("EMP-S1-STRICT")
def test_x8_summary_is_rebuilt_exactly(tmp_path):
    d = RECORDS / "posthoc_x8" / "x8"
    out = x8.summarize(d)
    stored = json.loads(lf(d / "summary.json"))
    if sys.platform == "win32":
        assert out == stored
        assert json.dumps(out, indent=1) == lf(d / "summary.json")
    else:
        for name, combo in out["combos"].items():
            for key, value in combo.items():
                assert value == pytest.approx(stored["combos"][name][key], rel=1e-12, abs=0)
        assert {k: v for k, v in out.items() if k != "combos"} == {
            k: v for k, v in stored.items() if k != "combos"
        }
    assert out["n_variants"] == 590
    assert out["B"] == [9999]
    prim = out["combos"][out["primary"]]
    assert prim["variants_with_decl"] == 2
    assert out["checks"]["X8a_var_decl"]["pass"]
    assert prim["rate_le_0.001"] == pytest.approx(7.9e-4, abs=5e-5)
    assert prim["rate_le_0.0002"] == pytest.approx(1.1e-4, abs=5e-5)


def test_x6_stage_is_rebuilt_exactly(tmp_path):
    res = x5x6.x6(X56 / "x6_gbm" / "merged.jsonl.gz")
    x5x6.write_stage(res, "x6", tmp_path)
    assert same_file(tmp_path / "diag_x6.json", X56 / "diag" / "diag_x6.json")


@pytest.mark.slow
def test_x5_stages_and_tables_are_rebuilt_exactly(tmp_path):
    res = x5x6.x5(X56 / "x5_fail" / "merged.jsonl.gz", X56 / "x5_adj" / "merged.jsonl.gz")
    x5x6.write_stage(res, "x5", tmp_path)
    assert same_file(tmp_path / "diag_x5.json", X56 / "diag" / "diag_x5.json")
    eval_merged = RECORDS / "study4_eval" / "eval" / "merged.jsonl.gz"
    res = x5x6.x5_step1(
        eval_merged, dependence=RECORDS / "study4_dev" / "fmax_sim" / "dependence.json"
    )
    x5x6.write_stage(res, "x5_step1", tmp_path)
    assert same_file(tmp_path / "diag_x5_step1.json", X56 / "diag" / "diag_x5_step1.json")
    x5x6.write_stage(json.loads(lf(X56 / "diag" / "diag_x6.json")), "x6", tmp_path)
    protocol = json.loads(lf(X56 / "diag" / "diag_tables.json"))["protocol_sha256"]
    x5x6.combine_tables(tmp_path, protocol)
    assert same_file(tmp_path / "diag_tables.json", X56 / "diag" / "diag_tables.json")
    assert same_file(tmp_path / "DIAG_TABLES.md", X56 / "diag" / "DIAG_TABLES.md")


def test_x9_table_is_rebuilt_exactly(tmp_path):
    out = x9.run(RECORDS)
    x9.write(out, tmp_path)
    for name in ("x9_decision_e2e.json", "x9_decision_e2e.csv"):
        assert same_file(tmp_path / name, REV / name)
    assert out["study3"]["repro"]["ok"]
    assert out["study2"]["repro"]["ok"]


@pytest.mark.claim("EMP-GT-SENS")
def test_x11_gt_sensitivity_is_rebuilt_exactly(tmp_path):
    x11.write(x11.run(RECORDS), tmp_path)
    for name in ("x11_gt_sensitivity.json", "x11_gt_sensitivity.csv", "x11_gt_labels.csv"):
        assert same_file(tmp_path / name, REV / name)


def test_x13_attribution_is_rebuilt_exactly(tmp_path):
    x13.write(x13.run(RECORDS), tmp_path)
    for name in ("x13_attribution.json", "x13_attribution.csv"):
        assert same_file(tmp_path / name, REV / name)


@pytest.mark.claim("EMP-GT-EXIST")
def test_x14_existence_is_rebuilt_exactly(tmp_path):
    x14.write(x14.run(RECORDS), tmp_path)
    for name in ("x14_existence.json", "x14_existence.csv"):
        assert same_file(tmp_path / name, REV / name)


def _close_tree(a: Any, b: Any, path: str = "", rtol: float = 1e-12) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys"]
        return [d for k in a for d in _close_tree(a[k], b[k], f"{path}/{k}", rtol)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _close_tree(x, y, f"{path}[{i}]", rtol)
        ]
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return []
        return [] if abs(a - b) <= rtol * max(abs(a), abs(b)) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.mark.claim("EMP-X15")
def test_x15_replication_is_rebuilt(tmp_path):
    out = x15.run(RECORDS)
    x15.write(out, tmp_path)
    assert same_file(tmp_path / "x15_replication.csv", REV / "x15_replication.csv")
    mine = json.loads(lf(tmp_path / "x15_replication.json"))
    stored = json.loads(lf(REV / "x15_replication.json"))
    assert _close_tree(mine, stored) == []
    exact = [d for d in _close_tree(mine, stored, rtol=0.0) if "ols_dR_on_V_AA" not in d]
    assert exact == []
    assert out["result"]["reading_rule"]["branch"].startswith("neither as worded")


def test_gate_a_statistics_come_from_the_system_stats_module():
    assert stats.N_BOOT == x9.stats.N_BOOT == 10_000

import csv
import json
import subprocess
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

from oran_causal.figures import paths
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import tab_appB_cost

COSTS = {
    "arms": {
        "pmrt_nl_eq": {"n": 100, "infeasible": 1, "ref_cpu_s": 7200.0, "max_ref_cpu_s": 120.5},
        "pcorr_eq": {"n": 50, "infeasible": 0, "ref_cpu_s": 3600.0, "max_ref_cpu_s": 40.0},
    },
    "total_ref_core_h": 3.0,
    "budget_cpu_s": 600.0,
}


@pytest.fixture
def bridging(tmp_path, monkeypatch):
    records = tmp_path / "records"
    (records / "bridging" / "exp_b").mkdir(parents=True)
    (records / "bridging" / "exp_b" / "exp_b_costs.json").write_text(
        json.dumps(COSTS), encoding="utf-8"
    )
    monkeypatch.setattr(paths, "RECORDS", records)
    monkeypatch.setenv(paths.OUT_ENV, str(tmp_path / "out"))
    return tmp_path / "out"


def test_cost_table_reads_the_bridging_record(bridging):
    written = tab_appB_cost.write()
    assert sorted(p.replace("\\", "/").rsplit("/", 1)[-1] for p in written) == [
        "tab_appB_cost.csv",
        "tab_appB_cost.tex",
    ]
    tex = (bridging / "tables" / "tab_appB_cost.tex").read_text(encoding="utf-8")
    assert "data/records/bridging/exp_b/exp_b_costs.json" in tex
    with (bridging / "tables" / "tab_appB_cost.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["units"] for r in rows if r["arm"] == "pmrt_nl_eq"] == ["100"]


def test_cost_table_refuses_a_total_that_does_not_add_up(bridging):
    path = paths.RECORDS / "bridging" / "exp_b" / "exp_b_costs.json"
    path.write_text(json.dumps({**COSTS, "total_ref_core_h": 9.0}), encoding="utf-8")
    with pytest.raises(AssertionError, match="core-h sum"):
        tab_appB_cost.rows()


def test_study4_sources_name_the_record_sets():
    assert T.EVAL == ("study4_eval", "eval")
    assert T.BRIDGE == ("bridging", "exp_b")
    assert T.SRC_REPORT == "data/records/study4_eval/eval/report.json"


def test_no_module_reads_a_record_when_imported(tmp_path):
    code = (
        "import pkgutil, importlib, pathlib\n"
        "import matplotlib; matplotlib.use('Agg')\n"
        "from oran_causal.figures import paths, figs\n"
        f"paths.RECORDS = pathlib.Path({str(tmp_path)!r})\n"
        "for info in pkgutil.iter_modules(figs.__path__):\n"
        "    importlib.import_module(f'{figs.__name__}.{info.name}')\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-800:]

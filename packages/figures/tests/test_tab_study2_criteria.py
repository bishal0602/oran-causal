"""tab_study2_criteria: the Study 2 criteria table (replaces tab_design_criteria as the home of
Study 2's criteria). Checks the row set and order (protocol s.7.1-7.2), the format shared with
tab_study1_criteria / tab_study3_criteria, and that a disagreement between FD and the analyzer
outputs stops the build.
"""

import copy

import pytest

from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import tab_study1_criteria as T1
from oran_causal.figures.figs import tab_study2_criteria as T2

ORDER = [
    "K0",
    "K0n",
    "G",
    "K1",
    "X1",
    "X2",
    "X3",
    "Verdict",
    "K0, K0n",
    "Elig",
    "D1",
    "D2",
    "R",
    "Verdict",
]


pytestmark = pytest.mark.usefixtures("records")


@pytest.fixture(scope="module")
def rows():
    return T2.rows()


def test_rows_in_protocol_order(rows):
    assert [r["criterion"] for r in rows] == ORDER


def test_results_and_labels(rows):
    allowed = {
        "PASS",
        "FAIL",
        "--",
        "PARTIAL",
        "INVALID",
        "NO-CHAIN",
        "UNDERPOWERED",
        "NOT ELIGIBLE",
    }
    assert all(r["result"] in allowed for r in rows)
    assert not any(
        "DISC-" in r["rule"] + r["result"] for r in rows
    )  # discovery labels printed without prefix
    assert rows[-1]["result"] == ST.fd()["option_a"]["eval"]["verdict"]
    assert not any(
        r["criterion"] in ("E", "S", "S1") for r in rows
    )  # paper labels (TM.crit), not protocol


def test_latex_format_matches_study1(rows):
    lines = T2.latex(rows)
    head = [x for x in lines if not x.startswith("%")][:5]
    assert head == [x for x in T1.latex(T1.rows()) if not x.startswith("%")][:5]
    assert lines[-2:] == [r"\bottomrule", r"\end{tabular}"]
    assert sum(x.startswith("Verdict & \\multicolumn{2}") for x in lines) == 2


def test_disagreement_stops_the_build(monkeypatch):
    fd = copy.deepcopy(ST.fd())
    fd["option_a"]["disc"]["X2"]["n_true"] += 1
    monkeypatch.setattr(ST, "fd", lambda: fd)
    with pytest.raises(AssertionError, match="disagree"):
        T2.rows()

"""tab_decision_all: the ESM table of every decision arm. Checks the row count per block, that every
stored arm key is a row once, plain names only, that a wrong printed value stops the build, and that
every row of the manuscript's tab:decision_full appears with identical cells.
"""

import copy
import re

import pytest

from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import tab_decision_all as T

pytestmark = pytest.mark.usefixtures("records")


@pytest.fixture(scope="module")
def built():
    rs, meta = T.rows()
    return rs, meta, T.latex(rs, meta)


def test_row_counts(built):
    rs, _, tex = built
    assert sum(r["study"] == "study2" for r in rs) == 26
    assert sum(r["study"] == "study3" for r in rs) == 29  # 26 arms + 3 random rejection
    assert sum(x.startswith(r"\quad ") for x in tex.splitlines()) == 55
    assert tex.count(r"\begin{tabular}") == 2


def test_every_stored_arm_once(built):
    rs = built[0]
    for study, key in T.FD_KEY.items():
        arms = [r["arm"] for r in rs if r["study"] == study and r["source"] != "X4"]
        assert sorted(arms) == sorted(ST.fd()[key]["eval"]["arms"])


def test_plain_names_only(built):
    body = "\n".join(x for x in built[2].splitlines() if not x.startswith("%"))
    for bad in (
        "Study",
        "MG:",
        "CS:",
        "DEV",
        "@dev",
        "@plc",
        "sub:",
        "blanket",
        "B2",
        "noarb",
        "dagger",
        "_",
    ):
        assert bad not in body, bad


def test_wrong_value_stops_the_build(built):
    rs, meta, _ = built
    bad = copy.deepcopy(rs)
    next(r for r in bad if r["arm"] == "freeze")["R"] += 0.01
    with pytest.raises(AssertionError):
        T.check(bad, meta)


def _manuscript_rows(journal) -> list[list[str]]:
    """Cells of every row of tab:decision_full in the manuscript (read-only)."""
    txt = (journal / "sections" / "app_c.tex").read_text(encoding="utf-8")
    body = txt.split(r"\label{tab:decision_full}", 1)[1].split(r"\bottomrule", 1)[0]
    return [
        [c.strip() for c in x.strip().removesuffix(r"\\").split(" & ")[1:]]
        for x in body.splitlines()
        if x.strip().startswith(r"\quad ")
    ]


def test_manuscript_rows_identical(built, journal):
    rows = _manuscript_rows(journal)
    assert len(rows) >= 20
    cells = [T.cells(r) for r in built[0]]
    missing = [c for c in rows if c not in cells]
    assert not missing, missing


def test_marks_follow_aliases(built):
    for x in built[2].splitlines():
        m = re.search(r"\$\^\{\\mathrm\{([ab])\}\}\$", x)
        if m and m.group(1) == "b":
            assert x.startswith(r"\quad QACM")

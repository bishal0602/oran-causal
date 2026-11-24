import json

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from oran_causal.analysis.posthoc import x18 as X
from oran_causal.figures import checks, methods, paths, style
from oran_causal.figures.figs import fig_appC_x18_guard as F


def _row(r: float, eligible: bool) -> dict:
    return {"R": r, "R_ci90": [r - 0.06, r + 0.05], "eligible": eligible}


def _doc(pending_at: float | None = None) -> dict:
    coupled = {}
    for i, limit in enumerate(X.L_GRID):
        k = f"{X.cell_key(limit, F.E_SHOW)}|{X.key(X.rho_of(limit))}"
        c = {
            "best_static": "never_sleep",
            "arms": {
                "never_sleep": _row(0.26, True),
                "CS:PMRT": _row(0.26 + 0.03 * i, i < 4),
                "CS:GT": _row(0.22 + 0.02 * i, True),
            },
        }
        if limit == pending_at:
            c["pending"] = ["CS:PMRT"]
            del c["arms"]["CS:PMRT"]
        coupled[k] = c
    return {"grid": {"L": list(X.L_GRID)}, "coupled": coupled}


@pytest.fixture
def elig_file(tmp_path, monkeypatch):
    root = tmp_path / "elig_file"
    (root / "posthoc_x18").mkdir(parents=True)
    monkeypatch.setattr(paths, "RECORDS", root)
    return root / "posthoc_x18" / X.OUT_ELIG


def test_builds_and_passes_the_audit(elig_file):
    try:
        style.arial_files()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    elig_file.write_text(json.dumps(_doc()), encoding="utf-8")
    fig = F.build()
    w, _ = fig.get_size_inches()
    assert w == pytest.approx(style.FULL)
    bad = {k: v for k, v in checks.audit(fig).items() if v}
    bad |= {k: v for k, v in methods.audit(fig).items() if v}
    plt.close(fig)
    assert not bad


def test_a_pending_cell_stops_the_build(elig_file):
    elig_file.write_text(json.dumps(_doc(pending_at=1.20)), encoding="utf-8")
    with pytest.raises(ValueError, match="pending"):
        F.build()

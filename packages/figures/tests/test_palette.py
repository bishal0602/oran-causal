"""Harbor is the single palette (palette.py). Every figure colour must come from it (or be a tint /
neutral).
"""

from oran_causal.figures import methods as M
from oran_causal.figures import palette as P


def test_every_method_colour_is_harbor():
    for code in M.METHODS:
        assert M.color(code) in (*P.METHODS, *P.NEUTRALS), code


def test_method_list():
    """[DECIDED 2026-10-08, user] one fixed colour per method, paper-wide (DESIGN s.12.1)."""
    assert M.color("pmrt") == M.color("pmrt_gbm") == P.BLUE
    assert M.color("pmrt_lin") == M.color("pmrt_lin_r3") == P.PMRT_LIN
    assert M.color("corr") == P.GOLD
    assert M.color("granger_dev") == M.color("granger") == P.TEAL
    assert M.color("granger_by") == M.color("granger_adj") == P.TERRACOTTA
    assert M.color("shap_gbdt") == M.color("pcorr_adj") == P.PURPLE
    assert M.color("two_tower") == M.color("pcorr") == P.GREEN
    assert (M.color("qacm"), M.color("int")) == (P.SLATE, P.LILAC)

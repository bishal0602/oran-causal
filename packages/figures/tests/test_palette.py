"""Harbor is the single palette (palette.py). Every figure colour must come from it (or be a tint /
neutral).
"""

import matplotlib

from oran_causal.figures import methods as M
from oran_causal.figures import palette as P


def test_harbor_values():
    assert P.ENTITIES == ["#4273B0", "#C17154", "#3FAFA9", "#CBAA56", "#955C98", "#5C9B5E"]
    assert [
        *P.ENTITIES,
        "#7196C4",
        "#8CA7CF",
        "#B58DB7",
    ] == P.METHODS  # six hues + PMRT-Lin, QACM, INT tints
    assert P.HERO == P.BLUE
    for tint, base, amount in (
        (P.PMRT_LIN, P.BLUE, 0.75),
        (P.SLATE, P.SEQ[5], 0.6),
        (P.LILAC, P.PURPLE, 0.7),
    ):
        assert matplotlib.colors.to_hex(P.tint(base, amount)).upper() == tint, tint


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

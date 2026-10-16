"""Logic of the figure modules that reads no record set: strip words, draw rules, text wrapping,
protocol eligibility.
"""

import matplotlib

matplotlib.use("Agg")

from oran_causal.figures.figs import _tabm as TM
from oran_causal.figures.figs import fig_envs as F
from oran_causal.figures.figs import fig_logging_validity as V
from oran_causal.figures.figs import fig_study4_envs as F4
from oran_causal.figures.figs import tab_study2_criteria as T2


def test_strip_words_match_logging_validity():
    assert [s.lower() for s in F.STRIP_TITLE] == [s.replace("\n", " ") for s in V.DESIGN_LABEL]


def test_draws_follow_the_design_rules():
    d = F.draws()
    assert len(set(d["setpoint"])) == F.DITHER_BLOCKS
    assert abs(d["dither"] - d["setpoint"][d["block"]]).max() <= F.DITHER_DELTA


def test_wrap_balances_lines():
    lines = F4._wrap("E1: clean identifiable linear graph", 120 - F4.TAG_X, 7.5)
    assert lines == ["E1: clean identifiable", "linear graph"]


def test_eligibility_read_from_protocol():
    assert T2.eligibility() == TM.eligibility()  # Studies 2 and 3 share the definition

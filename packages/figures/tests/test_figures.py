"""Smoke tests: style applies, palette invariants hold, every registered figure renders."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from oran_causal.figures import palette as P
from oran_causal.figures import registry, style


def test_style_applies_font_and_sizes():
    style.apply()
    assert plt.rcParams["font.family"] == [style.FONT_FAMILY]
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["xtick.labelsize"] == style.TICK_PT


def test_palette_slots():
    assert P.METHODS[0] == P.HERO
    assert len(P.ENTITIES) == len(P.ENTITY_MARKERS) == 6
    with pytest.raises(IndexError):
        P.entity(6)


def test_figure_builds(build_figure, figure_name):
    fig = build_figure(figure_name)
    w, _ = fig.get_size_inches()
    assert w == pytest.approx(
        {"full": style.FULL, "half": style.HALF}[registry.discover()[figure_name].SPEC.width]
    )


def test_text_floor(build_figure, paper_figure_name):
    """Journal artwork rule: every visible text of a paper figure is Arial at >=
    style.MIN_PT (8 pt).
    """
    from matplotlib.text import Text

    fig = build_figure(paper_figure_name)
    fig.draw_without_rendering()
    bad = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_figure() is not None:
            if t.get_fontsize() < style.MIN_PT - 1e-9 or t.get_fontname() != style.FONT_FAMILY:
                bad.append((t.get_text()[:30], t.get_fontsize(), t.get_fontname()))
    assert not bad, f"{paper_figure_name}: {bad[:8]}"

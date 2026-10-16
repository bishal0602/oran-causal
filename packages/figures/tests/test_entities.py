"""xApp and KPI colour slots (oran_causal.figures.entities, DESIGN s.12.2) [DECIDED 2026-10-06,
user; slots per panel]: table invariants, the slots match DESIGN, the five rules on every registered
figure, and each rule firing on a broken figure.
"""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from oran_causal.figures import diagram as D
from oran_causal.figures import entities as E
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots, style


# --- table invariants -----------------------------------------------------------------------------------------------
def test_xapp_slots():
    assert {k: e.slot for k, e in E.XAPPS.items()} == {
        "ES": 2,
        "SliceGuarantee": 3,
        "PowerES": 4,
        "Coverage": None,
    }
    assert E.XAPPS["Coverage"].color == P.NEUTRAL
    hues = [e.color for e in E.XAPPS.values() if e.slot]
    assert len(set(hues)) == len(hues)
    assert P.ENTITIES[0] not in hues
    assert P.ENTITIES[4] not in hues


def test_xapp_names_are_paper_names():
    assert [e.name for e in E.XAPPS.values()] == ["ES", "Slice Guarantee", "Power ES", "Coverage"]
    assert E.param_label("sleep") == "sleep (ES)"
    assert E.param_label("prot_min") == "prot_min (Slice Guarantee)"
    assert E.param_label("ptx") == "ptx (Power ES)"  # ptx keeps Power ES's hue (Coverage gray)
    for m in M.METHODS.values():
        assert not any(c in m.name for c in E.CODE_NAMES), m.name


def test_kpi_slots():
    assert E.PROVISIONAL
    assert {k: e.slot for k, e in E.KPIS.items()} == {"rlf": 1, "load": 2, "pv": 3, "v": 4, "e": 5}
    terracotta, green = P.ENTITIES[1], P.ENTITIES[5]
    for k in ("v", "pv", "rlf"):  # violation KPIs: no red-adjacent hue
        assert E.KPIS[k].color != terracotta
    assert green not in {e.color for e in E.KPIS.values()}  # reads as "good"
    assert [{"e", "rlf"}] == E.WEAK_PAIRS
    assert {E.KPIS["e"].color, E.KPIS["rlf"].color} == {P.ENTITIES[0], P.ENTITIES[4]}


def test_planned_method_figures_use_no_entity_colour(planned):
    for fig in planned:
        assert fig not in E.FIGURE_USE, fig


def _design_section(text: str, title: str) -> str:
    start = text.index(title)
    return text[start : text.index("\n## ", start)]


def test_slots_match_design(design):
    sec = _design_section(design, "### Entities: xApps and KPIs")
    for e in [*E.XAPPS.values(), *E.KPIS.values()]:
        row = next((ln for ln in sec.splitlines() if ln.startswith(f"| {e.name} |")), None)
        assert row is not None, f"DESIGN entity table has no row for {e.name}"
        assert e.color.lower() in row.lower(), f"{e.name}: {e.color} not in DESIGN row {row}"
        if e.slot:
            assert re.search(rf"\|\s*{e.slot}\b", row), (
                f"{e.name}: slot {e.slot} not in DESIGN row {row}"
            )


# --- rules on every registered figure -------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", E.RULES)
def test_entity_rules(build_figure, figure_name, rule):
    problems = E.audit(build_figure(figure_name))[rule]
    assert not problems, f"{figure_name}: " + "; ".join(problems)


# --- each rule fires on a deliberately broken figure ----------------------------------------------------------------
def _bad(rule):
    if rule in ("ink arrows", "code names"):
        fig, ax = D.canvas("half", 80)
        E.xapp_box(ax, 10, 10, 60, 24, "ES")
        E.param_node(ax, 100, 22, "sleep")
        if rule == "ink arrows":
            D.arrow(ax, [(70, 22), (94, 22)])
            ax.patches[-1].set_edgecolor(E.XAPPS["ES"].color)  # an arrow in the writer's hue
        else:
            D.box(ax, 120, 10, 60, 24, "PowerES")
        return fig
    fig, ax = plots.figure("half")
    x = np.arange(4.0)
    if rule == "entity in method figure":
        M.plot(ax, x, x, "pmrt_nl_eq", hero=True)
        ax.plot(x, x + 1)  # default cycle = entity blue
    elif rule == "one entity type":
        E.plot_xapp(ax, x, x, "ES")
        E.plot_kpi(ax, x, x + 1, "pv")
    elif rule == "e and rlf":
        E.plot_kpi(ax, x, x, "e")
        E.plot_kpi(ax, x, x + 1, "rlf", direct_label=False)
    return fig


@pytest.mark.parametrize("rule", E.RULES)
def test_rule_fires(rule):
    style.apply()
    fig = _bad(rule)
    problems = E.audit(fig)[rule]
    plt.close(fig)
    assert problems, f"{rule} did not flag its broken figure"


def test_good_xapp_diagram_passes():
    style.apply()
    fig, ax = D.canvas("half", 90)
    for i, k in enumerate(("ES", "PowerES", "SliceGuarantee")):
        E.xapp_box(ax, 10 + 60 * i, 10, 52, 22, k)
    E.param_node(ax, 30, 60, "sleep")
    E.param_node(ax, 90, 60, "ptx")
    E.kpi_node(ax, 150, 60, "pv")
    D.arrow(ax, [(36, 60), (144, 60)])
    D.box(ax, 10, 70, 60, 16, "Referee", role="proposed")
    problems = E.audit(fig)
    plt.close(fig)
    assert not any(problems.values()), problems


def test_e_and_rlf_with_labels_and_markers_pass():
    style.apply()
    fig, ax = plots.figure("half")
    x = np.arange(4.0)
    E.plot_kpi(ax, x, x, "e")
    E.plot_kpi(ax, x, x + 1, "rlf")
    problems = E.audit(fig)
    plt.close(fig)
    assert not any(problems.values()), problems

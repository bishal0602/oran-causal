"""No internal code names in visible text (oran_causal.figures.wording): figure text, caption hints,
generated LaTeX tables.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from oran_causal.figures import paths, plots, registry, style, wording

FIGS = registry.discover()
TABLES = sorted((paths.PAPER / "tables").glob("*.tex"))


def test_figure_text(build_figure, figure_name):
    problems = wording.audit(build_figure(figure_name))
    assert not problems, f"{figure_name}: " + "; ".join(problems)


@pytest.mark.parametrize("name", list(FIGS))
def test_caption_hint(name):
    assert not wording.check(FIGS[name].SPEC.caption_hint)


@pytest.mark.parametrize("tex", TABLES, ids=[t.stem for t in TABLES])
def test_table_text(tex, journal):
    assert not wording.check(wording.tex_visible(tex)), wording.check(wording.tex_visible(tex))


@pytest.mark.parametrize(
    "bad",
    [
        "E6-P bridging",
        "E6P logs",
        "Study A",
        "Exp B",
        "MapGateV2",
        "MSCR+",
        "arbiter",
        "knob",
        "CDL",
        "CDD-ORAN",
        "option (a)",
        "PowerES",
        "pmrt_nl_eq",
        "pcorr_native",
    ],
)
def test_rule_fires(bad):
    style.apply()
    fig, ax = plots.figure("half")
    ax.set_title(bad)
    assert wording.audit(fig), bad
    plt.close(fig)


@pytest.mark.parametrize(
    "ok",
    [
        "Simulator logs (bridging)",
        "PMRT-GBM",
        "Power ES only",
        "CDD O-RAN discovery",
        "Design-adjusted partial correlation",
        "E1",
        "E4 R3",
        "MSCR (n ≤ 1000)",
    ],
)
def test_paper_names_pass(ok):
    assert not wording.check(ok), ok


def test_tex_comments_ignored(tmp_path):
    p = tmp_path / "t.tex"
    p.write_text(
        "% Sources: CDD-ORAN/scratchpad/e6_dev E6P\n50\\% & PMRT \\\\ % E6-P note\n",
        encoding="utf-8",
    )
    assert not wording.check(wording.tex_visible(p))

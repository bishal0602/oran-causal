"""fig_study4_envs (the five Study 4 ground-truth graphs, main text): same graphs as the Appendix A
pages, every text >= 8 pt at print size, no node squeezed by an edge or a neighbour, one legend with
only the entries used, and the house panel headers. The generic layout checks (overlap, intrusion,
on canvas ...) run in test_layout.py.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import FancyArrowPatch

from oran_causal.figures import envgraph as EG
from oran_causal.figures import style
from oran_causal.figures.figs import _env_page as E
from oran_causal.figures.figs import fig_study4_envs as F

TEXTWIDTH_BP = 370.62  # \textwidth = 372.0 pt (sn-jnl build log)
MIN_CLEAR_PT = 6.0  # an edge stays this far from every node it does not start or end at
MIN_GAP_PT = 6.0  # free space between two neighbouring nodes of a column


pytestmark = pytest.mark.usefixtures("records")


@pytest.fixture(scope="module")
def fig():
    style.apply()
    f = F.build()
    yield f
    plt.close(f)


def _canvas(fig, w):
    return next(a for a in fig.axes if a.get_label() == f"graph {w}")


def _nodes(ax) -> dict:
    return {
        t.get_text(): np.array(t.get_position())
        for t in ax.texts
        if t.get_gid() is None and t.get_text() != EG.LAG
    }


def test_size_fits_the_page(fig):
    w_in, h_in = fig.get_size_inches()
    assert w_in == pytest.approx(style.FULL)
    assert h_in <= style.MAX_HEIGHT
    fig.draw_without_rendering()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    assert tight.width * 72 + 2 * 72 * plt.rcParams["savefig.pad_inches"] <= TEXTWIDTH_BP, (
        "LaTeX would shrink it"
    )


@pytest.mark.parametrize("w", F.WORLDS)
def test_same_graphs_as_the_appendix_pages(fig, w):
    g = E.GRAPHS[w]
    want = set(g.acts) | set(g.kpis) | set(g.placebos) | ({"Z"} if g.z else set())
    assert set(_nodes(_canvas(fig, w))) == want
    arrows = [p for p in _canvas(fig, w).patches if isinstance(p, FancyArrowPatch)]
    assert arrows, f"{w}: no edges drawn"


def test_every_text_at_least_min_pt(fig):
    small = [
        (t.get_text(), t.get_fontsize())
        for ax in fig.axes
        for t in ax.texts
        if t.get_fontsize() < style.MIN_PT
    ]
    for leg in fig.legends:
        small += [
            (t.get_text(), t.get_fontsize())
            for t in leg.get_texts()
            if t.get_fontsize() < style.MIN_PT
        ]
    assert not small


@pytest.mark.parametrize("w", F.WORLDS)
def test_edges_clear_other_nodes(fig, w):
    ax = _canvas(fig, w)
    nodes = _nodes(ax)
    for p in ax.patches:
        if not isinstance(p, FancyArrowPatch):
            continue
        pts = np.vstack([seg(np.linspace(0, 1, 40)) for seg, _ in p.get_path().iter_bezier()])
        ends = {
            n for n, c in nodes.items() for e in (pts[0], pts[-1]) if np.hypot(*(e - c)) < EG.R + 10
        }
        for n, c in nodes.items():
            if n not in ends:
                clear = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]).min() - EG.R
                assert clear >= MIN_CLEAR_PT, f"{w}: an edge passes {clear:.1f} pt from {n}"


@pytest.mark.parametrize("w", F.WORLDS)
def test_nodes_not_squeezed(fig, w):
    nodes = _nodes(_canvas(fig, w))
    for col in (lambda n: n.startswith("K"), lambda n: not n.startswith("K")):
        ys = sorted(c[1] for n, c in nodes.items() if col(n))
        assert all(b - a >= 2 * EG.R + MIN_GAP_PT for a, b in zip(ys, ys[1:], strict=False)), (
            f"{w}: nodes {np.diff(ys)} apart"
        )


def test_one_legend_with_only_the_used_entries(fig):
    assert len(fig.legends) == 1
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    graphs = [E.GRAPHS[w] for w in F.WORLDS]
    used = {lab for g in graphs for lab in EG.legend_handles(g)[1]}
    used = {("Placebo action (Plc: R3)" if lab.startswith("Placebo") else lab) for lab in used}
    assert sorted(labels) == sorted(used)
    assert len(labels) == len(set(labels))


def test_panel_headers(fig):
    fig.draw_without_rendering()
    base = {}
    for letter, w in zip("abcde", F.WORLDS, strict=False):
        ax = _canvas(fig, w)
        lets = [t for t in ax.texts if t.get_gid() == "panel-letter"]
        tags = [t.get_text() for t in ax.texts if t.get_gid() == "panel-tag"]
        assert [t.get_text() for t in lets] == [letter]
        assert " ".join(tags) == f"{w}: {E.BLURB[w]}"
        base[w] = lets[0].get_window_extent().y0
    assert base["E1"] == pytest.approx(base["E2"]) == pytest.approx(base["E3"])
    assert base["E4"] == pytest.approx(base["E5"])

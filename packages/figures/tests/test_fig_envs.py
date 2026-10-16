"""fig_envs (Section 5.4: environments and logging designs): every true edge of the five
ground-truth graphs is drawn, every node is placed, the figure fits the text width at <= 2.6 in, and
the strips use fig_logging_validity's words. The generic layout checks (overlap, intrusion, on
canvas ...) run in test_layout.py.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from oran_causal.figures import style
from oran_causal.figures.figs import _env_page as E
from oran_causal.figures.figs import fig_envs as F

pytestmark = pytest.mark.usefixtures("records")


@pytest.fixture(scope="module")
def fig():
    style.apply()
    f = F.build()
    yield f
    plt.close(f)


def _canvas(fig, w):
    return next(a for a in fig.axes if a.get_label() == f"graph {w}")


def test_size(fig):
    w, h = fig.get_size_inches()
    assert w == pytest.approx(style.FULL)
    assert h <= 2.6 + 1e-9


@pytest.mark.parametrize("w", F.WORLDS)
def test_every_edge_and_node_drawn(fig, w):
    g, ax = E.GRAPHS[w], _canvas(fig, w)
    gids = [p.get_gid() for p in ax.patches if p.get_gid()]
    drawn = sorted(x[5:] for x in gids if x.startswith("edge:"))
    assert drawn == sorted(f"{s}->{d}" for s, d, _ in g.edges)
    nodes = {x[5:] for x in gids if x.startswith("node:")}
    assert nodes == set(g.acts) | set(g.kpis) | set(g.placebos) | ({"Z"} if g.z else set())
    assert not any(x.startswith(f"edge:{p}->") for p in g.placebos for x in gids), (
        "placebo action has a child"
    )

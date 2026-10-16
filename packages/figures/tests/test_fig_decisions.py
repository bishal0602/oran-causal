"""fig_decisions [F7]: the build-time assertions fire on doctored FD values, the direct labels carry
the FD numbers, the size stays within the L3 variant's, and no text touches a line or a whisker.
"""

import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from oran_causal.figures import style
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import fig_decisions as F

pytestmark = pytest.mark.usefixtures("records")


@pytest.fixture(scope="module")
def fig():
    style.apply()
    f = F.build()
    f.draw_without_rendering()
    yield f
    plt.close(f)


def _texts(fig):
    return [t for ax in fig.axes for t in ax.texts if t.get_visible() and t.get_text().strip()]


def test_size(fig):
    w, h = fig.get_size_inches()
    assert w == pytest.approx(style.FULL)
    assert h <= 3.0 + 1e-9


def test_labels_read_fd(fig):
    g = ST.fd()["option_a"]["eval"]["arms"]["MG:PMRT"]["guard_ratio"]["rlf"]
    texts = {t.get_text() for t in _texts(fig)}
    assert f"PMRT map, no gate\nRLF ratio {g:.2f}" in texts
    assert {"breaks RLF limit", "with gate"} <= texts


def _segments(ax):
    """Every line and whisker of ``ax`` in display coordinates, as point arrays."""
    out = []
    for art in ax.get_children():
        if (
            isinstance(art, Line2D)
            and art.get_visible()
            and art.get_linestyle() not in ("None", "", " ")
        ):
            out.append(art.get_transform().transform(art.get_xydata()))
        elif isinstance(art, LineCollection) and art.get_visible():
            out += [art.get_transform().transform(p.vertices) for p in art.get_paths()]
    return out


def test_no_text_on_a_line(fig):
    r = fig.canvas.get_renderer()
    pad = fig.dpi / 72  # 1 pt
    hits = []
    for t in _texts(fig):
        bb = t.get_window_extent(r).expanded(1, 1)
        x0, x1, y0, y1 = bb.x0 - pad, bb.x1 + pad, bb.y0 - pad, bb.y1 + pad
        for ax in fig.axes:
            for seg in _segments(ax):
                for (ax0, ay0), (ax1, ay1) in zip(seg[:-1], seg[1:], strict=False):
                    s = np.linspace(0, 1, 200)
                    px, py = ax0 + (ax1 - ax0) * s, ay0 + (ay1 - ay0) * s
                    if np.any((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)):
                        hits.append(t.get_text()[:30])
                        break
    assert not hits, hits


def _doctored(monkeypatch, edit):
    d = copy.deepcopy(ST.fd())
    edit(d)
    monkeypatch.setattr(ST, "fd", lambda: d)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["certsafe"]["eval"]["arms"]["CS:PMRT"].__setitem__("R", 0.3),
        lambda d: d["certsafe"]["eval"]["arms"]["CS:PMRT"].__setitem__("R_ci90", [0.1, 0.4]),
        lambda d: d["certsafe"]["eval"]["arms"]["CS:PMRT"]["guard_ratio"].__setitem__("rlf", 0.97),
        lambda d: d["certsafe"]["eval"]["arms"]["CS:PMRT"]["guard_ratio_ci90"].__setitem__(
            "rlf", [0.9, 1.0]
        ),
        lambda d: d["option_a"]["eval"]["arms"]["MG:PMRT"]["guard_ratio"].__setitem__("rlf", 1.05),
    ],
    ids=["R", "R_ci", "rlf", "rlf_ci", "breach"],
)
def test_assertions_fire(monkeypatch, edit):
    _doctored(monkeypatch, edit)
    before = set(plt.get_fignums())
    with pytest.raises(ValueError):
        F.build()
    for num in set(plt.get_fignums()) - before:
        plt.close(num)

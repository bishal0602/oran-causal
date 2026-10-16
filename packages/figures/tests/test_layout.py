"""Layout checks (oran_causal.figures.checks) on every registered figure: the classes of the
2026-10-05 figure review, plus clipped markers and touching INVALID crosses (2026-10-06).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from oran_causal.figures import checks, style

CHECKS = [
    "text overlap",
    "text intrusion",
    "legend placement",
    "anchored labels",
    "lone markers",
    "coincident series",
    "tick hygiene",
    "facet width",
    "on canvas",
    "clipped markers",
    "touching crosses",
]


@pytest.mark.parametrize("check", CHECKS)
def test_layout(build_figure, figure_name, check):
    problems = checks.audit(build_figure(figure_name))[check]
    assert not problems, f"{figure_name}: " + "; ".join(problems[:6])


# --- each check fires on a deliberately broken figure -------------------------------------------------------------
def _bad(kind):
    import numpy as np

    from oran_causal.figures import plots

    if kind == "facet width":
        fig, axs = plots.figure("full", ncols=6)
        return fig
    fig, (a, b) = plots.figure("full", ncols=2)
    x = np.arange(5.0)
    if kind == "text overlap":
        a.annotate("first label", xy=(0.5, 0.5), xycoords="axes fraction")
        a.annotate("second label", xy=(0.5, 0.5), xycoords="axes fraction")
    elif kind == "text intrusion":
        a.annotate(
            "reaches into b",
            xy=(1.0, 0.5),
            xycoords="axes fraction",
            xytext=(-10, 0),
            textcoords="offset points",
            fontsize=14,
        ).set_in_layout(False)
    elif kind == "legend placement":
        a.plot(x, x, label="s")
        fig.legend(loc="outside lower center")
    elif kind == "anchored labels":
        a.annotate("floating", xy=(2, 2), xytext=(0, 22), textcoords="offset points")
    elif kind == "lone markers":
        a.plot(x, [1, np.nan, 2, np.nan, 3], marker="o")
    elif kind == "coincident series":
        a.plot(x, x, marker="o", label="one")
        a.plot(x, x, marker="s", label="two")
    elif kind == "tick hygiene":
        a.plot(x, x)
        b.plot(x, x)
        a.tick_params(axis="x", labelbottom=False)
    elif kind == "clipped markers":
        a.plot(x, x, marker="o", ms=6)
        a.set_ylim(0, 4)  # the point at y = 4 sits on the top edge
    elif kind == "touching crosses":
        from oran_causal.figures import layout as L

        L.invalid_marks(a, [2.0], "k")
        L.invalid_marks(a, [2.0], "k")  # not finalized: both crosses at the same spot
    elif kind == "on canvas":
        a.annotate(
            "off the page",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-400, 0),
            textcoords="offset points",
        ).set_in_layout(False)
    return fig


@pytest.mark.parametrize(
    "check",
    [
        "text overlap",
        "text intrusion",
        "legend placement",
        "anchored labels",
        "lone markers",
        "coincident series",
        "tick hygiene",
        "facet width",
        "on canvas",
        "clipped markers",
        "touching crosses",
    ],
)
def test_check_fires(check):
    style.apply()
    fig = _bad(check)
    problems = checks.audit(fig)[check]
    plt.close(fig)
    assert problems, f"{check} did not flag its broken figure"

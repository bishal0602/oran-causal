"""Study 4 validity by environment (ESM; [G5 2026-10-12, P3] panel (b) of fig_study4_validity moved
to the ESM, restructure plan "Fig 12b to the ESM"): the null-edge rejection rate of each method and
regime in each environment E1-E5, printed in cells shaded on harbor_div centred at the nominal .05
(two-slope norm 0-1, labelled colour bar). Rows, data and shading as fig_study4_validity (shared
builder ``build_panel("env")``); see that module's docstring. One panel, so no panel letter (DESIGN
s.12, as fig_study2_maps).
"""

from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import fig_study4_validity as V
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_validity_env",
    width="full",
    paper=True,
    sources=[T.SRC_REPORT, T.SRC_CELLS],
    caption_hint="Null-edge rejection rate per method and regime in each environment E1-E5 (Study 4, kappa "
    ".25; mean of the environment's conditions), shaded on a diverging scale centred at the "
    "nominal .05: blue below, rust above; a dash marks no condition",
)


# plain names for the ESM image (the body figure keeps its own labels): logging designs and
# environments
REG_PLAIN = {
    "R1": "Independent",
    "R2": "Dithered",
    "R1+R2": "Indep. + dithered",
    "R3": "Confounded",
}
ENV_PLAIN = ["Linear\ngraph", "Rare\nregime", "Delayed\nharm", "Confounding\n(placebo)", "Composed"]


def build():
    keep = dict(V.REG_LABEL)
    V.REG_LABEL.update(REG_PLAIN)
    try:
        fig = V.build_panel("env")
    finally:
        V.REG_LABEL.clear()
        V.REG_LABEL.update(keep)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        w if w != "E4" else "E4\nplacebo" for w in T.WORLDS
    ]
    ax.set_xticks(range(len(ENV_PLAIN)), ENV_PLAIN)
    return fig

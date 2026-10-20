"""Appendix C.6, X6 told-width grid (post hoc, exploratory; declared before the run, fresh seeds;
changes none of C1-C3): truth-null raw rejection rate (p <= .05) of PMRT-GBM and PMRT-Lin when the
dither width is told to the method as x .5, .8, 1 (exact), 1.25, 1.5 or 2 of the true width (told
law used for both centring and redraws). One cell: E1 R2, n 1000, kappa .25, 200 fresh datasets, 12
truth-null candidates per dataset (2400 per point). Shown next to fig_appC_x3_misspec; extends X3
panel (a) on fresh data.

Source: DIAG ``diag/diag_tables.json``, ``x6.curve`` (one row per base arm x told width; key
``null_raw``: rate, 95 % bootstrap CI). Row selection only. The 2 x 2 centring x redraw ablation
(rows X.6a, X.6c) is the table ``tab_appC_x6``, not drawn here.
"""

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures.figs import _appc as C
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_appC_x6_width",
    width="half",
    paper=True,
    sources=[C.SRC_DIAG],
    caption_hint="Post hoc, exploratory X6 told-width grid (declared before the run, fresh seeds; does "
    "not change C1-C3): truth-null rejection rate of PMRT-GBM and PMRT-Lin vs the dither width "
    "told to the method (told / true; 1 = exact; levels evenly spaced), told law used for centring and redraws. "
    "E1 R2, n 1000, 200 fresh datasets (2400 truth-null candidates per point). Bands: 95 % "
    "bootstrap CIs; symmetric-log scale, linear below 0.01; dashed line = nominal 0.05; "
    "series dodged horizontally. " + C.COLOUR_NOTE,
)

ARMS = ["pmrt_nl_eq", "pmrt_eq"]


def build():
    curve = C.diag()["x6"]["curve"]
    widths = sorted({r["width"] for r in curve})
    fig, ax = plots.figure("half", aspect=0.82)
    for arm in M.order(ARMS)[::-1]:  # PMRT-Lin first, the hero on top
        rows = {r["width"]: r for r in curve if r["base"] == arm}
        C.draw(
            ax, arm, ARMS, *C.points(C.at(widths, [(w, rows.get(w)) for w in widths]), "null_raw")
        )
    plots.ref_line(ax, 0.05)
    C.level_axis(ax, widths, "Told dither width (× true)")
    L.rate_axis(ax, "y", "Truth-null rate")
    plots.finish(ax)
    M.legend(fig, ARMS, hero=C.HERO)
    return L.finalize(fig)

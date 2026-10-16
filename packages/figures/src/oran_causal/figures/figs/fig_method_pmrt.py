"""Section 4: the PMRT pipeline (assignment score, predictable adjustment, matched filter,
randomization test), the weighted BY declaration layer, the causal map and the referee. Content from
TERMINOLOGY s.6.1 / s.7 and CDD-ORAN docs/ARCHITECTURE.md s.3 (diagram; no data). One hero
component: the PMRT block (DESIGN s.9); arrows ink.
"""

from matplotlib.patches import FancyBboxPatch

from oran_causal.figures import diagram as D
from oran_causal.figures import palette as P
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_method_pmrt",
    width="full",
    paper=True,
    sources=["CDD-ORAN/docs/ARCHITECTURE.md s.3", "TERMINOLOGY.md s.6.1, s.7"],
    caption_hint="The PMRT pipeline. Each logged request gets a design-centred assignment score from its "
    "logged propensity; a predictable (past-only) adjustment removes what earlier data "
    "predict; a matched filter frozen from earlier data combines the receiving cells and time "
    "bins; the randomisation test redraws the modes from the logging policy (B = 9999). "
    "Weighted BY declares signed edges over the 60 hypotheses, which form the causal map "
    "that drives the referee. Asymptotically valid, not exact",
)

IN_X, IN_W = 6.0, 88.0
PM_X, PM_W = 108.0, 132.0
OUT_X, OUT_W = 258.0, 106.0
STEP_H = 32.0


def build():
    fig, ax = D.canvas("full", height_pt=200)
    # inputs
    ins = [
        ("Logged requests", "mode and propensity", 30),
        ("KPI series", "per cell, observed only", 72),
        ("Frozen weights", "fitted on earlier data", 114),
    ]  # level with the step each one feeds
    for lab, sub, y in ins:
        D.box(ax, IN_X, y, IN_W, STEP_H, lab, sub)
    # PMRT: the one hero component, with its four steps inside
    ax.add_patch(
        FancyBboxPatch(
            (PM_X, 8),
            PM_W,
            186,
            boxstyle="round,pad=0,rounding_size=3",
            fc=P.tint(P.HERO, 0.16),
            ec=P.HERO,
            lw=1.0,
            zorder=1.5,
        )
    )
    ax.text(PM_X + 8, 19, "PMRT", fontsize=8, fontweight="bold", va="center", zorder=4)
    steps = [
        ("Assignment score", "from the logged propensities"),
        ("Predictable adjustment", "past-only prediction"),
        ("Matched filter", "frozen space-time profile"),
        ("Randomisation test", "redraw modes, B = 9999"),
    ]
    sx, sw = PM_X + 9, PM_W - 18
    ys = [30 + i * 42 for i in range(len(steps))]
    for (lab, sub), y in zip(steps, ys, strict=False):
        D.box(ax, sx, y, sw, STEP_H, lab, sub)
    for y0, y1 in zip(ys[:-1], ys[1:], strict=False):
        D.arrow(ax, [(sx + sw / 2, y0 + STEP_H), (sx + sw / 2, y1 - 1)])
    D.arrow(ax, [(IN_X + IN_W, ins[0][2] + 16), (sx - 1, ys[0] + 16)])
    D.arrow(ax, [(IN_X + IN_W, ins[1][2] + 16), (sx - 1, ys[1] + 16)])
    D.arrow(ax, [(IN_X + IN_W, ins[2][2] + 16), (sx - 1, ys[2] + 16)])
    # declaration layer, map, referee
    outs = [
        ("Weighted BY", "signed, 60 hypotheses", 30),
        ("Causal map", "declared signed edges", 96),
        ("Referee", "rejects harmful requests", 162),
    ]
    for lab, sub, y in outs:
        D.box(ax, OUT_X, y, OUT_W, STEP_H, lab, sub)
    ox = OUT_X + OUT_W / 2
    D.arrow(ax, [(ox, 30 + STEP_H), (ox, 95)])
    D.arrow(ax, [(ox, 96 + STEP_H), (ox, 161)])
    D.arrow(
        ax, [(sx + sw, ys[3] + 16), (249, ys[3] + 16), (249, 46), (OUT_X - 1, 46)]
    )  # p-values (caption)
    return fig

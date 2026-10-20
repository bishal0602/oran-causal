"""Appendix C.3, X3 design misspecification (post hoc, exploratory; declared before launch, fresh
seeds; changes none of C1-C3): the data are generated with the frozen design and the design is told
to the method wrong. Null-edge raw rejection rate (p <= .05) vs the misspecification level, one
panel per misspecified element: (a) dither width told x .5, .8, 1 (exact), 1.25, 2 and (b) setpoint
switch times told 2 % / 5 % of a block late, both on the truth-null edges of R2 (E1-E2, n 1000, 60
seeds); (c) logging-policy strength lambda told x .5 ... 2 on the confounded null edge of E4 R3 (lam
1, n 1000, 60 seeds: one edge per seed, so wide CIs).

Source: XR ``extras_tables.json``, ``pooled`` rows with experiment X3; R2 key ``null_raw``, R3 key
``conf_raw`` (rate, seed-cluster 95 % bootstrap CI). Rows are matched by base arm and ``design_key``
(``("width", 0.8)``, ``("shift", 0.02)``, ``("lam", 2.0)``; the exact design is ``("exact", 1.0)``).
Row selection only. Series (DESIGN s.12.1): PMRT-GBM (hero), PMRT-Lin, PMRT-Lin with R3 covariates
and design-adjusted partial correlation; the level is the x axis, not a series. Required wording
(posthoc protocol s.6) goes in the caption: a degradation curve, not evidence against C2a.
"""

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures.figs import _appc as C
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_appC_x3_misspec",
    width="full",
    paper=True,
    sources=[C.SRC_XR],
    caption_hint="Post hoc, exploratory X3 design misspecification (declared before launch, fresh seeds; "
    "does not change C1-C3): data generated with the frozen design, the design told to the "
    "method wrong. Null-edge rejection rate vs the told dither width (a) and the told "
    "setpoint switch delay (b), truth-null edges in R2 of E1-E2 (n 1000, 60 seeds), and vs "
    "the told logging-policy strength λ (c), confounded null edge in E4 R3 (n 1000, 60 seeds, "
    "one edge per seed). Multipliers are told / true (1 = exact); levels evenly spaced. "
    "Bands (a, b) and whiskers (c): 95 % seed-cluster bootstrap CIs; symmetric-log scale, "
    "linear below 0.01; dashed line = nominal 0.05; series dodged horizontally. This is a degradation curve showing where the known-design "
    "assumption breaks; it is not evidence against C2a, which assumes the design is known. "
    + C.COLOUR_NOTE
    + "; PMRT-Lin with R3 covariates is dotted",
)

ARMS = ["pmrt_nl_eq", "pmrt_eq", "pmrt_r3", "pcorr_eq"]
SHIFTS = [0.0, 0.02, 0.05]
EXACT = ("exact", 1.0)


def _rows(pool: dict, arm: str, kind: str, levels) -> list:
    """(level, pooled row) along one misspecification axis; level 1 (multipliers) or 0 (switch
    delay) = exact.
    """
    return [(v, pool.get((arm, EXACT if v in (0.0, 1.0) else (kind, v)))) for v in levels]


def build():
    tab = C.extras()
    r2, r3 = C.pooled(tab, "X3", "R2"), C.pooled(tab, "X3", "R3")
    fig, axs = plots.figure("full", aspect=0.42, ncols=3)
    a, b, c = axs
    for arm in M.order(ARMS)[::-1]:  # baselines first, the hero on top
        C.draw(a, arm, ARMS, *C.points(C.at(C.TOLD, _rows(r2, arm, "width", C.TOLD)), "null_raw"))
        C.draw(b, arm, ARMS, *C.points(C.at(SHIFTS, _rows(r2, arm, "shift", SHIFTS)), "null_raw"))
        C.draw(
            c,
            arm,
            ARMS,
            *C.points(C.at(C.TOLD, _rows(r3, arm, "lam", C.TOLD)), "conf_raw"),
            whiskers=True,
        )
    C.level_axis(a, C.TOLD, "Told dither width (× true)")
    C.level_axis(b, SHIFTS, "Told switch delay (of a block)", ["exact", "+2 %", "+5 %"])
    C.level_axis(c, C.TOLD, "Told policy λ (× true)")
    for ax in axs:
        plots.ref_line(ax, 0.05)
        L.rate_axis(ax, "y", "Null-edge rate")
        plots.finish(ax)
    L.tidy_grid(axs.reshape(1, 3))
    L.panel_header(a, "a", "Dithered setpoint")
    L.panel_header(b, "b", "Dithered setpoint")
    L.panel_header(c, "c", "Confounded policy")
    M.legend(fig, ARMS, hero=C.HERO)
    return L.finalize(fig)

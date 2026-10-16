"""Appendix C.2, X2 dither dose-response (post hoc, exploratory; declared before launch, fresh
seeds; changes none of C1-C3): pooled truth-null raw rejection rate (p <= .05) of the R2 (setpoint +
dither) design in E1-E3 (n 1000 and 4000, 40 measurement seeds per design) (a) vs the dither width
delta at 20 setpoint blocks and (b) vs the number of setpoint blocks at delta .10 (delta .10 / 20
blocks is the Study 4 R2 design).

Source: XR ``extras_tables.json``, ``pooled`` rows with experiment X2 (regime R2), key ``null_raw``
(rate, seed-cluster 95 % bootstrap CI). Row selection only. Raw rates, as in the Study 4 rate
figures; the XR / PAPER_PLAN row X.2 also quotes the BY-declared rate of correlation (0.771 to
0.514), which stays in the compact table. Series (DESIGN s.12.1 style row): PMRT-GBM (hero),
PMRT-Lin, design-adjusted partial correlation, correlation and plain partial correlation. PMRT-Lin
with R3 covariates was not an X2 arm (no X2 row in XR), so it is not drawn. NOTEARS and SHAP-DAG are
table-only: they have no paper style yet (oran_causal.figures.methods: no marker; DESIGN s.12.1
"tables only"). Design-adjusted partial correlation has no rate at 5 setpoint blocks: the frozen arm
stops on a rank-deficient source design there, so every candidate is untestable (XR Notes); that
point is left out (caption).
"""

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures.figs import _appc as C
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_appC_x2_dose",
    width="full",
    paper=True,
    sources=[C.SRC_XR],
    caption_hint="Post hoc, exploratory X2 dither dose-response (declared before launch, fresh seeds; does "
    "not change C1-C3): truth-null rejection rate in R2 of E1-E3 (n 1000 and 4000, 40 seeds "
    "per design) vs the dither width at 20 setpoint blocks (a) and vs the number of setpoint "
    "blocks at dither width 0.10 (b). Bands: 95 % seed-cluster bootstrap CIs; symmetric-log "
    "scale, linear below 0.01; dashed line = nominal 0.05; series dodged horizontally. "
    "Design-adjusted partial correlation has no rate at 5 blocks (rank-deficient design, not "
    "testable). " + C.COLOUR_NOTE,
)

ARMS = ["pmrt_nl_eq", "pmrt_eq", "pcorr_eq", "corr", "pcorr_native"]
DELTAS = [0.02, 0.05, 0.1, 0.2]
BLOCKS = [5, 20, 80]
RATE = "null_raw"


def build():
    pool = C.pooled(C.extras(), "X2", "R2")
    fig, axs = plots.figure("full", aspect=0.42, ncols=2, gridspec_kw={"width_ratios": [4, 3]})
    a, b = axs
    for arm in M.order(ARMS)[::-1]:  # baselines first, the hero on top
        C.draw(a, arm, ARMS, *C.points([(d, pool.get((arm, (d, 20)))) for d in DELTAS], RATE))
        C.draw(b, arm, ARMS, *C.points([(k, pool.get((arm, (0.1, k)))) for k in BLOCKS], RATE))
    for ax in axs:
        plots.ref_line(ax, 0.05)
        ax.set_xscale("log")
        ax.minorticks_off()
    a.set_xticks(DELTAS, ["0.02", "0.05", "0.10", "0.20"])
    a.set(xlim=(0.016, 0.25), xlabel="Dither width")
    b.set_xticks(BLOCKS, [str(k) for k in BLOCKS])
    b.set(xlim=(3.8, 105), xlabel="Setpoint blocks")
    for ax in axs:
        L.rate_axis(ax, "y", "Truth-null rate")
        plots.finish(ax)
    L.tidy_grid(axs.reshape(1, 2))
    L.panel_header(a, "a", "20 setpoint blocks")
    L.panel_header(b, "b", "Dither width 0.10")
    M.legend(fig, ARMS, hero=C.HERO)
    return L.finalize(fig)

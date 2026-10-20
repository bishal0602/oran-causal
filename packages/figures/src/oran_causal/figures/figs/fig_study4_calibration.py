"""F2: calibration of placebo-edge p-values, design-based / design-adjusted vs design-blind tests.
Empirical CDF of the p-values on log-log axes against the uniform diagonal (a calibrated test lies
on it). (a) Study 4, R2, kappa .25, measure datasets of E1-E5 pooled; (b) the bridging analysis,
the ten 60-episode slices.
"""

import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_calibration",
    width="full",
    paper=True,
    sources=[T.SRC_EVAL_MERGED, T.SRC_BRIDGE_MERGED],
    caption_hint="Placebo p-value calibration: Study 4 R2 (a) and the bridging analysis on the simulator logs at 60 episodes "
    "(b); design-based and design-adjusted tests lie on the uniform diagonal, design-blind "
    "tests do not. Colour identifies the method (the same in every figure); it does not "
    "mean validity. Plain and design-adjusted versions of a test share its marker",
)

ARMS = [
    "pmrt_nl_eq",
    "pcorr_eq",
    "granger_eq",
    "pcorr_native",
    "granger_native",
    "corr",
]  # legend order
GRID = np.logspace(-3, 0, 61)
PHASES = (
    0,
    3,
    6,
    1,
    4,
    8,
)  # marker phases on the 10-step grid: the three arms on the diagonal 3 steps apart
LIM = (1e-3, 1.0)
VIEW = (7.5e-4, 1.35)  # headroom: markers at 1 (and the 0.001 corner) never clipped


def _panel(ax, pv: dict):
    ax.plot(LIM, LIM, color=P.INK, alpha=0.6, lw=S.LW_REF, ls=(0, (4, 2)), marker="", zorder=1)
    ax.annotate(
        "uniform",
        xy=(0.2, 0.2),
        xytext=(4, -6),
        textcoords="offset points",
        rotation=45,
        rotation_mode="anchor",
        fontsize=S.ANNOT_PT,
        color=P.INK,
        ha="left",
        va="top",
    )
    for i, arm in enumerate(M.order(ARMS)):  # ECDF from the first positive p (DESIGN s.7)
        if arm in pv:
            x, y, mark = L.ecdf_line(pv[arm], GRID, phase=PHASES[i])
            M.plot(ax, x, y, arm, hero=arm == T.HERO, label=M.name(arm), markevery=mark)
    ax.set(
        xscale="log",
        yscale="log",
        xlim=VIEW,
        ylim=VIEW,
        xlabel="Threshold t",
        ylabel="Share of placebo p ≤ t",
    )
    ax.minorticks_off()
    ticks, labels = [1e-3, 1e-2, 1e-1, 1], ["0.001", "0.01", "0.1", "1"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    plots.ref_line(ax, 0.05, axis="x")
    L.label_line(ax, 0.05, 1.3e-3, "t = 0.05", where="right", va="bottom")
    plots.finish(ax)


def build():
    fig, (a, b) = plots.figure("full", aspect=0.56, ncols=2)
    arms = set(ARMS)
    _panel(a, T.placebo_p_study4(arms))
    _panel(b, T.placebo_p_bridge(arms))
    L.panel_header(a, "a", "Study 4, E1–E5, R2")
    L.panel_header(b, "b", "Simulator logs (bridging), 60 episodes")
    L.tidy_grid([[a, b]])
    M.legend(fig, ARMS, hero=T.HERO)
    return L.finalize(fig)

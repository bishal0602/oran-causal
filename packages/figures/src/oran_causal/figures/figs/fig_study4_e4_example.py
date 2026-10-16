"""Worked example (Study 4 s.4.3): E4 (latent confounding) under R1, R2 and R3 (lam 1), one method
per family. Top: raw rejection rate (p <= .05) of an edge with no effect: the placebo edge in R1 /
R2 (E4 has no truth-null candidate there) and the confounded null edge in R3 (the placebo column
drawn by the confounded policy, C3). Bottom: recall of the true edge (BY declarations; INVALID cells
carry no recall: a cross in the strip above the scale). Rates use the shared symlog rate axis (0
drawn at 0); series are dodged by a few points so equal values stay visible. Cells: 40 seeds, 95 %
cluster-bootstrap CIs over seeds (validity_cells.csv, recall_vs_n.csv).
"""

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures.data import load_csv
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_e4_example",
    width="full",
    paper=True,
    sources=[T.SRC_CELLS, T.SRC_RECALL],
    caption_hint="E4 under R1, R2 and R3 (lam 1): null-edge rejection rate (top) and recall (bottom) for "
    "a design-based, a design-adjusted and two design-blind methods. Colour identifies the "
    "method (the same in every figure); it does not mean validity",
)

REGS = [
    ("R1", "plac_raw", "R1: placebo edge"),
    ("R2", "plac_raw", "R2: placebo edge"),
    ("R3", "conf_raw", "R3: confounded null edge"),
]
ARMS = [
    "pmrt_nl_eq",
    "pcorr_eq",
    "corr",
    "pcorr_native",
]  # one per family + two design-blind (legend order)


def _xaxis(ax):
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(T.NS, ["0.5", "1", "4", "8", "24"])
    ax.set(xlim=(330, 36000), xlabel="n (×10³)")


def build():
    c = T.cells()
    c = c[(c.world == "E4") & (c.lam == 1.0)]
    r = load_csv(*T.EVAL, "csv", "recall_vs_n.csv")
    r = r[(r.world == "E4") & (r.lam == 1.0)]
    fig, axs = plots.figure("full", aspect=0.66, nrows=2, ncols=3)
    for j, (reg, col, tag) in enumerate(REGS):
        top, bot = axs[0, j], axs[1, j]
        for arm in ARMS:
            d = c[(c.arm == arm) & (c.regime == reg)].sort_values("n")
            T.series(top, d.n, d[f"{col}_rate"], d[f"{col}_lo"], d[f"{col}_hi"], arm, ARMS)
            q = r[(r.arm == arm) & (r.regime == reg)].sort_values("n")
            ok = q[q.state != "INVALID"]
            T.series(
                bot,
                ok.n,
                ok.recall_mean,
                ok.recall_lo,
                ok.recall_hi,
                arm,
                ARMS,
                invalid_x=q[q.state == "INVALID"].n,
            )
        plots.ref_line(top, 0.05)
        _xaxis(top)
        L.rate_axis(top, "y", "Null-edge rate" if j == 0 else None)
        plots.finish(top)
        _xaxis(bot)
        L.recall_axis(bot, "Recall" if j == 0 else None)
        plots.finish(bot)
        L.panel_header(top, "abc"[j], tag)
        L.panel_header(bot, "def"[j])
    L.tidy_grid(axs)
    M.legend(fig, ARMS, hero=T.HERO)
    return L.finalize(fig)

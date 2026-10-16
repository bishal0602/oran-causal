"""Specimen: method lines in the paper's mapping ([DECIDED 2026-10-06, user: one palette = specimen
set]; DESIGN s.12 banner), synthetic data. Every series goes through oran_causal.figures.methods by
arm code, each in its own method colour: PMRT-GBM hero (filled), PMRT-Lin (open), design-adjusted
partial correlation (solid), correlation and plain partial correlation (all solid, [DECIDED
2026-10-08]). The design-blind learned methods are shown in specimen_bars.
"""

import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures.figs import _synthetic as syn
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_methods", width="full", paper=False, sources=["synthetic"])
CODES = [
    "pcorr_native",
    "corr",
    "pcorr_eq",
    "pmrt_eq",
    "pmrt_nl_eq",
]  # baselines first, hero last (on top)
HERO = "pmrt_nl_eq"


def build():
    syn.reseed()
    fig, ax = plots.figure("full", aspect=0.5)
    steps = np.arange(21)
    y = np.vstack([syn.utility(steps), syn.utility(steps)[:1] - 0.08])  # 5 synthetic curves
    order = [4, 3, 2, 1, 0]  # hero gets the top curve
    for i, code in enumerate(CODES):
        yy = y[order[i]]
        if code == HERO:
            M.band(ax, steps, yy - 0.03, yy + 0.03, code, hero=True)
        M.plot(ax, steps, yy, code, hero=code == HERO, markevery=(i, 4))
    ax.set(xlabel="Decision step", ylabel="Synthetic score", xlim=(-0.5, 20.5), ylim=(0.3, 1.0))
    plots.finish(ax)
    M.legend(fig, CODES, hero=HERO)
    return L.finalize(fig)

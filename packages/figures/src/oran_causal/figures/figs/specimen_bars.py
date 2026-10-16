"""Specimen: Study 1 style method bars in the paper's mapping ([DECIDED 2026-10-06, user: one
palette = specimen set]; DESIGN s.12 banner, s.12.1 rule 5), synthetic data: PMRT hero solid;
correlation, Granger, Granger + BY, SHAP-GBDT and two-tower each a full-strength bar in its method
colour ([DECIDED 2026-10-08]); 95 % CI whiskers in ink.
"""

import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures.figs import _synthetic as syn
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_bars", width="full", paper=False, sources=["synthetic"])
CODES = ["pmrt", "corr", "granger_dev", "granger_by", "shap_gbdt", "two_tower"]
GROUPS = ["Precision", "Recall", "F1"]


def build():
    syn.reseed()
    vals = np.clip(
        np.array([[0.93, 0.62, 0.70, 0.78, 0.66, 0.58]]) + syn.RNG.normal(0, 0.04, (3, 6)),
        0.05,
        0.98,
    )
    err = 0.02 + syn.RNG.uniform(0, 0.03, (3, 6))
    fig, ax = plots.figure("full", aspect=0.42)
    w = 0.8 / len(CODES)
    for k, code in enumerate(CODES):
        x = np.arange(len(GROUPS)) - 0.4 + w * (k + 0.5)
        M.bar(ax, x, vals[:, k], code, width=w * 0.92, hero=code == "pmrt")
        ax.errorbar(
            x,
            vals[:, k],
            yerr=err[:, k],
            fmt="none",
            ecolor=P.INK,
            elinewidth=0.6,
            capsize=1.5,
            capthick=0.6,
            zorder=4,
        )
    ax.set_xticks(range(len(GROUPS)), GROUPS)
    ax.set(ylabel="Synthetic score", ylim=(0, 1))
    plots.finish(ax, trim=False)
    M.legend(fig, CODES, hero="pmrt")
    return L.finalize(fig)

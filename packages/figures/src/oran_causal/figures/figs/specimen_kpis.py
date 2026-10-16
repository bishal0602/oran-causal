"""Specimen: KPI lines in the paper's (PROVISIONAL) KPI slots ([DECIDED 2026-10-06, user: one
palette = specimen set]; DESIGN s.12.2), synthetic data, each KPI with its slot marker and a direct
label (energy and RLF share the plot, rule 6). No planned paper figure uses KPI hues yet.
"""

import numpy as np

from oran_causal.figures import entities as E
from oran_causal.figures import plots
from oran_causal.figures.figs import _synthetic as syn
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_kpis", width="full", paper=False, sources=["synthetic"])


def build():
    syn.reseed()
    fig, ax = plots.figure("full", aspect=0.42)
    steps = np.arange(21)
    for k, key in enumerate(E.KPIS):
        y = 0.22 + 0.16 * k + 0.03 * np.sin(steps / (3 + k)) + syn.RNG.normal(0, 0.006, steps.size)
        E.plot_kpi(ax, steps, y, key, markevery=(k, 4))
    ax.set(
        xlabel="Decision step",
        ylabel="Normalised KPI (synthetic)",
        xlim=(-0.5, 20.5),
        ylim=(0.1, 1.0),
    )
    plots.finish(ax)
    return fig

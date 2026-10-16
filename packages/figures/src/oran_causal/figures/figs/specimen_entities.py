"""Specimen: xApps in the paper's mapping ([DECIDED 2026-10-06, user: one palette = specimen set];
DESIGN s.12.2), synthetic data. (a) 100 % stacked shares of the three xApps of the studied scenario,
(b) one line per xApp (entities.plot_xapp: entity style, slot markers). KPIs are in specimen_kpis:
one entity type per figure.
"""

import numpy as np

from oran_causal.figures import entities as E
from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _synthetic as syn
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_entities", width="full", paper=False, sources=["synthetic"])
KEYS = ["ES", "PowerES", "SliceGuarantee"]


def build():
    syn.reseed()
    fig, (a, b) = plots.figure("full", aspect=0.5, ncols=2)
    shares = syn.SHARES[:, :3] / syn.SHARES[:, :3].sum(1, keepdims=True)
    rows = np.arange(len(shares))[::-1]
    left = np.zeros(len(shares))
    for k, key in enumerate(KEYS):
        e = E.xapp(key)
        a.barh(
            rows,
            shares[:, k],
            left=left,
            height=0.62,
            color=e.color,
            edgecolor=P.WHITE,
            linewidth=1.0,
            gid=f"xapp:{key}",
        )
        for r, (lft, s) in enumerate(zip(left, shares[:, k], strict=False)):
            a.text(
                lft + s / 2,
                rows[r],
                f"{s:.0%}",
                ha="center",
                va="center",
                fontsize=S.ANNOT_PT,
                color=plots._text_on(e.color),
            )
        left += shares[:, k]
    a.set_yticks(rows, [f"Scenario {i + 1}" for i in range(len(shares))])
    a.set(xlim=(0, 1), xlabel="Share of requests (synthetic)")
    a.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plots.finish(a, trim=False)
    steps = np.arange(21)
    for k, key in enumerate(KEYS):
        y = [0.62, 0.55, 0.70][k] + [0.25, 0.3, 0.12][k] * (1 - np.exp(-steps / [4, 7, 3][k]))
        E.plot_xapp(
            b,
            steps,
            y + syn.RNG.normal(0, 0.012, steps.size),
            key,
            markevery=(k, 5),
            label=E.xapp(key).name,
        )
    b.set(xlabel="Decision step", ylabel="Synthetic score", xlim=(-0.5, 20.5), ylim=(0.4, 1.0))
    plots.finish(b)
    L.panel_header(a, "a")
    L.panel_header(b, "b")
    L.fig_legend(fig, *b.get_legend_handles_labels())
    return L.finalize(fig)

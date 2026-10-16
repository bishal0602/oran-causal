"""Specimen: (a) sequential CMI heatmap with threshold outlines, (b) diverging signed-effect heatmap."""

from oran_causal.figures import layout as L
from oran_causal.figures import plots
from oran_causal.figures.figs import _synthetic as syn
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_heatmaps", width="full", paper=False, sources=["synthetic"])
TAU = 0.12


def build():
    fig, (a, b) = plots.figure("full", aspect=0.5, ncols=2)
    plots.heatmap(
        a,
        syn.CMI,
        syn.NCPS,
        syn.KPIS,
        fmt="{:.2f}",
        outline_mask=syn.CMI >= TAU,
        cbar_label="CMI (nats)",
    )
    plots.heatmap(
        b,
        syn.SIGNED,
        syn.NCPS,
        syn.KPIS,
        diverging=True,
        vmax=1.0,
        fmt="{:.2f}",
        cbar_label="Signed effect",
    )
    L.tidy_grid([[a, b]])
    L.panel_header(a, "a")
    L.panel_header(b, "b")
    return L.finalize(fig)

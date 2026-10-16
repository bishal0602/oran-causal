"""Specimen: recovered causal graph vs ground truth (half width)."""

from oran_causal.figures import plots
from oran_causal.figures.graph import Edge, draw_causal_graph
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_graph", width="half", paper=False, sources=["synthetic"])


def build():
    fig, ax = plots.figure("half", aspect=1.0)
    ncps, kpis = ["Ptx", "CIO", "TTT", "Hys"], ["Thr", "Lat", "HO", "RLF"]
    edges = [
        Edge("Ptx", "Thr", "correct"),
        Edge("Ptx", "RLF", "correct"),
        Edge("CIO", "HO", "correct"),
        Edge("TTT", "HO", "correct"),
        Edge("Hys", "HO", "missed"),
        Edge("CIO", "RLF", "correct"),
        Edge("CIO", "Lat", "spurious"),
        Edge("HO", "RLF", "correct"),
    ]
    draw_causal_graph(ax, ncps, kpis, edges)
    return fig

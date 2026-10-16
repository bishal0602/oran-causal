"""Specimen: architecture diagram (full width) in the paper's mapping ([DECIDED 2026-10-06, user:
one palette = specimen set]; DESIGN s.9 / s.12.2): the three xApps of the studied scenario in their
entity slots (1.0 pt outline, ink label), the referee as the one hero component (blue), everything
else neutral, arrows ink.
"""

from oran_causal.figures import diagram as D
from oran_causal.figures import entities as E
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_diagram", width="full", paper=False, sources=["synthetic"])


def build():
    fig, ax = D.canvas("full", height_pt=180)
    cx, cy, cw, ch = 112, 8, 254, 160
    D.container(ax, cx, cy, cw, ch, "Near-RT RIC")
    ax.text(8, 22, "E2 nodes", fontsize=8, fontweight="bold", va="center")
    for i, n in enumerate(["O-CU", "O-DU", "O-RU"]):
        D.box(ax, 8, 32 + i * 42, 72, 34, n)
    for i, k in enumerate(["ES", "PowerES", "SliceGuarantee"]):
        E.xapp_box(ax, cx + 12 + i * 80, cy + 24, 70, 32, k, sub="xApp")
        D.arrow(ax, [(cx + 47 + i * 80, cy + 56), (cx + 47 + i * 80, cy + 81)])
    D.box(ax, cx + 12, cy + 82, 150, 34, "Referee", "accepts or rejects requests", role="proposed")
    D.box(ax, cx + 176, cy + 82, 66, 34, "Causal map")
    D.box(ax, cx + 12, cy + 132, 230, 20, "Shared data layer (KPIs, NCP settings)")
    D.arrow(ax, [(cx + 175, cy + 99), (cx + 163, cy + 99)])
    D.arrow(ax, [(cx + 209, cy + 131), (cx + 209, cy + 117)])
    D.arrow(ax, [(80, 49), (cx - 1, 49)], label="E2 KPM")
    D.arrow(
        ax,
        [
            (cx + 87, cy + 116),
            (cx + 87, cy + 124),
            (cx + 248, cy + 124),
            (cx + 248, cy + 166),
            (cx - 14, cy + 166),
            (cx - 14, 91),
            (81, 91),
        ],
        dashed=True,
        label="E2 control",
        label_at=(96, 87),
    )
    return fig

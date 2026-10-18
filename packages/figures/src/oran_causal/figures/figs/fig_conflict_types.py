"""Figure 1: the three O-RAN WG3 conflict types differ in where two xApps meet.

Grammar (Sharma et al. 2025, Fig. 2; O-RAN WG3 TR ConMit R004 s.4.2.1): xApps in the near-RT RIC write RAN control
parameters over E2 (dashed, control), and parameters move KPIs in the RAN (solid). Each row is one conflict type on
the same three columns, so the rows differ only in where the conflict sits: at a shared parameter (direct), at a
shared KPI reached through different parameters (indirect), or outside the declared links, as a side effect of one
xApp's parameter on the other xApp's KPI (implicit). The conflict is the only coloured mark (semantic BAD, gid
``sem:bad``): a flagged box (``diagram.box(role="bad")``) on the shared parameter (a), the shared KPI (b) and the
degraded KPI (c); the hidden side effect itself is the dotted "degrades" arrow (c). Everything else is B0 diagram
furniture from ``diagram``: white boxes, a FIELD card for the near-RT RIC, plain ink arrows with the word beside the
line.

Examples and sources (cited in the text; unchanged from the earlier Figure 1, oran_causal/figures/svg/fig_intro_conflict.py):
  direct    ES lowers and CCO raises transmit power: QACM (Wadud et al., IEEE TGCN 2024), Sec. VIII-A.
  indirect  MRO writes the CIO, MLB the antenna tilt; both move the handover boundary: ConMit R004 s.4.2.1 ("antenna
            tilts and measurement offsets are different control points, but they both impact the handover boundary");
            Wadud et al., INFOCOM WKSHPS 2025, Sec. II.
  implicit  protecting GBR throughput degrades non-GBR throughput: ConMit R004 s.4.2.1. WG3 names no parameter, so
            both are drawn as "own parameter".
No data: definitions and literature examples only. Chosen 2026-10-13 from three concepts (F1R; the concept code is
kept in reviews/2026-10-13/briefs/F1R_concepts.py); revised 2026-10-14 after the user review (F1R2).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from oran_causal.figures import diagram as D
from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures import style as S
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_conflict_types",
    width="full",
    paper=True,
    sources=[
        "archive/reference-papers/O-RAN.WG3.TR.ConMit-R004-v01.00_.docx s.4.2.1",
        "archive/reference-papers/QACM_QoS-Aware_xApp_Conflict_Mitigation_in_Open_RAN.pdf s.VIII-A",
    ],
    caption_hint="O-RAN conflict types.",
)

H = 154.0  # canvas height (pt); the brief caps it at 2.2 in
HEADER_Y = 10.0
BOX_H, PITCH = 14.0, 22.0  # box height; vertical distance of the two boxes in a row
ROW_TOP, ROW_GAP = 20.0, 8.0
ROW_H = PITCH + BOX_H
NAME_X = 13.0
CARD_PAD = 6.0  # near-RT RIC card around the xApp boxes
SHARED_DY = 3.5  # two arrows meeting at one box side keep this far either side of its centre
MARGIN = 1.5  # an arrow stops this far from the box edge

# columns: centre x and box width (pt)
COLS = {"x": (114.5, 59.0), "p": (215.0, 60.0), "k": (328.0, 80.0)}

# one row per conflict type: xApps, parameters, KPIs (a single item is shared and drawn centred)
TYPES = [
    {
        "letter": "a",
        "name": "Direct",
        "sub": "same parameter",
        "x": ["ES", "CCO"],
        "p": ["transmit power"],
        "k": ["power consumption", "DL throughput"],
        "bad": ("p", 0),
        "edges": [
            ("x0", "p0", "lower"),
            ("x1", "p0", "raise"),
            ("p0", "k0", None),
            ("p0", "k1", None),
        ],
    },
    {
        "letter": "b",
        "name": "Indirect",
        "sub": "same KPI",
        "x": ["MRO", "MLB"],
        "p": ["CIO", "antenna tilt"],
        "k": ["handover boundary"],
        "bad": ("k", 0),
        "edges": [
            ("x0", "p0", "E2"),
            ("x1", "p1", None),
            ("p0", "k0", None),
            ("p1", "k0", None),
        ],
    },
    {
        "letter": "c",
        "name": "Implicit",
        "sub": "side effect only",
        "x": ["GBR xApp", "non-GBR xApp"],
        "p": ["own parameter", "own parameter"],
        "k": ["GBR throughput", "non-GBR throughput"],
        "bad": ("k", 1),
        "edges": [("x0", "p0", None), ("x1", "p1", None), ("p0", "k0", None), ("p1", "k1", None)],
        "side": ("p0", "k1", "degrades"),
    },
]
HEADERS = {"p": "RAN parameters", "k": "KPIs"}


def _text(ax, x, y, s, ha="center", va="baseline", weight="normal"):
    return ax.text(
        x, y, s, ha=ha, va=va, fontsize=S.ANNOT_PT, fontweight=weight, color=P.INK, zorder=6
    )


def _row(ax, i, ty):
    y0 = ROW_TOP + i * (ROW_H + ROW_GAP)
    ym = y0 + ROW_H / 2
    base = (
        0.36 * S.ANNOT_PT
    )  # half the cap height: a baseline this far below a centre aligns with a label
    let = ax.text(
        2.0,
        ym - PITCH / 2 + base,
        ty["letter"],
        fontsize=S.PANEL_PT,
        fontweight="bold",
        color=P.INK,
        va="baseline",
    )
    let.set_gid("panel-letter")
    _text(ax, NAME_X, ym - PITCH / 2 + base, ty["name"], ha="left", weight="bold")
    _text(ax, NAME_X, ym + PITCH / 2 + base, ty["sub"], ha="left")
    pos = {}
    for t in "xpk":
        cx, w = COLS[t]
        items = ty[t]
        for j, lab in enumerate(items):
            cy = ym if len(items) == 1 else ym + (j - 0.5) * PITCH
            pos[f"{t}{j}"] = (cx, cy, w)
            role = "bad" if ty["bad"] == (t, j) else "neutral"
            D.box(ax, cx - w / 2, cy - BOX_H / 2, w, BOX_H, lab, role=role)

    side = ty.get("side")

    def ends(a, b, dy_a=0.0, dy_b=0.0):
        (xa, ya, wa), (xb, yb, wb) = pos[a], pos[b]
        return (xa + wa / 2 + MARGIN, ya + dy_a), (xb - wb / 2 - MARGIN, yb + dy_b)

    for a, b, lab in ty["edges"]:
        # arrows sharing a box side keep apart: the side effect and its neighbours split the side
        dy_a = -SHARED_DY if side and a == side[0] else 0.0
        dy_b = (int(a[1]) - 0.5) * 2 * SHARED_DY if len(ty[b[0]]) == 1 else 0.0
        if side and b == side[1]:
            dy_a, dy_b = (SHARED_DY if a[0] == "p" else 0.0), SHARED_DY
        p0, p1 = ends(a, b, dy_a, dy_b)
        up = a[1] == "0"
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        D.arrow(
            ax,
            [p0, p1],
            lab,
            dashed=a[0] == "x",
            label_at=(mid[0], mid[1] + (-2.5 if up else 2.5)),
            label_va="bottom" if up else "top",
        )
    if side:
        a, b, lab = side
        p0, p1 = ends(a, b, SHARED_DY, -SHARED_DY)
        D.arrow(
            ax,
            [p0, p1],
            lab,
            dotted=True,
            flag=True,
            label_at=((p0[0] + p1[0]) / 2 - 1.0, (p0[1] + p1[1]) / 2 + 6.0),
            label_va="top",
        )
    return pos


def build():
    fig, ax = D.canvas("full", height_pt=H)
    cx, w = COLS["x"]
    top = 1.0
    bottom = ROW_TOP + len(TYPES) * ROW_H + (len(TYPES) - 1) * ROW_GAP + CARD_PAD
    D.container(
        ax,
        cx - w / 2 - CARD_PAD,
        top,
        w + 2 * CARD_PAD,
        bottom - top,
        "Near-RT RIC",
        title_dy=HEADER_Y - top,
        ha="center",
    )
    for t, label in HEADERS.items():
        _text(ax, COLS[t][0], HEADER_Y, label, weight="bold", va="center_baseline")
    for i, ty in enumerate(TYPES):
        _row(ax, i, ty)

    # build-time checks: one flagged box per row, plus the implicit row's dotted arrow and its head;
    # the side effect only in the implicit row, the E2 word once, everything inside the canvas
    marks = [a for a in (*ax.patches, *ax.lines) if a.get_gid() == "sem:bad"]
    boxes = [m for m in marks if hasattr(m, "get_boxstyle")]
    assert len(boxes) == len(TYPES) and len(marks) == len(TYPES) + 2, len(marks)
    assert sum(isinstance(m, FancyArrowPatch) for m in marks) == 1
    assert [("side" in ty) for ty in TYPES] == [False, False, True]
    assert sum(e[2] == "E2" for ty in TYPES for e in ty["edges"]) == 1
    assert bottom + 4.0 <= H
    assert (
        H / 72 + 2 * plt.rcParams["savefig.pad_inches"] <= 2.2
    )  # the PDF, tight-bbox padding included
    assert cx - w / 2 - CARD_PAD > NAME_X + 60.0  # the card clears the row names
    assert COLS["k"][0] + COLS["k"][1] / 2 <= S.FULL * 72 - D.EDGE_PT
    return L.finalize(fig)

"""Figure 1: the three O-RAN WG3 conflict types differ in where two xApps meet.

Grammar (Sharma et al. 2025, Fig. 2; O-RAN WG3 TR ConMit R004 s.4.2.1): xApps in the near-RT RIC write RAN control
parameters over E2 (dashed, control), and parameters move KPIs in the RAN (solid). Each row is one conflict type on
the same three columns, so the rows differ only in where the conflict sits: at a shared parameter (direct), at a
shared KPI reached through different parameters (indirect), or outside the declared links, as a side effect of one
xApp's parameter on the other xApp's KPI (implicit). The conflict is the only coloured mark (semantic BAD, gid
``sem:bad``), drawn as a solid outline on the shared parameter (a), the shared KPI (b) and the degraded KPI (c); the
hidden side effect itself is the dotted "degrades" arrow (c). Everything else is ink and gray; xApps are identified by
label only.

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
LW = 0.7  # structure: arrows
LW_BOX = 0.6  # structure: box and frame outlines
LW_BAD = 1.2  # emphasis: the conflict
HEAD = "-|>,head_length=4,head_width=2.2"
DASH = (0, (3, 1.6))  # E2 control (DESIGN s.10: dashed for control)
DOT = (0, (0.1, 2.0))  # hidden side effect: round dots at the emphasis weight
FS = S.ANNOT_PT  # 8 pt everywhere except the panel letters
BAD = P.BAD

# columns: centre x and box width (pt)
COLS = {"x": (112.0, 62.0), "p": (220.0, 60.0), "k": (326.8, 80.0)}
FRAME_PAD = (
    3.0  # near-RT RIC frame around the xApp boxes (sides, top); the bottom also carries its label
)
FRAME_PAD_BOTTOM = 5.0
BOX_H, PITCH = 14.0, 22.0  # box height; vertical distance of the two boxes in a row
HEADER_Y = 8.5
ROW_TOP, ROW_GAP = 15.0, 10.5
ROW_H = PITCH + BOX_H
NAME_X = 13.0
SHARED_DY = 3.5  # two arrows into one box land this far either side of its centre
SIDE_DX = 16.0  # the side effect leaves its parameter box this far right of the box centre

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
            ("x0", "p0", "E2 control"),
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
KIND = {"x": "xapp", "p": "param", "k": "kpi"}
HEADERS = {"x": "xApps", "p": "RAN parameters", "k": "KPIs"}


def _text(ax, x, y, s, ha="center", va="baseline", weight="normal", size=FS, **kw):
    return ax.text(
        x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight, color=P.INK, zorder=6, **kw
    )


def _box(ax, cx, cy, w, label, kind, bad=False):
    """xApp: gray tint, rounded, bold label; parameter: white, square; KPI: white pill. A conflict
    element keeps its fill and gets a solid BAD outline at the emphasis weight.
    """
    r = {"xapp": 2.5, "param": 0.8, "kpi": BOX_H / 2}[kind]
    fc = P.tint(P.NEUTRAL, 0.14) if kind == "xapp" else P.WHITE
    ec, lw = (BAD, LW_BAD) if bad else (P.NEUTRAL, LW_BOX)
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - BOX_H / 2),
            w,
            BOX_H,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=3,
            gid="sem:bad" if bad else f"diagram:{kind}",
        )
    )
    _text(ax, cx, cy, label, va="center_baseline", weight="bold" if kind == "xapp" else "normal")


def _arrow(ax, a, b, ls="-", bad=False):
    if ls != "-":  # a patterned shaft, then a solid head (a dashed head outline comes out ragged)
        (xa, ya), (xb, yb) = a, b
        n = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
        ux, uy = (xb - xa) / n, (yb - ya) / n
        ax.plot(
            [xa, xb - 4.5 * ux],
            [ya, yb - 4.5 * uy],
            color=P.INK,
            lw=LW,
            ls=ls,
            marker="",
            dash_capstyle="butt",
            zorder=4,
            gid="arrow",
        )
        a = (xb - 4.5 * ux, yb - 4.5 * uy)
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle=HEAD,
            mutation_scale=1,
            color=BAD if bad else P.INK,
            lw=LW_BAD if bad else LW,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
            gid="sem:bad" if bad else "arrow",
            joinstyle="miter",
            capstyle="butt",
        )
    )


def _side_effect(ax, src, dst, ym, label):
    """Dotted BAD arrow from below the source parameter box, along the row's midline (between the
    two box rows) and down into the top of the degraded KPI, its word on the arrow over the
    parameter-to-KPI gap.
    """
    (xs, ys, _), (xd, yd, wd) = src, dst
    x0, x1 = xs + SIDE_DX, xd - wd / 2 + 13.0
    ax.plot(
        [x0, x0, x1],
        [ys + BOX_H / 2 + 0.8, ym, ym],
        color=BAD,
        lw=LW_BAD,
        ls=DOT,
        marker="",
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=4,
        gid="sem:bad",
    )
    ax.add_patch(
        FancyArrowPatch(
            (x1, ym - 0.6),
            (x1, yd - BOX_H / 2 - 0.8),
            arrowstyle=HEAD,
            mutation_scale=1,
            color=BAD,
            lw=LW_BAD,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
            gid="sem:bad",
        )
    )
    gap_l, gap_r = COLS["p"][0] + COLS["p"][1] / 2, COLS["k"][0] - COLS["k"][1] / 2
    _text(
        ax,
        (gap_l + gap_r) / 2 + 0.6,
        ym,
        label,
        va="center_baseline",
        bbox={"boxstyle": "square,pad=0.12", "fc": P.WHITE, "ec": "none"},
    )


def _row(ax, i, ty):
    y0 = ROW_TOP + i * (ROW_H + ROW_GAP)
    ym = y0 + ROW_H / 2
    base = (
        0.36 * FS
    )  # half the cap height: a baseline this far below a box centre aligns with its label
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
            _box(ax, cx, cy, w, lab, KIND[t], ty["bad"] == (t, j))

    def ends(a, b, dy_b=0.0):
        (xa, ya, wa), (xb, yb, wb) = pos[a], pos[b]
        return (xa + wa / 2 + 1.5, ya), (xb - wb / 2 - 1.5, yb + dy_b)

    for a, b, lab in ty["edges"]:
        dy_b = (int(a[1]) - 0.5) * 2 * SHARED_DY if len(ty[b[0]]) == 1 else 0.0
        p0, p1 = ends(a, b, dy_b=dy_b)
        _arrow(ax, p0, p1, ls=DASH if a[0] == "x" else "-")
        if lab:  # above the upper arrow / below the lower arrow of the row
            up = a[1] == "0"
            _text(
                ax,
                (p0[0] + p1[0]) / 2,
                (p0[1] + p1[1]) / 2 + (-2.5 if up else 2.5),
                lab,
                va="bottom" if up else "top",
            )
    if "side" in ty:
        a, b, lab = ty["side"]
        _side_effect(ax, pos[a], pos[b], ym, lab)
    return pos


def build():
    fig, ax = D.canvas("full", height_pt=H)
    for t, (cx, _) in COLS.items():
        _text(ax, cx, HEADER_Y, HEADERS[t], weight="bold")
    rows = [_row(ax, i, ty) for i, ty in enumerate(TYPES)]
    # the near-RT RIC frame encloses the xApp boxes only (first to last), its name on the bottom
    # edge
    cx, w = COLS["x"]
    top = rows[0]["x0"][1] - BOX_H / 2 - FRAME_PAD
    bottom = rows[-1]["x1"][1] + BOX_H / 2 + FRAME_PAD_BOTTOM
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2 - FRAME_PAD, top),
            w + 2 * FRAME_PAD,
            bottom - top,
            boxstyle="round,pad=0,rounding_size=4",
            fc="none",
            ec=P.NEUTRAL_LIGHT,
            lw=LW_BOX,
            ls=(0, (3, 2)),
            zorder=1,
            gid="container",
        )
    )
    _text(
        ax,
        cx,
        bottom,
        "Near-RT RIC",
        va="center_baseline",
        bbox={"boxstyle": "square,pad=0.15", "fc": P.WHITE, "ec": "none"},
    )

    # build-time checks: one solid conflict outline per row, plus the implicit row's dotted path and
    # its head; the side effect only in the implicit row, the E2 label once, everything inside the
    # canvas
    marks = [a for a in (*ax.patches, *ax.lines) if a.get_gid() == "sem:bad"]
    outlines = [m for m in marks if isinstance(m, FancyBboxPatch)]
    assert len(marks) == len(TYPES) + 2 and len(outlines) == len(TYPES), len(marks)
    assert all(
        m.get_linestyle() in ("-", "solid") for m in outlines
    )  # every conflict outline is solid
    paths = [m for m in marks if not isinstance(m, FancyArrowPatch | FancyBboxPatch)]
    assert len(paths) == 1
    assert paths[0].get_linestyle() not in (
        "-",
        "solid",
    )
    assert [("side" in ty) for ty in TYPES] == [False, False, True]
    assert sum(e[2] == "E2 control" for ty in TYPES for e in ty["edges"]) == 1
    assert top > HEADER_Y + 2.0
    assert bottom + 4.0 <= H
    assert (
        H / 72 + 2 * plt.rcParams["savefig.pad_inches"] <= 2.2
    )  # the PDF, tight-bbox padding included
    assert COLS["x"][0] - COLS["x"][1] / 2 - FRAME_PAD > NAME_X + 60.0  # frame clears the row names
    assert COLS["k"][0] + COLS["k"][1] / 2 <= S.FULL * 72 - D.EDGE_PT
    return L.finalize(fig)

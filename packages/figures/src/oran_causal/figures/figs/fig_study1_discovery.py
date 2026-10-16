"""Study 1: discovery from randomised logs (system-level simulator, 600 EVAL episodes; verdict
PARTIAL).

Edge ledger in the Harbor grammar [DECIDED 2026-10-08, user; prototype "B v4", palette "H2";
replaces the five-panel bar figure]. Methods are labelled table rows under family banners; colour in
(a-c) shows edge status, not the method (DESIGN s.12.1 exception): teal = declared with the
ground-truth sign, terracotta = declared with the wrong sign (solid) or false (light terracotta: a
NULL edge or the placebo logs), gray = ground truth, not scored (INDET) or missed (open circle). A
triangle gives the sign of an effect (up +, down -). (a) Indirect edges: the 10 indirect
(neighbour-relation) edges, all GT-TRUE, one column each (v4_analysis
slices_result.pooled.metrics.<code>.declared, gt.cells); PMRT's "9 combinations" row counts the
statistic x layer combinations that declared each edge. (b) Other 50 edges: the remaining
declarations as a stacked bar (right sign / wrong sign / NULL / INDET), total at the end. (c)
Placebo logs: edges declared on the 40 placebo episodes (FD v4.placebo_declarations; v4_analysis
placebo_descriptive.<code>.declared). Text column: pooled indirect F1 and overall precision (FD
v4.pooled); bold below the P3 threshold (v4_verdict.json P3.rule.precision). [G5 2026-10-12, P3] The
old panels (d, e), PMRT minus each baseline per subsample, are the ESM figure fig_study1_subsamples
(restructure plan: Fig 4d-e to the ESM); the body figure keeps (a-c). Every count is re-derived from
the stored declarations and asserted against the stored TP, FP, sign accuracy, F1, precision,
placebo counts, P2 wins and S1.10 ranges (_study1.load). Derived and not stored as keys: the
wrong-sign split of each method's hits and the per-edge counts of the 9 PMRT combinations.
"""

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory, offset_copy

from oran_causal.figures import entities as E
from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.data import source_note
from oran_causal.figures.figs import _s12a as H
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _study1 as D
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study1_discovery",
    width="full",
    paper=True,
    sources=[ST.SRC_FD, ST.SRC_V4, source_note(*ST.DECISION, "v4_verdict.json")],
    caption_hint="Study 1 discovery from 600 randomised evaluation episodes of the system-level simulator, against the "
    "knockout ground truth. (a) The 10 indirect (neighbour-relation) edges, all TRUE; a triangle gives the "
    "sign of an effect (up +, down -): gray, the ground truth; teal, a declaration with the ground-truth "
    "sign; terracotta, a declaration with the wrong sign; an open gray circle is a missed edge. Chain edges "
    "are bold. The row under PMRT gives how many of its 9 statistic x declaration-layer combinations "
    "declared each edge. (b) The method's declarations among the other 50 edges (ground truth: 19 TRUE, "
    "21 NULL, 10 INDET): right sign (teal), wrong sign (terracotta), false, on a NULL edge (light "
    "terracotta), and not scored, on an INDET edge (gray), with the total. (c) Edges declared on the 40 "
    "placebo episodes, all false. Right: pooled indirect F1 and overall precision; bold, below the P3 "
    "threshold 0.80. Colours show edge status, not the method. QACM scores its own cell only, so "
    "its indirect F1 is 0 by construction. Source: figures_data.json (v4) and the Study 1 analysis file.",
)

KPI_ORDER = (
    "load",
    "e",
    "v",
    "pv",
    "rlf",
)  # the wide label first in each group, so neighbours have room
KPI_X = (
    0.0,
    1.45,
    2.4,
    3.45,
    4.65,
)  # [G5 2026-10-12] uneven, so the 8 pt labels clear (was 0-4.5 at 7 pt)
GROUP_X = 6.75  # offset of the second (sleep) group ([G5] was 6.6 at 7 pt;
# [G3] room between the two blocks for the separator)
SPAN_PT, GROUP_PT = (
    -14.5,
    -19.0,
)  # [G3] y offset (pt below the axes) of the group rule and group label
# ([G5] 8 pt labels; was -13.5, -18.0)
OK, ERR, MISS, GT = P.TEAL, P.TERRACOTTA, P.MISSED, P.NEUTRAL  # edge status (palette "H2")
NULL_FC, UNSC_FC = P.tint(P.TERRACOTTA, 0.25), P.tint(P.NEUTRAL, 0.45)
MS = 5.0  # status marker size (pt)
THICK = 0.56  # bar thickness (row units)
BASE7 = ["corr", "granger", "granger_by", "shap_gbdt", "two_tower", "qacm", "int"]
HEIGHT = 3.2  # [G5] was 4.5 with the (d, e) row
FAM_BASED, FAM_BLIND = "Design-based tests", "Design-blind methods"


def tri(sign):
    return "^" if sign > 0 else "v"


def rows():
    r = [
        ("Ground truth", ["gt"]),
        (FAM_BASED, None),
        ("PMRT", [D.PRIMARY, "all9"]),
        (FAM_BLIND, None),
    ]
    r += [(H.name13(c), [c]) for c in BASE7]
    return L.method_rows(r, gap=0.55)


def edge_status(d, code):
    """Per-method counts of the declarations outside the indirect-TRUE block, and the indirect cells."""
    g, dec = d["gt"], d["pooled"][code]["declared"]
    ind = set(d["fd"]["gt"]["nbr_true"])
    cells = {h: (dec[h], dec[h] == g[h]["sign"]) for h in ind if h in dec}
    other = {"ok": 0, "ws": 0, "null": 0, "indet": 0}
    for h, s in dec.items():
        if h in ind:
            continue
        st = g[h]["status"]
        other[
            "indet"
            if st == "INDET"
            else "null"
            if st == "NULL"
            else "ok"
            if s == g[h]["sign"]
            else "ws"
        ] += 1
    sc = d["pooled"][code]["score"]
    ws_ind = sum(1 for _, ok in cells.values() if not ok)
    assert len(cells) == sc["ind"]["tp"]
    assert other["null"] == sc["ov"]["fp"]
    assert other["ok"] + other["ws"] + len(cells) == sc["ov"]["tp"]
    assert round(sc["ov"]["tp"] * sc["ov"]["sign_acc"]) == other["ok"] + len(cells) - ws_ind
    assert sum(other.values()) + len(cells) == len(dec)
    return cells, other


def build():
    d = D.load()
    S.apply()
    fig = plt.figure(figsize=(S.FULL, HEIGHT), layout="constrained")
    a, b, t = fig.subplots(1, 3, sharey=True, gridspec_kw={"width_ratios": [1.5, 1.0, 1.17]})
    c = t  # [G5] (c) is the text column: placebo counts and scores
    banners, groups, subrows, ylim = rows()
    ylim = (ylim[0], ylim[1] + 0.9)  # head room for the text-column heads
    ys = {key: y for y, _, key in subrows}
    gt = d["gt"]
    # (a) indirect edges
    # ---------------------------------------------------------------------------------------------
    xs = {}
    for i, fam in enumerate(("ptx", "sleep")):
        for j, k in enumerate(KPI_ORDER):
            xs[f"{fam}|nbr|{k}"] = i * GROUP_X + KPI_X[j]
    order = list(xs)
    for sgn in (1, -1):
        hx = [xs[h] for h in order if gt[h]["sign"] == sgn]
        if hx:
            a.plot(
                hx,
                [ys["gt"]] * len(hx),
                ls="",
                marker=tri(sgn),
                ms=MS,
                mfc=GT,
                mec=P.WHITE,
                mew=0.4,
                zorder=3,
            )
    for code in [D.PRIMARY, *BASE7]:
        cells, _ = edge_status(d, code)
        y = ys[code]
        for ok in (True, False):
            for sgn in (1, -1):
                hx = [xs[h] for h, (s, o) in cells.items() if o == ok and s == sgn]
                if hx:
                    a.plot(
                        hx,
                        [y] * len(hx),
                        ls="",
                        marker=tri(sgn),
                        ms=MS,
                        mfc=OK if ok else ERR,
                        mec=P.WHITE,
                        mew=0.4,
                        zorder=3,
                    )
        miss = [xs[h] for h in order if h not in cells]
        if miss:
            a.plot(
                miss,
                [y] * len(miss),
                ls="",
                marker="o",
                ms=MS - 1.6,
                mfc=P.WHITE,
                mec=MISS,
                mew=0.7,
                zorder=3,
            )
    for h in order:  # all 9 combinations
        n = sum(h in d["pooled"][cc]["declared"] for cc in D.PMRT9)
        a.text(
            xs[h], ys["all9"], str(n), ha="center", va="center", fontsize=S.ANNOT_PT, color=P.INK
        )
    a.set_xlim(-0.7, GROUP_X + KPI_X[-1] + 0.5)
    a.set_xticks(
        [xs[h] for h in order], [h.split("|")[2] for h in order], fontsize=S.ANNOT_PT, rotation=0
    )
    for tl, h in zip(a.get_xticklabels(), order, strict=False):
        if h in D.CHAIN:
            tl.set_fontweight("bold")
    # [G3 2026-10-12] the two KPI blocks read as groups: a light rule between them through the
    # matrix, and a short rule under each block's KPI labels (a table spanner) above its
    # control-parameter label
    a.axvline((KPI_X[-1] + GROUP_X) / 2, color=P.GRID, lw=0.8, zorder=0.5, gid="group-separator")
    span = offset_copy(
        blended_transform_factory(a.transData, a.transAxes), fig=fig, y=SPAN_PT, units="points"
    )
    for i, fam in enumerate(("ptx", "sleep")):
        (rule,) = a.plot(
            [i * GROUP_X + KPI_X[0] - 0.45, i * GROUP_X + KPI_X[-1] + 0.35],
            [0, 0],
            color=P.NEUTRAL,
            lw=0.5,
            marker="",
            transform=span,
            clip_on=False,
            gid="group-rule",
        )
        rule.set_in_layout(False)
        a.annotate(
            E.param_label(fam),
            xy=(i * GROUP_X + KPI_X[-1] / 2, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, GROUP_PT),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=S.ANNOT_PT,
            color=P.INK,
        )
    a.tick_params(axis="x", length=0)
    a.grid(False)
    a.spines["bottom"].set_visible(False)
    # (b) other 50 edges
    # ---------------------------------------------------------------------------------------------
    stat = {cc: edge_status(d, cc)[1] for cc in [D.PRIMARY, *BASE7]}
    for code, o in stat.items():
        left = 0
        for key, fc, ec in (
            ("ok", OK, "none"),
            ("ws", ERR, "none"),
            ("null", NULL_FC, ERR),
            ("indet", UNSC_FC, "none"),
        ):
            if o[key]:
                b.barh(
                    ys[code],
                    o[key],
                    THICK,
                    left=left,
                    color=fc,
                    edgecolor=ec,
                    linewidth=0.6 if ec != "none" else 0,
                    zorder=3,
                )
                left += o[key]
        H.value_right(b, left, ys[code], str(left))
    omax = max(sum(o.values()) for o in stat.values())
    b.set_xlim(0, omax * 1.22)
    b.set_xticks([0, 10, 20, 30])
    b.set_xlabel("Declared edges")
    # (c) text column: placebo-log edges, then the pooled scores
    # -------------------------------------------------------- [G5 2026-10-12] the placebo bars
    # became a count column: at the journal's 8 pt floor four panels did not keep each data panel >=
    # 1.0 in wide; the counts were already printed at the bar ends
    assert all(len(d["placebo"][cc]) == 0 for cc in D.PMRT9)
    head = ylim[1] - 0.45
    t.text(0.16, head, "Placebo", ha="right", va="center", fontsize=S.ANNOT_PT, color=P.INK)
    for code in [D.PRIMARY, *BASE7]:
        t.text(
            0.16,
            ys[code],
            str(len(d["placebo"][code])),
            ha="right",
            va="center",
            fontsize=S.ANNOT_PT,
            color=P.INK,
        )
    t.text(0.16, ys["all9"], "0", ha="right", va="center", fontsize=S.ANNOT_PT, color=P.INK)
    for x, k, lab in ((0.55, "ind_f1", "F1"), (1.0, "ov_precision", "Precision")):
        t.text(x, head, lab, ha="right", va="center", fontsize=S.ANNOT_PT, color=P.INK)
        for code in [D.PRIMARY, *BASE7]:
            v = d["pooled"][code][k]
            fail = k == "ov_precision" and v < d["p3"]
            t.text(
                x,
                ys[code],
                H.fmt_score(v),
                ha="right",
                va="center",
                fontsize=S.ANNOT_PT,
                color=P.INK,
                fontweight="bold" if fail else "normal",
            )
        vals = [d["pooled"][cc][k] for cc in D.PMRT9]
        t.text(
            x,
            ys["all9"],
            f"{H.fmt_score(min(vals))}–{H.fmt_score(max(vals))}",
            ha="right",
            va="center",
            fontsize=S.ANNOT_PT,
            color=P.INK,
        )
    t.set_xlim(-0.02, 1.02)
    t.axis("off")
    # shared row axis, banners, labels
    # -------------------------------------------------------------------------------
    ticks = [y for y, _, _ in subrows]
    labs = [""] * len(ticks)
    for ax in (a, b):
        if ax is not a:
            plots.finish(ax)
        ax.spines["left"].set_visible(False)
        H.row_axes(ax, ylim, ticks, labs)
    b.grid(axis="x", visible=True)
    t.set_ylim(*ylim)
    t.patch.set_visible(False)
    L.tidy_grid([[a, b]])
    for y, text in banners:
        L.banner(fig, a, y, L.BANNER_H * 0.92, text)
    # one right-aligned label per row: method names bold (PMRT included), the 9-combination row and
    # the ground-truth row regular; the 9-combination row sits directly under PMRT, slightly smaller
    for y, _, key in subrows:
        name = {"gt": "Ground truth", "all9": "9 variants"}.get(key) or H.name13(key)
        a.annotate(
            name,
            xy=(0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=S.ANNOT_PT,
            fontweight="normal" if key in ("gt", "all9") else "bold",
            color=P.INK,
            gid="row-label",
        )
    L.panel_header(a, "a", "TRUE indirect edges")
    L.panel_header(b, "b", "Other 50 edges")
    L.panel_header(c, "c", "Placebo logs; scores")
    for ax in (a, b, c):  # as in the referee figures: inner letters / tags do not widen gaps
        for tt in ax.texts:
            if tt.get_gid() == "panel-tag" or (ax is not a and tt.get_gid() == "panel-letter"):
                tt.set_in_layout(False)
    tri_h = lambda m, fc: Line2D([], [], ls="", marker=m, ms=MS, mfc=fc, mec=P.WHITE, mew=0.4)
    sq = lambda fc, ec="none": Patch(fc=fc, ec=ec, lw=0.6)
    handles = [
        (tri_h("^", GT), tri_h("v", GT)),
        (tri_h("^", OK), sq(OK)),
        (tri_h("^", ERR), sq(ERR)),
        sq(NULL_FC, ERR),
        sq(UNSC_FC),
        Line2D([], [], ls="", marker="o", ms=MS - 1.6, mfc=P.WHITE, mec=MISS, mew=0.7),
    ]
    labels = [
        "Ground truth (up +, down −)",
        "Declared, right sign",
        "Declared, wrong sign",
        "Declared, false (NULL edge)",
        "Declared, not scored (INDET)",
        "Missed",
    ]
    L.fig_legend(
        fig,
        handles,
        labels,
        where="top",
        handlelength=1.6,
        handleheight=1.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
    )
    return L.finalize(fig)

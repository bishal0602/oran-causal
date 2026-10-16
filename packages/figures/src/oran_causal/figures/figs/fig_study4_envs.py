"""Study 4, Section 7.2 (beside Table tab_study4_design): the ground-truth causal graph of each controlled diagnostic
environment E1-E5, so the article alone shows what the validity and recall figures test (the per-environment pages
A1-A5 move to the electronic supplement). Same graphs (``_env_page.GRAPHS``), node roles, edge kinds and layered
bipartite drawing as panel (a) of those pages (oran_causal.figures.envgraph, DESIGN s.8.1, design G-A); only the canvas sizes, node
pitch, column gap and text sizes differ, so every label is >= 7 pt at print size.

Layout (user brief G4, 2026-10-12: one row was too cramped at \\textwidth): E2, with eight actions, takes a tall middle
column; E1 above E4 on the left and E3 above E5 on the right, so the letters a-e read across the top row, then the
bottom row. Each panel has the house header (bold letter, then "E1: <mechanism>" wrapped to the panel width, the
letter on the first line); the mechanism lines are TERMINOLOGY s.2 (``_env_page.BLURB``). One figure-level legend on
top lists only the node roles and edge kinds the five graphs use.
"""

from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from oran_causal.figures import envgraph as EG
from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures import style as S
from oran_causal.figures.figs import _env_page as E
from oran_causal.figures.registry import FigSpec

WORLDS = ["E1", "E2", "E3", "E4", "E5"]
SPEC = FigSpec(
    name="fig_study4_envs",
    width="full",
    paper=True,
    sources=[E.SRC_ENV.format(w.lower()) for w in WORLDS] + [E.SRC_GEN],
    caption_hint="Ground-truth causal graphs of the five controlled diagnostic environments E1-E5 (the "
    "same graphs as panel a of the environment pages): actions P, KPIs K, placebo action Pl (Plc: the R3 "
    "diagnostic placebo of E4) and confounder Z; every edge acts one step later.",
)

W_PT = S.FULL * 72
LEG_H = 34.0  # legend band at the top (pt)
LINE_PT = S.LEGEND_PT * 1.25  # pitch of a wrapped panel-tag line
TAG_X = L.LETTER_GAP_PT + 7  # tag offset after an inset letter (as layout.panel_header(inset=True))
# columns (pt from the left edge): E1 / E4, E2 (both rows), E3 / E5; 2 pt in from each side, so the
# saved PDF is not wider than   extwidth and LaTeX does not shrink the 7 pt labels (as
# diagram.EDGE_PT)
COLS = {
    "E1": (2.0, 120.0),
    "E4": (2.0, 120.0),
    "E2": (130.0, 104.0),
    "E3": (242.0, 124.0),
    "E5": (242.0, 124.0),
}
ROW_H = {1: 106.0, 2: 118.0}  # graph canvas heights (pt) of the top and bottom rows
HEAD = {1: 2, 2: 2}  # tag lines reserved above each row
GAP = 6.0  # space between the top row's canvases and the bottom row's header
# per-environment drawing: action column x, KPI column x (canvas pt), largest action / KPI pitch
GEOM = {
    "E1": {"xa": 16, "xk": 74, "act_step": 22, "kpi_step": 26},
    "E2": {"xa": 16, "xk": 88, "act_step": 40, "kpi_step": 40, "kpi_inset": 12},
    "E3": {"xa": 16, "xk": 74, "act_step": 22, "kpi_step": 22, "kpi_inset": 2},
    "E4": {"xa": 26, "xk": 84, "act_step": 34, "kpi_step": 26},
    "E5": {"xa": 16, "xk": 90, "act_step": 22, "kpi_step": 26},
}


def _width(text: str, size: float) -> float:
    return (
        TextPath((0, 0), text, size=size, prop=FontProperties(family=S.FONT_FAMILY))
        .get_extents()
        .width
    )


def _wrap(text: str, width_pt: float, size: float) -> list[str]:
    """Wrap ``text`` to ``width_pt`` at ``size`` pt in the figure font: the fewest lines, then the
    most even ones (no orphan word).
    """
    words = text.split()
    for n in range(1, len(words) + 1):
        best = None
        for cuts in combinations(range(1, len(words)), n - 1):
            b = (0, *cuts, len(words))
            lines = [" ".join(words[i:j]) for i, j in zip(b, b[1:], strict=False)]
            widest = max(_width(t, size) for t in lines)
            if widest <= width_pt and (best is None or widest < best[0]):
                best = (widest, lines)
        if best:
            return best[1]
    raise ValueError(f"cannot wrap {text!r} to {width_pt} pt")


def _header(ax, letter: str, tag: str, width_pt: float, nlines: int) -> None:
    """House panel header for a canvas without a label column (layout.panel_header(inset=True)),
    with the tag wrapped: the bold letter and the first tag line share a baseline; a row reserves
    ``nlines`` tag lines, the last HEADER_RISE_PT above the canvas, so the letters of a row stay on
    one baseline.
    """
    lines = _wrap(tag, width_pt - TAG_X, L.TAG_PT)
    assert len(lines) <= nlines, f"panel tag needs {len(lines)} lines: {lines}"
    top = L.HEADER_RISE_PT + (nlines - 1) * LINE_PT
    let = ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, top),
        textcoords="offset points",
        ha="left",
        va="baseline",
        fontsize=S.PANEL_PT,
        fontweight="bold",
        color=P.INK,
    )
    let.set_gid("panel-letter")
    for i, line in enumerate(lines):
        t = ax.annotate(
            line,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(TAG_X, top - i * LINE_PT),
            textcoords="offset points",
            ha="left",
            va="baseline",
            fontsize=L.TAG_PT,
            color=P.INK,
        )
        t.set_gid("panel-tag")


def legend_entries(graphs) -> tuple[list, list]:
    """Node roles and edge kinds used by any of ``graphs`` (labels and marks as on a page legend,
    envgraph), in three groups that fill the three legend columns: node roles; the confounder Z and
    its edges; the action edges and t−1.
    """
    hs, ls = {}, []
    for g in graphs:
        for h, lab in zip(*EG.legend_handles(g), strict=False):
            if lab.startswith("Placebo"):
                lab = "Placebo action" + (
                    " (Plc: R3)" if any("Plc" in x.placebos for x in graphs) else ""
                )
            if lab not in hs:
                hs[lab] = h
                ls.append(lab)
    lag = f"{EG.LAG}: previous step"
    groups = [
        ["Action", "KPI", next(x for x in ls if x.startswith("Placebo action"))],
        ["Confounder Z", EG.EDGE_LABEL["context"], EG.EDGE_LABEL["policy"]],
        [EG.EDGE_LABEL["true"], EG.EDGE_LABEL["gated"], lag],
    ]
    order = [x for grp in groups for x in grp if x in hs]
    assert sorted(order) == sorted(ls), "every used entry is in a group"
    return [hs[x] for x in order], order


def build():
    S.apply()
    top_row = LEG_H + HEAD[1] * LINE_PT + L.HEADER_RISE_PT + 4
    bot_row = top_row + ROW_H[1] + GAP + HEAD[2] * LINE_PT + L.HEADER_RISE_PT + 4
    h_pt = bot_row + ROW_H[2] + 2
    fig = plt.figure(figsize=(S.FULL, h_pt / 72))
    place = {
        "E1": (top_row, ROW_H[1]),
        "E3": (top_row, ROW_H[1]),
        "E4": (bot_row, ROW_H[2]),
        "E5": (bot_row, ROW_H[2]),
        "E2": (top_row, bot_row + ROW_H[2] - top_row),
    }
    for letter, w in zip("abcde", WORLDS, strict=False):
        x, cw = COLS[w]
        y, ch = place[w]
        ax = fig.add_axes((x / W_PT, 1 - (y + ch) / h_pt, cw / W_PT, ch / h_pt))
        ax.set_label(f"graph {w}")
        EG.draw_env_graph(
            ax, E.GRAPHS[w], cw, ch, legend=False, label_pt=S.MIN_PT, lag_pt=S.MIN_PT, **GEOM[w]
        )
        _header(
            ax, letter, f"{w}: {E.BLURB[w]}", cw, HEAD[1] if w in ("E1", "E2", "E3") else HEAD[2]
        )
    hs, ls = legend_entries([E.GRAPHS[w] for w in WORLDS])
    fig.legend(
        hs,
        ls,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=S.LEGEND_PT,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.35,
        borderaxespad=0.1,
    )
    return L.finalize(fig)

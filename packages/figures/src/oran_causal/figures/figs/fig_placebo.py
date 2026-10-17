"""Placebo check: edges each method declared on placebo logs, where no parameter has an effect (F4,
Section 6.1).

One message: on placebo data the baseline methods still report effects; PMRT reports none. A
lollipop strip per log type (a randomised, b confounded), one row per method, PMRT on top as the
hero on a light blue row band, a plain zero axis line (user review 2026-10-14: no teal zero band or
label), baselines sorted by their confounded-log count, the count printed beside each dot; no legend
(rows are labelled).

Variants are those of the map-quality table (tab:maps, s06_discovery.tex): the DEV-tuned threshold
for the learned and associational baselines, Granger + BY and PMRT without a tuned threshold (frozen
PMRT, loadsp_c+wby1s). The placebo-calibrated maps are left out: their threshold is set on these
placebo logs to allow at most one edge.

Sources: FD v4.placebo_declarations (randomised; re-counted from V4
placebo_descriptive.<code>.declared by _study1.load()), FD option_a.map_quality.<map>.placebo_edges
(confounded). Every printed count is asserted against the values printed in tab:maps.
"""

import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _study1 as S1
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_placebo",
    width="full",
    paper=True,
    sources=[ST.SRC_FD, ST.SRC_V4],
    caption_hint="Edges declared on placebo logs, where no parameter has an effect. "
    "40 randomised and 200 confounded placebo episodes.",
)

# (row label code, randomised-log code in FD v4, confounded-log map in FD option_a.map_quality)
ROWS = [
    ("PMRT", S1.PRIMARY, "MG:PMRT"),
    ("corr", "corr", "MG:corr@dev"),
    ("granger_dev", "granger", "MG:granger@dev"),
    ("granger_by", "granger_by", "MG:granger_by"),
    ("shap_gbdt", "shap_gbdt", "MG:shap_gbdt@dev"),
    ("two_tower", "two_tower", "MG:two_tower@dev"),
    ("qacm", "qacm", "MG:qacm@dev"),
    ("int", "int", "MG:int@dev"),
]
# tab:maps column P, (a) Study 1 DEV-tuned and (b) Study 2 DEV-tuned
EXPECT = {
    "PMRT": (0, 0),
    "corr": (3, 8),
    "granger_dev": (1, 19),
    "granger_by": (13, 29),
    "shap_gbdt": (8, 0),
    "two_tower": (6, 22),
    "qacm": (5, 0),
    "int": (6, 1),
}
PANELS = [("a", "Randomised logs"), ("b", "Confounded logs")]
XMAX = 32
X_PAD = 0.9  # room left of zero so the PMRT dot is not cut by the axes edge
STEM_LW = 1.6


def counts() -> dict[str, tuple[int, int]]:
    s1 = S1.load()
    mq = ST.fd()["option_a"]["map_quality"]
    out = {}
    for key, c1, c2 in ROWS:
        out[key] = (len(s1["placebo"][c1]), mq[c2]["placebo_edges"])
        assert out[key] == EXPECT[key], (key, out[key], EXPECT[key])
    assert all(
        len(s1["placebo"][c]) == 0 for c in S1.PMRT9
    )  # no PMRT variant declared a placebo edge
    return out


def build():
    n = counts()
    hero = "PMRT"
    # PMRT on top, then the baselines by their confounded-log count (most first)
    keys = [hero, *sorted((k for k in n if k != hero), key=lambda k: (-n[k][1], -n[k][0]))]
    ys = np.arange(len(keys))[::-1].astype(float)
    fig, axs = plots.figure("full", aspect=0.42, ncols=2, sharey=True)
    for j, (ax, (letter, tag)) in enumerate(zip(axs, PANELS, strict=False)):
        # our row: light blue band; zero: a plain axis line
        ax.axhspan(
            ys[0] - 0.5,
            ys[0] + 0.5,
            color=P.tint(P.OURS, 0.16),
            lw=0,
            zorder=0.4,
            gid="sem:ours-row",
        )
        ax.axvline(0, color=P.AXIS, lw=S.LW_AXIS, zorder=2, gid="zero-axis")
        for yy, k in zip(ys, keys, strict=False):
            v = n[k][j]
            is_hero = k == hero
            if v:
                ax.plot(
                    [0, v],
                    [yy, yy],
                    color=M.color(k),
                    lw=STEM_LW,
                    marker="",
                    zorder=2.5,
                    solid_capstyle="butt",
                    gid=f"stem:{k}",
                )  # the glyph carries the method tag
            M.row_point(ax, v, yy, None, k, hero=is_hero)
            ax.annotate(
                str(v),
                xy=(v, yy),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=S.ANNOT_PT,
                color=P.INK,
                fontweight="bold" if is_hero else "normal",
                gid="value-label",
            )
        ax.set_xlim(-X_PAD, XMAX)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.5)
        ax.grid(False)
        plots.finish(ax)
        ax.set_yticks(ys, [M.name(k) for k in keys])  # after finish: trimming resets tick labels
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        L.panel_header(ax, letter, tag)
    for t in axs[0].get_yticklabels():
        if t.get_text() == M.name(hero):
            t.set_fontweight("bold")
    L.tidy_grid([list(axs)])
    fig.supxlabel(
        "Edges declared on placebo logs (fewer is better)", fontsize=S.LABEL_PT, color=P.INK
    )
    return L.finalize(fig)

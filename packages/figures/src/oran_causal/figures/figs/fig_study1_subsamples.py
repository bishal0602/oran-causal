"""Study 1, the small-subsample comparison (ESM; [G5 2026-10-12, P3] split from
fig_study1_discovery, whose panels (d, e) it was: restructure plan, "Fig 4d-e to the ESM"). PMRT
minus each per-subsample baseline (correlation, Granger, Granger + BY) on the 10 subsamples of 60
episodes, where criterion P2 failed against Granger + BY: (a) indirect F1 with the P2 win counts (FD
v4.P2), (b) overall precision (FD v4.per_slice). PMRT blue dots, filled where PMRT >= baseline. Tied
values on a row are stacked 2.3 pt apart (a display offset; the caption states it). The win counts
are asserted against the stored P2 wins (_study1.load). The artifacts call a subsample a "slice";
the figure says "subsample" (restructure plan, terms: "slice" means the network slice only).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    name="fig_study1_subsamples",
    width="full",
    paper=True,
    sources=[ST.SRC_FD, ST.SRC_V4, source_note(*ST.DECISION, "v4_verdict.json")],
    caption_hint="Study 1, PMRT minus each baseline scored per subsample, on each of the 10 subsamples of 60 "
    "episodes: (a) indirect F1, the quantity of criterion P2, which needed at least 6 of 10 subsamples; "
    "(b) overall precision. Filled dots: PMRT at least the baseline; open: lower. Tied values are "
    "stacked 2.3 pt apart. Source: figures_data.json (v4) and the Study 1 analysis file.",
)

HEIGHT = 1.75
BASE = ["corr", "granger", "granger_by"]
DOT_WIN = {
    "ls": "",
    "marker": "o",
    "ms": 3.6,
    "mfc": P.BLUE,
    "mec": P.WHITE,
    "mew": 0.5,
    "zorder": 4,
}  # PMRT blue
DOT_LOSE = {
    "ls": "",
    "marker": "o",
    "ms": 3.6,
    "mfc": P.WHITE,
    "mec": P.BLUE,
    "mew": 0.8,
    "zorder": 4,
}


def subsample_panels(axs, d):
    """PMRT minus each per-subsample baseline on the 10 subsamples of 60 episodes (criterion P2's
    comparison).
    """
    pts = {ax: [] for ax in axs}
    s60 = [s for s in d["slice_names"] if d["slices"][D.PRIMARY][s]["n_eps"] == 60]
    assert len(s60) == 10
    ys = {c: -i for i, c in enumerate(BASE)}
    for ax, key in zip(axs, ("ind_f1", "ov_precision"), strict=False):
        for c in BASE:
            v = np.array([d["slices"][D.PRIMARY][s][key] - d["slices"][c][s][key] for s in s60])
            y = ys[c]
            win, lose = v[v >= 0], v[v < 0]
            if len(win):
                pts[ax] += ax.plot(win, [y] * len(win), **DOT_WIN)
            if len(lose):
                pts[ax] += ax.plot(lose, [y] * len(lose), **DOT_LOSE)
            if key == "ind_f1":
                wins = int((v >= 0).sum())
                assert wins == d["p2"][c]["wins60"], c
                ax.annotate(
                    f"{wins}/10" + ("" if wins >= 6 else ", fail"),
                    xy=(1, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(-1, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=S.ANNOT_PT,
                    fontweight="normal" if wins >= 6 else "bold",
                    color=P.INK,
                )
        ax.axvline(0, color=P.INK, alpha=0.6, lw=S.LW_REF, ls=(0, (4, 2)), zorder=2)
        top = key == "ind_f1"  # the label goes where the panel has no dot near 0
        ax.annotate(
            "tie",
            xy=(0, 1 if top else 0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -1 if top else 1),
            textcoords="offset points",
            ha="left",
            va="top" if top else "bottom",
            fontsize=S.ANNOT_PT,
            color=P.INK,
        )
        ax.set_xlabel("PMRT − baseline")
    axs[0].annotate(
        "Subsamples won",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(-1, -1),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=S.ANNOT_PT,
        color=P.INK,
    )
    axs[0].set_xlim(-0.24, 0.66)
    axs[0].set_xticks([-0.2, 0, 0.2, 0.4], ["−0.2", "0", "0.2", "0.4"])
    axs[1].set_xlim(-0.1, 0.36)
    axs[1].set_xticks([-0.1, 0, 0.1, 0.2, 0.3], ["−0.1", "0", "0.1", "0.2", "0.3"])
    for ax in axs:
        plots.finish(ax)
        H.row_axes(ax, (-2.6, 0.95), [ys[c] for c in BASE], [H.name13(c) for c in BASE])
    L.tidy_grid([axs])
    L.panel_header(axs[0], "a", "Indirect F1 per subsample")
    L.panel_header(axs[1], "b", "Overall precision per subsample")
    for ax in axs:  # [G5] tags and the inner letter stay out of the layout
        for t in ax.texts:
            if t.get_gid() == "panel-tag" or (ax is axs[1] and t.get_gid() == "panel-letter"):
                t.set_in_layout(False)
    return pts


def build():
    d = D.load()
    S.apply()
    fig = plt.figure(figsize=(S.FULL, HEIGHT), layout="constrained")
    axs = list(fig.subplots(1, 2))
    pts = subsample_panels(axs, d)
    nz = lambda st: {k: v for k, v in st.items() if k != "zorder"}
    L.fig_legend(
        fig,
        [Line2D([], [], **nz(DOT_WIN)), Line2D([], [], **nz(DOT_LOSE))],
        ["PMRT ≥ baseline", "PMRT < baseline"],
        where="top",
    )
    L.finalize(fig)
    for ax in axs:
        _stack(ax, pts[ax], step_pt=2.3)
    fig.draw_without_rendering()
    return fig


def _stack(ax, lines, step_pt: float, tol_frac: float = 0.6):
    """Tied values on one row (discrete F1 at 60 episodes) are stacked vertically, step_pt apart, so
    no dot hides another (a display offset like layout.dodge; the caption states it). Call once the
    layout is final.
    """
    fig = ax.figure
    while getattr(fig, "figure", fig) is not fig and fig.figure is not None:
        fig = fig.figure
    fig.draw_without_rendering()
    px = fig.dpi / 72
    placed = []
    for ln in lines:
        disp = ax.transData.transform(np.asarray(ln.get_xydata(), float))
        tol = tol_frac * ln.get_markersize() * px * 1.6
        new = []
        for p in disp:
            for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                q = p.copy()
                q[1] += k * step_pt * px
                if all(np.hypot(*(q - o)) >= tol for o in placed):
                    break
            placed.append(q)
            new.append(q)
        ln.set_data(*ax.transData.inverted().transform(np.array(new)).T)

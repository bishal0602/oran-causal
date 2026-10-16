"""F1: Study 4 recall vs n in R2 (E1-E5, kappa .25) beside the E6-P bridging-analysis slices, with
the frozen PMRT (E6-P v4 primary) as an ink reference line (DESIGN s.12: never styled as a member of
the PMRT-GBM family). INVALID cells carry no recall; they are marked with a cross in the strip above
the recall scale (layout.recall_axis). Series are dodged horizontally (layout.dodge) so equal
recalls stay visible. Styles: oran_causal.figures.methods.
"""

import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_recall",
    width="full",
    paper=True,
    sources=[T.SRC_RECALL, T.SRC_BRIDGE_TABLES],
    caption_hint="Recall of PMRT-GBM, PMRT-Lin and the design-adjusted tests: Study 4 dithered-setpoint logs (a-e) and the "
    "bridging analysis on the simulator logs (f, post hoc, descriptive; frozen PMRT = reference line). Colour identifies "
    "the method (the same in every figure); every series shown uses the logging design (solid)",
)

NS = [500, 1000, 4000, 8000, 24000]
WORLDS = ["E1", "E2", "E3", "E4", "E5"]
WORLD_NAME = {
    "E1": "Linear graph",
    "E2": "Rare regime",
    "E3": "Delayed harm",
    "E4": "Latent confounding",
    "E5": "Composed",
}  # plain names (ESM figures carry no codes); every panel is dithered setpoint
SLICE_TOTAL = T.SLICE_EPISODES[
    "pooled"
]  # 600 EVAL episodes; a slice of e episodes gives 600 / e clusters
MIN_CI_CLUSTERS = 2  # intervals on 2 or fewer clusters are not drawn (V2/V3 review, 2026-10-10)


def _study4_panel(ax, df, world):
    w = df[df.world == world]
    for arm in T.RECALL_ARMS:
        d = w[w.arm == arm].sort_values("n")
        if d.empty:
            continue
        ok = d[d.state != "INVALID"]
        T.series(
            ax,
            ok.n,
            ok.recall_mean,
            ok.recall_lo,
            ok.recall_hi,
            arm,
            T.RECALL_ARMS,
            invalid_x=d[d.state == "INVALID"].n,
        )
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(NS, ["0.5", "1", "4", "8", "24"])
    ax.set(xlim=(330, 36000), xlabel="n (×10³)")
    L.recall_axis(ax)
    plots.finish(ax)


def _bridge_panel(ax, tab):
    cells, ref = tab["cells"], tab["v4_frozen_reference"]
    kinds = list(T.SLICE_EPISODES)
    eps = np.array([T.SLICE_EPISODES[s] for s in kinds], float)
    M.reference_line(ax, eps, [ref[s]["recall"] for s in kinds])
    # label at the pooled end, text to the left: clear of the rising line and the 60-episode error
    # bars (team review 2026-10-09: measured 8.1 pt to other marks, 4.0 pt to its own line; was 1.5
    # pt / touching at 60 episodes)
    L.label_line(ax, eps[-1], ref[kinds[-1]]["recall"], "Frozen PMRT", where="above", ha="right")
    for arm in T.RECALL_ARMS:
        c = [cells[f"{arm}|{s}|by"]["rec"] for s in kinds]
        y = np.array([v["rate"] for v in c])
        tr = T.series(ax, eps, y, None, None, arm, T.RECALL_ARMS)
        n_pooled = cells[f"{arm}|pooled|by"]["rec"]["n"]
        for xi, yi, v in zip(eps, y, c, strict=False):
            # [2026-10-12, review] the interval rests on the slices (clusters): 600 / episodes of
            # them, i.e. n / pooled n. With 2 clusters (300-episode slices) it is not drawn; the
            # point stays
            clusters = round(SLICE_TOTAL / xi)
            assert v["n"] == clusters * n_pooled, (arm, xi, v["n"], n_pooled)
            if v.get("ci") and clusters > MIN_CI_CLUSTERS:
                plots.errorbars(
                    ax, [xi], [yi], [[yi - v["ci"][0]], [v["ci"][1] - yi]], transform=tr
                )
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(eps, ["60", "120", "300", "600"])
    ax.set(xlim=(46, 800), xlabel="Episodes")
    L.recall_axis(ax)
    plots.finish(ax)


def build():
    keep, L.INVALID_LABEL = L.INVALID_LABEL, "Not scored"
    try:
        return _build()
    finally:
        L.INVALID_LABEL = keep


def _build():
    fig, axs = plots.figure("full", aspect=0.74, nrows=2, ncols=3)
    axs = axs.ravel()
    df = T.recall_r2()
    for i, world in enumerate(WORLDS):
        _study4_panel(axs[i], df, world)
        L.panel_header(axs[i], "abcdef"[i], WORLD_NAME[world])
    _bridge_panel(axs[5], T.bridge_tables())
    L.panel_header(axs[5], "f", "Simulator logs (bridging)")
    L.tidy_grid(axs.reshape(2, 3))
    M.legend(fig, T.RECALL_ARMS, hero=T.HERO)
    return L.finalize(fig)

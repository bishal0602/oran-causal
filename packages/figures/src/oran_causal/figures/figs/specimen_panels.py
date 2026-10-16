"""Specimen: two panels in the paper's mapping ([DECIDED 2026-10-06, user: one palette = specimen
set]), synthetic data. (a) convergence with CI bands (PMRT-GBM hero vs design-adjusted partial
correlation), (b) per-seed box plots (box edge and seed dots in the method's marker colour; the
hero's box tinted and dots filled).
"""

import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.data import mean_ci
from oran_causal.figures.figs import _synthetic as syn
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(name="specimen_panels", width="full", paper=False, sources=["synthetic"])
CURVES = ["pcorr_eq", "pmrt_nl_eq"]
BOXES = ["pmrt_nl_eq", "pcorr_eq", "corr"]
HERO = "pmrt_nl_eq"


def build():
    syn.reseed()
    fig, (a, b) = plots.figure("full", aspect=0.5, ncols=2)
    x = np.arange(0, 51, 2)
    curves = syn.convergence(x)
    for k, code in zip((1, 0), CURVES, strict=False):
        m, lo, hi = mean_ci(curves[k], axis=0)
        M.band(a, x, lo, hi, code, hero=code == HERO)
        M.plot(a, x, m, code, hero=code == HERO, markevery=5)
    plots.ref_line(a, 0.9, "target 0.90")
    a.set(xlabel="Training samples (×10³)", ylabel="Synthetic F1", xlim=(-1, 51), ylim=(0, 1))
    plots.finish(a)

    f1 = syn.final_f1()
    for k, code in enumerate(BOXES):
        c, hero = M.marker_color(code), code == HERO
        bp = b.boxplot(
            f1[k],
            positions=[k - 0.12],
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "fc": P.tint(M.fill_color(code), 0.22) if hero else P.WHITE,
                "ec": c,
                "lw": 0.7,
            },
            medianprops={"color": c, "lw": 1.2},
            whiskerprops={"color": P.INK, "lw": 0.6},
            capprops={"color": P.INK, "lw": 0.6},
        )
        for art in [*bp["boxes"], *bp["medians"]]:
            art.set_gid(
                f"box:{M.resolve(code).key}"
            )  # styled from oran_causal.figures.methods (not a series)
        jitter = 0.16 + syn.RNG.uniform(0, 0.1, f1[k].size)
        b.scatter(
            k + jitter,
            f1[k],
            s=6,
            zorder=4,
            fc=c if hero else P.WHITE,
            ec=P.WHITE if hero else c,
            lw=0.3 if hero else 0.6,
            gid=f"method:{M.resolve(code).key}" + (":hero" if hero else ""),
        )
    b.set_xticks(
        range(len(BOXES)),
        ["PMRT-GBM", "Partial corr.\n(adjusted)", "Correlation"],
        fontsize=S.MIN_PT,
    )
    b.set(ylabel="Synthetic final F1", ylim=(0.4, 1.0))
    plots.finish(b)
    L.panel_header(a, "a")
    L.panel_header(b, "b")
    M.legend(fig, CURVES, hero=HERO)
    return L.finalize(fig)

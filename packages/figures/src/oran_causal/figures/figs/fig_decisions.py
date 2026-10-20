"""Conflict-mitigation decisions (F7): a safety-benefit plane. x = RLF ratio (vs accept-all), y =
recovery R, both with paired-seed-bootstrap 90 % CIs as thin ink crosses. Points: PMRT map and
oracle map without the gate (open) and with the gate (filled), arrows from each ungated to its gated
point, never-sleep and accept-all as references. The region beyond the 1.10 RLF limit is shaded.
Message: without the gate the PMRT referee recovers the most energy-side benefit but breaks the RLF
limit; with the gate it lands on never-sleep, as does the oracle map.

Every number is read from figures_data.json (FD) at build time: ``option_a.eval.arms`` (no gate, 160
seeds) and ``certsafe.eval.arms`` (gate, a different 160 seeds): R, R_ci90, guard_ratio.rlf,
guard_ratio_ci90.rlf; the tie label reads ``certsafe.eval.identical_seeds`` / ``n``. Asserted: gated
PMRT == never-sleep (R, RLF, both CIs), ungated PMRT RLF ratio > 1.10 (and its CI lower end > 1.0),
the annotated values. Not drawn (ESM scorecard tab_decision_e2e): design-blind maps, static subsets,
freeze, incumbent, static envelope, random map, no-map control. Never-sleep is the gated-block
(Study 3 seed) value, which coincides with the gated PMRT point.
"""

import matplotlib.pyplot as plt
import numpy as np

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _referee as R
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_decisions",
    width="full",
    paper=True,
    sources=[ST.SRC_FD],
    caption_hint="Recovery against RLF ratio without and with the gate; 160 seeds per arm, 90 % CIs.",
)

HEIGHT = 3.0
XLIM, YLIM = (0.68, 1.64), (-0.07, 0.72)


def _a(study: str, code: str) -> dict:
    return R.arms(study)[code]


def _pt(arm: dict):
    r, (rlo, rhi) = arm["R"], arm["R_ci90"]
    g, (glo, ghi) = arm["guard_ratio"]["rlf"], arm["guard_ratio_ci90"]["rlf"]
    return g, r, (glo, ghi), (rlo, rhi)


def _check(mg_pmrt: dict, cs_pmrt: dict, ns: dict) -> None:
    for k in ("R", "R_ci90"):
        if cs_pmrt[k] != ns[k]:
            raise ValueError(
                f"gated PMRT and never-sleep differ in {k}: the tie label no longer holds"
            )
    if (
        cs_pmrt["guard_ratio"]["rlf"] != ns["guard_ratio"]["rlf"]
        or cs_pmrt["guard_ratio_ci90"]["rlf"] != ns["guard_ratio_ci90"]["rlf"]
    ):
        raise ValueError("gated PMRT and never-sleep differ in the RLF ratio or its CI")
    if not mg_pmrt["guard_ratio"]["rlf"] > R.GUARD:
        raise ValueError("ungated PMRT RLF ratio is not above the 1.10 limit")
    if not ns["guard_ratio"]["rlf"] <= R.GUARD:
        raise ValueError("never-sleep RLF ratio is above the limit")


def _cross(ax, x, y, xci, yci):
    ax.errorbar(
        [x],
        [y],
        xerr=[[x - xci[0]], [xci[1] - x]],
        yerr=[[y - yci[0]], [yci[1] - y]],
        fmt="none",
        ecolor=P.INK,
        elinewidth=S.LW_WHISKER,
        capsize=1.5,
        capthick=S.LW_WHISKER,
        zorder=3,
        clip_on=False,
    )


def _point(ax, arm, code, *, filled, hero=False, cross=True):
    x, y, xci, yci = _pt(arm)
    if cross:
        _cross(ax, x, y, xci, yci)
    ax.plot(
        [x],
        [y],
        ls="",
        marker="o",
        ms=S.MARKER_SIZE + 3,
        mfc=P.WHITE,
        mec="none",
        zorder=3.5,
        clip_on=False,
        gid="halo",
    )
    M.row_point(ax, x, y, None, code, hero=hero, fill=filled)
    return x, y


def _labels(ax, mg_p, cs_p, mg_g, cs_g, p_mg_p, p_cs_p, fig):
    """Point labels as callouts, auto-placed clear of marks, whiskers and arrows, with a leader line
    when they sit away from their point; the hero arrow keeps its 'with gate' label.
    """
    g, r_, _, _ = _pt(mg_p)
    L.callout(
        ax,
        g,
        r_,
        f"PMRT map, no gate\nRLF ratio {g:.2f}",
        prefer=("n", "ne", "nw", "e"),
        fontweight="bold",
    )
    x, y, _, _ = _pt(cs_p)
    L.callout(ax, x, y, "PMRT map with gate\n= never-sleep", prefer=("nw", "w", "sw", "n"))
    x, y, _, _ = _pt(cs_g)
    L.callout(ax, x, y, "Oracle map\nwith gate", prefer=("sw", "w", "s", "nw"))
    x, y, _, _ = _pt(mg_g)
    L.callout(ax, x, y, "Oracle map\nno gate", prefer=("s", "se", "sw", "e"))
    (x0, y0), (x1, y1) = ax.transData.transform(p_mg_p), ax.transData.transform(p_cs_p)
    ang = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    ang = ang + 180 if ang > 90 else ang - 180 if ang < -90 else ang
    t = 0.38
    ax.annotate(
        "with gate",
        xy=(p_mg_p[0] + t * (p_cs_p[0] - p_mg_p[0]), p_mg_p[1] + t * (p_cs_p[1] - p_mg_p[1])),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        rotation=ang,
        rotation_mode="anchor",
        fontsize=S.ANNOT_PT,
        color=P.INK,
        fontweight="bold",
    )
    ax.set_xticks(
        [0.9, 1.0, R.GUARD, 1.2, 1.3, 1.4, 1.5, 1.6],
        ["0.9", "1.0", "1.10", "1.2", "1.3", "1.4", "1.5", "1.6"],
    )
    ax.set_yticks(np.round(np.arange(0, 0.71, 0.1), 1))
    ax.set_xlabel("RLF ratio vs accept-all (lower is better)")
    ax.set_ylabel("Recovery R (higher is better)")
    plots.finish(ax)
    return L.finalize(fig)


def build():
    S.apply()
    fig = plt.figure(figsize=(S.FULL, HEIGHT))
    ax = fig.add_axes([0.085, 0.17, 0.895, 0.80])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    mg_p, cs_p = _a("option_a", "MG:PMRT"), _a("certsafe", "CS:PMRT")
    mg_g, cs_g = _a("option_a", "MG:GT"), _a("certsafe", "CS:GT")
    ns, _aa = _a("certsafe", "never_sleep"), _a("certsafe", "noarb")
    _check(mg_p, cs_p, ns)
    _ev = ST.fd()["certsafe"]["eval"]
    # the breach region and the limit
    plots.zone(ax, XLIM[0], R.GUARD, "good", axis="x")
    plots.zone(ax, R.GUARD, XLIM[1], "bad", axis="x")
    plots.ref_line(ax, R.GUARD, axis="x")
    L.label_line(ax, R.GUARD, YLIM[0] + 0.03, "breaks RLF limit", where="right", fontweight="bold")
    L.label_line(ax, R.GUARD, YLIM[1] - 0.03, "within RLF limit", where="left", fontweight="bold")
    # references: accept-all (R = 0, ratio 1 by definition)
    ax.plot(
        [1.0],
        [0.0],
        ls="",
        marker="|",
        ms=S.MARKER_SIZE + 4,
        mew=1.0,
        color=P.NEUTRAL,
        mec=P.NEUTRAL,
        zorder=4,
        clip_on=False,
        gid="row:accept_all",
    )
    L.label_line(ax, 1.0, 0.0, "accept-all", where="below", ha="center")
    # oracle map: open without the gate, filled with it
    p_mg_g = _point(ax, mg_g, "MG:GT", filled=False)
    p_cs_g = _point(ax, cs_g, "CS:GT", filled=True)
    # PMRT map (hero); the gated point coincides with never-sleep
    p_mg_p = _point(ax, mg_p, "MG:PMRT", filled=False, hero=True)
    p_cs_p = _point(ax, cs_p, "CS:PMRT", filled=True, hero=True)
    plots.arrow(ax, p_mg_p, p_cs_p, shrink=(4, 4), color=P.OURS, lw=1.6, gid="sem:ours-arrow")
    plots.arrow(ax, p_mg_g, p_cs_g, shrink=(3, 3), gid="gate-arrow")
    return _labels(ax, mg_p, cs_p, mg_g, cs_g, p_mg_p, p_cs_p, fig)

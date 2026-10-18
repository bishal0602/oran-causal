"""Studies 2 and 3, lean forest [G7 2026-10-13, DRAFT for the user's review; L3 variant of tmp/fig7_variants/lean.py,
user-approved direction; the G5/G6 merged forest is kept in reviews/2026-10-12/backups-g7/]. Writes
fig_decision_referee_v3.pdf, so the paper's fig_decision_referee.pdf stays untouched until the user approves.

Two blocks, each a banner and three rows, PMRT map, oracle map and never-sleep:
  "Without gate": Study 2, the map-driven referee (MG:PMRT, MG:GT) and the static rule never_sleep;
  "With gate":    Study 3, the edge-certified referee (CS:PMRT, CS:GT) and never_sleep on the Study 3 seeds.
Left panel: recovery R; right panel: the RLF guard ratio, the region above the 1.10 limit shaded and the limit line
labelled. Paired-seed-bootstrap 90 % CIs as ink whiskers. No legend: the rows are labelled. Two direct labels: the
gated PMRT row's tie with never-sleep (identical seeds of n) and the ungated PMRT RLF ratio with its CI.
Every number is read from figures_data.json (FD) at build time, ``option_a.eval.arms`` (Study 2) and
``certsafe.eval.arms`` (Study 3): R, R_ci90, guard_ratio.rlf, guard_ratio_ci90.rlf; the tie label reads
``certsafe.eval.identical_seeds`` / ``n``. No re-aggregation. Asserted at build time: CS:PMRT equals never_sleep in R,
RLF and both CIs, and the MG:PMRT RLF ratio is above the 1.10 limit. Design-blind maps and the other static rules are
in the decision scorecard (tab_decision_e2e).
"""

import matplotlib.pyplot as plt

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _s23r as R
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_decision_referee_v3",
    width="full",
    paper=True,
    sources=[ST.SRC_FD],
    caption_hint="PMRT map, oracle map and never-sleep without the gate (Study 2) and with the gate "
    "(Study 3), 160 evaluation seeds each: recovery R and RLF ratio, paired seed-bootstrap "
    "90 % CIs; shaded: RLF ratio above the 1.10 limit. The studies used different seed sets, "
    "so rows are not paired across blocks",
)

HEIGHT = 3.3  # in; the L3 variant's height
BLOCKS = (("Without gate", "option_a", "MG"), ("With gate", "certsafe", "CS"))
ROWS = (
    "PMRT",
    "GT",
    "never_sleep",
)  # arm keys without the gate prefix; never_sleep is a static rule
ROW_NAME = {"PMRT": "PMRT map", "GT": "Oracle map", "never_sleep": "Never-sleep"}
PITCH = 1.0  # row pitch (row units)
BANNER_GAP = 1.15  # banner centre to its first row
BLOCK_GAP = 1.45  # last row of a block to the next banner
LEFT, GAP_PANELS, RIGHT = (
    0.155,
    0.04,
    0.985,
)  # figure fractions: row-label column, gap between panels, right edge
W_R = 0.475  # width of the recovery panel (figure fraction)
BOTTOM, TOP = 0.15, 0.985
R_LIM = (-0.02, 0.64)
RLF_LIM = (0.88, 1.62)
LIMIT_TINT = 0.14  # the shaded breach region: 14 % terracotta (L3)


def _code(gate: str, key: str) -> str:
    return key if key == "never_sleep" else f"{gate}:{key}"


def _layout():
    """Banner y and row y per block, top-down (y grows downward; the axes are inverted)."""
    y, banners, rows = 0.0, [], {}
    for title, study, gate in BLOCKS:
        banners.append((y, title))
        y += BANNER_GAP
        for key in ROWS:
            rows[(study, key)] = y
            y += PITCH
        y += BLOCK_GAP - PITCH
    return banners, rows, y - BLOCK_GAP + PITCH


def _check(cs: dict, mg: dict) -> None:
    """The two facts the figure states: behind the gate the PMRT referee is never-sleep, and without
    it the PMRT RLF ratio breaches the limit.
    """
    a, b = cs["CS:PMRT"], cs["never_sleep"]
    for k in ("R", "R_ci90"):
        if a[k] != b[k]:
            raise ValueError(
                f"CS:PMRT and never_sleep differ in {k}: the tie label no longer holds"
            )
    if (
        a["guard_ratio"]["rlf"] != b["guard_ratio"]["rlf"]
        or a["guard_ratio_ci90"]["rlf"] != b["guard_ratio_ci90"]["rlf"]
    ):
        raise ValueError(
            "CS:PMRT and never_sleep differ in the RLF ratio or its CI: the tie label no longer holds"
        )
    if not mg["MG:PMRT"]["guard_ratio"]["rlf"] > R.GUARD:
        raise ValueError(
            "MG:PMRT RLF ratio is not above the 1.10 limit: the breach label no longer holds"
        )


def build():
    S.apply()
    fig = plt.figure(figsize=(S.FULL, HEIGHT))
    h = TOP - BOTTOM
    a = fig.add_axes([LEFT, BOTTOM, W_R, h])
    b = fig.add_axes(
        [LEFT + W_R + GAP_PANELS, BOTTOM, RIGHT - LEFT - W_R - GAP_PANELS, h],
        sharey=a,
    )
    banners, rows, ybot = _layout()
    arms = {study: R.arms(study) for _, study, _ in BLOCKS}
    _check(arms["certsafe"], arms["option_a"])
    a.set_ylim(ybot - 0.35, -0.75)
    a.set_xlim(*R_LIM)
    b.set_xlim(*RLF_LIM)
    # (right) the breach region and the labelled limit line, under the data
    plots.span(b, R.GUARD, RLF_LIM[1], "bad", "x", tint=LIMIT_TINT, gid="limit-zone")
    plots.ref_line(b, R.GUARD, axis="x")
    L.label_line(b, R.GUARD, banners[0][0], "limit", where="right")
    for (study, key), y in rows.items():
        gate = {s: g for _, s, g in BLOCKS}[study]
        code = _code(gate, key)
        arm = arms[study][code]
        hero = key == "PMRT"
        rv, (rlo, rhi) = arm["R"], arm["R_ci90"]
        g, (glo, ghi) = arm["guard_ratio"]["rlf"], arm["guard_ratio_ci90"]["rlf"]
        M.row_point(a, rv, y, (rlo, rhi) if rhi > rlo else None, code, hero=hero)
        M.row_point(b, g, y, (glo, ghi) if ghi > glo else None, code, hero=hero)
        a.annotate(
            ROW_NAME[key],
            xy=(0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=S.ANNOT_PT,
            color=P.INK,
            fontweight="bold" if hero else "normal",
            gid="row-label",
        )
    # direct labels: the gated PMRT tie (Study 3) and the ungated PMRT RLF ratio (Study 2)
    ev, cs = ST.fd()["certsafe"]["eval"], arms["certsafe"]["CS:PMRT"]
    L.label_line(
        a,
        cs["R_ci90"][1],
        rows[("certsafe", "PMRT")],
        f"identical to never-sleep\non {ev['identical_seeds']} of {ev['n']} seeds",
        where="right",
        linespacing=0.95,
    )
    mg = arms["option_a"]["MG:PMRT"]
    g, (glo, ghi) = mg["guard_ratio"]["rlf"], mg["guard_ratio_ci90"]["rlf"]
    L.label_line(
        b,
        g,
        rows[("option_a", "PMRT")],
        f"{g:.2f} [{glo:.2f}, {ghi:.2f}]",
        where="below",
        ha="center",
    )
    # axes
    a.set_xticks([0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"])
    a.set_xlabel("Recovery R")
    b.set_xticks([0.9, 1.0, R.GUARD, 1.3, 1.5], ["0.9", "1.0", "1.10", "1.3", "1.5"])
    b.set_xlabel("RLF ratio")
    for ax in (a, b):
        ax.patch.set_visible(False)  # banners run underneath both panels
        plots.finish(ax)
        ax.grid(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
    for y, text in banners:
        L.banner(fig, a, y, L.BANNER_H * 0.8, text)
    return L.finalize(fig)

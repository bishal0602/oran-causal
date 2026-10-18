"""F6: validity across logging designs (replaces the 20-row forest fig_study4_validity in the body;
that figure, with its CIs and every arm, moves to the ESM). One message: tests that ignore how the
logs were produced lose their 5 % error rate once the logging is not independent; tests that use the
logging design keep it.

Two panels share a log rate axis: (a) design-based and design-adjusted tests, (b) design-blind
methods. x = logging design in order independent (R1), dithered setpoint (R2), confounded policy
(R3). Each point is the pooled raw null-edge rejection rate (p <= .05), kappa .25, the settings of
fig_study4_validity: truth-null edges for R1 / R2 (the four environments that have them; E4 has
none), the confounded null edge for R3 (E4 only, all four lam). Rates are pooled hits / tests over
the cells of report.json V1_validity + V6_E4; build-time assertions check that this reproduces every
rate the verdict table V4_verdicts prints (and its R1 + R2 pool for PMRT, which the table does not
split by regime). Panel (b) has only the two designs the verdict table scores for design-blind
methods (C1: R1 / R2); their descriptive confounded-policy rates go to the appendix text, not here
(advisor, 2026-10-13). Design-adjusted Granger has no R3 run: its line simply stops. CIs are
omitted: the PMRT per-regime rates have no cluster-bootstrap CI in the report, and the ESM forest
keeps every CI. Two arms of the forest are left out here and stay in the ESM forest: plain partial
correlation, HAC (fixed-b), which shares hue and marker with plain partial correlation and tracks it
within .004; MSCR, scored at n <= 1000 only; plain RCoT, which shares gray and marker with
design-adjusted RCoT (allowed only as labelled rows) and adds nothing the three design-blind lines
do not show (.06 independent, .33 dithered). Design-adjusted RCoT, the disclosed exception, stays in
(a).
"""

from collections import defaultdict

import numpy as np
from matplotlib.lines import Line2D

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.data import load_json
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_logging_validity",
    width="full",
    paper=True,
    sources=[T.SRC_REPORT],
    caption_hint="Null-edge rate by logging design, for tests that use (a) or ignore (b) the design; "
    "pooled over five environments.",
)

DESIGNS = ["R1", "R2", "R3"]
DESIGN_LABEL = ["independent", "dithered\nsetpoint", "confounded\npolicy"]
BLIND = ["corr", "pcorr_native", "granger_native"]
USES = [T.HERO, "pmrt_eq", "pcorr_eq", "granger_eq", "rcot2_eq"]
# designs the verdict table scores per arm (fig_study4_validity's rows): C1 scores design-blind
# methods on R1 / R2 only
SCORED = {**dict.fromkeys(BLIND, ("R1", "R2")), **{a: tuple(DESIGNS) for a in USES}}
YLIM = (0.02, 1.25)
YTICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
# [SOUL 2026-10-13] semantic zones (DESIGN s.5): terracotta above the report's validity limit
# valid_ub (a pooled rate above it cannot be valid, whatever its CI); a thin teal band marking the
# nominal level (visual width only, +-10 %)
NOMINAL_BAND = (0.045, 0.055)
BAND_TINT = 0.22
HERO_LW, HERO_MS = (
    S.LW_PROPOSED + 0.5,
    S.MARKER_SIZE + 1.0,
)  # the focal series: heavier than the default hero


def pooled_rates() -> dict[tuple[str, str], tuple[float, int]]:
    """(arm, regime) -> (pooled raw rate, tests) at kappa .25: truth-null edges in R1 / R2, the
    confounded null edge in R3. Cells are listed once per declaration rule (raw rates do not depend
    on it), so each key counts once.
    """
    t = load_json(*T.EVAL, "report.json")["tables"]
    acc, seen = defaultdict(lambda: [0.0, 0]), set()
    for r in t["V1_validity"] + t["V6_E4"]:
        if r["kappa"] != T.KAPPA or r["key"] in seen:
            continue
        seen.add(r["key"])
        reg = r["key"].split("|")[2]
        x = r.get("conf_raw" if reg == "R3" else "null_raw")
        if reg in DESIGNS and x and x.get("n"):
            a = acc[(r["arm"], reg)]
            a[0] += x["rate"] * x["n"]
            a[1] += x["n"]
    return {k: (h / n, n) for k, (h, n) in acc.items()}


def _check(rates: dict) -> float:
    """Every plotted rate equals the verdict table's (V4_verdicts) where it prints one; the figure's
    claims hold.
    """
    v = load_json(*T.EVAL, "report.json")["tables"]
    alpha, ub, v = v["alpha"], v["valid_ub"], v["V4_verdicts"]
    assert alpha == 0.05
    want = {}
    for arm in BLIND:
        for r in ("R1", "R2"):
            want[(arm, r)] = v["C1"]["arms"][arm][f"pooled_{r.lower()}"]["null_raw"]
    for arm in ("pcorr_eq", "granger_eq", "rcot2_eq"):
        for r in ("R1", "R2"):
            want[(arm, r)] = v["eq_arms_table"][arm][f"pooled_{r.lower()}"]["null_raw"]
    for arm in ("pcorr_eq", "rcot2_eq"):
        want[(arm, "R3")] = v["C3_eq_arms"][arm]["pooled"]["conf_raw"]
    want[(T.HERO, "R3")] = v["C3"]["pooled"]["conf_raw"]
    want[("pmrt_eq", "R3")] = v["pmrt_secondary"]["pmrt_eq"]["C3"]["pooled"]["conf_raw"]
    for k, w in want.items():
        assert abs(rates[k][0] - w["rate"]) < 1e-9 and rates[k][1] == w["n"], (
            k,
            rates.get(k),
            w["rate"],
            w["n"],
        )
    for arm, c2a in (
        (T.HERO, v["C2a"]),
        ("pmrt_eq", v["pmrt_secondary"]["pmrt_eq"]["C2a"]),
    ):  # R1 + R2 pool
        (a, na), (b, nb) = rates[(arm, "R1")], rates[(arm, "R2")]
        assert abs((a * na + b * nb) / (na + nb) - c2a["pooled"]["null_raw"]["rate"]) < 1e-9, arm
    assert ("granger_eq", "R3") not in rates
    # the message: independent logs keep every design-blind method near .05, dithered logs push all
    # of them past the validity bound; the design-based tests and design-adjusted partial
    # correlation and Granger stay below it
    assert all(rates[(a, "R1")][0] < ub < rates[(a, "R2")][0] for a in BLIND)
    assert all(rates[(a, r)][0] < ub for a in USES[:-1] for r in DESIGNS if (a, r) in rates)
    assert rates[("rcot2_eq", "R2")][0] > ub  # the disclosed exception
    assert ub == 0.075
    assert NOMINAL_BAND[0] < alpha < NOMINAL_BAND[1] < ub
    return ub


def _panel(ax, rates: dict, arms: list[str], designs: list[str], ub: float) -> None:
    x = np.arange(len(designs))
    plots.zone(ax, ub, YLIM[1], "bad")
    ax.axhspan(
        *NOMINAL_BAND, color=P.tint(P.GOOD, BAND_TINT), lw=0, zorder=0.5, gid="sem:good-band"
    )
    ax.text(
        0.03,
        0.97,
        "invalid",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=S.LABEL_PT,
        fontweight="bold",
        color=P.INK,
    )
    for arm in arms:
        y = [
            rates[(arm, r)][0] if (arm, r) in rates and r in SCORED[arm] else np.nan
            for r in designs
        ]
        kw = {"lw": HERO_LW, "ms": HERO_MS} if arm == T.HERO else {}
        T.series(ax, x, y, None, None, arm, arms, **kw)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS, ["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.minorticks_off()
    ax.set_xlim(-0.4, len(designs) - 0.6)
    ax.set_xticks(x, [DESIGN_LABEL[DESIGNS.index(r)] for r in designs])
    ax.set_xlabel("Logging design")
    plots.ref_line(ax, 0.05)
    plots.finish(ax)


def build():
    rates = pooled_rates()
    ub = _check(rates)
    # (a) uses the design, (b) ignores it: the legend's two columns (family order, design-based
    # first) then sit over their own panels (b) shows only the designs the verdict table scores for
    # design-blind methods (no empty column); width ratio 3 : 2 keeps the spacing between designs
    # equal in both panels. Their descriptive confounded-policy rates are in the appendix text
    # (reviews/2026-10-13/briefs/N3_confounded_designblind.md)
    blind_designs = ["R1", "R2"]
    fig, (a, b) = plots.figure(
        "full",
        aspect=0.62,
        ncols=2,
        sharey=True,
        gridspec_kw={"width_ratios": [len(DESIGNS), len(blind_designs)]},
    )
    _panel(a, rates, USES, DESIGNS, ub)
    _panel(b, rates, BLIND, blind_designs, ub)
    a.set_ylabel("Null-edge rate")
    L.label_line(b, 1, NOMINAL_BAND[0], "nominal 0.05", where="below", ha="center")
    L.panel_header(a, "a", "Uses the logging design")
    L.panel_header(b, "b", "Ignores the logging design")
    L.tidy_grid([[a, b]])
    # the top legend as two columns, each over its panel (the shorter design-blind column padded
    # with blanks); it fits the width, so finalize keeps it as registered
    pad = len(USES) - len(BLIND)
    handles = [M.handle(c, c == T.HERO) for c in USES + BLIND] + [
        Line2D([], [], ls="", marker="")
    ] * pad
    labels = [M.name(c) for c in USES + BLIND] + [""] * pad
    L.fig_legend(fig, handles, labels, where="top").remove()
    reg = L._reg(fig, "legends")
    reg[-1] = (
        fig,
        fig.legend(handles, labels, loc=L.LEGEND_TOP_LOC, ncol=2),
        handles,
        labels,
        "top",
        {},
    )
    return L.finalize(fig)

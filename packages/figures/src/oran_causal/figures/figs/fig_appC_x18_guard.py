"""Appendix C, X18 guard-limit sweep (post hoc, descriptive; declared before it was run; the frozen
1.10 limit stays primary): recovery R of the gated PMRT map referee, the gated oracle map referee
and the best eligible static rule at each guard limit L, the gate budget tied to L
(rho = 1 + (L - 1) / 2), energy floor 0.90, Study 3 EVAL seeds. Message: how far the guard limit
must move before a gated map referee leaves the best static rule.

Source: ``posthoc_x18/x18_eligibility.json``, ``coupled`` cells ``<L>|0.900|<rho(L)>``: per arm
``R``, ``R_ci90`` (90 % paired seed bootstrap) and ``eligible``; ``best_static`` names the static
rule drawn. A cell with a pending gated arm (no complete Part C records) stops the build, so no
point is drawn from an unsimulated policy. Every other cell, the energy floors and the rho sweep
are in the X18 table of the ESM.
"""

from matplotlib.lines import Line2D

from oran_causal.analysis.posthoc import x18 as X
from oran_causal.figures import data, plots
from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import style as S
from oran_causal.figures.registry import FigSpec

PARTS = ("posthoc_x18", X.OUT_ELIG)
SRC = data.source_note(*PARTS)
SPEC = FigSpec(
    name="fig_appC_x18_guard",
    width="full",
    paper=True,
    sources=[SRC],
    caption_hint="Recovery against guard limit; 160 seeds, 90 % CIs.",
)

E_SHOW = 0.90
DODGE = 0.011
HEIGHT = 2.6
LIFT = 0.06
LABEL_GAP = 0.08
GATED = (("CS:GT", 0.0, False), ("CS:PMRT", DODGE, True))
DRAWN = tuple(a for a, _, _ in GATED)
LABELS = {
    "CS:PMRT": "PMRT map with gate",
    "CS:GT": "Oracle map with gate",
    "static": "Highest-R eligible static rule",
    "ineligible": "Ineligible",
}


def cells(doc: dict) -> list[dict]:
    """The coupled cells at the shown energy floor, in guard-limit order.

    Raises:
        ValueError: A cell has a pending gated arm or lacks a drawn series.
    """
    out = []
    for limit in doc["grid"]["L"]:
        k = f"{X.cell_key(limit, E_SHOW)}|{X.key(X.rho_of(limit))}"
        c = doc["coupled"][k]
        if c.get("pending"):
            raise ValueError(f"X18 cell {k}: pending gated arms {c['pending']}")
        missing = [a for a, _, _ in GATED if a not in c["arms"]]
        if missing:
            raise ValueError(f"X18 cell {k}: no row for {missing}")
        out.append(c)
    return out


def _point(ax, x: float, row: dict, code: str, lift: float, hero: bool = False) -> None:
    y, (lo, hi) = row["R"], row["R_ci90"]
    ax.errorbar(
        [x],
        [y],
        yerr=[[y - lo], [hi - y]],
        fmt="none",
        ecolor=P.INK,
        elinewidth=S.LW_WHISKER,
        capsize=1.5,
        capthick=S.LW_WHISKER,
        zorder=3,
        clip_on=False,
    )
    st = M.style(code, hero)
    tick = st["marker"] == "|"
    marker = "_" if tick else st["marker"]
    ax.plot(
        [x],
        [y],
        ls="",
        marker=marker,
        color=st["color"],
        ms=st["ms"] + (3.4 if tick else 0.0),
        mfc=st["mfc"],
        mec=st["mec"],
        mew=1.4 if tick else st["mew"],
        zorder=st["zorder"] + 0.5,
        clip_on=False,
        gid=f"row:{M.resolve(code).key}" + (":hero" if hero else ""),
    )
    if not row["eligible"]:
        ax.plot(
            [x],
            [hi + lift],
            ls="",
            marker="x",
            ms=S.MARKER_SIZE + 2.0,
            mew=1.0,
            color=P.BAD,
            mec=P.BAD,
            zorder=5,
            clip_on=False,
            gid="sem:bad-ineligible",
        )


def _legend(fig) -> None:
    handles = [
        M.handle("CS:PMRT", hero=True),
        M.handle("CS:GT"),
        Line2D(
            [],
            [],
            ls="",
            marker="_",
            ms=S.MARKER_SIZE + 3.4,
            mew=1.4,
            color=P.NEUTRAL,
            mec=P.NEUTRAL,
        ),
        Line2D([], [], ls="", marker="x", ms=S.MARKER_SIZE + 2.0, mew=1.0, color=P.BAD, mec=P.BAD),
    ]
    labels = [LABELS["CS:PMRT"], LABELS["CS:GT"], LABELS["static"], LABELS["ineligible"]]
    L.fig_legend(fig, handles, labels)


def build():
    doc = data.load_json(*PARTS)
    cs = cells(doc)
    limits = [float(x) for x in doc["grid"]["L"]]
    drawn = [r for c in cs for a, r in c["arms"].items() if a in (c["best_static"], *DRAWN)]
    lift = LIFT * (max(r["R_ci90"][1] for r in drawn) - min(r["R_ci90"][0] for r in drawn))
    S.apply()
    fig, ax = plots.figure("full", aspect=HEIGHT / S.FULL)
    for x, c in zip(limits, cs, strict=True):
        bs = c["best_static"]
        if bs is not None:
            _point(ax, x - DODGE, c["arms"][bs], bs, lift)
    for code, dx, hero in GATED:
        xs = [x + dx for x in limits]
        ys = [c["arms"][code]["R"] for c in cs]
        M.plot(ax, xs, ys, code, hero=hero, marker="")
        for x, c in zip(xs, cs, strict=True):
            _point(ax, x, c["arms"][code], code, lift, hero=hero)
    plots.ref_line(ax, X.FROZEN_L, axis="x")
    hi = max(r["R_ci90"][1] for r in drawn)
    span = hi - min(r["R_ci90"][0] for r in drawn)
    ylab = hi + LABEL_GAP * span
    ax.set_ylim(top=ylab + LABEL_GAP * span)
    L.label_line(ax, X.FROZEN_L, ylab, "limit of the main analysis", where="right")
    ax.set_xticks(limits, [f"{x:.2f}" for x in limits])
    ax.set_xlim(limits[0] - 0.03, limits[-1] + 0.03)
    ax.set_xlabel("Guard limit (ratio to accept-all)")
    ax.set_ylabel("Recovery R (higher is better)")
    plots.finish(ax)
    _legend(fig)
    return L.finalize(fig)

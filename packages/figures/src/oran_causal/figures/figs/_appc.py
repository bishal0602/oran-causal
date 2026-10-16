"""Shared loaders and axis helpers of the Appendix C post hoc figures (X2 dither dose-response, X3 design
misspecification, X6 told-width grid). Exploratory checks declared before launch; they change none of C1-C3.

Sources (read-only):
  * XR   ``posthoc_x2_x3/extras_tables.json`` (X2 / X3): ``pooled`` rows, one per experiment x arm x
         design x regime, each with pooled rates (``null_raw``, ``conf_raw``, ...: rate, seed-cluster 95 % bootstrap CI,
         R-30 label). X2 pools E1-E3 R2 at n 1000 and 4000 (40 seeds per design); X3 pools E1-E2 R2 at n 1000 and the
         E4 R3 cell (lam 1, n 1000), 60 seeds.
  * DIAG ``posthoc_x5_x6/diag/diag_tables.json`` (X6): ``x6.curve``, one row per arm x told width
         (E1 R2, n 1000, 200 fresh datasets).
Nothing is recomputed: rates and CIs are read as stored; the figures only select rows.
"""

from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _study4 as T

EXTRAS_PARTS = ("posthoc_x2_x3", "extras_tables.json")
DIAG_PARTS = ("posthoc_x5_x6", "diag", "diag_tables.json")
SRC_XR = source_note(*EXTRAS_PARTS)
SRC_DIAG = source_note(*DIAG_PARTS)
HERO = T.HERO  # PMRT-GBM (pmrt_nl_eq), emphasis only
COLOUR_NOTE = "Colour identifies the method (the same in every figure); it does not mean validity"
TOLD = [0.5, 0.8, 1.0, 1.25, 2.0]  # X3 told-width / told-lambda multipliers


def extras() -> dict:
    return load_json(*EXTRAS_PARTS)


def diag() -> dict:
    return load_json(*DIAG_PARTS)


def pooled(tab: dict, experiment: str, regime: str) -> dict:
    """{(base arm, tuple(design_key)): pooled row} of one experiment and regime (XR ``pooled``; an
    X3 variant arm such as ``pmrt_eq.w080`` is found under its base arm and its design key, here
    ``("width", 0.8)``).
    """
    return {
        (r["base_arm"], tuple(r["design_key"])): r
        for r in tab["pooled"]
        if r["experiment"] == experiment and r["regime"] == regime
    }


def points(rows, rate: str):
    """(x, rate, lo, hi) lists of the (x, row) pairs that carry ``rate`` (a None rate = no cell: left out)."""
    keep = [(x, r[rate]) for x, r in rows if r is not None and r.get(rate) is not None]
    return (
        [x for x, _ in keep],
        [v["rate"] for _, v in keep],
        [v["ci"][0] for _, v in keep],
        [v["ci"][1] for _, v in keep],
    )


def draw(ax, arm: str, arms, xs, ys, lo, hi, step_pt: float = L.DODGE_PT, whiskers: bool = False):
    """One series in its paper style (oran_causal.figures.methods via the Study 4 helper): dodged by
    legend order, CI band for the design-based / design-adjusted arms, or (``whiskers``) ink CI
    whiskers instead of bands, for panels whose wide CIs would merge into one mass as overlapping
    bands.
    """
    if not whiskers:
        return T.series(ax, xs, ys, lo, hi, arm, arms, step_pt)
    tr = T.series(ax, xs, ys, None, None, arm, arms, step_pt)
    for x, y, a, b in zip(
        xs, ys, lo, hi, strict=False
    ):  # plots.errorbars style, drawn under the markers
        ax.errorbar(
            [x],
            [y],
            yerr=[[y - a], [b - y]],
            fmt="none",
            ecolor=P.INK,
            elinewidth=0.6,
            capsize=1.3,
            capthick=0.6,
            zorder=2.5,
            transform=tr,
        )
    return tr


def level_axis(ax, levels, label: str, names=None) -> list[int]:
    """Categorical x axis of discrete design levels (told / true multipliers, switch delays), evenly
    spaced at 0, 1, ... so neighbouring levels such as 1 and 1.25 never crowd. Returns the positions
    (index of each level).
    """
    pos = list(range(len(levels)))
    ax.set_xticks(pos, names or [f"{v:g}" for v in levels])
    ax.set(xlim=(-0.45, len(levels) - 0.55), xlabel=label)
    return pos


def at(levels, rows) -> list:
    """Map (level, row) pairs to (position, row) on a ``level_axis``."""
    return [(levels.index(v), r) for v, r in rows]

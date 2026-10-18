"""Study 2: what the map-driven referee rejected, PMRT map vs oracle (GT) map (system-level
simulator, 160 EVAL seeds).

One row per control-parameter family (labelled with its writer xApp, entities.param_label, and the
one direction the referee can reject for it); two bars per row, the map-driven referee on the PMRT
map (hero) and on the oracle map (ink, dash-dot). Values = rejected units summed over the 160 seeds.
Sources: FD option_a.eval.counts.<arm>.policy.defer_<family> (absent key = 0 rejected units). The
direction of each family's rejections is read from FD
option_a.decision_tables.<arm>["<family><+|->"]: the direction whose rule rejects (identical for
both maps; checked at build time). Consistency checks at build time (no new numbers): the ptx row
equals FD option_a.eval.ptx_up_defer_units and the sleep row FD option_a.eval.sleep_defer_units. No
re-aggregation.
"""

from oran_causal.figures import entities as E
from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _s12a as H
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study2_deferrals",
    width="full",
    paper=True,
    sources=[ST.SRC_FD],
    caption_hint="Units rejected by the map-driven referee on the PMRT map and on the oracle (GT) map, per control-"
    "parameter family (writer xApp in parentheses) and the direction the referee rejects for it, summed "
    "over the 160 Study 2 EVAL seeds of the system-level simulator. Without the ptx -> neighbour RLF edge, "
    "the PMRT-map referee rejected far more ptx-up units than the oracle-map referee. Source: "
    "figures_data.json (option_a.eval.counts, option_a.decision_tables).",
)

ARMS = ["MG:PMRT", "MG:GT"]  # legend / family order: design-based, then the oracle map
HERO = "MG:PMRT"
FAMILIES = ["carrier", "sleep", "ptx", "prot_min"]
DIRECTION = {"+": "up", "-": "down"}
THICK, STEP = 0.34, 0.40  # bar thickness and offset of the two bars of a row (row units)


def _direction(tables: dict, fam: str) -> str:
    dirs = {tuple(d for d in "+-" if "defer" in tables[a][f"{fam}{d}"]) for a in ARMS}
    if len(dirs) != 1 or len(next(iter(dirs))) != 1:
        raise ValueError(f"{fam}: deferral direction differs between maps or is not unique: {dirs}")
    return DIRECTION[next(iter(dirs))[0]]


def build():
    a = ST.fd()["option_a"]
    counts, tables = a["eval"]["counts"], a["decision_tables"]
    vals = {arm: {f: counts[arm]["policy"].get(f"defer_{f}", 0) for f in FAMILIES} for arm in ARMS}
    for arm in ARMS:  # the figure's ptx / sleep rows are the report's asserted numbers
        assert vals[arm]["ptx"] == a["eval"]["ptx_up_defer_units"][arm], arm
        assert vals[arm]["sleep"] == a["eval"]["sleep_defer_units"][arm], arm
    fig, ax = plots.figure("full", aspect=0.36)
    ys = list(range(len(FAMILIES)))[::-1]
    handles = {}
    for y, fam in zip(ys, FAMILIES, strict=False):
        for i, arm in enumerate(ARMS):
            yy = y + STEP / 2 - i * STEP
            b = H.hbar(ax, yy, vals[arm][fam], arm, THICK, hero=arm == HERO)
            if b is not None:
                handles.setdefault(arm, b)
            H.value_right(ax, vals[arm][fam], yy, H.fmt_count(vals[arm][fam]))
    top = max(max(v.values()) for v in vals.values())
    ax.set_xlim(0, top * 1.28)
    ticks = list(range(0, int(top) + 1, 2000))
    ax.set_xticks(ticks, [f"{t // 1000:g}" for t in ticks])
    ax.set_xlabel("Rejected units (×10³)")
    plots.finish(ax)
    labels = [f"{E.param_plain_label(f)}, {_direction(tables, f)}" for f in FAMILIES]
    H.row_axes(ax, (-0.6, len(FAMILIES) - 0.4), ys, labels)
    ax.tick_params(axis="y", labelsize=S.TICK_PT)
    names = [M.name(arm) + ("" if M.name(arm).endswith("map") else " map") for arm in ARMS]
    L.fig_legend(fig, [handles[arm] for arm in ARMS], names)
    return L.finalize(fig)

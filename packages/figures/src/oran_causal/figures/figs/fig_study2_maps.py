"""Study 2: quality of the causal maps learned from confounded incumbent logs (system-level simulator), scored against
the Study 2 knockout ground truth; shown descriptively (map quality was not a criterion fixed before evaluation).

Edge ledger: one row per map under
family banners (bold method name, DEV / placebo threshold sub-rows, as the referee figures), and
  * one stacked bar of the map's edges by ground-truth status, in the Study 1 edge-status colours (DESIGN s.12.1
    exception, as fig_study1_discovery): teal GT-TRUE with the right sign, terracotta GT-TRUE with the wrong sign, light
    terracotta GT-NULL (false), gray GT-INDET (not scored), with the total at the bar end;
  * a count table of the same four categories (a short bar in the category colour under each head ties the column to
    its segment), then, after a rule, the edges the same method declared on the placebo logs: a separate run, not part
    of the map, so not a segment of the bar and no colour key (a dash: no placebo counterpart).
One panel, so no panel letter. Source: FD option_a.map_quality.<map> (true_right, true_wrong, null, indet, edges,
unknown, placebo_edges); the four categories are asserted to add up to the stored edge count. blanket2 has no paper
name and is left out.
"""

from matplotlib.patches import Patch, Rectangle

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _method_rows as H
from oran_causal.figures.figs import _referee as R
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study2_maps",
    width="full",
    paper=True,
    sources=[ST.SRC_FD],
    caption_hint="Causal maps learned from 600 confounded incumbent episodes of the system-level simulator (Study 2), "
    "scored against the knockout ground truth (GT). One row per map (DEV = DEV-tuned threshold, placebo = "
    "placebo-calibrated threshold). The bar splits the map's edges into GT-TRUE edges with the right sign "
    "(teal) or the wrong sign (terracotta), GT-NULL edges (false, light terracotta) and GT-INDET edges "
    "(not scored, gray), with the total at its end; the table gives the same counts, then the edges the "
    "same method declared on the placebo logs, where every edge is false (a dash: no placebo counterpart). "
    "Shown descriptively: map quality was not a criterion fixed before evaluation. Source: "
    "figures_data.json (option_a.map_quality).",
)

OK, ERR = P.TEAL, P.TERRACOTTA  # edge status, as fig_study1_discovery (palette "H2")
NULL_FC, UNSC_FC = P.NULL_FILL, P.UNSCORED_FILL
# (key, column head, fill, edge, legend label)
CATS = [
    ("true_right", "Right", OK, "none", "GT-TRUE, right sign"),
    ("true_wrong", "Wrong", ERR, "none", "GT-TRUE, wrong sign"),
    ("null", "NULL", NULL_FC, ERR, "GT-NULL (false edge)"),
    ("indet", "INDET", UNSC_FC, "none", "GT-INDET (not scored)"),
]
PLACEBO = "placebo_edges"
BANNERS = [("based", "Design-based map"), ("blind", "Design-blind maps"), ("ref", "Reference maps")]
THICK = 0.56  # bar thickness (row units)
SEP = 0.45  # extra gap before the placebo column (table units)


def _maps(mq: dict) -> list[str]:
    codes = []
    for c in mq:
        try:
            M.resolve(c)
        except KeyError:  # no paper name (blanket2): never plotted
            continue
        codes.append(c)
    keys = list(M.METHODS)
    fam = list(M.FAMILIES)
    return sorted(
        codes,
        key=lambda c: (
            fam.index(M.resolve(c).method.family),
            keys.index(M.resolve(c).key),
            list(R.THRESHOLD | {None: ""}).index(M.resolve(c).threshold),
        ),
    )


def _family(code: str) -> str:
    f = M.resolve(code).method.family
    return "based" if f == "based" else "blind" if f.startswith("blind") else "ref"


def build():
    mq = ST.fd()["option_a"]["map_quality"]
    codes = _maps(mq)
    for code in codes:  # the four categories partition the map's edges
        q = mq[code]
        assert not q.get("unknown") and sum(q[k] for k, *_ in CATS) == q["edges"], (code, q)
    sections = [(text, [c for c in codes if _family(c) == key]) for key, text in BANNERS]
    rows, gcodes = R.sections_rows([s for s in sections if s[1]])
    banners, groups, subrows, (y0, y1) = L.method_rows(rows, gap=0.55)
    ys = [yy for yy, _, _ in subrows]
    flat = [c for g in gcodes for c in g]
    ylim = (y0 - 0.3, y1 + 1.0)
    fig, (a, t) = plots.figure(
        "full", aspect=0.95, ncols=2, sharey=True, width_ratios=[1.43, 1.22]
    )  # t wider for 7 pt heads
    cols = [k for k, *_ in CATS] + [PLACEBO]
    cx = lambda j: j + 0.5 + (SEP if cols[j] == PLACEBO else 0)
    for yy, code in zip(ys, flat, strict=False):
        q = mq[code]
        left = 0
        for k, _, fc, ec, _ in CATS:
            if q[k]:
                a.barh(
                    yy,
                    q[k],
                    THICK,
                    left=left,
                    color=fc,
                    edgecolor=ec,
                    linewidth=0.6 if ec != "none" else 0,
                    zorder=3,
                )
                left += q[k]
        H.value_right(a, left, yy, str(left))
        for j, k in enumerate(cols):
            v = q[k]
            t.text(
                cx(j),
                yy,
                "–" if v is None else str(v),
                ha="center",
                va="center",
                fontsize=S.ANNOT_PT,
                color=P.INK,
            )
    # table heads: a short bar in the category colour under each; the placebo column after a rule,
    # no colour key
    head = ylim[1] - 0.55
    for j, (_, lab, fc, ec, _) in enumerate(CATS):
        t.text(cx(j), head, lab, ha="center", va="bottom", fontsize=S.ANNOT_PT, color=P.INK)
        t.add_patch(
            Rectangle(
                (j + 0.2, head - 0.32),
                0.6,
                0.22,
                fc=fc,
                ec=ec,
                lw=0.6 if ec != "none" else 0,
                gid="column-key",
            )
        )
    jp = cols.index(PLACEBO)
    t.text(
        cx(jp),
        head,
        "Placebo\nlogs",
        ha="center",
        va="bottom",
        fontsize=S.ANNOT_PT,
        color=P.INK,
        linespacing=1.0,
    )
    t.plot(
        [cx(jp) - 0.5 - SEP / 2] * 2,
        [ylim[0] + 0.2, head + 0.9],
        color=P.GRID,
        lw=0.6,
        marker="",
        gid="column-rule",
    )
    t.set_xlim(0, len(cols) + SEP)
    t.set_ylim(*ylim)
    t.axis("off")
    t.patch.set_visible(False)
    a.set_xlim(0, max(mq[c]["edges"] for c in flat) * 1.12)
    a.set_xticks([0, 10, 20, 30, 40])
    a.set_xlabel("Edges in the map")
    plots.finish(a)
    H.row_axes(a, ylim, ys, [R.THRESHOLD.get(M.resolve(c).threshold, "") for c in flat])
    L.tidy_grid([[a]])
    for y, text in banners:
        L.banner(fig, a, y, L.BANNER_H * 0.92, text)
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    w = [x.get_window_extent(r).width for x in a.get_yticklabels() if x.get_text()]
    L.method_labels(a, groups, (max(w) if w else 0) * 72 / fig.dpi + 5)
    a.annotate(
        "Map edges, by ground truth",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, L.HEADER_RISE_PT),
        textcoords="offset points",
        ha="left",
        va="baseline",
        fontsize=L.TAG_PT,
        color=P.INK,
        gid="panel-tag",
    )
    handles = [Patch(fc=fc, ec=ec, lw=0.6 if ec != "none" else 0) for _, _, fc, ec, _ in CATS]
    L.fig_legend(fig, handles, [lab for *_, lab in CATS], where="top")
    return L.finalize(fig)

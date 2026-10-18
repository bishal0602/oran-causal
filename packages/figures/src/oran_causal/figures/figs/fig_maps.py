"""Causal maps as tile grids against the knockout ground truth, randomised logs (F5, Section 6.2).

One message: PMRT's map looks like the ground truth; the baselines' maps miss edges or get signs
wrong. Four facets of the 60 hypotheses (rows: 4 parameter families x 3 cell relations; columns: 5
KPIs): ground truth, PMRT and the two baselines with the highest F1 on these logs (Granger + BY,
Granger; asserted below). Each tile is empty or a triangle pointing in the sign of the effect of
raising the parameter. Tiles are the colour field (semantic roles, DESIGN s.5): ground truth - blue
tint, white triangle; right sign - teal tile, white triangle; wrong sign - terracotta tile, white
triangle; false edge (ground truth null) - light terracotta tile, terracotta triangle; missed -
white tile, gray outline triangle in the true sign; not scored (indeterminate) - light gray hatch;
true null - the gray panel field. PMRT's facet is framed in its blue. The legend row shows the tiles
themselves. Row labels and KPI labels on the left facet only (one row of facets).

Source: V4 gt.cells and slices_result.pooled.metrics.<code>.declared through _study1.load(), which
re-scores every map against the stored metrics. The tile counts are asserted against tab:maps (a)
(R, W, N, Ed.).
"""

from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from oran_causal.figures import entities as E
from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _study1 as S1
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_maps",
    width="full",
    paper=True,
    sources=[ST.SRC_V4, ST.SRC_FD],
    caption_hint="Causal maps learned from randomised logs, against the knockout ground truth.",
)

FAMILIES = ["carrier", "sleep", "ptx", "prot_min"]
RELATIONS = [("own", "own cell"), ("nbr", "neighbour"), ("far", "far cell")]
KPIS = [
    ("load", "load"),
    ("e", "energy"),
    ("v", "violations"),
    ("pv", "slice violations"),
    ("rlf", "RLF"),
]
BASELINES = ["corr", "granger", "granger_by", "shap_gbdt", "two_tower", "qacm", "int"]
# tab:maps (a), Study 1: Ed., R, W, N
EXPECT = {S1.PRIMARY: (28, 23, 0, 2), "granger": (32, 14, 4, 8), "granger_by": (45, 16, 6, 15)}
N_TRUE = 29

HATCH = "/////"
# category -> (tile face, tile edge, triangle face, triangle edge, semantic tag)
TILES = {
    "truth": (P.tint(P.OURS, 0.62), "none", P.WHITE, P.WHITE, "sem:ours-tile"),
    "right": (P.GOOD, "none", P.WHITE, P.WHITE, "sem:good-tile"),
    "wrong": (P.BAD, "none", P.WHITE, P.WHITE, "sem:bad-tile"),
    "false": (P.tint(P.BAD, 0.30), "none", P.BAD, P.BAD, "sem:bad-tile"),
    "missed": (P.WHITE, P.tint(P.MISSED, 0.7), "none", P.MISSED, "missed-tile"),
    "indet": (P.tint(P.NEUTRAL, 0.22), "none", None, None, "indet-tile"),
    None: (
        P.FIELD,
        "none",
        None,
        None,
        "null-tile",
    ),  # true null / nothing declared: the panel field
}
GAP = 0.9  # blank rows between family blocks (family header in the left facet)
FAMILY_IN_PT = 46  # family header starts left of the longest relation label
CATS = ["right", "wrong", "false", "missed"]
LABELS = {
    "right": "right sign",
    "wrong": "wrong sign",
    "false": "false edge",
    "missed": "missed",
    "indet": "not scored",
}


def _tile(gt: dict, declared: int | None) -> tuple[str | None, int]:
    """(category, sign to draw) of one hypothesis; declared: the declared sign, 0 = not declared,
    None = the ground-truth facet.
    """
    if gt["status"] == "INDET":
        return "indet", 0
    if declared is None:
        return ("truth", gt["sign"]) if gt["status"] == "TRUE" else (None, 0)
    if declared:
        s = declared
        if gt["status"] == "NULL":
            return "false", s
        return ("right" if s == gt["sign"] else "wrong"), s
    return ("missed", gt["sign"]) if gt["status"] == "TRUE" else (None, 0)


def _draw_tile(
    ax, x: float, y: float, cat: str | None, sign: int, size: float = 0.9, ms: float = 6.2
):
    """One tile at (x, y) (lower-left in data units, y axis inverted) with its triangle."""
    fc, ec, tf, te, gid = TILES[cat]
    pad = (1 - size) / 2
    ax.add_patch(
        Rectangle(
            (x + pad, y + pad),
            size,
            size,
            fc=fc,
            ec=_edge(cat, ec),
            lw=0.6 if ec != "none" else 0,
            hatch=HATCH if cat == "indet" else None,
            zorder=1,
            gid=gid,
        )
    )
    if tf is not None:
        ax.plot(
            [x + 0.5],
            [y + 0.5 + (0.04 if sign > 0 else -0.04)],
            marker="^" if sign > 0 else "v",
            ms=ms,
            ls="",
            mfc=tf,
            mec=te,
            mew=0.9 if tf == "none" else 0.4,
            zorder=3,
            gid=gid.replace("tile", "mark"),
        )


def _edge(cat: str | None, ec: str) -> str:
    """Tile edge colour; the hatch of a not-scored tile is drawn in its (white, zero-width) edge colour."""
    return P.WHITE if cat == "indet" else ec


class _Key:
    """Legend proxy: one tile category."""

    def __init__(self, cat: str):
        self.cat = cat


class _KeyHandler(HandlerBase):
    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        fc, ec, tf, te, gid = TILES[orig.cat]
        side = height * 1.45
        x0, y0 = xdescent + (width - side) / 2, -ydescent + (height - side) / 2
        out = [
            Rectangle(
                (x0, y0),
                side,
                side,
                fc=fc,
                ec=_edge(orig.cat, ec),
                lw=0.6 if ec != "none" else 0,
                hatch=HATCH if orig.cat == "indet" else None,
                transform=trans,
                gid=gid,
            )
        ]
        if tf is not None:
            out.append(
                Line2D(
                    [x0 + side / 2],
                    [y0 + side / 2],
                    marker="^",
                    ms=5.4,
                    ls="",
                    mfc=tf,
                    mec=te,
                    mew=0.9 if tf == "none" else 0.4,
                    transform=trans,
                    gid=gid.replace("tile", "mark"),
                )
            )
        return out


def data() -> tuple[dict, list[str]]:
    s1 = S1.load()
    gt = {tuple(h.split("|")): c for h, c in s1["gt"].items()}
    assert sum(c["status"] == "TRUE" for c in gt.values()) == N_TRUE
    f1 = {b: s1["pooled"][b]["score"]["ov"]["f1"] for b in BASELINES}
    top = sorted(BASELINES, key=lambda b: -f1[b])[:2]
    assert top == ["granger_by", "granger"], f1  # the two strongest baselines on these logs
    maps = {}
    for code in [S1.PRIMARY, *top]:
        dec = {tuple(h.split("|")): s for h, s in s1["pooled"][code]["declared"].items()}
        cells = {h: _tile(gt[h], dec.get(h, 0)) for h in gt}
        n = {c: sum(v[0] == c for v in cells.values()) for c in (*CATS, "indet")}
        ed = len(dec)
        nd_indet = sum(gt[h]["status"] == "INDET" for h in dec)
        assert (ed, n["right"], n["wrong"], n["false"]) == EXPECT[code], (code, ed, n)
        assert n["missed"] == N_TRUE - n["right"] - n["wrong"], code
        assert ed == n["right"] + n["wrong"] + n["false"] + nd_indet, code
        maps[code] = cells
    maps["GT"] = {h: _tile(gt[h], None) for h in gt}
    return maps, top


def _rows() -> list[tuple[float, str, str]]:
    """(y, family, relation) top to bottom, with a gap row before every family block."""
    out, y = [], 0.0
    for f in FAMILIES:
        y += GAP
        for r, _ in RELATIONS:
            out.append((y, f, r))
            y += 1
    return out


def build():
    maps, top = data()
    facets = [("GT", "Ground truth"), (S1.PRIMARY, "PMRT")] + [
        (c, M.resolve(c, study13=True).method.name) for c in top
    ]
    return draw(maps, facets, hero=S1.PRIMARY)


def draw(maps: dict, facets: list[tuple[str, str]], hero: str):
    """One row of tile-map facets; maps[code][(family, relation, kpi)] = (category, sign)."""
    rows = _rows()
    ymax = rows[-1][0] + 1
    fig, axs = plots.figure("full", aspect=0.74, ncols=len(facets), gridspec_kw={"wspace": 0.04})
    for j, (ax, (code, title)) in enumerate(zip(axs, facets, strict=False)):
        cells = maps[code]
        if code == hero:  # our map: framed in the hero colour
            ax.add_patch(
                Rectangle(
                    (-0.18, rows[0][0] - 0.25),
                    len(KPIS) + 0.36,
                    ymax - rows[0][0] + 0.43,
                    fc="none",
                    ec=P.OURS,
                    lw=1.1,
                    zorder=0,
                    clip_on=False,
                    gid="sem:ours-zone",
                )
            )
        for y, f, r in rows:
            for x, (k, _) in enumerate(KPIS):
                cat, sign = cells[(f, r, k)]
                _draw_tile(ax, x, y, cat, sign)
        ax.set_xlim(0, len(KPIS))
        ax.set_ylim(ymax, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.annotate(
            title,
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=S.LABEL_PT,
            color=P.INK,
            fontweight="bold" if code == hero else "normal",
            gid="facet-title",
        )
        for x, (_, lab) in enumerate(KPIS if j == 0 else []):
            ax.annotate(
                lab,
                xy=(x + 0.5, ymax),
                xycoords="data",
                xytext=(0, -3),
                textcoords="offset points",
                ha="center",
                va="top",
                rotation=90,
                fontsize=S.ANNOT_PT,
                color=P.INK,
                gid="kpi-label",
            )
        if j == 0:
            for y, f, r in rows:
                ax.annotate(
                    dict(RELATIONS)[r],
                    xy=(0, y + 0.5),
                    xycoords="data",
                    xytext=(-3, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=S.ANNOT_PT,
                    color=P.INK,
                    gid="relation-label",
                )
            for i, f in enumerate(FAMILIES):
                y0 = rows[3 * i][0]
                ax.text(
                    0,
                    y0,
                    E.param_label(f),
                    transform=offset_copy(
                        ax.transData, fig, x=-FAMILY_IN_PT, y=0.5, units="points"
                    ),
                    ha="left",
                    va="bottom",
                    fontsize=S.ANNOT_PT,
                    color=P.INK,
                    fontweight="bold",
                    gid="family-label",
                )
    keys = (*CATS, "indet")
    L.fig_legend(
        fig,
        [_Key(c) for c in keys],
        [LABELS[c] for c in keys],
        where="top",
        handler_map={_Key: _KeyHandler()},
        handlelength=1.2,
    )
    return L.finalize(fig)

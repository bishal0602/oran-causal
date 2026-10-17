"""Paper-wide method styling (DESIGN s.5.1 / s.12, Q8: one global method-colour mapping, decided 2026-10-05 by the user,
mapping filled 2026-10-06 overnight with the panel's changes). The single source of how a method looks: every figure
that shows a method (Studies 1-4, bridging analysis, Appendix C) styles it through this module, by arm code.

Encoding:
  * colour = method ([DECIDED 2026-10-08, user] DESIGN s.12 banner): one fixed colour per method for the whole
    paper, from Harbor's six hues plus three Harbor tints (palette.py). PMRT / PMRT-GBM blue; no figure shows two
    methods in one hue (Studies 1-3 and Study 4 reuse a hue only for methods that never share a figure). Methods
    that are drawn only in ink rows or tables are neutral gray. References / static rules ink / gray. Banners,
    legends and captions name the three families (design-based, design-adjusted, design-blind); colour never
    means "valid".
  * line style: solid for every method ([DECIDED 2026-10-08, user]: no dashes); references are ink / gray with
    their own dash.
  * marker = base method, the same shape in every family (plain and design-adjusted partial correlation are both
    ``^``). PMRT-GBM and the frozen PMRT never share a glyph.
  * exactly one hero method per figure (``hero=True``: filled marker, 1.9 pt, on top); its MG and CS referee arms
    both carry it. In Study 4 the hero is emphasis only; no legend or caption calls it "proposed" or "best".
  * edge-certified arms (``CS:<map>``) look exactly like their map's arm ([DECIDED 2026-10-08, user]: the ink gate ring
    is dropped; the banners name the gate); ``MG:<map>`` looks the same in Study 2 and Study 3 figures.
Artists carry ``gid`` tags (``method:<key>[:hero][:cs]``, ``row:<key>``, ``band:<key>``) that tests/test_methods.py
reads to check the rules on every registered figure.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures import style as S

DASH = (0, (3, 1.6))  # design-blind series (ignores the design)
DOT = (0, (1, 1.5))  # random map
DASHDOT = (0, (4, 1.5, 1, 1.5))  # oracle map
REF_DASH = (0, (4, 2))  # reference lines (DESIGN s.4 / s.5.4): ink at 60 %, LW_REF
BAND_ALPHA = {True: 0.14, False: 0.08}  # CI bands: hero / others (DESIGN s.7)


@dataclass(frozen=True)
class Family:
    paper: str  # the family name a banner / caption may use (three families + references)
    color: str  # default line colour of a family member without its own Method.color
    mcolor: str  # default marker edge and INVALID cross colour, likewise
    ls: object
    slot: int | None  # DESIGN s.5.1 hue slot (None = neutral)
    alpha: float = 1.0
    fill: str | None = None  # default band / bar fill colour; None = same as color


# [DECIDED 2026-10-08, user] Colour = method (fixed, paper-wide; palette.py). All methods solid; the
# family is named by banners, legends and captions. Methods without a colour (ink rows / tables
# only) are neutral gray.
FAMILIES = {
    "based": Family("Design-based tests", P.HERO, P.HERO, "-", 0),
    "adjusted": Family("Design-adjusted tests", P.NEUTRAL, P.NEUTRAL, "-", None),
    "blind_test": Family("Design-blind methods", P.NEUTRAL, P.NEUTRAL, "-", None),
    "blind_learned": Family("Design-blind methods", P.NEUTRAL, P.NEUTRAL, "-", None),
    "oracle": Family("Reference maps", P.INK, P.INK, DASHDOT, None, alpha=0.6),
    "random": Family("Reference maps", P.NEUTRAL, P.NEUTRAL, DOT, None),
    "static": Family("Static rules", P.NEUTRAL, P.NEUTRAL, "none", None),
}
FAMILY_ORDER = list(
    FAMILIES
)  # legend order: design-based, design-adjusted, design-blind, references (DESIGN s.5.1)


@dataclass(frozen=True)
class Method:
    name: str  # paper name (TERMINOLOGY s.6), as in legends
    family: str
    marker: (
        str | None
    )  # None = tables only so far; takes a free shape (recorded in DESIGN s.12) if plotted
    color: str | None = None  # Harbor method colour (palette.py); None = the family default
    rows_only: bool = False  # only ever drawn as labelled rows / bars, never as a line series
    ls: object = (
        None  # overrides the family line style (variant of a test drawn in the same colour)
    )
    filled: bool = False  # filled marker for a non-hero variant (PMRT-Lin, R3 covariates)


# key -> Method. Order inside a family = legend order. Variants that share hue and marker (partial
# correlation plain / HAC / HAC fixed-b; DEV-tuned vs plain Granger) may share a figure only as
# labelled rows (tests/test_methods.py).
METHODS = {
    # design-based tests (blue)
    "pmrt": Method("PMRT", "based", "o", color=P.BLUE),  # frozen test, Studies 1-3
    "pmrt_gbm": Method("PMRT-GBM", "based", "D", color=P.BLUE),
    "pmrt_lin": Method("PMRT-Lin", "based", "D", color=P.PMRT_LIN),
    "pmrt_lin_r3": Method("PMRT-Lin, R3 covariates", "based", "H", color=P.PMRT_LIN, filled=True),
    # design-adjusted tests (Study 4)
    "pcorr_adj": Method("Design-adjusted partial correlation", "adjusted", "^", color=P.PURPLE),
    "granger_adj": Method("Design-adjusted Granger", "adjusted", "s", color=P.TERRACOTTA),
    "rcot_adj": Method("Design-adjusted RCoT", "adjusted", "v"),  # ink rows only
    # design-blind statistical tests
    "corr": Method(
        "Correlation", "blind_test", "p", color=P.GOLD
    ),  # not X: reads as an INVALID cross
    "pcorr": Method("Plain partial correlation", "blind_test", "^", color=P.GREEN),
    "pcorr_hac": Method("Plain partial correlation, HAC", "blind_test", "^", color=P.GREEN),
    "pcorr_hac_fb": Method(
        "Plain partial correlation, HAC (fixed-b)", "blind_test", "^", color=P.GREEN
    ),
    "granger": Method("Plain Granger", "blind_test", "s", color=P.TEAL),  # Study 4 / bridging
    "granger_dev": Method(
        "Granger", "blind_test", "s", color=P.TEAL
    ),  # Studies 1-3 (DEV-tuned threshold)
    "granger_by": Method("Granger + BY", "blind_test", "P", color=P.TERRACOTTA),
    "rcot": Method("Plain RCoT", "blind_test", "v"),  # ink rows only
    "mscr": Method("MSCR (n ≤ 1000)", "blind_test", "8"),
    # design-blind learned / structure discovery (Studies 1-3)
    "shap_gbdt": Method("SHAP-GBDT", "blind_learned", ">", color=P.PURPLE),
    "two_tower": Method("Two-tower", "blind_learned", "<", color=P.GREEN),
    "qacm": Method(
        "QACM", "blind_learned", "X", color=P.SLATE
    ),  # Studies 1-3 only (no INVALID marks)
    "int": Method("INT", "blind_learned", "h", color=P.LILAC),
    "cdl": Method("CDD O-RAN discovery", "blind_learned", "d"),
    "pc": Method("PC", "blind_learned", None),
    "notears": Method("NOTEARS", "blind_learned", None),
    "shap_dag": Method("SHAP-DAG", "blind_learned", None),
    # reference maps and static rules (neutrals)
    "oracle": Method("Oracle map", "oracle", "*"),
    "random": Method("Random map", "random", "+"),
    "accept_all": Method("Accept-all", "static", "|", rows_only=True),
    "never_sleep": Method("Never-sleep", "static", "|", rows_only=True),
    "sg_only": Method(
        "Slice Guarantee only", "static", "|", rows_only=True
    ),  # xApp paper names (TERMINOLOGY s.3)
    "es_only": Method("ES only", "static", "|", rows_only=True),
    "poweres_only": Method("Power ES only", "static", "|", rows_only=True),
    "es_poweres": Method("ES + Power ES", "static", "|", rows_only=True),
    "freeze": Method("Freeze", "static", "|", rows_only=True),
    "incumbent": Method("Incumbent", "static", "|", rows_only=True),
}
# shared hue + marker on purpose (labelled-row figures only)
VARIANTS = [{"pcorr", "pcorr_hac", "pcorr_hac_fb"}, {"granger", "granger_dev"}]

# arm code (Study 4 / bridging / Appendix C, Studies 1-3 discovery, referee maps) -> method key
ALIASES = {
    "pmrt_nl_eq": "pmrt_gbm",
    "pmrt_eq": "pmrt_lin",
    "pmrt_r3": "pmrt_lin_r3",
    "pcorr_eq": "pcorr_adj",
    "granger_eq": "granger_adj",
    "rcot2_eq": "rcot_adj",
    "pcorr_native": "pcorr",
    "granger_native": "granger",
    "rcot2_native": "rcot",
    "mscr_native": "mscr",
    "pc_native": "pc",
    "granger_by": "granger_by",
    "PMRT": "pmrt",
    "GT": "oracle",
    "rand": "random",
    "noarb": "accept_all",
    "sub:SliceGuarantee": "sg_only",
    "sub:ES": "es_only",
    "sub:PowerES": "poweres_only",
    "sub:ES+PowerES": "es_poweres",
}
for _stat in ("plain_c", "loadsp_c", "max"):  # Study 1: the 9 PMRT statistic x layer combinations
    for _layer in ("by", "dagger1s", "wby1s"):
        ALIASES[f"{_stat}+{_layer}"] = "pmrt"
STUDY13 = {"granger": "granger_dev"}  # Studies 1-3 discovery codes that differ from Study 4


@dataclass(frozen=True)
class Arm:
    key: str
    method: Method
    gate: str | None = None  # "MG" map-driven referee, "CS" edge-certified referee
    threshold: str | None = None  # "dev" / "plc" (DEV-tuned / placebo-calibrated map threshold)


def resolve(code: str, study13: bool = False) -> Arm:
    """Arm code -> Arm. ``MG:<map>[@dev|@plc]`` / ``CS:<map>…`` are referee arms on a map;
    ``study13`` selects the Studies 1-3 meaning of an ambiguous discovery code (``granger`` =
    DEV-tuned Granger).
    """
    gate = thr = None
    c = code
    if c[:3] in ("MG:", "CS:"):
        gate, c = c[:2], c[3:]
    if "@" in c:
        c, thr = c.split("@")
    if study13 or gate:
        c = STUDY13.get(c, c)
    key = ALIASES.get(c, c)
    if key not in METHODS:
        raise KeyError(
            f"no paper style for arm '{code}' (no paper name yet: B2 / blanket2 / eq_min / pc_eq / mscr_eq)"
        )
    return Arm(key, METHODS[key], gate, thr)


def family(code: str) -> Family:
    return FAMILIES[resolve(code).method.family]


def name(code: str) -> str:
    return resolve(code).method.name


def color(code: str) -> str:
    m = resolve(code).method
    return m.color or FAMILIES[m.family].color


def marker_color(code: str) -> str:
    m = resolve(code).method
    return m.color or FAMILIES[m.family].mcolor


def fill_color(code: str) -> str:
    """Band / bar fill colour: the method's colour, else the family's fill, else the family's line colour."""
    m = resolve(code).method
    f = FAMILIES[m.family]
    return m.color or f.fill or f.color


def order(codes: Iterable[str]) -> list[str]:
    """Codes in legend order: family (design-based, design-adjusted, design-blind, references), then
    table order.
    """
    keys = list(METHODS)
    return sorted(
        codes,
        key=lambda c: (FAMILY_ORDER.index(resolve(c).method.family), keys.index(resolve(c).key)),
    )


def _gid(arm: Arm, hero: bool) -> str:
    return f"method:{arm.key}" + (":hero" if hero else "") + (":cs" if arm.gate == "CS" else "")


def style(code: str, hero: bool = False, label: str | None = None) -> dict:
    """Line2D keyword arguments of a series (line + markers)."""
    arm = resolve(code)
    m, f = arm.method, FAMILIES[arm.method.family]
    if m.marker is None:
        raise ValueError(
            f"{m.name} has no marker yet (tables only): pick a free shape and record it in DESIGN s.12"
        )
    c, mc = color(code), marker_color(code)
    filled = hero or m.filled
    st = {
        "color": c,
        "ls": m.ls if m.ls is not None else f.ls,
        "lw": S.LW_PROPOSED if hero else S.LW_BASELINE,
        "marker": m.marker,
        "mfc": mc if filled else P.WHITE,
        "mec": P.WHITE if filled else mc,
        "mew": 0.5 if filled else 0.8,
        "ms": S.MARKER_SIZE + (0.4 if hero else 0),
        "alpha": f.alpha,
        "zorder": 4 if hero else 3,
        "label": label,
        "gid": _gid(arm, hero),
    }
    if m.marker in ("+", "|"):  # line-only glyphs: no fill, coloured stroke
        st.update(mfc=mc, mec=mc, mew=1.0)
    if m.family == "oracle":  # open ink star, subordinate to the hero
        st.update(mfc="none", zorder=2.5, ms=S.MARKER_SIZE + 1.0)
    return st


def plot(ax, x, y, code: str, hero: bool = False, label: str | None = None, **kw):
    """One series in its paper style (edge-certified arms look like their map's arm; banners name
    the gate).
    """
    st = style(code, hero, label)
    st.update(kw)
    return ax.plot(x, y, **st)


def band(ax, x, lo, hi, code: str, hero: bool = False, **kw):
    """95 % CI band in the series hue (DESIGN s.7; [2026-10-06] 0.14 hero / 0.08 others, so up to
    three overlapping bands stay light instead of merging into one dark mass).
    """
    return ax.fill_between(
        x,
        lo,
        hi,
        color=fill_color(code),
        alpha=BAND_ALPHA[hero],
        lw=0,
        zorder=2,
        gid=f"band:{resolve(code).key}",
        **kw,
    )


def invalid(ax, xs, code: str, **kw) -> None:
    """INVALID crosses (no recall), always ink, matching the legend entry."""
    L.invalid_marks(ax, xs, P.INK, **kw)


def bar(ax, x, h, code: str, width: float, hero: bool = False, label: str | None = None, **kw):
    """Bars (DESIGN s.7, [DECIDED 2026-10-08, user]): every bar a full-strength fill in its method
    colour, no edge. Edge-certified arms (``CS:<map>``) add a light white ``////`` hatch.
    """
    arm = resolve(code)
    gid = _gid(arm, hero)
    cs = arm.gate == "CS"
    bars = ax.bar(
        x,
        h,
        width,
        color=fill_color(code),
        edgecolor=P.WHITE if cs else "none",
        linewidth=0,
        label=label,
        zorder=3,
        gid=gid,
        **kw,
    )
    if cs:
        for b in bars:
            b.set_hatch("////")
            b.set_hatch_linewidth(0.4)
    return bars


def row_point(
    ax,
    x,
    y,
    ci: Sequence[float] | None,
    code: str,
    hero: bool = False,
    ink: bool = False,
    fill: bool | None = None,
    **kw,
):
    """One labelled forest row: the estimate in the method's glyph with an ink CI whisker.
    ``ink=True`` keeps the glyph's shape and fill but draws it in ink (fig_study4_validity (a):
    family banners carry the family there). ``fill`` [G5 2026-10-12] overrides the fill state when
    fill encodes something other than the hero (the merged decision figure: open = Study 2 without
    the gate, filled = Study 3 edge-certified); None keeps the default.
    """
    arm = resolve(code)
    st = style(code, hero)
    mc = P.INK if ink else marker_color(code)
    filled = (hero or arm.method.filled) if fill is None else fill
    face = mc if filled else P.WHITE
    if arm.method.marker in ("+", "|"):
        face = mc
    tick = (
        arm.method.marker == "|"
    )  # [G3 2026-10-12] static rules: an unbroken interval with a taller estimate tick
    ms, mew = (st["ms"] + 2.4, 1.0) if tick else (st["ms"], 0.4 if hero else 0.8)
    if (
        ci is not None
    ):  # whisker below the glyph, broken at the glyph by a white disc (a CI narrower
        ax.errorbar(
            [x],
            [y],
            xerr=[[x - ci[0]], [ci[1] - x]],
            fmt="none",
            ecolor=P.INK,
            elinewidth=0.6,
            capsize=1.5,
            capthick=0.6,
            zorder=3,
            clip_on=False,
        )  # than the marker would read as an "A" otherwise)
        if not tick:  # a "|" tick has no inside to clear: a halo would cut a short interval into
            ax.plot(
                [x],
                [y],
                ls="",
                marker="o",
                ms=st["ms"] + 1.6,
                mfc=P.WHITE,
                mec="none",
                zorder=3.5,  # dots
                clip_on=False,
                gid="halo",
            )
    if fill is not None and not tick:
        mew = 0.4 if (hero and filled) else 0.8
    ax.plot(
        [x],
        [y],
        ls="",
        marker=st["marker"],
        color=mc,
        ms=ms,
        mfc=face,
        mec=P.WHITE if (hero and filled) else mc,
        mew=mew,
        zorder=4,
        clip_on=False,
        gid=f"row:{arm.key}" + (":hero" if hero else ""),
        **kw,
    )  # clip_on=False: a rate of 1 sits on the axis edge


def handle(code: str, hero: bool = False) -> Line2D:
    """Legend proxy for a series (line + marker in its paper style)."""
    st = style(code, hero)
    st.pop("gid")
    st.pop("zorder")
    return Line2D([], [], **st)


def legend(fig, codes: Sequence[str], hero: str | None = None, where: str = "auto", **kw):
    """Figure legend in family order with proxy handles (DESIGN s.6): the hero's entry is filled.
    ``auto`` (default): a right column when every panel keeps its minimum width, else a centred top
    row (layout.finalize decides).
    """
    codes = order(codes)
    return L.fig_legend(
        fig, [handle(c, c == hero) for c in codes], [name(c) for c in codes], where=where, **kw
    )


def reference_line(ax, x, y, **kw):
    """A reference series (DESIGN s.4 / s.5.4): ink at 60 %, dash (4, 2), LW_REF, no marker,
    labelled directly. The frozen PMRT next to the Study 4 PMRT arms is drawn this way, never in the
    PMRT-GBM family's blue.
    """
    return ax.plot(
        x,
        y,
        color=P.INK,
        alpha=0.6,
        lw=S.LW_REF,
        ls=REF_DASH,
        marker="",
        zorder=2,
        gid="reference",
        **kw,
    )


# --- rule checks (tests/test_methods.py runs them on every registered figure) ---------------------------------------------
BANNED_WORDS = ("proposed", "best")  # Study 4 hero is emphasis only (DESIGN s.12)


def tagged(fig) -> list[tuple[str, str, bool, object]]:
    """(kind, key, hero, artist) for every artist styled here: kind 'method' (series / bar), 'row'
    (labelled row).
    """
    out = []
    for art in fig.findobj(
        lambda a: isinstance(a.get_gid(), str) and a.get_gid().split(":")[0] in ("method", "row")
    ):
        parts = art.get_gid().split(":")
        out.append((parts[0], parts[1], "hero" in parts[2:], art))
    return out


def _glyph(key: str) -> tuple:
    m = METHODS[key]
    return (m.color or FAMILIES[m.family].color, m.marker, str(FAMILIES[m.family].ls))


def audit(fig) -> dict[str, list[str]]:
    """Method-mapping rules on one figure:
    one hero        - exactly one hero method (filled, heavy) when the figure shows any method;
    variants        - two methods with the same hue, marker and line style only as labelled rows (VARIANTS);
    frozen PMRT     - the frozen PMRT never appears as a blue series next to the Study 4 PMRT arms (ink reference);
    wording         - no legend entry says "proposed" / "best".
    """
    t = tagged(fig)
    keys = {k for _, k, _, _ in t}
    out = {"one hero": [], "variants": [], "frozen PMRT": [], "wording": []}
    heroes = {k for _, k, h, _ in t if h}
    if keys and len(heroes) != 1:
        out["one hero"].append(
            f"{len(heroes)} hero methods ({sorted(heroes)}); exactly one per figure"
        )
    lines = sorted({k for kind, k, _, _ in t if kind == "method"})
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            if _glyph(a) == _glyph(b):
                out["variants"].append(
                    f"'{METHODS[a].name}' and '{METHODS[b].name}' share hue + marker + line style "
                    "outside a labelled-row figure"
                )
    if "pmrt" in keys and keys & {"pmrt_gbm", "pmrt_lin", "pmrt_lin_r3"}:
        out["frozen PMRT"].append(
            "frozen PMRT drawn as a blue series next to PMRT-GBM / PMRT-Lin (use reference_line)"
        )
    for leg in fig.findobj(Legend):
        for txt in leg.get_texts():
            if any(w in txt.get_text().lower().split() for w in BANNED_WORDS):
                out["wording"].append(f"legend entry '{txt.get_text()}'")
    return out

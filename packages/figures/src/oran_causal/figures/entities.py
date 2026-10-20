"""Paper-wide xApp and KPI styling (DESIGN s.12.2; slots per panel). The single source of
how an xApp, the control parameter it writes, or a KPI looks; figures pass code keys (``ES``, ``PowerES``,
``SliceGuarantee``, ``Coverage``; ``sleep`` …; ``pv`` …) and get the paper names on the figure.

Rules (each checked by ``audit`` on every registered figure, tests/test_entities.py):
  * entity in method figure - xApp / parameter / KPI hues never share a figure with method series (oran_causal.figures.methods);
    any entity hue there is a violation, except inside a node-role graph panel (axes gid ``node-roles``).
  * one entity type        - xApps (with their parameters), KPIs and graph node roles never share a figure.
  * e and rlf              - energy (purple) and RLF (blue) are ΔE 5.5 under protan with equal lightness: when both are in
    one plot, each needs its marker and a direct label.
  * ink arrows             - edges and arrows are always ink, never a writer's hue.
  * code names             - no figure text uses the code names ``PowerES`` / ``SliceGuarantee``.
Artists carry gids ``xapp:<key>``, ``param:<param>:<writer>``, ``kpi:<key>``, ``kpi-label:<key>``, ``kpi-node``.
"""

from dataclasses import dataclass

from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from oran_causal.figures import diagram as D
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import style as S

LW_XAPP = 1.0  # xApp box / parameter node outline (DESIGN s.12.2)
COVERAGE_LS = (
    0,
    (3, 1.6),
)  # Coverage: neutral gray, dashed (not in P3; shown only where its ptx write matters)


@dataclass(frozen=True)
class Entity:
    name: str  # paper label
    slot: int | None  # DESIGN s.5.2 entity slot 1-6; None = neutral gray
    color: str
    marker: str | None


def _slot(name: str, k: int) -> Entity:
    return Entity(name, k, P.ENTITIES[k - 1], P.ENTITY_MARKERS[k - 1])


XAPPS = {
    "ES": _slot("ES", 2),  # terracotta
    "SliceGuarantee": _slot("Slice Guarantee", 3),  # teal
    "PowerES": _slot("Power ES", 4),  # gold
    "Coverage": Entity("Coverage", None, P.NEUTRAL, None),  # gray, dashed; not in P3 or any study
}
ALIASES = {"ESPico": "ES"}
WRITER = {
    "carrier": "ES",
    "sleep": "ES",
    "ptx": "PowerES",
    "prot_min": "SliceGuarantee",
}  # one writer each in P3

PARAM_PLAIN = {
    "carrier": "macro carrier",
    "sleep": "pico sleep",
    "ptx": "transmit power",
    "prot_min": "protected share",
}  # plain names for the ESM figures (no code names in print)


def param_plain_label(param: str) -> str:
    """Plain-name label of a control-parameter family with its writer, e.g. 'pico sleep (ES)'."""
    return f"{PARAM_PLAIN[param]} ({XAPPS[WRITER[param]].name})"


# PROVISIONAL (fixed at first use). Violation KPIs (v, pv, rlf) never take terracotta
# (red-adjacent); green is unused (reads as "good"); no KPI hue means good or bad.
KPIS = {
    "rlf": _slot("radio-link failures", 1),  # blue
    "load": _slot("served load", 2),  # terracotta
    "pv": _slot("protected-slice violations", 3),  # teal
    "v": _slot("service violations", 4),  # gold
    "e": _slot("energy", 5),  # purple
}
PROVISIONAL = True
WEAK_PAIRS = [{"e", "rlf"}]  # need markers + direct labels when they share a plot
CODE_NAMES = ("PowerES", "SliceGuarantee")

# Planned figures that use entity colours (DESIGN s.12.2); every other planned figure uses none.
FIGURE_USE = {
    "fig_intro_conflict": "xapps",
    "fig_system": "xapps",
    "fig_study4_env_e1": "node-roles",
    "fig_study4_env_e2": "node-roles",
    "fig_study4_env_e3": "node-roles",
    "fig_study4_env_e4": "node-roles",
    "fig_study4_env_e5": "node-roles",
}


def param_label(param: str) -> str:
    """Axis / node label of a control-parameter family with its writer, e.g. 'sleep (ES)'."""
    return f"{param} ({XAPPS[WRITER[param]].name})"


def xapp_box(ax, x, y, w, h, key: str, sub: str | None = None, **kw):
    """Diagram box of an xApp (DESIGN s.9 / s.12.2): 10 % tint, 1.0 pt outline in its hue, ink
    label. Coverage is a neutral box with a dashed gray outline.
    """
    k = ALIASES.get(key, key)
    e = XAPPS[k]
    if e.slot is None:
        return D.box(
            ax, x, y, w, h, e.name, sub, role="neutral", ls=COVERAGE_LS, gid=f"xapp:{k}", **kw
        )
    return D.box(
        ax, x, y, w, h, e.name, sub, role="entity", color=e.color, lw=LW_XAPP, gid=f"xapp:{k}", **kw
    )


def param_node(ax, x, y, param: str, r: float = 6.3, label: str | None = None):
    """Control-parameter node (circle, s.8 shape) in its writer's hue: 16 % tint fill and 1.0 pt
    outline only. Edges into / out of it stay ink.
    """
    w = WRITER[param]
    c = XAPPS[w].color
    ax.add_patch(
        Circle((x, y), r, fc=P.tint(c, 0.16), ec=c, lw=LW_XAPP, zorder=3, gid=f"param:{param}:{w}")
    )
    ax.text(x, y, label or param, ha="center", va="center", fontsize=S.ANNOT_PT, zorder=4)


def kpi_node(
    ax,
    x,
    y,
    label: str,
    r: float = 6.3,
    w: float | None = None,
    h: float | None = None,
    lw: float = 0.7,
):
    """KPI node in an xApp diagram: neutral (white fill, ink 0.7 pt outline, s.8 rounded square, or
    a rounded rectangle w x h for word labels), never a KPI hue. ``lw`` 1.4 marks the KPI where a
    conflict converges (fig_intro_conflict only, DESIGN s.8).
    """
    w, h = w or 2 * r, h or 2 * r
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={min(w, h) * 0.17}",
            fc=P.WHITE,
            ec=P.INK,
            lw=lw,
            zorder=3,
            gid="kpi-node",
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=S.ANNOT_PT, zorder=4)


def plot_kpi(ax, x, y, key: str, direct_label: bool = True, **kw):
    """One KPI series (entity style, s.5.2: filled marker, 1.3 pt) with a direct end label (rule "e
    and rlf").
    """
    e = KPIS[key]
    st = {
        "color": e.color,
        "marker": e.marker,
        "lw": S.LW_ENTITY,
        "mfc": e.color,
        "mec": P.WHITE,
        "mew": 0.5,
        "ms": S.MARKER_SIZE,
        "zorder": 3,
        "gid": f"kpi:{key}",
    }
    st.update(kw)
    lines = ax.plot(x, y, **st)
    if direct_label:
        ax.annotate(
            e.name,
            (x[-1], y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=S.ANNOT_PT,
            color=P.INK,
            gid=f"kpi-label:{key}",
        )
    return lines


def plot_xapp(ax, x, y, key: str, **kw):
    """One series per xApp (entity style; Coverage gray dashed, no marker)."""
    k = ALIASES.get(key, key)
    e = XAPPS[k]
    st = {"color": e.color, "lw": S.LW_ENTITY, "zorder": 3, "gid": f"xapp:{k}"}
    st.update(
        {"ls": COVERAGE_LS, "marker": ""}
        if e.slot is None
        else {"marker": e.marker, "mfc": e.color, "mec": P.WHITE, "mew": 0.5, "ms": S.MARKER_SIZE}
    )
    st.update(kw)
    return ax.plot(x, y, **st)


# --- rule checks ------------------------------------------------------------------------------------------------------
RULES = ["entity in method figure", "one entity type", "e and rlf", "ink arrows", "code names"]


def _rgb(c) -> tuple:
    return tuple(round(v, 3) for v in to_rgb(c))


ENTITY_RGB = {_rgb(P.tint(h, a)) for h in P.ENTITIES for a in (1.0, 0.10, 0.16, 0.28)}


def _gids(fig) -> list[tuple[str, object]]:
    return [(a.get_gid(), a) for a in fig.findobj(lambda a: isinstance(a.get_gid(), str))]


def _colours(art) -> list:
    """Colours the artist actually draws (a Line2D's line colour only with a line, its marker face
    only for a filled marker).
    """
    if isinstance(art, Line2D):
        out = []
        if art.get_linestyle() not in ("None", "", " ") and art.get_linewidth() > 0:
            out.append(art.get_color())
        m = art.get_marker()
        if m not in (None, "", "None", " "):
            out.append(art.get_markeredgecolor())
            if MarkerStyle(m).is_filled() and art.get_fillstyle() != "none":
                out.append(art.get_markerfacecolor())
        return out
    out = []
    for get in (
        "get_color",
        "get_markerfacecolor",
        "get_markeredgecolor",
        "get_facecolor",
        "get_edgecolor",
    ):
        f = getattr(art, get, None)
        if f is None:
            continue
        try:
            v = f()
        except AttributeError, TypeError:
            continue
        if isinstance(v, str) or (
            hasattr(v, "__len__") and len(v) in (3, 4) and not hasattr(v[0], "__len__")
        ):
            out.append(v)
        elif hasattr(v, "__len__"):
            out.extend(list(v))
    return out


def _entity_hued(fig) -> list[str]:
    """Methods and entities share the palette hues, so this is tag-based: untagged artists (no gid) drawn in a palette hue or its tint, outside node-role
    panels and colour bars = a series not styled through oran_causal.figures.methods (e.g. the
    default colour cycle).
    """
    hits = []
    for ax in fig.axes:
        if ax.get_gid() == "node-roles" or hasattr(ax, "_colorbar"):
            continue
        for art in ax.findobj():
            if art.get_gid() is not None or art is ax.patch:
                continue
            for c in _colours(art):
                try:
                    if isinstance(c, str) and c.lower() in ("none", ""):
                        continue
                    if to_rgba(c)[3] > 0 and _rgb(c) in ENTITY_RGB:
                        hits.append(f"untagged {type(art).__name__} in palette colour {c}")
                        break
                except ValueError:
                    continue
    return hits


def audit(fig) -> dict[str, list[str]]:
    out = {r: [] for r in RULES}
    g = _gids(fig)
    kinds = {gid.split(":")[0] for gid, _ in g}
    has_xapp = bool(kinds & {"xapp", "param"})
    has_kpi = "kpi" in kinds
    has_roles = any(ax.get_gid() == "node-roles" for ax in fig.axes)
    if M.tagged(fig):
        if has_xapp or has_kpi:
            out["entity in method figure"].append(
                "xApp / parameter / KPI artists in a figure with method series"
            )
        out["entity in method figure"] += _entity_hued(fig)[:5]
    types = [
        n for n, on in (("xApps", has_xapp), ("KPIs", has_kpi), ("node roles", has_roles)) if on
    ]
    if len(types) > 1:
        out["one entity type"].append(f"{' + '.join(types)} share the figure's colour encoding")
    for ax in fig.axes:
        lines = {
            gid.split(":")[1]: a
            for gid, a in g
            if gid.startswith("kpi:") and getattr(a, "axes", None) is ax
        }
        labels = {
            gid.split(":")[1]
            for gid, a in g
            if gid.startswith("kpi-label:") and getattr(a, "axes", None) is ax
        }
        for pair in WEAK_PAIRS:
            if pair <= set(lines):
                for k in sorted(pair):
                    if lines[k].get_marker() in (None, "", "None", " ") or k not in labels:
                        out["e and rlf"].append(
                            f"{KPIS[k].name}: needs its marker and a direct label next to "
                            f"{KPIS[(pair - {k}).pop()].name}"
                        )
    if has_xapp:
        for a in fig.findobj(lambda a: isinstance(a, FancyArrowPatch) or a.get_gid() == "arrow"):
            c = a.get_edgecolor() if isinstance(a, FancyArrowPatch) else a.get_color()
            if _rgb(c) != _rgb(P.INK):
                out["ink arrows"].append(f"arrow / edge in {c}, not ink")
    for t in fig.findobj(lambda a: hasattr(a, "get_text") and callable(a.get_text)):
        s = t.get_text() if isinstance(t.get_text(), str) else ""
        if any(n in s for n in CODE_NAMES):
            out["code names"].append(
                f"text '{s}' uses a code name (paper names: Power ES, Slice Guarantee)"
            )
    return out

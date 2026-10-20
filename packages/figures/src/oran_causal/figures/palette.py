"""Harbor palette: the single source of color for every journal figure.

Harbor's six entity hues are also the method palette: one fixed colour per method for the whole paper
(methods.py), plus three Harbor tints for PMRT-Lin, QACM and INT.
- Methods and entities (xApps, KPIs, NCPs) share these hues but never share a figure (DESIGN s.12.2).
- Heatmaps: SEQ for magnitudes, DIV for signed values (rates: centred at the nominal level).
- The six hues and three tints reach at least 3:1 on white (DESIGN s.2); panels sit on the FIELD gray.
No colour may be defined anywhere else.
"""

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb

# --- the six Harbor hues: entities (peers, slots 1-6) and methods -------------------------------------------------
ENTITIES = [
    "#3767A8",
    "#C2553A",
    "#1A7F8E",
    "#9A7414",
    "#8B4F9F",
    "#55803A",
]  # blue terracotta teal gold purple green
ENTITY_MARKERS = ["o", "s", "^", "D", "v", "P"]
BLUE, TERRACOTTA, TEAL, GOLD, PURPLE, GREEN = ENTITIES

# --- methods ----------------------------------------------------------------------------------------------------
HERO = BLUE  # PMRT / PMRT-GBM, and the hero component of a diagram
PMRT_LIN = "#638DC8"  # lighter BLUE: PMRT-Lin (open marker) and PMRT-Lin, R3 covariates
SLATE = "#6F8DB8"  # muted blue: QACM (Studies 1-3)
LILAC = "#A07AB2"  # lighter PURPLE: INT (Studies 1-3)
METHODS = [*ENTITIES, PMRT_LIN, SLATE, LILAC]

# --- neutrals and special roles ----------------------------------------------------------------------------------
INK = "#262B33"  # all text, CI whiskers, correct graph edges
AXIS = "#5A606B"  # tick labels and the few remaining axis lines
GRID = "#FFFFFF"  # white grid on the FIELD panels
NEUTRAL = "#707782"  # neutral diagram outlines; methods without a slot
NEUTRAL_LIGHT = "#9AA0AA"  # hairline outlines
MISSED = "#8A909C"  # missed (false-negative) graph edges
SPURIOUS = TERRACOTTA  # spurious (false-positive) graph edges
WHITE = "#FFFFFF"
FIELD = "#EFF1F4"  # panel background (facet field) and null map tiles
LEGEND_FC, LEGEND_EC = "#F6F7F9", "#D5D9DF"  # legend box

# --- semantic roles ---------------------------------------------------------------------------------------------
# Colour carries meaning paper-wide. Draw semantic artists with gid "sem:<role>" (e.g. "sem:good-zone").
OURS = BLUE  # our method and our components
GOOD = TEAL  # valid, safe, correct, right sign
BAD = TERRACOTTA  # invalid, unsafe, wrong sign, false edge
DESIGN_ROLE = (
    GOLD  # the logged randomisation / logging design (the ingredient that makes the test valid)
)
CONFOUNDER = PURPLE  # network state, confounding paths
ZONE_TINT = (
    0.10  # zone fills (bordered by a full-strength limit line); tiles and fills use 0.28-1.0
)
# --- continuous --------------------------------------------------------------------------------------------------
SEQ = [
    "#F2F7FF",
    "#C4DCFE",
    "#97BFFB",
    "#74A3E9",
    "#5787D0",
    "#406DAF",
    "#315488",
    "#263D5F",
    "#0E2648",
]
DIV = ["#0C3164", "#3963A1", "#A4B9D8", "#F2F2F2", "#DBB4A6", "#AE593A", "#67260C"]
SEQ_CMAP = LinearSegmentedColormap.from_list("harbor_seq", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "harbor_div", DIV
)  # negative = navy, zero = gray, positive = rust
ENTITY_CMAP = ListedColormap(ENTITIES, name="harbor_entities")

NEUTRALS = [
    INK,
    AXIS,
    GRID,
    NEUTRAL,
    NEUTRAL_LIGHT,
    MISSED,
    SPURIOUS,
    WHITE,
    FIELD,
    LEGEND_FC,
    LEGEND_EC,
    "#000000",
]
PALETTE = [*METHODS, *ENTITIES]


def allowed(color, tol: float = 0.006) -> bool:
    """True if ``color`` is a palette colour or a neutral, or a tint of one (mixed toward white).
    Alpha is ignored.
    """
    c = to_rgb(color)
    for base in (*PALETTE, *NEUTRALS):
        b = to_rgb(base)
        amounts = [(1 - ci) / (1 - bi) for ci, bi in zip(c, b, strict=False) if 1 - bi > 0.02]
        if not amounts:
            if all(abs(ci - bi) <= tol for ci, bi in zip(c, b, strict=False)):
                return True
            continue
        a = sum(amounts) / len(amounts)
        if 0 < a <= 1 + tol and all(
            abs((1 - (1 - bi) * a) - ci) <= tol for ci, bi in zip(c, b, strict=False)
        ):
            return True
    return False


def luminance(color) -> float:
    """WCAG 2 relative luminance of ``color`` (alpha ignored)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b) -> float:
    """WCAG 2 contrast ratio of two colours (1 to 21)."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def method(k: int) -> str:
    """Colour for method slot k (0 = hero blue, 1..5 = the other Harbor hues, 6..8 = the three
    tints). Specimens only: real methods are styled by arm code in methods.py.
    """
    return METHODS[k]


def entity(k: int) -> str:
    """Colour for entity slot k (0..5). A 7th entity must fold into 'Other' or a new panel."""
    if not 0 <= k < len(ENTITIES):
        raise IndexError(
            f"entity slot {k} out of range 0..{len(ENTITIES) - 1}; fold extras into 'Other'"
        )
    return ENTITIES[k]


def tint(color: str, amount: float) -> tuple[float, float, float]:
    """Mix `color` toward white; amount = 1 keeps the color, 0 gives white."""
    r, g, b = to_rgb(color)
    return (1 - (1 - r) * amount, 1 - (1 - g) * amount, 1 - (1 - b) * amount)


# --- edge-status fills (bars and column keys in the discovery figures) ----------------------------------------
NULL_FILL = tint(BAD, 0.25)  # a declared edge on a true null
UNSCORED_FILL = tint(NEUTRAL, 0.45)  # not scored

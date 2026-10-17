"""Global matplotlib + seaborn style. Call `apply()` once, before creating figures.

seaborn's `set_theme` is applied first (context/style), then our rcParams override
it. Never call `sns.set_theme` / `sns.set_style` anywhere else: it silently resets
the rcParams below.
"""

import functools
from pathlib import Path

import matplotlib as mpl
import seaborn as sns
from cycler import cycler
from matplotlib import font_manager

from oran_causal.figures import palette as P

# [G5 2026-10-12, user decision] Journal artwork rule (Springer: Helvetica or Arial, 8-12 pt):
# Arial, from the system's font files (not vendored: Arial may not be redistributed; its fsType 8
# permits embedding in the PDF).
FONT_FAMILY = "Arial"
ARIAL_FILES = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")
ARIAL_DIRS = (
    Path("C:/Windows/Fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/usr/share/fonts/truetype/msttcorefonts"),
    Path.home() / ".fonts",
)

# Page geometry: Springer sn-jnl single-column text block is 31 pc = 5.15 in.
FULL = 5.15  # \includegraphics[width=\textwidth]
HALF = 2.5  # \includegraphics[width=0.49\textwidth]
ASPECT = 0.55  # default height / width
MAX_HEIGHT = 4.5

# Type scale (pt) [G5 2026-10-12, user decision: journal rule 8-12 pt] floor raised to 8 pt; sizes 8
# / 9 / 10 keep the variance small
TICK_PT = 8.0  # was 7.5
LABEL_PT = 9.0  # was 8.5
LEGEND_PT = 8.0  # was 7.5
PANEL_PT = 10.0  # was 9.5
MIN_PT = 8.0  # floor for any figure text at print size (figures are built at print width); was 7.0
ANNOT_PT = MIN_PT  # was 7.0 (6.5 before that)

# Line weights (pt)
LW_PROPOSED = 1.9
LW_BASELINE = 1.1
LW_ENTITY = 1.3
LW_AXIS = 0.6
LW_GRID = 0.4
LW_REF = 0.7  # threshold / target lines
MARKER_SIZE = 3.6
SPINE_OFFSET = 3  # pt, see finish()

RC = {
    # output
    "figure.figsize": (FULL, FULL * ASPECT),
    "figure.dpi": 150,
    "figure.facecolor": P.WHITE,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.facecolor": P.WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # type
    "font.family": FONT_FAMILY,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "font.size": 8,
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": LABEL_PT,
    "legend.fontsize": LEGEND_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "axes.unicode_minus": True,
    # color
    "text.color": P.INK,
    "axes.labelcolor": P.INK,
    "axes.edgecolor": P.AXIS,
    "axes.facecolor": P.WHITE,
    "xtick.color": P.AXIS,
    "ytick.color": P.AXIS,
    "xtick.labelcolor": P.INK,
    "ytick.labelcolor": P.INK,
    "axes.prop_cycle": cycler(color=P.ENTITIES) + cycler(marker=P.ENTITY_MARKERS),
    "image.cmap": "harbor_seq",
    # axes
    "axes.linewidth": LW_AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.titlelocation": "left",
    "axes.labelpad": 3,
    "grid.color": P.GRID,
    "grid.linewidth": LW_GRID,
    "grid.linestyle": "-",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": LW_AXIS,
    "ytick.major.width": LW_AXIS,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    # marks
    "lines.linewidth": LW_ENTITY,
    "lines.markersize": MARKER_SIZE,
    "lines.markeredgecolor": P.WHITE,
    "lines.markeredgewidth": 0.5,
    "lines.solid_joinstyle": "round",
    "patch.edgecolor": "none",
    "patch.linewidth": 0.7,
    "errorbar.capsize": 2,
    "boxplot.flierprops.marker": "",
    # legend
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0,
    "legend.labelspacing": 0.35,
}


def arial_files() -> list[Path]:
    """The four Arial TTFs (regular, bold, italic, bold italic) from the first system font directory
    holding them.
    """
    for d in ARIAL_DIRS:
        fs = [d / f for f in ARIAL_FILES]
        if all(f.exists() for f in fs):
            return fs
    raise RuntimeError(
        f"Arial not found in {[str(d) for d in ARIAL_DIRS]}: install Arial (journal artwork rule)"
    )


def register_fonts() -> None:
    for ttf in arial_files():
        font_manager.fontManager.addfont(str(ttf))
    names = {f.name for f in font_manager.fontManager.ttflist}
    if FONT_FAMILY not in names:
        raise RuntimeError(f"{FONT_FAMILY} not registered")


def register_cmaps() -> None:
    for cmap in (P.SEQ_CMAP, P.DIV_CMAP, P.ENTITY_CMAP):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)


@functools.cache
def apply() -> None:
    """Apply the journal style globally. Idempotent."""
    register_fonts()
    register_cmaps()
    sns.set_theme(context="paper", style="ticks", palette=P.ENTITIES, font=FONT_FAMILY, rc=RC)
    mpl.rcParams.update(RC)  # second pass: our values win over anything seaborn's context scaled

"""Private helpers of fig_study1_discovery, fig_study2_maps and fig_study2_deferrals (Studies 1-2 on
the system-level simulator): number formats, horizontal method bars through
``oran_causal.figures.methods.bar`` and their value labels, and the row-wise layout (family banners,
bold method names, sub-row labels) shared by the row-wise panels.

Arm codes are resolved with Studies 1-3 semantics (``methods.resolve(code, study13=True)``:
``granger`` is the DEV-tuned Granger), and the resolved method key is what is passed on to
``oran_causal.figures.methods``.
"""

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import style as S

THIN = " "  # thin space as the thousands separator ("8 674")
KNOCKOUT = {
    "boxstyle": "square,pad=0.08",
    "fc": P.WHITE,
    "ec": "none",
}  # value labels sit on white over grid / ref lines


def key13(code: str) -> str:
    """Method key of a Studies 1-3 code (``granger`` -> ``granger_dev``; the 9 PMRT statistic x
    layer codes -> pmrt); referee map codes (``MG:<map>[@dev|@plc]``) keep their gate and threshold.
    """
    arm = M.resolve(code, study13=True)
    if arm.gate:
        return code  # MG:/CS: codes already resolve with Studies 1-3 semantics in oran_causal.figures.methods
    return arm.key


def name13(code: str) -> str:
    return M.resolve(code, study13=True).method.name


def fmt_count(v: int) -> str:
    s = f"{int(v):,}"
    return s.replace(",", THIN)


def fmt_score(v: float) -> str:
    """Two decimals without the leading zero (DESIGN cell style): .95, 1.00, 0 for exactly 0."""
    if v == 0:
        return "0"
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0") else s


def hbar(
    ax,
    y: float,
    length: float,
    code: str,
    thickness: float,
    hero: bool = False,
    label: str | None = None,
):
    """One horizontal method bar (methods.bar, orientation horizontal) from 0; a zero-length bar is
    not drawn (its value label carries the 0).
    """
    if not length:
        return None
    return M.bar(
        ax,
        0,
        thickness,
        code,
        width=length,
        hero=hero,
        label=label,
        bottom=y,
        orientation="horizontal",
    )


def vbar(
    ax,
    x: float,
    height: float,
    code: str,
    width: float,
    hero: bool = False,
    label: str | None = None,
):
    """One vertical method bar; zero height is not drawn (its value label carries the 0)."""
    if not height:
        return None
    return M.bar(ax, x, height, code, width=width, hero=hero, label=label)


def value_right(ax, x: float, y: float, text: str):
    """Value label just right of a horizontal bar end (data-anchored, DESIGN s.4)."""
    return L.label_line(ax, x, y, text, where="right", fontsize=S.ANNOT_PT, bbox=KNOCKOUT)


def value_above(ax, x: float, y: float, text: str):
    """Value label just above a vertical bar top, centred (data-anchored, DESIGN s.4)."""
    return L.label_line(
        ax, x, y, text, where="above", ha="center", fontsize=S.ANNOT_PT, bbox=KNOCKOUT
    )


def row_axes(ax, ylim, ticks, labels, xlabel: str | None = None) -> None:
    """A panel of a row-wise figure: rows on y (labels as tick labels, no tick marks, no left
    spine), vertical grid only, transparent background so family banners run underneath.
    """
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks, labels, fontsize=S.ANNOT_PT)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.patch.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)


def single_labels(ax, groups, pad_pt: float) -> None:
    """Bold method name of a one-row method, right-aligned at the tick-label edge (gid ``row-label``
    so the panel letter clears it).
    """
    for name, ys in groups:
        ax.annotate(
            name,
            xy=(0, ys[0]),
            xycoords=("axes fraction", "data"),
            xytext=(-pad_pt, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=S.ANNOT_PT,
            fontweight="bold",
            color=P.INK,
            gid="row-label",
        )


def tick_pad_pt(ax) -> float:
    """Distance (pt) from the axes' left edge to the right edge of its y tick labels (measured after
    a draw).
    """
    fig = ax.figure
    r = fig.canvas.get_renderer()
    x0 = ax.get_window_extent(r).x0
    right = [
        t.get_window_extent(r).x1 for t in ax.get_yticklabels() if t.get_text() and t.get_visible()
    ]
    return (x0 - max(right)) * 72 / fig.dpi if right else 3.5

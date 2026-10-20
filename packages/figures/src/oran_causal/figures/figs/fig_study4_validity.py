"""F3: Study 4 validity at a glance (C1-C3). Two panels share one row per method and regime: (a) the
pooled raw null-edge rejection rate (p <= .05) with its 95 % cluster-bootstrap CI, from the verdict
table (report.json V4_verdicts: design-blind methods C1 R1 / R2; design-adjusted tests eq_arms_table
R1 / R2 and C3_eq_arms R3; PMRT arms C2a R1 + R2 pooled, no per-regime split there, and C3 R3); (b)
the same rate in each environment E1-E5 (validity_cells.csv: the mean of the environment's n cells,
which is its pooled rate since every cell has the same edges x 40 seeds; no CI), printed in cells
shaded by the rate (harbor_div centred at the nominal .05, two-slope norm
0-1, labelled colour bar). Null edges are truth-null edges; E4 has no truth-null candidate in R1 /
R2 (REPORT V0), so its R1 / R2 rates are the placebo edge, and its R3 rate is the confounded null
edge (C3; R3 exists in E4 only, all four lam). Rates use the shared symlog rate axis (0 drawn at 0,
no floor). Rows are labelled directly (method, regime), so no legend. Method separation (DESIGN
s.7): an extra gap between methods, the bold method name on the method's
first regime row (layout.method_rows / method_labels). Points (DESIGN s.12): the
method's glyph from oran_causal.figures.methods drawn in ink (the family banners carry the family;
the light method colours would make some of the worst failures the faintest marks); PMRT-GBM is the
one filled (hero) glyph.
"""

from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.data import load_json
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study4_validity",
    width="full",
    paper=True,
    sources=[T.SRC_REPORT, T.SRC_CELLS],
    caption_hint="Null-edge rejection rate per method and regime, pooled over E1-E5 with 95 % cluster-bootstrap "
    "CIs (Study 4, kappa .25); nominal rate .05. Marker shape = base "
    "method (plain and design-adjusted versions share it); CIs narrower than the marker are hidden by it",
)

REG_LABEL = {"R1": "R1", "R2": "R2", "R1+R2": "R1 + R2", "R3": "R3"}
REG_ORDER = ["R1", "R2", "R1+R2", "R3"]
# (row label, None for a category banner | {regime: (verdict-table path..., rate key)})
ROWS = [
    ("Design-based tests", None),
    ("PMRT-GBM", {"R1+R2": ("C2a", "pooled", "null_raw"), "R3": ("C3", "pooled", "conf_raw")}),
    (
        "PMRT-Lin",
        {
            "R1+R2": ("pmrt_secondary", "pmrt_eq", "C2a", "pooled", "null_raw"),
            "R3": ("pmrt_secondary", "pmrt_eq", "C3", "pooled", "conf_raw"),
        },
    ),
    ("Design-adjusted tests", None),
    (
        "Partial correlation",
        {
            "R1": ("eq_arms_table", "pcorr_eq", "pooled_r1", "null_raw"),
            "R2": ("eq_arms_table", "pcorr_eq", "pooled_r2", "null_raw"),
            "R3": ("C3_eq_arms", "pcorr_eq", "pooled", "conf_raw"),
        },
    ),
    (
        "Granger",
        {
            "R1": ("eq_arms_table", "granger_eq", "pooled_r1", "null_raw"),
            "R2": ("eq_arms_table", "granger_eq", "pooled_r2", "null_raw"),
        },
    ),
    (
        "RCoT",
        {
            "R1": ("eq_arms_table", "rcot2_eq", "pooled_r1", "null_raw"),
            "R2": ("eq_arms_table", "rcot2_eq", "pooled_r2", "null_raw"),
            "R3": ("C3_eq_arms", "rcot2_eq", "pooled", "conf_raw"),
        },
    ),
    ("Design-blind methods", None),
] + [
    (
        name,
        {
            "R1": ("C1", "arms", arm, "pooled_r1", "null_raw"),
            "R2": ("C1", "arms", arm, "pooled_r2", "null_raw"),
        },
    )
    for name, arm in (
        ("Correlation", "corr"),
        ("Partial correlation", "pcorr_native"),
        ("Partial correlation,\nHAC (fixed-b)", "pcorr_hac_fb"),
        ("Granger", "granger_native"),
        ("RCoT", "rcot2_native"),
        ("MSCR (n ≤ 1000)", "mscr_native"),
    )
]
PITCH, BANNER_H = L.ROW_PITCH, L.BANNER_H  # data units on the shared row axis
# rate heatmaps: diverging harbor_div centred at the nominal .05 (blue
# below, near-white at .05, rust above), two-slope linear norm on 0-1, labelled colour bar
NORM = TwoSlopeNorm(vmin=0.0, vcenter=0.05, vmax=1.0)
CBAR_TICKS = [0, 0.05, 0.5, 1]
VALUE_PT = S.ANNOT_PT  # larger values make stacked rows overlap (checks.text_overlaps)


def _get(v: dict, path: tuple):
    for k in path:
        v = (v or {}).get(k)
    return v


def _arm(path: tuple) -> str:
    """Arm code of a verdict-table path (the C2a / C3 paths are the focal arm, PMRT-GBM)."""
    return next((k for k in path if k in T.PAPER), "pmrt_nl_eq")


def _env_rate(c, arm: str, world: str, r: str) -> float | None:
    if r == "R3":  # confounded null edge, E4 only, every lam
        if world != "E4":
            return None
        s = c[(c.arm == arm) & (c.world == "E4") & (c.regime == "R3")].conf_raw_rate.dropna()
        return float(s.mean()) if len(s) else None
    q = T.world_null_rate(c, arm, world, ("R1", "R2") if r == "R1+R2" else (r,))
    return q[0] if q else None


def _fmt(x: float) -> str:
    if x < 0.005:
        return "0"
    if x > 0.995:
        return "1"
    return f"{x:.2f}"[1:]


def _layout():
    """Row positions (DESIGN s.7 method separation): banners [(y, text)], groups [(name, [y of each
    regime row])], subrows [(y, name, regime, spec)], ylim.
    """
    rows = [
        (name, None if spec is None else [r for r in REG_ORDER if r in spec]) for name, spec in ROWS
    ]
    banners, groups, _, ylim = L.method_rows(rows)
    specs = [
        (subs, spec) for (_, subs), (_, spec) in zip(rows, ROWS, strict=False) if spec is not None
    ]  # same order as groups
    subrows = [
        (yy, name, r, spec)
        for (name, ys), (regs, spec) in zip(groups, specs, strict=False)
        for yy, r in zip(ys, regs, strict=False)
    ]
    return banners, groups, subrows, ylim


def build():
    return build_panel("pooled")


def build_panel(which: str):
    """``pooled``: panel (a) alone, the body figure; ``env``: panel (b) alone, the ESM figure
    fig_study4_validity_env. Same rows,
    banners and method labels in both.
    """
    v = load_json(*T.EVAL, "report.json")["tables"]["V4_verdicts"]
    c = T.cells()
    fig, ax = plots.figure(
        "full", aspect=0.8 if which == "pooled" else 10
    )  # (a) alone: 0.8 x 5.15 in
    banners, groups, subrows, ylim = _layout()
    for yy, name, r, spec in subrows:
        if which == "pooled":
            x = _get(v, spec[r])
            if x:  # ink glyph of the method (DESIGN s.12)
                M.row_point(ax, x["rate"], yy, x["ci"], _arm(spec[r]), hero=_arm(spec[r]) == T.HERO)
            continue
        for j, w in enumerate(T.WORLDS):
            e = _env_rate(c, _arm(spec[r]), w, r)
            fc = P.DIV_CMAP(NORM(e)) if e is not None else P.WHITE
            ax.add_patch(
                Rectangle(
                    (j - 0.5, yy - PITCH / 2),
                    1,
                    PITCH,
                    fc=fc,
                    ec=P.WHITE,
                    lw=1.0,
                    zorder=2,
                    gid="cmap-cell",
                )
            )
            ax.text(
                j,
                yy,
                "–" if e is None else _fmt(e),
                ha="center",
                va="center",
                fontsize=VALUE_PT,
                color=P.INK if e is None else plots._text_on(fc),
                zorder=3,
            )
    if which == "pooled":
        plots.ref_line(ax, 0.05, axis="x")
        L.rate_axis(ax, "x", "Null-edge rate")
        plots.finish(ax, grid="x")
    else:
        ax.set_xlim(-0.5, len(T.WORLDS) - 0.5)
        ax.set_xticks(range(len(T.WORLDS)), [w if w != "E4" else "E4\nplacebo" for w in T.WORLDS])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Environment")
        plots.finish(ax, grid=False)
    ax.set_ylim(*ylim)
    ax.set_yticks([s[0] for s in subrows], [REG_LABEL[s[2]] for s in subrows], fontsize=S.ANNOT_PT)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.patch.set_visible(False)  # banners run underneath the panel
    for y, text in banners:
        L.banner(fig, ax, y, BANNER_H * 0.92, text)
    # method names left of the regime labels, on each method's first regime row
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    wreg = max(t.get_window_extent(r).width for t in ax.get_yticklabels()) * 72 / fig.dpi
    L.method_labels(ax, groups, wreg + 9)
    if which == "env":
        _colour_bar(fig, ax)
    return L.finalize(fig)


def _colour_bar(fig, b):
    """Colour bar of (b): a thin horizontal bar under its environment labels, panel width, as an
    inset of (b) so the constrained layout reserves room for it below both panels (rows stay
    aligned).
    """
    fig.draw_without_rendering()
    h_ax = b.get_window_extent().height / fig.dpi  # panel height (in)
    drop, thick = 0.62, 0.055  # in: below the two-line labels and the x label
    cax = b.inset_axes((0.08, -(drop + thick) / h_ax, 0.84, thick / h_ax))
    cb = fig.colorbar(
        ScalarMappable(norm=NORM, cmap=P.DIV_CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=CBAR_TICKS,
    )
    cb.ax.set_xticklabels(["0", ".05", ".5", "1"], fontsize=S.ANNOT_PT)
    cb.ax.tick_params(length=2, width=0.5, pad=1.5)
    cb.outline.set_visible(False)
    cb.set_label("Null-edge rate (near-white = nominal .05)", fontsize=S.ANNOT_PT, labelpad=2)

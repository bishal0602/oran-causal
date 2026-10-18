"""Shared builder for the Studies 2-3 referee figures (fig_decision_referee [G5 2026-10-12] uses the
helpers; build_referee is the three-panel builder of the retired fig_study2_referee /
fig_study3_referee, kept for reference): one forest row per referee arm or static rule, labelled
directly (bold method name; DEV-tuned / placebo-calibrated map thresholds as sub-rows), with (a)
recovery R and its paired-seed-bootstrap 90 % CI on a broken R axis: one panel with two linear
segments and an axis break between them, so the arms near accept-all stay readable and the arms far
below it stay visible at their true values (no value or CI falls inside the break; checked at build
time). R = 0 (accept-all) and R = 1 (the strongest static single-xApp rule) are reference lines,
labelled directly; (b) the RLF guard ratio and its 90 % CI, with the 1.10 guard limit as a labelled
reference line; (c) the other eligibility inputs as a plain table [DECIDED 2026-10-08, user; sample
"RA", replaces the heat cells]: energy retention and the SVR, non-protected eMBB and LL guard ratios
in ink, a value that fails its limit (retention < 0.90 or ratio > 1.10) bold, then the eligibility
verdict ("no" bold); a note under the table states the limits. No fills, no colour bar. Every number
is read from figures_data.json (FD) at build time, ``<study>.eval.arms.<arm>``: R, R_ci90,
retention, guard_ratio, guard_ratio_ci90, matched, guard_ok, eligible. No re-aggregation: the only
computation is comparing the stored point values with the protocol limits (0.90, 1.10) to outline a
cell, and that comparison is checked against the stored matched / guard_ok / eligible flags of every
arm. Methods are styled through oran_causal.figures.methods by arm code (static rules gray ``|``
rows, PMRT the hero glyph, the oracle map the open ink star, the random map gray ``+``;
edge-certified and map-driven arms look the same, the banners name the gate).
"""

import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory, offset_copy

from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _studies as ST

# TERMINOLOGY s.7 "eligible": energy retention >= 0.90 and every guard ratio <= 1.10 (point
# estimates). Protocol constants (X_MATCH / GUARD of the referee analyzers); checked below against
# every arm's stored flags.
X_MATCH, GUARD = 0.90, 1.10
GUARDS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
GRID = [("retention", "Energy"), ("svr", "SVR"), ("nonprot_embb_viol", "eMBB"), ("ll_viol", "LL")]
COL_X = (
    -0.15,
    1.0,
    2.0,
    3.0,
)  # [2026-10-12] x of each value column; Energy (the wide head) sits apart
X_ELIG = len(GRID) - 0.35  # x of the eligibility column in the value grid (data units)
THRESHOLD = {"dev": "DEV", "plc": "placebo"}

# broken R axis: FAR and NEAR are linear segments; the break between them takes W_GAP of the panel
# width
FAR, NEAR = (
    (-13.4, -3.4),
    (-0.65, 1.08),
)  # [G3 2026-10-12] NEAR tightened and W_FAR / W_GAP cut (were -0.75, 1.12;
W_FAR, W_GAP = 0.30, 0.06  # 0.34, 0.07) so the 0-1 strip gets about a fifth more width
R_TICKS = [-10, -5, 0, 0.5, 1]
_XP = np.array([FAR[0], FAR[1], NEAR[0], NEAR[1]])
_UP = np.array([0.0, W_FAR, W_FAR + W_GAP, 1.0])
TOP, BOTTOM = 1.9, 1.5  # row units above the first banner / below the last row (label strips)
WHITE_BOX = L.KNOCKOUT  # a label crossing a reference line stays legible
GAP = 0.62  # extra space between two methods (row units; layout.METHOD_GAP is 0.95)


def _pw(x, xp, fp):
    """Piecewise-linear map through (xp, fp), extended linearly beyond both ends (FuncScale needs
    every real x).
    """
    x = np.asarray(x, float)
    y = np.interp(x, xp, fp)
    lo_s = (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi_s = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(x < xp[0], fp[0] + (x - xp[0]) * lo_s, y)
    return np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * hi_s, y)


def _fwd(x):
    return _pw(x, _XP, _UP)


def _inv(u):
    return _pw(u, _UP, _XP)


def arms(study: str) -> dict:
    return ST.fd()[study]["eval"]["arms"]


def associational(study: str, gate: str) -> list[str]:
    """The study's referee arms on design-blind (associational) maps, in legend order; arms without
    a paper name (blanket2, B2) are skipped.
    """
    out = []
    for c in arms(study):
        if not c.startswith(gate + ":"):
            continue
        try:
            fam = M.resolve(c).method.family
        except KeyError:
            continue
        if fam.startswith("blind"):
            out.append(c)
    return M.order(out)


def _fails(a: dict) -> dict[str, bool]:
    """Which stored point values fail their limit; checked against the stored flags."""
    f = {k: a["guard_ratio"][k] > GUARD for k in GUARDS}
    f["retention"] = a["retention"] < X_MATCH
    if bool(a["matched"]) == f["retention"] or bool(a["guard_ok"]) == any(f[k] for k in GUARDS):
        raise ValueError("stored matched / guard_ok flags disagree with the 0.90 / 1.10 limits")
    if bool(a["eligible"]) != (bool(a["matched"]) and bool(a["guard_ok"])):
        raise ValueError("eligible != matched and guard_ok")
    return f


def sections_rows(sections):
    """sections: [(banner, [arm codes])] -> method_rows input and the arm codes per block. Codes of
    one method (map thresholds @dev / @plc) form one block: the bold method name, then one sub-row
    per threshold.
    """
    rows, codes = [], []
    for banner, cs in sections:
        rows.append((banner, None))
        groups: dict[str, list[str]] = {}
        for c in cs:
            groups.setdefault(M.resolve(c).key, []).append(c)
        for members in groups.values():
            members = sorted(
                members, key=lambda c: ["dev", "plc", None].index(M.resolve(c).threshold)
            )
            rows.append((M.name(members[0]), members))
            codes.append(members)
    return rows, codes


def r_axis(ax, label="Recovery R") -> None:
    ax.set_xscale("function", functions=(_fwd, _inv))
    ax.set_xlim(FAR[0], NEAR[1])
    ax.xaxis.set_major_locator(FixedLocator(R_TICKS))
    ax.set_xticks(R_TICKS, [f"{t:g}".replace("-", "−") for t in R_TICKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_off()
    ax.set_xlabel(label)


def r_break(ax) -> None:
    """Axis-break marks on the offset bottom spine: the spine is interrupted between the two
    segments and a short slanted stroke closes each end.
    """
    tr = offset_copy(
        blended_transform_factory(ax.transData, ax.transAxes),
        fig=ax.figure,
        y=-S.SPINE_OFFSET,
        units="points",
    )
    ax.plot(
        [FAR[1], NEAR[0]],
        [0, 0],
        color=P.WHITE,
        marker="",
        lw=2.4,
        solid_capstyle="butt",
        transform=tr,
        clip_on=False,
        zorder=6,
        gid="axis-break",
    )
    for x in (FAR[1], NEAR[0]):
        ax.plot(
            [x],
            [0],
            ls="",
            marker=[(-0.5, -1.0), (0.5, 1.0)],
            ms=6,
            mew=0.8,
            color=P.AXIS,
            mec=P.AXIS,
            transform=tr,
            clip_on=False,
            zorder=7,
            gid="axis-break",
        )  # [G6] mec: the rc default (white) hid it


def _check_break(values) -> None:
    for v in values:
        if FAR[1] < v < NEAR[0] or not FAR[0] <= v <= NEAR[1]:
            raise ValueError(
                f"R value {v} falls in the axis break or outside the R axis: move the segments"
            )


def ratio_axis(ax, lo: float, hi: float) -> None:
    step = 0.2 if hi - lo > 0.45 else 0.1
    ticks = [round(t, 2) for t in np.arange(1.0 - 5 * step, hi + 1e-9, step) if t >= lo - 1e-9]
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
    ax.set_xlabel("RLF ratio")


def _fmt(v: float) -> str:
    return "0.00" if abs(v) < 0.005 else f"{v:.2f}".replace("-", "−")


def build_referee(fig, axs, study: str, sections, hero_codes, *, note=None):
    """Draw the three panels. ``hero_codes``: arm codes that carry the hero glyph (the PMRT arms:
    one hero method). ``note``: (arm code, text), a direct label left of that arm's R interval.
    """
    a, b, c = axs
    data = arms(study)
    rows, codes = sections_rows(sections)
    banners, groups, subrows, (y0, _) = L.method_rows(rows, gap=GAP)
    y0 -= BOTTOM
    ys = [yy for yy, _, _ in subrows]
    flat = [code for grp in codes for code in grp]
    if len(flat) != len(ys):
        raise ValueError("row bookkeeping")
    r_vals, rlf_vals = [], []
    for yy, code in zip(ys, flat, strict=False):
        arm = data[code]
        hero = code in hero_codes
        R, (rlo, rhi) = arm["R"], arm["R_ci90"]
        r_vals += [R, rlo, rhi]
        M.row_point(a, R, yy, (rlo, rhi) if rhi > rlo else None, code, hero=hero)
        g, (glo, ghi) = arm["guard_ratio"]["rlf"], arm["guard_ratio_ci90"]["rlf"]
        rlf_vals += [g, glo, ghi]
        M.row_point(b, g, yy, (glo, ghi) if ghi > glo else None, code, hero=hero)
        fails = _fails(arm)
        for j, (k, _) in enumerate(GRID):
            v = arm[k] if k == "retention" else arm["guard_ratio"][k]
            c.text(
                COL_X[j],
                yy,
                _fmt(v),
                ha="center",
                va="center",
                fontsize=S.ANNOT_PT,
                color=P.INK,
                fontweight="bold" if fails[k] else "normal",
                zorder=3,
            )
        c.text(
            X_ELIG,
            yy,
            "yes" if arm["eligible"] else "no",
            ha="left",
            va="center",
            fontsize=S.ANNOT_PT,
            color=P.INK,
            fontweight="normal" if arm["eligible"] else "bold",
        )
    _check_break(r_vals)
    ytop = TOP
    # the accept-all row and the reference single rule sit on their own reference lines (glyph and
    # line coincide)
    ref_arm = ST.fd()[study]["eval"]["ref_arm"]
    for code, val in (("noarb", 0.0), (ref_arm, 1.0)):
        if code in flat and data[code]["R"] == val and data[code]["R_ci90"] == [val, val]:
            L.label_line(
                a, val, ys[flat.index(code)], f"R = {val:g}", where="left", bbox=WHITE_BOX
            )  # short: stays
            # inside its own segment, clear of the other reference line (was "reference, R = 0 / 1")
    # (a) recovery R on the broken axis
    plots.ref_line(a, 0, axis="x")
    plots.ref_line(a, 1, axis="x")
    r_axis(a)
    a.grid(axis="x", visible=True)
    a.grid(axis="y", visible=False)
    L.label_line(a, 1, ytop - 0.75, "best static single", where="left", bbox=WHITE_BOX)
    L.label_line(a, 0, y0 + 0.75, "accept-all", where="left", bbox=WHITE_BOX)
    if note:
        code, text = note
        k = flat.index(code)
        L.label_line(
            a, data[code]["R_ci90"][0], ys[k], text, where="left", linespacing=0.95, bbox=WHITE_BOX
        )
    # (b) RLF guard ratio
    lo = min(0.9, np.floor(min(rlf_vals) * 20) / 20 - 0.02)
    hi = max(GUARD + 0.08, max(rlf_vals) + 0.05)
    plots.ref_line(b, GUARD, axis="x")
    L.label_line(b, GUARD, ytop - 0.75, "guard limit", where="right")
    ratio_axis(b, lo, hi)
    b.grid(axis="x", visible=True)
    b.grid(axis="y", visible=False)
    # (c) value grid: energy retention, other guard ratios, eligibility
    for j, (_, head) in enumerate(GRID):
        c.text(
            COL_X[j], ytop - 0.75, head, ha="center", va="center", fontsize=S.ANNOT_PT, color=P.INK
        )
    c.text(
        X_ELIG, ytop - 0.75, "Eligible", ha="left", va="center", fontsize=S.ANNOT_PT, color=P.INK
    )
    c.text(
        COL_X[0] - 0.5,
        ys[-1] - 0.6,
        f"bold: fails its limit\n(energy < {X_MATCH:.2f}, ratio > {GUARD:.2f})",
        ha="left",
        va="top",
        fontsize=S.ANNOT_PT,
        color=P.INK,
        linespacing=1.0,
    )  # [2026-10-12] hangs below the last row
    c.set_xlim(COL_X[0] - 0.55, X_ELIG + 1.05)
    c.axis("off")
    for ax in (a, b, c):
        ax.set_ylim(y0, ytop)
        ax.patch.set_visible(False)  # banners run underneath every panel
    for ax in (a, b):
        plots.finish(ax)
        ax.tick_params(axis="y", length=0)
    a.set_yticks(
        ys, [THRESHOLD.get(M.resolve(code).threshold, "") for code in flat], fontsize=S.ANNOT_PT
    )
    b.set_yticks(ys, [""] * len(ys))
    b.tick_params(axis="y", labelleft=False)
    r_break(a)
    for y, text in banners:
        L.banner(fig, a, y, L.BANNER_H * 0.92, text)
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    w = [t.get_window_extent(r).width for t in a.get_yticklabels() if t.get_text()]
    wreg = (max(w) if w else 0) * 72 / fig.dpi
    # [2026-10-12] a method without DEV / placebo sub-rows needs no sub-label column: its name sits
    # next to the axes, which frees the width the 7 pt table in (c) needs
    sub = {
        yy
        for yy, code in zip(ys, flat, strict=False)
        if THRESHOLD.get(M.resolve(code).threshold) or (note and code == note[0])
    }
    L.method_labels(a, [g for g in groups if sub & set(g[1])], wreg + 5)
    L.method_labels(a, [g for g in groups if not sub & set(g[1])], 5)
    L.panel_header(a, "a", "Recovery R, 90 % CI")
    L.panel_header(b, "b", "RLF, 90 % CI")
    L.panel_header(c, "c", "Eligibility inputs")
    for ax in (b, c):  # letters / tags of the inner panels sit above the panel to their
        for t in ax.texts:  # left; out of the layout so they do not widen the gaps (the
            if t.get_gid() in ("panel-letter", "panel-tag"):  # layout checks guard overlaps)
                t.set_in_layout(False)
    for t in a.texts:
        if t.get_gid() == "panel-tag":
            t.set_in_layout(False)
    return L.finalize(fig)

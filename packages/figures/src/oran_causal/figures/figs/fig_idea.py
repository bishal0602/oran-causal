"""Section 4 opener (F3): controller logs are confounded; a randomised accept / reject decision with
its probability logged cuts the confounding path, and redrawing the logged decisions gives the
reference distribution of the test.

(a) reactive controller: network state -> decision, network state -> KPI (the confounding path,
SPURIOUS hue, labelled once), decision -> KPI (the effect). (b) the same with a logged coin driving
the decision; the state -> decision arrow faint and crossed out. The decision node is the one hero
component (DESIGN s.10). (c) real redraws: the frozen Study 1 PMRT statistic (arm loadsp_c, the
primary) for sleep -> neighbour protected-slice violations, B = 9999 redraws of the logged decisions
on the pooled randomised logs (600 episodes) and on the placebo logs (40 episodes), observed value
as an ink line. The redraws come from figures/data/fig_idea_redraws.json, written by
figures/scripts/fig_idea_redraws.py from the stored evaluation logs with the frozen code; the build
re-derives the observed z and the p-values from them and asserts both against the stored analyzer
output (v4_analysis.json).
"""

import json

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from oran_causal.figures import diagram as D
from oran_causal.figures import layout as L
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.data import load_json
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.paths import DATA
from oran_causal.figures.registry import FigSpec

REDRAWS = DATA / "fig_idea_redraws.json"
HYP, ARM = "sleep|nbr|pv", "loadsp_c"

SPEC = FigSpec(
    name="fig_idea",
    width="full",
    paper=True,
    sources=[ST.SRC_V4, "packages/figures/data/fig_idea_redraws.json"],
    caption_hint="Confounding in controller logs and how randomised logging removes it; (c) one true edge, 9999 redraws",
)

H_IN = 2.2
PANEL_W = 114.0  # diagram panel width (pt)
GAP = 6.0  # between the two diagram panels (pt)
TAG_B_PT = 123.0  # the (b) header (letter + tag) reaches this far right of the panel's left edge
TOP = 18.0  # header band above the diagrams (pt)
HEAD = 12.0  # log-type heading band above each histogram (pt)


def load() -> dict:
    """Redraws per log type, with the observed z and the p-values re-derived and asserted against
    v4_analysis.json.
    """
    doc = json.loads(REDRAWS.read_text(encoding="utf-8"))
    v4 = load_json(*ST.V4_PARTS)
    assert doc["hypothesis"] == HYP
    assert doc["arm"] == ARM
    assert doc["B"] == 9999
    stored = {
        "randomised": v4["slices_result"]["pooled"]["hyp"][HYP],
        "placebo": v4["placebo_hyp"][HYP],
    }
    gt = ST.v4_gt_cells()[tuple(HYP.split("|"))]
    assert gt["status"] == "TRUE"
    assert gt["sign"] == 1
    out = {}
    for kind in ("randomised", "placebo"):
        d, ref = doc[kind], stored[kind]
        zb = np.asarray(d["z_redrawn"], float)
        z, o = d["z_obs"], d["orientation"]
        assert zb.size == doc["B"]
        assert abs(abs(z) - abs(ref[ARM]["z"])) < 1e-9 and np.sign(z) * o == np.sign(
            ref[ARM]["z"]
        ), (kind, z)
        tol = 1e-9 * (1 + abs(z))
        p2 = (1 + np.count_nonzero(np.abs(zb) >= abs(z) - tol)) / (zb.size + 1)
        pp = (1 + np.count_nonzero(o * zb >= o * z - tol)) / (zb.size + 1)
        # redraws rounded to 5 decimals: the counts must still reproduce the stored p-values exactly
        assert abs(p2 - ref[ARM]["p2"]) < 1e-12 and abs(pp - ref[ARM]["p_plus"]) < 1e-12, (
            kind,
            p2,
            pp,
            ref[ARM],
        )
        out[kind] = {"z": o * z, "zb": o * zb, "p": pp, "declared": ref["primary"]["declared"]}
    assert out["randomised"]["declared"]
    assert not out["placebo"]["declared"]
    assert out["randomised"]["p"] == 1 / 10000
    assert out["randomised"]["z"] > out["randomised"]["zb"].max()
    # room for the rotated "observed" label on the side of the line away from the bars
    assert out["randomised"]["z"] + 2.2 < XLIM[1]
    assert XLIM[0] + 2.2 < out["placebo"]["z"] < 0
    return out


def _canvas(fig, x0_pt: float, w_pt: float, h_pt: float):
    W, H = fig.get_size_inches() * 72
    ax = fig.add_axes((x0_pt / W, 0, w_pt / W, h_pt / H))
    ax.set_xlim(0, w_pt)
    ax.set_ylim(h_pt, 0)
    ax.axis("off")
    return ax


def _arrow(ax, a, b, color=P.INK, lw=0.7, ls="-", z=2, gid="arrow"):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>,head_length=4,head_width=2.2",
            mutation_scale=1,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
            gid=gid,
        )
    )


def _glow(ax, pts, color, role: str):
    """Soft highlighter band behind a causal path (semantic colour field)."""
    xs, ys = zip(*pts, strict=False)
    ax.plot(
        xs,
        ys,
        color=P.tint(color, 0.2),
        lw=8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=0.5,
        gid=f"sem:{role}",
    )


# node boxes (pt, y down): state top right, decision bottom left, KPI bottom right; the coin sits
# top left in (b)
BH = 20.0
STATE = (54.0, 8.0, 58.0)  # x, y, w
DECIS = (2.0, 96.0, 52.0)
KPI = (80.0, 96.0, 32.0)
SEM_LW = 1.1  # semantic arrows: a touch heavier than the 0.7 pt ink arrows


def _diagram(ax, randomised: bool):
    sx, sy, sw = STATE
    dx, dy, dw = DECIS
    kx, ky, kw = KPI
    D.box(
        ax,
        sx,
        sy,
        sw,
        BH,
        "Network state",
        role="entity",
        color=P.CONFOUNDER,
        lw=0.9,
        gid="sem:confounder",
    )
    D.box(ax, dx, dy, dw, BH, "Decision", role="proposed", gid="sem:ours")
    D.box(ax, kx, ky, kw, BH, "KPI").set_facecolor(P.tint(P.NEUTRAL, 0.16))
    # decision -> KPI: the effect we want
    _arrow(ax, (dx + dw, dy + BH / 2), (kx - 1, ky + BH / 2), lw=0.9)
    ax.text(
        (dx + dw + kx) / 2, dy + BH / 2 + 4, "effect", ha="center", va="top", fontsize=S.ANNOT_PT
    )
    s_dec = (sx + 3, sy + BH)  # state -> decision leaves the state's bottom-left corner
    e_dec = (dx + dw - 12, dy - 1)
    s_kpi, e_kpi = (kx + kw / 2, sy + BH), (kx + kw / 2, ky - 1)
    if not randomised:
        _glow(ax, [s_dec, e_dec], P.CONFOUNDER, "confounder-path")
        _glow(ax, [s_kpi, e_kpi], P.CONFOUNDER, "confounder-path")
        _arrow(ax, s_dec, e_dec, color=P.CONFOUNDER, lw=SEM_LW, gid="sem:confounder")
        _arrow(ax, s_kpi, e_kpi, color=P.CONFOUNDER, lw=SEM_LW, gid="sem:confounder")
        ym = (sy + BH + dy) / 2
        xl = s_dec[0] + (e_dec[0] - s_dec[0]) * (ym - s_dec[1]) / (e_dec[1] - s_dec[1])
        ax.text(
            (xl + s_kpi[0]) / 2,
            ym,
            "confounding",
            ha="center",
            va="center",
            fontsize=S.ANNOT_PT,
            fontweight="bold",
        )
        return
    # state -> decision cut: faint purple, crossed out; state -> KPI stays (no longer a back-door
    # path)
    _arrow(
        ax,
        s_dec,
        e_dec,
        color=P.tint(P.CONFOUNDER, 0.45),
        lw=0.7,
        ls=(0, (2, 1.6)),
        z=1,
        gid="sem:confounder-cut",
    )
    mx, my = (s_dec[0] + e_dec[0]) / 2, (s_dec[1] + e_dec[1]) / 2
    ax.text(
        mx,
        my,
        "×",
        ha="center",
        va="center",
        fontsize=11,
        color=P.INK,
        zorder=4,
        bbox={"boxstyle": "circle,pad=0.05", "fc": P.WHITE, "ec": "none"},
    )
    _arrow(ax, s_kpi, e_kpi, color=P.CONFOUNDER, lw=0.8, gid="sem:confounder")
    # the coin with its logged probability drives the decision
    cx, cy, r = 18.0, sy + BH / 2, 9.0
    _glow(ax, [(cx, cy + r + 18), (cx, dy)], P.DESIGN_ROLE, "design-path")
    ax.add_patch(
        Circle(
            (cx, cy),
            r,
            fc=P.tint(P.DESIGN_ROLE, 0.5),
            ec=P.DESIGN_ROLE,
            lw=0.9,
            zorder=3,
            gid="sem:design",
        )
    )
    ax.text(cx, cy, "p", ha="center", va="center", fontsize=S.TICK_PT, style="italic", zorder=4)
    ax.text(
        cx,
        cy + r + 3,
        "logged p",
        ha="center",
        va="top",
        fontsize=S.ANNOT_PT,
        fontweight="bold",
        zorder=4,
    )
    _arrow(ax, (cx, cy + r + 15), (cx, dy - 1), color=P.DESIGN_ROLE, lw=SEM_LW, gid="sem:design")


XLIM = (-4.5, 11.2)
BINS = np.arange(-4.5, 4.75, 0.25)


TAIL = 0.95  # the shaded tail: top 5 % of the redraws (one-sided, the tested direction)


def _hist(ax, d, label: str, tail_label: bool = False):
    zb, z = d["zb"], d["z"]
    assert BINS[0] <= zb.min() and zb.max() <= BINS[-1], (zb.min(), zb.max())
    q = float(np.quantile(zb, TAIL))
    real = z > q
    assert real == d["declared"], (
        label,
        z,
        q,
    )  # the line lands in the tail exactly when PMRT declares
    _, _, bars = ax.hist(zb, bins=BINS, color=P.tint(P.OURS, 0.45), ec=P.WHITE, lw=0.3, zorder=2)
    for b in bars:
        b.set_gid("sem:ours")
    n = np.histogram(zb, bins=BINS)[0]
    top = n.max() * 1.7
    ax.set_ylim(0, top)
    ax.set_xlim(*XLIM)
    ax.axvspan(q, XLIM[1], color=P.tint(P.GOOD, P.ZONE_TINT), lw=0, zorder=0.5, gid="sem:good-zone")
    if tail_label:
        ax.text(
            XLIM[1] - 0.3,
            top * 0.06,
            "top 5 %",
            ha="right",
            va="bottom",
            fontsize=S.ANNOT_PT,
            color=P.INK,
        )
    ax.axvline(
        z,
        color=P.GOOD if real else P.NEUTRAL,
        lw=1.8 if real else 1.3,
        zorder=3,
        gid="sem:good" if real else "observed-null",
    )
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=S.ANNOT_PT,
        color=P.INK,
    )
    away = "right" if z > np.median(zb) else "left"  # the side of the line away from the bars
    L.label_line(ax, z, top, "observed", where=away, va="top", rotation=90)
    if real:  # bars' side, inside the tail zone with the line
        L.label_line(ax, z, top, "real effect", where="left", va="top", fontweight="bold")
    else:  # outside side, beyond "observed": clear of the tail zone
        ax.annotate(
            "no effect",
            xy=(z, 1),
            xycoords=("data", "axes fraction"),
            xytext=(-16, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=S.ANNOT_PT,
            fontweight="bold",
            color=P.INK,
        )
    ax.axhline(0, color=P.AXIS, lw=S.LW_AXIS, zorder=2.5)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)


def build():
    data = load()
    S.apply()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(S.FULL, H_IN))
    W, H = S.FULL * 72, H_IN * 72
    dia_h = H - TOP
    axa = _canvas(fig, 2.0, PANEL_W, dia_h)
    axb = _canvas(fig, 2.0 + PANEL_W + GAP, PANEL_W, dia_h)
    _diagram(axa, randomised=False)
    _diagram(axb, randomised=True)
    L.panel_header(axa, "a", "Reactive controller", inset=True)
    L.panel_header(axb, "b", "Randomised, logged decision", inset=True)
    # (c): two stacked histograms sharing the x axis
    cx0 = 2.0 + PANEL_W + GAP + TAG_B_PT + 30.0  # letter c (about 20 pt) 10 pt clear of the (b) tag
    cw = W - cx0 - 6.0
    bottom, gap = 30.0, HEAD + 4.0
    ch = (H - TOP - HEAD - bottom - gap) / 2
    axt = fig.add_axes((cx0 / W, (bottom + ch + gap) / H, cw / W, ch / H))
    axl = fig.add_axes((cx0 / W, bottom / H, cw / W, ch / H))
    _hist(axt, data["randomised"], "Randomised logs")
    _hist(axl, data["placebo"], "Placebo logs", tail_label=True)
    L.hide_x(axt)
    axt.set_ylabel("Redraws")
    axl.set_xlabel("Test statistic")
    axl.set_xticks([-4, 0, 4, 8])
    for ax in (axt, axl):
        plots.finish(ax, trim=False, grid=False)
        ax.spines["left"].set_visible(False)
    axt.spines["bottom"].set_visible(False)
    L.panel_header(axt, "c", "Redraw test")
    L.finalize(fig)
    for t in axt.texts:  # lift the (c) header over the heading band: level with (a) and (b)
        if t.get_gid() in ("panel-letter", "panel-tag"):
            t.xyann = (t.xyann[0], t.xyann[1] + HEAD)
    fig.draw_without_rendering()
    return fig

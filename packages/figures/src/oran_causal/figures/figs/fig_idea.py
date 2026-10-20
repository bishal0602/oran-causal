"""Section 4 opener (F3): controller logs are confounded; a randomised accept / reject decision with
its probability logged cuts the confounding path, and redrawing the logged decisions gives the
reference distribution of the test.

(a) reactive controller: network state -> mode W, network state -> KPI (the confounding arch,
purple, labelled once), W -> KPI (the effect). (b) randomised logging: the state sets the logged
accept probability pi, an independent uniform draw U ~ Unif(0,1) turns it into the mode
(W = accept iff U < pi); the state -> KPI arch stays, muted. (c) real redraws: the frozen Study 1
PMRT statistic (arm loadsp_c, the primary) for sleep -> neighbour protected-slice violations,
B = 9999 redraws of the logged decisions on the pooled randomised logs (600 episodes) and on the
placebo logs (40 episodes), observed value as a bold line labelled with its z, one-sided p-value
(the declaration p_d, d = +1) in the panel subtitle. The redraws come from
figures/data/fig_idea_redraws.json, computed from the stored evaluation logs; the build
re-derives the observed z and the p-values from them and asserts both against the stored analyzer
output (v4_analysis.json).
"""

import json

import numpy as np

from oran_causal.figures import diagram as D
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

H_PT = 300.0
PAD_PT = 1.4  # savefig pad (0.02 in) on each side
XLIM = (-4.5, 10.0)
BINS = np.arange(-4.5, 4.75, 0.25)
TAIL = 0.95  # the shaded tail: top 5 % of the redraws (one-sided, the tested direction)
SEM_LW = 1.1


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
    # room for the z label on the side of the line away from the bars
    assert out["randomised"]["z"] + 1.0 < XLIM[1]
    assert XLIM[0] + 2.2 < out["placebo"]["z"] < 0
    return out


def _text(ax, x, y, s, ha="center", va="center", bold=False):
    return ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=S.ANNOT_PT,
        fontweight="bold" if bold else "normal",
        color=P.INK,
        zorder=5,
    )


def _header(ax, x, y, letter, tag):
    ax.text(
        x,
        y,
        letter,
        ha="left",
        va="baseline",
        fontsize=S.PANEL_PT,
        fontweight="bold",
        color=P.INK,
        gid="panel-letter",
    )
    ax.text(
        x + 21,
        y,
        tag,
        ha="left",
        va="baseline",
        fontsize=S.TAG_PT,
        fontweight="bold",
        color=P.INK,
        gid="panel-tag",
    )


def _card(ax, y, h):
    D.container(ax, 2, y, 366, h)


# node geometry shared by (a) and (b): x, width
STATE_X, STATE_W = 12.0, 52.0
PI_X, PI_W = 116.0, 56.0
W_X, W_W = 252.0, 44.0
KPI_X, KPI_W = 332.0, 32.0
U_X, U_W = 234.0, 80.0
NODE_H = 22.0


def _chain(ax, top: float, randomised: bool):
    by, ay = top + 22, top + 14
    mid = by + NODE_H / 2
    wc = W_X + W_W / 2
    D.box(
        ax,
        STATE_X,
        by,
        STATE_W,
        NODE_H,
        "State",
        role="entity",
        color=P.CONFOUNDER,
        lw=0.9,
        gid="sem:confounder",
    )
    D.box(ax, W_X, by, W_W, NODE_H, "W", role="proposed", gid="sem:ours")
    D.box(ax, KPI_X, by, KPI_W, NODE_H, "KPI")
    D.arrow(ax, [(W_X + W_W, mid), (KPI_X - 1, mid)], lw=0.9)
    _text(ax, (W_X + W_W + KPI_X) / 2, mid - 4, "effect", va="bottom")
    sc = STATE_X + STATE_W / 2
    kc = KPI_X + KPI_W / 2
    arch = [(sc, by), (sc, ay), (kc, ay), (kc, by - 1)]
    if not randomised:
        D.arrow(ax, arch, color=P.CONFOUNDER, lw=SEM_LW)
        D.arrow(ax, [(STATE_X + STATE_W, mid), (W_X - 1, mid)], color=P.CONFOUNDER, lw=SEM_LW)
        _text(ax, (sc + kc) / 2, ay - 3, "confounding", va="bottom", bold=True)
        return
    D.arrow(ax, arch, color=P.tint(P.CONFOUNDER, 0.55), lw=0.8)
    D.box(
        ax,
        PI_X,
        by - 2,
        PI_W,
        NODE_H + 4,
        "π",
        sub="logged",
        role="entity",
        color=P.DESIGN_ROLE,
        lw=1.4,
        gid="sem:design",
    )
    D.arrow(ax, [(STATE_X + STATE_W, mid), (PI_X - 1, mid)], color=P.CONFOUNDER, lw=0.9)
    D.arrow(ax, [(PI_X + PI_W, mid), (W_X - 1, mid)], color=P.DESIGN_ROLE, lw=SEM_LW)
    D.box(
        ax,
        U_X,
        by + 34,
        U_W,
        NODE_H,
        "U ~ Unif(0,1)",
        role="entity",
        color=P.DESIGN_ROLE,
        lw=1.4,
        gid="sem:design",
    )
    D.arrow(ax, [(wc, by + 33), (wc, by + NODE_H + 1)], color=P.DESIGN_ROLE, lw=SEM_LW)


HEADS = {"randomised": "600 episodes", "placebo": "Placebo, 40 episodes"}


def _hist(fig, ax0, x, y, w, h, d, kind: str, ylabel: bool = False):
    W, H = fig.get_size_inches() * 72
    ax = fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H))
    zb, z = d["zb"], d["z"]
    assert BINS[0] <= zb.min() and zb.max() <= BINS[-1], (zb.min(), zb.max())
    q = float(np.quantile(zb, TAIL))
    real = z > q
    assert real == d["declared"], (kind, z, q)  # the line is in the tail exactly when PMRT declares
    _, _, bars = ax.hist(zb, bins=BINS, color=P.tint(P.OURS, 0.45), ec=P.WHITE, lw=0.3, zorder=2)
    for b in bars:
        b.set_gid("sem:ours")
    top = np.histogram(zb, bins=BINS)[0].max() * 1.9
    ax.set_ylim(0, top)
    ax.set_xlim(*XLIM)
    plots.span(ax, q, XLIM[1], "good", "x", zorder=0.5)
    ax.axvline(
        z,
        color=P.GOOD if real else P.NEUTRAL,
        lw=2.6,
        zorder=3,
        gid="sem:good" if real else "observed-null",
    )
    side = -1 if real else 1  # the z label sits on the side of the line away from the bars
    ax.annotate(
        f"z = {z:.2f}".replace("-", "−"),
        xy=(z, top),
        xytext=(5 * side, -3),
        textcoords="offset points",
        ha="right" if real else "left",
        va="top",
        fontsize=S.ANNOT_PT,
        fontweight="bold",
        color=P.INK,
    )
    ax.axhline(0, color=P.AXIS, lw=S.LW_AXIS, zorder=2.5)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_xticks([-4, 0, 4, 8])
    plots.finish(ax, grid=False)
    ax.set_xlabel("Test statistic")
    if ylabel:
        ax.set_ylabel("Redraws")
    p = d["p"]
    ptxt = "p ≤ 1e-4" if p <= 1e-4 + 1e-12 else f"p = {p:.2f}"
    ax0.text(
        x,
        y - 2,
        HEADS[kind],
        ha="left",
        va="bottom",
        fontsize=S.ANNOT_PT,
        fontweight="bold",
        color=P.INK,
    )
    ax0.text(x + w, y - 2, ptxt, ha="right", va="bottom", fontsize=S.ANNOT_PT, color=P.INK)
    return ax


def build():
    data = load()
    S.apply()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(S.FULL, H_PT / 72))
    W = S.FULL * 72
    # the canvas spans the text width less the savefig pad on both sides, so the saved page is
    # 	extwidth wide (a wider page makes LaTeX shrink the 8 pt text)
    ax = fig.add_axes((PAD_PT / W, 0, (W - 2 * PAD_PT) / W, 1))
    ax.set_xlim(PAD_PT, W - PAD_PT)
    ax.set_ylim(H_PT, 0)
    ax.axis("off")
    for top, h, letter, tag, rand in (
        (15, 58, "a", "Reactive", False),
        (90, 92, "b", "Randomised", True),
    ):
        _header(ax, 2, top - 4, letter, tag)
        _card(ax, top, h)
        _chain(ax, top, rand)
    _header(ax, 2, 196, "c", "Redraw test")
    _hist(fig, ax, 34, 222, 156, 50, data["randomised"], "randomised", ylabel=True)
    _hist(fig, ax, 206, 222, 156, 50, data["placebo"], "placebo")
    fig.draw_without_rendering()
    return fig

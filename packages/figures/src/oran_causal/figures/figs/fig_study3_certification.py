"""Study 3 certification (PAPER_PLAN s.6.3, s.7 Discussion P2; claims S3.4, S3.5): for each
rejection class (f, d) the edge-certified referee may use, and each of its six guard edges (own /
nbr / far relation x service violations v / RLF), (a) the one-sided 90 % upper bound UB90 of the
rejection harm relative to the class threshold tau (log scale; the edge is certified iff UB90 <=
tau, so tau is the reference line at 1) for the PMRT map and the oracle map, and (b) the
certification status grid: per class and edge, certified (filled square), not certified (open
square) or declared in the map (dash: a declared edge is used by the map itself and needs no
certificate), plus the PMRT referee's rejected / withheld units per class. Sources (all in FD):
``certsafe.cert.<arm>.classes`` (edges: relation, KPI, status, UB90, tau; class certified flag),
``certsafe.tau`` (cross-check of every edge's tau), ``certsafe.pmrt_bound_ptx_nbr_rlf`` (the
deciding edge, labelled; cross-checked against the cert entry) and
``certsafe.eval.composition.CS:PMRT`` (rejected / withheld units). Re-aggregation: the ratio UB90 /
tau of two stored values per edge; the status is the stored one (checked: CERTIFIED iff UB90 <=
tau). Rows are class banners and labelled edge rows (no entity hues, DESIGN s.12.2).
"""

from matplotlib.lines import Line2D

from oran_causal.figures import entities as E
from oran_causal.figures import layout as L
from oran_causal.figures import methods as M
from oran_causal.figures import palette as P
from oran_causal.figures import plots
from oran_causal.figures import style as S
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.registry import FigSpec

SPEC = FigSpec(
    name="fig_study3_certification",
    width="full",
    paper=True,
    sources=[ST.SRC_FD],
    caption_hint="Study 3 certification per rejection class (f, d) and guard edge (own / nbr / far relation; "
    "v service violations, RLF radio-link failures): one-sided 90 % upper bound of the rejection "
    "harm relative to the class threshold tau (a, log scale; certified iff UB90 <= tau) for the "
    "PMRT map and the oracle map under the edge-certified gate, and the certification "
    "status (b: filled certified, open not certified, dash declared in the map, no certificate "
    "needed) with the PMRT referee's rejected or withheld units per class",
)

ARMS = [("CS:PMRT", "PMRT map"), ("CS:GT", "Oracle map")]
HERO = "CS:PMRT"
EDGES = [(r, k) for r in ("own", "nbr", "far") for k in ("v", "rlf")]
KPI = {"v": "violations", "rlf": "RLF"}
REL = {"own": "own", "nbr": "neighbour", "far": "far"}
DODGE = {
    "CS:PMRT": 0.2,
    "CS:GT": -0.2,
}  # row units: the two maps' bounds of one edge sit side by side
XLIM, XTICKS = (0.006, 20), [0.01, 0.1, 1, 10]
STATUS = {
    "CERTIFIED": {"marker": "s", "mfc": P.INK, "mec": P.INK},
    "UNRESOLVED": {"marker": "s", "mfc": P.WHITE, "mec": P.INK},
    "DECLARED": {"marker": "_", "mfc": P.INK, "mec": P.INK},
}
STATUS_LABEL = {
    "CERTIFIED": "certified (≤ threshold)",
    "UNRESOLVED": "not certified (> threshold)",
    "DECLARED": "declared in the map (no certificate needed)",
}
TOP = 1.6
WHITE_BOX = L.KNOCKOUT


def _class_label(cls: str) -> str:
    f, d = cls.split("|")
    return f"{E.param_plain_label(f)}, {'up' if int(d) > 0 else 'down'}"


def _units(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _status_glyph(ax, x, y, status: str) -> None:
    ax.plot(
        [x],
        [y],
        ls="",
        ms=S.MARKER_SIZE + (1.2 if status == "DECLARED" else 0),
        mew=0.8,
        **STATUS[status],
        zorder=4,
        clip_on=False,
        gid=f"status:{status.lower()}",
    )


def build():
    cs = ST.fd()["certsafe"]
    cert, tau_all, comp = cs["cert"], cs["tau"], cs["eval"]["composition"]["CS:PMRT"]
    classes = list(cert[HERO]["classes"])
    if set(classes) != set(cert["CS:GT"]["classes"]):
        raise ValueError("PMRT and oracle maps have different rejection classes")
    rows = []
    for cls in classes:
        rows += [(_class_label(cls), None), ("", [f"{r}|{k}" for r, k in EDGES])]
    banners, _, subrows, (y0, _) = L.method_rows(rows, gap=0.45)
    fig, (a, b) = plots.figure(
        "full", aspect=0.874, ncols=2, width_ratios=[1.55, 1.0]
    )  # [G5] 4.5 in (was 0.78): 8 pt rows
    ys = {}  # (class, edge) -> y
    k = 0
    for cls in classes:
        for r, kp in EDGES:
            ys[(cls, f"{r}|{kp}")] = subrows[k][0]
            k += 1
    ban_y = {cls: y for cls, (y, _) in zip(classes, banners, strict=False)}
    for j, (code, _) in enumerate(ARMS):
        for cls in classes:
            c = cert[code]["classes"][cls]
            f, d = cls.split("|")
            for rel, kpi, status, ub, tau in c["edges"]:
                if abs(tau - tau_all[f"{f}|{int(d):+d}|{kpi}"]) > 1e-9 * max(1.0, tau):
                    raise ValueError(f"tau mismatch for {code} {cls} {rel} {kpi}")
                y = ys[(cls, f"{rel}|{kpi}")]
                _status_glyph(b, j, y, status)
                if ub is None:
                    if status != "DECLARED":
                        raise ValueError(f"{code} {cls} {rel} {kpi}: no bound but status {status}")
                    continue
                if (status == "CERTIFIED") != (ub <= tau) or status == "DECLARED":
                    raise ValueError(
                        f"{code} {cls} {rel} {kpi}: status {status} vs UB90 {ub} / tau {tau}"
                    )
                ratio = ub / tau
                if not XLIM[0] < ratio < XLIM[1]:
                    raise ValueError(f"UB90 / tau {ratio} outside the axis")
                M.row_point(a, ratio, y + DODGE[code], None, code, hero=code == HERO)
            if all(e[2] != "UNRESOLVED" for e in c["edges"]) != bool(c["certified"]):
                raise ValueError(f"{code} {cls}: class flag disagrees with its edges")
            _status_glyph(b, j, ban_y[cls], "CERTIFIED" if c["certified"] else "UNRESOLVED")
    # PMRT referee units per class family (deferred, or withheld as uncertified)
    for cls in classes:
        f = cls.split("|")[0]
        dn, wn = comp["defer"][f], comp["withheld"][f]
        txt = f"{_units(dn)} rejected" if dn else (f"{_units(wn)} withheld" if wn else "none")
        b.text(1.75, ban_y[cls], txt, ha="left", va="center", fontsize=S.ANNOT_PT, color=P.INK)
    # the deciding uncertified edge (FD certsafe.pmrt_bound_ptx_nbr_rlf), cross-checked against the
    # cert entry
    pb = cs["pmrt_bound_ptx_nbr_rlf"]
    e = next(x for x in cert[HERO]["classes"]["ptx|1"]["edges"] if x[0] == "nbr" and x[1] == "rlf")
    if abs(e[3] - pb["ub90_defer_up"]) > 1e-9 or abs(e[4] - pb["tau"]) > 1e-9:
        raise ValueError("pmrt_bound_ptx_nbr_rlf disagrees with the cert entry")
    L.label_line(
        a,
        pb["ub90_defer_up"] / pb["tau"],
        ys[("ptx|1", "nbr|rlf")] + DODGE[HERO],
        f"bound {pb['ub90_defer_up']:.3f} vs threshold {pb['tau']:.3f}",
        where="left",
        bbox=WHITE_BOX,
    )
    # (a) axis
    plots.ref_line(a, 1, axis="x")
    L.label_line(a, 1, TOP - 0.7, "bound = threshold", where="right")
    a.set_xscale("log")
    a.set_xlim(*XLIM)
    a.set_xticks(XTICKS, ["0.01", "0.1", "1", "10"])
    a.minorticks_off()
    a.set_xlabel("Bound ÷ threshold (log scale)")
    a.grid(axis="x", visible=True)
    a.grid(axis="y", visible=False)
    # (b) status grid
    for j, (_, name) in enumerate(ARMS):
        b.text(
            j,
            TOP - 0.7,
            name.replace(" map", "\nmap"),
            ha="center",
            va="center",
            fontsize=S.ANNOT_PT,
            color=P.INK,
            linespacing=0.95,
        )
    b.text(
        1.75,
        TOP - 0.7,
        "PMRT referee\nunits",
        ha="left",
        va="center",
        fontsize=S.ANNOT_PT,
        color=P.INK,
        linespacing=0.95,
    )
    b.set_xlim(-0.6, 3.3)
    b.axis("off")
    for ax in (a, b):
        ax.set_ylim(y0, TOP)
        ax.patch.set_visible(False)
    plots.finish(a)
    a.tick_params(axis="y", length=0)
    a.set_yticks(
        [y for y, _, _ in subrows],
        [f"{REL[s.split('|')[0]]}, {KPI[s.split('|')[1]]}" for _, _, s in subrows],
        fontsize=S.ANNOT_PT,
    )
    for y, text in banners:
        L.banner(fig, a, y, L.BANNER_H * 0.92, text)
    L.panel_header(a, "a", "90 % upper bound of the rejection harm")
    L.panel_header(b, "b", "Certification status")
    for t in b.texts:
        if t.get_gid() in ("panel-letter", "panel-tag"):
            t.set_in_layout(False)
    handles = []
    for code, name in ARMS:
        st = M.style(code, code == HERO)
        for key in ("gid", "zorder", "label"):
            st.pop(key)
        st["ls"] = "none"
        handles.append((Line2D([], [], **st), name))
    for s in ("CERTIFIED", "UNRESOLVED", "DECLARED"):
        handles.append(
            (
                Line2D(
                    [],
                    [],
                    ls="none",
                    ms=S.MARKER_SIZE + (1.2 if s == "DECLARED" else 0),
                    mew=0.8,
                    **STATUS[s],
                ),
                STATUS_LABEL[s],
            )
        )
    L.fig_legend(fig, [h for h, _ in handles], [n for _, n in handles])
    return L.finalize(fig)

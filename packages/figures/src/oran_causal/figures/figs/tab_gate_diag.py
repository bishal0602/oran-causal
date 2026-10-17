"""Post hoc analysis X10, gate diagnosis of the edge-certified PMRT referee (Section 6.3 or Appendix
C; booktabs LaTeX + csv): for every rejection class (control-parameter family and direction) the
PMRT referee may reject, the guard edges left uncertified, with the rejection harm h x beta of the
PMRT bound, its one-sided 90 % upper bound UB90, the class tolerance tau, the Study 2 knockout
ground-truth effect of a rejection (h x GT mean, 95 % CI) and label, and whether that ground-truth
harm exceeds tau. Not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_gate_diag``.

Source: extras/revision/x10_gate_diag.json (X10, EXTRAS_PROTOCOL amendment 2026-10-11; the script
reproduced the frozen certification, the uncertified set and CS:PMRT = never-sleep on 160 / 160
seeds before writing), keys ``repro.ok``, ``classes.<i>.{class, family, d, certified,
withheld_units_eval, deferred_units_eval, unresolved}`` and ``edges.<i>.{class, rel, kpi, status,
h_beta, ub90, tau, h_gt_mean, h_gt_ci95, gt_label, gt_harm_beyond_tau}``. Cross-checked against FD
``certsafe.cert.CS:PMRT`` (uncertified set; UB90 and tau of every edge, 7 significant digits),
``certsafe.pmrt_bound_ptx_nbr_rlf`` and ``certsafe.eval.composition.CS:PMRT`` (rejected / withheld
units). Post hoc, descriptive, no verdict; Study 3 is unchanged.
"""

import math

from oran_causal.figures import entities as E
from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _taba as A

NAME = "tab_gate_diag"
X10 = ("posthoc_x9_x16", "revision", "x10_gate_diag.json")
SOURCES = [source_note(*X10), ST.SRC_FD]
REL = {"own": "own", "nbr": "neighbour", "far": "far"}
KPI = {"v": "v", "rlf": "RLF"}
ARM = "CS:PMRT"


def num(x) -> str:
    """Three significant digits, math minus, leading zero below 1 (0.0364, 20.8, 34.9, 112)."""
    if A.nan(x):
        return A.DASH
    if x == 0:
        return "0"
    s = f"{abs(x):.{max(0, 2 - math.floor(math.log10(abs(x))))}f}"
    return (A.MINUS if x < 0 else "") + s


# W21 (V7 A13): prot_min is a code name; the printed table names it as the body does ("protected
# share")
PLAIN = {"prot_min": "protected share"}


def class_label(f: str, d: int) -> str:
    lab = E.param_label(f)
    if f in PLAIN:
        lab = PLAIN[f] + lab[len(f) :]
    return lab.replace("_", r"\_") + ", " + ("$+1$" if d > 0 else "$-1$")


def check(t: dict):
    """The frozen certification as the paper's figure reads it (FD), and the X10 reproduction flag."""
    if (
        not t["repro"]["ok"]
        or t["repro"]["cs_pmrt_identical_to_never_sleep"]["identical_seeds"] != 160
    ):
        raise AssertionError("X10 reproduction flag not set")
    cs = ST.fd()["certsafe"]
    fz = cs["cert"][ARM]
    unc = sorted([c["family"], c["d"]] for c in t["classes"] if not c["certified"])
    if unc != sorted(fz["uncertified"]):
        raise AssertionError(f"uncertified set {unc} != FD {fz['uncertified']}")
    for e in t["edges"]:
        fe = next(
            x for x in fz["classes"][e["class"]]["edges"] if x[0] == e["rel"] and x[1] == e["kpi"]
        )
        if fe[2] != e["status"] or not math.isclose(fe[4], e["tau"], rel_tol=1e-6):
            raise AssertionError(f"{e['class']} {e['rel']} {e['kpi']}: status / tau differ from FD")
        if e["status"] != "DECLARED" and not math.isclose(fe[3], e["ub90"], rel_tol=1e-6):
            raise AssertionError(f"{e['class']} {e['rel']} {e['kpi']}: UB90 differs from FD")
    pb = cs["pmrt_bound_ptx_nbr_rlf"]
    e = next(x for x in t["edges"] if (x["class"], x["rel"], x["kpi"]) == ("ptx|1", "nbr", "rlf"))
    if not (
        math.isclose(pb["ub90_defer_up"], e["ub90"], rel_tol=1e-6)
        and math.isclose(pb["tau"], e["tau"], rel_tol=1e-6)
    ):
        raise AssertionError("ptx -> neighbour RLF bound differs from FD pmrt_bound_ptx_nbr_rlf")
    comp = cs["eval"]["composition"][ARM]
    for c in t["classes"]:
        want = comp["withheld"][c["family"]] if not c["certified"] else 0
        if (
            c["withheld_units_eval"] != want
            or c["deferred_units_eval"] != comp["defer"][c["family"]]
        ):
            raise AssertionError(f"{c['class']}: units differ from FD composition")


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X10)
    check(t)
    out = []
    for c in t["classes"]:
        unres = [e for e in t["edges"] if e["class"] == c["class"] and e["status"] == "UNRESOLVED"]
        if [[e["rel"], e["kpi"]] for e in unres] != c["unresolved"]:
            raise AssertionError(f"{c['class']}: unresolved edges disagree")
        base = {
            "class": c["class"],
            "family": c["family"],
            "d": c["d"],
            "certified": c["certified"],
            "units_withheld": c["withheld_units_eval"],
            "units_rejected": c["deferred_units_eval"],
        }
        if not unres:
            out.append(dict(base, rel=None, kpi=None))
        for e in unres:
            lo, hi = e["h_gt_ci95"]
            out.append(
                dict(
                    base,
                    rel=e["rel"],
                    kpi=e["kpi"],
                    h_beta=e["h_beta"],
                    ub90=e["ub90"],
                    tau=e["tau"],
                    h_gt=e["h_gt_mean"],
                    h_gt_lo=lo,
                    h_gt_hi=hi,
                    gt_label=e["gt_label"],
                    gt_harm_beyond_tau=e["gt_harm_beyond_tau"],
                )
            )
    # largest class first: the withheld / rejected volume decides the referee's behaviour
    order = sorted(
        {r["class"] for r in out},
        key=lambda k: (
            -max(r["units_withheld"] + r["units_rejected"] for r in out if r["class"] == k)
        ),
    )
    out.sort(key=lambda r: order.index(r["class"]))
    return out, {}


def latex(rs: list[dict], meta: dict) -> str:
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc analysis X10: descriptive, no verdict; Study 3 unchanged.",
            "h = -d (harm of a rejection); UB90 = h x beta + 1.28 SE (PMRT design slope); an edge is certified iff UB90 <= tau.",
            "GT: Study 2 knockout ground truth (40 episodes, 95 % episode-cluster CI), harm of a rejection = h x GT mean.",
            "Units: Study 3 evaluation, 160 seeds; withheld = rejections the referee did not make (class uncertified).",
        ],
        module=NAME,
    )
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l r r r r l c@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{PMRT bound} & \multicolumn{3}{c}{Ground truth} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Uncertified guard edge & $h\beta$ & UB90 & $\varepsilon$ & $h\cdot$GT [95\,\% CI] & Label & Harm $>\varepsilon$ \\",
    ]
    ncol = 7
    seen = None
    for r in rs:
        if r["class"] != seen:
            seen = r["class"]
            if r["certified"]:
                vol = f"certified; {A.thousands(r['units_rejected'])} units rejected"
            else:
                vol = f"uncertified; {A.thousands(r['units_withheld'])} units withheld"
            out += [r"\midrule", A.group_row(f"{class_label(r['family'], r['d'])} ({vol})", ncol)]
        if r["rel"] is None:
            out.append(rf"\quad none & \multicolumn{{{ncol - 1}}}{{l}}{{--}} \\")
            continue
        edge = f"{REL[r['rel']]} {KPI[r['kpi']]}"
        gt = rf"{num(r['h_gt'])} [{num(r['h_gt_lo'])}, {num(r['h_gt_hi'])}]"
        out.append(
            rf"\quad {edge} & {num(r['h_beta'])} & {num(r['ub90'])} & {num(r['tau'])} & {gt} & "
            rf"{r['gt_label']} & {'yes' if r['gt_harm_beyond_tau'] else 'no'} \\"
        )
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

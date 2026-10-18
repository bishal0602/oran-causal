"""Post hoc analysis X11, ground-truth sensitivity (Appendix C; booktabs LaTeX + csv): the knockout
labels of Studies 1 and 2 re-derived with the practical-effect threshold delta halved and doubled,
the protected-violation floor halved and doubled, and the bootstrap CI at 90 % and 99 % (one factor
at a time), with the label counts and the PMRT map's precision (INDET counted wrong / right as
bounds), recall and sign accuracy against each label set; Granger + BY's precision as the one
comparator column. Every other method's scores are in the csv only. Not a figure (no SPEC): run ``uv
run python -m oran_causal.figures.figs.tab_appC_gt_sensitivity``.

Source: extras/revision/x11_gt_sensitivity.json (X11, EXTRAS_PROTOCOL amendment 2026-10-11; the
script reproduced every stored GT cell and every method's stored scores at the frozen setting before
writing), keys ``studies.<S1|S2>.reproduced``, ``.variants.<v>.{rel_frac, floor_pv, ci_level,
delta.pv, counts, methods.<code>.{precision, precision_indet_wrong, precision_indet_right, recall,
recall_indet_wrong, recall_indet_right, sign_acc, n_declared, ...}}``. PMRT = loadsp_c+wby1s (Study
1, pooled 600 episodes) and the MG:PMRT map (Study 2, pooled DISC; the same 22 declarations as
loadsp_c+wby1s, asserted). Granger + BY = granger_by (Study 1) and MG:granger_by (Study 2). Post
hoc, descriptive, no verdict; Studies 1 and 2 are unchanged.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_gt_sensitivity"
X11 = ("posthoc_x9_x16", "revision", "x11_gt_sensitivity.json")
SOURCES = [source_note(*X11)]
VARIANTS = [
    ("frozen", "Rule used"),
    ("delta_x0.5", r"$\delta \times 0.5$"),
    ("delta_x2", r"$\delta \times 2$"),
    ("floor_x0.5", r"Floor $\times 0.5$"),
    ("floor_x2", r"Floor $\times 2$"),
    ("ci90", r"90\,\% CI"),
    ("ci99", r"99\,\% CI"),
]
STUDIES = [
    ("S1", "Randomised logs, 600 episodes pooled", "loadsp_c+wby1s", "granger_by"),
    ("S2", "Confounded logs, discovery episodes pooled", "MG:PMRT", "MG:granger_by"),
]
KEYS = (
    "precision",
    "precision_indet_wrong",
    "precision_indet_right",
    "recall",
    "recall_indet_wrong",
    "recall_indet_right",
    "sign_acc",
    "n_declared",
    "tp",
    "fp",
    "fn",
    "indet_declared",
    "indet_undeclared",
)


def check(t: dict) -> None:
    """Frozen values the paper states, and the identity of the Study 2 PMRT rows."""
    s1, s2 = t["studies"]["S1"], t["studies"]["S2"]
    if not (s1["reproduced"] and s2["reproduced"]):
        raise AssertionError("X11 reproduction flag not set")
    f1, f2 = s1["variants"]["frozen"], s2["variants"]["frozen"]
    p = f1["methods"]["loadsp_c+wby1s"]
    got = (
        round(p["precision"], 2),
        round(p["recall"], 2),
        tuple(f1["counts"][k] for k in ("TRUE", "NULL", "INDET")),
    )
    if got != (0.92, 0.79, (29, 21, 10)):
        raise AssertionError(f"Study 1 frozen PMRT / labels not as the paper: {got}")
    m = f2["methods"]["MG:PMRT"]
    if (m["n_true_right"], m["n_true_wrong"], m["n_null"], m["n_indet"]) != (18, 0, 3, 1):
        raise AssertionError("Study 2 PMRT map counts not 18 / 0 / 3 / 1")
    if tuple(f2["counts"][k] for k in ("TRUE", "NULL", "INDET")) != (29, 24, 7):
        raise AssertionError("Study 2 frozen labels not 29 / 24 / 7")
    for v, _ in VARIANTS:
        a, b = (
            s2["variants"][v]["methods"]["MG:PMRT"],
            s2["variants"][v]["methods"]["loadsp_c+wby1s"],
        )
        if any(a[k] != b[k] for k in KEYS):
            raise AssertionError(f"Study 2 MG:PMRT and loadsp_c+wby1s differ under {v}")


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X11)
    check(t)
    out = []
    for st, _, _, _ in STUDIES:
        for v, _ in VARIANTS:
            o = t["studies"][st]["variants"][v]
            for code, m in o["methods"].items():
                out.append(
                    {
                        "study": st,
                        "variant": v,
                        "rel_frac": o["rel_frac"],
                        "floor_pv": o["floor_pv"],
                        "ci_level": o["ci_level"],
                        "delta_pv": o["delta"]["pv"],
                        "gt_true": o["counts"]["TRUE"],
                        "gt_null": o["counts"]["NULL"],
                        "gt_indet": o["counts"]["INDET"],
                        "method_code": code,
                        "method": A.M.name(code) if A.has_name(code) else code,
                        "kind": m["kind"],
                        **{k: m.get(k) for k in KEYS},
                    }
                )
    return out, t


def latex(rs: list[dict]) -> str:
    by = {(r["study"], r["variant"], r["method_code"]): r for r in rs}
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc analysis X11: descriptive, no verdict; Studies 1 and 2 unchanged.",
            "One factor at a time around the frozen rule (delta = 5 % of the KPI's largest |mean|, protected-violation floor"
            " 0.5 UE-s, 95 % episode-cluster bootstrap CI); the 90 / 99 % CIs reuse the frozen bootstrap draws.",
            "Precision [INDET wrong, right]: a declared indeterminate edge counted false / true; recall bounds in the csv.",
            "Every method's scores are in the csv.",
        ],
        module=NAME,
    )
    ncol = 8
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l c c c c c c c@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Ground-truth labels} & \multicolumn{3}{c}{PMRT} & Granger + BY \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
        r"Variant & True & Null & Indet. & Precision [bounds] & Recall & Sign acc. & precision \\",
    ]
    for st, title, pm, gb in STUDIES:
        out += [r"\midrule", A.group_row(title, ncol)]
        for v, lab in VARIANTS:
            p, g = by[(st, v, pm)], by[(st, v, gb)]
            out.append(
                rf"\quad {lab} & {p['gt_true']} & {p['gt_null']} & {p['gt_indet']} & "
                rf"{A.prob(p['precision'], 2)} {A.ci(p['precision_indet_wrong'], p['precision_indet_right'], nd=2)}"
                rf" & {A.prob(p['recall'], 2)} & {A.prob(p['sign_acc'], 2)} & {A.prob(g['precision'], 2)} \\"
            )
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, _ = rows()
    return A.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

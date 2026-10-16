"""Post hoc analysis X13, relation attribution (Appendix C; booktabs LaTeX + csv): (a) the 20
far-relation hypotheses of Studies 1 and 2 by knockout label, per control-parameter family; (b) the
neighbour and far declarations of PMRT and Granger + BY by the label of the declared edge, and how
many are own-attributable (declared edge null or indeterminate while the same family and KPI's own
edge is true). Not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_appC_attribution``.

Source: extras/revision/x13_attribution.json (X13, EXTRAS_PROTOCOL amendment 2026-10-11; the script
reproduced every stored declaration count before writing), keys ``studies.<S1|S2>.reproduced``,
``.far_hypotheses_by_gt_label.{all_20, by_family}``, ``.methods.<code>.slices.pooled`` and
``.methods.<code>.by_slice_kind.{s60, s120, s300}`` (counts ``<rel>_declared``,
``<rel>_gt_<TRUE|NULL|INDET>``, ``<rel>_own_attributable`` for rel in nbr, far). Allowed
re-aggregation: the Study 1 slice row sums the 17 slices (10 x 60, 5 x 120, 2 x 300 episodes). PMRT
= loadsp_c+wby1s (Study 1) and the MG:PMRT map (Study 2); Granger + BY = granger_by and
MG:granger_by. Post hoc, descriptive, no verdict; Studies 1 and 2 are unchanged.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_attribution"
X13 = ("posthoc_x9_x16", "revision", "x13_attribution.json")
SOURCES = [source_note(*X13)]
FAMILIES = [
    ("carrier", "carrier (ES)"),
    ("sleep", "sleep (ES)"),
    ("ptx", "ptx (Power ES)"),
    ("prot_min", r"prot\_min (Slice Guarantee)"),
]
LABELS = ("TRUE", "NULL", "INDET")
ROWS = [
    ("S1", "pooled", "Study 1, pooled 600 episodes"),
    ("S1", "slices", "Study 1, sum over 17 subsamples"),
    ("S2", "pooled", "Study 2, pooled discovery episodes"),
]
METHODS = {
    "S1": [("loadsp_c+wby1s", "PMRT"), ("granger_by", "Granger + BY")],
    "S2": [("MG:PMRT", "PMRT"), ("MG:granger_by", "Granger + BY")],
}
RELS = (("nbr", "Neighbour"), ("far", "Far"))


def counts(m: dict, which: str) -> dict:
    if which == "pooled":
        return m["slices"]["pooled"]
    tot: dict = {}
    for k in ("s60", "s120", "s300"):
        for a, b in m["by_slice_kind"][k].items():
            tot[a] = tot.get(a, 0) + b
    if tot["n_slices"] != 17:
        raise AssertionError(f"expected 17 slices, got {tot['n_slices']}")
    return tot


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X13)
    for st in ("S1", "S2"):
        if not t["studies"][st]["reproduced"]:
            raise AssertionError(f"X13 {st} reproduction flag not set")
    far = {st: t["studies"][st]["far_hypotheses_by_gt_label"] for st in ("S1", "S2")}
    if (
        far["S1"]["all_20"].get("TRUE"),
        sum(far["S1"]["by_family"][f].get("TRUE", 0) for f in ("carrier", "sleep", "ptx")),
    ) != (8, 8):
        raise AssertionError("Study 1 far TRUE count changed (audit: 8 of 15 without prot_min)")
    out = []
    for st in ("S1", "S2"):
        for f, _ in [*FAMILIES, ("all", "")]:
            c = far[st]["all_20"] if f == "all" else far[st]["by_family"][f]
            out.append(
                {
                    "table": "far_hypotheses",
                    "study": st,
                    "family": f,
                    **{f"gt_{s.lower()}": c.get(s, 0) for s in LABELS},
                }
            )
    for st, which, _ in ROWS:
        for code, name in METHODS[st]:
            c = counts(t["studies"][st]["methods"][code], which)
            r = {
                "table": "declarations",
                "study": st,
                "scope": which,
                "method_code": code,
                "method": name,
            }
            for rel, _ in RELS:
                r[f"{rel}_declared"] = c.get(f"{rel}_declared", 0)
                for s in LABELS:
                    r[f"{rel}_gt_{s.lower()}"] = c.get(f"{rel}_gt_{s}", 0)
                r[f"{rel}_own_attributable"] = c.get(f"{rel}_own_attributable", 0)
                if r[f"{rel}_declared"] != sum(r[f"{rel}_gt_{s.lower()}"] for s in LABELS):
                    raise AssertionError(f"{st} {which} {code} {rel}: labels do not add up")
            out.append(r)
    return out, t


def latex(rs: list[dict]) -> str:
    far = {(r["study"], r["family"]): r for r in rs if r["table"] == "far_hypotheses"}
    dec = {(r["study"], r["scope"], r["method"]): r for r in rs if r["table"] == "declarations"}
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc analysis X13: descriptive, no verdict; Studies 1 and 2 unchanged.",
            "Own-attributable = declared neighbour / far edge null or indeterminate while the same family and KPI's own edge"
            " is true (a flag, not proof of misattribution).",
            "Per-edge rows (all methods, all slices) in data/records/posthoc_x9_x16/revision/"
            "x13_attribution.csv.",
        ],
        module=NAME,
    )
    # (a) far hypotheses
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l c c c c c c@{}}",
        r"\toprule",
        r"\multicolumn{7}{@{}l}{(a) Far-relation hypotheses by ground-truth label} \\",
        r"\midrule",
        r" & \multicolumn{3}{c}{Study 1} & \multicolumn{3}{c}{Study 2} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Parameter (writer) & True & Null & Indet. & True & Null & Indet. \\",
        r"\midrule",
    ]
    for f, lab in [*FAMILIES, ("all", "All 20")]:
        if f == "all":
            out.append(r"\midrule")
        cells = " & ".join(
            str(far[(st, f)][f"gt_{s.lower()}"]) for st in ("S1", "S2") for s in LABELS
        )
        out.append(rf"{lab} & {cells} \\")
    out += [r"\bottomrule", r"\end{tabular}", "", r"\par\medskip", ""]
    # (b) declarations
    out += [
        r"\begin{tabular}{@{}l c c c c c c c c c c@{}}",
        r"\toprule",
        r"\multicolumn{11}{@{}l}{(b) Neighbour and far declarations by the label of the declared edge} \\",
        r"\midrule",
        r" & \multicolumn{5}{c}{Neighbour} & \multicolumn{5}{c}{Far} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r"Method & Decl. & True & Null & Indet. & Own-attr. & Decl. & True & Null & Indet. & Own-attr. \\",
    ]
    for st, which, title in ROWS:
        out += [r"\midrule", A.group_row(title, 11)]
        for _, name in METHODS[st]:
            r = dec[(st, which, name)]
            cells = " & ".join(
                str(r[f"{rel}_{k}"])
                for rel, _ in RELS
                for k in ("declared", "gt_true", "gt_null", "gt_indet", "own_attributable")
            )
            out.append(rf"\quad {name} & {cells} \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, _ = rows()
    return A.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

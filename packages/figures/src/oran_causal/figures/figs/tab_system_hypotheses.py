"""The hypothesis space of the system-level simulator: 4 control-parameter families x 3 relations x
5 KPIs = 60 candidate edges, each with its Study 1 knockout ground-truth status (TRUE with the sign
of the effect of a +1 parameter step, NULL, INDET). Booktabs LaTeX + csv; not a figure (no SPEC):
run ``uv run python -m oran_causal.figures.figs.tab_system_hypotheses``.

Source: ``v4_analysis.json`` gt.cells (via ``_studies.v4_gt_cells()``), status and sign as stored.
The status counts are checked against FD ``v4.gt.counts`` (re-aggregation: counting cells). Families
are labelled with their plain name and writer xApp (``entities.param_plain_label``), relations and KPIs with the
TERMINOLOGY s.4 paper names. The Study 2 ground truth (confounded logs) has its own labels and is
not shown here.
"""

from collections import Counter

from oran_causal.figures import entities as E
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _tabm as TM

NAME = "tab_system_hypotheses"
SOURCES = [ST.SRC_V4, ST.SRC_FD]
FAMILIES = ["carrier", "sleep", "ptx", "prot_min"]  # TERMINOLOGY s.3 order
RELATIONS = {"own": "own", "nbr": "neighbour", "far": "far"}  # TERMINOLOGY s.4
KPIS = {
    "pv": ("protected-slice", "violations"),
    "v": ("service", "violations"),
    "rlf": ("radio-link", "failures"),
    "e": ("energy", ""),
    "load": ("served", "load"),
}  # TERMINOLOGY s.4 (entities.KPIS names)


def _label(c: dict) -> str:
    if c["status"] == "TRUE":
        return {1: "true $+$", -1: "true $-$"}[c["sign"]]
    if c["status"] in ("NULL", "INDET"):
        return {"NULL": "null", "INDET": "indeterminate"}[
            c["status"]
        ]  # prose words (TERMINOLOGY GT row)
    raise AssertionError(f"unknown ground-truth status {c['status']}")


def rows() -> list[dict]:
    g = ST.v4_gt_cells()
    want = {(f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS}
    if set(g) != want:
        raise AssertionError(f"hypothesis space differs from 4 x 3 x 5: {sorted(set(g) ^ want)}")
    counts = Counter(c["status"] for c in g.values())
    if dict(counts) != ST.fd()["v4"]["gt"]["counts"]:
        raise AssertionError(
            f"ground-truth counts {dict(counts)} differ from FD {ST.fd()['v4']['gt']['counts']}"
        )
    for k, (a, b) in KPIS.items():
        if E.KPIS[k].name != f"{a} {b}".strip():
            raise AssertionError(f"KPI name of {k} changed: {E.KPIS[k].name}")
    out = []
    for f in FAMILIES:
        for r in RELATIONS:
            row = {"family": f, "family_label": E.param_plain_label(f), "relation": RELATIONS[r]}
            for k in KPIS:
                c = g[(f, r, k)]
                row[k] = _label(c)
                row[f"{k}_status"], row[f"{k}_sign"] = c["status"], c["sign"]
            out.append(row)
    return out


def latex(rs: list[dict]) -> list[str]:
    counts = ST.fd()["v4"]["gt"]["counts"]
    out = TM.header(
        NAME,
        SOURCES,
        [
            "Knockout ground truth of Study 1 (randomised logs); sign = accept-minus-reject effect of a "
            "request that raises the parameter (60 s mode contrast); "
            + ", ".join(f"{k} {v}" for k, v in counts.items())
            + "."
        ],
    )
    head = [rf"\shortstack{{{a}\\{b}}}" if b else a for a, b in KPIS.values()]
    out += [
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l l c c c c c@{}}",
        r"\toprule",
        r"\shortstack[l]{Parameter\\(writer)} & Relation & " + " & ".join(head) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rs):
        j = i % len(RELATIONS)  # row within the family block
        if i and j == 0:
            out.append(r"\midrule")
        param, _, writer = r["family_label"].partition(
            " ("
        )  # 'protected share (Slice Guarantee)' over two rows
        fam = {0: TM.tex(param), 1: f"({writer}"}.get(j, "")
        out.append(rf"{fam} & {r['relation']} & " + " & ".join(r[k] for k in KPIS) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


def write() -> list[str]:
    rs = rows()
    return TM.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

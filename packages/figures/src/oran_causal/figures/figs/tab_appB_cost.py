"""Appendix B bridging-analysis cost (booktabs LaTeX + csv): per named method, units (method x data
set), infeasible units, total CPU time in Kaggle-reference core-hours and the largest CPU-s of one
unit. Not a figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_appB_cost``.

Source: exp_b_costs.json (EXP_B s.9 Cost; EXP_B_COSTS.md), keys ``arms.<arm>.{n, infeasible,
ref_cpu_s, max_ref_cpu_s}``, ``total_ref_core_h`` and ``budget_cpu_s``. Allowed re-aggregation:
core-h = ref_cpu_s / 3600; the two arms without a paper name (pc_eq, mscr_eq) are summed into one
unnamed row so the column adds up to the stored total (checked against ``total_ref_core_h``).
Infeasible units are counted at the CPU budget (EXP_B s.6).
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import _taba as A

NAME = "tab_appB_cost"
SRC_COSTS = source_note(*T.BRIDGE, "exp_b_costs.json")
SOURCES = [SRC_COSTS]


def rows() -> tuple[list[dict], dict]:
    c = load_json(*T.BRIDGE, "exp_b_costs.json")
    out, other = (
        [],
        {
            "arm": "other",
            "method": "further arms without a paper name",
            "units": 0,
            "infeasible": 0,
            "ref_core_h": 0.0,
            "max_ref_cpu_s": 0.0,
            "members": [],
        },
    )
    for arm, a in c["arms"].items():
        r = {
            "arm": arm,
            "method": A.M.name(arm) if A.has_name(arm) else "",
            "units": a["n"],
            "infeasible": a["infeasible"],
            "ref_core_h": a["ref_cpu_s"] / 3600,
            "max_ref_cpu_s": a["max_ref_cpu_s"],
        }
        if A.has_name(arm):
            out.append(r)
        else:
            other["units"] += a["n"]
            other["infeasible"] += a["infeasible"]
            other["ref_core_h"] += r["ref_core_h"]
            other["max_ref_cpu_s"] = max(other["max_ref_cpu_s"], a["max_ref_cpu_s"])
            other["members"].append(arm)
    total = sum(r["ref_core_h"] for r in out) + other["ref_core_h"]
    if abs(total - c["total_ref_core_h"]) > 1e-6:
        raise AssertionError(f"core-h sum {total} != stored total {c['total_ref_core_h']}")
    other["members"] = " ".join(other["members"])
    meta = {
        "total": c["total_ref_core_h"],
        "budget": c["budget_cpu_s"],
        "other": other,
        "units": sum(r["units"] for r in out) + other["units"],
        "infeasible": sum(r["infeasible"] for r in out) + other["infeasible"],
    }
    return out, meta


def latex(rs: list[dict], meta: dict) -> str:
    ncol = 5
    o = meta["other"]
    out = A.header(
        NAME,
        SOURCES,
        [
            "Bridging analysis cost: post hoc, descriptive (EXP_B s.9 Cost). CPU time in Kaggle-reference core-hours"
            " (other hosts converted with the stored speed factors); infeasible units counted at the budget of"
            f" {meta['budget']:.0f} CPU-s per unit.",
            f"Unnamed row = the arms without a paper name ({o['members']}), summed so the total adds up.",
        ],
    )
    out += [
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}l r r r r@{}}",
        r"\toprule",
        r"Method & Units & Infeasible & Core-h & Max CPU-s \\",
    ]

    def line(label, r):
        return (
            rf"{label} & {A.thousands(r['units'])} & {r['infeasible']} & {r['ref_core_h']:.2f} & "
            rf"{A.sig3(r['max_ref_cpu_s'])} \\"
        )

    by = {r["arm"]: r for r in rs}
    for fam, codes in A.blocks(list(by)):
        out += [r"\midrule", A.group_row(fam, ncol)]
        out += [line(rf"\quad {A.short(code)}", by[code]) for code in codes]
    out += [
        r"\midrule",
        line(r"Further arms not in the method set (2)", o),
        r"\midrule",
        rf"Total & {A.thousands(meta['units'])} & {meta['infeasible']} & {meta['total']:.1f} & \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), [*rs, meta["other"]])


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

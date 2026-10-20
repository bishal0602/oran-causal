"""Appendix B bridging-analysis headline (booktabs LaTeX + csv): every named Study 4 method on the
simulator logs of Studies 1-3, by slice (60 / 120 / 300 episodes, pooled), with recall of the
GT-TRUE edges, the GT-NULL rejection rate, the placebo-column rejection rate (60-episode slices) and
the rejection rate on the placebo logs; the frozen PMRT row is the reference. Not a figure (no
SPEC): run ``uv run python -m oran_causal.figures.figs.tab_appB_bridge``.

Source: exp_b_tables.json (bridging protocol s.9), keys ``cells["<arm>|<slice>|<rule>"]`` (rec / null / plac /
plac_log rate and n_datasets) and ``v4_frozen_reference`` (recall and null_rate per slice; placebo
logs = its placebo null_rate). The rule is each method's primary declaration rule as in the bridging
protocol's s.9 headline: BY (q .05) for the p-value methods, the leave-one-slice-out placebo threshold tau for
the threshold-only methods (in-sample for pooled and placebo logs). Allowed re-aggregation: the
coverage column sums ``n_datasets`` over the four measure slice kinds (of 72). Arms without a paper
name (pc_eq, mscr_eq) are left out; PC is the paper's PC (native covariates).
"""

from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import _taba as A

NAME = "tab_appB_bridge"
SOURCES = [T.SRC_BRIDGE_TABLES]
SLICES = ["60", "120", "300", "pooled"]
REF = "PMRT"  # frozen PMRT (Studies 1-3), the reference row


def rows() -> list[dict]:
    t = T.bridge_tables()
    cells = t["cells"]
    arms = sorted({k.split("|")[0] for k in cells})
    full = sum(cells[f"pmrt_nl_eq|{s}|by"]["n_datasets"] for s in SLICES)
    out = []
    ref = t["v4_frozen_reference"]
    r = {
        "arm": "frozen_v4_primary",
        "method": f"{A.M.name(REF)} of the simulator (reference)",
        "rule": "",
        "datasets": "",
        "of": full,
    }
    for s in SLICES:
        r[f"recall_{s}"] = ref[s]["recall"]
        r[f"gtnull_{s}"] = ref[s]["null_rate"]
    r["placebo_col_60"] = None
    r["placebo_logs"] = ref["placebo"]["null_rate"]
    out.append(r)
    for arm in arms:
        if not A.has_name(arm):
            continue
        rule = "by" if f"{arm}|60|by" in cells else "tau"
        c = {s: cells[f"{arm}|{s}|{rule}"] for s in [*SLICES, "placebo"]}
        ds = sum(c[s]["n_datasets"] for s in SLICES)
        r = {"arm": arm, "method": A.M.name(arm), "rule": rule, "datasets": ds, "of": full}
        for s in SLICES:
            ok = c[s]["n_datasets"] > 0 and "rec" in c[s]
            r[f"recall_{s}"] = c[s]["rec"]["rate"] if ok else None
            r[f"gtnull_{s}"] = c[s]["null"]["rate"] if ok else None
            r[f"gtnull_n_{s}"] = c[s]["null"]["n"] if ok else 0
        r["placebo_col_60"] = c["60"]["plac"]["rate"] if c["60"]["n_datasets"] else None
        r["placebo_logs"] = c["placebo"]["plac_log"]["rate"] if c["placebo"]["n_datasets"] else None
        out.append(r)
    return out


def row_label(code: str, partial: bool) -> str:
    """Table label of a method, marked 'partial' when it covers only some data sets."""
    lab = (
        A.short(code)
        .replace(r"n $\le$ 1000", r"$n \le 1000$")
        .replace("R3 covariates", "confounder covariates")
    )
    if not partial:
        return lab
    return lab[:-1] + ", partial)" if lab.endswith(")") else lab + " (partial)"


def latex(rs: list[dict]) -> str:
    ncol = 1 + 1 + 1 + 2 * len(SLICES) + 2
    out = A.header(
        NAME,
        SOURCES,
        [
            "Bridging analysis on the simulator logs of Studies 1-3: post hoc, descriptive, no verdict.",
            "Rule: BY = Benjamini-Yekutieli at q .05; Plac. = placebo-calibrated threshold (leave one slice out; in-sample"
            " for pooled and the placebo logs). Rates are shares of hypotheses; slices 60 / 120 / 300 episodes and"
            " pooled (all 600). GT-NULL is an equivalence label, not a sharp null. Data sets = measure data sets with a"
            " record (of 72); dagger = partial coverage (MSCR: n <= 1000 only; CDD O-RAN discovery: 12 units over the"
            " CPU budget), not comparable with full-coverage rows. Slice-cluster CIs (2-10 clusters) stay in the"
            " source, not here. Left out (no paper name): pc_eq, mscr_eq.",
            "The frozen PMRT row is the reference (Studies 1-3 primary), not a competitor.",
        ],
    )
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1pt}",
        r"\begin{tabular}{@{}l c c" + " c" * len(SLICES) + " c" * len(SLICES) + r" c c@{}}",
        r"\toprule",
        r" & & Data & \multicolumn{4}{c}{Recall} & \multicolumn{4}{c}{Null rate} & Placebo & Placebo \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        r"Method & Rule & sets & 60 & 120 & 300 & All & 60 & 120 & 300 & All & column & logs \\",
    ]
    ref, rest = rs[0], rs[1:]

    def nums(r):
        return " & ".join(
            [A.prob(r[f"recall_{s}"], 2) for s in SLICES]
            + [A.prob(r[f"gtnull_{s}"], 2) for s in SLICES]
            + [
                A.prob(r["placebo_col_60"], 2) if r["placebo_col_60"] is not None else "",
                A.prob(r["placebo_logs"], 2),
            ]
        )

    out += [r"\midrule", rf"{ref['method']} & & & {nums(ref)} \\"]
    by = {r["arm"]: r for r in rest}
    rule_tex = {
        "by": "BY",
        "tau": "Plac.",
    }  # not tau: tau names the estimands in the paper (V3, 2026-10-12)
    for fam, codes in A.blocks(list(by)):
        out += [r"\midrule", A.group_row(fam, ncol)]
        for code in codes:
            r = by[code]
            out.append(
                rf"\quad {row_label(code, r['datasets'] < r['of'])} & {rule_tex[r['rule']]} & {r['datasets']} & {nums(r)} \\"
            )
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs = rows()
    return A.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

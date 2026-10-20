"""Post hoc check X2 (Appendix C.2; booktabs LaTeX + csv): the compact table next to
fig_appC_x2_dose. Pooled truth-null
raw rejection rate (p <= .05) of the R2 design in E1-E3 (n 1000 and 4000, 40 measurement seeds per
design) per method, against the dither width at 20 setpoint blocks and the number of setpoint blocks
at width 0.10, with the cell-validity mark. Not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_appC_x2``.

Source: XR ``extras_tables.json``, ``pooled`` rows with experiment X2, regime R2, key ``null_raw``
(rate, hits, n, seed-cluster 95 % CI, validity). Raw rates, as the figure and Study 4; the csv also carries the BY-declared rates (``null_decl``). Rows
only select stored values. NOTEARS and SHAP-DAG declare by a threshold and give no p-values, so they
have no raw rate (csv only); pc_eq has no paper name. Design-adjusted partial correlation has no
rate at 5 setpoint blocks: the frozen arm stops on a rank-deficient source design there (XR Notes).
"""

from oran_causal.figures.figs import _appc as C
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_x2"
SOURCES = [C.SRC_XR]
ARMS = ["pmrt_nl_eq", "pmrt_eq", "pcorr_eq", "corr", "pcorr_native"]
NO_RAW = ["notears", "shap_dag"]  # threshold-declared: no p-values, csv only
DELTAS = [0.02, 0.05, 0.1, 0.2]  # at 20 setpoint blocks
BLOCKS = [5, 80]  # at width 0.10 (20 blocks is the 0.10 column)
COLS = [(d, 20) for d in DELTAS] + [(0.1, k) for k in BLOCKS]
MISSING_OK = {("pcorr_eq", (0.1, 5))}  # rank-deficient design: not testable


def rows(tab: dict) -> list[dict]:
    pool = C.pooled(tab, "X2", "R2")
    out = []
    for arm in ARMS + NO_RAW:
        for col in COLS:
            r = pool.get((arm, col))
            if r is None:
                raise AssertionError(f"X2: no pooled row for {arm} {col}")
            raw, decl = r["null_raw"], r["null_decl"]
            if arm in NO_RAW:
                assert raw is None, (arm, col)
            elif raw is None:
                assert (arm, col) in MISSING_OK, (arm, col)
            out.append(
                {
                    "arm": arm,
                    "width": col[0],
                    "blocks": col[1],
                    "declare": r["declare"],
                    "null_raw_rate": raw and raw["rate"],
                    "null_raw_lo": raw and raw["ci"][0],
                    "null_raw_hi": raw and raw["ci"][1],
                    "null_raw_hits": raw and raw["hits"],
                    "null_raw_n": raw and raw["n"],
                    "null_raw_validity": raw and raw["validity"],
                    "null_decl_rate": decl and decl["rate"],
                    "null_decl_validity": decl and decl["validity"],
                }
            )
    return out


def latex(tab: dict, rs: list[dict]) -> str:
    cell = {(r["arm"], (r["width"], r["blocks"])): r for r in rs}
    n = {r["null_raw_n"] for r in rs if r["null_raw_n"]}
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc check X2: dither width; exploratory, no verdict.",
            f"Pooled truth-null raw p <= .05 rejection rates, R2 of E1-E3, n 1000 and 4000, {'/'.join(map(str, sorted(n)))} candidate tests per"
            " design; mark = condition-validity label (* INVALID, unmarked VALID).",
            "The 0.10 column is the 20-block design (the Study 4 R2 design). -- : design-adjusted partial correlation is"
            " not testable at 5 setpoint blocks (rank-deficient design). NOTEARS / SHAP-DAG give no p-values (csv only).",
        ],
        module=NAME,
    )
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l c c c c c c@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Dither width (20 setpoint blocks)} & \multicolumn{2}{c}{Setpoint blocks"
        r" (width 0.10)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"Method & 0.02 & 0.05 & 0.10 & 0.20 & 5 & 80 \\",
        r"\midrule",
    ]
    for i, (fam, codes) in enumerate(A.blocks(ARMS)):
        if i:
            out.append(r"\addlinespace")
        out.append(A.group_row(fam, 7))
        for arm in codes:
            vals = []
            for col in COLS:
                r = cell[(arm, col)]
                vals.append(
                    A.DASH
                    if r["null_raw_rate"] is None
                    else A.prob(r["null_raw_rate"]) + A.MARK[r["null_raw_validity"]]
                )
            out.append(rf"\quad {A.short(arm)} & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    tab = C.extras()
    rs = rows(tab)
    return A.write(NAME, latex(tab, rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

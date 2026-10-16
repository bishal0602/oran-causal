"""Appendix A kappa sweep (booktabs LaTeX + csv): Study 4 recall of the true edges in R2 at n 1000
for the measurement- noise levels kappa .125, .25 (primary) and .5, per named method x environment.
Not a figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_appA_kappa``.

Source: report.json ``tables.V9_kappa_sweep`` (REPORT s.7 / V9), one record per arm x environment x
kappa (E4 at lam 1). Each cell is the record's mean recall with its validity mark as in the Appendix
A cell tables (* INVALID, ~ INCONCLUSIVE, unmarked VALID, ° no validity read). Recall is not
reported for an INVALID cell (power is descriptive among cells that are not INVALID, REPORT s.6), so
those cells read "--*". Rows only select records; nothing is recomputed. Arms without a paper name
(eq_min variants, pc_eq, mscr_eq) are left out. CDD O-RAN discovery and MSCR were run at the primary
kappa only; Granger variants exist in E3 only.
"""

from oran_causal.figures.data import load_json
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import _taba as A

NAME = "tab_appA_kappa"
SOURCES = [T.SRC_REPORT]
KAPPAS = [0.125, 0.25, 0.5]


def records() -> list[dict]:
    rep = load_json(*T.EVAL, "report.json")["tables"]["V9_kappa_sweep"]
    out = []
    for r in rep:
        if r["status"] != "ok" or r["regime"] != "R2" or r["n"] != 1000 or not A.has_name(r["arm"]):
            continue
        rec = r["recall"]
        out.append(
            {
                "arm": r["arm"],
                "method": A.M.name(r["arm"]),
                "world": r["world"],
                "lam": r["lam"],
                "kappa": r["kappa"],
                "seeds": rec["n"],
                "validity": r["validity"],
                "recall_mean": rec["mean"],
                "recall_lo": rec["ci"][0],
                "recall_hi": rec["ci"][1],
                "recall_shown": "" if r["validity"] == "INVALID" else rec["mean"],
            }
        )
    keys = [(x["arm"], x["world"], x["kappa"]) for x in out]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate arm x world x kappa records in V9")
    return out


def _cell(x: dict | None) -> str:
    if x is None:
        return ""
    val = A.DASH if x["validity"] == "INVALID" else A.prob(x["recall_mean"], 2)
    return val + A.MARK[x["validity"]]


def latex(rows: list[dict]) -> str:
    by = {(x["arm"], x["world"], x["kappa"]): x for x in rows}
    seeds = sorted({(x["kappa"], x["seeds"]) for x in rows})
    ncol = 2 + len(T.WORLDS)
    out = A.header(
        NAME,
        SOURCES,
        [
            "Secondary, descriptive Study 4 kappa sweep specified in the frozen protocol (PROTOCOL_A V9); no verdict.",
            "Entry: mean recall of the true edges, R2, n = 1000 (E4 at lam 1); mark = the condition's validity (* INVALID,"
            " ~ INCONCLUSIVE, unmarked VALID, ° no validity read); --* = INVALID condition, recall not reported;"
            " blank = not in the method's grid.",
            "Seeds per condition: " + ", ".join(f"kappa {k:g}: {s}" for k, s in seeds) + ".",
            "Left out (no paper name): eq_min variants, pc_eq, mscr_eq.",
        ],
    )
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l c" + " c" * len(T.WORLDS) + "@{}}",
        r"\toprule",
        r"Method & $\kappa$ & " + " & ".join(T.WORLDS) + r" \\",
    ]
    arms = sorted({x["arm"] for x in rows})
    for fam, codes in A.blocks(arms):
        out += [r"\midrule", A.group_row(fam, ncol)]
        for code in codes:
            ks = [k for k in KAPPAS if any((code, w, k) in by for w in T.WORLDS)]
            for i, k in enumerate(ks):
                label = A.short(code) if i == 0 else ""
                cells = " & ".join(_cell(by.get((code, w, k))) for w in T.WORLDS)
                out.append(rf"\quad {label} & {A.prob(k, 3).rstrip('0')} & {cells} \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rows = records()
    return A.write(NAME, latex(rows), rows)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

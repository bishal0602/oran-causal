"""Post hoc check X5 (Appendix C.5; booktabs LaTeX + csv): fresh re-run of the Study 4 E4 R3 cells
in which PMRT-Lin's null-edge rate had been labelled INVALID, plus the five adjacent E4 R3 cells.
Two stacked tabulars: (a) the declared primary rates (PMRT-Lin, one rate per re-run cell) with
Clopper-Pearson CIs, one-sided exact binomial p vs .05 and Holm-adjusted p; (b) every placebo and
confounded-null rate of PMRT-Lin and PMRT-Lin with R3 covariates in the eight cells. Not a figure
(no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_appC_x5``.

Source: extras/diag/diag_tables.json (DIAG s.X5): ``x5.primary[*].{lam, n, n_, hits, rate, rate_key,
cp_ci, p_binom_gt05, p_holm, outcome}`` and ``x5.cells[*].{arm, spec, lam, n, seeds, plac, conf}``
with each rate's ``{rate, hits, n, cp_ci, r30}`` (r30 = the cell-validity label, shown with the
Appendix A marks). Rows only select stored values; the primary rates are checked against the
matching x5.cells entries. Post hoc, exploratory, declared before the run, fresh seeds; the frozen
C3 result is unchanged.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_x5"
DIAG = ("posthoc_x5_x6", "diag", "diag_tables.json")
SOURCES = [source_note(*DIAG)]
ARMS = ["pmrt_eq", "pmrt_r3"]
KINDS = [("plac", "placebo"), ("conf", "confounded null")]
KEY = {"plac_raw": "plac", "conf_raw": "conf"}
SPEC = {"x5_fail": "Re-run conditions", "x5_adj": "Adjacent conditions"}


def _lam(x: float) -> str:
    if x == 0:
        return "0"
    return A.prob(x, 2).rstrip("0").rstrip(".") if x < 1 else f"{x:g}"


def cell_label(lam: float, n: int) -> str:
    return rf"$\lambda = {_lam(lam)}$, $n = {A.thousands(n)}$"


def rows(t: dict) -> tuple[list[dict], list[dict]]:
    x5 = t["x5"]
    cells = {(c["arm"], c["lam"], c["n"]): c for c in x5["cells"] if c["status"] == "ok"}
    prim = []
    for p in x5["primary"]:
        c = cells[("pmrt_eq", p["lam"], p["n"])][KEY[p["rate_key"]]]
        if (c["hits"], c["n"]) != (p["hits"], p["n_"]):
            raise AssertionError(
                f"primary {p['lam']} {p['n']}: {p['hits']}/{p['n_']} != cells {c['hits']}/{c['n']}"
            )
        prim.append(
            {
                "arm": "pmrt_eq",
                "lam": p["lam"],
                "n": p["n"],
                "null": KEY[p["rate_key"]],
                "hits": p["hits"],
                "datasets": p["n_"],
                "rate": p["rate"],
                "cp_lo": p["cp_ci"][0],
                "cp_hi": p["cp_ci"][1],
                "p_one_sided": p["p_binom_gt05"],
                "p_holm": p["p_holm"],
                "outcome": p["outcome"],
            }
        )
    allr = []
    order = [(c["spec"], c["lam"], c["n"]) for c in x5["cells"]]
    seen = []
    for spec, lam, n in order:
        if (spec, lam, n) in seen:
            continue
        seen.append((spec, lam, n))
        r = {"spec": spec, "lam": lam, "n": n}
        for arm in ARMS:
            c = cells[(arm, lam, n)]
            r["datasets"] = c["seeds"]
            for k, _ in KINDS:
                v = c[k]
                r.update(
                    {
                        f"{arm}_{k}_rate": v["rate"],
                        f"{arm}_{k}_hits": v["hits"],
                        f"{arm}_{k}_n": v["n"],
                        f"{arm}_{k}_cp_lo": v["cp_ci"][0],
                        f"{arm}_{k}_cp_hi": v["cp_ci"][1],
                        f"{arm}_{k}_label": v["r30"],
                    }
                )
        allr.append(r)
    return prim, allr


def latex(prim: list[dict], allr: list[dict]) -> str:
    cp_hi = max(r[f"{a}_{k}_cp_hi"] for r in allr for a in ARMS for k, _ in KINDS)
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc check X5: exploratory, declared before the run, fresh seeds; does not change C3 (PMRT-Lin under R3"
            " stays as reported). Environment E4, regime R3, kappa .25.",
            "(a) declared primary rate per re-run condition (PMRT-Lin), 95 % Clopper-Pearson CI, one-sided exact binomial p vs"
            " .05, Holm over the three conditions; outcome = the declared label (DIAG s.X5).",
            "(b) raw p <= .05 rejection rates; mark = condition-validity label (* INVALID, ~ INCONCLUSIVE, unmarked VALID);"
            f" dagger = the primary rate of (a); largest 95 % CP upper bound in (b): {A.prob(cp_hi, 3)}.",
            "Secondary, descriptive, unadjusted for the adjacent conditions.",
        ],
        module=NAME,
    )
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l l r c c c c@{}}",
        r"\toprule",
        r"\multicolumn{7}{@{}l}{(a) Re-run conditions, PMRT-Lin: declared primary rate} \\",
        r"\midrule",
        r" & Null & Hits / & Rate & $p$ & $p$ & \\",
        r"Condition & edge & data sets & [95\,\% CI] & one-sided & Holm & Outcome \\",
        r"\midrule",
    ]
    kind = dict(KINDS)
    for p in prim:
        out.append(
            rf"{cell_label(p['lam'], p['n'])} & {kind[p['null']]} & {A.thousands(p['hits'])} / "
            rf"{A.thousands(p['datasets'])} & {A.prob(p['rate'])} {A.ci(p['cp_lo'], p['cp_hi'])} & "
            rf"{A.prob(p['p_one_sided'])} & {A.prob(p['p_holm'])} & {p['outcome'].lower()} \\"
        )
    out += [r"\bottomrule", r"\end{tabular}", "", r"\par\medskip", ""]
    prim_keys = {(p["lam"], p["n"], p["null"]) for p in prim}
    names = [A.M.name(a) for a in ARMS]
    out += [
        r"\begin{tabular}{@{}l r c c c c@{}}",
        r"\toprule",
        r"\multicolumn{6}{@{}l}{(b) Every rate in the re-run and adjacent conditions} \\",
        r"\midrule",
        r" & & \multicolumn{2}{c}{" + names[0] + r"} & \multicolumn{2}{c}{" + names[1] + r"} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Condition & Data sets & "
        + " & ".join(kind[k] for _ in ARMS for k, _ in KINDS)
        .replace("confounded null", "Confounded null")
        .replace("placebo", "Placebo")
        + r" \\",
    ]
    for spec in SPEC:
        out += [r"\midrule", A.group_row(SPEC[spec], 6)]
        for r in allr:
            if r["spec"] != spec:
                continue
            cells = []
            for a in ARMS:
                for k, _ in KINDS:
                    dag = (
                        r"$^{\dagger}$"
                        if a == "pmrt_eq" and (r["lam"], r["n"], k) in prim_keys
                        else ""
                    )
                    cells.append(A.prob(r[f"{a}_{k}_rate"]) + A.MARK[r[f"{a}_{k}_label"]] + dag)
            out.append(
                rf"\quad {cell_label(r['lam'], r['n'])} & {A.thousands(r['datasets'])} & "
                + " & ".join(cells)
                + r" \\"
            )
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    t = load_json(*DIAG)
    prim, allr = rows(t)
    return A.write(
        NAME,
        latex(prim, allr),
        [dict(r, table="a") for r in prim] + [dict(r, table="b") for r in allr],
    )


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

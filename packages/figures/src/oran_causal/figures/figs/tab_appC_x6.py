"""Post hoc check X6 (Appendix C.6; booktabs LaTeX + csv): PMRT-GBM under a told dither twice the
true width, 2 x 2 ablation of the law used for centring the statistic and the law used for the
randomization redraws (each told / true), next to the exact design. Per arm: truth-null and placebo
raw rejection rates with 95 % seed-cluster bootstrap CIs and cell-validity marks, and the mean of
the declared first-order bias term. Not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_appC_x6``. The told-width grid is fig_appC_x6_width (not here).

Source: extras/diag/diag_tables.json (DIAG s.X6): ``x6.arms[*]`` with base ``pmrt_nl_eq`` and width
2 (the four centring x redraw arms) or centre ``exact`` (width 1): ``null_raw`` / ``plac_raw``
``{rate, hits, n, ci, validity}``, ``bias.null.{mean, ci}``, ``seeds``; ``x6.predictions.outcome``.
Rows only select stored values. Post hoc, exploratory, one cell (E1 R2, n 1000), fresh seeds; C1-C3
unchanged.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_x6"
DIAG = ("posthoc_x5_x6", "diag", "diag_tables.json")
SOURCES = [source_note(*DIAG)]
ARM = "pmrt_nl_eq"
ORDER = [("told", "told"), ("told", "true"), ("true", "told"), ("true", "true")]


def rows(t: dict) -> list[dict]:
    arms = [a for a in t["x6"]["arms"] if a["base"] == ARM]
    pick = {}
    for a in arms:
        if a["width"] == 2.0 and (a["centre"], a["redraw"]) in ORDER:
            pick[(a["centre"], a["redraw"])] = a
        elif a["centre"] == "exact":
            pick[("exact", "exact")] = a
    if len(pick) != 5:
        raise AssertionError(f"X6: expected 5 PMRT-GBM arms, found {sorted(pick)}")
    out = []
    for key in [*ORDER, ("exact", "exact")]:
        a = pick[key]
        b = a.get("bias", {}).get("null")
        out.append(
            {
                "arm": a["arm"],
                "centring": key[0],
                "redraw": key[1],
                "told_width": a["width"],
                "seeds": a["seeds"],
                "null_hits": a["null_raw"]["hits"],
                "null_n": a["null_raw"]["n"],
                "null_rate": a["null_raw"]["rate"],
                "null_lo": a["null_raw"]["ci"][0],
                "null_hi": a["null_raw"]["ci"][1],
                "null_validity": a["null_raw"]["validity"],
                "plac_rate": a["plac_raw"]["rate"],
                "plac_lo": a["plac_raw"]["ci"][0],
                "plac_hi": a["plac_raw"]["ci"][1],
                "plac_validity": a["plac_raw"]["validity"],
                "bias_mean": b["mean"] if b else None,
                "bias_lo": b["ci"][0] if b else None,
                "bias_hi": b["ci"][1] if b else None,
            }
        )
    return out


def latex(t: dict, rs: list[dict]) -> str:
    n_null, _n_plac = rs[0]["null_n"], None
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc check X6: exploratory, declared before the run, one condition (E1 R2, n = 1000, kappa .25),"
            f" {rs[0]['seeds']} fresh data sets; does not change C1-C3. Declared outcome: "
            f"{t['x6']['predictions']['outcome']} (DIAG s.X6).",
            f"Raw p <= .05 rejection rates over {n_null} truth-null candidates (placebo: 800); 95 % seed-cluster bootstrap"
            " CIs; mark = condition-validity label (* INVALID, unmarked VALID). Bias term = the declared first-order bias"
            " term (mean over truth-null candidates, 95 % CI); it did not account for the size of the inflation (row"
            " X.6c of the plan).",
        ],
        module=NAME,
    )
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l l c c c@{}}",
        r"\toprule",
        r"Centring & Redraw & Truth-null rate & Placebo rate & Mean bias term \\",
        r"law & law & [95\,\% CI] & [95\,\% CI] & [95\,\% CI] \\",
        r"\midrule",
        A.group_row(rf"{A.M.name(ARM)}, told dither width $\times 2$", 5),
    ]
    cap = {"told": "Told", "true": "True", "exact": "Exact design"}

    def line(r, first):
        bias = (
            (
                rf"{A.signed(r['bias_mean'], 3, plus=False)} "
                rf"{A.ci(r['bias_lo'], r['bias_hi'], A.signed, plus=False)}"
            )
            if r["bias_mean"] is not None
            else ""
        )
        return (
            rf"{first} & {A.prob(r['null_rate'])}{A.MARK[r['null_validity']]} {A.ci(r['null_lo'], r['null_hi'])}"
            rf" & {A.prob(r['plac_rate'])}{A.MARK[r['plac_validity']]} {A.ci(r['plac_lo'], r['plac_hi'])} & {bias} \\"
        )

    for r in rs[:4]:
        out.append(line(r, rf"\quad {cap[r['centring']]} & {cap[r['redraw']].lower()}"))
    out += [
        r"\midrule",
        A.group_row(rf"{A.M.name(ARM)}, exact design (width $\times 1$)", 5),
        line(rs[4], r"\multicolumn{2}{@{}l}{\quad Exact design}"),
    ]
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    t = load_json(*DIAG)
    rs = rows(t)
    return A.write(NAME, latex(t, rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

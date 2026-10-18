"""Post hoc check X3 (Appendix C.3; booktabs LaTeX + csv): the compact table next to fig_appC_x3_misspec (PAPER_PLAN
question 14: a small figure plus a compact table). The data are generated with the frozen design and the design is
told to the method wrong. Null-edge raw rejection rate (p <= .05) with the cell-validity mark:
  * truth-null edges of R2 (E1-E2, n 1000, 60 seeds) against the told dither width (x .5, .8, 1, 1.25, 2) and the told
    setpoint switch delay (+2 %, +5 % of a block);
  * the confounded null edge of E4 R3 (lambda 1, n 1000, 60 seeds, one edge per seed) against the told logging-policy
    strength lambda (x .5 ... 2).
Not a figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_appC_x3``.

Source: XR ``extras_tables.json``, ``pooled`` rows with experiment X3; R2 key ``null_raw``, R3 key ``conf_raw`` (rate,
hits, n, seed-cluster 95 % CI, validity), matched by base arm and ``design_key`` (``("exact", 1.0)`` = the exact
design). Rows only select stored values. Required wording (PAPER_PLAN s.5 item 6) belongs to the caption: a
degradation curve, not evidence against C2a.
"""

from oran_causal.figures.figs import _appc as C
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_x3"
SOURCES = [C.SRC_XR]
ARMS = ["pmrt_nl_eq", "pmrt_eq", "pmrt_r3", "pcorr_eq"]
EXACT = ("exact", 1.0)
R2_COLS = [("width", v) if v != 1.0 else EXACT for v in C.TOLD] + [("shift", 0.02), ("shift", 0.05)]
R3_COLS = [("lam", v) if v != 1.0 else EXACT for v in C.TOLD]


def plain(label: str) -> str:
    """Method label with the confounder named in words."""
    return label.replace("R3 covariates", "confounder covariates")


def rows(tab: dict) -> list[dict]:
    out = []
    for reg, key, cols in (("R2", "null_raw", R2_COLS), ("R3", "conf_raw", R3_COLS)):
        pool = C.pooled(tab, "X3", reg)
        for arm in ARMS:
            for col in cols:
                r = pool.get((arm, col))
                if r is None or r[key] is None:
                    raise AssertionError(f"X3: no {key} for {arm} {reg} {col}")
                v = r[key]
                out.append(
                    {
                        "arm": arm,
                        "regime": reg,
                        "edge": key,
                        "element": col[0],
                        "told": col[1],
                        "rate": v["rate"],
                        "lo": v["ci"][0],
                        "hi": v["ci"][1],
                        "hits": v["hits"],
                        "n": v["n"],
                        "validity": v["validity"],
                    }
                )
    return out


def latex(tab: dict, rs: list[dict]) -> str:
    cell = {(r["arm"], r["regime"], (r["element"], r["told"])): r for r in rs}
    n2 = {r["n"] for r in rs if r["regime"] == "R2"}
    n3 = {r["n"] for r in rs if r["regime"] == "R3"}
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc check X3: " + tab["wording"],
            f"Raw p <= .05 rejection rates; R2 truth-null edges of E1-E2 (n 1000, {'/'.join(map(str, sorted(n2)))} candidate"
            f" tests per design), E4 R3 confounded null edge (lambda 1, n 1000, {'/'.join(map(str, sorted(n3)))} tests: one"
            " edge per seed). Mark = condition-validity"
            " label (* INVALID, ~ INCONCLUSIVE, unmarked VALID). Told multipliers are told / true; 1 = the exact design.",
            "Caption must carry the required X3 sentence (PAPER_PLAN s.5 item 6).",
        ],
        module=NAME,
    )

    def line(arm, reg, cols, pad):
        vals = [
            A.prob(cell[(arm, reg, c)]["rate"]) + A.MARK[cell[(arm, reg, c)]["validity"]]
            for c in cols
        ]
        return (
            rf"\quad {plain(A.tex_name(arm))} & " + " & ".join(vals + [""] * pad) + r" \\"
        )  # full names: no family rows

    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}l c c c c c c c@{}}",
        r"\toprule",
        A.group_row("Truth-null edges, dithered setpoint, linear graph and rare regime", 8),
        r" & \multicolumn{5}{c}{Told dither width ($\times$ true)} & \multicolumn{2}{c}{Told setpoint delay} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}",
        r"Method & 0.5 & 0.8 & 1 & 1.25 & 2 & +2\,\% & +5\,\% \\",
        r"\midrule",
    ]
    out += [line(arm, "R2", R2_COLS, 0) for arm in A.M.order(ARMS)]
    out += [
        r"\midrule",
        A.group_row(
            r"Confounded null edge, latent confounding, confounded policy ($\lambda = 1$)", 8
        ),
        r" & \multicolumn{5}{c}{Told $\lambda$ ($\times$ true)} & & \\",
        r"\cmidrule(lr){2-6}",
        r"Method & 0.5 & 0.8 & 1 & 1.25 & 2 & & \\",
        r"\midrule",
    ]
    out += [line(arm, "R3", R3_COLS, 2) for arm in A.M.order(ARMS)]
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    tab = C.extras()
    rs = rows(tab)
    return A.write(NAME, latex(tab, rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

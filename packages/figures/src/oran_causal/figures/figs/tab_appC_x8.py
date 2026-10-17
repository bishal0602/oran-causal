"""Post hoc check X8, stricter-threshold null-outcome check of the frozen Study 1 PMRT (Appendix C;
booktabs LaTeX + csv): on 590 null-outcome variants of the evaluation logs (every cyclic outcome
shift inside the ten 60-episode groups, B 9999), the number of variants with any declaration and the
counts of used p-values at or below .05, 1e-3, 2e-4 and 1e-4, for the primary combination and the
eight descriptive ones; then the redraw p against its normal approximation. Not a figure (no SPEC):
run ``uv run python -m oran_causal.figures.figs.tab_appC_x8``.

Source: extras/x8/summary.json (written by scratchpad/e6_dev/k0n_tail_check.py summarize; X8,
EXTRAS_PROTOCOL amendment 2026-10-11, declared before the run), keys ``n_variants``, ``B``,
``primary``, ``checks``, ``combos.<stat+layer>.{n_p, variants_with_decl, n_le_<a>, rate_le_<a>}``,
``arms_p2_vs_normal.<stat>``. Post hoc, descriptive apart from the three pre-stated checks X8a-c; it
changes no verdict.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_x8"
X8 = ("posthoc_x8", "x8", "summary.json")
SOURCES = [source_note(*X8)]
STATS = [("plain_c", "Plain"), ("loadsp_c", "Matched filter"), ("max", "Maximum")]
LAYERS = [("by", "BY"), ("wby1s", "Weighted one-sided BY"), ("dagger1s", "One-sided DAGGER")]
ALPHAS = [
    ("0.05", "0.05"),
    ("0.001", r"$10^{-3}$"),
    ("0.0002", r"$2\times10^{-4}$"),
    ("0.0001", r"$10^{-4}$"),
]


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X8)
    if t["B"] != [9999] or t["n_variants"] != 590 or t["primary"] != "loadsp_c+wby1s":
        raise AssertionError(
            f"X8 summary is not the declared run: {t['n_variants']} variants, B {t['B']}"
        )
    if not all(c["pass"] for c in t["checks"].values()):
        raise AssertionError(
            "an X8 check failed: the table text assumes CONSISTENT; revise the caption"
        )
    out = []
    for s, sn in STATS:
        for lay, ln in LAYERS:
            c = t["combos"][f"{s}+{lay}"]
            row = {
                "combo": f"{s}+{lay}",
                "statistic": sn,
                "layer": ln,
                "primary": f"{s}+{lay}" == t["primary"],
                "n_p": c["n_p"],
                "variants_with_decl": c["variants_with_decl"],
            }
            for a, _ in ALPHAS:
                row[f"n_le_{a}"] = c[f"n_le_{a}"]
            out.append(row)
    arms = []
    for s, sn in STATS:
        if s in t["arms_p2_vs_normal"]:
            v = t["arms_p2_vs_normal"][s]
            arms.append(
                {
                    "statistic": sn,
                    "n": v["n"],
                    **{f"redraw_{a}": v[f"redraw_le_{a}"] for a, _ in ALPHAS},
                    **{f"normal_{a}": v[f"normal_le_{a}"] for a, _ in ALPHAS},
                }
            )
    return out, {"n_variants": t["n_variants"], "B": t["B"][0], "arms": arms, "n_p": out[0]["n_p"]}


def sci(x: float) -> str:
    """A small rate in LaTeX scientific notation: 7.9e-4 -> $7.9\\times10^{-4}$; >= .01 -> .047."""
    if x >= 0.01:
        return A.prob(x, 3)
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def latex(rs: list[dict], meta: dict) -> str:
    n = meta["n_p"]
    exp = [n * float(a) for a, _ in ALPHAS]
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc check X8 (declared before the run): pre-stated checks on the primary combination all pass; other"
            " combinations descriptive; no verdict changes.",
            f"{meta['n_variants']} null-outcome variants (10 groups x 59 cyclic shifts), B {meta['B']}; {n} used p-values"
            " per combination; expected counts under exact validity in the last row.",
            "(b) two-sided redraw p versus the normal approximation 2(1 - Phi(|z|)), share of all tested hypotheses.",
        ],
        module=NAME,
    )
    na = len(ALPHAS)
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l l c" + " r" * na + "@{}}",
        r"\toprule",
        rf"\multicolumn{{{3 + na}}}{{@{{}}l}}{{(a) Declarations and small p-values on null outcomes}} \\",
        r"\midrule",
        rf" & & Variants with & \multicolumn{{{na}}}{{c}}{{Used p-values at or below}} \\",
        rf"\cmidrule(lr){{4-{3 + na}}}",
        r"Statistic & Declaration layer & a declaration & "
        + " & ".join(h for _, h in ALPHAS)
        + r" \\",
        r"\midrule",
    ]
    for r in rs:
        cells = " & ".join(str(r[f"n_le_{a}"]) for a, _ in ALPHAS)
        lab = rf"{r['statistic']} & {r['layer']}"
        line = rf"{lab} & {r['variants_with_decl']} & {cells} \\"
        out.append(
            r"\textbf{" + line.replace(" & ", r"} & \textbf{").replace(r" \\", r"} \\")
            if r["primary"]
            else line
        )
    out += [
        r"\midrule",
        r"Expected if exact & & & " + " & ".join(f"{e:.1f}" for e in exp) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\par\medskip",
        "",
    ]
    out += [
        r"\begin{tabular}{@{}l l" + " r" * na + "@{}}",
        r"\toprule",
        rf"\multicolumn{{{2 + na}}}{{@{{}}l}}{{(b) Redraw p-value versus its normal approximation}} \\",
        r"\midrule",
        r"Statistic & p-value & " + " & ".join(h for _, h in ALPHAS) + r" \\",
        r"\midrule",
    ]
    for a_ in meta["arms"]:
        for kind, kn in (("redraw", "Redraw"), ("normal", "Normal")):
            out.append(
                rf"{a_['statistic'] if kind == 'redraw' else ''} & {kn} & "
                + " & ".join(sci(a_[f"{kind}_{a}"]) for a, _ in ALPHAS)
                + r" \\"
            )
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

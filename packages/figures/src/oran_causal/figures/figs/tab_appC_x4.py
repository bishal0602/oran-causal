"""Post hoc check X4 (Appendix C.4; booktabs LaTeX + csv): the random-rejection referee at three
rejection rates next to the stored Study 3 arms on the same 160 evaluation seeds. Three stacked
tabulars: (a) rejection rate, recovery R, energy retention, guard ratios and eligibility; (b) paired
differences in R (random-rejection referee minus each comparator); (c) rejected requests by
control-parameter family. Not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_appC_x4``.

Source: extras/x4/x4_tables.json (XR4, X4_REPORT.md for column meaning only):
``rows.<arm>.{defer_rate, R, R_ci90, retention, guard_ratio, eligible, requests, rejected,
defer_by_family_phase}``, ``contrasts.<X4 arm>.<comparator>.{dR, ci90}``, ``p``, ``n_seeds``.
Allowed re-aggregation: rejected requests per family = pre + post warm-up counts of
``defer_by_family_phase`` (XR4 does the same). Post hoc, descriptive, no verdict; Study 3 is
unchanged.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_x4"
X4 = ("posthoc_x4", "x4", "x4_tables.json")
SOURCES = [source_note(*X4)]
STORED = ["noarb", "never_sleep", "MG:PMRT", "CS:PMRT", "CS:GT"]
RANDOM = ["X4:rand@0.25", "X4:rand@0.352", "X4:rand@0.5"]
GUARDS = [("svr", "SVR"), ("nonprot_embb_viol", "eMBB"), ("ll_viol", "LL"), ("rlf", "RLF")]
FAMILIES = ["prot_min", "ptx", "carrier", "sleep"]


GROUPS = [
    ("Static rules", ["noarb", "never_sleep"]),
    ("Map-driven referees", ["MG:PMRT", "CS:PMRT", "CS:GT"]),
    ("Random-rejection referee", RANDOM),
]


def label(arm: str, p: dict) -> str:
    """Row label inside its group (paper terms: TERMINOLOGY s.7; method names via
    oran_causal.figures.methods).
    """
    if arm in p:
        return rf"$p = {A.short_prob(p[arm])}$"
    gate = arm.split(":", maxsplit=1)[0] if ":" in arm else None
    if gate in ("MG", "CS"):
        m = A.M.name(arm).replace(" map", "")
        return f"{m} map, " + ("ungated" if gate == "MG" else "edge-certified")
    return A.M.name(arm)


def full_label(arm: str, p: dict) -> str:
    """Stand-alone label (csv and table (c))."""
    if arm in p:
        return "Random rejection, " + label(arm, p)
    return label(arm, p)


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X4)
    out = []
    for arm in STORED + RANDOM:
        r = t["rows"][arm]
        fam = {
            f: sum(v for k, v in r["defer_by_family_phase"].items() if k.rsplit("_", 1)[0] == f)
            for f in FAMILIES
        }
        if sum(fam.values()) != r["deferred"]:
            raise AssertionError(
                f"{arm}: family counts {sum(fam.values())} != deferred {r['deferred']}"
            )
        row = {
            "arm": arm,
            "label": full_label(arm, t["p"]),
            "requests": r["requests"],
            "deferred": r["deferred"],
            "defer_rate": r["defer_rate"],
            "R": r["R"],
            "R_lo": r["R_ci90"][0],
            "R_hi": r["R_ci90"][1],
            "retention": r["retention"],
            "eligible": r["eligible"],
        }
        row.update({f"guard_{g}": r["guard_ratio"][g] for g, _ in GUARDS})
        row.update({f"deferred_{f}": fam[f] for f in FAMILIES})
        for comp in STORED:
            if arm in t["contrasts"]:
                c = t["contrasts"][arm][comp]
                row.update(
                    {
                        f"dR_vs_{comp}": c["dR"],
                        f"dR_vs_{comp}_lo": c["ci90"][0],
                        f"dR_vs_{comp}_hi": c["ci90"][1],
                    }
                )
        out.append(row)
    return out, {"p": t["p"], "n_seeds": t["n_seeds"]}


def latex(rs: list[dict], meta: dict) -> str:
    by = {r["arm"]: r for r in rs}
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc check X4: descriptive, no hypothesis, no criterion, no verdict; Study 3 unchanged.",
            f"{meta['n_seeds']} Study 3 evaluation seeds; R and paired differences with 90 % paired seed-bootstrap CIs."
            " Guard ratios vs accept-all; eligible = retention >= .90 and every guard ratio <= 1.10 (point).",
            "Host note (PAPER_PLAN X4.4): X4 ran on a different host than the stored arms; |dR| <= 4e-4 there.",
            "(c) rejected requests summed over warm-up and scored window.",
        ],
        module=NAME,
    )
    out += [r"\footnotesize", r"\setlength{\tabcolsep}{2pt}"]
    # (a) outcomes
    g = len(GUARDS)
    out += [
        r"\begin{tabular}{@{}l c c c" + " c" * g + r" c@{}}",
        r"\toprule",
        rf"\multicolumn{{{5 + g}}}{{@{{}}l}}{{(a) Outcomes}} \\",
        r"\midrule",
        r" & Rejection & & & \multicolumn{" + str(g) + r"}{c}{Guard ratio} & \\",
        rf"\cmidrule(lr){{5-{4 + g}}}",
        r"Arm & rate & $R$ [90\,\% CI] & Retention & "
        + " & ".join(h for _, h in GUARDS)
        + r" & Elig. \\",
        r"\midrule",
    ]
    for i, (group, arms) in enumerate(GROUPS):
        out += ([r"\midrule"] if i else []) + [A.group_row(group, 5 + g)]
        for arm in arms:
            r = by[arm]
            guards = " & ".join(f"{r[f'guard_{k}']:.2f}" for k, _ in GUARDS)
            out.append(
                rf"\quad {label(arm, meta['p'])} & {A.prob(r['defer_rate'], 3)} & {A.signed(r['R'])} "
                rf"{A.ci(r['R_lo'], r['R_hi'], A.signed, plus=False)} & {r['retention']:.3f} & {guards} & "
                rf"{'yes' if r['eligible'] else 'no'} \\"
            )
    out += [r"\bottomrule", r"\end{tabular}", "", r"\par\medskip", ""]
    # (b) paired differences, comparators as rows
    out += [
        r"\begin{tabular}{@{}l" + " c" * len(RANDOM) + "@{}}",
        r"\toprule",
        rf"\multicolumn{{{1 + len(RANDOM)}}}{{@{{}}l}}{{(b) Paired difference in $R$, random-rejection referee"
        r" minus comparator [90\,\% CI]} \\",
        r"\midrule",
        "Comparator & "
        + " & ".join(rf"$p = {A.short_prob(meta['p'][a])}$" for a in RANDOM)
        + r" \\",
        r"\midrule",
    ]
    for i, (group, arms) in enumerate(GROUPS[:2]):
        out += ([r"\midrule"] if i else []) + [A.group_row(group, 1 + len(RANDOM))]
        for comp in arms:
            cells = " & ".join(
                rf"{A.signed(by[a][f'dR_vs_{comp}'])} "
                rf"{A.ci(by[a][f'dR_vs_{comp}_lo'], by[a][f'dR_vs_{comp}_hi'], A.signed, plus=False)}"
                for a in RANDOM
            )
            out.append(rf"\quad {label(comp, meta['p'])} & {cells} \\")
    out += [r"\bottomrule", r"\end{tabular}", "", r"\par\medskip", ""]
    # (c) deferred requests by family
    out += [
        r"\begin{tabular}{@{}l" + " r" * len(FAMILIES) + r" r r@{}}",
        r"\toprule",
        rf"\multicolumn{{{3 + len(FAMILIES)}}}{{@{{}}l}}{{(c) Rejected requests by control-parameter family}} \\",
        r"\midrule",
        "Arm & "
        + " & ".join(f.replace("_", r"\_") for f in FAMILIES)
        + r" & Rejected & Requests \\",
        r"\midrule",
    ]
    for i, (group, arms) in enumerate([GROUPS[2], (GROUPS[1][0], ["MG:PMRT"])]):
        out += ([r"\midrule"] if i else []) + [A.group_row(group, 3 + len(FAMILIES))]
        for arm in arms:
            r = by[arm]
            fam = " & ".join(A.thousands(r[f"deferred_{f}"]) for f in FAMILIES)
            out.append(
                rf"\quad {label(arm, meta['p'])} & {fam} & {A.thousands(r['deferred'])} & "
                rf"{A.thousands(r['requests'])} \\"
            )
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

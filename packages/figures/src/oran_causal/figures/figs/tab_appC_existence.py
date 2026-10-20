"""Post hoc analysis X14, map scores against an "effect exists" ground truth (Appendix C; booktabs
LaTeX + csv): each method's declared edges against the knockout labels with no practical-effect
threshold (the frozen 95 % episode-cluster interval excludes zero, any magnitude), next to its
precision under the frozen material-effect rule. Declared edges are confirmed (effect exists,
declared sign), wrong sign, or not established (interval contains zero). Blocks: Study 1 pooled,
Study 1 over the 17 slices (PMRT and Granger + BY only, the rows Section 6.2 cites; summed counts;
median [min, max] precision; every method's slice rows stay in the csv), Study 2 pooled discovery
maps (DEV- tuned thresholds; the placebo-calibrated maps, the random map and every unnamed
comparator are in the csv). Not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_appC_existence``.

Source: posthoc_x9_x16/revision/x14_existence.json (X14, posthoc protocol amendment 2026-10-11; the script
reproduced the frozen material-rule metrics of every stored analysis before writing), keys
``studies.<S1|S2>.reproduced``, ``.levels.<0.95|0.9|0.99>.counts`` and
``.methods.<code>.<slice>.{mat_n_declared, mat_precision, ex_confirmed, ex_wrong_sign,
ex_not_established, ex_unsupported, ex_precision, ex_recall_right_sign, ex_n_exists}``. PMRT =
loadsp_c+wby1s (Study 1) and the MG:PMRT map (Study 2). Allowed re-aggregation: slice counts summed,
slice precisions as median / min / max. Post hoc, descriptive, no verdict; the frozen material rule
stays the primary ground truth.
"""

import statistics

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A
from oran_causal.figures.figs._tabm import baseline_name

NAME = "tab_appC_existence"
X14 = ("posthoc_x9_x16", "revision", "x14_existence.json")
SOURCES = [source_note(*X14)]
LEVELS = ("0.95", "0.9", "0.99")
S1_POOLED = [
    "loadsp_c+wby1s",
    "corr",
    "granger",
    "granger_by",
    "shap_gbdt",
    "int",
    "qacm",
    "two_tower",
]
S1_SLICES = [
    "loadsp_c+wby1s",
    "granger_by",
]  # the slice block keeps the two rows Section 6.2 cites
S2_POOLED = [
    "MG:PMRT",
    "MG:corr@dev",
    "MG:granger@dev",
    "MG:granger_by",
    "MG:shap_gbdt@dev",
    "MG:int@dev",
    "MG:qacm@dev",
    "MG:two_tower@dev",
]
COUNTS = ("mat_n_declared", "ex_confirmed", "ex_wrong_sign", "ex_not_established", "ex_unsupported")


def label(code: str) -> str:
    if code in ("loadsp_c+wby1s", "MG:PMRT"):
        return "PMRT"
    if code.startswith("MG:"):
        return baseline_name(code) + (" (untuned)" if code == "MG:granger_by" else "")
    return baseline_name(code)


def check(t: dict) -> None:
    for st in ("S1", "S2"):
        if not t["studies"][st]["reproduced"]:
            raise AssertionError(f"X14 {st} reproduction flag not set")
    p1 = t["studies"]["S1"]["levels"]["0.95"]["methods"]["loadsp_c+wby1s"]["pooled"]
    p2 = t["studies"]["S2"]["levels"]["0.95"]["methods"]["MG:PMRT"]["pooled"]
    if (round(p1["mat_precision"], 2), round(p1["mat_recall"], 2), p1["mat_n_declared"]) != (
        0.92,
        0.79,
        28,
    ):
        raise AssertionError("Study 1 frozen PMRT not P .92, R .79, 28 declarations")
    if (p2["mat_true_right"], p2["mat_true_wrong"], p2["mat_null"], p2["mat_indet"]) != (
        18,
        0,
        3,
        1,
    ):
        raise AssertionError("Study 2 PMRT map not 18 / 0 / 3 / 1")
    for st in ("S1", "S2"):
        for L in LEVELS:
            for m, v in t["studies"][st]["levels"][L]["methods"].items():
                for sl, q in v.items():
                    n = (
                        q["ex_confirmed"]
                        + q["ex_wrong_sign"]
                        + q["ex_not_established"]
                        + q["ex_unsupported"]
                    )
                    if n != q["mat_n_declared"]:
                        raise AssertionError(f"{st} {L} {m} {sl}: existence classes do not add up")


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X14)
    check(t)
    out = []
    for st in ("S1", "S2"):
        for L in LEVELS:
            lv = t["studies"][st]["levels"][L]
            for m, v in lv["methods"].items():
                for sl, q in v.items():
                    out.append(
                        {
                            "study": st,
                            "ci_level": float(L),
                            "slice": sl,
                            "method_code": m,
                            "method": label(m) if A.has_name(m) or m == "loadsp_c+wby1s" else m,
                            **q,
                            "n_exists": lv["counts"]["EXISTS_pos"] + lv["counts"]["EXISTS_neg"],
                            "n_not_established": lv["counts"]["NOT_ESTABLISHED"],
                        }
                    )
    return out, t


def med(xs):
    xs = [x for x in xs if not A.nan(x)]
    return statistics.median(xs), min(xs), max(xs)


def latex(t: dict) -> str:
    S = t["studies"]
    c1, c2 = S["S1"]["levels"]["0.95"]["counts"], S["S2"]["levels"]["0.95"]["counts"]
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc analysis X14: descriptive, no verdict; the frozen material-effect rule stays primary.",
            "Effect exists = the frozen 95 % episode-cluster bootstrap interval excludes zero (any magnitude); "
            f"Study 1: {c1['EXISTS_pos'] + c1['EXISTS_neg']} of 60 hypotheses, Study 2: {c2['EXISTS_pos'] + c2['EXISTS_neg']}"
            " of 60; no hypothesis unsupported.",
            "Material = frozen rule, precision TP / (TP + FP) as in the studies. Existence precision = confirmed / "
            "(confirmed + wrong sign + not established); recall = confirmed / hypotheses with an effect.",
            "Subsample block: counts summed over the 17 subsamples, precision median [min, max]. Placebo-calibrated maps, the "
            "random map and unnamed comparators in the csv.",
        ],
        module=NAME,
    )
    ncol = 9
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1pt}",
        r"\begin{tabular}{@{}l c c c c c c c c@{}}",
        r"\toprule",
        r" & & Material & \multicolumn{6}{c}{Effect exists (any magnitude)} \\",
        r"\cmidrule(lr){3-3}\cmidrule(lr){4-9}",
        r" & & & & Wrong & Not & & & Prec. \\",
        r"Method & Decl. & precision & Confirmed & sign & est. & Precision & Recall & 90 / 99\,\% \\",
    ]

    def pooled_row(st, code):
        q = {L: S[st]["levels"][L]["methods"][code]["pooled"] for L in LEVELS}
        p = q["0.95"]
        return (
            rf"\quad {label(code)} & {p['mat_n_declared']} & {A.prob(p['mat_precision'], 2)} & "
            rf"{p['ex_confirmed']} & {p['ex_wrong_sign']} & {p['ex_not_established']} & "
            rf"{A.prob(p['ex_precision'], 2)} & {A.prob(p['ex_recall_right_sign'], 2)} & "
            rf"{A.prob(q['0.9']['ex_precision'], 2)} / {A.prob(q['0.99']['ex_precision'], 2)} \\"
        )

    out += [r"\midrule", A.group_row("Randomised logs, 600 episodes pooled", ncol)]
    out += [pooled_row("S1", m) for m in S1_POOLED]
    out += [
        r"\midrule",
        A.group_row(
            "Randomised logs, 17 subsamples: counts summed, precision median [min, max]", ncol
        ),
    ]
    for m in S1_SLICES:
        sl = {
            L: {k: v for k, v in S["S1"]["levels"][L]["methods"][m].items() if k != "pooled"}
            for L in LEVELS
        }
        if len(sl["0.95"]) != 17:
            raise AssertionError(f"{m}: {len(sl['0.95'])} slices, expected 17")
        s = {k: sum(q[k] for q in sl["0.95"].values()) for k in COUNTS}
        mp, ep = (
            med([q[k] for q in sl["0.95"].values()]) for k in ("mat_precision", "ex_precision")
        )
        rc = med([q["ex_recall_right_sign"] for q in sl["0.95"].values()])
        e90, e99 = (med([q["ex_precision"] for q in sl[L].values()])[0] for L in ("0.9", "0.99"))
        out.append(
            rf"\quad {label(m)} & {s['mat_n_declared']} & {A.prob(mp[0], 2)} {A.ci(mp[1], mp[2], nd=2)} & "
            rf"{s['ex_confirmed']} & {s['ex_wrong_sign']} & {s['ex_not_established']} & "
            rf"{A.prob(ep[0], 2)} {A.ci(ep[1], ep[2], nd=2)} & {A.prob(rc[0], 2)} & "
            rf"{A.prob(e90, 2)} / {A.prob(e99, 2)} \\"
        )
    out += [
        r"\midrule",
        A.group_row("Confounded logs, pooled discovery maps (tuned thresholds)", ncol),
    ]
    out += [pooled_row("S2", m) for m in S2_POOLED]
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, t = rows()
    return A.write(NAME, latex(t), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

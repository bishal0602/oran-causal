"""Post hoc check X7 (Appendix C.7; booktabs LaTeX + csv): the edge-certified map-driven referees
with the four associational maps next to the static ES + Power ES subset on the 160 Study 3
evaluation seeds. Two stacked tabulars: (a) per-request agreement with the subset's accept / reject
decision (scored window), on the subset's request stream by control-parameter family and on each
referee's own stream; (b) recovery R, energy retention and eligibility of the subset, accept-all,
each referee and its two attribution replays (only its prot_min rejections; every rejection except
prot_min), with the paired difference in R vs the subset and the seeds on which the record equals
the subset's. Not a figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_appC_x7``.

Source: posthoc_x7/x7/x7_tables.json (XR7, X7_REPORT.md for column meaning only):
``agreement.stream_ref.<arm>.post. {overall, prot_min|+1, ptx, carrier, sleep}.{agree, n}``,
``agreement.stream_own.<arm>.post.overall.agree``, ``rows.<arm>.{R, R_ci90, retention, eligible}``,
``diff_vs_ref.<arm>.{dR, dR_ci90, seeds_bit_identical_to_ref}`` (stored referee vs stored subset,
one host) and ``diff_same_platform["<replay> - X7:sub|shadow"].{dR, dR_ci90, seeds_bit_identical}``
(replays vs the subset's re-run on the same host, as XR7 reports them). Rows only select stored
values. Post hoc, descriptive, no verdict; Study 3 unchanged.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _taba as A

NAME = "tab_appC_x7"
X7 = ("posthoc_x7", "x7", "x7_tables.json")
SOURCES = [source_note(*X7)]
REF = "sub:ES+PowerES"
SHADOW = "X7:sub|shadow"
FAMS = [
    ("prot_min|+1", "Prot. share up"),
    ("ptx", "Tx power"),
    ("carrier", "Carrier"),
    ("sleep", "Sleep"),
]


def map_name(arm: str) -> str:
    name = A.M.name(arm)
    return f"{name} (placebo-cal.)" if A.M.resolve(arm).threshold == "plc" else name


def referee_group(map_label: str) -> str:
    """Group-row title of a gated referee, e.g. 'Gated referee, Granger map'."""
    base = map_label.removesuffix(" (placebo-cal.)")
    tail = ", placebo-calibrated threshold" if base != map_label else ""
    return f"Gated referee, {base} map{tail}"


def rows(t: dict) -> tuple[list[dict], list[dict]]:
    agree, outcome = [], []
    for arm in t["refs"]:
        s = t["agreement"]["stream_ref"][arm]["post"]
        r = {
            "arm": arm,
            "map": map_name(arm),
            "overall": s["overall"]["agree"],
            "n_overall": s["overall"]["n"],
            "own_overall": t["agreement"]["stream_own"][arm]["post"]["overall"]["agree"],
            "n_own": t["agreement"]["stream_own"][arm]["post"]["overall"]["n"],
        }
        for k, _ in FAMS:
            r[k] = s[k]["agree"]
            r[f"n_{k}"] = s[k]["n"]
        agree.append(r)

    def out_row(key, kind, label, diff):
        x = t["rows"][key]
        d = {
            "arm": key,
            "kind": kind,
            "label": label,
            "R": x["R"],
            "R_lo": x["R_ci90"][0],
            "R_hi": x["R_ci90"][1],
            "retention": x["retention"],
            "eligible": x["eligible"],
        }
        if diff is not None:
            d.update(
                {
                    "dR": diff["dR"],
                    "dR_lo": diff["dR_ci90"][0],
                    "dR_hi": diff["dR_ci90"][1],
                    "identical_seeds": diff.get(
                        "seeds_bit_identical_to_ref", diff.get("seeds_bit_identical")
                    ),
                }
            )
        return d

    outcome.append(out_row(REF, "subset", "Static subset", None))
    outcome.append(out_row("noarb", "accept_all", A.M.name("noarb"), None))
    for arm in t["refs"]:
        outcome.append(out_row(arm, "referee", "Referee", t["diff_vs_ref"][arm]))
        for rep, lab in (
            ("pm", "Replay: protected-share rejections only"),
            ("other", "Replay: without it"),
        ):
            key = f"X7:{arm}|{rep}"
            outcome.append(
                out_row(key, f"replay_{rep}", lab, t["diff_same_platform"][f"{key} - {SHADOW}"])
            )
        outcome[-3]["map"] = outcome[-2]["map"] = outcome[-1]["map"] = map_name(arm)
    return agree, outcome


def latex(t: dict, agree: list[dict], outcome: list[dict]) -> str:
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc check X7: descriptive, no hypothesis, no criterion, no verdict; Study 3 unchanged.",
            f"{t['n_seeds']} Study 3 evaluation seeds. (a) share of scored-window requests (t >= 120 s) on which the"
            " referee's accept / reject decision equals the subset's; subset stream = the subset applied, the referee"
            " queried in shadow (primary); own stream = the referee applied, the subset in shadow.",
            "(b) R and paired differences with 90 % paired seed-bootstrap CIs; dR = arm minus the subset; referee rows"
            " compare stored records, replay rows compare with the subset's re-run on the replay host (never mixing"
            " hosts). Identical seeds = seeds whose record equals the subset's in every outcome field (of 160)."
            " Dagger = not eligible (retention < .90); every other row is eligible.",
            "Host note: re-runs vs stored records differ on some seeds by a few violations,"
            " dR within [-0.0004, +0.0003].",
        ],
        module=NAME,
    )
    nf = len(FAMS)
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}l" + " c" * nf + r" c c@{}}",
        r"\toprule",
        rf"\multicolumn{{{3 + nf}}}{{@{{}}l}}{{(a) Agreement with the static subset's decision, per request}} \\",
        r"\midrule",
        rf" & \multicolumn{{{nf + 1}}}{{c}}{{Subset's request stream}} & Own stream \\",
        rf"\cmidrule(lr){{2-{nf + 2}}}\cmidrule(lr){{{nf + 3}-{nf + 3}}}",
        "Map & " + " & ".join(h for _, h in FAMS) + r" & All & All \\",
        r"\midrule",
    ]
    for r in agree:
        cells = " & ".join(A.prob(r[k], 4) for k, _ in FAMS)
        out.append(
            rf"{r['map']} & {cells} & {A.prob(r['overall'], 4)} & {A.prob(r['own_overall'], 4)} \\"
        )
    first = agree[0]
    if any(r["n_overall"] != first["n_overall"] for r in agree):
        raise AssertionError("subset-stream request counts differ across referees")
    out += [
        r"\midrule",
        "Requests & "
        + " & ".join(A.thousands(first[f"n_{k}"]) for k, _ in FAMS)
        + rf" & {A.thousands(first['n_overall'])} & \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\par\medskip",
        "",
    ]
    # (b) outcomes, replays, paired differences
    out += [
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.7cm} c c c c@{}}",
        r"\toprule",
        r"\multicolumn{5}{@{}l}{(b) Outcomes, attribution replays and paired differences vs the static subset} \\",
        r"\midrule",
        r" & & & $\Delta R$ vs subset & Identical \\",
        r"Arm & $R$ [90\,\% CI] & Retention & [90\,\% CI] & seeds \\",
        r"\midrule",
    ]

    def line(label, r):
        d = (
            rf"{A.signed(r['dR'])} {A.ci(r['dR_lo'], r['dR_hi'], A.signed, plus=False)} & {r['identical_seeds']}"
            if "dR" in r
            else " & "
        )
        mark = "" if r["eligible"] else r"$^{\dagger}$"
        return (
            rf"{label} & {A.signed(r['R'])} {A.ci(r['R_lo'], r['R_hi'], A.signed, plus=False)} & {r['retention']:.3f}{mark} & "
            rf"{d} \\"
        )

    out += [line(outcome[0]["label"], outcome[0]), line(outcome[1]["label"], outcome[1])]
    for r in outcome[2:]:
        if r["kind"] == "referee":
            out += [
                r"\midrule",
                A.group_row(referee_group(r["map"]), 5),
            ]
        out.append(line(rf"\quad {r['label']}", r))
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    t = load_json(*X7)
    agree, outcome = rows(t)
    return A.write(
        NAME,
        latex(t, agree, outcome),
        [dict(r, table="a") for r in agree] + [dict(r, table="b") for r in outcome],
    )


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

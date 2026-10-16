"""Study 3 (edge-certified referee) criteria E, D1, D2, D3, S1 with the frozen rule, the measured
value and pass / fail, plus the verdict and its precedence. Booktabs LaTeX + csv; not a figure (no
SPEC): run ``uv run python -m oran_causal.figures.figs.tab_study3_criteria``.

Values: FD ``certsafe.eval`` (E, D1, D2, D3, S1, verdict, identical_seeds, n, arms.CS:PMRT: R,
R_ci90, retention, guard_ratio). Rule constants and the precedence: the analyzer verdict
``cs_verdict.json`` (D1.margin, D2.alpha, D2.n_hypotheses, D1.best_selected_share = the gated
associational maps, D3.ref, S1.cap, K0 / K0n carried from Study 2, verdict.precedence). The
eligibility thresholds (retention, guard cap) are stored in no JSON: read from the frozen Study 3
protocol s.7 text. Re-aggregation: the count of rejected D2 hypotheses and of gated associational
maps; FD and cs_verdict.json are checked to agree on every value shown.
"""

from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _tabm as TM

NAME = "tab_study3_criteria"
SOURCES = [ST.SRC_FD, TM.SRC_CS_VERDICT, TM.SRC_CS_PROTOCOL]
HERO = "CS:PMRT"


def _check(ev: dict, vd: dict) -> None:
    pairs = [
        (ev["E"], vd["E"]["pass"]),
        (ev["verdict"], vd["verdict"]["verdict"]),
        (ev["n"], vd["n_seeds"]),
    ]
    for k in ("D1", "D2", "D3", "S1"):
        pairs.append((ev[k]["pass"], vd[k]["pass"]))
    pairs += [
        (round(ev["D1"]["delta1"], 6), round(vd["D1"]["delta1"], 6)),
        (round(ev["D1"]["lb90"], 6), round(vd["D1"]["lb90"], 6)),
        (ev["D1"]["best_point"], vd["D1"]["best_point"]),
        (ev["D2"]["n_hypotheses"], vd["D2"]["n_hypotheses"]),
        (ev["D3"]["delta3"], vd["D3"]["delta3"]),
        (round(ev["S1"]["rlf_ratio_ub90"], 6), round(vd["S1"]["rlf_ratio_ub90"], 6)),
    ]
    for i, (a, b) in enumerate(pairs):
        if a != b:
            raise AssertionError(f"FD and cs_verdict.json disagree (check {i}): {a!r} vs {b!r}")


def rows() -> list[dict]:
    ev = ST.fd()["certsafe"]["eval"]
    vd = TM.load(TM.CS_VERDICT_PARTS)
    _check(ev, vd)
    a = ev["arms"][HERO]
    ret, cap = TM.eligibility()
    n_maps = len(vd["D1"]["best_selected_share"])
    hyp = ev["D2"]["hypotheses"]
    n_rej = sum(h["rejected"] for h in hyp.values())
    if vd["D3"]["ref"] != "never_sleep":
        raise AssertionError(f"D3 reference changed: {vd['D3']['ref']}")
    guards = "; ".join(f"{lab} {TM.f(a['guard_ratio'][k])}" for k, lab in TM.GUARDS.items())
    out = [
        {
            "criterion": "K0, K0n",
            "rule": "Validity of the PMRT map, carried over from Study 2",
            "value": "--",
            "ok": vd["K0"] and vd["K0n"],
        },
        {
            "criterion": TM.crit("E"),
            "rule": (
                f"Edge-certified PMRT referee eligible: energy retention {TM.ge(ret)} and every guard "
                f"ratio {TM.le(cap)} (point estimates)"
            ),
            "value": f"retention {TM.f(a['retention'])}; guard ratios {guards}",
            "ok": ev["E"],
        },
        {
            "criterion": "D1",
            "rule": (
                f"$R^*$ minus the best gated associational map's $R^*$ (of {n_maps}, re-selected per "
                f"resample) {TM.ge(TM.t(vd['D1']['margin']))} and LB90 $>$ 0"
            ),
            "value": (
                f"{TM.fs(ev['D1']['delta1'])} {TM.ci(*ev['D1']['ci90'], 3)} vs "
                f"the {TM.inline(TM.map_name(ev['D1']['best_point']))} map"
            ),
            "ok": ev["D1"]["pass"],
        },
        {
            "criterion": "D2",
            "rule": (
                f"Beats each gated associational map: one-sided paired bootstrap per distinct "
                f"map signature (identical rejection rules pooled), Holm at {TM.t(vd['D2']['alpha'])}, all rejected"
            ),
            "value": f"{n_rej} of {ev['D2']['n_hypotheses']} rejected",
            "ok": ev["D2"]["pass"],
        },
        {
            "criterion": "D3",
            "rule": "$R^*$ minus the never-sleep rule's $R^*$ $>$ 0 and paired LB90 $>$ 0",
            "value": (
                f"{TM.fs(ev['D3']['delta3'])} {TM.ci(*ev['D3']['ci90'], 3)}; identical to never-sleep on "
                f"{ev['identical_seeds']} of {ev['n']} seeds"
            ),
            "ok": ev["D3"]["pass"],
        },
        {
            "criterion": TM.crit("S1"),
            "rule": f"Upper 90\\,\\% bound of the RLF ratio {TM.le(TM.t(vd['S1']['cap']))}",
            "value": f"{TM.f(ev['S1']['rlf_ratio_ub90'], 3)}",
            "ok": ev["S1"]["pass"],
        },
    ]
    for r in out:
        r["result"] = TM.result(r.pop("ok"))
    out.append(
        {
            "criterion": "R",
            "rule": "Recovery of the edge-certified PMRT referee (descriptive, no criterion)",
            "value": f"{TM.f(a['R'], 3)} {TM.ci(*a['R_ci90'], 3)}",
            "result": "--",
        }
    )
    out.append(
        {
            "criterion": "Verdict",
            "rule": "Precedence: " + TM.precedence(vd["verdict"]["precedence"]),
            "value": "",
            "result": ev["verdict"],
        }
    )
    return out


def latex(rs: list[dict]) -> list[str]:
    out = TM.header(
        NAME,
        SOURCES,
        [
            f"{ST.fd()['certsafe']['eval']['n']} evaluation seeds; 90 % intervals from the "
            "paired seed bootstrap; LB90 = lower 90 % bound."
        ],
    )
    out += [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l p{5.0cm} p{4.3cm} l@{}}",
        r"\toprule",
        r"Criterion & Rule & Measured & Result \\",
        r"\midrule",
    ]
    for r in rs:
        if r["criterion"] == "R":
            out.append(r"\midrule")
        if r["criterion"] == "Verdict":
            out += [
                r"\midrule",
                rf"Verdict & \multicolumn{{2}}{{p{{9.5cm}}}}{{\raggedright {r['rule']}}} & "
                rf"{r['result']} \\",
            ]
            continue
        out.append(
            rf"{r['criterion']} & \raggedright {r['rule']} & \raggedright {r['value']} & {r['result']} \\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


def write() -> list[str]:
    rs = rows()
    return TM.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

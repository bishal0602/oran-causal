"""Study 2 (discovery from confounded logs and the map-driven referee) criteria with the frozen
rule, the measured value and pass / fail: the discovery stage K0, K0n, G, K1, X1, X2, X3 and its
label precedence, then the referee stage (K0 / K0n carried over, Elig, D1, D2), the descriptive
recovery R and the verdict precedence. Booktabs LaTeX + csv; not a figure (no SPEC): run ``uv run
python -m oran_causal.figures.figs.tab_study2_criteria``. Replaces tab_design_criteria (Table 6) as
the place where Study 2's criteria appear (advisor, 2026-10-12).

Rules and thresholds: the frozen protocol protocols/study2/PROTOCOL.md s.7.1-7.2, with every threshold
read at build time from the analyzer outputs' ``rule`` fields: conf_disc_conf.json (K0, K0n, K1, X2,
X3 ``need``; G rule; the discovery ``label.precedence``) and conf_verdict.json (D1 margin, D2 alpha
and procedure, the D1 map set, the verdict precedence). The eligibility thresholds are stored in no
JSON and are read from the protocol s.7.2 text. Values: FD ``option_a.disc`` (K0, K0n, K1, G, X1-X3,
label) and ``option_a.eval`` (E, D1, D2, verdict, n, arms.MG:PMRT: R, Rstar, R_ci90, retention,
guard_ratio), each checked against conf_disc_conf.json / conf_verdict.json before anything is
written. Re-aggregation: min-max of the K0n family rates and the count of rejected D2 hypotheses;
nothing else is computed. The discovery labels DISC-PASS / DISC-PARTIAL / DISC-FAIL are typeset as
pass / partial / fail, as in the text and Table 3 ("discovery pass").
"""

import re

from oran_causal.figures.data import protocol_note
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _tabm as TM
from oran_causal.figures.paths import protocol

NAME = "tab_study2_criteria"
PROTOCOL_PARTS = ("study2", "PROTOCOL.md")  # s.7.2 eligibility definition (text)
SOURCES = [ST.SRC_FD, ST.SRC_CONF, TM.SRC_CONF_VERDICT, protocol_note(*PROTOCOL_PARTS)]
HERO = "MG:PMRT"


def eligibility() -> tuple[str, str]:
    """(retention floor, guard-ratio cap) of 'eligible' in the frozen Study 2 protocol s.7.2."""
    txt = protocol(*PROTOCOL_PARTS).read_text(encoding="utf-8")
    m = re.search(
        r"\*\*eligible\*\* = retention >= (\.\d+) and every guard ratio <= (\d\.\d+)", txt
    )
    if m is None:
        raise AssertionError("eligibility rule not found in the Study 2 protocol s.7.2")
    return "0" + m.group(1), m.group(2)


def _check(ds: dict, ev: dict, cd: dict, vd: dict) -> None:
    """FD (the report's asserted figure data) and the analyzer outputs must agree on every value shown."""
    hyp_fd = {k: v["rejected"] for k, v in ev["D2"]["hypotheses"].items()}
    hyp_vd = {k: v["rejected"] for k, v in vd["D2"]["hypotheses"].items()}
    pairs = [
        (ds["K0"]["pass"], cd["K0"]["pass"]),
        (ds["K0"]["n_reject"], cd["K0"]["n_reject"]),
        (ds["K0"]["m"], cd["K0"]["m"]),
        (ds["K0"]["n_declared"], cd["K0"]["n_declared"]),
        (ds["K0n"]["pass"], cd["K0n"]["pass"]),
        (ds["K0n"]["rate05"], round(cd["K0n"]["rate05"], 8)),
        (ds["K0n"]["n_variants_with_decl"], cd["K0n"]["n_variants_with_decl"]),
        (ds["K1"]["pass"], cd["K1"]["pass"]),
        (ds["K1"]["sleep_units"], cd["K1"]["sleep_units"]),
        (ds["K1"]["sleep_rejects"], cd["K1"]["sleep_rejects"]),
        (ds["G"]["status"], cd["G"]["cell"]["status"]),
        (ds["G"]["sign"], cd["G"]["cell"]["sign"]),
        (round(ds["G"]["mean"], 4), round(cd["G"]["cell"]["mean"], 4)),
        (ds["X1"]["pass"], cd["X"]["X1"]["pass"]),
        (ds["X1"]["premise"], cd["X"]["X1"]["premise"]),
        (round(ds["X2"]["precision"], 6), round(cd["X"]["X2"]["precision"], 6)),
        (ds["X2"]["sign_accuracy"], cd["X"]["X2"]["sign_accuracy"]),
        (ds["X2"]["n_scored"], cd["X"]["X2"]["n_scored"]),
        (ds["X2"]["n_true"], cd["X"]["X2"]["n_true"]),
        (ds["X3"]["chain_hits"], cd["X"]["X3"]["chain_hits"]),
        (ds["X3"]["chain_true"], cd["X"]["X3"]["chain_true"]),
        (ds["label"], cd["label"]["label"]),
        (ds["label"], vd["discovery_label"]),
        (cd["K0"]["pass"], vd["K0"]),
        (cd["K0n"]["pass"], vd["K0n"]),
        (ev["E"], vd["E"]["pass"]),
        (ev["E"], ev["arms"][HERO]["eligible"]),
        (ev["n"], vd["n_seeds"]),
        (ev["D1"]["pass"], vd["D1"]["pass"]),
        (round(ev["D1"]["delta1"], 6), round(vd["D1"]["delta1"], 6)),
        (round(ev["D1"]["lb90"], 6), round(vd["D1"]["lb90"], 6)),
        (ev["D1"]["best_point"], vd["D1"]["best_point"]),
        (ev["D2"]["pass"], vd["D2"]["pass"]),
        (ev["D2"]["n_hypotheses"], vd["D2"]["n_hypotheses"]),
        (hyp_fd, hyp_vd),
        (ev["verdict"], vd["verdict"]["verdict"]),
    ]
    for i, (a, b) in enumerate(pairs):
        if a != b:
            raise AssertionError(
                f"FD and the Study 2 analyzer outputs disagree (check {i}): {a!r} vs {b!r}"
            )


def _label(lab: str) -> str:
    """A discovery label as the paper prints it (DISC-PASS -> PASS); the referee labels are unchanged."""
    return lab.replace("DISC-", "")


def rows() -> list[dict]:
    oa = ST.fd()["option_a"]
    ds, ev = oa["disc"], oa["eval"]
    cd = TM.load(ST.CONF_PARTS)
    vd = TM.load(TM.CONF_VERDICT_PARTS)
    _check(ds, ev, cd, vd)
    k0, k0n, k1 = (cd[k]["rule"] for k in ("K0", "K0n", "K1"))
    x1, x2, x3 = (cd["X"][k] for k in ("X1", "X2", "X3"))
    if x3["need"] != min(3, x3["chain_true"]):
        raise AssertionError("X3 need is not min(3, |C*|)")
    if (
        cd["G"]["cell"]["family"] != "sleep"
        or cd["G"]["cell"]["relation"] != "nbr"
        or cd["G"]["cell"]["kpi"] != "pv"
    ):
        raise AssertionError("G is no longer the premise edge sleep -> nbr pv")
    if not (x1["premise"]["declared"] and x1["premise"]["sign"] == 1):
        raise AssertionError("X1 premise not declared (+): the value text says it is")
    g = cd["G"]["cell"]
    a = ev["arms"][HERO]
    ret, cap = eligibility()
    n_maps = len(vd["D1"]["best_selected_share"])
    n_rej = sum(h["rejected"] for h in ev["D2"]["hypotheses"].values())
    fam = cd["K0n"]["rate05_family"].values()
    lo, hi = min(fam), max(fam)
    guards = "; ".join(f"{lab} {TM.f(a['guard_ratio'][k])}" for k, lab in TM.GUARDS.items())
    disc = [
        {
            "criterion": "K0",
            "what": "placebo validity",
            "rule": (
                f"Placebo logs, {cd['K0']['m']} hypotheses: binomial tail of the rejections at "
                f"$p$ {TM.le(TM.t(k0['alpha']))} is {TM.ge(TM.t(k0['binom_level']))}, and {TM.le(k0['max_decl'])} "
                f"declaration"
            ),
            "value": (
                f"{ds['K0']['n_reject']} of {ds['K0']['m']} at $p$ {TM.le(TM.t(k0['alpha']))} (tail "
                f"{TM.f(cd['K0']['binom_tail'], 4)}); {ds['K0']['n_declared']} declared"
            ),
            "ok": ds["K0"]["pass"],
        },
        {
            "criterion": "K0n",
            "what": "null-outcome validity",
            "rule": (
                f"{cd['K0n']['n_variants']} null-outcome variants (shifted KPI series): rate of $p$ "
                f"{TM.le(TM.t(k0n['alpha']))} is {TM.le(TM.t(k0n['rate']))}, every family "
                f"{TM.le(TM.t(k0n['family_rate']))}, {TM.le(k0n['max_var_decl'])} variants with a declaration"
            ),
            "value": (
                f"rate {TM.f(ds['K0n']['rate05'], 4)} (families {TM.f(lo, 3)}--{TM.f(hi, 3)}); "
                f"{ds['K0n']['n_variants_with_decl']} of {cd['K0n']['n_variants']} variants declared"
            ),
            "ok": ds["K0n"]["pass"],
        },
        {
            "criterion": "G",
            "what": "premise in ground truth",
            "rule": "Ground truth labels the premise edge sleep $\\rightarrow$ neighbour protected-slice violations "
            "true ($+$)",
            "value": f"true ($+$), mean {TM.fs(g['mean'], 2)} {TM.ci(*g['ci'])}, {g['n_eps']} episodes",
            "ok": cd["G"]["pass"],
        },
        {
            "criterion": "K1",
            "what": "support",
            "rule": (
                f"{TM.ge(TM.n(k1['min_sleep_units']))} tested sleep units with "
                f"{TM.ge(k1['min_sleep_rejects'])} logged rejections in the discovery logs"
            ),
            "value": f"{TM.n(ds['K1']['sleep_units'])} units, {ds['K1']['sleep_rejects']} rejections",
            "ok": ds["K1"]["pass"],
        },
        {
            "criterion": "X1",
            "what": "premise recovered",
            "rule": "PMRT declares the premise edge ($+$) on the pooled discovery logs",
            "value": f"declared ($+$), $p$ = {TM.f(x1['premise']['p'], 4)}",
            "ok": ds["X1"]["pass"],
        },
        {
            "criterion": "X2",
            "what": "map quality",
            "rule": (
                f"PMRT's declarations on the {', '.join(x2['rule']['kpis'][:-1])} and {x2['rule']['kpis'][-1]} KPIs, scored against the ground "
                f"truth (indeterminate excluded): precision {TM.ge(TM.t(x2['rule']['precision']))} and sign "
                f"accuracy {TM.ge(TM.t(x2['rule']['sign_acc']))}"
            ),
            "value": (
                f"precision {TM.f(ds['X2']['precision'])} ({ds['X2']['n_true']} of {ds['X2']['n_scored']} "
                f"scored); sign accuracy {TM.f(ds['X2']['sign_accuracy'])}"
            ),
            "ok": x2["pass"],
        },
        {
            "criterion": "X3",
            "what": "chain",
            "rule": (
                "$\\ge$ min(3, $|C^*|$) of the chain edges true in the ground truth ($C^*$) declared with the "
                "ground-truth sign"
            ),
            "value": f"{ds['X3']['chain_hits']} of {ds['X3']['chain_true']}",
            "ok": x3["pass"],
        },
    ]
    referee = [
        {
            "criterion": "K0, K0n",
            "what": "validity",
            "rule": "Validity of the PMRT map, from the discovery stage",
            "value": "--",
            "ok": vd["K0"] and vd["K0n"],
        },
        {
            "criterion": TM.crit("E"),
            "what": "eligibility",
            "rule": (
                f"Map-driven PMRT referee eligible: energy retention {TM.ge(ret)} and every guard ratio "
                f"{TM.le(cap)} (point estimates)"
            ),
            "value": f"retention {TM.f(a['retention'])}; guard ratios {guards}",
            "ok": ev["E"],
        },
        {
            "criterion": "D1",
            "what": "best associational map",
            "rule": (
                f"$R^*$ minus the best associational map's $R^*$ (of {n_maps}, re-selected per resample) "
                f"{TM.ge(TM.t(vd['D1']['margin']))} and LB90 $>$ 0"
            ),
            "value": (
                f"{TM.fs(ev['D1']['delta1'])} {TM.ci(*ev['D1']['ci90'], 3)} vs the "
                f"{TM.inline(TM.map_name(ev['D1']['best_point']))} map"
            ),
            "ok": ev["D1"]["pass"],
        },
        {
            "criterion": "D2",
            "what": "each associational map",
            "rule": (
                f"Beats each associational map: one-sided paired bootstrap per distinct map signature (identical "
                f"rejection rules pooled), Holm at {TM.t(vd['D2']['alpha'])}, all rejected"
            ),
            "value": f"{n_rej} of {ev['D2']['n_hypotheses']} rejected",
            "ok": ev["D2"]["pass"],
        },
    ]
    for r in disc + referee:
        r["result"] = TM.result(r.pop("ok"))
    out = disc
    out.append(
        {
            "criterion": "Verdict",
            "what": "discovery",
            "rule": "Discovery. Precedence: " + TM.precedence(_label(cd["label"]["precedence"])),
            "value": "",
            "result": _label(ds["label"]),
        }
    )
    out += referee
    out.append(
        {
            "criterion": "R",
            "what": "recovery",
            "rule": "Recovery of the map-driven PMRT referee (descriptive, no criterion)",
            "value": (
                f"{TM.f(a['R'], 3)} {TM.ci(*a['R_ci90'], 3)}; $R^*$ = {TM.f(a['Rstar'], 0)} "
                f"(not eligible)"
                if not a["eligible"]
                else f"{TM.f(a['R'], 3)} {TM.ci(*a['R_ci90'], 3)}"
            ),
            "result": "--",
        }
    )
    out.append(
        {
            "criterion": "Verdict",
            "what": "referee",
            "rule": "Referee. Precedence: " + TM.precedence(vd["verdict"]["precedence"]),
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
            f"{ST.fd()['option_a']['eval']['n']} evaluation seeds; 90 % intervals from the "
            "paired seed bootstrap; LB90 = lower 90 % bound; G: 95 % interval of the ground "
            "truth. Discovery labels DISC-* printed without the prefix."
        ],
    )
    out += [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l p{5.0cm} p{4.3cm} l@{}}",
        r"\toprule",
        r"Criterion & Rule & Measured & Result \\",
        r"\midrule",
    ]
    for i, r in enumerate(rs):
        if r["criterion"] == "Verdict":
            out += [
                r"\midrule",
                rf"Verdict & \multicolumn{{2}}{{p{{9.5cm}}}}{{\raggedright {r['rule']}}} & "
                rf"{r['result']} \\",
            ]
            if i < len(rs) - 1:
                out.append(r"\midrule")
            continue
        if r["criterion"] == "R":
            out.append(r"\midrule")
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

"""Study 1 (discovery from randomised logs) criteria K0, K0n, G, K1, P1, P2, P3, S: the frozen rule,
the measured value and pass / fail, plus the verdict and its precedence. Booktabs LaTeX + csv; not a
figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_study1_criteria``.

Values: FD ``v4.*`` (figures_data.json: K0, K0n, G via gt.premise, K1, P1, P2, P3_values, S,
verdict). Rule thresholds, the per-criterion pass flags FD does not copy (K1, S, P3, G) and the
precedence: the analyzer verdict ``v4_verdict.json`` (``<criterion>.rule``, ``.pass``,
``verdict.precedence``). Every FD value is checked against the verdict file before anything is
written. Re-aggregation: min-max over the five 120-episode slices (K1 units / rejects, P1 chain
hits) and over the four families (K0n); nothing else is computed.
"""

from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _tabm as TM

NAME = "tab_study1_criteria"
SOURCES = [ST.SRC_FD, TM.SRC_V4_VERDICT]
BASELINES = ["corr", "granger", "granger_by"]  # P2 baselines, protocol order


def _range(vals, nd=None) -> str:
    lo, hi = min(vals), max(vals)
    fmt = (lambda x: TM.f(x, nd)) if nd is not None else TM.n
    return fmt(lo) if lo == hi else f"{fmt(lo)}--{fmt(hi)}"


def _check(fd: dict, vd: dict) -> None:
    """FD (the report's asserted figure data) and the analyzer verdict must agree on every value shown."""
    pairs = [
        (fd["K0"]["n_reject"], vd["K0"]["n_reject"]),
        (fd["K0"]["m"], vd["K0"]["m"]),
        (fd["K0"]["n_declared"], vd["K0"]["n_declared"]),
        (fd["K0"]["pass"], vd["K0"]["pass"]),
        (fd["K0n"]["rate05"], round(vd["K0n"]["rate05"], 8)),
        (fd["K0n"]["pass"], vd["K0n"]["pass"]),
        (fd["K0n"]["n_variants_with_decl"], vd["K0n"]["n_variants_with_decl"]),
        (fd["G"], vd["G"]["pass"]),
        (fd["gt"]["premise"]["status"], vd["G"]["cell"]["status"]),
        (fd["P1_pass"], vd["P1"]["pass"]),
        (fd["P2_pass"], vd["P2"]["pass"]),
        (fd["P3_values"]["ov_precision"], vd["P3"]["values"]["overall_precision"]),
        (fd["P3_values"]["sign_acc"], vd["P3"]["values"]["sign_accuracy"]),
        (fd["P3_values"]["chain_hits"], vd["P3"]["values"]["chain_hits"]),
        (fd["S"]["pooled"], vd["S"]["pooled"]),
        (fd["S"]["mean120"], vd["S"]["mean120"]),
        (fd["verdict"], vd["verdict"]["verdict"]),
    ]
    pairs += [(fd["K1"][s]["sleep_units"], vd["K1"]["slices"][s]["sleep_units"]) for s in fd["K1"]]
    pairs += [(fd["P2"][b]["wins60"], vd["P2"]["per_baseline"][b]["wins60"]) for b in BASELINES]
    pairs += [(fd["P2"][b]["wins120"], vd["P2"]["per_baseline"][b]["wins120"]) for b in BASELINES]
    for i, (a, b) in enumerate(pairs):
        if a != b:
            raise AssertionError(f"FD and v4_verdict.json disagree (check {i}): {a!r} vs {b!r}")


def rows() -> list[dict]:
    fd = ST.fd()["v4"]
    vd = TM.load(TM.V4_VERDICT_PARTS)
    _check(fd, vd)
    k0, k0n, k1, p1, p2, p3, s = (vd[k]["rule"] for k in ("K0", "K0n", "K1", "P1", "P2", "P3", "S"))
    prem = fd["gt"]["premise"]
    n120 = len(vd["K1"]["slices"])
    n60 = vd["P2"]["per_baseline"][BASELINES[0]]["n60"]
    pb = vd["P2"]["per_baseline"]
    fam = fd["K0n"]["rate05_family"].values()
    out = [
        {
            "criterion": "K0",
            "what": "placebo validity",
            "rule": (
                f"Placebo logs, {fd['K0']['m']} hypotheses: binomial tail of the rejections at "
                f"$p$ {TM.le(TM.t(k0['alpha']))} is {TM.ge(TM.t(k0['binom_level']))}, and {TM.le(k0['max_decl'])} declaration"
            ),
            "value": (
                f"{fd['K0']['n_reject']} of {fd['K0']['m']} at $p$ {TM.le(TM.t(k0['alpha']))} (tail "
                f"{TM.f(fd['K0']['binom_tail'], 4)}); {fd['K0']['n_declared']} declared"
            ),
            "ok": fd["K0"]["pass"],
        },
        {
            "criterion": "K0n",
            "what": "null-outcome validity",
            "rule": (
                f"{vd['K0n']['n_variants']} null-outcome variants (shifted KPI series): rate of $p$ "
                f"{TM.le(TM.t(k0n['alpha']))} is {TM.le(TM.t(k0n['rate']))}, every family {TM.le(TM.t(k0n['family_rate']))}, "
                f"{TM.le(k0n['max_var_decl'])} variants with a declaration"
            ),
            "value": (
                f"rate {TM.f(fd['K0n']['rate05'], 4)} (families {_range(fam, 3)}); "
                f"{fd['K0n']['n_variants_with_decl']} of {fd['K0n']['n_variants']} variants declared"
            ),
            "ok": fd["K0n"]["pass"],
        },
        {
            "criterion": "G",
            "what": "premise in ground truth",
            "rule": "Ground truth labels the premise edge sleep $\\rightarrow$ neighbour protected-slice violations "
            "true ($+$)",
            "value": (
                f"true ($+$), mean {TM.fs(prem['mean'], 2)} {TM.ci(*prem['ci'])}, "
                f"{prem['n_eps']} episodes"
            ),
            "ok": vd["G"]["pass"],
        },
        {
            "criterion": "K1",
            "what": "support",
            "rule": (
                f"{TM.ge(k1['min_sleep_units'])} tested sleep units with {TM.ge(k1['min_sleep_rejects'])} "
                f"logged rejections, in every 120-episode subsample"
            ),
            "value": (
                f"{_range([v['sleep_units'] for v in vd['K1']['slices'].values()])} units, "
                f"{_range([v['sleep_rejects'] for v in vd['K1']['slices'].values()])} rejections "
                f"({n120} subsamples)"
            ),
            "ok": vd["K1"]["pass"],
        },
        {
            "criterion": "P1",
            "what": "chain at 120 episodes",
            "rule": (
                f"In {TM.ge(p1['min_slices'])} of {vd['P1']['n_slices']} 120-episode subsamples: premise declared "
                f"($+$) and {TM.ge(p1['chain_min'])} chain edges with the ground-truth sign"
            ),
            "value": (
                f"{vd['P1']['n_ok']} of {vd['P1']['n_slices']} subsamples; chain "
                f"{_range([x['chain_hits'] for x in vd['P1']['slices']])} of "
                f"{_range([x['chain_true'] for x in vd['P1']['slices']])} in each"
            ),
            "ok": fd["P1_pass"],
        },
        {
            "criterion": "P2",
            "what": "small-subsample F1",
            "rule": (
                f"PMRT indirect F1 {TM.ge("the baseline's")} in {TM.ge(p2['min_60'])} of {n60} 60-episode and "
                f"{TM.ge(p2['min_120'])} of {n120} 120-episode subsamples, for each baseline"
            ),
            "value": "; ".join(
                f"{TM.baseline_name(b)} {pb[b]['wins60']}/{pb[b]['n60']}, "
                f"{pb[b]['wins120']}/{pb[b]['n120']}"
                for b in BASELINES
            ),
            "ok": fd["P2_pass"],
        },
        {
            "criterion": "P3",
            "what": "pooled discovery",
            "rule": (
                f"All {fd['slices']['n_episodes']['pooled']} episodes: premise declared ($+$), "
                f"{TM.ge(p3['chain_min'])} chain edges, overall precision {TM.ge(TM.t(p3['precision']))}, "
                f"sign accuracy {TM.ge(TM.t(p3['sign_acc']))}"
            ),
            "value": (
                f"premise ($+$); chain {fd['P3_values']['chain_hits']} of {vd['P3']['values']['chain_true']}; "
                f"precision {TM.f(fd['P3_values']['ov_precision'])}; sign accuracy "
                f"{TM.f(fd['P3_values']['sign_acc'])}"
            ),
            "ok": vd["P3"]["pass"],
        },
        {
            "criterion": TM.crit("S"),
            "what": "sign",
            "rule": (
                f"Overall sign accuracy {TM.ge(TM.t(s['sign_acc']))} pooled and as the mean over the "
                f"120-episode subsamples"
            ),
            "value": f"{TM.f(fd['S']['pooled'])} pooled; {TM.f(fd['S']['mean120'])} mean",
            "ok": vd["S"]["pass"],
        },
    ]
    if not fd["P3_parts"]["premise_declared_plus"]:
        raise AssertionError("P3 premise not declared: the value text says it is")
    for r in out:
        r["result"] = TM.result(r.pop("ok"))
    out.append(
        {
            "criterion": "Verdict",
            "what": "",
            "rule": "Precedence: " + TM.precedence(vd["verdict"]["precedence"]),
            "value": "",
            "result": fd["verdict"],
        }
    )
    return out


def latex(rs: list[dict]) -> list[str]:
    out = TM.header(
        NAME,
        SOURCES,
        ["Rule thresholds and pass flags from the analyzer verdict; values from FD v4."],
    )
    out += [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l p{5.0cm} p{4.3cm} l@{}}",
        r"\toprule",
        r"Criterion & Rule & Measured & Result \\",
        r"\midrule",
    ]
    for r in rs:
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

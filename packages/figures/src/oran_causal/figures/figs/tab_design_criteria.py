"""Criteria definitions of Studies 1-3 (discovery K0, K0n, G, K1, P1, P2, P3, S; referee E, D1, D2,
D3, S1; paper labels via _tabm.crit) and Study 2's verdict precedence (Studies 1 and 3 carry theirs
in their own criteria tables). Booktabs LaTeX + csv; not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_design_criteria``. Kept separate from tab_design_studies (PAPER_PLAN
s.4 [DECIDE] merge).

Definitions are text, shortened from the frozen protocols (E6P_DISCOVERY_PROTOCOL_V4.md s.5,
E6P_CONFOUNDED_PROTOCOL.md s.7, E6P_CERTSAFE_PROTOCOL.md s.7) and TERMINOLOGY s.7-8; no new claims.
Every threshold is read at build time from the analyzer JSONs' ``rule`` fields: v4_verdict.json (K0,
K0n, K1, P1, P2, P3, S; P2 slice counts), conf_disc_conf.json (Study 2 K1 thresholds, K0 hypothesis
count), conf_verdict.json (D1 margin, D2 alpha, precedence) and cs_verdict.json (D1 margin, D2
alpha, S1 cap, precedence); the eligibility thresholds are in no JSON and are read from the Study 3
protocol s.7 text. Study 2 and Study 3 must share the D1 margin and the D2 alpha (checked). The
Study 2 discovery criteria X1-X3 are not in the planned list and are not shown.
"""

from oran_causal.figures.data import protocol_note
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _tabm as TM

NAME = "tab_design_criteria"
SOURCES = [
    TM.SRC_V4_VERDICT,
    ST.SRC_CONF,
    TM.SRC_CONF_VERDICT,
    TM.SRC_CS_VERDICT,
    TM.SRC_CS_PROTOCOL,
    protocol_note("study1", "E6P_DISCOVERY_PROTOCOL_V4.md"),
    protocol_note("study2", "E6P_CONFOUNDED_PROTOCOL.md"),
    "TERMINOLOGY.md",
]


def rows() -> list[dict]:
    v4 = TM.load(TM.V4_VERDICT_PARTS)
    cd = TM.load(ST.CONF_PARTS)
    cv = TM.load(TM.CONF_VERDICT_PARTS)
    cs = TM.load(TM.CS_VERDICT_PARTS)
    k0, k0n, k1, p1, p2, p3, s = (v4[k]["rule"] for k in ("K0", "K0n", "K1", "P1", "P2", "P3", "S"))
    if cd["K0"]["rule"] != k0 or cd["K0n"]["rule"] != k0n or cd["K0"]["m"] != v4["K0"]["m"]:
        raise AssertionError("Study 2 K0 / K0n rule differs from Study 1: split the rows")
    if cv["D1"]["margin"] != cs["D1"]["margin"] or cv["D2"]["alpha"] != cs["D2"]["alpha"]:
        raise AssertionError("Study 2 and Study 3 D1 / D2 constants differ: split the rows")
    k1b = cd["K1"]["rule"]
    ret, cap = TM.eligibility()
    pb = v4["P2"]["per_baseline"]["corr"]
    nsl = v4["P1"]["n_slices"]
    guards = ", ".join(TM.GUARDS.values())
    out = [
        (
            "K0",
            "1, 2",
            f"Placebo validity. On placebo logs (every request applied whatever its logged mode, so no "
            f"effect exists), with $m$ = {v4['K0']['m']} hypotheses and $r$ of them at $p$ "
            f"{TM.le(TM.t(k0['alpha']))}: P(Binomial($m$, {TM.t(k0['alpha'])}) $\\ge r$) "
            f"{TM.ge(TM.t(k0['binom_level']))}, and "
            f"{TM.le(k0['max_decl'])} declaration.",
        ),
        (
            "K0n",
            "1, 2",
            f"Null-outcome validity. {v4['K0n']['n_variants']} variants with KPI series cyclically shifted "
            f"across episodes: rate of $p$ {TM.le(TM.t(k0n['alpha']))} is {TM.le(TM.t(k0n['rate']))} "
            f"pooled and {TM.le(TM.t(k0n['family_rate']))} in every family; "
            f"{TM.le(k0n['max_var_decl'])} variants with a declaration.",
        ),
        (
            "G",
            "1, 2",
            "Premise in the ground truth: the knockout ground truth labels sleep $\\rightarrow$ neighbour "
            "protected-slice violations true ($+$).",
        ),
        (
            "K1",
            "1, 2",
            f"Support. Study 1: {TM.ge(k1['min_sleep_units'])} tested sleep units with "
            f"{TM.ge(k1['min_sleep_rejects'])} logged rejections in every 120-episode slice. Study 2: "
            f"{TM.ge(k1b['min_sleep_units'])} sleep units with {TM.ge(k1b['min_sleep_rejects'])} logged "
            f"rejections.",
        ),
        (
            "P1",
            "1",
            f"Chain at 120 episodes: in {TM.ge(p1['min_slices'])} of {nsl} slices, the premise edge declared "
            f"($+$) and {TM.ge(p1['chain_min'])} chain edges declared with the ground-truth sign.",
        ),
        (
            "P2",
            "1",
            f"Small-slice F1: PMRT indirect F1 {TM.ge("the baseline's")} in "
            f"{TM.ge(p2['min_60'])} of {pb['n60']} 60-episode and {TM.ge(p2['min_120'])} of {pb['n120']} "
            f"120-episode slices, separately for correlation, Granger and Granger + BY.",
        ),
        (
            "P3",
            "1",
            f"Pooled discovery on all evaluation episodes: premise declared ($+$), {TM.ge(p3['chain_min'])} "
            f"chain edges with the ground-truth sign, overall precision {TM.ge(TM.t(p3['precision']))} and sign "
            f"accuracy {TM.ge(TM.t(p3['sign_acc']))}.",
        ),
        (
            TM.crit("S"),
            "1",
            f"Sign: overall sign accuracy {TM.ge(TM.t(s['sign_acc']))} pooled and as the mean over the "
            f"120-episode slices.",
        ),
        (
            TM.crit("E"),
            "2, 3",
            f"The PMRT referee (Study 2: map-driven; Study 3: edge-certified) is eligible: energy retention "
            f"{TM.ge(ret)} and every guard ratio ({guards} violations, each relative to "
            f"accept-all) {TM.le(cap)}, point estimates.",
        ),
        (
            "D1",
            "2, 3",
            f"Beats the best associational map: $R^*$(PMRT) minus the largest $R^*$ of the associational "
            f"maps {TM.ge(TM.t(cs['D1']['margin']))}, and its lower 90\\,\\% bound (LB90) $>$ 0, the largest "
            f"map re-selected in every bootstrap resample.",
        ),
        (
            "D2",
            "2, 3",
            f"Beats each associational map: one-sided paired bootstrap $p$ per distinct map signature "
            f"(identical rejection rules pooled), Holm at {TM.t(cs['D2']['alpha'])}; every hypothesis "
            f"rejected.",
        ),
        (
            "D3",
            "3",
            "Beats the static never-sleep rule: $R^*$ difference $>$ 0 and paired LB90 $>$ 0.",
        ),
        (
            TM.crit("S1"),
            "3",
            f"Secondary safety check: upper 90\\,\\% bound of the RLF ratio {TM.le(TM.t(cs['S1']['cap']))}; "
            f"does not change the verdict label.",
        ),
        # Slimmed 2026-10-11 (outline v2 s.5, review: Table 4 too heavy): the R / R* notation row
        # moved to the Section 5 text; the Study 1 and Study 3 precedence rows are in
        # tab_study1_criteria / tab_study3_criteria, so only Study 2 (no criteria table of its own)
        # keeps its precedence here.
        ("Verdict", "2", "Referee: " + TM.precedence(cv["verdict"]["precedence"])),
    ]
    return [{"criterion": c, "study": st, "definition": d} for c, st, d in out]


def latex(rs: list[dict]) -> list[str]:
    out = TM.header(NAME, SOURCES, ["'>' in the verdict rows = takes precedence over."])
    out += [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l l p{10.1cm}@{}}",
        r"\toprule",
        r"Criterion & Study & Definition \\",
        r"\midrule",
    ]
    prev = None
    for r in rs:
        block = (
            "verdict"
            if r["criterion"] == "Verdict"
            else (
                "referee"
                if r["study"] != "1" and "K" not in r["criterion"] and r["criterion"] != "G"
                else "disc"
            )
        )
        if prev is not None and block != prev:
            out.append(r"\midrule")
        prev = block
        out.append(
            rf"{r['criterion']} & {r['study']} & \raggedright {r['definition']} \tabularnewline"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


def write() -> list[str]:
    rs = rows()
    return TM.write(NAME, latex(rs), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

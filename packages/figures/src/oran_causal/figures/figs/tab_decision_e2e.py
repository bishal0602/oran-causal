"""Post hoc analysis X9, end-to-end decision table (Section 6.3; booktabs LaTeX + csv): every Study
3 arm on the 160 evaluation seeds, static rules, map-driven referees and the gated associational
maps, with recovery R, the paired difference in R against never-sleep, energy retention, the four
guard ratios, the request-level rejection rate and eligibility. Not a figure (no SPEC): run ``uv run
python -m oran_causal.figures.figs.tab_decision_e2e``.

Source: posthoc_x9_x16/revision/x9_decision_e2e.json (X9, posthoc protocol amendment 2026-10-11; the script
reproduced the frozen cs_eval.json values for all 26 arms before writing), key ``study3.rows``: per
arm ``R``, ``R_ci90``, ``retention``, ``guard_ratio``, ``guard_ratio_ci90``, ``rejection_rate``,
``eligible``, ``alias_of``, ``dR_vs_never_sleep`` (+ ``_ci90``). Aliased arms (identical seed
records to their target) are not rows; the caption note names them. Post hoc, descriptive, no
verdict; Study 3 is unchanged.

The three random-rejection rows of post hoc check X4 form a fourth group, read from
posthoc_x4/x4/x4_tables.json (the artifact of tab_appC_x4):
``rows.<arm>.{defer_rate, R, retention, guard_ratio, eligible}`` and
``contrasts.<arm>.never_sleep.{dR, ci90}`` (random-rejection referee minus never-sleep, 90 % paired
seed bootstrap). X4 ran on another host; its stored accept-all and never-sleep R must equal X9's
within 1e-3 (the host note: |dR| <= 4e-4).

A Study 2 block comes first,
so that the primary metric and the guards of both studies sit in one table. Same columns, read from
X9 key ``study2.rows`` (the stored option (a) EVAL arms with the same fields); every value the Study
2 figure shows (R, retention, guard ratios, eligibility) is checked against its source, FD
``option_a.eval.arms``, before writing. Study 2 has no gate, so its map arms are the ungated
referees; as in the Study 3 block, aliased arms and placebo-calibrated thresholds are not rows. The
random-rejection rows ran on the Study 3 seeds and stay in the Study 3 block.

A "Slice Guarantee only" static row in both blocks, the R
= 1 reference of recovery, so that its retention (about 0, not eligible) is printed beside
never-sleep. X9 has no such row, so it is read from FD ``option_a.eval.arms`` /
``certsafe.eval.arms`` key ``sub:SliceGuarantee`` and checked against the analyzer outputs
conf_eval.json / cs_eval.json. Neither stores its rejection rate or a paired difference against
never-sleep, so Rej. and Delta R are dashes.
"""

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _taba as A

NAME = "tab_decision_e2e"
X9 = ("posthoc_x9_x16", "revision", "x9_decision_e2e.json")
X4 = ("posthoc_x4", "x4", "x4_tables.json")
SOURCES = [source_note(*X9), source_note(*X4), ST.SRC_FD + " (sub:SliceGuarantee)"]
RANDOM = ["X4:rand@0.25", "X4:rand@0.352", "X4:rand@0.5"]
GUARDS = [("svr", "SVR"), ("nonprot_embb_viol", "eMBB"), ("ll_viol", "LL"), ("rlf", "RLF")]
SG = "sub:SliceGuarantee"  # the R = 1 static rule, from FD (not in X9)
GROUPS = [
    ("Static rules", ["noarb", "never_sleep", SG]),
    ("Map-driven referees", ["MG:PMRT", "CS:PMRT", "CS:GT"]),
    ("Edge-certified referees on associational maps (DEV-tuned thresholds)", None),
    ("Random-rejection referee", RANDOM),
]
GROUPS2 = [
    ("Static rules", ["noarb", "never_sleep", SG]),  # Study 2 block
    ("Map-driven referees", ["MG:PMRT", "MG:GT", "MG:rand"]),
    ("Ungated referees on associational maps (DEV-tuned thresholds)", None),
]
STUDY_ROWS = {
    "study2": r"\textbf{Study~2}, 160 evaluation seeds, no gate",
    "study3": r"\textbf{Study~3}, 160 other evaluation seeds",
}


def label(arm: str, p: dict | None = None) -> str:
    """Row label (paper terms; method names via oran_causal.figures.methods), as tab_appC_x4."""
    if p and arm in p:
        return rf"$p = {A.short_prob(p[arm])}$"
    gate = arm.split(":", maxsplit=1)[0] if ":" in arm else None
    if gate in ("MG", "CS") and arm.split(":")[1] not in ("PMRT", "GT", "rand"):
        n = A.M.name(arm).replace(
            " map", ""
        )  # associational block: the group row names gate and threshold
        return n + (" (untuned)" if arm.endswith(":granger_by") else "")
    if gate in ("MG", "CS"):
        m = A.M.name(arm).replace(" map", "")
        return f"{m}, " + ("ungated" if gate == "MG" else "edge-certified")
    return A.M.name(arm)


def ret(x) -> str:
    """Retention at two decimals; three when it would round to zero (Slice Guarantee only, about
    -0.002).
    """
    if not A.nan(x) and x != 0 and abs(x) < 0.005:
        return A.signed(x, 3, plus=False)
    return A.prob(x, 2)


def _sg_row(study: str, n_seeds: int) -> dict:
    """Slice Guarantee only from FD, checked against the study's analyzer output; no Rej. and
    no Delta R.
    """
    key, parts = {
        "study2": ("option_a", ST.CONF_EVAL_PARTS),
        "study3": ("certsafe", ST.CS_EVAL_PARTS),
    }[study]
    ev = ST.fd()[key]["eval"]
    if ev["n"] != n_seeds:
        raise AssertionError(f"FD {key} seeds {ev['n']} != X9 {study} seeds {n_seeds}")
    f, a = ev["arms"][SG], load_json(*parts)["arms"][SG]
    pairs = (
        [(f["R"], a["R"]), (f["retention"], a["retention"])]
        + [(f["guard_ratio"][g], a["guard_ratio"][g]) for g, _ in GUARDS]
        + list(zip(f["R_ci90"], a["R_ci90"], strict=False))
    )
    if any(abs(x - y) > 1e-6 * max(1.0, abs(y)) for x, y in pairs) or bool(f["eligible"]) != bool(
        a["eligible"]
    ):
        raise AssertionError(f"{study} {SG}: FD differs from {parts[-1]}")
    if f["R"] != 1.0:
        raise AssertionError(f"{study} {SG}: R {f['R']} is not the R = 1 reference")
    row = {
        "study": study,
        "arm": SG,
        "label": label(SG),
        "R": f["R"],
        "R_lo": f["R_ci90"][0],
        "R_hi": f["R_ci90"][1],
        "dR": None,
        "dR_lo": None,
        "dR_hi": None,
        "retention": f["retention"],
        "rejection_rate": None,
        "eligible": f["eligible"],
    }
    for g, _ in GUARDS:
        row[f"guard_{g}"] = f["guard_ratio"][g]
        row[f"guard_{g}_lo"], row[f"guard_{g}_hi"] = f["guard_ratio_ci90"][g]
    return row


def num(x, plus: bool = False) -> str:
    """R-scale number: three decimals, two when |x| >= 1 (the associational arms' large negative R)."""
    return A.signed(x, 3 if abs(x) < 1 else 2, plus=plus)


def _row(arm: str, r: dict, study: str) -> dict:
    row = {
        "study": study,
        "arm": arm,
        "label": label(arm),
        "R": r["R"],
        "R_lo": r["R_ci90"][0],
        "R_hi": r["R_ci90"][1],
        "dR": r["dR_vs_never_sleep"],
        "dR_lo": r["dR_vs_never_sleep_ci90"][0],
        "dR_hi": r["dR_vs_never_sleep_ci90"][1],
        "retention": r["retention"],
        "rejection_rate": r["rejection_rate"],
        "eligible": r["eligible"],
    }
    for g, _ in GUARDS:
        row[f"guard_{g}"] = r["guard_ratio"][g]
        row[f"guard_{g}_lo"], row[f"guard_{g}_hi"] = r["guard_ratio_ci90"][g]
    return row


def rows2() -> tuple[list[dict], list[str]]:
    """Study 2 block: X9 ``study2.rows``, checked against the Study 2 figure's source (FD
    option_a.eval.arms).
    """
    t = load_json(*X9)["study2"]
    by = {r["arm"]: r for r in t["rows"]}
    fd = ST.fd()["option_a"]["eval"]
    if t["n_seeds"] != fd["n"]:
        raise AssertionError(f"X9 Study 2 seeds {t['n_seeds']} != FD {fd['n']}")
    assoc = [
        r["arm"]
        for r in t["rows"]
        if r["group"] == "ungated associational"
        and r["alias_of"] is None
        and not r["arm"].endswith("@plc")
    ]
    if SG in by:
        raise AssertionError(f"X9 Study 2 now has {SG}; read it from X9")
    out = []
    for arm in GROUPS2[0][1] + GROUPS2[1][1] + assoc:
        if arm == SG:
            out.append(_sg_row("study2", t["n_seeds"]))
            continue
        r, f = by[arm], fd["arms"][arm]
        if by[arm]["alias_of"] is not None:
            raise AssertionError(f"{arm} is aliased in Study 2")
        pairs = (
            [(r["R"], f["R"]), (r["retention"], f["retention"])]
            + [(r["guard_ratio"][g], f["guard_ratio"][g]) for g, _ in GUARDS]
            + [(lo, flo) for lo, flo in zip(r["R_ci90"], f["R_ci90"], strict=False)]
        )
        if any(abs(a - b) > 1e-6 * max(1.0, abs(b)) for a, b in pairs) or bool(
            r["eligible"]
        ) != bool(f["eligible"]):
            raise AssertionError(f"{arm}: X9 Study 2 row differs from FD option_a.eval.arms")
        out.append(_row(arm, r, "study2"))
    return out, assoc


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X9)["study3"]
    by = {r["arm"]: r for r in t["rows"]}
    if by["CS:PMRT"]["alias_of"] not in (None, "never_sleep"):
        raise AssertionError("CS:PMRT alias changed")
    assoc = [
        r["arm"]
        for r in t["rows"]
        if r["group"] == "gated associational"
        and r["alias_of"] is None
        and not r["arm"].endswith("@plc")
    ]
    aliased = [(r["arm"], r["alias_of"]) for r in t["rows"] if r["alias_of"] is not None]
    if SG in by:
        raise AssertionError(f"X9 Study 3 now has {SG}; read it from X9")
    out = []
    for arm in GROUPS[0][1] + GROUPS[1][1] + assoc:
        out.append(_sg_row("study3", t["n_seeds"]) if arm == SG else _row(arm, by[arm], "study3"))
    x4 = load_json(*X4)
    if x4["n_seeds"] != t["n_seeds"]:
        raise AssertionError(f"X4 seeds {x4['n_seeds']} != X9 seeds {t['n_seeds']}")
    for arm in ("noarb", "never_sleep"):
        if abs(x4["rows"][arm]["R"] - by[arm]["R"]) > 1e-3:
            raise AssertionError(f"{arm}: X4 R {x4['rows'][arm]['R']} != X9 R {by[arm]['R']}")
    for arm in RANDOM:
        r, c = x4["rows"][arm], x4["contrasts"][arm]["never_sleep"]
        row = {
            "study": "study3",
            "arm": arm,
            "label": "Random rejection, " + label(arm, x4["p"]),
            "R": r["R"],
            "R_lo": r["R_ci90"][0],
            "R_hi": r["R_ci90"][1],
            "dR": c["dR"],
            "dR_lo": c["ci90"][0],
            "dR_hi": c["ci90"][1],
            "retention": r["retention"],
            "rejection_rate": r["defer_rate"],
            "eligible": r["eligible"],
        }
        for g, _ in GUARDS:
            row[f"guard_{g}"] = r["guard_ratio"][g]
            row[f"guard_{g}_lo"], row[f"guard_{g}_hi"] = r["guard_ratio_ci90"][g]
        out.append(row)
    s2, assoc2 = rows2()
    return s2 + out, {
        "n_seeds": t["n_seeds"],
        "assoc": assoc,
        "aliased": aliased,
        "p": x4["p"],
        "assoc2": assoc2,
    }


def latex(rs: list[dict], meta: dict) -> str:
    by = {(r["study"], r["arm"]): r for r in rs}
    al = ", ".join(f"{label(a) if A.has_name(a) else a} = {label(b)}" for a, b in meta["aliased"])
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc analysis X9: descriptive, no verdict; Study 3 unchanged.",
            "Study 2 block: X9 study2.rows (= FD option_a.eval.arms, checked); 160 other seeds, no gate.",
            f"{meta['n_seeds']} Study 3 evaluation seeds; R and dR (arm minus never-sleep) with 90 % paired seed-bootstrap"
            " CIs (R CI and guard-ratio CIs in the csv); guard ratios vs accept-all; rejection rate = rejected / requests reaching the referee.",
            f"Aliased arms (identical records, not shown): {al}.",
            "Study 2: aliased arms and placebo-calibrated thresholds not shown (as in the Study 3 block).",
            "Slice Guarantee only: FD option_a / certsafe .eval.arms.sub:SliceGuarantee (= conf_eval / cs_eval, checked);"
            " R = 1 by definition; no stored rejection rate or dR (--).",
            "Random-rejection rows: post hoc check X4 (x4_tables.json), run on another host; dR = random-rejection"
            " referee minus never-sleep.",
        ],
        module=NAME,
    )
    g = len(GUARDS)
    ncol = 6 + g
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}l c c c c" + " c" * g + r" c@{}}",
        r"\toprule",
        r" & & & $\Delta R$ vs never-sleep & & \multicolumn{" + str(g) + r"}{c}{Guard ratio} & \\",
        rf"\cmidrule(lr){{6-{5 + g}}}",
        r"Arm & Rej. & $R$ & [90\,\% CI] & Ret. & "
        + " & ".join(h for _, h in GUARDS)
        + r" & Elig. \\",
    ]
    blocks = [
        ("study2", [GROUPS2[0], GROUPS2[1], (GROUPS2[2][0], meta["assoc2"])]),
        ("study3", [GROUPS[0], GROUPS[1], (GROUPS[2][0], meta["assoc"]), GROUPS[3]]),
    ]
    for study, groups in blocks:
        out += [r"\midrule", rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{{STUDY_ROWS[study]}}} \\"]
        for gi, (group, arms) in enumerate(groups):
            out += ([] if gi == 0 else [r"\midrule"]) + [A.group_row(group, ncol)]
            for arm in arms:
                r = by[(study, arm)]
                guards = " & ".join(A.prob(r[f"guard_{k}"], 2) for k, _ in GUARDS)
                dr = (
                    A.DASH
                    if arm == "never_sleep" or A.nan(r["dR"])
                    else rf"{num(r['dR'], True)} [{num(r['dR_lo'])}, {num(r['dR_hi'])}]"
                )
                out.append(
                    rf"\quad {label(arm, meta['p'])} & {A.prob(r['rejection_rate'], 3)} & {num(r['R'])} & "
                    rf"{dr} & {ret(r['retention'])} & "
                    rf"{guards} & {'yes' if r['eligible'] else 'no'} \\"
                )
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

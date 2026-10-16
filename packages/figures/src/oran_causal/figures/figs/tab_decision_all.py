"""ESM complete-results table: every evaluated decision arm of both decision datasets (booktabs LaTeX + csv). The
paper's Appendix C table tab:decision_full (tab_decision_e2e) prints a selection; this table prints every arm key of
FD ``option_a.eval.arms`` (without the gate) and ``certsafe.eval.arms`` (with the gate, other seeds), plus the three
random-rejection rows of post hoc check X4. Not a figure (no SPEC): run ``uv run python -m oran_causal.figures.figs.tab_decision_all``.

Sources (read-only), per row:
  * X9 rows (extras/revision/x9_decision_e2e.json, ``study2.rows`` / ``study3.rows``): R, retention, guard ratios,
    eligibility, rejection rate and Delta R vs never-sleep [90 % CI]; R, its CI, retention, guard ratios and
    eligibility are checked against FD before writing (as tab_decision_e2e).
  * Arms X9 does not store (freeze, the xApp subsets, the incumbent, the static envelope): FD, checked against the
    analyzer outputs conf_eval.json / cs_eval.json. No rejection rate or paired Delta R is stored for them: dashes.
  * Random rejection: X4 (extras/x4/x4_tables.json), the rows tab_decision_e2e prints.
Aliased arms (X9 ``alias_of``: identical seed records to their target) are printed as their own rows with a note mark.

Build-time assertions: every arm key of both FD dicts is a row exactly once; every printed number is within half a
unit of its last printed digit of the source value; every row of tab_decision_e2e (tab:decision_full) appears here with
identical cells.
"""

import re

from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _taba as A
from oran_causal.figures.figs import tab_decision_e2e as E

NAME = "tab_decision_all"
GUARDS = E.GUARDS
SOURCES = [source_note(*E.X9), source_note(*E.X4), ST.SRC_FD, ST.SRC_CONF_EVAL, ST.SRC_CS_EVAL]
FD_KEY = {"study2": "option_a", "study3": "certsafe"}
EVAL = {"study2": ST.CONF_EVAL_PARTS, "study3": ST.CS_EVAL_PARTS}
X9_BLOCK = {"study2": "study2", "study3": "study3"}
BLOCK = {"study2": r"\textbf{Without the gate}", "study3": r"\textbf{With the gate} (other seeds)"}
MAPS = [
    "corr",
    "granger",
    "shap_gbdt",
    "int",
    "qacm",
    "two_tower",
]  # the six baselines, two thresholds each

NAMES = {  # plain row names (Appendix B arm glossary); the group row carries the threshold
    "noarb": "Accept-all",
    "never_sleep": "Never-sleep",
    "freeze": "Freeze",
    "sub:SliceGuarantee": "Slice Guarantee only",
    "sub:ES": "ES only",
    "sub:PowerES": "Power ES only",
    "sub:ES+PowerES": "ES + Power ES only",
    "incumbent": "Incumbent",
    "B2": "Static envelope",
    "PMRT": "PMRT map",
    "GT": "Oracle map",
    "rand": "Random map",
    "blanket2": "No-map control",
    "corr": "Correlation",
    "granger": "Granger",
    "shap_gbdt": "SHAP-GBDT",
    "int": "INT",
    "qacm": "QACM",
    "two_tower": "Two-tower",
    "granger_by": "Granger + BY",
}
STATIC = [
    "noarb",
    "never_sleep",
    "freeze",
    "sub:SliceGuarantee",
    "sub:ES",
    "sub:PowerES",
    "sub:ES+PowerES",
    "incumbent",
    "B2",
]


def groups(study: str) -> list[tuple[str, list[str]]]:
    """(group heading, arm keys) of one block, in print order."""
    g = "MG" if study == "study2" else "CS"
    maps = (
        [f"MG:{m}" for m in ("PMRT", "GT", "rand")] + ["blanket2"]
        if study == "study2"
        else ["CS:PMRT", "MG:PMRT", "CS:GT", "CS:rand", "CS:blanket2"]
    )
    out = [
        (
            "Static rules",
            STATIC if study == "study2" else STATIC[:-1],
        ),  # no static envelope with the gate
        ("Map-driven referees", maps),
        (
            "Referees on associational maps (tuned thresholds)",
            [f"{g}:{m}@dev" for m in MAPS] + [f"{g}:granger_by"],
        ),
        (
            "Referees on associational maps (placebo-calibrated thresholds)",
            [f"{g}:{m}@plc" for m in MAPS],
        ),
    ]
    if study == "study3":
        out.append(("Random rejection", list(E.RANDOM)))
    return out


def label(arm: str, study: str, p: dict) -> str:
    if arm in p:
        return rf"$p = {A.short_prob(p[arm])}$"
    if arm == "MG:PMRT" and study == "study3":
        return "Ungated PMRT map"
    base = arm.split(":", 1)[1] if arm[:3] in ("MG:", "CS:") else arm
    return NAMES[base.split("@")[0]]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def _fd_row(study: str, arm: str) -> dict:
    """An arm X9 does not store: FD, checked against the analyzer output; no Rej. and no Delta R."""
    f = ST.fd()[FD_KEY[study]]["eval"]["arms"][arm]
    a = load_json(*EVAL[study])["arms"][arm]
    pairs = (
        [(f["R"], a["R"]), (f["retention"], a["retention"])]
        + list(zip(f["R_ci90"], a["R_ci90"], strict=False))
        + [(f["guard_ratio"][g], a["guard_ratio"][g]) for g, _ in GUARDS]
    )
    if not all(_close(x, y) for x, y in pairs) or bool(f["eligible"]) != bool(a["eligible"]):
        raise AssertionError(f"{study} {arm}: FD differs from {EVAL[study][-1]}")
    row = {
        "study": study,
        "arm": arm,
        "source": "FD",
        "alias_of": None,
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


def _x9_row(study: str, r: dict) -> dict:
    """An X9 row (as tab_decision_e2e), checked against FD."""
    f = ST.fd()[FD_KEY[study]]["eval"]["arms"][r["arm"]]
    pairs = (
        [(r["R"], f["R"]), (r["retention"], f["retention"])]
        + list(zip(r["R_ci90"], f["R_ci90"], strict=False))
        + [(r["guard_ratio"][g], f["guard_ratio"][g]) for g, _ in GUARDS]
    )
    if not all(_close(x, y) for x, y in pairs) or bool(r["eligible"]) != bool(f["eligible"]):
        raise AssertionError(
            f"{study} {r['arm']}: X9 row differs from FD {FD_KEY[study]}.eval.arms"
        )
    row = {
        "study": study,
        "arm": r["arm"],
        "source": "X9",
        "alias_of": r["alias_of"],
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


def rows() -> tuple[list[dict], dict]:
    x9 = load_json(*E.X9)
    e2e, meta = E.rows()  # also runs tab_decision_e2e's own source checks
    rand = {r["arm"]: r for r in e2e if r["arm"] in E.RANDOM}
    out = []
    for study in ("study2", "study3"):
        fd = ST.fd()[FD_KEY[study]]["eval"]
        if fd["n"] != x9[X9_BLOCK[study]]["n_seeds"]:
            raise AssertionError(
                f"{study}: FD seeds {fd['n']} != X9 seeds {x9[X9_BLOCK[study]]['n_seeds']}"
            )
        by = {r["arm"]: r for r in x9[X9_BLOCK[study]]["rows"]}
        printed = [a for _, arms in groups(study) for a in arms]
        keys = set(fd["arms"]) | set(by)
        if sorted(a for a in printed if a not in E.RANDOM) != sorted(keys):
            raise AssertionError(
                f"{study}: printed arms differ from the stored arms: "
                f"missing {sorted(keys - set(printed))}, extra {sorted(set(printed) - keys - set(E.RANDOM))}"
            )
        for _, arms in groups(study):
            for arm in arms:
                if arm in E.RANDOM:
                    r = {k: v for k, v in rand[arm].items() if k != "label"}
                    out.append({**r, "source": "X4", "alias_of": None})
                elif arm in by:
                    out.append(_x9_row(study, by[arm]))
                else:
                    out.append(_fd_row(study, arm))
    for r in out:
        r["label"] = label(r["arm"], r["study"], meta["p"])
    return out, {"p": meta["p"], "e2e": e2e, "e2e_meta": meta}


# --- formatting -------------------------------------------------------------------------------------------------------
MARK = {"noarb": "a"}  # alias target -> note mark; any other target (a tuned-threshold map) -> b


def mark(r: dict) -> str:
    if r["alias_of"] is None:
        return ""
    return rf"$^{{\mathrm{{{MARK.get(r['alias_of'], 'b')}}}}}$"


def cells(r: dict) -> list[str]:
    """The printed cells after the label, formatted exactly as tab_decision_e2e."""
    dr = (
        A.DASH
        if r["arm"] == "never_sleep" or A.nan(r["dR"])
        else rf"{E.num(r['dR'], True)} [{E.num(r['dR_lo'])}, {E.num(r['dR_hi'])}]"
    )
    return (
        [A.prob(r["rejection_rate"], 3), E.num(r["R"]), dr, E.ret(r["retention"])]
        + [A.prob(r[f"guard_{k}"], 2) for k, _ in GUARDS]
        + ["yes" if r["eligible"] else "no"]
    )


def _val(s: str) -> float:
    return float(s.replace("$-$", "-").replace("[", "").replace("]", "").replace(",", "").strip())


def source(r: dict) -> dict:
    """The stored values of a row, re-read from the artifacts: FD for R, retention, guards and
    eligibility (X4 for the random-rejection rows); X9 / X4 for the rejection rate and Delta R.
    """
    if r["source"] == "X4":
        x, c = (
            load_json(*E.X4)["rows"][r["arm"]],
            load_json(*E.X4)["contrasts"][r["arm"]]["never_sleep"],
        )
        return {
            "R": x["R"],
            "retention": x["retention"],
            "guard_ratio": x["guard_ratio"],
            "eligible": x["eligible"],
            "rejection_rate": x["defer_rate"],
            "dR": [c["dR"], *c["ci90"]],
        }
    f = ST.fd()[FD_KEY[r["study"]]]["eval"]["arms"][r["arm"]]
    out = {
        "R": f["R"],
        "retention": f["retention"],
        "guard_ratio": f["guard_ratio"],
        "eligible": f["eligible"],
        "rejection_rate": None,
        "dR": None,
    }
    if r["source"] == "X9":
        x = {a["arm"]: a for a in load_json(*E.X9)[X9_BLOCK[r["study"]]]["rows"]}[r["arm"]]
        out["rejection_rate"] = x["rejection_rate"]
        out["dR"] = [x["dR_vs_never_sleep"], *x["dR_vs_never_sleep_ci90"]]
    return out


def _check_printed(r: dict, c: list[str]) -> None:
    """[assertion] every printed number is the stored source value rounded to its printed digits."""
    s0 = source(r)
    r = {
        **r,
        "R": s0["R"],
        "retention": s0["retention"],
        "eligible": s0["eligible"],
        "rejection_rate": s0["rejection_rate"],
        **{f"guard_{k}": s0["guard_ratio"][k] for k, _ in GUARDS},
    }
    r["dR"], r["dR_lo"], r["dR_hi"] = s0["dR"] or (None, None, None)
    srcs = [r["rejection_rate"], r["R"], r["retention"]] + [r[f"guard_{k}"] for k, _ in GUARDS]
    for s, v in zip(c[:2] + c[3:-1], srcs, strict=False):
        if A.nan(v) or s == A.DASH:
            if not (A.nan(v) and s == A.DASH):
                raise AssertionError(f"{r['study']} {r['arm']}: printed '{s}', source {v}")
            continue
        nd = len(s.split(".")[1]) if "." in s else 0
        if abs(_val(s) - v) > 0.5 * 10**-nd + 1e-12:
            raise AssertionError(f"{r['study']} {r['arm']}: printed {s} != source {v}")
    if (c[2] == A.DASH) != (r["arm"] == "never_sleep" or A.nan(r["dR"])):
        raise AssertionError(f"{r['study']} {r['arm']}: Delta R dash")
    if c[2] != A.DASH:
        parts = re.match(r"(\S+) \[(\S+), (\S+)\]$", c[2]).groups()
        for s, v in zip(parts, (r["dR"], r["dR_lo"], r["dR_hi"]), strict=False):
            nd = len(s.split(".")[1])
            if abs(_val(s.replace("+", "")) - v) > 0.5 * 10**-nd + 1e-12:
                raise AssertionError(f"{r['study']} {r['arm']}: printed dR {s} != source {v}")
    if c[-1] != ("yes" if r["eligible"] else "no"):
        raise AssertionError(f"{r['study']} {r['arm']}: eligibility")


def e2e_cells(meta: dict) -> dict[tuple[str, str], list[str]]:
    """(study, arm) -> the cells tab_decision_e2e prints (tab:decision_full), parsed from its LaTeX output."""
    m = meta["e2e_meta"]
    order = [("study2", a) for a in E.GROUPS2[0][1] + E.GROUPS2[1][1] + m["assoc2"]]
    order += [("study3", a) for a in E.GROUPS[0][1] + E.GROUPS[1][1] + m["assoc"] + E.GROUPS[3][1]]
    lines = [x for x in E.latex(meta["e2e"], m).splitlines() if x.startswith(r"\quad ")]
    if len(lines) != len(order):
        raise AssertionError(f"tab_decision_e2e prints {len(lines)} rows, expected {len(order)}")
    return {
        k: [c.strip() for c in x.removesuffix(r"\\").split(" & ")[1:]]
        for k, x in zip(order, lines, strict=False)
    }


def check(rs: list[dict], meta: dict) -> None:
    by = {(r["study"], r["arm"]): r for r in rs}
    if len(by) != len(rs):
        raise AssertionError("an arm is printed twice")
    for r in rs:
        _check_printed(r, cells(r))
    for k, c in e2e_cells(meta).items():
        if cells(by[k]) != c:
            raise AssertionError(f"{k}: tab:decision_full prints {c}, this table {cells(by[k])}")


def tabular(rs: list[dict], study: str) -> list[str]:
    g = len(GUARDS)
    ncol = 6 + g
    out = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}l c c c c" + " c" * g + r" c@{}}",
        r"\toprule",
        r" & & & $\Delta R$ vs never-sleep & & \multicolumn{" + str(g) + r"}{c}{Guard ratio} & \\",
        rf"\cmidrule(lr){{6-{5 + g}}}",
        r"Arm & Rej. & $R$ & [90\,\% CI] & Ret. & "
        + " & ".join(h for _, h in GUARDS)
        + r" & Elig. \\",
        r"\midrule",
        rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{{BLOCK[study]}}} \\",
    ]
    by = {(r["study"], r["arm"]): r for r in rs}
    for gi, (group, arms) in enumerate(groups(study)):
        out += ([] if gi == 0 else [r"\midrule"]) + [A.group_row(group, ncol)]
        for arm in arms:
            r = by[(study, arm)]
            out.append(rf"\quad {r['label']}{mark(r)} & " + " & ".join(cells(r)) + r" \\")
    return [*out, r"\bottomrule", r"\end{tabular}"]


def latex(rs: list[dict], meta: dict) -> str:
    check(rs, meta)
    n = {s: sum(r["study"] == s for r in rs) for s in ("study2", "study3")}
    out = A.header(
        NAME,
        SOURCES,
        [
            "ESM complete-results table: every arm key of FD option_a.eval.arms (without the gate) and certsafe.eval.arms"
            " (with the gate, other seeds), plus the three X4 random-rejection rows.",
            f"Two tabulars (one table each; the whole does not fit one page): without the gate {n['study2']} rows,"
            f" with the gate {n['study3']} rows.",
            "Rej. and Delta R vs never-sleep [90 % paired seed-bootstrap CI] as stored (X9 / X4); -- where not stored"
            " (freeze, xApp subsets, incumbent, static envelope: FD, = conf_eval / cs_eval, checked).",
            "Aliased arms (identical seed records): a = same decisions as accept-all; b = same decisions as the"
            " tuned-threshold map.",
            "Rows of tab:decision_full (tab_decision_e2e) are checked to print identical cells.",
        ],
        module=NAME,
    )
    out += ["% ---- Table 1 of 2: without the gate ----", *tabular(rs, "study2")]
    out += ["", "% ---- Table 2 of 2: with the gate (other seeds) ----", *tabular(rs, "study3"), ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

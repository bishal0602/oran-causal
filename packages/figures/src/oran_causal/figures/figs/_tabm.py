"""Private helpers of the Studies 1-3 and experimental-design tables (tab_study1_criteria,
tab_study2_maps, tab_study3_criteria, tab_design_studies, tab_design_criteria,
tab_system_hypotheses): the analyzer verdict files that figures_data.json (FD, ``_studies.fd()``)
does not copy, number formatting and the shared .tex / .csv writer.

Every value is read from the record sets at build time; nothing here holds a result.
"""

import csv
import re

from oran_causal.figures import methods as M
from oran_causal.figures.data import load_json, protocol_note, source_note
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.paths import protocol, tables_dir

V4_VERDICT_PARTS = (*ST.DECISION, "v4_verdict.json")  # Study 1 analyzer verdict (rules, pass flags)
CONF_VERDICT_PARTS = (*ST.DECISION, "conf_verdict.json")  # Study 2 referee verdict
CS_VERDICT_PARTS = (*ST.DECISION, "cs_verdict.json")  # Study 3 referee verdict
SRC_V4_VERDICT = source_note(*V4_VERDICT_PARTS)
SRC_CONF_VERDICT = source_note(*CONF_VERDICT_PARTS)
SRC_CS_VERDICT = source_note(*CS_VERDICT_PARTS)

CS_PROTOCOL_PARTS = ("study3", "E6P_CERTSAFE_PROTOCOL.md")  # s.7 eligibility definition (text)
SRC_CS_PROTOCOL = protocol_note(*CS_PROTOCOL_PARTS)
GUARDS = {
    "svr": "SVR",
    "nonprot_embb_viol": "non-protected eMBB",
    "ll_viol": "LL",
    "rlf": "RLF",
}  # TERMINOLOGY s.7

_cache: dict = {}


def eligibility() -> tuple[str, str]:
    """(retention floor, guard-ratio cap) of 'eligible', read from the frozen Study 3 protocol s.7
    (the analyzer JSONs store only the eligible flag).
    """
    if "elig" not in _cache:
        txt = protocol(*CS_PROTOCOL_PARTS).read_text(encoding="utf-8")
        m = re.search(r"eligible = retention >= (\.\d+) and every guard ratio\s+<= (\d\.\d+)", txt)
        if m is None:
            raise AssertionError("eligibility rule not found in the Study 3 protocol s.7")
        _cache["elig"] = ("0" + m.group(1), m.group(2))
    return _cache["elig"]


def load(parts: tuple) -> dict:
    if parts not in _cache:
        _cache[parts] = load_json(*parts)
    return _cache[parts]


# --- formatting -------------------------------------------------------------------------------------------------------
def f(x: float, nd: int = 2) -> str:
    """Plain decimal with a leading zero (the minus is typeset in math mode)."""
    s = f"{x:.{nd}f}"
    return f"${s}$" if s.startswith("-") else s


def fs(x: float, nd: int = 3) -> str:
    """Signed decimal in math mode (``$+0.161$``)."""
    return f"${x:+.{nd}f}$"


def t(x) -> str:
    """A rule threshold as stored, with at least two decimals (0.1 -> 0.10, 0.075 stays); integers
    unchanged.
    """
    if isinstance(x, int):
        return str(x)
    s = repr(float(x))
    return s if len(s.split(".")[1]) >= 2 else f"{x:.2f}"


# Paper labels of protocol criteria whose protocol label collides with other notation (outline v2
# s.5; advisor 2026-10-11): E (eligibility) vs environments E1-E5, S (sign) vs S1, S1 (RLF bound) vs
# S. Protocol JSON keys unchanged.
CRIT_LABEL = {"E": "Elig", "S": "SA", "S1": "RLF-UB"}


def crit(label: str) -> str:
    """Paper label of a protocol criterion."""
    return CRIT_LABEL.get(label, label)


def precedence(p: str) -> str:
    """An analyzer precedence string in table wording: '>' typeset, 'A / B' as 'A; B', '(P1 P2 P3 S)' as
    '(P1, P2, P3, SA)'; criterion labels renamed by ``crit`` inside parentheses.
    """
    p = re.sub(
        r"\(([A-Z0-9]+(?: [A-Z0-9]+)+)\)", lambda m: "(" + ", ".join(m.group(1).split()) + ")", p
    )
    p = re.sub(
        r"\(([^()]*)\)",
        lambda m: "(" + re.sub(r"\b(S1|E|S)\b", lambda t: crit(t.group(1)), m.group(1)) + ")",
        p,
    )
    return p.replace(" > ", r" $>$ ").replace(" / ", "; ")


def ci(lo: float, hi: float, nd: int = 2) -> str:
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


def n(k: int) -> str:
    """Integer with a thin-space thousands separator (``89\\,067``)."""
    return f"{int(k):,}".replace(",", r"\,")


def le(x) -> str:
    return rf"$\le$ {x}"


def ge(x) -> str:
    return rf"$\ge$ {x}"


def result(ok: bool) -> str:
    return "PASS" if ok else "FAIL"  # TM.write sets them in small caps (\verdict)


def tex(s: str) -> str:
    """Escape the LaTeX specials that occur in names (underscores in parameter codes, ampersands)."""
    return s.replace("_", r"\_").replace("&", r"\&")


def map_name(code: str) -> str:
    """Paper name of a referee map arm (``MG:corr@dev`` -> "Correlation (DEV-tuned)")."""
    arm = M.resolve(code)
    thr = {"dev": "DEV-tuned", "plc": "placebo-calibrated"}.get(arm.threshold)
    return arm.method.name + (f" ({thr})" if thr else "")


def inline(name: str) -> str:
    """A paper name inside a sentence: 'Correlation' -> 'correlation'; acronyms (PMRT, SHAP-GBDT)
    unchanged.
    """
    return name[0].lower() + name[1:] if len(name) > 1 and name[1].islower() else name


def baseline_name(code: str) -> str:
    """Studies 1-3 discovery baseline (``granger`` = the DEV-tuned Granger of Studies 1-3)."""
    return M.resolve(code, study13=True).method.name


# --- writer -----------------------------------------------------------------------------------------------------------
def header(name: str, sources: list[str], notes: list[str]) -> list[str]:
    return (
        [
            f"% {name}: generated by oran-causal-figures (packages/figures/src/oran_causal/figures/figs/{name}.py); do not edit.",
            "% Sources: " + ", ".join(sources),
        ]
        + [f"% {s}" for s in notes]
        + ["% Needs \\usepackage{booktabs} and the \\verdict macro (sn-article.tex)."]
    )


VERDICT = re.compile(
    r"(?<![\w{-])(NOT ELIGIBLE|NOT SUPPORTED|NOT EVALUABLE|NO-CHAIN|UNDERPOWERED|INCONCLUSIVE|SUPPORTED|ELIGIBLE"
    r"|PARTIAL|INVALID|VALID|PASS|FAIL|KILL)(?![\w-])"
)


def verdicts(line: str) -> str:
    """Verdict labels outside LaTeX comments as ``\\verdict{partial}`` (small caps; macro in sn-article.tex)."""
    m = re.search(r"(?<!\\)%", line)
    pre, post = (line[: m.start()], line[m.start() :]) if m else (line, "")
    return VERDICT.sub(lambda v: r"\verdict{" + v.group(1).lower() + "}", pre) + post


def write(name: str, lines: list[str], rows: list[dict]) -> list[str]:
    """Write ``<tables_dir>/<name>.tex`` (the given lines, verdict labels in small caps) and
    ``<name>.csv``.
    """
    lines = [verdicts(x) for x in lines]
    out = tables_dir()
    out.mkdir(parents=True, exist_ok=True)
    tex_path, csv_path = out / f"{name}.tex", out / f"{name}.csv"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return [str(tex_path), str(csv_path)]

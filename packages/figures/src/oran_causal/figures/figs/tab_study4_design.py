"""Study 4 design: one row per controlled diagnostic environment E1-E5 (mechanism, true and null candidate edges,
regimes), as a booktabs LaTeX tabular plus a csv for Section 7.2. Not a figure (no SPEC): run
``uv run python -m oran_causal.figures.figs.tab_study4_design``.

Source: the frozen protocol, packages/experiments/protocols/study4/PROTOCOL.md s.2 (read as text at build time):
  true / null   the "Candidates true / null" sentence (nulls include the placebo edges, placebo action -> each KPI;
                E4 has no real-action null, so its one null is the placebo edge)
  regimes       the world table's regimes column (ranges expanded), restricted to R1-R3 (Section
                7.2 shows R1-R3; R4, the negative control with nothing logged, is summarised in Section 7.2 and
                given on the E4 page, Online Resource 1, Section S4, and named in a table note, checked against the protocol's regime table); E4's confounding strengths from the lambda clause
  mechanism     the world table's mechanism column in paper wording (MECHANISM below); each row is checked against a
                key phrase of the protocol row so a protocol change fails the build
The regimes R1-R4 form a second block: per regime, how actions are logged and its role, from the protocol's regime
table (REGIME below, each row checked against a key phrase of the protocol row); E4's regimes list R4, which the block
defines.
The counts agree with the EVAL scores (merged.jsonl.gz measure records: n_true, and n_null plus
the placebo edges, identical in every regime of a world).
"""

import re

from oran_causal.figures.data import protocol_note
from oran_causal.figures.figs import _study4 as T
from oran_causal.figures.figs import _tabm as TM
from oran_causal.figures.paths import protocol

NAME = "tab_study4_design"
PROTOCOL_PARTS = ("study4", "PROTOCOL.md")
SOURCES = [protocol_note(*PROTOCOL_PARTS)]
SHOWN = ("R1", "R2", "R3", "R4")  # R4 shown too, since the regime block defines it
# regime -> (protocol design phrase, protocol role phrase, paper wording of the design, paper
# wording of the role)
REGIME = {
    "R1": (
        "every action redrawn i.i.d. per row",
        "reference",
        "Every action redrawn independently per observation",
        "Reference",
    ),
    # the +/-25 % envelope as in oran_causal.sim.controlled.datasets.generate (R2: 20 setpoints
    # drawn once per corpus uniformly within +/-25 % of the range around the midpoint, each held
    # for n/20 consecutive rows; DITHER_ENVELOPE = 0.25)
    "R2": (
        "20 setpoints per corpus (+/-25 % envelope), held n/20 rows, + i.i.d. U(+/-.10 range) dither",
        "C1 / C2 test bed",
        r"Setpoint plus dither: 20 setpoints drawn within $\pm 25$~\% of the range around its midpoint, each held "
        r"for $n/20$ observations, plus independent dither of $\pm 10$~\% of the range",
        "Main test bed",
    ),
    "R3": (
        "E4 behaviour policy on an OBSERVED confounder, per-row probabilities logged",
        "C3",
        "E4 only: a behaviour policy on an observed confounder, with its probabilities logged",
        "Confounded policy",
    ),
    "R4": (
        "E4 latent confounder, nothing logged",
        "negative control",
        "E4 only: latent confounder, nothing logged",
        "Negative control",
    ),
}
# environment -> (protocol key phrase, paper wording of the mechanism)
MECHANISM = {
    "E1": ("clean identifiable linear graph", "Clean, identifiable linear graph"),
    "E2": ("gated edge, active only in a rare regime", "Gated edge, active only in a rare regime"),
    "E3": ("delayed harm", "Delayed harm through lagged KPI cascades (depth 3)"),
    "E4": (
        "latent confounding",
        "Latent confounding; the associational sign is wrong at large confounding strength",
    ),
    "E5": ("E2-E4 mechanisms composed", "The E2--E4 mechanisms composed"),
}


def _section() -> str:
    txt = protocol(*PROTOCOL_PARTS).read_text(encoding="utf-8")
    m = re.search(r"^## 2\. .*?(?=^## 3\. )", txt, re.S | re.M)
    if m is None:
        raise AssertionError("PROTOCOL_A s.2 not found")
    return m.group(0)


def _regimes(cell: str) -> list[str]:
    out = []
    for part in cell.split(","):
        a = re.fullmatch(r"\s*R(\d)(?:-R(\d))?\s*", part)
        if a is None:
            raise AssertionError(f"unreadable regimes {cell!r}")
        lo, hi = int(a.group(1)), int(a.group(2) or a.group(1))
        out += [f"R{i}" for i in range(lo, hi + 1)]
    return out


def rows() -> list[dict]:
    s = _section()
    flat = " ".join(s.split())
    cand = re.search(r"Candidates true / null \(.*?\): (E1 .*?);", flat)
    if cand is None:
        raise AssertionError("candidate sentence not found in PROTOCOL_A s.2")
    counts = {w: (int(t), int(n)) for w, t, n in re.findall(r"(E\d) (\d+)/(\d+)", cand.group(1))}
    lam = re.search(r"lambda in \{([^}]*)\}", flat)
    if lam is None:
        raise AssertionError("E4 lambda grid not found in PROTOCOL_A s.2")
    lams = [x.strip() for x in lam.group(1).split(",")]
    table = {
        m.group(1): (m.group(2).strip(), m.group(3).strip())
        for m in re.finditer(r"^\| (E\d) \| (.*?) \| (.*?) \|$", s, re.M)
    }
    if sorted(counts) != T.WORLDS or sorted(table) != T.WORLDS:
        raise AssertionError(f"environments differ: {sorted(counts)}, {sorted(table)}")
    for g, (design, role, _, _) in REGIME.items():
        m = re.search(rf"^\| {g} \| (.*?) \| (.*?) \|$", s, re.M)
        if m is None or design not in m.group(1) or not m.group(2).strip().startswith(role):
            raise AssertionError(f"PROTOCOL_A s.2 regime row {g} changed; check REGIME")
    out = []
    for w in T.WORLDS:
        key, mech = MECHANISM[w]
        if key not in table[w][0]:
            raise AssertionError(f"{w}: protocol mechanism changed: {table[w][0]!r}")
        out.append(
            {
                "environment": w,
                "mechanism": mech,
                "true": counts[w][0],
                "null": counts[w][1],
                "regimes": [g for g in _regimes(table[w][1]) if g in SHOWN],
                "lambda": lams if w == "E4" else [],
            }
        )
    return out


def _lam(xs: list[str]) -> str:
    return ", ".join(x if not x.startswith(".") else "0" + x for x in xs)


def latex(rs: list[dict]) -> list[str]:
    out = TM.header(
        NAME,
        SOURCES,
        [
            "Null candidates include the placebo edges (placebo action -> each KPI); E4 has no "
            "real-action null.",
            "E4: R1 / R2 at confounding strength 1 only; R3 / R4 at every strength.",
            "Regime block: the protocol's s.2 regime table (design and role columns).",
        ],
    )
    out += [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l p{5.4cm} c c p{3.0cm}@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Candidate edges} & \tabularnewline",
        r"\cmidrule(lr){3-4}",
        r"Environment & Mechanism & True & Null & Regimes \tabularnewline",
        r"\midrule",
    ]
    for r in rs:
        reg = ", ".join(r["regimes"])
        if r["lambda"]:
            reg += rf"; $\lambda \in \{{{_lam(r['lambda'])}\}}$"
        out.append(
            rf"{r['environment']} & \raggedright {r['mechanism']} & {r['true']} & {r['null']} & "
            rf"\raggedright {reg} \tabularnewline"
        )
    out += [
        r"\midrule",
        r"Regime & How actions are logged & \multicolumn{3}{l}{Role} \tabularnewline",
        r"\midrule",
    ]
    for g, (_, _, design, role) in REGIME.items():
        out.append(
            rf"{g} & \raggedright {design} & \multicolumn{{3}}{{l}}{{{role}}} \tabularnewline"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return out


def write() -> list[str]:
    rs = rows()
    csv_rows = [
        {**r, "regimes": " ".join(r["regimes"]), "lambda": " ".join(r["lambda"])} for r in rs
    ]
    return TM.write(NAME, latex(rs), csv_rows)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

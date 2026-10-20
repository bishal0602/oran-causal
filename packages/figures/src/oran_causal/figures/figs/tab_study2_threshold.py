"""Post hoc analysis X12, threshold choice of the Study 2 baselines (Section 6.2; booktabs LaTeX +
csv): each tuned design-blind baseline with its DEV-tuned threshold next to its placebo-calibrated
one, with the declared edges, edges whose ground truth is TRUE with the right / wrong sign, GT-NULL
edges, placebo edges and recall (whatever the sign); the untuned Granger + BY and PMRT (no tuned threshold) close the
table, PMRT last. The placebo-calibrated threshold is set on the same placebo logs that give its
placebo-edge count, so that count is in-sample. Not a figure (no SPEC): run ``uv run python -m
oran_causal.figures.figs.tab_study2_threshold``.

Source: posthoc_x9_x16/revision/x12_threshold.json (X12, posthoc protocol amendment 2026-10-11; the script
reproduced conf_disc_conf.json edge_quality and FD option_a.map_quality for all 14 maps before
writing), keys ``repro.ok``, ``gt.n_true`` and ``rows.<i>.{arm, method, variant, edges, true_right,
true_wrong, null, indet, placebo_edges, recall_true_right}``. Cross-checked against FD
``option_a.map_quality`` (the source of tab_study2_maps). Recall = (right + wrong) / GT-TRUE cells,
whatever the sign, as in Section 5; the stored right-sign recall stays in the csv as
``recall_true_right``. Post hoc, descriptive, no verdict; Study 2 is unchanged.
"""

import math

from oran_causal.figures import methods as M
from oran_causal.figures.data import load_json, source_note
from oran_causal.figures.figs import _studies as ST
from oran_causal.figures.figs import _taba as A

NAME = "tab_study2_threshold"
X12 = ("posthoc_x9_x16", "revision", "x12_threshold.json")
SOURCES = [source_note(*X12), ST.SRC_FD]
COLS = [
    ("edges", "Edges"),
    ("true_right", "Right"),
    ("true_wrong", "Wrong"),
    ("null", "NULL"),
    ("placebo_edges", "Plac."),
    ("recall", "Recall"),
]
CSV_ONLY = ("recall_true_right",)
FD_KEYS = ("edges", "true_right", "true_wrong", "null", "indet", "placebo_edges")
LAST = ["MG:granger_by", "MG:PMRT"]  # no tuned threshold; PMRT is the last row
MAX_FP = 1  # the @plc rule's placebo budget (protocols/study2/PROTOCOL.md s.4.2)


def check(t: dict):
    if not t["repro"]["ok"]:
        raise AssertionError("X12 reproduction flag not set")
    mq = ST.fd()["option_a"]["map_quality"]
    for r in t["rows"]:
        if any(r[k] != mq[r["arm"]][k] for k in FD_KEYS):
            raise AssertionError(f"{r['arm']}: counts differ from FD option_a.map_quality")
        if not math.isclose(r["recall_true_right"], r["true_right"] / t["gt"]["n_true"]):
            raise AssertionError(f"{r['arm']}: stored recall != right / GT-TRUE")
        if r["variant"] == "plc" and r["placebo_edges"] > MAX_FP:
            raise AssertionError(f"{r['arm']}: placebo-calibrated map exceeds the in-sample budget")


def rows() -> tuple[list[dict], dict]:
    t = load_json(*X12)
    check(t)
    n_true = t["gt"]["n_true"]
    by = {r["arm"]: r | {"recall": (r["true_right"] + r["true_wrong"]) / n_true} for r in t["rows"]}
    tuned = M.order(
        sorted({r["arm"].split("@")[0] for r in t["rows"] if r["variant"] in ("dev", "plc")})
    )
    out = []
    for base in tuned:
        d, p = by[f"{base}@dev"], by[f"{base}@plc"]
        row = {"arm": base, "label": M.resolve(f"{base}@dev").method.name, "tuned": True}
        for v, r in (("dev", d), ("plc", p)):
            row.update({f"{k}_{v}": r[k] for k in (*(c for c, _ in COLS), *CSV_ONLY)})
            row[f"indet_{v}"] = r["indet"]
        out.append(row)
    for arm in LAST:
        r = by[arm]
        row = {"arm": arm, "label": M.resolve(arm).method.name, "tuned": False}
        row.update({f"{k}_dev": r[k] for k in (*(c for c, _ in COLS), *CSV_ONLY)})
        row["indet_dev"] = r["indet"]
        out.append(row)
    return out, {"n_true": t["gt"]["n_true"], "n_cells": t["gt"]["n_cells"]}


def cell(k: str, x) -> str:
    return A.prob(x, 2) if k == "recall" else str(x)


def latex(rs: list[dict], meta: dict) -> str:
    out = A.header(
        NAME,
        SOURCES,
        [
            "Post hoc analysis X12: descriptive, no verdict; Study 2 unchanged.",
            f"Right / wrong = ground-truth TRUE edge with the right / wrong sign; recall = (right + wrong) / {meta['n_true']} "
            f"GT-TRUE cells of {meta['n_cells']}, whatever the sign; Plac. = edges declared on the 200 placebo episodes.",
            "DEV-tuned: far-FPR rule on 20 DEV episodes. Placebo-calibrated: set on the placebo episodes themselves, so its "
            "placebo count is in-sample (at most 1 by construction).",
            "INDET edges in the csv.",
        ],
        module=NAME,
    )
    k = len(COLS)
    ncol = 1 + 2 * k
    out += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}l" + " r" * k + r" @{\hspace{8pt}}" + " r" * k + r"@{}}",
        r"\toprule",
        rf" & \multicolumn{{{k}}}{{c}}{{DEV-tuned threshold}} & \multicolumn{{{k}}}{{c}}{{Placebo-calibrated threshold}} \\",
        rf"\cmidrule(lr){{2-{1 + k}}}\cmidrule(l){{{2 + k}-{ncol}}}",
        "Method & "
        + " & ".join(h for _, h in COLS)
        + " & "
        + " & ".join(h for _, h in COLS)
        + r" \\",
        r"\midrule",
        A.group_row("Design-blind baselines, two thresholds", ncol),
    ]
    first_untuned = True
    for r in rs:
        dev = " & ".join(cell(c, r[f"{c}_dev"]) for c, _ in COLS)
        if r["tuned"]:
            plc = " & ".join(cell(c, r[f"{c}_plc"]) for c, _ in COLS)
        else:
            if first_untuned:
                out += [r"\midrule", A.group_row("No tuned threshold", ncol)]
                first_untuned = False
            plc = rf"\multicolumn{{{k}}}{{c}}{{{A.DASH}}}"
        out.append(rf"\quad {r['label']} & {dev} & {plc} \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write() -> list[str]:
    rs, meta = rows()
    return A.write(NAME, latex(rs, meta), rs)


if __name__ == "__main__":
    for p in write():
        print("wrote", p)

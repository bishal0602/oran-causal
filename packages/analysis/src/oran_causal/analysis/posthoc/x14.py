"""X14: map scores against an "effect exists" ground truth.

Post hoc and descriptive: no verdict, and the frozen material-effect rule stays the primary ground
truth. The existence rule uses the same knockout records, pooled means and episode-cluster
bootstrap draws as the frozen GT table (`x11.cell_stats`, identical draws), with the same support
rule:

* EXISTS(sign): the interval excludes 0, any magnitude; the sign is that of the mean.
* NOT_ESTABLISHED: the interval contains 0.
* UNSUPPORTED: fewer than 10 units or 3 episodes (the frozen INDET-unsupported), unscored.

Per method, over its declared edges: confirmed = EXISTS with the declared sign; wrong sign = EXISTS
with the opposite sign; not established; unsupported, counted apart. Existence precision is
confirmed / (confirmed + wrong sign + not established). Recall against the EXISTS set is the
declared EXISTS count (any sign) over EXISTS, and with the right sign confirmed over EXISTS. The
frozen material-rule scores sit beside them.

Methods are every method of every stored analysis: Study 1 pooled and its 17 slices; Study 2 the
nine PMRT combinations and every DISC map except `MG:GT`. Reproduction comes first and is a hard
stop: the GT table and the material-rule scores of every analysis must equal the stored ones.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc import x11
from oran_causal.analysis.posthoc.paths import PRIMARY, PROTOCOL_BASENAME, RECORDS, write_text
from oran_causal.sim.system.decision import gt as GT
from oran_causal.sim.system.decision.vocab import HYPOTHESES

LEVELS = (0.95, 0.90, 0.99)
STORED_KEYS = {
    "ov_precision": "precision",
    "ov_recall": "recall",
    "ov_f1": "f1",
    "sign_acc": "sign_acc",
    "ind_precision": "ind_precision",
    "ind_recall": "ind_recall",
    "ind_f1": "ind_f1",
    "n_declared": "n_declared",
    "far_declared": "far_declared",
}
DECLARED = f"{PROTOCOL_BASENAME} amendment 2026-10-11 X14"
Analysis = tuple[str, str, str, str, x11.Decl, Mapping[str, Any]]


def existence_labels(
    stats: Mapping[tuple, Mapping[str, Any]], level: float
) -> dict[tuple, dict[str, Any]]:
    """Existence label `{status, sign, mean, ci}` per hypothesis.

    The status is EXISTS, NOT_ESTABLISHED or UNSUPPORTED.
    """
    out = {}
    for h, v in stats.items():
        lo, hi = v["ci"][level]
        if v["n"] < GT.GT_MIN_UNITS or v["n_eps"] < GT.GT_MIN_EPS or not np.isfinite(v["mean"]):
            st, s = "UNSUPPORTED", 0
        elif lo > 0 or hi < 0:
            st, s = "EXISTS", int(np.sign(v["mean"]))
        else:
            st, s = "NOT_ESTABLISHED", 0
        out[h] = {"status": st, "sign": s, "mean": v["mean"], "ci": [lo, hi]}
    return out


def existence_scores(decl: x11.Decl, ex: Mapping[tuple, Mapping[str, Any]]) -> dict[str, Any]:
    """Existence scores of a declaration."""
    dec_h = [h for h in HYPOTHESES if decl[h]["declared"]]
    c = {"confirmed": 0, "wrong_sign": 0, "not_established": 0, "unsupported": 0}
    for h in dec_h:
        e = ex[h]
        if e["status"] == "EXISTS":
            c["confirmed" if decl[h]["sign"] == e["sign"] else "wrong_sign"] += 1
        else:
            c["not_established" if e["status"] == "NOT_ESTABLISHED" else "unsupported"] += 1
    n_ex = sum(e["status"] == "EXISTS" for e in ex.values())
    scored = c["confirmed"] + c["wrong_sign"] + c["not_established"]
    return {
        **{f"ex_{k}": v for k, v in c.items()},
        "ex_n_exists": n_ex,
        "ex_precision": x11.nanf(x11.safe(c["confirmed"], scored)),
        "ex_recall": x11.nanf(x11.safe(c["confirmed"] + c["wrong_sign"], n_ex)),
        "ex_recall_right_sign": x11.nanf(x11.safe(c["confirmed"], n_ex)),
    }


def material_scores(decl: x11.Decl, ref: Mapping, cells: Sequence[x11.Cell]) -> dict[str, Any]:
    """The frozen material-rule scores, renamed with the `mat_` prefix."""
    s = x11.scores(decl, ref, cells)
    signed = x11.safe(s["n_true_right"], s["n_true_right"] + s["n_true_wrong"] + s["n_null"])
    return {
        "mat_n_declared": s["n_declared"],
        "mat_precision": s["precision"],
        "mat_recall": s["recall"],
        "mat_sign_acc": s["sign_acc"],
        "mat_precision_signed": x11.nanf(signed),
        "mat_true_right": s["n_true_right"],
        "mat_true_wrong": s["n_true_wrong"],
        "mat_null": s["n_null"],
        "mat_indet": s["n_indet"],
    }


def analyses(study: str, a: Mapping[str, Any]) -> list[Analysis]:
    """`(slice, slice kind, method, kind, declaration, stored scores)` of a stored analysis."""
    out: list[Analysis] = []
    if study == "S1":
        for sl, r in a["slices_result"].items():
            for m, mm in r["metrics"].items():
                if m == PRIMARY:
                    kind = "PMRT"
                elif "+" in m and not m.startswith("v2"):
                    kind = "PMRT variant"
                else:
                    kind = "baseline"
                stored = {v: mm[k] for k, v in STORED_KEYS.items()}
                out.append((sl, r["kind"], m, kind, x11.decl_from_list(mm["declared"]), stored))
        return out
    meth, stored = x11.methods_of(study, a)
    for m, (kind, decl) in meth.items():
        out.append(("pooled", "pooled", m, kind, decl, stored[m]))
    return out


def _check_material(
    study: str, items: Sequence[Analysis], ref0: Mapping, cells: Sequence[x11.Cell]
) -> None:
    bad = []
    for sl, _, m, _, decl, st in items:
        sc = x11.scores(decl, ref0, cells, x2=(study == "S2" and m == PRIMARY))
        bad += [(sl, m, k, sc[k], v) for k, v in st.items() if not x11.close(sc[k], v)]
    if bad:
        raise x11.ReproductionError(f"REPRODUCTION FAILED ({study} material metrics): {bad[:8]}")


def _level(
    study: str,
    level: float,
    stats: Mapping[tuple, Mapping[str, Any]],
    frozen: Mapping[str, Any],
    ref0: Mapping,
    items: Sequence[Analysis],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ex = existence_labels(stats, level)
    cnt = {
        "EXISTS_pos": sum(e["status"] == "EXISTS" and e["sign"] > 0 for e in ex.values()),
        "EXISTS_neg": sum(e["status"] == "EXISTS" and e["sign"] < 0 for e in ex.values()),
        "NOT_ESTABLISHED": sum(e["status"] == "NOT_ESTABLISHED" for e in ex.values()),
        "UNSUPPORTED": sum(e["status"] == "UNSUPPORTED" for e in ex.values()),
    }
    mat_true = {h for h, c in x11.gt_reference(frozen).items() if c["status"] == "TRUE"}
    exists = {h for h, e in ex.items() if e["status"] == "EXISTS"}
    lv: dict[str, Any] = {
        "counts": cnt,
        "frozen_true_not_exists": sorted("|".join(h) for h in mat_true - exists),
        "cells": {"|".join(h): {**e, "material": ref0[h]["status"]} for h, e in ex.items()},
        "methods": {},
    }
    rows = []
    for sl, kind_sl, m, kind, decl, _ in items:
        r = {
            "study": study,
            "ci_level": level,
            "slice": sl,
            "slice_kind": kind_sl,
            "method": m,
            "kind": kind,
            **material_scores(decl, ref0, frozen["cells"]),
            **existence_scores(decl, ex),
        }
        skip = ("study", "ci_level", "slice", "method")
        lv["methods"].setdefault(m, {})[sl] = {k: v for k, v in r.items() if k not in skip}
        rows.append(r)
    return lv, rows


def analyse_study(study: x11.Study) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """X14 for one study.

    Returns:
        The study's output and its CSV rows.

    Raises:
        x11.ReproductionError: If the frozen GT cells, the material-rule scores or (Study 1) the
            paper values are not reproduced.
    """
    a = json.loads(study.analysis.read_text(encoding="utf-8"))
    files, prov = x11.gt_files(study)
    units, n_eps = x11.load_rows(files)
    x11.check_cells(
        GT.gt_table(units)["cells"], a["gt"]["cells"], f"{study.name} gt_table vs stored"
    )
    stats = x11.cell_stats(units)
    frozen = x11.label_table(stats, *x11.VARIANTS["frozen"])
    x11.check_cells(frozen["cells"], a["gt"]["cells"], f"{study.name} frozen draws vs stored")
    ref0 = x11.gt_reference(frozen)
    items = analyses(study.name, a)
    _check_material(study.name, items, ref0, frozen["cells"])
    if study.name == "S1":
        decl = x11.decl_from_list(a["slices_result"]["pooled"]["metrics"][PRIMARY]["declared"])
        p = x11.scores(decl, ref0, frozen["cells"])
        got = (round(p["precision"], 2), round(p["recall"], 2), tuple(frozen["counts"].values()))
        if got != x11.PAPER_S1:
            raise x11.ReproductionError(f"S1 paper values not reproduced: {got}")
    out: dict[str, Any] = {
        "analysis": study.label,
        "analysis_sha256": x11.sha(study.analysis),
        "gt_records": prov,
        "n_episodes": n_eps,
        "n_labels": len(units),
        "reproduced": True,
        "n_analyses": len(items),
        "levels": {},
    }
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        lv, lrows = _level(study.name, level, stats, frozen, ref0, items)
        out["levels"][str(level)] = lv
        rows += lrows
    return out, rows


def run(
    records: Path = RECORDS, study_set: Mapping[str, x11.Study] | None = None
) -> dict[str, Any]:
    """X14 over both studies (`study_set`: other inputs than `x11.studies(records)`).

    Returns:
        `{"result", "rows"}`: the JSON output and the CSV rows.
    """
    res: dict[str, Any] = {
        "item": "X14",
        "declared": DECLARED,
        "post_hoc": True,
        "rule": "EXISTS(sign): CI excludes 0, any magnitude; NOT_ESTABLISHED: CI contains 0; "
        f"UNSUPPORTED: < {GT.GT_MIN_UNITS} units or < {GT.GT_MIN_EPS} episodes (unscored); "
        "frozen gt_p draws",
        "scores": "existence precision = confirmed / (confirmed + wrong sign + not established); "
        "ex_recall = declared EXISTS (any sign) / EXISTS; ex_recall_right_sign = confirmed / "
        "EXISTS; mat_* = frozen material rule (mat_precision = TP / (TP + FP), the paper's; "
        "mat_precision_signed = TRUE with the declared sign / (TRUE + NULL declared))",
        "levels": list(LEVELS),
        "studies": {},
    }
    rows: list[dict[str, Any]] = []
    for name, study in (study_set or x11.studies(records)).items():
        out, srows = analyse_study(study)
        res["studies"][name] = out
        rows += srows
    return {"result": res, "rows": rows}


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x14_existence.json` and `x14_existence.csv`."""
    d = Path(out_dir)
    write_text(d / "x14_existence.json", json.dumps(out["result"], indent=1))
    x11.write_csv(d / "x14_existence.csv", out["rows"])


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: recompute X14 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.records))
    write(out, a.out)
    for name, s in out["result"]["studies"].items():
        print(f"{name}: reproduced ({s['n_episodes']} eps, {s['n_labels']} labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

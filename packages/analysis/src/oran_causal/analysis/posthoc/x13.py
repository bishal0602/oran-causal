"""X13: relation attribution of declared neighbour (nbr) and far edges.

Post hoc and descriptive: no verdict, no frozen file changes, no new run. For every declared nbr or
far edge `(f, rel, k)` of every method it records the frozen GT label of the declared edge and of
the own edge of the same (family, KPI). A declared edge is "own-attributable" when it is GT-NULL
or GT-INDET while the own edge is GT-TRUE: the declaration may be the own effect attributed to the
wrong relation. It also counts the 20 far hypotheses (4 families x 5 KPIs) by GT label.

Study 1 is every method of every stored slice of `analysis_v4.json` (pooled, 10 x 60, 5 x 120 and
2 x 300 episodes; PMRT primary `loadsp_c+wby1s`). Study 2 is the nine PMRT combinations of the
pooled DISC and every DISC map but `MG:GT` of `disc_conf.json` (edge sign = sign of beta).

Reproduction comes first and is a hard stop: per method and slice the declaration counts and far
declaration counts must equal the stored metrics.
"""

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.posthoc.paths import PRIMARY, RECORDS, write_text

STATUSES = ("TRUE", "NULL", "INDET")
FAMILY_ORDER = ("carrier", "sleep", "ptx", "prot_min")
DECLARED = "EXTRAS_PROTOCOL.md amendment 2026-10-11 X9-X13"
RULE = (
    "own-attributable = declared nbr/far edge GT-NULL or GT-INDET while the same (family, KPI) "
    "own edge is GT-TRUE"
)
INPUTS = {
    "S1": (
        "study1_raw/e6p-disc-v4an-pmrt-1/out/analysis_v4.json",
        "scratchpad/e6_dev/runs/e6p-disc-v4an-pmrt-1/out/analysis_v4.json",
    ),
    "S2": (
        "study2_raw/e6p-conf-discan-2/out/disc_conf.json",
        "scratchpad/e6_dev/runs/e6p-conf-discan-2/out/disc_conf.json",
    ),
}
Edge = tuple[str, str, str, int]
Analysis = tuple[str, str, str, str, list[Edge], int, int]


class ReproductionError(RuntimeError):
    """A method's declaration counts do not equal the stored metrics."""


def label(c: Mapping[str, Any]) -> str:
    """`TRUE(+1)`, `NULL`, ... of a GT cell."""
    return c["status"] + ("" if not c["sign"] else f"({c['sign']:+d})")


def method_kind(study: str, m: str) -> str:
    """Kind of a Study 1 method or a Study 2 map."""
    if study == "S1":
        if m == PRIMARY:
            return "PMRT"
        return "PMRT variant" if "+" in m and not m.startswith("v2") else "baseline"
    if m in ("MG:rand", "blanket2"):
        return "comparator map"
    return "PMRT map" if m == "MG:PMRT" else "baseline map"


def edges_of(study: str, a: Mapping[str, Any]) -> list[Analysis]:
    """`(slice, slice kind, method, kind, edges, stored n_declared, stored far_declared)`."""
    out: list[Analysis] = []
    if study == "S1":
        for sl, r in a["slices_result"].items():
            for m, mm in r["metrics"].items():
                e = [(*h.split("|"), int(s)) for h, s in mm["declared"]]
                out.append(
                    (
                        sl,
                        r["kind"],
                        m,
                        method_kind(study, m),
                        e,
                        mm["n_declared"],
                        mm["far_declared"],
                    )
                )
        return out
    for m, mm in a["pmrt_combos"].items():
        e = [(*h.split("|"), int(s)) for h, s in mm["declared"]]
        kind = "PMRT" if m == PRIMARY else "PMRT variant"
        out.append(
            (
                "pooled",
                "pooled",
                m,
                kind,
                e,
                mm["metrics"]["n_declared"],
                mm["metrics"]["far_declared"],
            )
        )
    for m, rows in a["maps"].items():
        if m == "MG:GT":
            continue
        e = [(f, rel, k, (b > 0) - (b < 0)) for f, rel, k, b in rows]
        out.append(
            (
                "pooled",
                "pooled",
                m,
                method_kind(study, m),
                e,
                a["edge_quality"][m]["n_edges"],
                sum(1 for x in e if x[1] == "far"),
            )
        )
    return out


def _edge_row(
    study: str,
    item: Analysis,
    edge: Edge,
    cells: Mapping[tuple, Mapping[str, Any]],
) -> dict[str, Any]:
    sl, kind_sl, m, kind, _e, _n, _far = item
    f, rel, k, s = edge
    d, o = cells[(f, rel, k)], cells[(f, "own", k)]
    return {
        "study": study,
        "slice": sl,
        "slice_kind": kind_sl,
        "method": m,
        "kind": kind,
        "edge": f"{f}|{rel}|{k}",
        "relation": rel,
        "declared_sign": s,
        "gt_edge": label(d),
        "gt_edge_mean": d["mean"],
        "gt_own": label(o),
        "gt_own_mean": o["mean"],
        "declared_sign_matches_gt": d["status"] == "TRUE" and s == d["sign"],
        "declared_sign_equals_own_sign": o["status"] == "TRUE" and s == o["sign"],
        "own_attributable": d["status"] in ("NULL", "INDET") and o["status"] == "TRUE",
    }


def _slice_counts(rows: Sequence[Mapping[str, Any]], cells: Mapping[tuple, Any]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        f, rel, k = r["edge"].split("|")
        d, o = cells[(f, rel, k)], cells[(f, "own", k)]
        c[f"{rel}_declared"] += 1
        c[f"{rel}_gt_{d['status']}"] += 1
        c[f"{rel}_own_true"] += o["status"] == "TRUE"
        c[f"{rel}_own_attributable"] += r["own_attributable"]
    return c


def _by_slice_kind(v: dict[str, Any]) -> dict[str, Any]:
    tot: dict[str, Counter] = {}
    for sl, c in v["slices"].items():
        kk = "pooled" if sl == "pooled" else sl.split("_")[0]
        t = tot.setdefault(kk, Counter({"n_slices": 0}))
        t["n_slices"] += 1
        t.update(c)
    return {k: dict(t) for k, t in tot.items()}


def far_counts(gt_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The 20 far hypotheses by GT label: all, per family and per cell."""
    far = [c for c in gt_cells if c["relation"] == "far"]
    return {
        "all_20": dict(Counter(c["status"] for c in far)),
        "by_family": {
            f: dict(Counter(c["status"] for c in far if c["family"] == f)) for f in FAMILY_ORDER
        },
        "cells": {f"{c['family']}|far|{c['kpi']}": label(c) for c in far},
    }


def analyse_study(study: str, a: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """X13 for one study's stored analysis.

    Returns:
        The study's output and its CSV rows.

    Raises:
        ReproductionError: If a method's declaration counts differ from the stored metrics.
    """
    cells = {(c["family"], c["relation"], c["kpi"]): c for c in a["gt"]["cells"]}
    summ: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    bad = []
    for item in edges_of(study, a):
        sl, _kind_sl, m, kind, e, n_dec, far_dec = item
        if len(e) != n_dec or sum(1 for x in e if x[1] == "far") != far_dec:
            bad.append((sl, m, len(e), n_dec, far_dec))
        srows = [_edge_row(study, item, x, cells) for x in e if x[1] != "own"]
        rows += srows
        summ.setdefault(m, {"kind": kind, "slices": {}})["slices"][sl] = dict(
            _slice_counts(srows, cells)
        )
    if bad:
        raise ReproductionError(f"REPRODUCTION FAILED ({study}): {bad[:5]}")
    for v in summ.values():
        v["by_slice_kind"] = _by_slice_kind(v)
    out = {"far_hypotheses_by_gt_label": far_counts(a["gt"]["cells"]), "methods": summ}
    return out, rows


def run(
    records: Path = RECORDS, inputs: Mapping[str, tuple[str | Path, str]] = INPUTS
) -> dict[str, Any]:
    """X13 from the frozen record sets (`inputs`: study -> analysis path, recorded label).

    Returns:
        `{"result", "rows"}`: the JSON output and the CSV rows.
    """
    res: dict[str, Any] = {
        "item": "X13",
        "declared": DECLARED,
        "post_hoc": True,
        "rule": RULE,
        "studies": {},
    }
    rows: list[dict[str, Any]] = []
    for study, (rel_path, label_path) in inputs.items():
        a = json.loads((records / rel_path).read_text(encoding="utf-8"))
        out, srows = analyse_study(study, a)
        res["studies"][study] = {
            "analysis": label_path,
            "reproduced": True,
            "far_hypotheses_by_gt_label": out["far_hypotheses_by_gt_label"],
            "methods": out["methods"],
        }
        rows += srows
    return {"result": res, "rows": rows}


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x13_attribution.json` and `x13_attribution.csv`."""
    d = Path(out_dir)
    write_text(d / "x13_attribution.json", json.dumps(out["result"], indent=1))
    with (d / "x13_attribution.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(out["rows"][0]))
        w.writeheader()
        w.writerows(out["rows"])


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: recompute X13 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.records))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write(out, a.out)
    for study, s in out["result"]["studies"].items():
        print(f"{study}: far cells {s['far_hypotheses_by_gt_label']['all_20']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

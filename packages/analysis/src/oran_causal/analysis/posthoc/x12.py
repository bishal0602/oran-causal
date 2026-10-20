"""X12: threshold choice of the Study 2 associational baselines, tuned on dev versus on placebo.

Post hoc and descriptive: no verdict, no frozen file changes, no new run. Per Study 2 map (corr,
granger, shap_gbdt, int, qacm and two_tower at `@dev` and `@plc`, `granger_by` untuned, `MG:PMRT`
as the reference row) it reports the declared map edges against the Study 2 GT cells (TRUE with the
right sign, TRUE with the wrong sign, NULL, INDET), the placebo declarations, the threshold and its
source.

* `@dev`: the far-FPR rule (max_far 1) on the 20 incumbent episodes of `dev_conf`.
* `@plc`: tau is the (MAX_FP + 1)-th largest finite score on the 200 confounded placebo episodes
  (MAX_FP 1), so it is set on the placebo logs themselves and its placebo-declaration count is
  in-sample for the placebo criterion.

Reproduction comes first and is a hard stop: the recomputed edge counts must equal the stored
`edge_quality` and the report's `option_a.map_quality` for every map.
"""

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.posthoc.paths import PROTOCOL_NAME, RECORDS, write_text
from oran_causal.analysis.posthoc.refs import RefereeTools

METHODS = ("corr", "granger", "shap_gbdt", "int", "qacm", "two_tower")
KEYS = ("edges", "true_right", "true_wrong", "null", "indet", "placebo_edges")
DECLARED = f"{PROTOCOL_NAME} amendment 2026-10-11 X12"
COLUMNS = (
    "arm",
    "method",
    "variant",
    "edges",
    "true_right",
    "true_wrong",
    "null",
    "indet",
    "unknown",
    "placebo_edges",
    "placebo_declared_hyp",
    "disc_declared_hyp",
    "precision_true_right",
    "recall_true_right",
    "threshold",
    "threshold_source",
)
DEV_SOURCE = "dev_conf, 20 incumbent episodes, far-FPR rule max_far 1 (baselines_disc.tune_tau)"
PLC_SOURCE = (
    "placebo, 200 PlaceboIncumbent episodes: (MAX_FP + 1)-th largest finite score, MAX_FP 1 "
    "(K-B placebo_tau); IN-SAMPLE for the placebo criterion"
)


class ReproductionError(RuntimeError):
    """The recomputed edge counts do not equal the stored ones."""


def arms() -> list[str]:
    """The twelve Study 2 maps plus the untuned granger and the PMRT reference, in table order."""
    return (
        ["MG:PMRT"] + [f"MG:{m}@{v}" for m in METHODS for v in ("dev", "plc")] + ["MG:granger_by"]
    )


def _arm_row(arm: str, d: Mapping[str, Any], q: Mapping[str, Any], pl: Any) -> dict[str, Any]:
    r: dict[str, Any] = {
        "arm": arm,
        "method": arm[3:].partition("@")[0],
        "variant": arm.split("@")[1]
        if "@" in arm
        else ("untuned" if arm == "MG:granger_by" else "PMRT"),
        "edges": q["n_edges"],
        "true_right": q["n_true_right"],
        "true_wrong": q["n_true_wrong"],
        "null": q["n_null"],
        "indet": q["n_indet"],
        "unknown": q["n_unknown"],
        "placebo_edges": None if pl is None else len(pl),
        "precision_true_right": q["precision_true_right"],
        "recall_true_right": q["recall_true_right"],
    }
    m = r["method"]
    if arm == "MG:PMRT":
        r.update(
            threshold=None,
            threshold_source=f"PMRT {d['primary']} (design p-values, alpha .05; no tuned tau); "
            "placebo declarations = K0 n_declared",
            placebo_declared_hyp=d["K0"]["n_declared"],
            disc_declared_hyp=None,
        )
    elif arm == "MG:granger_by":
        r.update(
            threshold=None,
            threshold_source="BY at q .05 (untuned, one variant)",
            placebo_declared_hyp=len(d["provenance"][arm].get("placebo_declared") or []),
            disc_declared_hyp=len(d["provenance"][arm].get("declared") or []),
        )
    else:
        b = d["baselines"][m]
        v = r["variant"]
        t = b["tau_dev"] if v == "dev" else b["tau_plc"]
        r.update(
            threshold=t.get("tau", t.get("tau_r2")),
            threshold_rule=t,
            threshold_source=DEV_SOURCE if v == "dev" else PLC_SOURCE,
            placebo_declared_hyp=b["n_placebo_declared"][v],
            disc_declared_hyp=b["n_disc_declared"][v],
        )
    return r


def _matches(r: Mapping[str, Any], eq: Mapping[str, Any], f: Mapping[str, Any]) -> bool:
    counts = (r["edges"], r["true_right"], r["true_wrong"], r["null"], r["indet"], r["unknown"])
    stored = (
        eq["n_edges"],
        eq["n_true_right"],
        eq["n_true_wrong"],
        eq["n_null"],
        eq["n_indet"],
        eq["n_unknown"],
    )
    return counts == stored and all(r[k] == f[k] for k in KEYS)


def analyse(
    disc: Mapping[str, Any], figures_data: Mapping[str, Any], tools: RefereeTools
) -> dict[str, Any]:
    """The X12 table.

    Args:
        disc: Stored Study 2 discovery output (`conf_disc_conf.json`).
        figures_data: The Studies 1-3 report data (`figures_data.json`).
        tools: Referee functions (map decoder and edge quality).

    Returns:
        The output JSON.

    Raises:
        ReproductionError: If an edge count differs from the stored map quality.
    """
    fd = figures_data["option_a"]["map_quality"]
    cells = disc["gt"]["cells"]
    n_true = sum(c["status"] == "TRUE" for c in cells)
    names = arms()
    rows, bad = [], []
    for a in names:
        m = tools.map_from_json(disc["maps"][a])
        q = tools.edge_quality(m, cells)
        r = _arm_row(a, disc, q, disc["placebo_maps"].get(a))
        if not _matches(r, disc["edge_quality"][a], fd[a]):
            bad.append(a)
        rows.append(r)
    repro = {
        "ok": not bad,
        "mismatch": bad,
        "n_maps": len(names),
        "checked_against": [
            "conf_disc_conf.json edge_quality",
            "figures_data.json option_a.map_quality",
        ],
    }
    if bad:
        raise ReproductionError(
            f"X12: edge counts do not reproduce the frozen map quality: {repro}"
        )
    return {
        "schema": "x12-threshold/1",
        "declared": DECLARED,
        "status": "POST HOC, DESCRIPTIVE",
        "repro": repro,
        "gt": {"n_cells": len(cells), "n_true": n_true, "n_episodes": disc["gt"]["n_episodes"]},
        "note_plc": "@plc thresholds are set on the 200 placebo episodes that also define the "
        "placebo-declaration count, so @plc's placebo declarations (<= MAX_FP = 1 by "
        "construction, ties aside) are in-sample; @dev thresholds come from dev_conf and its "
        "placebo count is out-of-sample.",
        "placebo_edges_def": "edges of the map the same declaration rule builds on the placebo "
        "episodes (conf_disc_conf.json placebo_maps; = figures_data.json placebo_edges); "
        "placebo_declared_hyp counts declared hypotheses before the sign-0 drop",
        "rows": rows,
    }


def run(records: Path = RECORDS, *, tools: RefereeTools) -> dict[str, Any]:
    """X12 from the frozen record sets."""
    sysd = records / "system_artifacts"
    disc = json.loads((sysd / "decision" / "conf_disc_conf.json").read_text(encoding="utf-8"))
    fd = json.loads((sysd / "report_2026-10-02" / "figures_data.json").read_text(encoding="utf-8"))
    return analyse(disc, fd, tools)


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x12_threshold.json` and `x12_threshold.csv`."""
    d = Path(out_dir)
    write_text(d / "x12_threshold.json", json.dumps(out, indent=1, default=float))
    d.mkdir(parents=True, exist_ok=True)
    with (d / "x12_threshold.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in out["rows"]:
            w.writerow([r.get(c) for c in COLUMNS])


def main(argv: Sequence[str] | None = None, *, tools: RefereeTools) -> int:
    """Command line: recompute X12 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.records), tools=tools)
    write(out, a.out)
    print(f"repro ok; GT TRUE cells {out['gt']['n_true']} / {out['gt']['n_cells']}")
    return 0

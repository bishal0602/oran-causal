"""X10: diagnosis of the certified-safe PMRT referee's gate (Study 3).

Post hoc and descriptive: no verdict, no frozen file changes, no new simulation. For every request
class (family, d) that MapGateV2 of the PMRT map can defer, and each of its guard edges (relation
own, nbr or far; KPI v or rlf), it gives the frozen status (DECLARED, CERTIFIED or UNRESOLVED),
h = -d, h * beta and its standard error (the PMRT design slope in the artifact's `pmrt` bounds), the
one-sided 90 % upper bound UB90 = h * beta + z90 * se, the tolerance tau(f, d, k), and the Study 2
ground truth of the same edge (mean, 95 % interval, TRUE, NULL or INDET) as the harm of a deferral
h * GT. An edge is "harmful beyond tau" when h * GT mean > tau and "CI beyond tau" when the whole
h * GT interval lies above tau. The GT oracle's status of the same edge is shown for comparison.

Reproduction comes first and is a hard stop: certifying the artifact's PMRT map with its bounds
and tau must give the frozen classes and uncertified set, and the stored EVAL records must give
`CS:PMRT` identical to `never_sleep` on all 160 seeds.
"""

import argparse
import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc.paths import ARTIFACTS, PROTOCOL_NAME, RECORDS, write_text
from oran_causal.analysis.posthoc.refs import RefereeTools
from oran_causal.analysis.posthoc.x9 import REF, eval_files, identity
from oran_causal.analysis.system import stats, study3
from oran_causal.core.gate import DECLARED, RHO, UNRESOLVED, Z90, bound_ub

ARM = "CS:PMRT"
RTOL = 1e-9
FD_RTOL = 1e-6
STATUS = "POST HOC, DESCRIPTIVE"
DECLARED_BY = f"{PROTOCOL_NAME} amendment 2026-10-11 X10"
COLUMNS = (
    "class",
    "rel",
    "kpi",
    "status",
    "h",
    "beta",
    "se",
    "h_beta",
    "ub90",
    "ub90_used",
    "tau",
    "gt_mean",
    "gt_ci_lo",
    "gt_ci_hi",
    "gt_label",
    "h_gt_mean",
    "gt_harm_beyond_tau",
    "gt_ci_beyond_tau",
    "cs_gt_status",
    "cs_gt_ub",
)


class ReproductionError(RuntimeError):
    """The frozen certification outcome is not reproduced; the analysis stops."""


def _close(a: Any, b: Any, rtol: float = RTOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=1e-15))


def same_cert(got: Mapping[str, Any], frozen: Mapping[str, Any], rtol: float = RTOL) -> bool:
    """Whether certified classes equal frozen ones (status, bound and tau at `rtol`)."""
    if set(got) != set(frozen):
        return False
    for c, g in got.items():
        f = frozen[c]
        if (
            g["can_defer"] != f["can_defer"]
            or g["certified"] != f["certified"]
            or len(g["edges"]) != len(f["edges"])
        ):
            return False
        for ge, fe in zip(g["edges"], f["edges"], strict=True):
            if ge[:3] != fe[:3] or not _close(ge[3], fe[3], rtol) or not _close(ge[4], fe[4], rtol):
                return False
    return True


def reproduction(
    cert: Mapping[str, Any],
    art: Mapping[str, Any],
    fd_cert: Mapping[str, Any],
    bounds_json: Mapping[str, Any],
    R: stats.Records,
) -> dict[str, Any]:
    """The reproduction checks of the certification and of `CS:PMRT` against `never_sleep`."""
    arm = art["arms"][ARM]
    seeds = sorted(s for s in R if ARM in R[s] and REF in R[s])
    rp: dict[str, Any] = {
        "bounds_equal_cs_bounds_json": art["bounds"]["pmrt"] == bounds_json["pmrt"],
        "certify_equals_artifact": same_cert(cert["classes"], arm["certification"]),
        "uncertified_equals_artifact": sorted(map(list, cert["uncertified"]))
        == sorted(arm["uncertified"]),
        "uncertified_equals_fd": sorted(map(list, cert["uncertified"]))
        == sorted(fd_cert[ARM]["uncertified"]),
        "fd_classes_equal_7sig": all(
            same_cert({c: cert["classes"][c]}, {c: v}, FD_RTOL)
            for c, v in fd_cert[ARM]["classes"].items()
        ),
        "cs_pmrt_identical_to_never_sleep": identity(R, seeds, ARM, REF),
    }
    rp["ok"] = bool(
        all(v for k, v in rp.items() if k != "cs_pmrt_identical_to_never_sleep")
        and rp["cs_pmrt_identical_to_never_sleep"]["identical_seeds"] == 160
    )
    return rp


def _edge_row(
    key: str,
    f: str,
    d: int,
    edge: Sequence[Any],
    bounds: Mapping,
    gt: Mapping,
    gt_cert: Mapping[str, Any],
) -> dict[str, Any]:
    h = -d
    rel, k, status, _ub, t = edge
    b = bounds[(f, rel, k)]
    g = gt.get((f, rel, k))
    hci = sorted([h * g["ci"][0], h * g["ci"][1]]) if g else None
    gte = {(e[0], e[1]): e for e in gt_cert[key]["edges"]}
    e = {
        "class": key,
        "family": f,
        "d": d,
        "h": h,
        "rel": rel,
        "kpi": k,
        "status": status,
        "beta": b["beta"],
        "se": b["se"],
        "h_beta": h * b["beta"],
        "ub90": bound_ub(b, h),
        "ub90_used": status != DECLARED,
        "tau": t,
        "gt_mean": g["mean"] if g else None,
        "gt_ci95": g["ci"] if g else None,
        "gt_label": g["status"] if g else None,
        "gt_delta": g["delta"] if g else None,
        "h_gt_mean": h * g["mean"] if g else None,
        "h_gt_ci95": hci,
        "gt_harm_beyond_tau": bool(g and h * g["mean"] > t),
        "gt_ci_beyond_tau": bool(g and hci is not None and hci[0] > t),
        "cs_gt_status": gte[(rel, k)][2] if (rel, k) in gte else None,
        "cs_gt_ub": gte[(rel, k)][3] if (rel, k) in gte else None,
    }
    if not math.isfinite(e["ub90"]):
        e["ub90"] = None
    return e


def diagnose(
    cert: Mapping[str, Any],
    art: Mapping[str, Any],
    bounds: Mapping,
    gt_cells: Sequence[Mapping[str, Any]],
    cs_eval: Mapping[str, Any],
) -> tuple[list[dict], list[dict]]:
    """The class table and the edge table of the deferrable classes."""
    gt = {(c["family"], c["relation"], c["kpi"]): c for c in gt_cells}
    gt_cert = art["arms"]["CS:GT"]["certification"]
    pol = cs_eval["arms"][ARM]["counts"]["policy"]
    classes, rows = [], []
    for key, c in cert["classes"].items():
        if not c["can_defer"]:
            continue
        f, d = key.split("|")[0], int(key.split("|")[1])
        edges = [_edge_row(key, f, d, e, bounds, gt, gt_cert) for e in c["edges"]]
        rows += edges
        unres = [e for e in edges if e["status"] == UNRESOLVED]
        sign = "+" if d > 0 else "-"
        classes.append(
            {
                "class": key,
                "family": f,
                "d": d,
                "certified": c["certified"],
                "frozen_decision": art["arms"][ARM]["decision_table"][f"{f}{sign}"],
                "withheld_units_eval": pol.get(f"uncert_{f}", 0) if not c["certified"] else 0,
                "deferred_units_eval": pol.get(f"defer_{f}", 0),
                "unresolved": [[e["rel"], e["kpi"]] for e in unres],
                "unresolved_gt_harm_beyond_tau": [
                    [e["rel"], e["kpi"]] for e in unres if e["gt_harm_beyond_tau"]
                ],
                "cs_gt_certified": gt_cert[key]["certified"],
            }
        )
    return classes, rows


def analyse(
    records: Path,
    artifacts: Path,
    tools: RefereeTools,
    certify: Callable[..., dict] | None = None,
) -> dict[str, Any]:
    """X10 from the frozen record sets.

    Args:
        records: The record sets.
        artifacts: The frozen artifacts.
        tools: Referee functions (decoders and MapGateV2's deferrable classes).
        certify: Certification of a map; defaults to `tools.certify_map`.

    Returns:
        `{"result", "rows"}`: the JSON output and the CSV edge rows.

    Raises:
        ReproductionError: If the frozen certification outcome is not reproduced.
    """
    sysd = records / "system_artifacts"
    art = json.loads((artifacts / "study3_certsafe.json").read_text(encoding="utf-8"))
    fd = json.loads((sysd / "report_2026-10-02" / "figures_data.json").read_text(encoding="utf-8"))[
        "certsafe"
    ]
    disc = json.loads((sysd / "decision" / "conf_disc_conf.json").read_text(encoding="utf-8"))
    bjs = json.loads((sysd / "decision" / "cs_bounds.json").read_text(encoding="utf-8"))
    m = tools.map_from_json(art["arms"][ARM]["map"])
    tau = tools.tau_from_json(art["tau"])
    pm = tools.bounds_from_json(art["bounds"]["pmrt"]) or {}
    cert = (certify or tools.certify_map)(m, pm, tau)
    R, _ = stats.load_jobs(eval_files(records, "e6p-cs-eval-1"), study3.job_filter("eval"))
    rp = reproduction(cert, art, fd["cert"], bjs, R)
    if not rp["ok"]:
        raise ReproductionError(
            "X10: the frozen certification outcome is not reproduced: "
            + json.dumps(rp, indent=1, default=str)
        )
    ev = json.loads((sysd / "decision" / "cs_eval.json").read_text(encoding="utf-8"))
    classes, rows = diagnose(cert, art, pm, disc["gt"]["cells"], ev)
    rep = {
        "schema": "x10-gate-diag/1",
        "declared": DECLARED_BY,
        "status": STATUS,
        "repro": rp,
        "conventions": {
            "h": "-d (harm direction of a deferral; certsafe.py)",
            "z90": Z90,
            "rho": RHO,
            "tau_rule": art["constants"]["tau_rule"],
            "gt": "Study 2 knockout GT (conf_disc_conf.json gt.cells, 40 episodes, 95 % "
            "episode-cluster CI, label TRUE / NULL / INDET with the frozen relative delta), the "
            "scale and orientation certsafe.gt_ci_bounds uses for CS:GT",
        },
        "classes": classes,
        "edges": rows,
    }
    return {"result": rep, "rows": rows}


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x10_gate_diag.json` and `x10_gate_diag.csv`."""
    d = Path(out_dir)
    write_text(d / "x10_gate_diag.json", json.dumps(out["result"], indent=1, default=float))
    with (d / "x10_gate_diag.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for e in out["rows"]:
            w.writerow(
                [
                    e["class"],
                    e["rel"],
                    e["kpi"],
                    e["status"],
                    e["h"],
                    e["beta"],
                    e["se"],
                    e["h_beta"],
                    e["ub90"],
                    e["ub90_used"],
                    e["tau"],
                    e["gt_mean"],
                    *(e["gt_ci95"] or [None, None]),
                    e["gt_label"],
                    e["h_gt_mean"],
                    e["gt_harm_beyond_tau"],
                    e["gt_ci_beyond_tau"],
                    e["cs_gt_status"],
                    e["cs_gt_ub"],
                ]
            )


def main(argv: Sequence[str] | None = None, *, tools: RefereeTools) -> int:
    """Command line: recompute X10 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = analyse(Path(a.records), Path(a.artifacts), tools)
    write(out, a.out)
    print(f"repro ok {out['result']['repro']['ok']}")
    for c in out["result"]["classes"]:
        print(c)
    return 0

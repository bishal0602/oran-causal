"""Measured cost of the bridging study in Kaggle-reference CPU-s (protocol EXP_B section 6).

Kaggle records count as measured. A record of another host is multiplied by that host's factor:
the arm's own `f` when the factor file has that exact arm name, else the pooled `*` factor, with
no transfer across arm names. An infeasible unit counts at the budget.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from oran_causal.analysis.records import Record, arm_of, file_sha256, iter_records, status_of
from oran_causal.analysis.system.report_data import FROZEN_NEWLINE
from oran_causal.analysis.system.stats import dumps

SCHEMA = "xm-expb-costs/1"
REF_PLATFORM = "kaggle"
COSTS_JSON = "exp_b_costs.json"
COSTS_MD = "EXP_B_COSTS.md"

type Factors = Mapping[str, Mapping[str, Mapping[str, Any]]]


def _new_arm() -> dict[str, Any]:
    return {
        "n": 0,
        "ok": 0,
        "infeasible": 0,
        "ref_cpu_s": 0.0,
        "max_ref_cpu_s": 0.0,
        "raw_cpu_s": defaultdict(float),
        "factor": {},
    }


def ref_cpu_s(r: Record, arm: str, a: dict[str, Any], factors: Factors) -> float:
    """Reference CPU-s of one record's raw CPU-s; notes the factor used in `a["factor"]`."""
    host = r.get("host") or {}
    raw = float(r.get("cpu_s") or 0.0)
    a["raw_cpu_s"][host.get("platform")] += raw
    if host.get("platform") == REF_PLATFORM:
        return raw
    hk = f"{host.get('platform')}|{host.get('cpu_model')}"
    src = arm if hk in factors.get(arm, {}) else "*"
    f = float(factors[src][hk]["f"])
    a["factor"][hk] = {"from": src, "f": f}
    return raw * f


def per_arm(
    records: Iterable[Record], factors: Factors, budget: float
) -> dict[str, dict[str, Any]]:
    """Unit counts and reference CPU-s per arm, in first-seen order."""
    arms: dict[str, dict[str, Any]] = defaultdict(_new_arm)
    for r in records:
        arm, st = arm_of(r), status_of(r)
        a = arms[arm]
        a["n"] += 1
        a[st] = a.get(st, 0) + 1
        ref = ref_cpu_s(r, arm, a, factors)
        if st == "infeasible":
            ref = budget
        a["ref_cpu_s"] += ref
        a["max_ref_cpu_s"] = max(a["max_ref_cpu_s"], ref)
    return arms


def build(
    records: Iterable[Record], factors_path: str | Path, budget: float, label: str | None = None
) -> dict[str, Any]:
    """The cost document (`exp_b_costs.json`).

    Args:
        records: Merged records, in file order.
        factors_path: Non-reference host factors (`{"factors": {arm | "*": {host: {"f"}}}}`).
        budget: Per-unit CPU-s budget.
        label: Factor file name recorded in the output; default the path in POSIX form.

    Returns:
        `{"schema", "reference", "factors_file", "factors_sha256", "budget_cpu_s", "arms",
        "total_ref_core_h"}`; arms sorted by name.
    """
    path = Path(factors_path)
    fac = json.loads(path.read_text(encoding="utf-8"))["factors"]
    arms = per_arm(records, fac, float(budget))
    res: dict[str, Any] = {
        "schema": SCHEMA,
        "reference": REF_PLATFORM,
        "factors_file": label if label is not None else path.as_posix(),
        "factors_sha256": file_sha256(path),
        "budget_cpu_s": float(budget),
        "arms": {k: {**v, "raw_cpu_s": dict(v["raw_cpu_s"])} for k, v in sorted(arms.items())},
    }
    res["total_ref_core_h"] = sum(v["ref_cpu_s"] for v in arms.values()) / 3600
    return res


def markdown(res: Mapping[str, Any]) -> str:
    """`EXP_B_COSTS.md` of a cost document."""
    lines = [
        "# Experiment B measured cost (Kaggle reference CPU-s)",
        "",
        f"Factors: `{res['factors_file']}` (sha256 {res['factors_sha256'][:12]}...); infeasible "
        f"units at the {res['budget_cpu_s']:.0f} s budget. Total {res['total_ref_core_h']:.1f} "
        "ref core-h.",
        "",
        "| arm | units | ok | infeasible | ref core-h | max ref CPU-s / unit | non-ref factor |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in res["arms"].items():
        fs = "; ".join(f"{x['from']} {x['f']:.3f}" for x in v["factor"].values())
        lines.append(
            f"| {k} | {v['n']} | {v['ok']} | {v['infeasible']} | {v['ref_cpu_s'] / 3600:.2f} | "
            f"{v['max_ref_cpu_s']:.0f} | {fs} |"
        )
    return "\n".join(lines) + "\n"


def write(res: Mapping[str, Any], out_dir: str | Path, newline: str = FROZEN_NEWLINE) -> None:
    """Writes `exp_b_costs.json` (`newline` line ends; stored: CRLF) and the LF markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / COSTS_JSON).write_text(dumps(res), encoding="utf-8", newline=newline)
    (out / COSTS_MD).write_text(markdown(res), encoding="utf-8", newline="\n")


def costs(
    merged: str | Path,
    factors: str | Path,
    budget: float,
    out_dir: str | Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Builds the cost document of a merged record file and writes it to `out_dir` if given."""
    res = build(iter_records(merged), factors, budget, label)
    if out_dir is not None:
        write(res, out_dir)
    return res

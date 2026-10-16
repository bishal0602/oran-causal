"""Integrity checks of a record set (V11) and the FINAL / PROVISIONAL label.

FINAL needs every check to pass: the protocol file is frozen and hashed, all records come from
the freeze commit (or a listed amendment), are clean and stamped, no expected unit is missing,
every candidate edge is present, the datasets agree across arms, one platform and one package
set produced the records, and the BY declarations re-derive from the recorded p-values.
"""

import dataclasses
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.controlled.protocol import FROZEN_MARK, Q_BY
from oran_causal.analysis.controlled.spec import Spec, not_in_grid
from oran_causal.analysis.records import (
    PathLike,
    Record,
    file_sha256,
    status_of,
    to_result,
    unit_of,
)
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.core.scoring import PLACEBO_CONF, not_testable_edges
from oran_causal.sim.controlled.datasets.truth import truth_for

Integrity = dict[str, Any]
Amendments = dict[str, list[dict[str, Any]]]
LIST_LIMIT = 20
LONG_LIST_LIMIT = 50
NO_AMENDMENTS: Amendments = {"amendments": [], "persistent_errors": []}


@dataclasses.dataclass(frozen=True)
class IntegrityInputs:
    """External facts the integrity checks compare the records with.

    Attributes:
        freeze_commit: Commit the protocol was frozen at.
        spec_sha: LF sha256 of the spec file.
        protocol: Result of `protocol_check`.
        amendments: Result of `load_amendments`.
        dep: Dependence table (`fmax.load_dependence`).
        dev_records: DEV records for the tune reproducibility check.
    """

    freeze_commit: str | None = None
    spec_sha: str | None = None
    protocol: dict[str, Any] | None = None
    amendments: Amendments | None = None
    dep: dict[str, Any] | None = None
    dev_records: list[Record] | None = None


def by_declare(ps: Sequence[float], q: float = Q_BY, m: int | None = None) -> list[bool]:
    """Benjamini-Yekutieli step-up at level `q`.

    Args:
        ps: P-values of the tested hypotheses.
        q: FDR level.
        m: Family size when larger than `len(ps)`: untested members count in the correction.

    Returns:
        Whether each hypothesis is rejected.
    """
    if not ps:
        return []
    size = max(len(ps), m or 0)
    padded = np.array([*ps, *([math.nan] * (size - len(ps)))], float)
    return [bool(x) for x in weighted_step_up(padded, None, q, True)[: len(ps)]]


def load_amendments(path: PathLike | None) -> Amendments:
    """Reads the amendments file; an empty set without a path."""
    if not path:
        return {"amendments": [], "persistent_errors": []}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "amendments": data.get("amendments", []),
        "persistent_errors": data.get("persistent_errors", []),
    }


def protocol_check(spec: Spec, protocol_path: PathLike | None) -> dict[str, Any]:
    """Checks that the protocol file is frozen and its LF sha256 equals the spec's constant."""
    want = spec.get("protocol_sha256")
    if not protocol_path or not Path(protocol_path).exists():
        return {"ok": False, "why": "protocol file missing"}
    got = file_sha256(protocol_path)
    text = Path(protocol_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    frozen = any(line.startswith(FROZEN_MARK) for line in text.split("\n"))
    ok = bool(want) and str(want).lower() == got and frozen
    return {"ok": ok, "spec_protocol_sha256": want, "file_sha256": got, "frozen": frozen}


def _allowed_commits(amendments: Amendments) -> dict[str, set]:
    allowed: dict[str, set] = {}
    for amendment in amendments["amendments"]:
        for key in amendment.get("keys", []):
            allowed.setdefault(key, set()).add(amendment.get("commit"))
    return allowed


def _code_checks(
    records: list[Record], freeze_commit: str | None, allowed: dict[str, set]
) -> dict[str, Any]:
    commits = sorted({str((r.get("code") or {}).get("commit")) for r in records})
    bad_commit = sorted(
        r["key"]
        for r in records
        if (r.get("code") or {}).get("commit") != freeze_commit
        and (r.get("code") or {}).get("commit") not in allowed.get(r["key"], set())
    )
    dirty = sorted(r["key"] for r in records if (r.get("code") or {}).get("dirty") is not False)
    return {"commits": commits, "bad_commit": bad_commit, "dirty": dirty}


def _bad_stamps(records: list[Record], spec: Spec, spec_sha: str | None) -> list[str]:
    protocol_sha = spec.get("protocol_sha256")
    return sorted(
        r["key"]
        for r in records
        if (r.get("run_mode") or {}).get("protocol_sha256") != protocol_sha
        or (spec_sha is not None and (r.get("integrity") or {}).get("spec_file_sha256") != spec_sha)
    )


def _incomplete_candidates(records: list[Record]) -> list[str]:
    truths: dict = {}
    incomplete = []
    for record in records:
        if status_of(record) != "ok":
            continue
        u = unit_of(record)
        truth = truths.setdefault((u["world"], u["regime"]), truth_for(u["world"], u["regime"]))
        recorded = {(e["source"], e["target"]) for e in record.get("edges", [])}
        if recorded != set(truth.edges) | set(truth.null_edges):
            incomplete.append(record["key"])
    return incomplete


def _dataset_checks(records: list[Record]) -> dict[str, Any]:
    hashes: dict[tuple, set] = defaultdict(set)
    platforms: dict[tuple, set] = defaultdict(set)
    unstamped, unstamped_not_ok = [], []
    for record in records:
        u = unit_of(record)
        dataset = (u["world"], u["regime"], u["lam"], u["n"], u["kappa"], u["seed"])
        if record.get("dataset_sha256"):
            hashes[dataset].add(record["dataset_sha256"])
        elif status_of(record) == "ok":
            unstamped.append(record["key"])
        else:
            unstamped_not_ok.append(record["key"])
        platforms[dataset].add(str((record.get("host") or {}).get("platform")))
    return {
        "mismatch": sorted(str(k) for k, v in hashes.items() if len(v) > 1),
        "unstamped": unstamped,
        "unstamped_not_ok": unstamped_not_ok,
        "multi_platform": sorted(str(k) for k, v in platforms.items() if len(v) > 1),
    }


def _packages_uniform(records: list[Record], spec: Spec) -> tuple[bool, int]:
    sets = Counter(
        json.dumps(r.get("pkgs"), sort_keys=True) for r in records if status_of(r) == "ok"
    )
    lock = spec.get("pkgs_lock")
    ok = len(sets) == 1 and (lock is None or json.loads(next(iter(sets))) == lock)
    return ok, len(sets)


def by_recheck(records: list[Record], spec: Spec) -> dict[str, Any]:
    """Re-derives the BY declarations of every p-arm record from its recorded p-values.

    The family is the primary action sources including `P_placebo`, without the confounded
    placebo, the lagged KPIs and the not-testable edges.
    """
    total = agree = 0
    disagree: list[str] = []
    for record in records:
        if (
            status_of(record) != "ok"
            or spec["arms"].get(unit_of(record)["arm"], {}).get("declare") != "by"
        ):
            continue
        untestable = {f"{s}->{t}" for s, t in not_testable_edges(to_result(record))}
        family = [
            e
            for e in record.get("edges", [])
            if not e["source"].startswith("K")
            and e["source"] != PLACEBO_CONF
            and e.get("p") is not None
            and f"{e['source']}->{e['target']}" not in untestable
        ]
        mine = by_declare(
            [float(e["p"]) for e in family], m=(record.get("notes") or {}).get("by_family_m")
        )
        total += 1
        if all(bool(e.get("declared")) == m for e, m in zip(family, mine, strict=True)):
            agree += 1
        elif len(disagree) < LIST_LIMIT:
            disagree.append(record["key"])
    return {"records": total, "agree": agree, "disagree_first": disagree}


def _tune_reproducibility(
    records: list[Record], dev_records: list[Record] | None
) -> dict[str, int] | None:
    if dev_records is None:
        return None
    dev = {r["key"]: r for r in dev_records if status_of(r) == "ok"}
    pairs = [
        (r, dev[r["key"]])
        for r in records
        if r["key"] in dev and status_of(r) == "ok" and unit_of(r)["role"] == "tune"
    ]
    return {
        "compared": len(pairs),
        "dataset_sha_equal": sum(
            a.get("dataset_sha256") == b.get("dataset_sha256") for a, b in pairs
        ),
        "declarations_equal": sum(
            [bool(e.get("declared")) for e in a["edges"]]
            == [bool(e.get("declared")) for e in b["edges"]]
            for a, b in pairs
        ),
    }


def _missing_units(want: dict | None, records: list[Record], screened: dict[str, Any]) -> list[str]:
    if want is None:
        return [f"expected units unknown: {screened['expected_error']}"]
    return sorted(set(want) - {r["key"] for r in records})


def _check_flags(facts: dict[str, Any]) -> dict[str, bool]:
    screened = facts["screened"]
    checks = {
        "protocol_frozen_sha": facts["protocol"]["ok"],
        "freeze_commit_given": facts["freeze_commit"] is not None,
        "commits": not facts["code"]["bad_commit"],
        "clean": not facts["code"]["dirty"],
        "stamps": not facts["bad_stamps"],
        "missing": not facts["missing"],
        "unexpected": not screened["unexpected"],
        "role": not screened["role_mismatch"],
        "duplicates": not screened["duplicates"],
        "candidates_complete": not facts["incomplete"],
        "errors_listed": not facts["errors_unlisted"],
        "dataset_hash": not facts["datasets"]["mismatch"],
        "dataset_hash_stamped": not facts["datasets"]["unstamped"],
        "one_platform_per_dataset": not facts["datasets"]["multi_platform"],
        "pkgs_uniform": facts["pkgs_ok"],
        "fmax_dependence": facts["dep_ok"],
    }
    if facts["grid"]["has_rules"]:
        checks["grid_gaps_ruled"] = not facts["grid"]["unexplained"]
    return checks


def _report(facts: dict[str, Any], checks: dict[str, bool], records: list[Record]) -> Integrity:
    screened, code, datasets = facts["screened"], facts["code"], facts["datasets"]
    roles = Counter(f"{unit_of(r)['role']}:{status_of(r)}" for r in records)
    want = screened["expected"]
    return {
        "label": "FINAL" if all(checks.values()) else "PROVISIONAL",
        "checks": checks,
        "failed": sorted(k for k, v in checks.items() if not v),
        "freeze_commit": facts["freeze_commit"],
        "commits": code["commits"],
        "commit_violations": code["bad_commit"][:LIST_LIMIT],
        "n_commit_violations": len(code["bad_commit"]),
        "amendments": [a.get("id") for a in facts["amendments"]["amendments"]],
        "not_clean": code["dirty"][:LIST_LIMIT],
        "n_not_clean": len(code["dirty"]),
        "stamp_violations": facts["bad_stamps"][:LIST_LIMIT],
        "n_stamp_violations": len(facts["bad_stamps"]),
        "protocol": facts["protocol"],
        "spec_sha256": facts["spec_sha"],
        "n_records_used": len(records),
        "n_expected": None if want is None else len(want),
        "missing": facts["missing"][:LONG_LIST_LIMIT],
        "n_missing": len(facts["missing"]),
        "unexpected": screened["unexpected"][:LIST_LIMIT],
        "n_unexpected": len(screened["unexpected"]),
        "role_mismatch": screened["role_mismatch"][:LIST_LIMIT],
        "n_role_mismatch": len(screened["role_mismatch"]),
        "duplicates": screened["duplicates"][:LIST_LIMIT],
        "n_duplicates": len(screened["duplicates"]),
        "candidates_incomplete": facts["incomplete"][:LIST_LIMIT],
        "n_candidates_incomplete": len(facts["incomplete"]),
        "by_role_status": dict(sorted(roles.items())),
        "errors": facts["errors"][:LONG_LIST_LIMIT],
        "errors_not_listed_persistent": facts["errors_unlisted"][:LONG_LIST_LIMIT],
        "infeasible": sorted(r["key"] for r in records if status_of(r) == "infeasible")[
            :LONG_LIST_LIMIT
        ],
        "dataset_hash_mismatch": datasets["mismatch"][:LIST_LIMIT],
        "n_dataset_hash_mismatch": len(datasets["mismatch"]),
        "dataset_hash_unstamped": sorted(datasets["unstamped"])[:LIST_LIMIT],
        "n_dataset_hash_unstamped": len(datasets["unstamped"]),
        "n_dataset_hash_unstamped_not_ok": len(datasets["unstamped_not_ok"]),
        "multi_platform_datasets": datasets["multi_platform"][:LIST_LIMIT],
        "not_in_grid_cells": facts["grid"]["n_cells"],
        "not_in_grid_unexplained": facts["grid"]["unexplained"][:LIST_LIMIT],
        "pkgs_sets": facts["pkgs_sets"],
        "fmax_dependence_sha256": (facts["dep"] or {}).get("_sha"),
        "by_recheck": facts["by_recheck"],
        "tune_reproducibility_vs_dev": facts["repro"],
    }


def integrity(
    records: list[Record], screened: dict[str, Any], spec: Spec, inputs: IntegrityInputs
) -> Integrity:
    """Runs the V11 integrity checks.

    Args:
        records: Screened records.
        screened: Screening report from `cells.screen`.
        spec: Campaign spec.
        inputs: Freeze commit, stamps, amendments and the other external facts.

    Returns:
        The label (FINAL only if every check passes), the checks and the evidence behind each.
    """
    amendments = inputs.amendments or NO_AMENDMENTS
    persistent = {p["key"] for p in amendments["persistent_errors"]}
    errors = sorted(r["key"] for r in records if status_of(r) == "error")
    pkgs_ok, pkgs_sets = _packages_uniform(records, spec)
    dep = inputs.dep
    facts = {
        "screened": screened,
        "freeze_commit": inputs.freeze_commit,
        "spec_sha": inputs.spec_sha,
        "protocol": inputs.protocol or {"ok": False, "why": "not checked"},
        "amendments": amendments,
        "dep": dep,
        "code": _code_checks(records, inputs.freeze_commit, _allowed_commits(amendments)),
        "bad_stamps": _bad_stamps(records, spec, inputs.spec_sha),
        "missing": _missing_units(screened["expected"], records, screened),
        "incomplete": _incomplete_candidates(records),
        "errors": errors,
        "errors_unlisted": sorted(set(errors) - persistent),
        "datasets": _dataset_checks(records),
        "pkgs_ok": pkgs_ok,
        "pkgs_sets": pkgs_sets,
        "dep_ok": spec.get("fmax_dependence_sha256") is None
        or (dep or {}).get("_sha") == spec["fmax_dependence_sha256"],
        "by_recheck": by_recheck(records, spec),
        "repro": _tune_reproducibility(records, inputs.dev_records),
        "grid": not_in_grid(spec),
    }
    return _report(facts, _check_flags(facts), records)

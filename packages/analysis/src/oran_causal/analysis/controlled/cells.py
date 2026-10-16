"""Per-seed scoring and the arm-by-cell summary table.

A cell is one `(arm, world, regime, lam, n, kappa)` combination. Its measurement records are
scored seed by seed under the arm's declaration rule (BY flags for p arms, the conformal
placebo threshold for score-only arms), then summarised into rates with validity classes and
power. Truth enters only the scoring.
"""

import dataclasses
import functools
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from oran_causal.analysis.controlled.protocol import (
    ALPHA,
    CONFORMAL_LEVEL,
    COUNTED,
    LIKE_RAW_KEYS,
    MIN_FLAG_SEEDS,
    RATE_KEYS,
)
from oran_causal.analysis.controlled.spec import (
    Arm,
    Spec,
    analysis_block,
    arm_kind,
    cost_field,
    expected_units,
    planned_units,
    power_not_applicable,
)
from oran_causal.analysis.controlled.validity import combine, mean_ci, rate
from oran_causal.analysis.records import Record, Unit, cell_of, status_of, to_result, unit_of
from oran_causal.core.data import EdgeResult, Result, Truth
from oran_causal.core.scoring import (
    PLACEBO,
    PLACEBO_CONF,
    apply_threshold,
    placebo_tau,
    score,
)
from oran_causal.sim.controlled.datasets.truth import truth_for

Cell = dict[str, Any]
SeedRow = dict[str, Any]
Cells = dict[str, Cell]
Rows = dict[str, list[SeedRow]]


def is_primary(source: str) -> bool:
    """Tells whether a candidate source is a real action (not a lagged KPI, not a placebo)."""
    return not source.startswith("K") and source not in (PLACEBO, PLACEBO_CONF)


def conformal_tau(results: Iterable[Result], alpha: float = CONFORMAL_LEVEL) -> float:
    """Returns the conformal placebo cutoff of the tune results (`core.scoring.placebo_tau`)."""
    return float(placebo_tau(list(results), alpha=alpha))


def raw_declare(result: Result) -> Result:
    """Copies a result with `declared = p <= .05` per edge; no p means not declared."""
    edges = tuple(
        EdgeResult(
            e.source, e.target, e.score, e.p, e.sign, e.p is not None and float(e.p) <= ALPHA
        )
        for e in result.edges
    )
    return Result(
        method=result.method,
        version=result.version,
        edges=edges,
        cpu_s=result.cpu_s,
        config={**result.config, "declare": "raw_p"},
        notes=result.notes,
    )


def fixed_declare(result: Result, threshold: float) -> Result:
    """Copies a result with `declared = score >= threshold`, the arm's own untuned threshold."""
    edges = tuple(
        EdgeResult(
            e.source,
            e.target,
            e.score,
            e.p,
            e.sign,
            e.score is not None and math.isfinite(e.score) and float(e.score) >= threshold,
        )
        for e in result.edges
    )
    return Result(
        method=result.method,
        version=result.version,
        edges=edges,
        cpu_s=result.cpu_s,
        config={**result.config, "declare": "fixed", "fixed_threshold": threshold},
        notes=result.notes,
    )


def _rate_counts(record: Record, scored: dict[str, Any], truth: Truth) -> dict[str, list[int]]:
    not_testable = set(scored["not_testable_edges"])
    declared = set(scored["declared_edges"])
    nulls = {f"{s}->{t}" for s, t in truth.null_edges if is_primary(s)}
    counts = {k: [0, 0] for k in RATE_KEYS}
    for edge in record.get("edges", []):
        name = f"{edge['source']}->{edge['target']}"
        if edge["source"] == PLACEBO:
            kind = "plac"
        elif edge["source"] == PLACEBO_CONF:
            kind = "conf"
        else:
            kind = "null" if name in nulls else None
        if kind is None or name in not_testable:
            continue
        counts[f"{kind}_decl"][1] += 1
        counts[f"{kind}_decl"][0] += int(name in declared)
        p = edge.get("p")
        if p is not None:
            counts[f"{kind}_raw"][1] += 1
            counts[f"{kind}_raw"][0] += int(float(p) <= ALPHA)
    return counts


def _sign_counts(record: Record, scored: dict[str, Any], truth: Truth) -> tuple[int, int, int]:
    declared = set(scored["declared_edges"])
    by_edge = {(e["source"], e["target"]): e for e in record.get("edges", [])}
    signed = [(k, v) for k, v in truth.signs.items() if k in truth.edges and is_primary(k[0])]

    def read_sign(edge: tuple[str, str]) -> int:
        return int(np.sign((by_edge.get(edge) or {}).get("sign") or 0))

    ok = sum(1 for k, v in signed if f"{k[0]}->{k[1]}" in declared and read_sign(k) == v)
    bad = sum(1 for k, v in signed if f"{k[0]}->{k[1]}" in declared and read_sign(k) not in (0, v))
    return ok, bad, len(signed)


def seed_row(record: Record, declared: Result, truth: Truth) -> SeedRow:
    """Scores one measurement record under the declaration `declared`.

    Args:
        record: Measurement record.
        declared: Its result with the declarations to score.
        truth: Ground truth of the record's world and regime.

    Returns:
        Recall, FDP, hit counts per rate key, sign counts, not-testable counts and the
        secondary KPI-to-KPI family.
    """
    scored = score(declared, truth)
    sign_ok, sign_bad, n_signed = _sign_counts(record, scored, truth)
    not_applicable = (record.get("notes") or {}).get("not_applicable") or {}
    secondary = scored["secondary"]
    return {
        "seed": int(record["job"]["seed"]),
        "world": record["job"]["world"],
        "recall": scored["recall"],
        "fdp": scored["fdp"],
        "n_declared": scored["n_declared"],
        "tp": scored["tp"],
        "n_true": scored["n_true"],
        "cnt": _rate_counts(record, scored, truth),
        "sign_ok": sign_ok,
        "sign_bad": sign_bad,
        "n_signed_true": n_signed,
        "nt_true": scored["n_not_testable_true"],
        "nt_null": scored["n_not_testable_null"],
        "n_not_applicable": sum(1 for k in not_applicable if is_primary(str(k).partition("->")[0])),
        "sec": {
            "recall": secondary["recall"],
            "fdp": secondary["fdp"],
            "fp": secondary["fp"],
            "n_null_testable": secondary["n_null"] - secondary["n_not_testable_null"],
            "n_true": secondary["n_true"],
        },
    }


def validity_keys(*, has_p: bool) -> tuple[str, ...]:
    """Returns the rates that decide a cell's validity.

    p arms: raw p and BY declarations. Score-only arms: the truth-null and confounded-placebo
    declarations (`P_placebo` is their tuning column).
    """
    if has_p:
        return ("null_raw", "plac_raw", "conf_raw", "null_decl", "plac_decl", "conf_decl")
    return ("null_decl", "conf_decl")


def _sign_summary(rows: list[SeedRow]) -> tuple[dict[str, Any] | None, float | None]:
    sign_acc = rate([x["sign_ok"] for x in rows], [x["sign_ok"] + x["sign_bad"] for x in rows])
    if sign_acc:
        sign_acc.pop("validity")
    n_signed = sum(x["n_signed_true"] for x in rows)
    wrong = sum(x["sign_bad"] for x in rows) / n_signed if n_signed else None
    return sign_acc, wrong


def summarise(
    rows: list[SeedRow],
    *,
    has_p: bool,
    vkeys: Sequence[str] | None = None,
    power_na: bool = False,
    min_flag_seeds: int = MIN_FLAG_SEEDS,
) -> dict[str, Any]:
    """Summarises per-seed rows into rates, validity and power.

    Args:
        rows: Per-seed rows of one cell.
        has_p: Whether the arm has raw p-values.
        vkeys: Rates that decide validity; defaults to `validity_keys`.
        power_na: Power is not applicable (PMRT where the design is unknown): recall and sign
            are None, never 0.
        min_flag_seeds: Fewer seeds make the validity `few_seeds`.

    Returns:
        The cell summary.
    """
    rows = sorted(rows, key=lambda x: x["seed"])
    out: dict[str, Any] = {"n_seeds": len(rows)}
    for key in RATE_KEYS:
        if key.endswith("_raw") and not has_p:
            continue
        out[key] = rate([x["cnt"][key][0] for x in rows], [x["cnt"][key][1] for x in rows])
    if len(rows) < min_flag_seeds:
        out["validity"] = "few_seeds"
    else:
        keys = vkeys or validity_keys(has_p=has_p)
        out["validity"] = combine((out.get(k) or {}).get("validity") for k in keys)
    out["recall"] = mean_ci([x["recall"] for x in rows])
    out["fdp"] = mean_ci([x["fdp"] for x in rows])
    out["sign_acc"], out["wrong_sign_rate"] = _sign_summary(rows)
    out["n_declared"] = float(np.mean([x["n_declared"] for x in rows])) if rows else None
    out.update(
        nt_true=int(sum(x["nt_true"] for x in rows)),
        nt_null=int(sum(x["nt_null"] for x in rows)),
        n_not_applicable=int(sum(x["n_not_applicable"] for x in rows)),
    )
    out["secondary"] = {
        "recall": mean_ci([x["sec"]["recall"] for x in rows]),
        "fdp": mean_ci([x["sec"]["fdp"] for x in rows]),
        "null_decl": rate(
            [x["sec"]["fp"] for x in rows], [x["sec"]["n_null_testable"] for x in rows]
        ),
    }
    out["per_seed_recall"] = {str(x["seed"]): x["recall"] for x in rows}
    out["power_not_applicable"] = power_na
    if power_na:
        out.update(recall=None, sign_acc=None, wrong_sign_rate=None, per_seed_recall={})
    return out


def screen(records: list[Record], spec: Spec) -> tuple[list[Record], dict[str, Any]]:
    """Selects the records that enter the tables: one per expected key with the expected role.

    Others are listed, never used. Without a known unit list (placeholder seeds) every record of
    a spec arm is used and the output is provisional.

    Args:
        records: Campaign records.
        spec: Campaign spec.

    Returns:
        The records to use and the screening report (`expected`, `expected_error`,
        `duplicates`, `unexpected`, `role_mismatch`).
    """
    try:
        want, why = expected_units(spec), None
    except ValueError as ex:
        want, why = None, str(ex)
    keys = Counter(r.get("key", "") for r in records)
    duplicates = sorted(k for k, v in keys.items() if v > 1)
    seen: set[str | None] = set()
    use, unexpected, role_bad = [], [], []
    for record in records:
        key = record.get("key")
        if key in seen:
            continue
        seen.add(key)
        if want is None:
            if unit_of(record)["arm"] in spec["arms"]:
                use.append(record)
        elif key not in want:
            unexpected.append(key)
        elif unit_of(record)["role"] != want[key]["role"]:
            role_bad.append(key)
        else:
            use.append(record)
    report = {
        "expected": want,
        "expected_error": why,
        "duplicates": duplicates,
        "unexpected": sorted(unexpected),
        "role_mismatch": sorted(role_bad),
    }
    return use, report


def _grid_cells(spec: Spec) -> Cells:
    try:
        units = planned_units(spec)
    except ValueError:
        return {}
    out: Cells = {}
    for u in units.values():
        if u["role"] != "measure":
            continue
        key = f"{u['arm']}|{cell_of(u['world'], u['regime'], u['lam'], u['n'], u['kappa'])}"
        entry = out.setdefault(
            key,
            {
                **{f: u[f] for f in ("arm", "world", "regime", "lam", "n", "kappa")},
                "planned_seeds": 0,
                "t3": u["t3"],
            },
        )
        entry["planned_seeds"] += 1
    return out


@dataclasses.dataclass
class _Group:
    tune: list[Record] = dataclasses.field(default_factory=list)
    measure: list[Record] = dataclasses.field(default_factory=list)
    infeasible: list[Record] = dataclasses.field(default_factory=list)
    error: list[Record] = dataclasses.field(default_factory=list)

    def add(self, bucket: str, record: Record) -> None:
        getattr(self, bucket).append(record)


@dataclasses.dataclass
class _SeedRows:
    by: list[SeedRow] = dataclasses.field(default_factory=list)
    tau: list[SeedRow] = dataclasses.field(default_factory=list)
    raw: list[SeedRow] = dataclasses.field(default_factory=list)
    fixed: list[SeedRow] = dataclasses.field(default_factory=list)


def _group_records(records: list[Record]) -> tuple[dict[str, _Group], dict[str, Unit]]:
    groups: dict[str, _Group] = defaultdict(_Group)
    meta: dict[str, Unit] = {}
    for record in records:
        u = unit_of(record)
        key = f"{u['arm']}|{cell_of(u['world'], u['regime'], u['lam'], u['n'], u['kappa'])}"
        meta[key] = u
        status = status_of(record)
        groups[key].add(u["role"] if status == "ok" else status, record)
    return groups, meta


def _seed_rows(arm: Arm, group: _Group, truth: Truth, tau: float | None) -> _SeedRows:
    rows = _SeedRows()
    threshold = arm.get("fixed_threshold")
    for record in group.measure:
        result = to_result(record)
        if arm["declare"] == "by":
            rows.by.append(seed_row(record, result, truth))
            rows.raw.append(seed_row(record, raw_declare(result), truth))
        if threshold is not None:
            rows.fixed.append(seed_row(record, fixed_declare(result, float(threshold)), truth))
        if tau is not None:
            rows.tau.append(seed_row(record, apply_threshold(result, tau), truth))
    return rows


def _cell_status(
    group: _Group, n_primary: int, plan: Cell | None, *, untuned: bool, min_flag_seeds: int
) -> str:
    planned = (plan or {}).get("planned_seeds")
    checks = (
        (bool((plan or {}).get("t3")), "infeasible_t3"),
        (bool(group.infeasible), "infeasible"),
        (not group.measure, "missing"),
        (untuned, "untuned"),
        (n_primary < min_flag_seeds, "few_seeds"),
        (planned is not None and n_primary < planned, "under_seeded"),
    )
    return next((status for hit, status in checks if hit), "ok")


def _infeasible_cost(arm: Arm, group: _Group, plan: Cell | None, unit: Unit) -> Cell:
    field = cost_field(arm)
    if (plan or {}).get("t3"):
        return {f"infeasible_cost_{field}": (arm.get(f"t3_cost_{field}") or {}).get(str(unit["n"]))}
    if group.infeasible:
        cost: list[float] = [x[field] for x in group.infeasible if x.get(field) is not None]
        return {f"infeasible_cost_{field}": max(cost) if cost else None}
    return {}


def _cell_entry(  # noqa: PLR0913
    spec: Spec,
    unit: Unit,
    group: _Group,
    *,
    plan: Cell | None,
    truth: Truth,
    min_flag_seeds: int,
) -> tuple[Cell, list[SeedRow]]:
    arm = spec["arms"][unit["arm"]]
    mode = arm["declare"]
    tau = conformal_tau([to_result(r) for r in group.tune]) if group.tune else None
    rows = _seed_rows(arm, group, truth, tau)
    primary = rows.by if mode == "by" else rows.tau
    entry: Cell = {
        "arm": unit["arm"],
        "kind": arm_kind(arm),
        "block": analysis_block(unit["arm"], spec),
        "declare": mode,
        "world": unit["world"],
        "regime": unit["regime"],
        "lam": unit["lam"],
        "n": unit["n"],
        "kappa": unit["kappa"],
        "tau": tau if tau is None or math.isfinite(tau) else None,
        "tau_is_pos_inf": tau == math.inf,
        "n_tune": len(group.tune),
        "n_measure": len(group.measure),
        "planned_seeds": (plan or {}).get("planned_seeds"),
        "n_infeasible": len(group.infeasible),
        "n_error": len(group.error),
    }
    entry["status"] = _cell_status(
        group,
        len(primary),
        plan,
        untuned=mode == "tau" and tau is None,
        min_flag_seeds=min_flag_seeds,
    )
    entry.update(_infeasible_cost(arm, group, plan, unit))
    na = power_not_applicable(arm, unit["regime"])
    entry["power_not_applicable"] = na
    summarise_here = functools.partial(summarise, power_na=na, min_flag_seeds=min_flag_seeds)
    entry["primary"] = summarise_here(primary, has_p=mode == "by") if primary else None
    if mode == "by" and rows.tau:
        entry["secondary_tau"] = summarise_here(rows.tau, has_p=False)
    if rows.raw:
        entry["like_raw"] = summarise_here(rows.raw, has_p=False, vkeys=LIKE_RAW_KEYS)
    if rows.fixed:
        entry["like_fixed"] = summarise_here(rows.fixed, has_p=False, vkeys=LIKE_RAW_KEYS)
    return entry, primary


def build_cells(
    records: list[Record], spec: Spec, *, min_flag_seeds: int = MIN_FLAG_SEEDS
) -> tuple[Cells, Rows]:
    """Builds the arm-by-cell summaries from screened records.

    Planned cells without records get status `missing`; T3 cells get `infeasible_t3` and their
    DEV cost.

    Args:
        records: Screened records (`screen`).
        spec: Campaign spec.
        min_flag_seeds: Fewer measurement seeds make a cell `few_seeds`.

    Returns:
        `cells[arm|cell]` summaries and `rows[arm|cell]` per-seed rows of the primary
        declaration.
    """
    groups, meta = _group_records(records)
    plan = _grid_cells(spec)
    truths: dict[tuple[str, str], Truth] = {}
    cells: Cells = {}
    rows_out: Rows = {}
    for key in sorted(set(groups) | set(plan)):
        unit = plan.get(key) or meta[key]
        if unit["arm"] not in spec["arms"]:
            continue
        group = groups[key] if key in groups else _Group()
        truth = truths.setdefault(
            (unit["world"], unit["regime"]), truth_for(unit["world"], unit["regime"])
        )
        cells[key], rows_out[key] = _cell_entry(
            spec, unit, group, plan=plan.get(key), truth=truth, min_flag_seeds=min_flag_seeds
        )
    return cells, rows_out


def counts(cell: Cell) -> bool:
    """Tells whether an arm-cell enters verdicts: a counted status and a validity read."""
    return (
        cell["status"] in COUNTED
        and bool(cell.get("primary"))
        and cell["primary"]["validity"] != "few_seeds"
    )


def valid(cell: Cell) -> bool:
    """Tells whether an arm-cell is counted and VALID."""
    return counts(cell) and cell["primary"]["validity"] == "VALID"


def powered(cell: Cell) -> bool:
    """Tells whether an arm-cell enters the power tables: counted and not INVALID."""
    return counts(cell) and cell["primary"]["validity"] != "INVALID"


def cell_class(cell: Cell, keys: Iterable[str]) -> str:
    """Returns the validity of a counted cell restricted to a component's rate keys."""
    primary = cell["primary"]
    return combine((primary.get(k) or {}).get("validity") for k in keys)

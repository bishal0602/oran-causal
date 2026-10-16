"""What a campaign spec plans: its units, its nominal grid and the roles of its arms.

A spec is the JSON object of a campaign (`arms`, `blocks`, optional `focal`, `not_in_grid`,
`protocol_sha256`). It is read as data; no method is imported.
"""

import json
from collections.abc import Iterator
from typing import Any

from oran_causal.analysis.controlled.protocol import (
    DEFAULT_E4_LAMS,
    FOCAL,
    NO_DESIGN_REGIMES,
)
from oran_causal.analysis.records import Unit, cell_of, job_key
from oran_causal.sim.controlled.datasets.truth import REGIMES_OF

Spec = dict[str, Any]
Arm = dict[str, Any]
CELL_FIELDS = ("arm", "world", "regime", "lam", "n", "kappa")
PLACEHOLDER_SEED = 3_100_000


def seeds_of(seeds: Any) -> list[int]:
    """Expands a block's seeds: a placeholder string is refused, `[lo, hi]` with a gap is a range.

    Raises:
        ValueError: If the seeds are still a placeholder string.
    """
    if isinstance(seeds, str):
        raise ValueError(
            f"spec seeds are a placeholder ({seeds!r}): fill them at the freeze (PROTOCOL_A T1)"
        )
    if len(seeds) == 2 and seeds[1] > seeds[0] + 1:
        return list(range(int(seeds[0]), int(seeds[1]) + 1))
    return [int(x) for x in seeds]


def _block_arms(spec: Spec, block: Spec) -> list[str]:
    return list(spec["arms"]) if block.get("arms", "all") == "all" else list(block["arms"])


def _block_cells(block: Spec) -> Iterator[tuple[str, str, float | None, float, int]]:
    lam_map = {**DEFAULT_E4_LAMS, **block.get("e4_lams", {})}
    for world in block["worlds"]:
        for regime in block["regimes"]:
            if regime not in REGIMES_OF[world]:
                continue
            lams = [float(x) for x in lam_map[regime]] if world == "E4" else [None]
            for lam in lams:
                for kappa in block["kappas"]:
                    for n in block["ns"]:
                        yield world, regime, lam, float(kappa), int(n)


def _arm_has_world(arm: Arm, world: str) -> bool:
    return "worlds" not in arm or world in arm["worlds"]


def planned_units(spec: Spec) -> dict[str, Unit]:
    """Lists every unit of a spec, including those above an arm's T3 `max_n` (flag `t3`).

    Mirrors the campaign expansion, which drops the T3 units, without its seed guard.

    Args:
        spec: Campaign spec.

    Returns:
        Unit key to unit fields.
    """
    arms = spec["arms"]
    out: Unit = {}
    for block in spec["blocks"]:
        names = _block_arms(spec, block)
        for world, regime, lam, kappa, n in _block_cells(block):
            for seed in seeds_of(block["seeds"]):
                for name in names:
                    arm = arms[name]
                    if not _arm_has_world(arm, world):
                        continue
                    key = job_key(name, world, regime, lam, n, int(seed), kappa)
                    out.setdefault(
                        key,
                        {
                            "arm": name,
                            "world": world,
                            "regime": regime,
                            "lam": lam,
                            "n": n,
                            "kappa": kappa,
                            "seed": int(seed),
                            "role": block["role"],
                            "t3": "max_n" in arm and n > int(arm["max_n"]),
                        },
                    )
    return out


def expected_units(spec: Spec) -> dict[str, Unit]:
    """Lists the units the campaign runs: `planned_units` without the T3-infeasible ones."""
    return {k: u for k, u in planned_units(spec).items() if not u["t3"]}


def _measure_cells(spec: Spec, *, every_arm: bool) -> dict[str, Unit]:
    spec = json.loads(json.dumps(spec))
    for block in spec["blocks"]:
        if isinstance(block["seeds"], str):
            block["seeds"] = [PLACEHOLDER_SEED]
        if every_arm:
            block["arms"] = "all"
    out: dict[str, Unit] = {}
    for unit in planned_units(spec).values():
        if unit["role"] == "measure":
            cell = cell_of(unit["world"], unit["regime"], unit["lam"], unit["n"], unit["kappa"])
            out[f"{unit['arm']}|{cell}"] = {f: unit[f] for f in CELL_FIELDS}
    return out


def _rule_hits(rule: Spec, cell: Unit) -> bool:
    return (
        cell["arm"] in rule["arms"]
        and (rule.get("regimes") is None or cell["regime"] in rule["regimes"])
        and (rule.get("ns") is None or cell["n"] in rule["ns"])
        and (rule.get("kappas") is None or cell["kappa"] in rule["kappas"])
    )


def not_in_grid(spec: Spec) -> dict[str, Any]:
    """Finds the cells of the nominal grid that an arm does not have.

    The nominal grid holds every arm in every measurement block. Gaps are pre-registered grid
    choices (spec `not_in_grid` rules): they never count as missing and never cap a verdict.
    A gap no rule explains is listed as unexplained.

    Args:
        spec: Campaign spec.

    Returns:
        The gap cells with the rule that explains each, per-rule counts and unexplained gaps.
    """
    have = _measure_cells(spec, every_arm=False)
    rules = spec.get("not_in_grid") or []
    cells, unexplained = [], []
    for key, cell in sorted(_measure_cells(spec, every_arm=True).items()):
        if key in have:
            continue
        index = next((i for i, rule in enumerate(rules) if _rule_hits(rule, cell)), None)
        if index is None:
            unexplained.append(key)
        ruling = None if index is None else rules[index]["ruling"]
        cells.append({**cell, "key": key, "rule": index, "ruling": ruling})
    by_rule = [
        {
            "ruling": rule["ruling"],
            "reason": rule.get("reason"),
            "arms": rule["arms"],
            "where": {f: rule.get(f) for f in ("regimes", "ns", "kappas")},
            "n_cells": sum(c["rule"] == i for c in cells),
        }
        for i, rule in enumerate(rules)
    ]
    return {
        "cells": cells,
        "n_cells": len(cells),
        "by_rule": by_rule,
        "unexplained": unexplained,
        "has_rules": bool(rules),
    }


def arm_kind(arm: Arm) -> str:
    """Returns `pmrt`, `eq`, `eq_min` or `native` for a spec arm."""
    if "pmrt_core" in arm["ref"]:
        return "pmrt"
    config_arm = (arm.get("config") or {}).get("arm")
    return config_arm if config_arm in ("eq", "eq_min") else "native"


def focal_arm(spec: Spec) -> str:
    """Returns the primary PMRT arm: the spec's `focal`, else `pmrt_eq`."""
    return spec.get("focal") or FOCAL


def power_not_applicable(arm: Arm, regime: str) -> bool:
    """Tells whether an arm's power is not applicable in a regime.

    A PMRT arm where the real actions have no known design tests no true edge, so its power is
    NA (never 0).
    """
    return arm_kind(arm) == "pmrt" and regime in NO_DESIGN_REGIMES


def cost_field(arm: Arm) -> str:
    """Returns the record field of an arm's budget: `wall_s` for a GPU arm, else `cpu_s`."""
    return "wall_s" if arm.get("budget_wall_s") else "cpu_s"


def native_partner(name: str, spec: Spec) -> str | None:
    """Returns the native arm that an `eq` or `eq_min` arm is paired with.

    The spec's `native_partner`, else the native arm that shares the adapter ref.
    """
    arm = spec["arms"][name]
    if arm.get("native_partner"):
        return arm["native_partner"]
    return next(
        (
            other
            for other, d in sorted(spec["arms"].items())
            if d["ref"] == arm["ref"] and arm_kind(d) == "native"
        ),
        None,
    )


def analysis_block(name: str, spec: Spec) -> str:
    """Returns `primary` or `secondary`: the spec's `analysis`, else by arm kind."""
    arm = spec["arms"][name]
    if arm.get("analysis"):
        return arm["analysis"]
    return "primary" if name == focal_arm(spec) or arm_kind(arm) == "eq" else "secondary"


def arm_label(name: str, spec: Spec) -> str | None:
    """Returns the reporting label a ruling attaches to an arm."""
    return spec["arms"][name].get("label")

"""Partition of units into parts, and the relaunch rule for datasets completed elsewhere."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.experiments.campaign.integrity import record_status
from oran_causal.experiments.campaign.merge import iter_jsonl
from oran_causal.experiments.campaign.spec import DatasetKey, Unit

type CostTable = Mapping[str, float]


def unit_cost(u: Unit, cost_table: CostTable | None) -> float:
    """Estimated CPU-s of a unit: the table's mean for (arm, world, regime, n), else n / 1000."""
    if cost_table:
        c = cost_table.get(f"{u.arm}|{u.world}|{u.regime}|n{u.n}")
        if c is not None:
            return float(c)
    return u.n / 1000.0


def _groups(units: Iterable[Unit]) -> dict[DatasetKey, list[Unit]]:
    groups: dict[DatasetKey, list[Unit]] = defaultdict(list)
    for u in units:
        groups[u.dataset].append(u)
    return groups


def _part_order(u: Unit) -> tuple[int, str, str]:
    return (u.n, str(u.dataset), u.arm)


def partition(
    units: Sequence[Unit], parts: int, cost_table: CostTable | None = None
) -> list[list[Unit]]:
    """Whole dataset groups to parts by greedy longest-processing-time on the estimated cost.

    Deterministic. Within a part: ascending n, then dataset, so a session cut by its wall
    limit loses the largest units first.
    """
    groups = _groups(units)
    cost = {g: sum(unit_cost(u, cost_table) for u in us) for g, us in groups.items()}
    load = [0.0] * parts
    out: list[list[Unit]] = [[] for _ in range(parts)]
    for g in sorted(groups, key=lambda g: (-cost[g], str(g))):
        i = int(np.argmin(load))
        out[i].extend(groups[g])
        load[i] += cost[g]
    return [sorted(p, key=_part_order) for p in out]


def assignment(
    units: Sequence[Unit], parts: int, cost_table: CostTable | None = None
) -> dict[str, int]:
    """Part index of every unit key."""
    return {u.key: i for i, p in enumerate(partition(units, parts, cost_table)) for u in p}


def final_keys(
    paths: Iterable[Path], accept: Callable[[dict[str, Any]], bool] | None = None
) -> set[str]:
    """Keys with a final record (ok or infeasible) in `paths`, optionally filtered by `accept`."""
    return {
        r["key"]
        for p in paths
        for r in iter_jsonl(p)
        if r.get("key")
        and record_status(r) in ("ok", "infeasible")
        and (accept is None or accept(r))
    }


def complete_dataset_keys(
    units: Iterable[Unit],
    paths: Iterable[Path],
    accept: Callable[[dict[str, Any]], bool] | None = None,
) -> set[str]:
    """Keys of the units whose whole dataset (every arm of it in `units`) is final in `paths`.

    A relaunch skips those and re-runs every arm of the incomplete datasets, so each
    dataset's arms come from one platform.
    """
    fin = final_keys(paths, accept)
    groups = _groups(units)
    return {u.key for us in groups.values() if all(u.key in fin for u in us) for u in us}

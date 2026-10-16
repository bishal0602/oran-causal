"""Study 4 driver on the controlled environments: spec -> shards -> run -> merge -> integrity."""

import dataclasses
import json
from pathlib import Path
from typing import Any

from oran_causal.experiments.campaign.cost import cost_table_from_agg
from oran_causal.experiments.campaign.merge import merge
from oran_causal.experiments.campaign.run import RunOptions, Shard, run_part
from oran_causal.experiments.campaign.shard import CostTable, partition, unit_cost
from oran_causal.experiments.campaign.spec import ROLES, ProtocolTexts, expand, is_eval, sha_lf

INTEGRITY_LISTS = ("missing", "errors", "unexpected", "conflicts", "mixed_platform_datasets")

type Spec = dict[str, Any]


def load_spec(path: Path) -> tuple[Spec, str]:
    """The spec and the LF sha256 of its file."""
    data = path.read_bytes()
    return json.loads(data), sha_lf(data)


def load_cost_table(path: Path | None) -> CostTable | None:
    """A cost table file, or the cost table of an aggregate file."""
    if path is None:
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    return cost_table_from_agg(table) if "cost" in table else table


def plan(
    spec: Spec, parts: int, cost_table: CostTable | None = None, texts: ProtocolTexts | None = None
) -> dict[str, Any]:
    """Unit counts by role and arm and the estimated CPU-s of each part."""
    units = expand(spec, texts)
    shards = partition(units, parts, cost_table)
    return {
        "mode": "eval" if is_eval(spec) else "dev",
        "units": len(units),
        "datasets": len({u.dataset for u in units}),
        "by_role": {r: sum(u.role == r for u in units) for r in ROLES},
        "by_arm": {k: sum(u.arm == k for u in units) for k in spec["arms"]},
        "part_est_cpu_s": [round(sum(unit_cost(u, cost_table) for u in p)) for p in shards],
    }


def run_shard(spec_path: Path, shard: Shard, out: Path, opts: RunOptions | None = None) -> int:
    """Runs one part of a spec file; the spec file's sha is stamped on the records."""
    spec, file_sha = load_spec(spec_path)
    opts = dataclasses.replace(opts or RunOptions(), spec_file_sha256=file_sha)
    return run_part(spec, shard, out, opts)


def integrity_problems(summary: dict[str, Any]) -> list[str]:
    """Problems that keep a merged run from being final."""
    problems = [
        f"{len(summary[n])} {n}: {summary[n][:5]}" for n in INTEGRITY_LISTS if summary.get(n)
    ]
    if summary.get("bad_lines"):
        problems.append(f"{summary['bad_lines']} unparseable lines")
    if summary.get("dirty_records"):
        problems.append(f"{summary['dirty_records']} records from a dirty or unknown code tree")
    return problems


def run_study(
    spec_path: Path,
    workdir: Path,
    parts: int,
    cost_table: CostTable | None = None,
    opts: RunOptions | None = None,
) -> dict[str, Any]:
    """Runs every part of a spec in this process, merges the shards and reports integrity.

    Returns:
        The merge summary with an `integrity_problems` list.
    """
    opts = opts or RunOptions()
    shards = [workdir / f"res_{i}.jsonl" for i in range(parts)]
    for i, out in enumerate(shards):
        run_shard(spec_path, Shard(i, parts, cost_table), out, opts)
    spec, _ = load_spec(spec_path)
    existing = [p for p in shards if p.exists()]
    summary = merge(spec, existing, workdir / "merged.jsonl.gz", opts.providers.texts())
    return {**summary, "integrity_problems": integrity_problems(summary)}

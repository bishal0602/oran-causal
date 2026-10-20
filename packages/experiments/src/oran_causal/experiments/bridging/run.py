"""Bridging-study runs: the Study 4 method set on the frozen Study 1 (v4) data.

A unit is (arm, family, slice) of the frozen v4 data, keyed as a campaign unit
`<arm>|E6|<family>|k0|n<units>|s<dataset seed>` with role `measure` (evaluation slices) or
`placebo_log`; `n` is the dataset's unit count from the spec's counts file, so the campaign's
over-budget propagation is in rows. The campaign machinery (partition, isolated units under the
CPU budget, resumable shard files, merge rules) is reused unchanged; only the dataset source
differs (`Loader`, the runner's `Providers.dataset`).
"""

import dataclasses
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np

from oran_causal.analysis.bridging.plan import (
    WORLD,
    Slice,
    Spec,
    dataset_seed,
    load_gt,
    seed_index,
    spec_families,
    spec_slices,
)
from oran_causal.core.data import Dataset, Truth
from oran_causal.experiments.bridging import bridge
from oran_causal.experiments.campaign.integrity import MERGE_SETS, MergeTally, accept_rule
from oran_causal.experiments.campaign.merge import merge_index, read_entry, summary_path
from oran_causal.experiments.campaign.run import RunOptions, run_units
from oran_causal.experiments.campaign.shard import CostTable, partition
from oran_causal.experiments.campaign.spec import Unit
from oran_causal.experiments.paths import WORKSPACE_ROOT
from oran_causal.sim.system.decision.vocab import FAMILIES

COUNTS_SCHEMA = "xm-expb-counts/1"
ROLE_OF_KIND = {"placebo": "placebo_log"}
RESULTS = WORKSPACE_ROOT / "data" / "records" / "bridging" / "exp_b"
SUMMARY_EXTRA = ("mode", "platforms", "protocol_sha256", "spec_sha256", *MERGE_SETS)

type Counts = dict[str, dict[str, int]]


def repo_path(path: str) -> Path:
    """A path field of a spec, relative to the workspace root (absolute paths unchanged)."""
    return WORKSPACE_ROOT / path


def load_spec(path: str | Path) -> dict[str, Any]:
    """A spec, or a sub-spec `{"base": <spec path>, "slice_names" | "exclude_slices": [...]}`.

    A sub-spec's other fields override its base (same arms, so the same unit keys); the result
    carries `base_spec`.

    Raises:
        ValueError: An arm lacks `ref` or a `declare` in (by, tau), its name holds "|", or a
            family is unknown.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("base"):
        base = load_spec(repo_path(spec["base"]))
        spec = {**base, **{k: v for k, v in spec.items() if k != "base"}, "base_spec": spec["base"]}
    for a, d in spec["arms"].items():
        if "|" in a or not d.get("ref") or d.get("declare") not in ("by", "tau"):
            raise ValueError(f"arm {a!r}: needs 'ref', 'declare' in (by, tau), no '|'")
    bad = set(spec_families(spec)) - set(FAMILIES)
    if bad:
        raise ValueError(f"unknown families {bad}")
    return spec


def campaign_spec(spec: Spec) -> dict[str, Any]:
    """The dict the campaign runner reads: name, budget and arms; no blocks (never EVAL)."""
    return {
        "name": spec["name"],
        "budget_cpu_s": spec.get("budget_cpu_s"),
        "arms": spec["arms"],
        "blocks": [],
    }


def load_counts(path: str | Path) -> Counts:
    """{slice: {family: units}} of a counts file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["counts"]


def spec_counts(spec: Spec) -> Counts:
    """The counts of the spec's `counts` file."""
    return load_counts(repo_path(spec["counts"]))


def _arm_takes(d: Mapping[str, Any], s: Slice, family: str, n: int) -> bool:
    if "kinds" in d and s.kind not in d["kinds"]:
        return False
    if "families" in d and family not in d["families"]:
        return False
    return not ("max_n" in d and n > int(d["max_n"]))


def expand(spec: Spec, counts: Counts) -> list[Unit]:
    """Every unit of the spec, sorted, deduplicated; arms restricted by kinds, families, max_n."""
    out: dict[str, Unit] = {}
    for s in spec_slices(spec):
        for f in spec_families(spec):
            n = int(counts[s.name][f])
            if n == 0:
                continue
            role = ROLE_OF_KIND.get(s.kind, "measure")
            for a, d in spec["arms"].items():
                if _arm_takes(d, s, f, n):
                    u = Unit(WORLD, f, None, n, 0.0, dataset_seed(s.split, f), a, role)
                    out.setdefault(u.key, u)
    return sorted(out.values())


def table_path(tables: str | Path, stage: str) -> Path:
    """`<tables>/e6units_<stage>.npz`."""
    return Path(tables) / f"e6units_{stage}.npz"


def counts_from_tables(spec: Spec, tables: str | Path) -> Counts:
    """Unit counts per slice and family of the built unit tables."""
    tabs: dict[str, bridge.Table] = {}
    out = {}
    for s in spec_slices(spec):
        if s.stage not in tabs:
            tabs[s.stage] = bridge.Table.load(table_path(tables, s.stage))
        out[s.name] = {f: len(tabs[s.stage].rows(f, s.seeds)) for f in FAMILIES}
    return out


def counts_from_v4(spec: Spec, analysis: Mapping[str, Any]) -> Counts:
    """Unit counts as a stored v4 analysis reports them (same H 90 unit set); planning only.

    Raises:
        ValueError: The analysis lacks a slice of the spec.
    """
    out = {n: dict(v["units"]["by_family"]) for n, v in analysis["slices_result"].items()}
    pu = analysis.get("placebo_units") or {}
    if isinstance(pu, dict) and "by_family" in pu:
        out["placebo"] = dict(pu["by_family"])
    want = {s.name for s in spec_slices(spec)}
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"v4 analysis has no unit counts for {missing}")
    return {k: v for k, v in out.items() if k in want}


def check_counts(counts: Counts, ref: Counts) -> dict[str, list[Any]]:
    """{slice: [reference, found]} of every slice whose counts differ from `ref`."""
    return {k: [ref.get(k), v] for k, v in counts.items() if ref.get(k) != v}


@dataclasses.dataclass
class Loader:
    """Dataset source of a run: (world, regime, n, seed) -> (Dataset, Truth) of the v4 data.

    Attributes:
        index: (slice, family) per dataset seed.
        tables: Directory of the `e6units_<stage>.npz` tables.
        gt: The v4 GT cells.
    """

    index: dict[int, tuple[Slice, str]]
    tables: Path
    gt: dict[tuple[str, str, str], dict[str, Any]]
    _tabs: dict[str, bridge.Table] = dataclasses.field(default_factory=dict)

    @classmethod
    def of(cls, spec: Spec, tables: str | Path, gt: str | Path | None = None) -> Loader:
        """The loader of a spec (GT from `gt`, default the spec's `gt` path)."""
        return cls(seed_index(spec), Path(tables), load_gt(gt or repo_path(spec["gt"])))

    def table(self, stage: str) -> bridge.Table:
        """The unit table of a stage, loaded once."""
        if stage not in self._tabs:
            self._tabs[stage] = bridge.Table.load(table_path(self.tables, stage))
        return self._tabs[stage]

    def __call__(self, u: Unit) -> tuple[Dataset, Truth]:
        """The dataset and truth of a unit; its `lam` and `kappa` are unused.

        Raises:
            ValueError: The unit does not match its dataset seed, or the unit count differs.
        """
        s, f = self.index[int(u.seed)]
        if u.world != WORLD or u.regime != f:
            raise ValueError(
                f"unit ({u.world}, {u.regime}) does not match dataset seed {u.seed} ({f})"
            )
        ds = bridge.make_dataset(self.table(s.stage), f, s.seeds, s.split, s.name)
        if ds.n != int(u.n):
            raise ValueError(f"{s.name} {f}: {ds.n} units, the counts file says {u.n}")
        return ds, bridge.truth_from_gt(self.gt, f, placebo_log=s.kind == "placebo")


def with_loader(opts: RunOptions, loader: Loader) -> RunOptions:
    """`opts` with `loader` as the campaign runner's dataset source."""
    return dataclasses.replace(opts, providers=dataclasses.replace(opts.providers, dataset=loader))


def part_units(
    spec: Spec, part: int, parts: int, cost_table: CostTable | None = None
) -> list[Unit]:
    """The units of part `part` of `parts`."""
    return partition(expand(spec, spec_counts(spec)), parts, cost_table)[part]


def run(
    spec: Spec,
    tables: str | Path,
    out: str | Path,
    part: int = 0,
    parts: int = 1,
    cost_table: CostTable | None = None,
    opts: RunOptions | None = None,
) -> int:
    """Runs one part of the spec's units into the shard file `out` (resumable).

    The budget defaults to the spec's `budget_cpu_s`; records carry no part stamp, as the
    stored ones.

    Returns:
        The number of records written.
    """
    opts = opts or RunOptions()
    if opts.budget is None:
        opts = dataclasses.replace(opts, budget=spec.get("budget_cpu_s"))
    units = part_units(spec, part, parts, cost_table)
    return run_units(
        campaign_spec(spec), units, Path(out), with_loader(opts, Loader.of(spec, tables))
    )


def _open_out(out: Path) -> IO[str]:
    if out.suffix == ".gz":
        return gzip.open(out, "wt", encoding="utf-8", newline="\n")
    return out.open("w", encoding="utf-8", newline="\n")


def merge(
    spec: Spec, inputs: Sequence[str | Path], out: str | Path, newline: str = "\n"
) -> dict[str, Any]:
    """Merges shard files by the campaign rules; expected keys are the spec's units.

    Args:
        spec: The spec.
        inputs: Shard files.
        out: Merged JSONL (`.gz` writes gzip), key-sorted.
        newline: Line end of the summary file (the stored one was written on Windows, CRLF).

    Returns:
        The summary, also written to `<out without .gz>.summary.json`.
    """
    paths = [Path(p) for p in inputs]
    out = Path(out)
    best, stats = merge_index(paths, accept_rule(campaign_spec(spec)))
    want = {u.key for u in expand(spec, spec_counts(spec))}
    tally = MergeTally()
    fhs = [p.open("rb") for p in paths]
    try:
        with _open_out(out) as fo:
            for k in sorted(best):
                r = read_entry(fhs, best[k])
                fo.write(json.dumps(r, allow_nan=False) + "\n")
                tally.add(r)
    finally:
        for fh in fhs:
            fh.close()
    full = tally.summary(want, stats, "dev")
    summ = {k: v for k, v in full.items() if k not in SUMMARY_EXTRA}
    summary_path(out).write_text(json.dumps(summ, indent=1), encoding="utf-8", newline=newline)
    return summ


def _unit_cost(pts: Sequence[tuple[int, float]], n: int) -> float:
    x = np.log([p[0] for p in pts])
    y = np.log([max(p[1], 0.05) for p in pts])
    ln = math.log(n)
    if len(pts) == 1:
        return math.exp(y[0]) * n / pts[0][0]
    if ln <= x[-1]:
        return math.exp(np.interp(ln, x, y))
    slope = min(max((y[-1] - y[-2]) / (x[-1] - x[-2]), 0.0), 2.0)
    return math.exp(y[-1] + slope * (ln - x[-1]))


def project(spec: Spec, dev_costs: Mapping[str, Any]) -> dict[str, Any]:
    """Cost projection of the spec's units from Study 4 DEV costs.

    Per arm, the DEV CPU-s per dataset (max over worlds of the cell mean) interpolated log-log in
    n, extrapolated past the last point with the last segment's slope clipped to [0, 2], times
    the arm's `size_factor` (default 1).

    Returns:
        `{n_units, total_core_h, per_arm, projected_over_budget, cost_table}`; `cost_table` is a
        campaign cost table keyed `arm|E6|family|n<n>`.
    """
    units = expand(spec, spec_counts(spec))
    table: dict[str, float] = {}
    per_arm: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0.0])
    over = []
    budget = spec.get("budget_cpu_s")
    for u in units:
        d = spec["arms"][u.arm]
        dev = dev_costs[d.get("cost_from", u.arm)]
        pts = sorted((int(n), max(v[0] for v in w.values())) for n, w in dev.items())
        c = _unit_cost(pts, u.n) * float(d.get("size_factor", 1.0))
        table[f"{u.arm}|{u.world}|{u.regime}|n{u.n}"] = c
        pa = per_arm[u.arm]
        pa[0] += c
        pa[1] += 1
        pa[2] = max(pa[2], c)
        if budget and c > float(budget):
            over.append(u.key)
    return {
        "n_units": len(units),
        "total_core_h": sum(v[0] for v in per_arm.values()) / 3600.0,
        "per_arm": {
            a: {"core_h": v[0] / 3600.0, "units": v[1], "max_unit_s": v[2]}
            for a, v in per_arm.items()
        },
        "projected_over_budget": over,
        "cost_table": table,
    }

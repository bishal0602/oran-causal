"""Command line of the campaign runner: `python -m oran_causal.experiments.campaign <command>`.

`list` prints a spec's units and the estimated CPU-s per part, `run` runs one part into a
resumable shard file, `merge` merges shard files by the campaign rules. In an unpacked cloud
bundle the protocol texts come from the bundle (its protocol file and its frozen copy).
"""

import argparse
import dataclasses
import glob
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from oran_causal.experiments.campaign.cost import cost_table_from_agg
from oran_causal.experiments.campaign.merge import merge
from oran_causal.experiments.campaign.run import Providers, RunOptions, Shard, run_part
from oran_causal.experiments.campaign.shard import CostTable, partition, unit_cost
from oran_causal.experiments.campaign.spec import (
    ROLES,
    ProtocolTexts,
    expand,
    is_eval,
    sha_lf,
    shipped_protocol,
)
from oran_causal.experiments.launch.bundle import PROTOCOL_REL, bundled_texts

PROG = "python -m oran_causal.experiments.campaign"


def protocol_texts(root: Path) -> ProtocolTexts:
    """The bundle's protocol texts when `root` holds the protocol, else the shipped protocol."""
    return bundled_texts(root) if (root / PROTOCOL_REL).exists() else shipped_protocol()


def read_cost_table(path: Path | None) -> CostTable | None:
    """A cost table file; an aggregate (with `cost`) is converted to its cost table."""
    if path is None:
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    return cost_table_from_agg(table) if "cost" in table else table


def expand_globs(patterns: Iterable[str]) -> list[Path]:
    """The sorted files matching any of `patterns` (recursive `**`)."""
    return [Path(f) for f in sorted({f for p in patterns for f in glob.glob(p, recursive=True)})]


def parse_part(text: str) -> tuple[int, int]:
    """`"i/P"` as (i, P).

    Raises:
        argparse.ArgumentTypeError: Not two integers with 0 <= i < P.
    """
    try:
        i, p = (int(x) for x in text.split("/"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--part {text!r}: expected i/P") from e
    if not 0 <= i < p:
        raise argparse.ArgumentTypeError(f"--part {text!r}: needs 0 <= i < P")
    return i, p


def build_parser(prog: str = PROG) -> argparse.ArgumentParser:
    """Builds the argument parser."""
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="bundle or workspace root")
    sub = ap.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list", help="units, datasets and estimated CPU-s per part")
    ls.add_argument("--spec", required=True, type=Path)
    ls.add_argument("--parts", type=int, default=4)
    ls.add_argument("--cost-table", type=Path, default=None)

    run = sub.add_parser("run", help="run one part into a resumable shard file")
    run.add_argument("--spec", required=True, type=Path)
    run.add_argument("--part", type=parse_part, default=(0, 1), help="i/P")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--budget", type=float, default=None)
    run.add_argument("--cost-table", type=Path, default=None)
    run.add_argument("--no-isolate", action="store_true")
    run.add_argument("--registry", type=Path, default=None, help="shared DEV infeasibility file")
    run.add_argument(
        "--skip-complete-from",
        nargs="*",
        default=[],
        help="globs of an earlier session's shards: skip the datasets complete there",
    )

    mg = sub.add_parser("merge", help="merge shard files by the campaign rules")
    mg.add_argument("--spec", required=True, type=Path)
    mg.add_argument("--inputs", nargs="+", required=True, help="shard files or globs")
    mg.add_argument("--out", required=True, type=Path)
    return ap


def list_units(
    spec: dict[str, Any], parts: int, table: CostTable | None, texts: ProtocolTexts
) -> dict[str, Any]:
    """The `list` summary: mode, unit and dataset counts, units per role and arm, CPU-s per part."""
    units = expand(spec, texts)
    return {
        "mode": "eval" if is_eval(spec) else "dev",
        "units": len(units),
        "datasets": len({u.dataset for u in units}),
        "by_role": {r: sum(u.role == r for u in units) for r in ROLES},
        "by_arm": {k: sum(u.arm == k for u in units) for k in spec["arms"]},
        "part_est_cpu_s": [
            round(sum(unit_cost(u, table) for u in p)) for p in partition(units, parts, table)
        ],
    }


def _run(a: argparse.Namespace, spec_bytes: bytes, providers: Providers) -> int:
    spec = json.loads(spec_bytes)
    i, p = a.part
    shard = Shard(i, p, read_cost_table(a.cost_table), expand_globs(a.skip_complete_from))
    opts = RunOptions(
        budget=a.budget,
        isolate=False if a.no_isolate else None,
        spec_file_sha256=sha_lf(spec_bytes),
        registry=a.registry,
        providers=providers,
    )
    n = run_part(spec, shard, a.out, opts)
    providers.log(f"[xm-c] part {i}/{p}: {n} units run")
    return 0


def _summary_counts(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: len(v) if isinstance(v, list | dict) else v for k, v in summary.items()}


def main(
    argv: Sequence[str] | None = None,
    out: Callable[[str], None] = print,
    providers: Providers | None = None,
) -> int:
    """Runs one command.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.
        out: Output sink.
        providers: Environment queries of `run` (its protocol texts are replaced by the root's).

    Returns:
        The process exit code.
    """
    a = build_parser().parse_args(argv)
    texts = protocol_texts(a.root)
    spec_bytes = a.spec.read_bytes()
    match a.command:
        case "list":
            table = read_cost_table(a.cost_table)
            out(json.dumps(list_units(json.loads(spec_bytes), a.parts, table, texts)))
        case "run":
            base = providers or Providers(log=out)
            return _run(a, spec_bytes, dataclasses.replace(base, texts=lambda: texts))
        case _:
            summary = merge(json.loads(spec_bytes), expand_globs(a.inputs), a.out, texts)
            out(json.dumps(_summary_counts(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

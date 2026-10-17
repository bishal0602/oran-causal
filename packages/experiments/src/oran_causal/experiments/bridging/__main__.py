"""Command line of the bridging study: `python -m oran_causal.experiments.bridging <command>`.

`prep` builds the unit tables from the v4 collections, `counts` writes the unit counts, `list`
summarises a spec's units, `run` runs one part, `merge` merges shard files, `table` and `costs`
write the analysis documents, `project` projects the CPU cost from Study 4 DEV costs.
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.bridging import costs, tables
from oran_causal.experiments.bridging import bridge, run
from oran_causal.experiments.campaign.__main__ import expand_globs, parse_part
from oran_causal.experiments.campaign.run import Providers, RunOptions
from oran_causal.experiments.campaign.spec import sha_lf

PROG = "python -m oran_causal.experiments.bridging"
COUNTS_MISMATCH = 3

type Out = Callable[[str], None]


def build_parser(prog: str = PROG) -> argparse.ArgumentParser:
    """Builds the argument parser."""
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prep", help="build e6units_<stage>.npz from the v4 collections")
    p.add_argument("--eval", required=True, help="collection dirs or JSONL files, comma-separated")
    p.add_argument("--placebo", required=True)
    p.add_argument("--dev", default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("counts", help="unit counts per slice and family")
    p.add_argument("--spec", required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tables", type=Path, default=None)
    src.add_argument("--v4-analysis", dest="v4", type=Path, default=None)
    p.add_argument("--check", type=Path, default=None, help=f"exit {COUNTS_MISMATCH} on a mismatch")
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("list", help="units and datasets per arm")
    p.add_argument("--spec", required=True, type=Path)

    p = sub.add_parser("run", help="run one part into a resumable shard file")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--tables", required=True, type=Path)
    p.add_argument("--part", type=parse_part, default=(0, 1), help="i/P")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--cost-table", type=Path, default=None, help="a `project` output")
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--no-isolate", action="store_true")

    p = sub.add_parser("merge", help="merge shard files")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--inputs", nargs="+", required=True, help="shard files or globs")
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("table", help="exp_b_tables.json and EXP_B_TABLES.md")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--merged", required=True, type=Path)
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--v4-analysis", dest="v4", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("costs", help="exp_b_costs.json and EXP_B_COSTS.md")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--merged", required=True, type=Path)
    p.add_argument("--factors", required=True, type=Path, help="non-reference host factors")
    p.add_argument("--label", default=None, help="factors file name stored in the document")
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("project", help="cost projection from Study 4 DEV costs")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--dev-costs", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    return ap


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8", newline="\n")


def _prep(a: argparse.Namespace, out: Out) -> int:
    sources = {"eval": a.eval, "placebo": a.placebo} | ({"dev": a.dev} if a.dev else {})
    out(json.dumps(bridge.build_table(sources, a.out, a.workers), indent=1))
    return 0


def _counts(a: argparse.Namespace, spec: run.Spec, out: Out) -> int:
    if a.tables:
        counts, source = run.counts_from_tables(spec, a.tables), "tables"
    else:
        counts, source = run.counts_from_v4(spec, _read_json(a.v4)), "v4 analysis"
    doc: dict[str, Any] = {"schema": run.COUNTS_SCHEMA, "source": source, "counts": counts}
    if a.check:
        bad = run.check_counts(counts, run.load_counts(a.check))
        doc["check"] = {"against": str(a.check), "mismatches": bad}
    _write_json(a.out, doc)
    out(json.dumps(doc.get("check", {"n_slices": len(counts)})))
    return COUNTS_MISMATCH if doc.get("check", {}).get("mismatches") else 0


def _list(spec: run.Spec, out: Out) -> int:
    units = run.expand(spec, run.spec_counts(spec))
    summary = {
        "n_units": len(units),
        "n_datasets": len({u.dataset for u in units}),
        "per_arm": dict(Counter(u.arm for u in units)),
    }
    out(json.dumps(summary, indent=1))
    return 0


def _run(a: argparse.Namespace, spec: run.Spec, providers: Providers) -> int:
    i, p = a.part
    table = _read_json(a.cost_table)["cost_table"] if a.cost_table else None
    opts = RunOptions(
        budget=a.budget,
        isolate=False if a.no_isolate else None,
        spec_file_sha256=sha_lf(a.spec.read_bytes()),
        registry=a.registry,
        providers=providers,
    )
    n = run.run(spec, a.tables, a.out, i, p, table, opts)
    providers.log(f"[exp_b] part {i}/{p}: {n} units run")
    return 0


def _merge(a: argparse.Namespace, spec: run.Spec, out: Out) -> int:
    summary = run.merge(spec, expand_globs(a.inputs), a.out)
    short = {k: v for k, v in summary.items() if not isinstance(v, list) or len(v) < 20}
    out(json.dumps(short, indent=1))
    return 0


def _table(a: argparse.Namespace, spec: run.Spec) -> int:
    v4 = _read_json(a.v4) if a.v4 else None
    tables.tables(spec, a.merged, a.gt or run.repo_path(spec["gt"]), a.out, v4)
    return 0


def _costs(a: argparse.Namespace, spec: run.Spec, out: Out) -> int:
    res = costs.costs(a.merged, a.factors, float(spec["budget_cpu_s"]), a.out, a.label)
    out(json.dumps({k: res[k] for k in ("total_ref_core_h", "factors_sha256")}))
    return 0


def _project(a: argparse.Namespace, spec: run.Spec, out: Out) -> int:
    res = run.project(spec, _read_json(a.dev_costs))
    if a.out:
        _write_json(a.out, res)
    out(json.dumps({k: v for k, v in res.items() if k != "cost_table"}, indent=1))
    return 0


def main(
    argv: Sequence[str] | None = None, out: Out = print, providers: Providers | None = None
) -> int:
    """Runs one command.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.
        out: Output sink.
        providers: Environment queries of `run`.

    Returns:
        The process exit code (`counts --check` gives 3 on a mismatch).
    """
    a = build_parser().parse_args(argv)
    if a.command == "prep":
        return _prep(a, out)
    spec = run.load_spec(a.spec)
    handlers: dict[str, Callable[[], int]] = {
        "counts": lambda: _counts(a, spec, out),
        "list": lambda: _list(spec, out),
        "run": lambda: _run(a, spec, providers or Providers(log=out)),
        "merge": lambda: _merge(a, spec, out),
        "table": lambda: _table(a, spec),
        "costs": lambda: _costs(a, spec, out),
        "project": lambda: _project(a, spec, out),
    }
    return handlers[a.command]()


if __name__ == "__main__":
    raise SystemExit(main())

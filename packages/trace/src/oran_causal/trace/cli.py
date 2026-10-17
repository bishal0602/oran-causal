"""Command line: `oran-causal-trace check [--strict]` and `oran-causal-trace matrix`."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from oran_causal.trace.manifests import read_manifests
from oran_causal.trace.matrix import render
from oran_causal.trace.paper import read_exemptions, read_generated, scan_paper
from oran_causal.trace.registry import load_registry
from oran_causal.trace.results import DEFAULT_PATH, read_results
from oran_causal.trace.rules import Context, Finding, Rule, Status, run_rules
from oran_causal.trace.spec import spec_labels

SHOWN_MESSAGES = 8


@dataclass(frozen=True, slots=True)
class Paths:
    """Input and output locations of the checker."""

    claims: Path
    schema: Path
    sections: Path
    tests: Path
    manifests: Path
    matrix: Path
    paper: Path
    exemptions: Path
    records: Path

    @classmethod
    def under(cls, root: Path) -> Self:
        """The standard layout under a workspace root."""
        return cls(
            claims=root / "spec" / "claims.yaml",
            schema=root / "spec" / "claims.schema.yaml",
            sections=root / "spec" / "sections",
            tests=root / DEFAULT_PATH,
            manifests=root / "data" / "manifests",
            matrix=root / "docs" / "TRACEABILITY.md",
            paper=root / "paper",
            exemptions=root / "spec" / "paper_exempt.yaml",
            records=root / "data" / "records",
        )


def find_root(start: Path) -> Path:
    """The nearest directory at or above `start` that holds `spec/claims.yaml`, else `start`."""
    for candidate in (start, *start.parents):
        if (candidate / "spec" / "claims.yaml").is_file():
            return candidate
    return start


def build_context(paths: Paths) -> Context:
    """Load every input the rules need."""
    registry = load_registry(paths.claims, paths.schema)
    tests = read_results(paths.tests)
    manifests = read_manifests(paths.manifests)
    return Context(
        registry=registry,
        tests=tests,
        manifests=manifests,
        spec_labels=spec_labels(paths.sections),
        matrix_path=paths.matrix,
        matrix_text=render(registry, tests, manifests),
        paper=scan_paper(paths.paper),
        generated=read_generated(paths.paper),
        exemptions=read_exemptions(paths.exemptions),
        records_root=paths.records,
    )


def _label(status: Status, *, strict: bool) -> str:
    if status is Status.FAIL and not strict:
        return "warn"
    return str(status)


def _report(results: list[tuple[Rule, Finding]], *, strict: bool, verbose: bool) -> list[str]:
    lines = []
    for rule, finding in results:
        lines.append(f"{rule.code:<4} {_label(finding.status, strict=strict):<14} {rule.title}")
        shown = finding.messages if verbose else finding.messages[:SHOWN_MESSAGES]
        lines += [f"       {message}" for message in shown]
        if len(finding.messages) > len(shown):
            lines.append(f"       ... {len(finding.messages) - len(shown)} more (use --verbose)")
    counts = {status: sum(f.status is status for _, f in results) for status in Status}
    lines.append(
        f"{len(results)} rules: {counts[Status.PASS]} pass, {counts[Status.FAIL]} fail, "
        f"{counts[Status.NOT_CHECKABLE]} not checkable ({'strict' if strict else 'non-strict'})"
    )
    return lines


def check(paths: Paths, *, strict: bool, verbose: bool) -> int:
    """Run the rules and print the summary.

    Returns:
        1 in strict mode when a rule fails, else 0.
    """
    results = run_rules(build_context(paths))
    print("\n".join(_report(results, strict=strict, verbose=verbose)))
    failed = any(finding.status is Status.FAIL for _, finding in results)
    return 1 if strict and failed else 0


def write_matrix(paths: Paths) -> int:
    """Regenerate `docs/TRACEABILITY.md`."""
    context = build_context(paths)
    paths.matrix.parent.mkdir(parents=True, exist_ok=True)
    paths.matrix.write_text(context.matrix_text, encoding="utf-8", newline="\n")
    print(f"wrote {paths.matrix}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oran-causal-trace", description=__doc__)
    parser.add_argument("--root", type=Path, help="workspace root (default: found from the cwd)")
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="run the traceability rules")
    check_parser.add_argument("--strict", action="store_true", help="exit 1 when a rule fails")
    check_parser.add_argument("--verbose", action="store_true", help="print every message")
    commands.add_parser("matrix", help="write docs/TRACEABILITY.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the `oran-causal-trace` console script."""
    args = _parser().parse_args(argv)
    paths = Paths.under(args.root or find_root(Path.cwd()))
    if args.command == "check":
        return check(paths, strict=args.strict, verbose=args.verbose)
    return write_matrix(paths)


if __name__ == "__main__":
    sys.exit(main())

"""Traceability rules F1-F14 over the registry, test results, manifests and the matrix."""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from oran_causal.trace.manifests import Manifests
from oran_causal.trace.registry import Claim, Evidence, Registry
from oran_causal.trace.results import MarkedTest


class Status(StrEnum):
    """Result of one rule."""

    PASS = "pass"
    FAIL = "fail"
    NOT_CHECKABLE = "not checkable"


@dataclass(frozen=True, slots=True)
class Finding:
    """Status of a rule with one message per problem found."""

    status: Status
    messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Context:
    """Everything the rules read."""

    registry: Registry
    tests: tuple[MarkedTest, ...] | None
    manifests: Manifests | None
    spec_labels: frozenset[str] | None
    matrix_path: Path
    matrix_text: str


@dataclass(frozen=True, slots=True)
class Rule:
    """A named check."""

    code: str
    title: str
    check: Callable[[Context], Finding]


def _finding(problems: list[str], *, unchecked: list[str] | None = None) -> Finding:
    if problems:
        return Finding(Status.FAIL, tuple(problems + (unchecked or [])))
    if unchecked:
        return Finding(Status.NOT_CHECKABLE, tuple(unchecked))
    return Finding(Status.PASS)


def schema_and_ids(context: Context) -> Finding:
    """F1: the registry matches its schema and ids are unique."""
    return _finding(list(context.registry.errors))


def _cycles(graph: Mapping[str, tuple[str, ...]]) -> list[list[str]]:
    state: dict[str, int] = {}
    cycles: list[list[str]] = []

    def visit(node: str, trail: list[str]) -> None:
        state[node] = 1
        for target in graph.get(node, ()):
            if state.get(target) == 1:
                cycles.append([*trail[trail.index(target) :], target])
            elif target in graph and target not in state:
                visit(target, [*trail, target])
        state[node] = 2

    for node in sorted(graph):
        if node not in state:
            visit(node, [node])
    return cycles


def _reference_problems(claim: Claim, by_id: Mapping[str, Claim]) -> list[str]:
    problems = []
    for field_name, targets in (("depends_on", claim.depends_on), ("probes", claim.probes)):
        for target in targets:
            if target not in by_id:
                problems.append(f"{claim.id}: {field_name} names unknown claim {target}")
            elif not by_id[target].active:
                problems.append(f"{claim.id}: {field_name} names retracted claim {target}")
    return problems


def references(context: Context) -> Finding:
    """F2: dependencies, probes and spec labels resolve, and dependencies are acyclic."""
    by_id = context.registry.by_id
    problems = [p for claim in context.registry.claims for p in _reference_problems(claim, by_id)]
    graph = {claim.id: claim.depends_on for claim in context.registry.claims}
    problems += [f"dependency cycle: {' -> '.join(cycle)}" for cycle in _cycles(graph)]
    if context.spec_labels is None:
        return _finding(problems, unchecked=["spec/sections not found: spec labels unchecked"])
    problems += [
        f"{claim.id}: spec label {claim.spec} is not anchored in spec/sections"
        for claim in context.registry.claims
        if claim.spec not in context.spec_labels
    ]
    return _finding(problems)


def _no_tests_message() -> list[str]:
    return ["build/trace/tests.json not found: run pytest from the workspace root first"]


def markers(context: Context) -> Finding:
    """F3: every claim marker names a registered claim."""
    if context.tests is None:
        return _finding([], unchecked=_no_tests_message())
    problems = [
        f"{test.nodeid}: claim marker without an id" for test in context.tests if not test.claims
    ]
    problems += [
        f"{test.nodeid}: unknown claim {claim_id}"
        for test in context.tests
        for claim_id in test.claims
        if claim_id not in context.registry.by_id
    ]
    return _finding(problems)


def _claim_problems(claim: Claim, passed: Counter[str] | None) -> list[str]:
    problems = []
    if claim.kind == "theorem-cited" and not (claim.cited and claim.cited.verified):
        problems.append(f"{claim.id}: cited theorem locator not verified")
    if claim.kind == "assumption" and not (claim.probes or claim.tests_min):
        problems.append(f"{claim.id}: assumption has neither a probe nor a required test")
    if passed is not None and passed[claim.id] < claim.tests_min:
        problems.append(
            f"{claim.id}: {passed[claim.id]} of {claim.tests_min} required passing tests"
        )
    return problems


def coverage(context: Context) -> Finding:
    """F4: required tests pass, assumptions are probed, cited locators are verified."""
    passed = None
    if context.tests is not None:
        passed = Counter(c for test in context.tests if test.passed for c in set(test.claims))
    active = (claim for claim in context.registry.claims if claim.active)
    problems = [p for claim in active for p in _claim_problems(claim, passed)]
    return _finding(problems, unchecked=None if passed is not None else _no_tests_message())


def _hash_problem(evidence: Evidence, listing: Mapping[str, str]) -> str | None:
    listed = listing.get(evidence.record_path)
    if listed is None:
        return f"{evidence.record}: not listed in data/manifests/{evidence.record_set}.sha256"
    if listed != evidence.sha256.lower():
        return (
            f"{evidence.record}: sha256 {evidence.sha256[:12]} differs from manifest {listed[:12]}"
        )
    return None


def record_hashes(context: Context) -> Finding:
    """F6: every evidence sha256 equals its data manifest entry."""
    if context.manifests is None:
        return _finding([], unchecked=["data/manifests not found: record hashes unchecked"])
    problems, unchecked = [], []
    for claim in context.registry.claims:
        for evidence in claim.evidence:
            listing = context.manifests.get(evidence.record_set)
            if listing is None:
                unchecked.append(f"{claim.id}: no manifest for set {evidence.record_set}")
            elif problem := _hash_problem(evidence, listing):
                problems.append(f"{claim.id}: {problem}")
    return _finding(problems, unchecked=unchecked)


def matrix_fresh(context: Context) -> Finding:
    """F13: the committed matrix equals the regenerated one."""
    if not context.matrix_path.is_file():
        return _finding([f"{context.matrix_path} missing: run `oran-causal-trace matrix`"])
    current = context.matrix_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if current != context.matrix_text:
        return _finding([f"{context.matrix_path} is stale: run `oran-causal-trace matrix`"])
    return _finding([])


def pending(reason: str) -> Callable[[Context], Finding]:
    """A rule whose inputs do not exist yet in this phase of the migration."""

    def check(_: Context) -> Finding:
        return Finding(Status.NOT_CHECKABLE, (reason,))

    return check


PAPER_PENDING = "paper/ and its LaTeX build output arrive in migration phase 10"

RULES = (
    Rule("F1", "registry matches the schema, ids unique", schema_and_ids),
    Rule("F2", "references resolve, no dependency cycle", references),
    Rule("F3", "test markers name registered claims", markers),
    Rule("F4", "required tests pass, assumptions probed, citations verified", coverage),
    Rule(
        "F5",
        "Lean declarations exist, axiom audit clean",
        pending("proofs/ arrives with the Lean phase L"),
    ),
    Rule("F6", "evidence sha256 equals the data manifest", record_hashes),
    Rule(
        "F7",
        "record pointers resolve, expect values hold",
        pending("records are resolved from migration phase 9 on"),
    ),
    Rule("F8", "\\claim usages name registered claims", pending(PAPER_PENDING)),
    Rule("F9", "every empirical and theorem claim is \\claim-ed", pending(PAPER_PENDING)),
    Rule("F10", "\\val keys are generated and backed by evidence", pending(PAPER_PENDING)),
    Rule("F11", "no bare decimals in paper prose", pending(PAPER_PENDING)),
    Rule("F12", "code symbols import", pending("members are ported in migration phases 2-9")),
    Rule("F13", "docs/TRACEABILITY.md is current", matrix_fresh),
    Rule("F14", "post hoc claims carry the post hoc wording", pending(PAPER_PENDING)),
)


def run_rules(context: Context) -> list[tuple[Rule, Finding]]:
    """Run every rule in order."""
    return [(rule, rule.check(context)) for rule in RULES]

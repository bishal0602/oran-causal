from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from oran_causal.trace.cli import Paths, build_context
from oran_causal.trace.results import MarkedTest
from oran_causal.trace.rules import (
    RULES,
    Context,
    Status,
    coverage,
    markers,
    matrix_fresh,
    record_hashes,
    references,
    run_rules,
    schema_and_ids,
)

type Workspace = Callable[..., Path]
type MakeClaim = Callable[..., dict[str, Any]]

SHA = "b" * 64


def context_for(root: Path, **changes: Any) -> Context:
    return replace(build_context(Paths.under(root)), **changes)


def passed(nodeid: str, *claims: str) -> MarkedTest:
    return MarkedTest(nodeid, claims, "passed")


def test_f1_passes_on_a_valid_registry(workspace: Workspace, new_claim: MakeClaim) -> None:
    assert schema_and_ids(context_for(workspace([new_claim("DEF-A")]))).status is Status.PASS


def test_f1_fails_on_duplicate_ids(workspace: Workspace, new_claim: MakeClaim) -> None:
    finding = schema_and_ids(context_for(workspace([new_claim("DEF-A"), new_claim("DEF-A")])))
    assert finding.status is Status.FAIL
    assert "duplicate claim id" in finding.messages[0]


def test_f2_flags_unknown_retracted_and_cyclic_references(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    root = workspace(
        [
            new_claim("DEF-A", depends_on=["DEF-B"]),
            new_claim("DEF-B", depends_on=["DEF-A", "DEF-X"]),
            new_claim("ASM-C", kind="assumption", probes=["EMP-R"]),
            new_claim("EMP-R", kind="empirical", status="retracted"),
        ]
    )
    messages = references(context_for(root)).messages
    assert "DEF-B: depends_on names unknown claim DEF-X" in messages
    assert "ASM-C: probes names retracted claim EMP-R" in messages
    assert any(
        message.startswith("dependency cycle: DEF-A -> DEF-B -> DEF-A") for message in messages
    )


def test_f2_flags_unanchored_spec_labels(workspace: Workspace, new_claim: MakeClaim) -> None:
    root = workspace([new_claim("DEF-A", spec="spec:b")], labels=("spec:a",))
    finding = references(context_for(root))
    assert finding.status is Status.FAIL
    assert "spec label spec:b is not anchored" in finding.messages[0]


def test_f2_without_spec_sections_is_partial(workspace: Workspace, new_claim: MakeClaim) -> None:
    finding = references(context_for(workspace([new_claim("DEF-A")]), spec_labels=None))
    assert finding.status is Status.NOT_CHECKABLE


def test_f3_flags_unknown_and_empty_markers(workspace: Workspace, new_claim: MakeClaim) -> None:
    tests = (passed("t::a", "DEF-A"), passed("t::b", "DEF-Z"), passed("t::c"))
    finding = markers(context_for(workspace([new_claim("DEF-A")]), tests=tests))
    assert finding.messages == ("t::c: claim marker without an id", "t::b: unknown claim DEF-Z")


def test_f3_needs_test_results(workspace: Workspace, new_claim: MakeClaim) -> None:
    finding = markers(context_for(workspace([new_claim("DEF-A")]), tests=None))
    assert finding.status is Status.NOT_CHECKABLE


def test_f4_counts_passing_tests(workspace: Workspace, new_claim: MakeClaim) -> None:
    root = workspace([new_claim("DEF-A", tests={"min": 2}), new_claim("DEF-B", tests={"min": 1})])
    tests = (
        passed("t::a", "DEF-A", "DEF-B"),
        MarkedTest("t::b", ("DEF-A",), "failed"),
    )
    finding = coverage(context_for(root, tests=tests))
    assert finding.messages == ("DEF-A: 1 of 2 required passing tests",)


def test_f4_checks_citations_and_assumptions(workspace: Workspace, new_claim: MakeClaim) -> None:
    cited = {"bibkey": "k", "locator": "Thm. 1", "verified": False}
    root = workspace(
        [
            new_claim("CIT-A", kind="theorem-cited", cited=cited),
            new_claim("ASM-B", kind="assumption"),
            new_claim("ASM-C", kind="assumption", probes=["CIT-A"]),
        ]
    )
    finding = coverage(context_for(root, tests=()))
    assert finding.messages == (
        "CIT-A: cited theorem locator not verified",
        "ASM-B: assumption has neither a probe nor a required test",
    )


def test_f4_without_results_still_checks_citations(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    root = workspace([new_claim("DEF-A", tests={"min": 1})])
    assert coverage(context_for(root, tests=None)).status is Status.NOT_CHECKABLE


def write_manifest(root: Path, name: str, lines: list[str]) -> None:
    directory = root / "data" / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_f6_compares_evidence_with_manifests(workspace: Workspace, new_claim: MakeClaim) -> None:
    evidence = [
        {"record": "set_a/ok.json", "sha256": SHA},
        {"record": "set_a/bad.json", "sha256": SHA},
        {"record": "set_a/absent.json", "sha256": SHA},
    ]
    root = workspace([new_claim("EMP-A", kind="empirical", evidence=evidence)])
    write_manifest(root, "set_a", [f"{SHA}  ok.json", f"{'c' * 64} *bad.json"])
    finding = record_hashes(context_for(root))
    assert finding.status is Status.FAIL
    assert finding.messages == (
        f"EMP-A: set_a/bad.json: sha256 {SHA[:12]} differs from manifest {'c' * 12}",
        "EMP-A: set_a/absent.json: not listed in data/manifests/set_a.sha256",
    )


def test_f6_without_a_manifest_is_not_checkable(workspace: Workspace, new_claim: MakeClaim) -> None:
    evidence = [{"record": "set_b/x.json", "sha256": SHA}]
    root = workspace([new_claim("EMP-A", kind="empirical", evidence=evidence)])
    write_manifest(root, "set_a", [])
    finding = record_hashes(context_for(root))
    assert finding.status is Status.NOT_CHECKABLE
    assert finding.messages == ("EMP-A: no manifest for set set_b",)


def test_f13_detects_missing_stale_and_fresh_matrix(
    workspace: Workspace, new_claim: MakeClaim
) -> None:
    root = workspace([new_claim("DEF-A")])
    context = context_for(root)
    assert matrix_fresh(context).status is Status.FAIL
    context.matrix_path.parent.mkdir(parents=True)
    context.matrix_path.write_text("old\n", encoding="utf-8")
    assert "stale" in matrix_fresh(context).messages[0]
    context.matrix_path.write_text(context.matrix_text, encoding="utf-8")
    assert matrix_fresh(context).status is Status.PASS


@pytest.mark.parametrize("code", ["F5", "F7", "F8", "F9", "F10", "F11", "F12", "F14"])
def test_pending_rules_say_why_they_are_not_checkable(
    code: str, workspace: Workspace, new_claim: MakeClaim
) -> None:
    context = context_for(workspace([new_claim("DEF-A")]))
    results = {rule.code: finding for rule, finding in run_rules(context)}
    assert results[code].status is Status.NOT_CHECKABLE
    assert results[code].messages


def test_rules_cover_f1_to_f14_in_order() -> None:
    assert [rule.code for rule in RULES] == [f"F{n}" for n in range(1, 15)]

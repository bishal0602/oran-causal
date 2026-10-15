from collections.abc import Callable
from pathlib import Path
from typing import Any

from oran_causal.trace.matrix import render
from oran_causal.trace.registry import load_registry
from oran_causal.trace.results import MarkedTest

type Workspace = Callable[..., Path]
type MakeClaim = Callable[..., dict[str, Any]]

SHA = "d" * 64


def registry_of(root: Path):
    return load_registry(root / "spec" / "claims.yaml", root / "spec" / "claims.schema.yaml")


def test_render_lists_claims_tests_and_records(workspace: Workspace, new_claim: MakeClaim) -> None:
    evidence = [{"record": "set_a/x.json", "sha256": SHA}]
    root = workspace(
        [
            new_claim(
                "EMP-A", kind="empirical", tests={"min": 1}, evidence=evidence, statement="a | b"
            ),
            new_claim("DEF-B", status="retracted"),
        ]
    )
    text = render(
        registry_of(root),
        (MarkedTest("t::x", ("EMP-A",), "passed"),),
        {"set_a": {"x.json": SHA}},
    )
    assert "| EMP-A | empirical | a \\| b |" in text
    assert "| 1/1 | `set_a/x.json`@dddddddd (ok) |" in text
    assert "| DEF-B (retracted) |" in text
    assert "| `t::x` | EMP-A | passed |" in text
    assert "| `set_a/x.json` | EMP-A |" in text
    assert "| empirical | 1 | 1 | 1 |" in text


def test_render_without_results_or_manifests(workspace: Workspace, new_claim: MakeClaim) -> None:
    evidence = [{"record": "set_a/x.json", "sha256": SHA}]
    root = workspace([new_claim("EMP-A", kind="empirical", tests={"min": 1}, evidence=evidence)])
    text = render(registry_of(root), None, None)
    assert "| n/a/1 | `set_a/x.json`@dddddddd (no manifest) |" in text
    assert "No test results" in text


def test_render_is_deterministic(workspace: Workspace, new_claim: MakeClaim) -> None:
    registry = registry_of(workspace([new_claim("DEF-A")]))
    assert render(registry, (), {}) == render(registry, (), {})

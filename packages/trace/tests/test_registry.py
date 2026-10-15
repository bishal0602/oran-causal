from collections.abc import Callable
from pathlib import Path
from typing import Any

from oran_causal.trace.registry import Registry, load_registry

type Workspace = Callable[..., Path]
type MakeClaim = Callable[..., dict[str, Any]]


def load(root: Path) -> Registry:
    return load_registry(root / "spec" / "claims.yaml", root / "spec" / "claims.schema.yaml")


def test_loads_typed_claims(workspace: Workspace, new_claim: MakeClaim) -> None:
    evidence = {"record": "set_a/x.json", "sha256": "a" * 64, "pointer": "/k", "values": ["v.k"]}
    cited = {"bibkey": "key", "locator": "Thm. 1", "verified": True}
    root = workspace(
        [
            new_claim("DEF-A", tests={"min": 2, "tier": "unit"}, evidence=[evidence]),
            new_claim("CIT-B", kind="theorem-cited", cited=cited, depends_on=["DEF-A"]),
        ]
    )
    registry = load(root)
    assert registry.errors == ()
    first, second = registry.claims
    assert first.tests_min == 2
    assert first.evidence[0].record_set == "set_a"
    assert first.evidence[0].record_path == "x.json"
    assert second.cited is not None
    assert second.cited.verified
    assert registry.by_id["CIT-B"].depends_on == ("DEF-A",)


def test_schema_errors_prevent_loading(workspace: Workspace, new_claim: MakeClaim) -> None:
    registry = load(workspace([new_claim("bad id"), new_claim("DEF-A", kind="opinion")]))
    assert registry.claims == ()
    assert len(registry.errors) == 2


def test_duplicate_ids_are_errors(workspace: Workspace, new_claim: MakeClaim) -> None:
    registry = load(workspace([new_claim("DEF-A"), new_claim("DEF-A")]))
    assert registry.errors == ("duplicate claim id 'DEF-A' (2 entries)",)


def test_missing_file_is_an_error(tmp_path: Path, schema_path: Path) -> None:
    registry = load_registry(tmp_path / "claims.yaml", schema_path)
    assert "file not found" in registry.errors[0]


def test_invalid_yaml_is_an_error(tmp_path: Path, schema_path: Path) -> None:
    path = tmp_path / "claims.yaml"
    path.write_text("claims: [", encoding="utf-8")
    assert "invalid YAML" in load_registry(path, schema_path).errors[0]


def test_seeded_registry_is_valid(repo_root: Path) -> None:
    registry = load(repo_root)
    assert registry.errors == ()
    assert len(registry.claims) == 80

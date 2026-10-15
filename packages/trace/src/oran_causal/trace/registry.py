"""Typed loader for `spec/claims.yaml`."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from oran_causal.trace.schema import validate

CLAIM_KINDS = (
    "definition",
    "assumption",
    "theorem-paper",
    "theorem-lean",
    "theorem-cited",
    "empirical",
    "protocol",
)


@dataclass(frozen=True, slots=True)
class Evidence:
    """A record that backs a claim, pinned by its sha256."""

    record: str
    sha256: str
    pointer: str | None = None
    expect: Any = None
    derive: str | None = None
    values: tuple[str, ...] = ()

    @property
    def record_set(self) -> str:
        """Name of the data set: the first path segment of `record`."""
        return self.record.split("/", 1)[0]

    @property
    def record_path(self) -> str:
        """Path of the record inside its set."""
        return self.record.split("/", 1)[1] if "/" in self.record else ""


@dataclass(frozen=True, slots=True)
class Cited:
    """A theorem taken from the literature."""

    bibkey: str
    locator: str
    verified: bool


@dataclass(frozen=True, slots=True)
class Claim:
    """One traceable claim of the paper."""

    id: str
    kind: str
    statement: str
    paper: tuple[str, ...]
    spec: str
    status: str = "active"
    depends_on: tuple[str, ...] = ()
    probes: tuple[str, ...] = ()
    code: tuple[str, ...] = ()
    lean_decl: str | None = None
    cited: Cited | None = None
    tests_min: int = 0
    tests_tier: str | None = None
    evidence: tuple[Evidence, ...] = ()
    posthoc: bool = False

    @property
    def active(self) -> bool:
        """Whether the claim is still made by the paper."""
        return self.status == "active"


@dataclass(frozen=True, slots=True)
class Registry:
    """The loaded registry, or the errors that prevented loading it."""

    claims: tuple[Claim, ...] = ()
    errors: tuple[str, ...] = ()
    by_id: dict[str, Claim] = field(default_factory=dict)


def _evidence(raw: dict[str, Any]) -> Evidence:
    return Evidence(
        record=raw["record"],
        sha256=raw["sha256"],
        pointer=raw.get("pointer"),
        expect=raw.get("expect"),
        derive=raw.get("derive"),
        values=tuple(raw.get("values", ())),
    )


def _claim(raw: dict[str, Any]) -> Claim:
    cited = raw.get("cited")
    tests = raw.get("tests", {})
    return Claim(
        id=raw["id"],
        kind=raw["kind"],
        statement=raw["statement"],
        paper=tuple(raw["paper"]),
        spec=raw["spec"],
        status=raw.get("status", "active"),
        depends_on=tuple(raw.get("depends_on", ())),
        probes=tuple(raw.get("probes", ())),
        code=tuple(raw.get("code", ())),
        lean_decl=raw.get("lean", {}).get("decl"),
        cited=Cited(cited["bibkey"], cited["locator"], cited["verified"]) if cited else None,
        tests_min=tests.get("min", 0),
        tests_tier=tests.get("tier"),
        evidence=tuple(_evidence(item) for item in raw.get("evidence", ())),
        posthoc=raw.get("posthoc", False),
    )


def _read_yaml(path: Path) -> tuple[Any, list[str]]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), []
    except FileNotFoundError:
        return None, [f"{path}: file not found"]
    except yaml.YAMLError as error:
        return None, [f"{path}: invalid YAML: {error}"]


def _duplicate_errors(raw_claims: list[dict[str, Any]]) -> list[str]:
    counts = Counter(item["id"] for item in raw_claims)
    return [
        f"duplicate claim id {key!r} ({n} entries)" for key, n in sorted(counts.items()) if n > 1
    ]


def load_registry(claims_path: Path, schema_path: Path) -> Registry:
    """Load and validate the claims registry.

    Args:
        claims_path: Path of `claims.yaml`.
        schema_path: Path of `claims.schema.yaml`.

    Returns:
        The typed registry, or an empty registry with the schema and
        duplicate-id errors when validation fails.
    """
    schema, errors = _read_yaml(schema_path)
    document, document_errors = _read_yaml(claims_path)
    errors += document_errors
    if errors:
        return Registry(errors=tuple(errors))
    errors = validate(document, schema)
    if errors:
        return Registry(errors=tuple(errors))
    errors = _duplicate_errors(document["claims"])
    if errors:
        return Registry(errors=tuple(errors))
    claims = tuple(_claim(item) for item in document["claims"])
    return Registry(claims=claims, by_id={claim.id: claim for claim in claims})

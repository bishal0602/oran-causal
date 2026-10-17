"""Per-study manifests: what a study is, what it costs and how its records are checked.

Every study or post hoc item has `manifests/<id>.yaml` in the experiments member. The file ties
the item to its claims, frozen protocol, seed blocks, stages (driver, inputs, outputs, cost),
frozen inputs, stored record sets, reproduce tiers and equivalence rules. `load` and `load_all`
parse a manifest into frozen dataclasses; `check` verifies it against the files, the claim
registry, the seed registry and the importable drivers.

Costs are reference CPU-hours (Kaggle host, Xeon 2.20 GHz). A cost is a number or a `[low, high]`
extrapolation. A stage listed in `frozen_in` reuses the frozen records in that tier and costs
nothing there. `tier_cost` sums a tier over manifests; `manifests/tiers.yaml` prices the same
stages and `reproduce.plan` reads it, so the two must agree (tested).
"""

import dataclasses
import hashlib
import importlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from oran_causal.experiments.paths import MANIFESTS, MEMBER_ROOT, PROTOCOLS, WORKSPACE_ROOT

TIERS = ("t0", "t1", "t2", "t3")
COST_TIERS = ("t2", "t3")
BASES = ("M", "P", "E")
ORIGIN_PREFIX = "CDD-ORAN@0c89106:"
SEEDS_FILE = MANIFESTS / "seeds.json"
CLAIMS_FILE = WORKSPACE_ROOT / "spec" / "claims.yaml"
DATA_ROOT = WORKSPACE_ROOT / "data"
ZENODO_FILE = DATA_ROOT / "zenodo.yaml"


class ManifestError(ValueError):
    """A manifest does not follow the schema."""


@dataclasses.dataclass(frozen=True)
class Protocol:
    """A frozen protocol file under `protocols/`.

    Attributes:
        path: Path relative to `protocols/`.
        sha256: SHA-256 of the file with LF line endings.
    """

    path: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class Seed:
    """A seed block.

    Attributes:
        block: Registry claim name or stage label.
        range: Inclusive `(low, high)`.
        role: What the block is used for.
        world: Key of `seeds.json` the block belongs to.
    """

    block: str
    range: tuple[int, int]
    role: str
    world: str


@dataclasses.dataclass(frozen=True)
class Cost:
    """Reference CPU-hours of a stage.

    Attributes:
        low: Lower end.
        high: Upper end (equal to `low` unless extrapolated).
        basis: `M` measured sum of record times, `P` protocol estimate, `E` extrapolation.
    """

    low: float
    high: float
    basis: str


@dataclasses.dataclass(frozen=True)
class Stage:
    """One stage of an item.

    Attributes:
        name: Stage name, as in `tiers.yaml`.
        driver: `module:callable` entry point.
        args: Fixed keyword arguments the driver is called with (a stage selector, say).
        inputs: Frozen files or record paths the stage reads.
        outputs: Files or record paths it writes.
        cost: Reference cost.
        frozen_in: Tiers in which the stage reuses the frozen records.
    """

    name: str
    driver: str
    args: Mapping[str, Any]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cost: Cost
    frozen_in: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class FrozenInput:
    """A frozen file under the experiments member root."""

    path: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class RecordSet:
    """A stored record set.

    Attributes:
        set: Name in `data/zenodo.yaml`.
        manifest: `data/`-relative path of its `.sha256` manifest.
        zenodo_doi: DOI once deposited, else None.
    """

    set: str
    manifest: str
    zenodo_doi: str | None


@dataclasses.dataclass(frozen=True)
class Equivalence:
    """A same-conclusion rule for a re-run."""

    metric: str
    rule: str


@dataclasses.dataclass(frozen=True)
class Manifest:
    """One study or post hoc item.

    Attributes:
        id: Item id, equal to the file stem.
        platform: Reference lane of its re-run (`kaggle`, `colab`, `vps`), as in `tiers.yaml`.
        claims: Registered claim IDs the item supports.
        protocol: Frozen protocol, if the item has one of its own.
        code_origin: `CDD-ORAN@0c89106:<path>` entries the item was ported from.
        seeds: Seed blocks.
        stages: Stages in run order.
        frozen_inputs: Frozen files with their hashes.
        records: Stored record sets.
        tiers: What each tier does for the item; None when the item is not in that tier.
        equivalence: Rules under which a re-run agrees with the stored result.
    """

    id: str
    platform: str
    claims: tuple[str, ...]
    protocol: Protocol | None
    code_origin: tuple[str, ...]
    seeds: tuple[Seed, ...]
    stages: tuple[Stage, ...]
    frozen_inputs: tuple[FrozenInput, ...]
    records: tuple[RecordSet, ...]
    tiers: Mapping[str, str | None]
    equivalence: tuple[Equivalence, ...]

    def stage_cost(self, tier: str) -> tuple[float, float]:
        """Reference CPU-h `(low, high)` of the stages that run in `tier`."""
        if self.tiers.get(tier) is None:
            return 0.0, 0.0
        live = [s.cost for s in self.stages if tier not in s.frozen_in]
        return sum(c.low for c in live), sum(c.high for c in live)


def _fail(where: str, message: str) -> ManifestError:
    return ManifestError(f"{where}: {message}")


def _table(
    value: Any, where: str, keys: Iterable[str], optional: Iterable[str] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, "expected a mapping")
    need, allowed = set(keys), set(keys) | set(optional)
    if missing := sorted(need - value.keys()):
        raise _fail(where, f"missing {', '.join(missing)}")
    if extra := sorted(value.keys() - allowed):
        raise _fail(where, f"unknown {', '.join(extra)}")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(where, "expected a non-empty string")
    return value


def _texts(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail(where, "expected a list")
    return tuple(_text(v, f"{where}[{i}]") for i, v in enumerate(value))


def _items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(where, "expected a list")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise _fail(where, "expected a non-negative number")
    return float(value)


def _sha(value: Any, where: str) -> str:
    text = _text(value, where)
    if len(text) != 64 or set(text) - set("0123456789abcdef"):
        raise _fail(where, "expected a lowercase hex SHA-256")
    return text


def _cost(value: Any, where: str) -> Cost:
    row = _table(value, where, ("cpu_h", "basis"))
    basis = _text(row["basis"], f"{where}.basis")
    if basis not in BASES:
        raise _fail(where, f"basis {basis!r} is not one of {', '.join(BASES)}")
    cpu = row["cpu_h"]
    if isinstance(cpu, list):
        if len(cpu) != 2:
            raise _fail(where, "cpu_h range needs [low, high]")
        low, high = _number(cpu[0], where), _number(cpu[1], where)
        if low > high:
            raise _fail(where, "cpu_h range is reversed")
        return Cost(low, high, basis)
    value = _number(cpu, f"{where}.cpu_h")
    return Cost(value, value, basis)


def _stage(value: Any, where: str) -> Stage:
    row = _table(
        value,
        where,
        ("name", "driver", "inputs", "outputs", "cost"),
        ("args", "frozen_in"),
    )
    args = row.get("args", {})
    if not isinstance(args, dict):
        raise _fail(where, "args: expected a mapping")
    frozen_in = _texts(row.get("frozen_in", []), f"{where}.frozen_in")
    if bad := [t for t in frozen_in if t not in TIERS]:
        raise _fail(where, f"frozen_in names unknown tier {bad[0]!r}")
    return Stage(
        _text(row["name"], f"{where}.name"),
        _text(row["driver"], f"{where}.driver"),
        args,
        _texts(row["inputs"], f"{where}.inputs"),
        _texts(row["outputs"], f"{where}.outputs"),
        _cost(row["cost"], f"{where}.cost"),
        frozen_in,
    )


def _seed(value: Any, where: str) -> Seed:
    row = _table(value, where, ("block", "range", "role", "world"))
    span = row["range"]
    if (
        not isinstance(span, list)
        or len(span) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in span)
        or span[0] > span[1]
    ):
        raise _fail(where, "range: expected [low, high] integers")
    return Seed(
        _text(row["block"], f"{where}.block"),
        (span[0], span[1]),
        _text(row["role"], f"{where}.role"),
        _text(row["world"], f"{where}.world"),
    )


def _record_set(value: Any, where: str) -> RecordSet:
    row = _table(value, where, ("set", "manifest", "zenodo_doi"))
    doi = row["zenodo_doi"]
    if doi is not None:
        doi = _text(doi, f"{where}.zenodo_doi")
    return RecordSet(
        _text(row["set"], f"{where}.set"), _text(row["manifest"], f"{where}.manifest"), doi
    )


def _tiers(value: Any, where: str) -> dict[str, str | None]:
    row = _table(value, where, TIERS)
    return {t: None if row[t] is None else _text(row[t], f"{where}.{t}") for t in TIERS}


def parse(doc: Any, source: str = "manifest") -> Manifest:
    """Parses a loaded YAML document.

    Args:
        doc: The document.
        source: Name used in error messages.

    Raises:
        ManifestError: A key is missing, unknown or has the wrong shape.
    """
    row = _table(
        doc,
        source,
        (
            "id",
            "platform",
            "claims",
            "code_origin",
            "seeds",
            "stages",
            "frozen_inputs",
            "records",
            "tiers",
            "equivalence",
        ),
        ("protocol",),
    )
    protocol = None
    if (p := row.get("protocol")) is not None:
        entry = _table(p, f"{source}.protocol", ("path", "sha256"))
        protocol = Protocol(
            _text(entry["path"], f"{source}.protocol.path"),
            _sha(entry["sha256"], f"{source}.protocol.sha256"),
        )
    frozen = []
    for i, f in enumerate(_items(row["frozen_inputs"], f"{source}.frozen_inputs")):
        entry = _table(f, f"{source}.frozen_inputs[{i}]", ("path", "sha256"))
        frozen.append(
            FrozenInput(
                _text(entry["path"], f"{source}.frozen_inputs[{i}].path"),
                _sha(entry["sha256"], f"{source}.frozen_inputs[{i}].sha256"),
            )
        )
    rules = []
    for i, e in enumerate(_items(row["equivalence"], f"{source}.equivalence")):
        entry = _table(e, f"{source}.equivalence[{i}]", ("metric", "rule"))
        rules.append(
            Equivalence(
                _text(entry["metric"], f"{source}.equivalence[{i}].metric"),
                _text(entry["rule"], f"{source}.equivalence[{i}].rule"),
            )
        )
    out = Manifest(
        _text(row["id"], f"{source}.id"),
        _text(row["platform"], f"{source}.platform"),
        _texts(row["claims"], f"{source}.claims"),
        protocol,
        _texts(row["code_origin"], f"{source}.code_origin"),
        tuple(
            _seed(s, f"{source}.seeds[{i}]")
            for i, s in enumerate(_items(row["seeds"], f"{source}.seeds"))
        ),
        tuple(
            _stage(s, f"{source}.stages[{i}]")
            for i, s in enumerate(_items(row["stages"], f"{source}.stages"))
        ),
        tuple(frozen),
        tuple(
            _record_set(r, f"{source}.records[{i}]")
            for i, r in enumerate(_items(row["records"], f"{source}.records"))
        ),
        _tiers(row["tiers"], f"{source}.tiers"),
        tuple(rules),
    )
    if not out.stages:
        raise _fail(source, "no stages")
    if not out.records:
        raise _fail(source, "no records")
    return out


def ids(root: Path = MANIFESTS) -> list[str]:
    """Manifest ids on disk, sorted (every `<id>.yaml` except `tiers.yaml`)."""
    return sorted(p.stem for p in root.glob("*.yaml") if p.stem != "tiers")


def load(item: str, root: Path = MANIFESTS) -> Manifest:
    """Loads `<root>/<item>.yaml`.

    Raises:
        ManifestError: The file is missing, malformed, or its `id` differs from `item`.
    """
    path = root / f"{item}.yaml"
    if not path.is_file():
        raise ManifestError(f"{item}: no manifest at {path}")
    out = parse(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)
    if out.id != item:
        raise _fail(path.name, f"id {out.id!r} differs from the file name")
    return out


def load_all(root: Path = MANIFESTS) -> dict[str, Manifest]:
    """Loads every manifest, keyed by id."""
    return {i: load(i, root) for i in ids(root)}


def tier_cost(manifests: Iterable[Manifest], tier: str) -> tuple[float, float]:
    """Reference CPU-h `(low, high)` of `tier` over `manifests`.

    Raises:
        ValueError: `tier` is not t2 or t3.
    """
    if tier not in COST_TIERS:
        raise ValueError(f"tier {tier!r}: costs exist for {', '.join(COST_TIERS)}")
    spans = [m.stage_cost(tier) for m in manifests]
    return sum(s[0] for s in spans), sum(s[1] for s in spans)


def lf_sha256(path: Path) -> str:
    """SHA-256 of `path` with CRLF read as LF (the hash `SHA256SUMS` records for text)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def resolve_driver(ref: str) -> Callable[..., Any]:
    """Imports `module:callable`.

    Raises:
        ManifestError: The reference is malformed, or the module or callable is missing.
    """
    module, sep, name = ref.partition(":")
    if not sep or not module or not name:
        raise ManifestError(f"driver {ref!r}: expected module:callable")
    try:
        target = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ManifestError(f"driver {ref!r}: {exc}") from exc
    if not callable(target):
        raise ManifestError(f"driver {ref!r}: not callable")
    return target


def claim_ids(path: Path = CLAIMS_FILE) -> set[str]:
    """IDs registered in `spec/claims.yaml`."""
    return {c["id"] for c in yaml.safe_load(path.read_text(encoding="utf-8"))["claims"]}


def _pairs(node: Any) -> Iterable[tuple[int, int]]:
    if isinstance(node, list):
        if len(node) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in node):
            yield node[0], node[1]
        else:
            for child in node:
                yield from _pairs(child)
    elif isinstance(node, dict):
        for key, child in node.items():
            if key != "note":
                yield from _pairs(child)


def registered_ranges(world: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Every `[low, high]` pair listed anywhere under one world of `seeds.json`."""
    return list(_pairs(world))


def _inside(span: tuple[int, int], pairs: Iterable[tuple[int, int]]) -> bool:
    return any(lo <= span[0] and span[1] <= hi for lo, hi in pairs)


def _check_seeds(m: Manifest, registry: Mapping[str, Any]) -> list[str]:
    problems = []
    for s in m.seeds:
        world = registry.get(s.world)
        if not isinstance(world, dict):
            problems.append(f"{m.id}: seed block {s.block}: world {s.world!r} not in seeds.json")
            continue
        pairs = registered_ranges(world)
        if not _inside(s.range, pairs):
            problems.append(
                f"{m.id}: seed block {s.block} {list(s.range)} is not registered for {s.world}"
            )
    return problems


def _check_files(m: Manifest, protocols: Path, member: Path) -> list[str]:
    problems = []
    if m.protocol is not None:
        path = protocols / m.protocol.path
        if not path.is_file():
            problems.append(f"{m.id}: protocol {m.protocol.path} is missing")
        elif (got := lf_sha256(path)) != m.protocol.sha256:
            problems.append(
                f"{m.id}: protocol {m.protocol.path} has sha256 {got}, "
                f"manifest says {m.protocol.sha256}"
            )
    for f in m.frozen_inputs:
        path = member / f.path
        if not path.is_file():
            problems.append(f"{m.id}: frozen input {f.path} is missing")
        elif (got := lf_sha256(path)) != f.sha256:
            problems.append(
                f"{m.id}: frozen input {f.path} has sha256 {got}, manifest says {f.sha256}"
            )
    return problems


def _check_records(m: Manifest, data: Path, zenodo: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems = []
    for r in m.records:
        if not (data / r.manifest).is_file():
            problems.append(f"{m.id}: record manifest {r.manifest} is missing")
        entry = zenodo.get(r.set)
        if entry is None:
            problems.append(f"{m.id}: record set {r.set} is not in zenodo.yaml")
        elif entry["manifest"] != r.manifest:
            problems.append(
                f"{m.id}: record set {r.set} lives in {entry['manifest']}, not {r.manifest}"
            )
        elif entry.get("doi") != r.zenodo_doi:
            problems.append(
                f"{m.id}: record set {r.set} doi {entry.get('doi')!r} differs from the manifest"
            )
    return problems


def check(
    m: Manifest,
    *,
    claims: Iterable[str] | None = None,
    registry: Mapping[str, Any] | None = None,
    protocols: Path = PROTOCOLS,
    member: Path = MEMBER_ROOT,
    data: Path = DATA_ROOT,
    zenodo: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Checks a manifest against the repository; returns the problems found.

    Verifies that claim IDs are registered, seed ranges agree with `seeds.json`, the protocol and
    frozen inputs exist with the stated hashes, `code_origin` names CDD-ORAN paths, every driver
    imports, `frozen_in` stages belong to a tier of the item, and the record sets match
    `data/zenodo.yaml`.

    Args:
        m: The manifest.
        claims: Registered claim IDs (default: `spec/claims.yaml`).
        registry: Parsed `seeds.json` (default: the shipped one).
        protocols: Directory of the protocol files.
        member: Experiments member root that `frozen_inputs` paths are relative to.
        data: The `data/` directory.
        zenodo: Record sets by name (default: `data/zenodo.yaml`).
    """
    known = set(claims) if claims is not None else claim_ids()
    if registry is None:
        registry = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))
    if zenodo is None:
        sets = yaml.safe_load(ZENODO_FILE.read_text(encoding="utf-8"))["sets"]
        zenodo = {s["name"]: s for s in sets}
    problems = [f"{m.id}: claim {c} is not registered" for c in m.claims if c not in known]
    problems += [
        f"{m.id}: code_origin {o!r} lacks the {ORIGIN_PREFIX} prefix"
        for o in m.code_origin
        if not o.startswith(ORIGIN_PREFIX)
    ]
    for s in m.stages:
        try:
            resolve_driver(s.driver)
        except ManifestError as exc:
            problems.append(f"{m.id}: stage {s.name}: {exc}")
        problems += [
            f"{m.id}: stage {s.name} is frozen in {t} but the item has no {t}"
            for t in s.frozen_in
            if m.tiers[t] is None
        ]
    problems += _check_seeds(m, registry)
    problems += _check_files(m, protocols, member)
    problems += _check_records(m, data, zenodo)
    return problems

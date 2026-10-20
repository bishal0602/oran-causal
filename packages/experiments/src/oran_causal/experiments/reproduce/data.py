"""Record sets: `data/record_sets.yaml`, the manifests in `data/manifests` and `data/records`.

`verify` hashes the local record sets against their manifests.
"""

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml

from oran_causal.experiments.paths import WORKSPACE_ROOT
from oran_causal.experiments.reproduce.checks import Check, Status

DATA = WORKSPACE_ROOT / "data"
RECORDS = DATA / "records"
INDEX = DATA / "record_sets.yaml"
CHUNK = 1 << 20
KNOWN_MISSING: dict[str, tuple[str, ...]] = dict.fromkeys(
    ("study1_raw", "study1_train", "study2_raw"), (".pyc",)
)


@dataclasses.dataclass(frozen=True)
class RecordSet:
    """One record set of `record_sets.yaml`.

    Attributes:
        name: Directory name under `data/records`.
        manifest: The sha256 manifest (`<sha256>  <path>` lines, paths relative to the set).
        files: Number of files.
        bytes: Total size.
    """

    name: str
    manifest: Path
    files: int
    bytes: int


def record_sets(index: Path = INDEX) -> dict[str, RecordSet]:
    """The record sets of `record_sets.yaml` by name."""
    doc = yaml.safe_load(index.read_text(encoding="utf-8"))
    return {
        s["name"]: RecordSet(
            s["name"], index.parent / s["manifest"], int(s["files"]), int(s["bytes"])
        )
        for s in doc["sets"]
    }


def read_manifest(path: Path) -> dict[str, str]:
    """{relative path: sha256} of a manifest."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, rel = line.split(maxsplit=1)
            out[rel.strip()] = sha
    return out


def file_sha256(path: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_set(rs: RecordSet, root: Path = RECORDS, ignore: Sequence[str] = ()) -> Check:
    """Hashes every manifest entry of a set under `root/<name>`.

    Args:
        rs: The record set.
        root: Records directory.
        ignore: Suffixes of manifest entries known to be missing (reported, not failed).
    """
    base = root / rs.name
    if not base.is_dir():
        return Check("data", rs.name, Status.SKIP, f"not present under {root}")
    want = read_manifest(rs.manifest)
    missing, bad, excused = [], [], 0
    for rel, sha in want.items():
        path = base / rel
        if not path.exists():
            if rel.endswith(tuple(ignore)):
                excused += 1
            else:
                missing.append(rel)
        elif file_sha256(path) != sha:
            bad.append(rel)
    if missing or bad:
        first = (missing + bad)[0]
        detail = f"{len(missing)} missing, {len(bad)} differ (first {first})"
        return Check("data", rs.name, Status.DIFF, detail)
    note = f"{len(want) - excused} files verified" + (
        f", {excused} known missing" if excused else ""
    )
    return Check("data", rs.name, Status.PASS, note)


def _chosen(names: Iterable[str] | None, index: Path) -> list[RecordSet]:
    sets = record_sets(index)
    chosen = list(sets) if names is None else list(names)
    unknown = sorted(set(chosen) - set(sets))
    if unknown:
        raise KeyError(f"unknown record sets {unknown}; known: {sorted(sets)}")
    return [sets[n] for n in chosen]


def verify(
    names: Iterable[str] | None = None,
    root: Path = RECORDS,
    *,
    known_missing: Mapping[str, Sequence[str]] = KNOWN_MISSING,
    index: Path = INDEX,
) -> list[Check]:
    """Verifies the chosen sets under `root` against their manifests; absent sets are SKIP.

    Args:
        names: Sets to verify (None: every set).
        root: Records directory.
        known_missing: {set: suffixes of entries known to be missing}.
        index: The record set index.

    Raises:
        KeyError: An unknown set name.
    """
    return [verify_set(rs, root, known_missing.get(rs.name, ())) for rs in _chosen(names, index)]

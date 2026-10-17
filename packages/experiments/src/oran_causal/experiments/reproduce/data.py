"""Record sets: `data/zenodo.yaml`, the sha256 manifests in `data/manifests` and `data/records`.

`fetch` verifies the local record sets against their manifests; the Zenodo download is a stub
until the sets have DOIs.
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
ZENODO = DATA / "zenodo.yaml"
CHUNK = 1 << 20
KNOWN_MISSING: dict[str, tuple[str, ...]] = dict.fromkeys(
    ("study1_raw", "study1_train", "study2_raw"), (".pyc",)
)


@dataclasses.dataclass(frozen=True)
class RecordSet:
    """One record set of `zenodo.yaml`.

    Attributes:
        name: Directory name under `data/records`.
        manifest: The sha256 manifest (`<sha256>  <path>` lines, paths relative to the set).
        files: Number of files.
        bytes: Total size.
        doi: Zenodo DOI, None until deposited.
    """

    name: str
    manifest: Path
    files: int
    bytes: int
    doi: str | None


def record_sets(zenodo: Path = ZENODO) -> dict[str, RecordSet]:
    """The record sets of `zenodo.yaml` by name."""
    doc = yaml.safe_load(zenodo.read_text(encoding="utf-8"))
    return {
        s["name"]: RecordSet(
            s["name"], zenodo.parent / s["manifest"], int(s["files"]), int(s["bytes"]), s["doi"]
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


def download(rs: RecordSet, root: Path) -> Check:
    """Zenodo download of a set (stub: no set has a DOI yet)."""
    if rs.doi is None:
        return Check(
            "data", rs.name, Status.SKIP, "no Zenodo DOI yet; place the set under " + str(root)
        )
    return Check("data", rs.name, Status.ERROR, f"download of {rs.doi} not implemented")


def fetch(
    names: Iterable[str] | None = None,
    root: Path = RECORDS,
    *,
    verify_only: bool = False,
    known_missing: Mapping[str, Sequence[str]] = KNOWN_MISSING,
    zenodo: Path = ZENODO,
) -> list[Check]:
    """Downloads absent sets (stub) and verifies the present ones against their manifests.

    Args:
        names: Sets to handle (None: every set).
        root: Records directory.
        verify_only: Never download.
        known_missing: {set: suffixes of entries known to be missing}.
        zenodo: The record set index.

    Raises:
        KeyError: An unknown set name.
    """
    sets = record_sets(zenodo)
    chosen = list(sets) if names is None else list(names)
    unknown = sorted(set(chosen) - set(sets))
    if unknown:
        raise KeyError(f"unknown record sets {unknown}; known: {sorted(sets)}")
    out = []
    for name in chosen:
        rs = sets[name]
        if not (root / name).is_dir() and not verify_only:
            out.append(download(rs, root))
            continue
        out.append(verify_set(rs, root, known_missing.get(name, ())))
    return out

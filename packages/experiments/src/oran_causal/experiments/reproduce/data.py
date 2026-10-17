"""Record sets: `data/zenodo.yaml`, the sha256 manifests in `data/manifests` and `data/records`.

`fetch` verifies the local record sets against their manifests and downloads absent ones from
their Zenodo record; `pack` builds the deterministic `<set>.tar.zst` archives for the deposit.
"""

import dataclasses
import hashlib
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from compression.zstd import CompressionParameter
from pathlib import Path

import yaml

from oran_causal.experiments.paths import WORKSPACE_ROOT
from oran_causal.experiments.reproduce.checks import Check, Status

DATA = WORKSPACE_ROOT / "data"
RECORDS = DATA / "records"
ZENODO = DATA / "zenodo.yaml"
CHUNK = 1 << 20
ZENODO_RECORDS = "https://zenodo.org/records"
DOI_PREFIX = "10.5281/zenodo."
ZSTD_OPTIONS: Mapping[int, int] = {CompressionParameter.compression_level: 10}
SHA256SUMS = "SHA256SUMS"
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
        doi: Zenodo DOI of the record holding the archive, None until deposited.
        archive: File name of the set's archive inside the record.
        archive_sha256: sha256 of the archive, None until it is deposited.
    """

    name: str
    manifest: Path
    files: int
    bytes: int
    doi: str | None
    archive: str
    archive_sha256: str | None


def record_sets(zenodo: Path = ZENODO) -> dict[str, RecordSet]:
    """The record sets of `zenodo.yaml` by name."""
    doc = yaml.safe_load(zenodo.read_text(encoding="utf-8"))
    return {
        s["name"]: RecordSet(
            s["name"],
            zenodo.parent / s["manifest"],
            int(s["files"]),
            int(s["bytes"]),
            s["doi"],
            s.get("archive", f"{s['name']}.tar.zst"),
            s.get("archive_sha256"),
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


def archive_url(rs: RecordSet, base_url: str = ZENODO_RECORDS) -> str:
    """URL of a set's archive inside its Zenodo record.

    Args:
        rs: The record set; its DOI must be `10.5281/zenodo.<id>`.
        base_url: Records endpoint (a `file://` directory works offline).

    Raises:
        ValueError: The set has no Zenodo DOI of that form.
    """
    if rs.doi is None or not rs.doi.startswith(DOI_PREFIX):
        raise ValueError(f"{rs.name}: no Zenodo DOI of the form {DOI_PREFIX}<id>")
    url = f"{base_url.rstrip('/')}/{rs.doi.removeprefix(DOI_PREFIX)}/files/{rs.archive}"
    return url if base_url.startswith("file:") else f"{url}?download=1"


def _fetch_archive(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh, CHUNK)


def download(rs: RecordSet, root: Path, base_url: str = ZENODO_RECORDS) -> Check:
    """Downloads a set's archive from its Zenodo record and extracts it to `root/<name>`.

    The archive is checked against `archive_sha256` before extraction and extracted with the
    `data` filter into a scratch directory beside `root`, so a failure leaves `root` untouched.
    The caller still verifies the extracted files against the set's manifest.

    Args:
        rs: The record set.
        root: Records directory.
        base_url: Records endpoint.

    Returns:
        SKIP without a DOI; ERROR on a failed transfer, a missing pin or an unsafe archive;
        DIFF on a checksum mismatch; PASS once extracted.
    """
    if rs.doi is None:
        return Check(
            "data", rs.name, Status.SKIP, "no Zenodo DOI yet; place the set under " + str(root)
        )
    if rs.archive_sha256 is None:
        return Check("data", rs.name, Status.ERROR, "zenodo.yaml pins no archive_sha256")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".fetch-") as scratch:
        staged = _stage(rs, base_url, Path(scratch))
        if isinstance(staged, Check):
            return staged
        shutil.move(staged / rs.name, root / rs.name)
    return Check("data", rs.name, Status.PASS, f"extracted {rs.archive}")


def _stage(rs: RecordSet, base_url: str, scratch: Path) -> Path | Check:
    archive = scratch / rs.archive
    try:
        _fetch_archive(archive_url(rs, base_url), archive)
    except (OSError, ValueError) as exc:
        return Check("data", rs.name, Status.ERROR, f"download failed: {exc}")
    got = file_sha256(archive)
    if got != rs.archive_sha256:
        detail = f"{rs.archive} sha256 {got}, expected {rs.archive_sha256}"
        return Check("data", rs.name, Status.DIFF, detail)
    staged = scratch / "extracted"
    try:
        with tarfile.open(archive, "r:zst") as tar:
            tar.extractall(staged, filter="data")
    except (tarfile.TarError, OSError) as exc:
        return Check("data", rs.name, Status.ERROR, f"unsafe or unreadable archive: {exc}")
    if not (staged / rs.name).is_dir():
        return Check("data", rs.name, Status.ERROR, f"archive has no {rs.name}/ directory")
    return staged


def build_archive(
    rs: RecordSet, out: Path, root: Path = RECORDS, ignore: Sequence[str] = ()
) -> int:
    """Writes the deterministic `tar.zst` of a set: sorted manifest entries as `<name>/<rel>`.

    Args:
        rs: The record set.
        out: Archive path.
        root: Records directory.
        ignore: Suffixes of manifest entries known to be missing (left out).

    Returns:
        The number of members written.

    Raises:
        FileNotFoundError: A manifest entry that is not excused is absent.
    """
    members = 0
    with tarfile.open(out, "w:zst", options=ZSTD_OPTIONS) as tar:
        for rel in sorted(read_manifest(rs.manifest)):
            src = root / rs.name / rel
            if not src.exists():
                if rel.endswith(tuple(ignore)):
                    continue
                raise FileNotFoundError(src)
            info = tar.gettarinfo(src, arcname=f"{rs.name}/{rel}")
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            info.mode = 0o644
            with src.open("rb") as fh:
                tar.addfile(info, fh)
            members += 1
    return members


def verify_archive(rs: RecordSet, archive: Path, ignore: Sequence[str] = ()) -> Check:
    """Streams an archive back and hashes every member against the set's manifest.

    Args:
        rs: The record set.
        archive: A `tar.zst` built by `build_archive`.
        ignore: Suffixes of manifest entries known to be missing.
    """
    want = read_manifest(rs.manifest)
    seen: dict[str, str] = {}
    with tarfile.open(archive, "r:zst") as tar:
        for info in tar:
            member = tar.extractfile(info)
            if member is None:
                continue
            h = hashlib.sha256()
            with member:
                while chunk := member.read(CHUNK):
                    h.update(chunk)
            seen[info.name.removeprefix(rs.name + "/")] = h.hexdigest()
    missing = [r for r in want if r not in seen and not r.endswith(tuple(ignore))]
    bad = [r for r, sha in seen.items() if want.get(r) != sha]
    if missing or bad:
        first = (missing + bad)[0]
        detail = f"{len(missing)} missing, {len(bad)} differ (first {first})"
        return Check("data", rs.name, Status.DIFF, detail)
    return Check("data", rs.name, Status.PASS, f"{len(seen)} members verified")


def pack(
    names: Iterable[str] | None,
    out: Path,
    root: Path = RECORDS,
    *,
    known_missing: Mapping[str, Sequence[str]] = KNOWN_MISSING,
    zenodo: Path = ZENODO,
) -> list[Check]:
    """Builds and verifies the deposit files of the chosen sets under `out`.

    Writes `<set>.tar.zst`, a copy of `<set>.sha256` and `SHA256SUMS` (archive hashes).

    Args:
        names: Sets to pack (None: every set).
        out: Output directory, created if absent.
        root: Records directory.
        known_missing: {set: suffixes of entries known to be missing}.
        zenodo: The record set index.

    Raises:
        KeyError: An unknown set name.
    """
    sets = _chosen(names, zenodo)
    out.mkdir(parents=True, exist_ok=True)
    rows, sums = [], []
    for rs in sets:
        ignore = known_missing.get(rs.name, ())
        archive = out / rs.archive
        build_archive(rs, archive, root, ignore)
        shutil.copyfile(rs.manifest, out / f"{rs.name}.sha256")
        sums.append(f"{file_sha256(archive)}  {rs.archive}")
        rows.append(verify_archive(rs, archive, ignore))
    (out / SHA256SUMS).write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    return rows


def _chosen(names: Iterable[str] | None, zenodo: Path) -> list[RecordSet]:
    sets = record_sets(zenodo)
    chosen = list(sets) if names is None else list(names)
    unknown = sorted(set(chosen) - set(sets))
    if unknown:
        raise KeyError(f"unknown record sets {unknown}; known: {sorted(sets)}")
    return [sets[n] for n in chosen]


def fetch(
    names: Iterable[str] | None = None,
    root: Path = RECORDS,
    *,
    verify_only: bool = False,
    known_missing: Mapping[str, Sequence[str]] = KNOWN_MISSING,
    zenodo: Path = ZENODO,
    base_url: str = ZENODO_RECORDS,
) -> list[Check]:
    """Downloads absent sets from Zenodo and verifies the present ones against their manifests.

    Args:
        names: Sets to handle (None: every set).
        root: Records directory.
        verify_only: Never download.
        known_missing: {set: suffixes of entries known to be missing}.
        zenodo: The record set index.
        base_url: Zenodo records endpoint.

    Raises:
        KeyError: An unknown set name.
    """
    out = []
    for rs in _chosen(names, zenodo):
        if not (root / rs.name).is_dir() and not verify_only:
            fetched = download(rs, root, base_url)
            if fetched.status is not Status.PASS:
                out.append(fetched)
                continue
        out.append(verify_set(rs, root, known_missing.get(rs.name, ())))
    return out

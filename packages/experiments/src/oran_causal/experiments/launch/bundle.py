"""The code bundle a cloud session runs: a zip of the files a run needs plus a MANIFEST."""

import dataclasses
import hashlib
import io
import json
import subprocess
import tarfile
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.experiments.campaign.provenance import GitRunner, code_info, run_git
from oran_causal.experiments.campaign.spec import ProtocolTexts
from oran_causal.experiments.launch import fingerprint
from oran_causal.experiments.paths import STUDY4_PROTOCOL, WORKSPACE_ROOT

MANIFEST = "MANIFEST.json"
DOT = "_dot_"
FROZEN_COPY_REL = "_bundle/PROTOCOL.frozen.md"
PROTOCOL_REL = STUDY4_PROTOCOL.relative_to(WORKSPACE_ROOT).as_posix()
CORE_FILES = ("pyproject.toml", "uv.lock", ".python-version")
RUN_MEMBERS = ("core", "sim", "methods", "analysis", "experiments")
CORE_GLOBS = (
    *(f"packages/{m}/src/**/*" for m in RUN_MEMBERS),
    "packages/*/pyproject.toml",
    "packages/experiments/protocols/**/*",
    "packages/experiments/manifests/**/*",
    "packages/experiments/artifacts/**/*",
)
TEXT_SUFFIXES = (".py", ".md", ".json", ".txt", ".sh", ".cfg", ".toml", ".yaml", ".yml", ".lock")
SKIP_PARTS = ("__pycache__",)
SKIP_SUFFIXES = (".pyc", ".pyo")

type BlobReader = Callable[[str, str], bytes | None]
type Fingerprinter = Callable[[], dict[str, Any]]


def sha256(data: bytes) -> str:
    """Hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def git_blob(commit: str, rel: str, root: Path = WORKSPACE_ROOT) -> bytes | None:
    """Bytes of `rel` at `commit` (`git show commit:rel`), or None if git cannot give them."""
    try:
        p = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{rel}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return p.stdout if p.returncode == 0 else None


def encode_dots(rel: str) -> str:
    """Kaggle drops dot-directories and dot-files from datasets: a leading `.` becomes `_dot_`."""
    return "/".join(DOT + c[1:] if c.startswith(".") else c for c in rel.split("/"))


def decode_dots(rel: str) -> str:
    """Inverse of `encode_dots`."""
    return "/".join("." + c[len(DOT) :] if c.startswith(DOT) else c for c in rel.split("/"))


def _keep(path: Path) -> bool:
    return path.is_file() and not (
        any(part in SKIP_PARTS for part in path.parts) or path.suffix in SKIP_SUFFIXES
    )


def _expand(root: Path, entry: str) -> Iterable[Path]:
    path = (root / entry).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"bundle path outside the workspace: {entry}")
    if path.is_dir():
        return (p for p in sorted(path.rglob("*")) if _keep(p))
    if path.is_file():
        return (path,)
    raise FileNotFoundError(f"bundle path not found: {entry}")


def collect(root: Path, extra: Iterable[str] = ()) -> dict[str, Path]:
    """Bundle path -> local file for the base set plus the `extra` files or directories.

    The base set is every member's `src`, the member `pyproject.toml` files, the root
    `pyproject.toml`, `uv.lock`, `.python-version`, the protocols, the seed manifest and the
    frozen artifacts. Extras are workspace-relative (a directory adds its files recursively).
    """
    base = root.resolve()
    files: dict[str, Path] = {}
    for name in CORE_FILES:
        if (base / name).is_file():
            files[name] = base / name
    for pattern in CORE_GLOBS:
        for p in sorted(base.glob(pattern)):
            if _keep(p):
                files[p.relative_to(base).as_posix()] = p
    for entry in extra:
        for p in _expand(base, entry):
            files[p.resolve().relative_to(base).as_posix()] = p
    return files


def read_blob(rel: str, path: Path) -> bytes:
    """File bytes, with CRLF turned into LF for text files (binary files are never touched)."""
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if rel.endswith(TEXT_SUFFIXES) else data


@dataclasses.dataclass(frozen=True)
class BundleRequest:
    """What goes into one bundle.

    Attributes:
        name: Job name (the bundle is `<name>_bundle.zip`).
        root: Workspace root.
        extra: Workspace-relative files or directories added to the base set (spec, cost table,
            earlier shards to skip, tests for a selftest).
        frozen_protocol: Verified frozen protocol text for the cloud-side EVAL guard, stored at
            `FROZEN_COPY_REL`.
        cmd: The session command, recorded in the manifest.
        sources: Kaggle kernel sources the session reads, recorded in the manifest.
        with_fingerprint: Record the local numeric fingerprint (a 300 s accept-all episode).
        markers: Extra top-level members written verbatim (name -> bytes).
        dot_encode: Encode leading dots of path components (Kaggle datasets).
    """

    name: str
    root: Path = WORKSPACE_ROOT
    extra: Sequence[str] = ()
    frozen_protocol: bytes | None = None
    cmd: str | None = None
    sources: Sequence[str] = ()
    with_fingerprint: bool = False
    markers: Mapping[str, bytes] = dataclasses.field(default_factory=dict)
    dot_encode: bool = False


@dataclasses.dataclass(frozen=True)
class Bundle:
    """A built bundle.

    Attributes:
        zip_path: The archive.
        manifest: The MANIFEST it carries.
        files: Bundle paths of the payload (excluding MANIFEST and markers).
    """

    zip_path: Path
    manifest: dict[str, Any]
    files: tuple[str, ...]


def build_blobs(req: BundleRequest) -> dict[str, bytes]:
    """Payload bytes by bundle path: collected files, LF-normalised, plus the frozen copy."""
    blobs = {rel: read_blob(rel, p) for rel, p in collect(req.root, req.extra).items()}
    if req.frozen_protocol is not None:
        blobs[FROZEN_COPY_REL] = req.frozen_protocol
    return blobs


def build_manifest(
    req: BundleRequest,
    blobs: Mapping[str, bytes],
    git: GitRunner | None = None,
    fingerprinter: Fingerprinter = fingerprint.fingerprint,
) -> dict[str, Any]:
    """MANIFEST of a bundle.

    `git_head` and `dirty` come from `code_info` on the workspace; `local_env` is the numeric
    environment of this machine and `local_fp` its fingerprint when requested.
    """
    code = code_info({}, git or (lambda args: run_git(args, req.root)), req.root)
    man: dict[str, Any] = {
        "git_head": code["commit"],
        "dirty": code["dirty"],
        "dirty_paths": code["dirty_paths"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmd": req.cmd,
        "sources": list(req.sources),
        "sha256": {rel: sha256(b) for rel, b in sorted(blobs.items())},
        "local_env": fingerprint.numeric_env(),
    }
    if req.with_fingerprint:
        man["local_fp"] = fingerprinter()
        man["local_env"] = man["local_fp"]["env"]
    return man


def write_zip(
    zip_path: Path, blobs: Mapping[str, bytes], manifest: Mapping[str, Any], req: BundleRequest
) -> None:
    """Writes the payload, the markers and the manifest into a deflated zip."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in blobs.items():
            z.writestr(encode_dots(rel) if req.dot_encode else rel, data)
        for name, data in req.markers.items():
            z.writestr(name, data)
        z.writestr(MANIFEST, json.dumps(manifest, indent=1))


def tar_bytes(blobs: Mapping[str, bytes], manifest: Mapping[str, Any]) -> bytes:
    """The payload and the manifest as a gzipped tar (the SSH lane's transfer format)."""
    now = int(time.time())
    buf = io.BytesIO()
    members = {**blobs, MANIFEST: json.dumps(manifest, indent=1).encode()}
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, data in members.items():
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode = len(data), now, 0o644
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def build_bundle(
    req: BundleRequest,
    out_dir: Path,
    git: GitRunner | None = None,
    fingerprinter: Fingerprinter = fingerprint.fingerprint,
) -> Bundle:
    """Builds `<out_dir>/<name>_bundle.zip` from the workspace."""
    blobs = build_blobs(req)
    man = build_manifest(req, blobs, git, fingerprinter)
    zip_path = out_dir / f"{req.name}_bundle.zip"
    write_zip(zip_path, blobs, man, req)
    return Bundle(zip_path, man, tuple(sorted(blobs)))


def frozen_protocol(
    spec: Mapping[str, Any], root: Path = WORKSPACE_ROOT, reader: BlobReader | None = None
) -> bytes | None:
    """The protocol text to bundle for an EVAL guard.

    The text at the spec's `freeze_commit` when git has it, else the live protocol file.
    """
    read = reader or (lambda commit, rel: git_blob(commit, rel, root))
    commit = spec.get("freeze_commit")
    frozen = read(str(commit), PROTOCOL_REL) if commit else None
    if frozen is not None:
        return frozen
    live = root / PROTOCOL_REL
    return live.read_bytes() if live.exists() else None


def bundled_texts(root: Path) -> ProtocolTexts:
    """`ProtocolTexts` of an unpacked bundle: its protocol file and its frozen copy."""
    live, copy = root / PROTOCOL_REL, root / FROZEN_COPY_REL
    return ProtocolTexts(
        live=live.read_bytes() if live.exists() else None,
        frozen_copy=copy.read_bytes() if copy.exists() else None,
    )

"""Code commit, interpreter, packages and lock pins of the process that writes records."""

import dataclasses
import importlib.metadata as md
import json
import os
import platform
import subprocess
import tomllib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from oran_causal.experiments.campaign.spec import sha_lf
from oran_causal.experiments.paths import WORKSPACE_ROOT

COMMIT_ENV = "ORAN_CAUSAL_CODE_COMMIT"
DIRTY_ENV = "ORAN_CAUSAL_CODE_DIRTY"
LOCK_ENV = "ORAN_CAUSAL_LOCK_FILE"
PYTHON_VERSION_FILE = ".python-version"
CODE_PREFIXES = ("packages/",)
SPEC_DIR = "packages/experiments/specs/"
BUNDLE_MANIFESTS = ("MANIFEST.json", "JOB_MANIFEST.json")
PKGS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "causal-learn",
    "xgboost",
    "shap",
    "torch",
    "threadpoolctl",
    "tigramite",
    "dcor",
    "momentchi2",
)

type GitRunner = Callable[[list[str]], str | None]


def _porcelain_path(line: str) -> str:
    return line[3:].strip().strip('"').split(" -> ")[-1]


def _counts_as_dirt(flag: str, path: str) -> bool:
    if "__pycache__" in path or path.endswith((".pyc", ".pyo")):
        return False
    if flag not in ("??", "!!"):
        return True
    code = path.endswith(".py") and path.startswith(CODE_PREFIXES)
    return code or path.startswith(SPEC_DIR)


def dirty_paths(porcelain: str) -> list[str]:
    """Paths of `git status --porcelain --untracked-files=all --ignored=matching` that are dirt.

    Any tracked change counts; untracked or ignored files count only when they are Python code
    under `packages/` or a campaign spec. Caches never count.
    """
    out = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        flag, path = line[:2], _porcelain_path(line)
        if _counts_as_dirt(flag, path):
            out.append(path)
    return out


def run_git(args: list[str], root: Path = WORKSPACE_ROOT) -> str | None:
    """Output of `git -C root <args>`, or None without git or on failure."""
    try:
        p = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError, subprocess.SubprocessError:
        return None
    return p.stdout if p.returncode == 0 else None


def _env_code(env: Mapping[str, str]) -> dict[str, Any] | None:
    commit = env.get(COMMIT_ENV)
    if not commit:
        return None
    d = env.get(DIRTY_ENV)
    return {
        "commit": commit,
        "dirty": None if d is None else d == "1",
        "source": "env (launcher git check)",
        "dirty_paths": None,
    }


def _git_code(git: GitRunner) -> dict[str, Any] | None:
    head = (git(["rev-parse", "HEAD"]) or "").strip()
    if not head:
        return None
    status = git(
        ["status", "--porcelain", "--untracked-files=all", "--ignored=matching", "--", "."]
    )
    dp = dirty_paths(status or "")
    return {"commit": head, "dirty": bool(dp), "source": "git", "dirty_paths": dp[:20]}


def _bundle_code(root: Path) -> dict[str, Any] | None:
    for name in BUNDLE_MANIFESTS:
        p = root / name
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            return {
                "commit": m.get("git_head"),
                "dirty": m.get("dirty"),
                "source": f"{name} (bundle manifest)",
                "dirty_paths": None,
            }
    return None


def code_info(
    env: Mapping[str, str] | None = None, git: GitRunner = run_git, root: Path = WORKSPACE_ROOT
) -> dict[str, Any]:
    """Commit of the code that produces records.

    Order: the launcher's environment variables, then git, then a bundle manifest.
    """
    found = _env_code(os.environ if env is None else env) or _git_code(git) or _bundle_code(root)
    return found or {"commit": None, "dirty": None, "source": None, "dirty_paths": None}


def lock_versions(text: str) -> dict[str, frozenset[str]]:
    """`{normalised name: locked versions}` of the third-party packages of a `uv.lock`.

    Workspace members (editable or virtual sources) are left out. A package locked at several
    versions (per platform or extra, e.g. the torch builds) keeps all of them.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for p in tomllib.loads(text).get("package", []):
        src = p.get("source", {})
        if "version" in p and "editable" not in src and "virtual" not in src:
            out[canonicalize_name(p["name"])].add(p["version"])
    return {n: frozenset(v) for n, v in out.items()}


def is_torch_build(name: str) -> bool:
    """Packages of the torch build: torch, triton and the CUDA 12 runtime wheels."""
    return name in ("torch", "triton") or (name.startswith("nvidia-") and name.endswith("-cu12"))


def installed_versions(dists: Iterable[md.Distribution] | None = None) -> dict[str, str]:
    """Installed distributions; the first copy on `sys.path` wins, as for imports."""
    out: dict[str, str] = {}
    for d in md.distributions() if dists is None else dists:
        if d.metadata["Name"]:
            out.setdefault(canonicalize_name(d.metadata["Name"]), d.version)
    return out


def _mismatch(name: str, locked: frozenset[str], have: str | None) -> bool:
    if have is None or (is_torch_build(name) and name != "torch"):
        return False
    if name == "torch":
        return Version(have).public not in {Version(v).public for v in locked}
    return all(Version(have) != Version(v) for v in locked)


@dataclasses.dataclass(frozen=True)
class LockFile:
    """A `uv.lock` and its display name."""

    text: str
    name: str
    sha256: str


def read_lock(path: Path | None = None, root: Path = WORKSPACE_ROOT) -> LockFile | None:
    """Reads `path`, else `ORAN_CAUSAL_LOCK_FILE`, else the workspace `uv.lock`; None if absent."""
    env = os.environ.get(LOCK_ENV)
    target = path or (Path(env) if env else root / "uv.lock")
    if not target.exists():
        return None
    data = target.read_bytes()
    return LockFile(data.decode("utf-8"), target.as_posix(), sha_lf(data))


@dataclasses.dataclass(frozen=True)
class PythonPin:
    """The workspace interpreter pin.

    Attributes:
        version: `.python-version` (a release prefix such as `3.14`).
        requires: `requires-python` of the lock.
    """

    version: str
    requires: str

    def admits(self, python: str) -> bool:
        """Whether interpreter version `python` matches the pin and `requires-python`."""
        pin, have = Version(self.version).release, Version(python)
        return have.release[: len(pin)] == pin and have in SpecifierSet(self.requires)

    def __str__(self) -> str:
        return f"{self.version} ({self.requires})"


def workspace_python(root: Path = WORKSPACE_ROOT) -> PythonPin:
    """The pin of `root`: its `.python-version` and the `requires-python` of its `uv.lock`."""
    version = (root / PYTHON_VERSION_FILE).read_text(encoding="utf-8").strip()
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    return PythonPin(version, lock.get("requires-python", ""))


def pin_check(
    lock: LockFile | None, python: PythonPin | None = None, installed: dict[str, str] | None = None
) -> dict[str, Any]:
    """Installed versions against a `uv.lock`.

    Every installed package the lock names must have a locked version, except the torch build:
    torch needs a locked public version in any build, and triton / nvidia-*-cu12 are not
    checked. Locked packages that are not installed are not checked (platform markers and
    extras decide them). With `python`, an interpreter the pin does not admit is a `python`
    mismatch.

    Returns:
        `lock`, `lock_sha256`, `n_pins`, `mismatches` (name -> [locked, installed], None
        without a lock), `torch_build`, `python`, `python_required`, `torch_build_exempt`.
    """
    inst = installed_versions() if installed is None else installed
    py = platform.python_version()
    if lock is None:
        return {
            "lock": None,
            "n_pins": 0,
            "mismatches": None,
            "torch_build": inst.get("torch"),
            "python": py,
        }
    pins = lock_versions(lock.text)
    bad: dict[str, list[str | None]] = {
        n: [" | ".join(sorted(v)), inst.get(n)]
        for n, v in pins.items()
        if _mismatch(n, v, inst.get(n))
    }
    if python is not None and not python.admits(py):
        bad["python"] = [str(python), py]
    return {
        "lock": lock.name,
        "lock_sha256": lock.sha256,
        "n_pins": len(pins),
        "mismatches": bad,
        "torch_build": inst.get("torch"),
        "python": py,
        "python_required": None if python is None else str(python),
        "torch_build_exempt": {
            n: inst.get(n) for n in pins if is_torch_build(n) and n != "torch" and n in inst
        },
    }


def pkg_versions() -> dict[str, str | None]:
    """Versions of the study's numerical packages (None when not installed)."""
    out: dict[str, str | None] = {}
    for p in PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out

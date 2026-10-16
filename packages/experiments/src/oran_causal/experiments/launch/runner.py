"""Injectable process runner and clock, so launchers are testable without a network."""

import dataclasses
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from oran_causal.experiments.paths import WORKSPACE_ROOT

RUNS_DIR = WORKSPACE_ROOT / "records" / "runs"

type Runner = Callable[[Sequence[str], Path | None, float], str]


def run_process(
    argv: Sequence[str],
    cwd: Path | None = None,
    timeout: float = 900,
    env: Mapping[str, str] | None = None,
) -> str:
    """Runs `argv` and returns stdout and stderr joined and stripped.

    The text is decoded as UTF-8 with replacement: a CLI's progress output under Task Scheduler
    can hold bytes the console code page cannot decode.
    """
    p = subprocess.run(
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})},
        check=False,
    )
    return ((p.stdout or "") + (p.stderr or "")).strip()


@dataclasses.dataclass(frozen=True)
class Clock:
    """Wall clock and sleep, replaceable in tests."""

    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep


@dataclasses.dataclass(frozen=True)
class Dirs:
    """Local directories of a launch.

    Attributes:
        work: Where bundles and metadata are staged (one subdirectory per job).
        runs: Where pulled results and launch records go (one subdirectory per job).
        root: The workspace the bundle is built from.
    """

    work: Path
    runs: Path = RUNS_DIR
    root: Path = WORKSPACE_ROOT

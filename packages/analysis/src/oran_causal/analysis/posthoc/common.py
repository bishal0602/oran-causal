"""Helpers shared by the post hoc analyses and their drivers."""

import dataclasses
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.system import stats, study3


def js_default(x: Any) -> Any:
    """`json.dump` default: NumPy scalars and arrays as Python values, anything else as text."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def ci_close(ci: Sequence[float | None], expected: Sequence[float], rtol: float = 1e-9) -> bool:
    """Whether an interval equals the expected one at `rtol`; a missing bound never does."""
    return bool(
        np.allclose(np.asarray(ci, dtype=float), np.asarray(expected, dtype=float), rtol=rtol)
    )


def sub_filter(sub: str) -> stats.JobFilter:
    """Job filter of the EVAL records a post hoc driver writes with the certified-safe schema.

    Headers of the stage are kept whatever their driver, so a run directory is read as a whole.
    """
    return dataclasses.replace(study3.job_filter("eval"), sub=sub, driver=None)


def shard_files(root: Path, pattern: str = "res_*.jsonl") -> list[Path]:
    """The shard files under `root` (any depth), in path order."""
    return sorted(root.rglob(pattern))


def read_json(path: Path) -> Any:
    """Parses a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def alias_filled(
    files: Iterable[Path], filt: stats.JobFilter, alias_of: dict[str, str]
) -> tuple[stats.Records, list[dict[str, Any]]]:
    """Records of the shards with aliased arms filled from their targets."""
    R, heads = stats.load_jobs(files, filt)
    stats.alias_fill(R, alias_of)
    return R, heads

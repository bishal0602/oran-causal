"""Read-only loaders for the frozen record sets, plus small statistics helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from oran_causal.figures.paths import record


def load_json(*parts: str):
    """Parse a JSON file under data/records."""
    return json.loads(record(*parts).read_text(encoding="utf-8"))


def load_csv(*parts: str, **kw) -> pd.DataFrame:
    """Read a CSV file under data/records into a DataFrame."""
    return pd.read_csv(record(*parts), **kw)


def mean_ci(
    samples, axis: int = 0, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and two-sided t-interval over `axis` (e.g. seeds). Returns (mean, lo, hi)."""
    a = np.asarray(samples, dtype=float)
    n = a.shape[axis]
    m = a.mean(axis=axis)
    if n < 2:
        return m, m, m
    half = stats.sem(a, axis=axis) * stats.t.ppf((1 + level) / 2, n - 1)
    return m, m - half, m + half


def source_note(*parts: str) -> str:
    """Workspace-relative record path for figure provenance comments."""
    return str(Path("data", "records").joinpath(*parts).as_posix())


def protocol_note(*parts: str) -> str:
    """Workspace-relative frozen-protocol path for figure provenance comments."""
    return str(Path("packages", "experiments", "protocols").joinpath(*parts).as_posix())

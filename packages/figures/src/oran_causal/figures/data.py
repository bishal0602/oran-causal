"""Read-only loaders for the frozen record sets, plus small statistics helpers."""

import json
from pathlib import Path

import pandas as pd

from oran_causal.figures.paths import record, resolve


def load_json(*parts: str):
    """Parse a JSON file under data/records."""
    return json.loads(record(*parts).read_text(encoding="utf-8"))


def load_csv(*parts: str, **kw) -> pd.DataFrame:
    """Read a CSV file under data/records into a DataFrame."""
    return pd.read_csv(record(*parts), **kw)


def source_note(*parts: str) -> str:
    """Workspace-relative path of the record file `record` reads, for provenance comments."""
    return Path("data", "records", *resolve(*parts)).as_posix()


def protocol_note(*parts: str) -> str:
    """Workspace-relative frozen-protocol path for figure provenance comments."""
    return str(Path("packages", "experiments", "protocols").joinpath(*parts).as_posix())

"""Filesystem layout. Figures read the frozen record sets read-only and write PDFs and tables for
the manuscript in ``paper/``.
"""

import os
from pathlib import Path

MEMBER_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = MEMBER_ROOT.parents[1]
RECORDS = WORKSPACE_ROOT / "data" / "records"
PROTOCOLS = WORKSPACE_ROOT / "packages" / "experiments" / "protocols"
PAPER = WORKSPACE_ROOT / "paper"
PREVIEW = MEMBER_ROOT / "preview"
DATA = MEMBER_ROOT / "data"
OUT_ENV = "ORAN_CAUSAL_FIGURES_OUT"


def _out_root() -> Path | None:
    value = os.environ.get(OUT_ENV)
    return Path(value) if value else None


def figures_dir() -> Path:
    r"""Directory of the LaTeX ``\includegraphics`` PDFs: the manuscript's ``paper/figures`` unless redirected."""
    root = _out_root()
    return root / "figures" if root else PAPER / "figures"


def tables_dir() -> Path:
    """Directory of the booktabs ``.tex`` and ``.csv`` tables: the manuscript's ``paper/tables`` unless redirected."""
    root = _out_root()
    return root / "tables" if root else PAPER / "tables"


def preview_dir() -> Path:
    """Directory of the 300 dpi review PNGs; never referenced by LaTeX."""
    root = _out_root()
    return root / "preview" if root else PREVIEW


def record(*parts: str) -> Path:
    """Path inside ``data/records``; raises if it does not exist (figures never invent data)."""
    path = RECORDS.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"record missing: {path}")
    return path


def protocol(*parts: str) -> Path:
    """Path inside the frozen protocols of the experiments member; raises if it does not exist."""
    path = PROTOCOLS.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"protocol missing: {path}")
    return path

"""Reads the `label{spec:...}` anchors of the LaTeX specification."""

import re
from pathlib import Path

LABEL = re.compile(r"\\label\{(spec:[a-z0-9.]+)\}")


def spec_labels(sections: Path) -> frozenset[str] | None:
    """Collect every spec label defined under a directory of `.tex` files.

    Returns:
        The labels, or `None` when the directory holds no `.tex` file.
    """
    files = sorted(sections.glob("*.tex")) if sections.is_dir() else []
    if not files:
        return None
    return frozenset(m for f in files for m in LABEL.findall(f.read_text(encoding="utf-8")))

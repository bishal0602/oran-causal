"""Pass/diff rows of the reproduce tiers and the comparisons that produce them."""

import dataclasses
import enum
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


class Status(enum.StrEnum):
    """Outcome of one check."""

    PASS = "pass"
    DIFF = "diff"
    SKIP = "skip"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class Check:
    """One row of a tier's result table.

    Attributes:
        item: The tier item (study or analysis).
        target: The file or rule checked.
        status: The outcome.
        detail: What differed, why it was skipped, or a short note.
    """

    item: str
    target: str
    status: Status
    detail: str = ""


def passed(checks: Iterable[Check]) -> bool:
    """True when no check differs or failed (skips pass)."""
    return all(c.status in (Status.PASS, Status.SKIP) for c in checks)


def table(checks: Sequence[Check]) -> str:
    """A markdown table of the checks."""
    rows = ["| Item | Target | Status | Detail |", "|---|---|---|---|"]
    rows += [f"| {c.item} | {c.target} | {c.status} | {c.detail} |" for c in checks]
    return "\n".join(rows)


def lf(data: bytes) -> bytes:
    """`data` with CRLF line ends as LF."""
    return data.replace(b"\r\n", b"\n")


def compare_bytes(item: str, mine: Path, stored: Path, *, eol: bool = False) -> Check:
    """Byte comparison of a regenerated file with the stored one.

    Args:
        item: The tier item.
        mine: The regenerated file.
        stored: The stored file.
        eol: Compare with CRLF read as LF (stored files written in Windows text mode).
    """
    if not stored.exists():
        return Check(item, stored.name, Status.ERROR, f"stored file missing: {stored}")
    if not mine.exists():
        return Check(item, stored.name, Status.ERROR, "not regenerated")
    a, b = mine.read_bytes(), stored.read_bytes()
    if eol:
        a, b = lf(a), lf(b)
    if a == b:
        return Check(item, stored.name, Status.PASS, "bytes equal" + (" (LF)" if eol else ""))
    return Check(item, stored.name, Status.DIFF, f"{len(a)} vs {len(b)} bytes")


def compare_text(item: str, target: str, mine: str, stored: str) -> Check:
    """Text comparison of a regenerated document with the stored one (CRLF read as LF)."""
    a, b = mine.replace("\r\n", "\n"), stored.replace("\r\n", "\n")
    if a == b:
        return Check(item, target, Status.PASS, "text equal")
    return Check(item, target, Status.DIFF, f"{len(a)} vs {len(b)} characters")


def _close_float(a: float, b: float, rtol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def close_tree(a: Any, b: Any, rtol: float, path: str = "") -> list[str]:
    """Paths where two JSON trees differ: floats beyond `rtol`, anything else when not equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in close_tree(a[k], b[k], rtol, f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        pairs = enumerate(zip(a, b, strict=True))
        return [d for i, (x, y) in pairs for d in close_tree(x, y, rtol, f"{path}[{i}]")]
    same = _close_float(a, b, rtol) if isinstance(a, float) and isinstance(b, float) else a == b
    return [] if same else [f"{path}: {a!r} != {b!r}"]


def compare_json_close(
    item: str, mine: Path, stored: Path, rtol: float, skip: Sequence[str] = ()
) -> Check:
    """Parsed JSON comparison, floats to a relative `rtol`, top-level `skip` keys ignored."""
    a = json.loads(mine.read_text(encoding="utf-8"))
    b = json.loads(stored.read_text(encoding="utf-8"))
    diffs = close_tree(
        {k: v for k, v in a.items() if k not in skip},
        {k: v for k, v in b.items() if k not in skip},
        rtol,
    )
    if diffs:
        return Check(item, stored.name, Status.DIFF, f"{len(diffs)} values, first {diffs[0]}")
    note = f"equal to rtol {rtol:g}" + (f", {', '.join(skip)} not compared" if skip else "")
    return Check(item, stored.name, Status.PASS, note)

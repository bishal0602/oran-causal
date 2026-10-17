"""JSON pointers into stored records and the comparison of an `expect` value with what is found."""

import gzip
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

SELECTORS = ("=", "[", "*")


class PointerError(ValueError):
    """A pointer that does not resolve, or a record that cannot be read."""


def is_plain(pointer: str) -> bool:
    """Whether `pointer` is an RFC 6901 pointer without selector characters."""
    return pointer.startswith("/") and not any(mark in pointer for mark in SELECTORS)


def follow(document: Any, pointer: str) -> Any:
    """The value at an RFC 6901 `pointer`.

    Raises:
        PointerError: A step names a missing key or index.
    """
    try:
        for step in pointer.split("/")[1:]:
            part = step.replace("~1", "/").replace("~0", "~")
            document = document[int(part)] if isinstance(document, list) else document[part]
    except KeyError, IndexError, ValueError, TypeError:
        raise PointerError(f"pointer {pointer} does not resolve") from None
    return document


def read_json(path: Path) -> Any:
    """A `.json` or `.json.gz` record.

    Raises:
        PointerError: The file is not JSON of those two kinds.
    """
    if path.name.endswith(".json.gz"):
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
    elif path.suffix == ".json":
        text = path.read_text(encoding="utf-8")
    else:
        raise PointerError(f"{path.name}: only .json and .json.gz records are readable")
    try:
        return json.loads(text)
    except ValueError as error:
        raise PointerError(f"{path.name}: invalid JSON: {error}") from None


def _numbers_agree(found: float, expected: float) -> bool:
    """Equal; or, for a decimal `expected`, equal once `found` is rounded half up to it."""
    if math.isclose(found, expected, rel_tol=1e-9, abs_tol=0.0):
        return True
    if isinstance(expected, int):
        return False
    step = Decimal(repr(expected))
    return Decimal(repr(found)).quantize(step, rounding=ROUND_HALF_UP) == step


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def mismatches(found: Any, expected: Any, path: str = "") -> list[str]:
    """Where `found` does not hold the `expected` value.

    An expected mapping names a subset of the found mapping's keys; numbers agree when equal or
    when `found` rounded to the decimals of `expected` equals it; lists compare element-wise;
    everything else compares equal.
    """
    where = path or "/"
    if isinstance(expected, dict):
        if not isinstance(found, dict):
            return [f"{where}: expected a mapping, found {type(found).__name__}"]
        out: list[str] = []
        for key, value in expected.items():
            if key not in found:
                out.append(f"{path}/{key}: missing")
            else:
                out += mismatches(found[key], value, f"{path}/{key}")
        return out
    if isinstance(expected, list):
        if not isinstance(found, list) or len(found) != len(expected):
            return [f"{where}: expected {expected!r}, found {found!r}"]
        return [
            m
            for i, (f, e) in enumerate(zip(found, expected, strict=True))
            for m in mismatches(f, e, f"{path}[{i}]")
        ]
    if _is_number(expected) and _is_number(found):
        return (
            []
            if _numbers_agree(found, expected)
            else [f"{where}: expected {expected!r}, found {found!r}"]
        )
    return (
        []
        if found == expected and type(found) is type(expected)
        else [f"{where}: expected {expected!r}, found {found!r}"]
    )

"""Named functions over frozen records for printed values that no single pointer reaches.

`spec/values.toml` calls these from an `expr`; each takes record subtrees that the entry's `args`
select and returns a number. They hold no state and read nothing themselves.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

ALPHA = 0.05
SLICE_PREFIX = "s60_"


def flat(*parts: Any) -> list[Any]:
    """Flattens numbers and arbitrarily nested lists into one list."""
    out: list[Any] = []
    for part in parts:
        if isinstance(part, list | tuple):
            out.extend(flat(*part))
        else:
            out.append(part)
    return out


def _null_hypotheses(cells: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [name for name, cell in cells.items() if cell["material"] == "NULL"]


def _unestablished_nulls(cells: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [name for name in _null_hypotheses(cells) if cells[name]["status"] != "EXISTS"]


def _rate_at_alpha(names: Iterable[str], slices: Mapping[str, Mapping[str, Any]]) -> float:
    p_values = [
        slice_["hyp"][name]["primary"]["p"]
        for key, slice_ in slices.items()
        if key.startswith(SLICE_PREFIX)
        for name in names
    ]
    return sum(p <= ALPHA for p in p_values) / len(p_values)


def null_hypothesis_count(cells: Mapping[str, Mapping[str, Any]]) -> int:
    """Number of hypotheses the knockout ground truth labels null (existence-rule cells)."""
    return len(_null_hypotheses(cells))


def unestablished_null_count(cells: Mapping[str, Mapping[str, Any]]) -> int:
    """Null hypotheses whose knockout interval also contains zero."""
    return len(_unestablished_nulls(cells))


def null_rejection_rate(
    cells: Mapping[str, Mapping[str, Any]], slices: Mapping[str, Mapping[str, Any]]
) -> float:
    """Share of primary p-values at or below 0.05 over the null hypotheses and 60-episode slices."""
    return _rate_at_alpha(_null_hypotheses(cells), slices)


def unestablished_null_rejection_rate(
    cells: Mapping[str, Mapping[str, Any]], slices: Mapping[str, Mapping[str, Any]]
) -> float:
    """The same share over the null hypotheses whose knockout interval contains zero."""
    return _rate_at_alpha(_unestablished_nulls(cells), slices)


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "flat": flat,
    "null_hypothesis_count": null_hypothesis_count,
    "unestablished_null_count": unestablished_null_count,
    "null_rejection_rate": null_rejection_rate,
    "unestablished_null_rejection_rate": unestablished_null_rejection_rate,
}

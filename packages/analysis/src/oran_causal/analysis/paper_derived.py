"""Named functions over frozen records for printed values that no single pointer reaches.

`spec/values.toml` calls these from an `expr`; each takes record subtrees that the entry's `args`
select and returns a number. They hold no state and read nothing themselves.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

ALPHA = 0.05
SLICE_PREFIX = "s60_"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 0


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


def _rejection_matrix(names: list[str], slices: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    keys = sorted(
        (k for k in slices if k.startswith(SLICE_PREFIX)), key=lambda k: int(k[len(SLICE_PREFIX) :])
    )
    return np.array(
        [[slices[k]["hyp"][name]["primary"]["p"] <= ALPHA for name in names] for k in keys],
        dtype=float,
    )


def _two_way_bounds(
    names: list[str], slices: Mapping[str, Mapping[str, Any]]
) -> tuple[float, float]:
    hits = _rejection_matrix(names, slices)
    n_slices, n_hyps = hits.shape
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rates = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        rows = rng.integers(0, n_slices, n_slices)
        cols = rng.integers(0, n_hyps, n_hyps)
        rates[i] = hits[np.ix_(rows, cols)].mean()
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def null_rejection_bound(
    cells: Mapping[str, Mapping[str, Any]],
    slices: Mapping[str, Mapping[str, Any]],
    unestablished: bool,
    upper: bool,
) -> float:
    """End of the 95 % percentile interval of the null rejection rate.

    The interval resamples hypotheses and 60-episode subsamples together (two-way bootstrap,
    20 000 draws, fixed seed), as the earlier supplement describes.
    """
    names = _unestablished_nulls(cells) if unestablished else _null_hypotheses(cells)
    return _two_way_bounds(names, slices)[1 if upper else 0]


def high_violation_ratios(
    arms: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
    prefix: str,
    floor: float = 2.0,
) -> list[float]:
    """Protected-slice violations over the accept-all level for the map arms far above it.

    Takes the arms named `prefix...` (`MG:` ungated, `CS:` gated) whose ratio exceeds `floor`.
    """
    ratios = [arm["V"] / gate["V_AA"] for name, arm in arms.items() if name.startswith(prefix)]
    return [ratio for ratio in ratios if ratio > floor]


def nonzero_count(values: Mapping[str, Any]) -> int:
    """Entries of a mapping that are not zero (directions set by the prior)."""
    return sum(value != 0 for value in values.values())


def indeterminate_declared_count(
    declared: Iterable[Iterable[Any]], cells: Iterable[Mapping[str, Any]]
) -> int:
    """Declared edges (`[hypothesis, sign]` pairs) whose knockout label is indeterminate."""
    status = {f"{c['family']}|{c['relation']}|{c['kpi']}": c["status"] for c in cells}
    return sum(status[name] == "INDET" for name, _sign in declared)


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "flat": flat,
    "null_hypothesis_count": null_hypothesis_count,
    "unestablished_null_count": unestablished_null_count,
    "null_rejection_rate": null_rejection_rate,
    "unestablished_null_rejection_rate": unestablished_null_rejection_rate,
    "null_rejection_bound": null_rejection_bound,
    "high_violation_ratios": high_violation_ratios,
    "nonzero_count": nonzero_count,
    "indeterminate_declared_count": indeterminate_declared_count,
}

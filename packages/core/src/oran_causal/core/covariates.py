"""Shared builder of the design covariates every equal-information method conditions on.

Blocks, in order: lagged KPIs (`lag_kpi:<K>`), context (`ctx:<name>`), setpoints of dither
designs (`sp:<P>`), lagged actions (`<P>@t-<L>`, plus `gap@t-<L>` when some row follows a
time gap) and, when a focal action is given, the concurrent action columns
(`concurrent:<P>`). Rows are processed in information order and returned in row order.
An all-NaN column is dropped; a partly missing column is 0-filled and followed by its
`<name>:missing` indicator. Lag L of a row is the action vector L rows earlier only when
that row is a true previous step, else 0.
"""

import dataclasses
from typing import Literal

import numpy as np

from oran_causal.core.data import Dataset

DESIGNED_KINDS = ("iid", "dither", "logged")


@dataclasses.dataclass(frozen=True)
class CovariateSpec:
    """Which covariate blocks to build.

    Attributes:
        setpoints: `"all"` dither setpoints, only the `"focal"` one, or `"none"`.
        lagged_actions: Include lagged action columns.
        lags: Number of lags.
        focal: Focal action index or name; adds its concurrent columns.
        concurrent: `"designed"` (designed columns only) or `"all"`.
        kpi_lag: Include lagged KPIs.
        context: Include the observed context.
    """

    setpoints: Literal["all", "focal", "none"] = "all"
    lagged_actions: bool = True
    lags: int = 2
    focal: int | str | None = None
    concurrent: str = "designed"
    kpi_lag: bool = True
    context: bool = True


EQ = CovariateSpec()
R3 = CovariateSpec(setpoints="none", lagged_actions=False)


def info_order(data: Dataset) -> np.ndarray:
    """Information order of the rows: stable argsort of `time_index`, else row order."""
    if data.time_index is None:
        return np.arange(data.n)
    return np.argsort(np.asarray(data.time_index), kind="stable")


def finite_block(m: np.ndarray | None, names: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """Usable columns of a block: all-NaN dropped, partly missing 0-filled plus an indicator."""
    if m is None:
        return [], []
    m = np.asarray(m, float)
    if m.ndim == 1:
        m = m[:, None]
    cols: list[np.ndarray] = []
    out: list[str] = []
    for j in range(m.shape[1]):
        c = m[:, j]
        ok = np.isfinite(c)
        if not ok.any():
            continue
        if ok.all():
            cols.append(c)
            out.append(names[j])
        else:
            cols += [np.where(ok, c, 0.0), (~ok).astype(float)]
            out += [names[j], f"{names[j]}:missing"]
    return cols, out


def concurrent_indices(data: Dataset, focal: int, concurrent: str = "designed") -> list[int]:
    """Other action columns usable at t for the focal column.

    `"designed"` keeps every other designed column and none when the focal column has no
    known design; `"all"` keeps every other action column.
    """
    if concurrent == "all":
        return [j for j in range(len(data.action_names)) if j != focal]
    if concurrent != "designed":
        raise ValueError(f"concurrent must be 'designed' or 'all', got {concurrent!r}")
    if data.designs[focal].kind not in DESIGNED_KINDS:
        return []
    return [j for j, d in enumerate(data.designs) if j != focal and d.kind in DESIGNED_KINDS]


def _focal_index(data: Dataset, focal: int | str | None) -> int | None:
    if focal is None:
        return None
    return data.action_names.index(focal) if isinstance(focal, str) else int(focal)


def _context_names(data: Dataset, c_dim: int) -> list[str]:
    given = tuple((data.meta or {}).get("context_names") or ())
    return [f"ctx:{x}" for x in (given if len(given) == c_dim else [f"c{j}" for j in range(c_dim)])]


def _static_blocks(
    data: Dataset, order: np.ndarray, spec: CovariateSpec, fi: int | None
) -> tuple[list[np.ndarray], list[str]]:
    cols: list[np.ndarray] = []
    names: list[str] = []
    if spec.kpi_lag and data.X_kpi_lag is not None:
        lag_names = [f"lag_kpi:{k}" for k in data.kpi_names]
        c, nm = finite_block(np.asarray(data.X_kpi_lag)[order], lag_names)
        cols, names = cols + c, names + nm
    if spec.context and data.context is not None:
        ctx = np.asarray(data.context)
        c, nm = finite_block(ctx[order], _context_names(data, 1 if ctx.ndim == 1 else ctx.shape[1]))
        cols, names = cols + c, names + nm
    if spec.setpoints != "none":
        for j, (a, d) in enumerate(zip(data.action_names, data.designs, strict=True)):
            if spec.setpoints == "focal" and j != fi:
                continue
            if d.kind == "dither" and d.fixed_part is not None:
                c, nm = finite_block(np.asarray(d.fixed_part, float)[order], [f"sp:{a}"])
                cols, names = cols + c, names + nm
    return cols, names


def _lag_blocks(
    data: Dataset, order: np.ndarray, lags: int
) -> tuple[list[np.ndarray], list[str], np.ndarray]:
    n = data.n
    cols: list[np.ndarray] = []
    names: list[str] = []
    mask = np.ones(n, bool)
    a = np.asarray(data.X_action, float)[order]
    t = None if data.time_index is None else np.asarray(data.time_index)[order]
    for lag in range(1, min(lags, n - 1) + 1):
        ok = np.zeros(n, bool)
        ok[lag:] = True if t is None else (t[lag:] - t[:-lag]) == lag
        lagged = np.zeros_like(a)
        lagged[lag:] = a[:-lag]
        lagged[~ok] = 0.0
        cols += [lagged[:, j] for j in range(a.shape[1])]
        names += [f"{x}@t-{lag}" for x in data.action_names]
        if t is not None and not ok[lag:].all():
            cols.append((~ok).astype(float))
            names.append(f"gap@t-{lag}")
        mask &= ok
    return cols, names, mask


def design_covariates(
    data: Dataset, spec: CovariateSpec = EQ
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Builds the design covariates of `data`.

    Args:
        data: Row corpus.
        spec: Blocks to build; `EQ` is the equal-information set, `R3` its reduced set.

    Returns:
        (names, matrix of shape `[n, c]` in row order, row_mask of shape `[n]`); row_mask is
        True when every included lag of the row is a true previous step.
    """
    if spec.setpoints not in ("all", "focal", "none"):
        raise ValueError(f"setpoints must be 'all', 'focal' or 'none', got {spec.setpoints!r}")
    fi = _focal_index(data, spec.focal)
    if spec.setpoints == "focal" and fi is None:
        raise ValueError("setpoints='focal' needs a focal action")
    n = data.n
    order = info_order(data)
    cols, names = _static_blocks(data, order, spec, fi)
    mask = np.ones(n, bool)
    if spec.lagged_actions and spec.lags > 0:
        c, nm, mask = _lag_blocks(data, order, spec.lags)
        cols, names = cols + c, names + nm
    if fi is not None:
        conc = concurrent_indices(data, fi, spec.concurrent)
        a = np.asarray(data.X_action, float)[order]
        cols += [a[:, j] for j in conc]
        names += [f"concurrent:{data.action_names[j]}" for j in conc]
    m_info = np.column_stack(cols) if cols else np.zeros((n, 0))
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    return tuple(names), m_info[inv], mask[inv]


def eq_min_covariates(
    data: Dataset, focal: int | str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Minimal design set: lagged KPIs, context, every other action at t, the focal setpoint."""
    spec = CovariateSpec(setpoints="focal", lagged_actions=False, focal=focal, concurrent="all")
    return design_covariates(data, spec)

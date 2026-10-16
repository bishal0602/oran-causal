"""Partial correlation of each source given all other sources (E1 discovery v2 score).

Score |rho| of z-standardized columns; a collinear source is a recorded STOP. p-value: the
exact partial-correlation t-test, t = rho sqrt(df / (1 - rho^2)), df = n - 2 - |Z|. The E1
v2 rule (per-target largest-gap cut) is kept as `declared_native`. Sign: the sign of rho.
"""

from typing import Any

import numpy as np
from scipy import stats

from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._citests import CITestBase, Prepared

EPS_COND = 1e-8
FROZEN_FLOOR = 0.0


def largest_gap(values: np.ndarray, floor: float = 1e-3) -> float:
    """Midpoint of the biggest gap between the sorted values.

    Only gaps whose lower end is at or above `floor` are eligible; with none eligible the
    global biggest gap is used.
    """
    v = np.sort(np.asarray(values, dtype=float))
    if v.size < 2:
        return float(v[0]) if v.size else 0.0
    gaps = np.diff(v)
    eligible = v[:-1] >= floor
    idx = (
        int(np.argmax(np.where(eligible, gaps, -np.inf)))
        if eligible.any()
        else int(np.argmax(gaps))
    )
    return float((v[idx] + v[idx + 1]) / 2.0)


def partial_correlation_scores(
    xs: np.ndarray, ys: np.ndarray, eps_cond: float = EPS_COND
) -> np.ndarray:
    """Signed partial correlations, shape `[k, d]`, of every input given all other inputs.

    Branch order per edge: a target residual variance below `eps_cond` gives rho = 0; else
    an input residual variance below `eps_cond` is a STOP; else the residual correlation.

    Args:
        xs: Standardized inputs, `[n, d]`.
        ys: Standardized targets, `[n, k]`.
        eps_cond: Residual-variance floor.

    Raises:
        ValueError: An input is near-collinear with the others (STOP).
    """
    n, d = xs.shape
    k = ys.shape[1]
    partial = np.zeros((k, d), dtype=np.float64)
    for i in range(d):
        others = np.concatenate([np.delete(xs, i, axis=1), np.ones((n, 1))], axis=1)
        rhs = np.concatenate([xs[:, [i]], ys], axis=1)
        solution, *_ = np.linalg.lstsq(others, rhs, rcond=None)
        resid = rhs - others @ solution
        r_i = resid[:, 0]
        feat_resid_var = float(r_i @ r_i) / n
        for j in range(k):
            r_yj = resid[:, 1 + j]
            if float(r_yj @ r_yj) / n < eps_cond:
                partial[j, i] = 0.0
            elif feat_resid_var < eps_cond:
                raise ValueError(
                    f"discovery_v2: input column {i} is near-collinear with the other inputs "
                    f"(||r_i||^2/n = {feat_resid_var:.3e} < {eps_cond:.0e}); "
                    "the partial correlation "
                    "is non-identifiable. This is a recorded STOP, not an absent edge."
                )
            else:
                partial[j, i] = (r_i @ r_yj) / (np.sqrt(r_i @ r_i) * np.sqrt(r_yj @ r_yj))
    return partial


def _zstd(a: np.ndarray, what: str) -> np.ndarray:
    sd = a.std(axis=0)
    if (sd == 0).any():
        raise ValueError(f"pcorr: zero-variance {what} column(s) {np.nonzero(sd == 0)[0].tolist()}")
    return (a - a.mean(axis=0)) / sd


def t_test_p(rho: np.ndarray, df: np.ndarray | float) -> np.ndarray:
    """Two-sided t-test p-value of partial correlations `rho` with `df` degrees of freedom."""
    r2 = np.clip(rho**2, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        t = np.abs(rho) * np.sqrt(df / np.maximum(1.0 - r2, 1e-300))
    return 2.0 * stats.t.sf(t, df)


def _native_declared(rho: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    native = np.zeros(len(pairs), dtype=bool)
    for j in {j for _, j in pairs}:
        es = [e for e, (_, jj) in enumerate(pairs) if jj == j]
        row = np.abs(rho[es])
        if row.max() > row.min():
            native[es] = row >= largest_gap(row, floor=FROZEN_FLOOR)
    return native


class PCorrMethod(CITestBase):
    """Partial-correlation t-test arm."""

    name = "pcorr"
    version = "xm-pcorr-v1"
    method_key = 2
    native_sign = True

    def _test(self, prep: Prepared, data: Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
        xs, ys = _zstd(prep.S, "source"), _zstd(prep.Y, "target")
        n = xs.shape[0]
        rho, df = np.zeros(len(prep.pairs)), np.zeros(len(prep.pairs))
        for cols, es in prep.column_groups():
            xg = xs[:, cols]
            d = len(cols)
            if np.linalg.matrix_rank(np.concatenate([xg, np.ones((n, 1))], axis=1)) < d + 1:
                raise ValueError("pcorr: rank-deficient source design")
            part = partial_correlation_scores(xg, ys, EPS_COND)
            for e in es:
                i, j = prep.pairs[e]
                rho[e], df[e] = part[j, cols.index(i)], n - 2 - (d - 1)
        return {
            "score": np.abs(rho),
            "p": t_test_p(rho, df),
            "sign": rho,
            "declared_native": _native_declared(rho, prep.pairs),
            "notes": {
                "df": sorted({int(v) for v in df}),
                "native_rule": "per-target largest_gap(floor=0) over the target's candidates",
            },
        }

"""Fixed-b reference distribution of a Bartlett-kernel HAC t statistic.

Kiefer and Vogelsang (2005), "A new asymptotic theory for heteroskedasticity-autocorrelation
robust tests", Econometric Theory 21, 1130-1164. With bandwidth M = b T (Bartlett weights
1 - j / M) the t statistic converges to W(1) / sqrt(Q(b)), Q(b) = (2 / b) [int_0^1 B(r)^2 dr
- int_0^(1-b) B(r + b) B(r) dr] for a Brownian bridge B independent of W(1). The two-sided
p-value E_Q[2 Phi(-|t| sqrt(Q(b)))] is averaged over `N_DRAWS` simulated Q draws from
normalised partial sums of `T_GRID` iid normals, from a fixed seed. A b between grid lags
interpolates the p-value linearly; b < `B0` interpolates between the normal p (b = 0) and
the p at `B0`.
"""

import functools

import numpy as np
from scipy import stats

T_GRID = 1000
N_DRAWS = 20_000
SEED = (7803, 3801)
B0 = 10 / T_GRID
_CHUNK = 2_000
KV_TABLE1_BARTLETT = (
    (0.90, 1.2816, 1.3040, 0.5135, -0.3386),
    (0.95, 1.6449, 2.1859, 0.3142, -0.3427),
    (0.975, 1.9600, 2.9694, 0.4160, -0.5324),
    (0.99, 2.3263, 4.1618, 0.5368, -0.9060),
)


@functools.lru_cache(maxsize=1)
def _bridges() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    b = np.empty((N_DRAWS, T_GRID), np.float32)
    r = np.arange(1, T_GRID + 1) / T_GRID
    for i in range(0, N_DRAWS, _CHUNK):
        s = np.cumsum(rng.standard_normal((min(_CHUNK, N_DRAWS - i), T_GRID)), axis=1) / np.sqrt(
            T_GRID
        )
        b[i : i + len(s)] = s - r * s[:, -1:]
    i0 = np.einsum("ij,ij->i", b, b, dtype=np.float64) / T_GRID
    return b, i0


@functools.lru_cache(maxsize=4096)
def q_draws(m: int) -> np.ndarray:
    """Q(b) draws at grid lag `m` (b = m / `T_GRID`), clipped at 0.

    Raises:
        ValueError: `m` is outside 1..`T_GRID`.
    """
    if not 1 <= m <= T_GRID:
        raise ValueError(f"grid lag must be in 1..{T_GRID}, got {m}")
    b, i0 = _bridges()
    frac = m / T_GRID
    if m == T_GRID:
        im = np.zeros(N_DRAWS)
    else:
        parts = [
            np.einsum("ij,ij->i", b[i : i + _CHUNK, m:], b[i : i + _CHUNK, :-m], dtype=np.float64)
            for i in range(0, N_DRAWS, _CHUNK)
        ]
        im = np.concatenate(parts) / T_GRID
    return np.maximum(2.0 / frac * (i0 - im), 0.0)


def _p_grid(t: float, m: int) -> float:
    return float(np.mean(2.0 * stats.norm.sf(abs(t) * np.sqrt(q_draws(m)))))


def fixedb_p(t: float, b: float) -> float:
    """Two-sided fixed-b p-value of a Bartlett HAC t statistic `t` at bandwidth fraction `b`."""
    if not np.isfinite(t):
        return 0.0 if np.isinf(t) else float("nan")
    b = float(min(max(b, 0.0), 1.0))
    p0 = float(2.0 * stats.norm.sf(abs(t)))
    if b <= 0.0:
        return p0
    if b < B0:
        w = b / B0
        return (1.0 - w) * p0 + w * _p_grid(t, round(B0 * T_GRID))
    x = b * T_GRID
    lo, hi = int(np.floor(x)), int(np.ceil(x))
    if lo == hi:
        return _p_grid(t, lo)
    w = x - lo
    return (1.0 - w) * _p_grid(t, lo) + w * _p_grid(t, hi)


def kv_cv(b: float, percentile: float) -> float:
    """Kiefer-Vogelsang (2005) Table I Bartlett critical value function cv(b) at `percentile`.

    Raises:
        KeyError: `percentile` is not one of .90, .95, .975, .99.
    """
    for pc, a0, a1, a2, a3 in KV_TABLE1_BARTLETT:
        if abs(pc - percentile) < 1e-12:
            return a0 + a1 * b + a2 * b**2 + a3 * b**3
    raise KeyError(percentile)

"""Predictable adjustments: ridge fits on past rows and running residuals.

Every array is in information order. A weight for row t uses rows before t only.
"""

import dataclasses
import itertools
import math
from collections.abc import Callable

import numpy as np

LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

Predict = Callable[[np.ndarray], np.ndarray]


def _gcv_path(
    xs: np.ndarray, yc: np.ndarray, lams: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    uty = u.T @ yc
    best_g = np.full(yc.shape[1], np.inf)
    coef = np.zeros((xs.shape[1], yc.shape[1]))
    fit_best = np.zeros_like(yc)
    for lam in lams:
        big_l = lam * n
        f = s**2 / (s**2 + big_l)
        fit = u @ (f[:, None] * uty)
        rss = ((yc - fit) ** 2).sum(0)
        g = rss / max(n - float(f.sum()), 1.0) ** 2
        better = g < best_g
        if better.any():
            best_g[better] = g[better]
            coef[:, better] = (vt.T @ ((s / (s**2 + big_l))[:, None] * uty))[:, better]
            fit_best[:, better] = fit[:, better]
    return coef, fit_best


def ridge_fit(
    x: np.ndarray, y: np.ndarray, lams: tuple[float, ...] = LAMS
) -> tuple[Predict, np.ndarray]:
    """Per-target ridge with standardised covariates and an unpenalised intercept.

    Lambda (times n) is chosen by GCV per target column.

    Args:
        x: Covariates, shape `[n, d]`.
        y: Targets, shape `[n, k]`.
        lams: Lambda grid.

    Returns:
        (predict, in-sample residuals); `predict(x_new)` has shape `[m, k]`.
    """
    n, d = x.shape
    mx = x.mean(0) if n else np.zeros(d)
    sx = x.std(0) if n else np.ones(d)
    keep = sx > 1e-12
    my = y.mean(0)
    yc = y - my
    if not keep.any():
        return (lambda xn: np.broadcast_to(my, (len(xn), y.shape[1])).copy()), yc
    xs = (x[:, keep] - mx[keep]) / sx[keep]
    coef, fit_best = _gcv_path(xs, yc, lams)
    mxk, sxk = mx[keep], sx[keep]

    def predict(xn: np.ndarray) -> np.ndarray:
        return my[None, :] + ((xn[:, keep] - mxk) / sxk) @ coef

    return predict, yc - fit_best


def mad_scale(r: np.ndarray) -> np.ndarray:
    """Per-column MAD scale (1.4826 MAD), falling back to the RMS, then to 1."""
    med = np.median(r, 0)
    s = 1.4826 * np.median(np.abs(r - med), 0)
    rms = np.sqrt(np.mean(r**2, 0))
    return np.where(s > 0, s, np.where(rms > 0, rms, 1.0))


@dataclasses.dataclass(frozen=True)
class AdjustConfig:
    """Expanding-window ridge settings.

    Attributes:
        min_burn: Minimum number of leading rows with weight 0.
        burn_per_cov: Burn-in rows per covariate.
        growth: Refit boundaries grow geometrically by this factor.
        lams: Ridge lambda grid.
        huber_c: Huber clip in residual MAD units; None for no clip.
    """

    min_burn: int = 30
    burn_per_cov: float = 1.0
    growth: float = 1.25
    lams: tuple[float, ...] = LAMS
    huber_c: float | None = None


def burn_rows(config: AdjustConfig, d: int) -> int:
    """Leading rows with weight 0: max(min_burn, ceil(burn_per_cov * (d + 1)))."""
    return int(max(config.min_burn, math.ceil(config.burn_per_cov * (d + 1))))


def refit_bounds(burn: int, n: int, growth: float) -> list[int]:
    """Geometric refit boundaries from `burn` to `n`."""
    bounds = [burn]
    while bounds[-1] < n:
        bounds.append(min(n, max(bounds[-1] + 1, math.ceil(bounds[-1] * growth))))
    return bounds


def predictable_weights(
    x: np.ndarray, y: np.ndarray, config: AdjustConfig
) -> tuple[np.ndarray, dict]:
    """Past-only ridge residuals, optionally Huber-clipped, as CRT weights.

    Rows in [b_i, b_{i+1}) are predicted by a ridge fitted on rows before b_i.

    Args:
        x: Predictable covariates, shape `[n, d]`.
        y: Targets, shape `[n, k]`.
        config: Adjustment settings.

    Returns:
        (W, info) with W of shape `[n, k]` and info keys `burn`, `n_refits`, `d`.
    """
    n, k = y.shape
    d = x.shape[1]
    burn = burn_rows(config, d)
    w = np.zeros((n, k))
    e_all = np.zeros((n, k))
    if burn >= n:
        return w, {"burn": burn, "n_refits": 0, "d": d}
    bounds = refit_bounds(burn, n, config.growth)
    for a, b in itertools.pairwise(bounds):
        predict, res_in = ridge_fit(x[:a], y[:a], config.lams)
        e = y[a:b] - predict(x[a:b])
        e_all[a:b] = e
        if config.huber_c is None:
            w[a:b] = e
            continue
        scale = mad_scale(e_all[burn:a]) if a - burn >= 10 else mad_scale(res_in)
        w[a:b] = scale[None, :] * np.clip(e / scale[None, :], -config.huber_c, config.huber_c)
    return w, {"burn": burn, "n_refits": len(bounds) - 1, "d": d}


def ridge_gcv(
    x: np.ndarray, y: np.ndarray, lams: tuple[float, ...] = LAMS
) -> tuple[np.ndarray, float]:
    """Multi-output ridge on centred covariates with one lambda (times n) chosen by GCV.

    Returns:
        (coefficients of shape `[d, p]`, chosen lambda).
    """
    n = len(x)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    uty = u.T @ y
    best: tuple[float, float] | None = None
    for lam in lams:
        big_l = lam * n
        f = s**2 / (s**2 + big_l)
        fit = u @ (f[:, None] * uty)
        rss = float(((y - fit) ** 2).sum())
        g = rss / max(n - float(f.sum()), 1.0) ** 2
        if best is None or g < best[0]:
            best = (g, lam)
    lam = best[1] if best else lams[0]
    big_l = lam * n
    return vt.T @ ((s / (s**2 + big_l))[:, None] * uty), lam


@dataclasses.dataclass(frozen=True)
class ResidualConfig:
    """Running-residual settings.

    Attributes:
        burn: Units of a family whose residual never enters the scale.
        prior_n: Pseudo-units shrinking a stratum's running means to the pooled means.
        ridge: Ridge on the pooled within-stratum slope, relative to its scale.
        use_pre: Adjust for the pre-window value.
    """

    burn: int = 10
    prior_n: float = 20.0
    ridge: float = 1e-6
    use_pre: bool = True


@dataclasses.dataclass
class _RunningState:
    strata: dict[tuple[int, float], tuple[int, float, float]] = dataclasses.field(
        default_factory=dict
    )
    pool: dict[float, list[float]] = dataclasses.field(
        default_factory=lambda: {1.0: [0, 0.0, 0.0], -1.0: [0, 0.0, 0.0]}
    )
    cxy: float = 0.0
    cpp: float = 0.0
    r2: float = 0.0
    nr: int = 0


def _centre(
    st: _RunningState, key: tuple[int, float], yu: float, pu: float, k: float
) -> tuple[float, float]:
    ns, my, mp = st.strata.get(key, (0, 0.0, 0.0))
    pn, py, pp = st.pool[key[1]]
    if pn == 0:
        return (my, mp) if ns else (yu, pu)
    if ns + k <= 0:
        return py, pp
    return (ns * my + k * py) / (ns + k), (ns * mp + k * pp) / (ns + k)


def _absorb(
    st: _RunningState, key: tuple[int, float], obs: tuple[float, float, float], burn: int
) -> None:
    yu, pu, r = obs
    ns, my, mp = st.strata.get(key, (0, 0.0, 0.0))
    pn, py, pp = st.pool[key[1]]
    if pn >= burn:
        rc = min(abs(r), 5.0 * np.sqrt(st.r2 / st.nr)) if st.nr >= 5 else abs(r)
        st.r2 += rc * rc
        st.nr += 1
    if ns > 0:
        f = ns / (ns + 1.0)
        st.cxy += f * (pu - mp) * (yu - my)
        st.cpp += f * (pu - mp) ** 2
    st.strata[key] = (ns + 1, my + (yu - my) / (ns + 1), mp + (pu - mp) / (ns + 1))
    st.pool[key[1]] = [pn + 1, py + (yu - py) / (pn + 1), pp + (pu - pp) / (pn + 1)]


def predictable_residuals(
    y: np.ndarray,
    pre: np.ndarray | None,
    episode: np.ndarray,
    sgn: np.ndarray,
    config: ResidualConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Running past-only residuals of decision units in information order.

    r_u = y_u - a_s - b (pre_u - p_s), with the running (episode, sign) stratum means a_s
    and p_s shrunk to the same-sign pooled running means and the running pooled
    within-stratum slope b. sigma_u is the RMS of the past residuals (NaN before any).

    Args:
        y: Outcomes.
        pre: Pre-window values, or None.
        episode: Episode id per unit.
        sgn: Request direction per unit.
        config: Running-residual settings; None uses the defaults.

    Returns:
        (r, sigma), each of shape `[n]`.
    """
    config = config or ResidualConfig()
    y = np.asarray(y, float)
    pre_v = np.zeros_like(y) if (pre is None or not config.use_pre) else np.asarray(pre, float)
    res = np.zeros(len(y))
    sigs = np.full(len(y), np.nan)
    st = _RunningState()
    k = float(config.prior_n)
    for u in range(len(y)):
        key = (int(episode[u]), 1.0 if sgn[u] > 0 else -1.0)
        a_y, a_p = _centre(st, key, y[u], pre_v[u], k)
        b = st.cxy / (st.cpp + config.ridge * max(st.cpp, 1e-300) + 1e-300) if st.cpp > 0 else 0.0
        r = y[u] - a_y - b * (pre_v[u] - a_p)
        res[u] = r
        if st.nr > 0:
            sigs[u] = np.sqrt(st.r2 / st.nr)
        _absorb(st, key, (y[u], pre_v[u], r), config.burn)
    return res, sigs

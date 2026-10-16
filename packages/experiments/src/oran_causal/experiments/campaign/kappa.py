"""Calibration of the observation-noise level kappa (ruling R-24) from system-level DEV episodes.

Model: one HistGradientBoostingRegressor per KPI pooled over cells, fit on the even episodes
and evaluated on the odd ones (primary), the reverse (swap) and both (xfit). u_k = 1 - R^2_oos
and kappa_k = sqrt(u_k / (1 - u_k)). CI: episode bootstrap within each half, B reps, refit,
RNG `default_rng([7824, b])`. The design matrices come from the episode records; building them
needs the system-level decision layer and is not part of this module.
"""

import dataclasses
import hashlib
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

KPIS = ("pv", "v", "e", "rlf", "load")
FIELD = {"pv": "pv", "v": "v", "e": "e", "rlf": "rlf", "load": "ue"}
STEP = 10
L_KPI, L_KNOB, L_NBR = 6, 3, 3
SEED0, N_DEV = 188000, 20
VARIANTS = ("base", "lead", "nbr", "time", "all")
GROUPS = {
    "base": {"base"},
    "lead": {"base", "lead"},
    "nbr": {"base", "nbr"},
    "time": {"base", "time"},
    "all": {"base", "lead", "nbr", "time"},
}
TAG = 7824
SPLITS = (("primary", (0,)), ("swap", (1,)), ("xfit", (0, 1)))

type Fit = dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


def lag(a: np.ndarray, steps: int) -> np.ndarray:
    """`a` (K, C) shifted down by `steps` along axis 0 with NaN fill; negative steps lead."""
    out = np.full_like(a, np.nan, dtype=float)
    if steps > 0:
        out[steps:] = a[:-steps]
    elif steps < 0:
        out[:steps] = a[-steps:]
    else:
        out[:] = a
    return out


def data_sha(x: np.ndarray, y: np.ndarray) -> str:
    """sha256 of the design matrix (float32) and targets (float64)."""
    xb = np.ascontiguousarray(x, np.float32).tobytes()
    return hashlib.sha256(xb + np.ascontiguousarray(y, np.float64).tobytes()).hexdigest()


def columns(names: Sequence[str], variant: str) -> list[int]:
    """Feature indices of a predictor variant."""
    return [i for i, n in enumerate(names) if n.split("|")[0] in GROUPS[variant]]


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=0)


@dataclasses.dataclass(frozen=True)
class DesignData:
    """Pooled design matrices of the DEV episodes.

    Attributes:
        x: Features, shape `[rows, features]`.
        y: KPI targets, shape `[rows, 5]`.
        ep: Episode index of each row.
        cell: Cell index of each row.
    """

    x: np.ndarray
    y: np.ndarray
    ep: np.ndarray
    cell: np.ndarray


def r2(y: np.ndarray, yh: np.ndarray, grp: np.ndarray | None = None) -> float:
    """Out-of-sample R^2; with `grp`, SST about per-group means."""
    sse = float(((y - yh) ** 2).sum())
    if grp is None:
        sst = float(((y - y.mean()) ** 2).sum())
    else:
        _, inv = np.unique(grp, return_inverse=True)
        mu = np.bincount(inv, y) / np.bincount(inv)
        sst = float(((y - mu[inv]) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def kappa_of(u: Any) -> np.ndarray:
    """Kappa = sqrt(u / (1 - u)); u >= 1 gives inf."""
    u = np.asarray(u, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(u >= 1, np.inf, np.sqrt(np.clip(u, 0, None) / (1 - u)))


def _rows(
    ep: np.ndarray, cell: np.ndarray, episodes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    idx, grp = [], []
    for d, e in enumerate(episodes):
        r = np.flatnonzero(ep == e)
        idx.append(r)
        grp.append(d * 1000 + cell[r])
    return np.concatenate(idx), np.concatenate(grp)


def fit_eval(d: DesignData, train: Sequence[int], test: Sequence[int], cols: Sequence[int]) -> Fit:
    """One model per KPI fit on `train` episodes (repeats = bootstrap weights), evaluated on `test`.

    Returns:
        `{kpi: (y_test, y_hat, group)}`; group = test (episode draw, cell) id.
    """
    itr, _ = _rows(d.ep, d.cell, train)
    ite, gte = _rows(d.ep, d.cell, test)
    out = {}
    for i, k in enumerate(KPIS):
        m = _model().fit(d.x[itr][:, cols], d.y[itr, i])
        out[k] = (d.y[ite, i], m.predict(d.x[ite][:, cols]), gte)
    return out


def summary(us: dict[str, float]) -> dict[str, Any]:
    """U and kappa per KPI with their medians and means."""
    u = np.array([us[k] for k in KPIS])
    kap = kappa_of(u)
    return {
        "u": dict(zip(KPIS, u.tolist(), strict=True)),
        "kappa": dict(zip(KPIS, kap.tolist(), strict=True)),
        "kappa_median": float(np.median(kap)),
        "kappa_mean": float(np.mean(kap)),
        "kappa_from_median_u": float(kappa_of(np.median(u))),
        "u_median": float(np.median(u)),
    }


def _us(parts: Sequence[Fit], within: bool) -> dict[str, float]:
    us = {}
    for k in KPIS:
        y = np.concatenate([p[k][0] for p in parts])
        yh = np.concatenate([p[k][1] for p in parts])
        g = np.concatenate([p[k][2] + 10**6 * j for j, p in enumerate(parts)])
        us[k] = 1.0 - r2(y, yh, g if within else None)
    return us


def point(
    d: DesignData, train: Sequence[int], test: Sequence[int], cols: Sequence[int]
) -> dict[str, Any]:
    """Primary split, swap and cross-fit, each with pooled and within R^2."""
    fits = (fit_eval(d, train, test, cols), fit_eval(d, test, train, cols))
    res = {}
    for name, which in SPLITS:
        parts = [fits[i] for i in which]
        for kind in ("pooled", "within"):
            res[f"{name}/{kind}"] = summary(_us(parts, kind == "within"))
    return res


def boot_one(args: tuple[Any, ...]) -> tuple[int, dict[str, dict[str, float]]]:
    """One bootstrap replicate: resample train and test episodes within their halves and refit."""
    d, train, test, cols, b = args
    rng = np.random.default_rng([TAG, b])
    tr = list(rng.choice(train, len(train), replace=True))
    te = list(rng.choice(test, len(test), replace=True))
    f = fit_eval(d, tr, te, cols)
    out = {}
    for kind in ("pooled", "within"):
        out[kind] = {
            k: 1.0 - r2(f[k][0], f[k][1], f[k][2] if kind == "within" else None) for k in KPIS
        }
    return b, out


def bootstrap(
    d: DesignData,
    cols: Sequence[int],
    reps: Sequence[int],
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> dict[int, dict[str, dict[str, float]]]:
    """Bootstrap replicates `reps` (u per KPI, pooled and within)."""
    train, test = halves()
    jobs = [(d, train, test, cols, b) for b in reps]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            return dict(ex.map(boot_one, jobs, chunksize=4))
    out = {}
    for j in jobs:
        b, o = boot_one(j)
        out[b] = o
        if b % 10 == 0:
            log(f"boot {b}")
    return out


def halves() -> tuple[list[int], list[int]]:
    """Even (train) and odd (test) DEV episode indices."""
    return [e for e in range(N_DEV) if e % 2 == 0], [e for e in range(N_DEV) if e % 2 == 1]


def _q(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def bootstrap_ci(boots: dict[int, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """Percentile 95 % CIs of u, kappa and their summaries from the bootstrap replicates."""
    ci = {}
    for kind in ("pooled", "within"):
        u = np.array([[boots[b][kind][k] for k in KPIS] for b in range(len(boots))])
        kap = kappa_of(u)
        ci[kind] = {
            "u": {k: _q(u[:, i]) for i, k in enumerate(KPIS)},
            "kappa": {k: _q(kap[:, i]) for i, k in enumerate(KPIS)},
            "kappa_median": _q(np.median(kap, 1)),
            "kappa_mean": _q(np.mean(kap, 1)),
            "kappa_from_median_u": _q(kappa_of(np.median(u, 1))),
        }
    return ci

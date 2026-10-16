"""RCoT numerics of the frozen E2 RCoT discovery (random Fourier features, statistic, nulls).

Source: CDD-ORAN@0c89106 `cdd_oran/e2slice/discovery_rcot.py` (lifted there from the
calibration study) and `discovery_rcot_v2.py` (`block_perm_reps` = 299). The statistic is
T = n ||Cxy||_F^2 of the RFF residuals of candidate and target on Z (Strobl, Zhang and
Visweswaran 2019, RCoT).
"""

import dataclasses
import math

import numpy as np

NULL_ANALYTIC = "analytic_hbe"
NULL_BLOCK_PERM = "block_perm"
FROZEN_BLOCK_PERM_REPS_V2 = 299


@dataclasses.dataclass(frozen=True)
class RCoTDiscoveryConfig:
    """RCoT constants; defaults are the frozen E2 protocol values.

    Attributes:
        dz: RFF features of the conditioner Z.
        dxy: RFF features of candidate and target.
        ridge: Ridge of the Z residualization.
        q: Per-target BH level.
        null_method: `"block_perm"` or `"analytic_hbe"`.
        block_perm_reps: Permutations of the block null.
        block_size: Z-neighbour block length.
        residual_epsilon: Relative residual-collapse guard.
        max_guard_fraction: Guard fraction at which the run halts.
        rng_seed: Base RNG seed.
    """

    dz: int = 25
    dxy: int = 5
    ridge: float = 1e-6
    q: float = 0.05
    null_method: str = NULL_BLOCK_PERM
    block_perm_reps: int = 99
    block_size: int = 25
    residual_epsilon: float = 1e-12
    max_guard_fraction: float = 0.05
    rng_seed: int = 0

    def validate(self) -> None:
        """Checks the null method and feature counts.

        Raises:
            ValueError: Unknown null method or a feature count below 1.
        """
        if self.null_method not in (NULL_ANALYTIC, NULL_BLOCK_PERM):
            raise ValueError(
                f"RCoTDiscoveryConfig: null_method must be {NULL_ANALYTIC!r} or "
                f"{NULL_BLOCK_PERM!r}, got {self.null_method!r}"
            )
        if self.dz < 1 or self.dxy < 1:
            raise ValueError("RCoTDiscoveryConfig: dz and dxy must be >= 1")


def frozen_config_v2() -> RCoTDiscoveryConfig:
    """The frozen RCoT-v2 config: the v1 defaults with `block_perm_reps` = 299."""
    return RCoTDiscoveryConfig(block_perm_reps=FROZEN_BLOCK_PERM_REPS_V2)


def _dist_matrix_1d(v: np.ndarray) -> np.ndarray:
    return np.abs(v[:, None] - v[None, :])


def _dist_matrix_euclidean(z: np.ndarray) -> np.ndarray:
    sq = np.einsum("ij,ij->i", z, z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def _median_sigma(v: np.ndarray) -> float:
    m = min(v.shape[0], 200)
    vs = v[:m]
    d = (
        _dist_matrix_euclidean(vs)
        if vs.ndim == 2 and vs.shape[1] > 1
        else _dist_matrix_1d(vs.ravel())
    )
    med = np.median(d[np.triu_indices(m, 1)])
    return med if med > 0 else 1.0


def _rff(v: np.ndarray, d: int, sig: float, rng: np.random.Generator) -> np.ndarray:
    if v.ndim == 1:
        v = v[:, None]
    w = rng.normal(size=(v.shape[1], d)) / sig
    b = rng.uniform(0, 2 * np.pi, size=d)
    f = math.sqrt(2.0 / d) * np.cos(v @ w + b)
    return f - f.mean(0)


def _gamma_series(a: float, xx: float, gln: float) -> float:
    ap = a
    summ = 1.0 / a
    delv = summ
    for _ in range(1000):
        ap += 1.0
        delv *= xx / ap
        summ += delv
        if abs(delv) < abs(summ) * 1e-14:
            break
    return 1.0 - summ * math.exp(-xx + a * math.log(xx) - gln)


def _gamma_fraction(a: float, xx: float, gln: float) -> float:
    b0 = xx + 1.0 - a
    c = 1e300
    d = 1.0 / b0
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b0 += 2.0
        d = an * d + b0
        if abs(d) < 1e-300:
            d = 1e-300
        c = b0 + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delv = d * c
        h *= delv
        if abs(delv - 1.0) < 1e-14:
            break
    return math.exp(-xx + a * math.log(xx) - gln) * h


def _gamma_sf(x: float, k_shape: float, theta: float) -> float:
    """Survival function of Gamma(shape `k_shape`, scale `theta`)."""
    if x <= 0:
        return 1.0
    xx = x / theta
    gln = math.lgamma(k_shape)
    if xx < k_shape + 1.0:
        return _gamma_series(k_shape, xx, gln)
    return _gamma_fraction(k_shape, xx, gln)


def rff_residuals(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: RCoTDiscoveryConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(fx, fy, rx, ry): RFF of x, y, z drawn in that order, fx and fy ridge-residualized on fz."""
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n = x.shape[0]
    fx = _rff(x, cfg.dxy, _median_sigma(x), rng)
    fy = _rff(y, cfg.dxy, _median_sigma(y), rng)
    fz = _rff(z, cfg.dz, _median_sigma(z), rng)
    g = fz.T @ fz + cfg.ridge * n * np.eye(fz.shape[1])
    bx = np.linalg.solve(g, fz.T @ fx)
    by = np.linalg.solve(g, fz.T @ fy)
    return fx, fy, fx - fz @ bx, fy - fz @ by


def rcot_statistic(rx: np.ndarray, ry: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(T = n ||Cxy||_F^2, row products, centred row products)."""
    n = rx.shape[0]
    prod = np.einsum("ni,nj->nij", rx, ry).reshape(n, -1)
    cmean = prod.mean(0)
    stat = n * float(cmean @ cmean)
    return stat, prod, prod - cmean


def _hbe_pvalue(stat: float, prodc: np.ndarray) -> float:
    """Hall-Buckley-Eagleson weighted chi^2 p-value from the centred products."""
    n = prodc.shape[0]
    w = np.linalg.eigvalsh((prodc.T @ prodc) / n)
    w = w[w > 1e-12]
    if w.size == 0:
        return 1.0
    mean = float(w.sum())
    var = float(2.0 * (w**2).sum())
    g = var / (2.0 * mean)
    kdf = 2.0 * mean * mean / var
    return _gamma_sf(stat, kdf / 2.0, 2.0 * g)


def residual_collapse(
    fx: np.ndarray, fy: np.ndarray, rx: np.ndarray, ry: np.ndarray, eps: float
) -> bool:
    """True when the RFF residuals of candidate and target both collapse relative to the raw RFF."""
    ssx = float((fx * fx).sum())
    ssy = float((fy * fy).sum())
    if ssx <= 0.0 or ssy <= 0.0:
        return True
    ratio_x = float((rx * rx).sum()) / ssx
    ratio_y = float((ry * ry).sum()) / ssy
    return math.sqrt(max(ratio_x, 0.0) * max(ratio_y, 0.0)) <= eps


def z_blocks(zs: np.ndarray, block: int = 25) -> list[np.ndarray]:
    """Greedy nearest-neighbour tour of Z split into contiguous blocks of `block` rows."""
    n = zs.shape[0]
    d = _dist_matrix_euclidean(zs).copy()
    np.fill_diagonal(d, np.inf)
    order = np.empty(n, dtype=int)
    cur = 0
    order[0] = 0
    for t in range(1, n):
        d[:, cur] = np.inf
        cur = int(np.argmin(d[cur]))
        order[t] = cur
    return [order[i : i + block] for i in range(0, n, block)]


def rcot_pvalue_analytic(
    cand: np.ndarray, target: np.ndarray, z: np.ndarray, cfg: RCoTDiscoveryConfig, seed: int
) -> tuple[float, float, bool]:
    """RCoT with the HBE null for cand independent of target given Z: (statistic, p, guarded)."""
    rng = np.random.default_rng(seed)
    fx, fy, rx, ry = rff_residuals(cand, target, z, cfg, rng)
    if residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon):
        return math.nan, 1.0, True
    stat, _, prodc = rcot_statistic(rx, ry)
    return stat, _hbe_pvalue(stat, prodc), False


def pair_seed(base: int, j: int, i: int) -> int:
    """Per-(target j, candidate i) RFF seed."""
    return int(
        np.random.SeedSequence(entropy=int(base), spawn_key=(int(j), int(i))).generate_state(1)[0]
    )


def standardize(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z-scores, mean, std) with population std.

    Raises:
        ValueError: A column has zero variance.
    """
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    if (std == 0.0).any():
        raise ValueError(
            f"discovery_rcot: zero-variance column(s) at {np.where(std == 0.0)[0].tolist()}; "
            "cannot z-standardize."
        )
    return (a - mean) / std, mean, std

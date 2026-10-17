"""PMRT statistic on decision-unit tables: learned matched-filter weights.

For one hypothesis, `P` holds the target's post-window bins per receiver slot, `X` the raw
predictable covariates and `okp` which post bins fit the episode. A ridge adjustment, a
Ledoit-Wolf covariance and per-variant kernels K = Sigma^-1 mu are learned on training
units; scores k_u = K'(P_u - B X_u) are centred by a predictable running stratum mean and
optionally reweighted (`_h`) or Huber-clipped (`_c`). Every learned part is fitted on
training episodes and frozen before it touches evaluation units.
"""

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingRegressor

from oran_causal.core.data import UnitTable
from oran_causal.core.design import v_design, v_var
from oran_causal.core.ridge import ResidualConfig, predictable_residuals, ridge_gcv

VARIANTS = ("plain", "plain4", "gls1", "full", "rank1", "loadsp", "prof")
MODS = ("", "_h", "_c")
SIGN_FROM_S = ("pred", "plain", "plain4", "gls1")
ARMS = ("pred", *(f"{v}{h}" for v in VARIANTS for h in MODS), "best", "best_gb")
COMBOS = {
    "max_pred_best": ("pred", "best"),
    "max_plain_best": ("plain", "best"),
    "max_pred_best_gb": ("pred", "best", "best_gb"),
    "max_plain_plain4_best": ("plain", "plain4", "best"),
}
F32 = np.float32
PMRT_VERSION = "pmrt-v1"
LEGACY_VERSION = "mscr-crt-units-plus-v0"
NSLOT = {"own": 1, "nbr": 4, "far": 2}
LOAD = "load"


@dataclasses.dataclass(frozen=True)
class PmrtConfig:
    """Settings of the decision-unit statistic.

    Attributes:
        B: Redraws.
        q: BY level.
        alpha: Nominal level.
        chunk: Redraws per chunk.
        max_chunk_bytes: Memory cap of one chunk.
        n0: Pseudo-units shrinking a stratum's running mean to the same-sign mean.
        n1: Pseudo-units shrinking the same-sign running mean to the training intercept.
        cv_reps: Repeats of the 2-fold episode cross-validation.
        cv_penalty: CV z handicap of the non-plain variants.
        best_exclude: Variants never chosen as best.
        h_clip: Clip of the `_h` weights.
        h_lam: Ridge of the log-variance model, relative to n.
        huber_c: Huber clip of `_c` in predicted standard deviations.
        h_pow: h = sigma_hat^-h_pow.
        lw: Ledoit-Wolf shrinkage of the residual covariance.
        gb: Fit the boosted score correction of arm `best_gb`.
        min_units: Minimum units of a family.
        min_accept: Minimum accepted units.
        min_reject: Minimum rejected units.
        min_episodes: Minimum episodes.
        seed: RNG seed.
    """

    B: int = 9999
    q: float = 0.05
    alpha: float = 0.05
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    n0: float = 20.0
    n1: float = 50.0
    cv_reps: int = 3
    cv_penalty: float = 0.25
    best_exclude: tuple[str, ...] = ("full",)
    h_clip: float = 10.0
    h_lam: float = 30.0
    huber_c: float = 2.5
    h_pow: float = 1.0
    lw: bool = True
    gb: bool = True
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3
    seed: int = 0


def slot_matrix(c: np.ndarray, rel_mask: np.ndarray, score: np.ndarray, s: int) -> np.ndarray:
    """Receiver-slot indicators, shape `[n, s, C]`.

    Slot j < s - 1 is the j-th relation cell by descending `score[c_u, c']` (ties go to the
    lower cell index); slot s - 1 holds the remaining relation cells.
    """
    n, ncell = rel_mask.shape
    a = np.zeros((n, s, ncell), F32)
    if s == 1:
        a[:, 0] = rel_mask
        return a
    sc = np.where(rel_mask, score[c] - 1e-9 * np.arange(ncell)[None, :], -np.inf)
    order = np.argsort(-sc, axis=1, kind="stable")
    ar = np.arange(n)
    for j in range(s - 1):
        cj = order[:, j]
        ok = rel_mask[ar, cj]
        a[ar[ok], j, cj[ok]] = 1.0
    a[:, s - 1] = rel_mask & (a[:, : s - 1].sum(1) == 0)
    return a


def demean(a: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Subtracts the stratum mean from every column."""
    _, inv = np.unique(st, return_inverse=True)
    k = int(inv.max()) + 1
    n = np.bincount(inv, minlength=k).astype(float)
    if a.ndim == 1:
        return a - (np.bincount(inv, a, k) / n)[inv]
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(inv, a[:, j], k) / n)[inv]
    return out


def _past_sums(key: np.ndarray, k2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(key, kind="stable")
    ks = key[order]
    v = k2[order]
    cs = np.empty_like(v)
    cnt = np.empty(len(ks))
    start = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1] if len(ks) else np.zeros(0, int)
    for a, b in zip(start, np.r_[start[1:], len(ks)], strict=True):
        cs[a:b] = np.cumsum(v[a:b], 0) - v[a:b]
        cnt[a:b] = np.arange(b - a)
    s_out, c_out = np.empty_like(cs), np.empty_like(cnt)
    s_out[order], c_out[order] = cs, cnt
    return s_out, c_out


@dataclasses.dataclass(frozen=True)
class Shrinkage:
    """Running-centre shrinkage: `n0` and `n1` pseudo-units and the training `prior` per sign."""

    n0: float
    n1: float
    prior: Mapping[int, Any]


def running_center(
    k: np.ndarray, episode: np.ndarray, sgn: np.ndarray, shrink: Shrinkage
) -> np.ndarray:
    """Predictable centring k - m in information order.

    m_u = (sum of past k in u's (episode, sign) stratum + n0 pm_u) / (count + n0) with
    pm_u = (sum of past k with u's sign + n1 prior[sign]) / (count + n1).

    Args:
        k: Scores, shape `[n]` or `[n, m]`.
        episode: Episode id per unit.
        sgn: Request direction per unit.
        shrink: Shrinkage settings; `prior` maps +1 and -1 to per-column intercepts.

    Returns:
        Centred scores with the shape of `k`.
    """
    k2 = k[:, None] if k.ndim == 1 else k
    g = (np.asarray(sgn) > 0).astype(np.int64)
    pr = np.where(
        g[:, None] == 1,
        np.asarray(shrink.prior[1], float)[None, :],
        np.asarray(shrink.prior[-1], float)[None, :],
    )
    ps, pc = _past_sums(g, k2)
    pm = (ps + shrink.n1 * pr) / (pc[:, None] + shrink.n1)
    ss, sc = _past_sums(np.asarray(episode).astype(np.int64) * 2 + g, k2)
    m = (ss + shrink.n0 * pm) / (sc[:, None] + shrink.n0)
    r = k2 - m
    return r[:, 0] if k.ndim == 1 else r


def lw_cov(e: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance toward a scaled identity on the standardised scale."""
    n, p = e.shape
    ec = e - e.mean(0)
    sd = ec.std(0)
    sd[sd <= 0] = 1.0
    z = ec / sd
    sm = z.T @ z / n
    mu = np.trace(sm) / p
    d2 = float(((sm - mu * np.eye(p)) ** 2).sum())
    step = max(1, n // 2000)
    b2 = float(sum(((np.outer(zr, zr) - sm) ** 2).sum() for zr in z[::step])) * step / n**2
    b2 = min(b2, d2)
    sh = b2 / d2 if d2 > 0 else 1.0
    c = sh * mu * np.eye(p) + (1 - sh) * sm
    return c * np.outer(sd, sd), sh


def kernels(
    mu: np.ndarray, sigma: np.ndarray, cols_nobin3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Matched filters Sigma^-1 mu: on every bin, and on the sub-block without bin 3."""
    reg = 1e-8 * np.trace(sigma) / len(sigma)
    kf = np.linalg.solve(sigma + reg * np.eye(len(sigma)), mu)
    a = cols_nobin3
    ks = np.zeros_like(mu)
    ks[a] = np.linalg.solve(sigma[np.ix_(a, a)] + reg * np.eye(len(a)), mu[a])
    return kf, ks


def pos_profile(a: np.ndarray) -> np.ndarray:
    """Same-signed unit-sum spatial profile (uniform when all zero)."""
    a = np.asarray(a, float)
    a = a * (1.0 if a.sum() >= 0 else -1.0)
    a = np.maximum(a, 0.0)
    return a / a.sum() if a.sum() > 0 else np.full(len(a), 1.0 / len(a))


def rank1_pos(m: np.ndarray, a: np.ndarray | None = None) -> np.ndarray:
    """Mu = a (x) t with a same-signed spatial profile and t its least-squares time profile."""
    if a is None:
        u, _, _ = np.linalg.svd(m, full_matrices=False)
        a = u[:, 0]
    a = pos_profile(a)
    t = a @ m / max(float(a @ a), 1e-300)
    return np.outer(a, t)


@dataclasses.dataclass(frozen=True)
class TrainSet:
    """Training units of one hypothesis.

    Attributes:
        P: Post bins per slot, shape `[n, S * 4]`; NaN where missing.
        X: Raw predictable covariates, shape `[n, d]`.
        okp: Which post bins fit, shape `[n, 4]`.
        v: Assignment scores.
        st: Stratum (episode, sign) code per unit.
    """

    P: np.ndarray
    X: np.ndarray
    okp: np.ndarray
    v: np.ndarray
    st: np.ndarray

    def subset(self, rows: np.ndarray) -> TrainSet:
        """The training set restricted to `rows`."""
        return TrainSet(self.P[rows], self.X[rows], self.okp[rows], self.v[rows], self.st[rows])


def _standardiser(x: np.ndarray) -> dict[str, np.ndarray]:
    fill = np.nanmedian(x, 0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    xf = np.where(np.isfinite(x), x, fill[None, :])
    mx, sx = xf.mean(0), xf.std(0)
    return {"fill": fill, "mx": mx, "sx": sx, "keep": sx > 1e-12}


def xs(core: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    """Covariates filled and standardised with the training statistics of `core`."""
    xf = np.where(np.isfinite(x), x, core["fill"][None, :])
    kp = core["keep"]
    return (xf[:, kp] - core["mx"][kp]) / core["sx"][kp]


def _variant_kernels(
    mu_full: np.ndarray, sigma: np.ndarray, s: int, load_a: np.ndarray | None
) -> dict:
    p = len(mu_full)
    bins = np.tile(np.arange(4), s)
    one03 = (bins < 3).astype(float)
    nob3 = np.nonzero(bins < 3)[0]
    mf = mu_full.reshape(s, 4)
    mus = {"gls1": np.ones(p), "full": mu_full, "rank1": rank1_pos(mf).ravel()}
    mus["loadsp"] = (
        rank1_pos(mf, load_a).ravel() if (load_a is not None and s > 1) else mus["rank1"]
    )
    ks = {"plain": (one03.copy(), one03.copy()), "plain4": (np.ones(p), one03.copy())}
    for nm, mu in mus.items():
        ks[nm] = kernels(mu, sigma, nob3)
    ks["prof"] = (mus["rank1"].copy(), mus["rank1"] * one03)
    return {"mus": mus, "K": ks, "one03": one03, "mf": mf}


def _directions(mus: Mapping[str, np.ndarray], one03: np.ndarray) -> dict[str, float]:
    sdir = {nm: float(np.sign(one03 @ mu)) for nm, mu in mus.items()}
    sdir["prof"] = sdir["rank1"]
    for nm in SIGN_FROM_S:
        sdir[nm] = 1.0
    return sdir


def _intercepts(k: Mapping[str, np.ndarray], st: np.ndarray) -> dict[str, dict[int, float]]:
    sg = np.where(np.asarray(st) % 2 == 1, 1, -1)
    return {
        nm: {
            1: float(np.nanmean(k[nm][sg > 0])) if (sg > 0).any() else 0.0,
            -1: float(np.nanmean(k[nm][sg < 0])) if (sg < 0).any() else 0.0,
        }
        for nm in k
    }


def fit_core(
    train: TrainSet, s: int, cfg: PmrtConfig, load_a: np.ndarray | None = None
) -> dict[str, Any]:
    """Learns the frozen parts of one hypothesis on training units.

    Args:
        train: Training units.
        s: Number of receiver slots.
        cfg: Statistic settings.
        load_a: Spatial profile of the family's load hypothesis, for variant `loadsp`.

    Returns:
        Fill and standardisation, ridge coefficients `B`, covariance `Sigma`, variant means
        `mus`, kernels `K`, sign directions `sdir`, training `z_plain`, `load_a` and the
        per-sign intercepts `prior` of every variant.
    """
    std = _standardiser(train.X)
    x_s = xs(std, train.X)
    full = train.okp.all(1)
    pd_ = demean(train.P[full], train.st[full])
    xd = demean(x_s[full], train.st[full])
    bm, lam = ridge_gcv(xd, pd_)
    e = pd_ - xd @ bm
    vf = train.v[full]
    vv = float(vf @ vf)
    mu_full = e.T @ vf / vv
    sigma, sh = lw_cov(e) if cfg.lw else (np.cov(e.T), 0.0)
    vk = _variant_kernels(mu_full, sigma, s, load_a)
    one03 = vk["one03"]
    z_plain = float(one03 @ mu_full / np.sqrt(max(one03 @ sigma @ one03 / vv, 1e-300)))
    load = (
        pos_profile(np.linalg.svd(vk["mf"], full_matrices=False)[0][:, 0]) if s > 1 else np.ones(1)
    )
    core = {
        **std,
        "B": bm,
        "lam": lam,
        "Sigma": sigma,
        "lw_shrink": sh,
        "mu_full": mu_full,
        "se_mu": np.sqrt(np.diag(sigma) / vv),
        "mus": vk["mus"],
        "K": vk["K"],
        "S": s,
        "z_plain": z_plain,
        "sdir": _directions(vk["mus"], one03),
        "load_a": load,
    }
    core["prior"] = _intercepts(variant_scores(core, train.P, train.X, train.okp), train.st)
    return core


def variant_scores(
    core: Mapping[str, Any], p: np.ndarray, x: np.ndarray, okp: np.ndarray
) -> dict[str, np.ndarray]:
    """Score k per variant: K'(P - B X), with the kernel without bin 3 where bin 3 is missing."""
    e = np.nan_to_num(p) - xs(core, x) @ core["B"]
    has3 = okp[:, 3]
    return {nm: np.where(has3, e @ kf, e @ kn) for nm, (kf, kn) in core["K"].items()}


def fit_hmodel(
    r: np.ndarray, x: np.ndarray, core: Mapping[str, Any], cfg: PmrtConfig
) -> dict[str, Any]:
    """Log-variance ridge of r on the standardised covariates (training)."""
    xs_ = xs(core, x)
    y = np.log(r**2 + 1e-3 * float(np.mean(r**2)) + 1e-300)
    ym = y.mean()
    n, d = xs_.shape
    xm = xs_.mean(0)
    xc = xs_ - xm
    w = np.linalg.solve(xc.T @ xc + cfg.h_lam * n / 100.0 * np.eye(d), xc.T @ (y - ym))
    pred = ym + (xs_ - xm) @ w
    med = float(np.median(np.exp(-0.5 * cfg.h_pow * pred)))
    kappa = float(np.median(np.abs(r) / np.exp(0.5 * pred)) / 0.6745) or 1.0
    return {"w": w, "xm": xm, "ym": ym, "med": med, "kappa": kappa}


def _pred(hm: Mapping[str, Any], core: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    return hm["ym"] + (xs(core, x) - hm["xm"]) @ hm["w"]


def h_weights(
    hm: Mapping[str, Any], core: Mapping[str, Any], x: np.ndarray, cfg: PmrtConfig
) -> np.ndarray:
    """Reweighting h = exp(-h_pow pred / 2) / median, clipped to [1 / h_clip, h_clip]."""
    return np.clip(
        np.exp(-0.5 * cfg.h_pow * _pred(hm, core, x)) / hm["med"], 1.0 / cfg.h_clip, cfg.h_clip
    )


def lin_z(v: np.ndarray, w: np.ndarray, var: np.ndarray) -> float:
    """Z = sum v w / sqrt(sum Var(v) w^2), and 0 when the design variance is 0."""
    sd = float(np.sqrt(var @ (w * w)))
    return float(v @ w / sd) if sd > 0 else 0.0


def split_arm(a: str) -> tuple[str, str]:
    """Splits an arm name into (variant, modifier), e.g. `"rank1_h"` to `("rank1", "_h")`."""
    for m in ("_h", "_c"):
        if a.endswith(m):
            return a[: -len(m)], m
    return a, ""


@dataclasses.dataclass(frozen=True)
class Fitted:
    """A fitted hypothesis: its `core`, the log-variance model `hm` and the settings."""

    core: Mapping[str, Any]
    hm: Mapping[str, Any]
    cfg: PmrtConfig


def modified(r: np.ndarray, mod: str, fitted: Fitted, x: np.ndarray) -> np.ndarray:
    """Weights from the running residual: r, h r (`_h`) or the Huber clip of r (`_c`)."""
    if mod == "":
        return r
    if mod == "_h":
        return h_weights(fitted.hm, fitted.core, x, fitted.cfg) * r
    sc = fitted.hm["kappa"] * np.exp(0.5 * _pred(fitted.hm, fitted.core, x))
    return sc * np.clip(r / sc, -fitted.cfg.huber_c, fitted.cfg.huber_c)


def center_all(
    core: Mapping[str, Any],
    k: Mapping[str, np.ndarray],
    ep: np.ndarray,
    sg: np.ndarray,
    cfg: PmrtConfig,
) -> dict[str, np.ndarray]:
    """Running-centred scores of every variant."""
    out = {}
    for nm in k:
        prior = {1: [core["prior"][nm][1]], -1: [core["prior"][nm][-1]]}
        out[nm] = running_center(k[nm], ep, sg, Shrinkage(cfg.n0, cfg.n1, prior))
    return out


@dataclasses.dataclass(frozen=True)
class CvData:
    """Training units with their design variance, episode and sign, for cross-validation."""

    train: TrainSet
    var: np.ndarray
    episode: np.ndarray
    sgn: np.ndarray

    def subset(self, rows: np.ndarray) -> CvData:
        """The data restricted to `rows`."""
        return CvData(self.train.subset(rows), self.var[rows], self.episode[rows], self.sgn[rows])


def _fold_scores(
    a: CvData, b: CvData, s: int, cfg: PmrtConfig, load_a: np.ndarray | None
) -> dict[str, float]:
    core = fit_core(a.train, s, cfg, load_a)
    ra = center_all(
        core, variant_scores(core, a.train.P, a.train.X, a.train.okp), a.episode, a.sgn, cfg
    )
    rb = center_all(
        core, variant_scores(core, b.train.P, b.train.X, b.train.okp), b.episode, b.sgn, cfg
    )
    out = {}
    for nm in ra:
        o = np.sign(lin_z(a.train.v, ra[nm], a.var)) if nm in SIGN_FROM_S else 1.0
        o = o or 1.0
        fitted = Fitted(core, fit_hmodel(ra[nm], a.train.X, core, cfg), cfg)
        for mod in MODS:
            w = modified(rb[nm], mod, fitted, b.train.X)
            out[nm + mod] = o * lin_z(b.train.v, w, b.var)
    return out


def cv_select(
    data: CvData,
    s: int,
    cfg: PmrtConfig,
    load_a: np.ndarray | None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Repeated 2-fold cross-validation over training episodes.

    Returns:
        `cv`: mean held-out oriented z per arm; `best`: the arm maximising it after the
        penalty on non-plain variants.
    """
    rng = rng or np.random.default_rng([cfg.seed, 6616, 31])
    eps = np.unique(data.episode)
    scores: dict[str, list[float]] = {}
    for _ in range(cfg.cv_reps):
        perm = rng.permutation(eps)
        halves = [
            np.isin(data.episode, perm[: len(perm) // 2]),
            np.isin(data.episode, perm[len(perm) // 2 :]),
        ]
        for ia, ib in ((0, 1), (1, 0)):
            a, b = halves[ia], halves[ib]
            if b.sum() < 30 or a.sum() < 30 or data.train.okp[a].all(1).sum() < 30:
                continue
            for k, v in _fold_scores(data.subset(a), data.subset(b), s, cfg, load_a).items():
                scores.setdefault(k, []).append(v)
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    if not mean:
        return {"cv": {}, "best": "plain"}
    pen = {
        k: mean[k] - (0.0 if k == "plain" else cfg.cv_penalty)
        for k in mean
        if not k.startswith(cfg.best_exclude)
    }
    return {"cv": mean, "best": max(pen, key=lambda k: pen[k])}


def fit_gb(
    k: np.ndarray, x: np.ndarray, st: np.ndarray, core: Mapping[str, Any]
) -> HistGradientBoostingRegressor:
    """Boosted correction of the within-stratum-demeaned score on the standardised covariates."""
    m = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=0,
    )
    m.fit(xs(core, x), demean(k, st))
    return m


Hyp = tuple[str, str, str]


@dataclasses.dataclass(frozen=True)
class PmrtData:
    """Per-unit features of a unit table for the decision-unit statistic.

    Attributes:
        ud: The unit table.
        post: Per-bin post-window sums, shape `[n, K, C, 4]` (float32); NaN where a bin does
            not fit the episode.
        pre: Per-bin pre-window sums, same shape.
        ctx: Observation-only context at t0, shape `[n, J]`; NaN where missing.
        ctx_keys: Names of the context columns.
        hist: Past design-centred modes per family at the unit's cell, then at its exposure
            neighbours, shape `[n, 2 F]`.
        tfrac: t0 / T.
    """

    ud: UnitTable
    post: np.ndarray
    pre: np.ndarray
    ctx: np.ndarray
    ctx_keys: list[str]
    hist: np.ndarray
    tfrac: np.ndarray

    @property
    def n(self) -> int:
        """Number of units."""
        return self.ud.n

    def subset(self, rows: npt.ArrayLike) -> PmrtData:
        """The features restricted to `rows` (indices or a boolean mask)."""
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.nonzero(rows)[0]
        return PmrtData(
            ud=self.ud.subset(rows),
            post=self.post[rows],
            pre=self.pre[rows],
            ctx=self.ctx[rows],
            ctx_keys=list(self.ctx_keys),
            hist=self.hist[rows],
            tfrac=self.tfrac[rows],
        )


def unit_order(ud: UnitTable, rows: npt.ArrayLike) -> np.ndarray:
    """`rows` in information order (episode, t0, cell)."""
    rows = np.asarray(rows, int)
    return rows[np.lexsort((ud.c[rows], ud.t0[rows], ud.episode[rows]))]


def strata(ud: UnitTable, rows: np.ndarray) -> np.ndarray:
    """Stratum code episode * 2 + (sgn > 0) of `rows`."""
    return ud.episode[rows].astype(np.int64) * 2 + (ud.sgn[rows] > 0)


def align_ctx(pd: PmrtData, rows: np.ndarray, keys: Sequence[str] | None = None) -> np.ndarray:
    """Context columns of `rows` in the order of `keys`; a key absent from `pd` gives NaN."""
    if keys is None or list(keys) == list(pd.ctx_keys):
        return pd.ctx[rows]
    out = np.full((len(rows), len(keys)), np.nan)
    ix = {k: j for j, k in enumerate(pd.ctx_keys)}
    for j, k in enumerate(keys):
        if k in ix:
            out[:, j] = pd.ctx[rows, ix[k]]
    return out


def _post_targets(
    pd: PmrtData, rows: np.ndarray, a: np.ndarray, ki: int
) -> tuple[np.ndarray, np.ndarray]:
    post = pd.post[rows, ki]
    okp = np.isfinite(post[:, 0, :])
    p = np.einsum("usc,ucb->usb", a, np.nan_to_num(post)).astype(float)
    p[~np.repeat(okp[:, None, :], a.shape[1], 1)] = np.nan
    return p.reshape(len(rows), -1), okp


def _pre_covariates(
    pd: PmrtData, rows: np.ndarray, a: np.ndarray, hyp: Hyp
) -> tuple[list[np.ndarray], np.ndarray]:
    ud = pd.ud
    _, rel, kpi = hyp
    ki = ud.kpis.index(kpi)
    pre = pd.pre[rows]
    okq = np.isfinite(pre[:, 0, 0, :])
    pre = np.nan_to_num(pre)
    q = np.einsum("usc,ucb->usb", a, pre[:, ki]).astype(float)
    miss = ~okq[:, 3]
    q[miss, :, 3] = 2.0 * q[miss, :, 2]
    cols = [q.reshape(len(rows), -1)]
    relm = ud.rel_mask[rel][rows].astype(F32)
    for j in range(len(ud.kpis)):
        if j == ki:
            continue
        s = np.einsum("uc,ucb->ub", relm, pre[:, j]).astype(float)
        cols.append(np.column_stack([s[:, 0], s[:, 1] + s[:, 2]]))
    if rel != "own":
        own = ud.rel_mask["own"][rows].astype(F32)
        cols.append(np.einsum("uc,ukcb->uk", own, pre[:, :, :, :3]).astype(float))
    return cols, miss


def hyp_design(
    pd: PmrtData,
    rows: np.ndarray,
    hyp: Hyp,
    rho: Mapping[str, np.ndarray],
    ctx_keys: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design of hypothesis (family, relation, KPI) on `rows`.

    Args:
        pd: Unit features.
        rows: Rows of the family, in information order.
        hyp: (family, relation, KPI).
        rho: Receiver score table per family, shape `[C, C]`.
        ctx_keys: Training context keys; None keeps the order of `pd`.

    Returns:
        `P`, the post bins per receiver slot, shape `[n, S * 4]` (NaN where missing); `X`, the
        raw predictable covariates; `okp`, which post bins fit, shape `[n, 4]`.
    """
    ud = pd.ud
    f, rel, kpi = hyp
    a = slot_matrix(ud.c[rows], ud.rel_mask[rel][rows], rho[f], NSLOT[rel])
    p, okp = _post_targets(pd, rows, a, ud.kpis.index(kpi))
    cols, miss = _pre_covariates(pd, rows, a, hyp)
    cols += [
        align_ctx(pd, rows, ctx_keys),
        pd.hist[rows],
        pd.tfrac[rows, None],
        miss[:, None].astype(float),
    ]
    return p, np.column_stack(cols), okp


def _cell_z(c: np.ndarray, v: np.ndarray, y: np.ndarray, n_cells: int) -> np.ndarray:
    z = np.zeros((n_cells, n_cells))
    for cell in range(n_cells):
        m = c == cell
        if m.sum() < 5:
            continue
        vv = float(v[m] @ v[m])
        if vv <= 0:
            continue
        b = v[m] @ y[m] / vv
        res = y[m] - np.outer(v[m], b)
        se = np.sqrt((v[m] ** 2) @ (res**2)) / vv
        z[cell] = np.where(se > 0, b / np.where(se > 0, se, 1), 0.0)
    return z


def receiver_table(pd: PmrtData, family: str, levels: np.ndarray) -> np.ndarray:
    """Receiver scores rho_f, shape `[C, C]`.

    Row c holds the design-centred z of the family's effect, at cells c', on the per-cell load
    change of units at cell c, signed so that the pooled neighbour-load effect is >= 0.
    """
    ud = pd.ud
    rows = ud.rows_of(family)
    li = ud.kpis.index(LOAD)
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows], levels)
    y = np.nan_to_num(pd.post[rows, li, :, :3].sum(-1) - pd.pre[rows, li, :, :3].sum(-1))
    y = demean(y.astype(float), strata(ud, rows))
    c = ud.c[rows]
    z = _cell_z(c, v, y, ud.n_cells)
    nbr = ud.rel_mask["nbr"][rows]
    tot = 0.0
    for cell in range(ud.n_cells):
        m = c == cell
        if m.any():
            tot += float(z[cell][nbr[m].any(0)].sum())
    return (1.0 if tot >= 0 else -1.0) * z


@dataclasses.dataclass(frozen=True)
class _FitRun:
    params: dict[str, Any]
    rho: Mapping[str, np.ndarray]
    levels: np.ndarray
    cfg: PmrtConfig
    log: Callable[[str], None] | None


@dataclasses.dataclass(frozen=True)
class _Family:
    rows: np.ndarray
    v: np.ndarray
    var: np.ndarray
    st: np.ndarray
    load_a: dict[str, np.ndarray]


def _hyp_rng(ud: UnitTable, hyp: Hyp, seed: int) -> np.random.Generator:
    f, rel, kpi = hyp
    idx = [ud.families.index(f), ud.relations.index(rel), ud.kpis.index(kpi)]
    return np.random.default_rng([seed, 6616, 31, *idx])


def _fit_hyp(pd: PmrtData, fam: _Family, hyp: Hyp, run: _FitRun) -> None:
    t = time.time()
    ud, cfg = pd.ud, run.cfg
    _, rel, kpi = hyp
    ep, sg = ud.episode[fam.rows], ud.sgn[fam.rows]
    p, x, okp = hyp_design(pd, fam.rows, hyp, run.rho, run.params["ctx_keys"])
    s = NSLOT[rel]
    la = fam.load_a.get(rel) if kpi != LOAD else None
    train = TrainSet(p, x, okp, fam.v, fam.st)
    sel = cv_select(CvData(train, fam.var, ep, sg), s, cfg, la, _hyp_rng(ud, hyp, cfg.seed))
    core = fit_core(train, s, cfg, la)
    if kpi == LOAD:
        fam.load_a[rel] = core["load_a"]
    k = variant_scores(core, p, x, okp)
    r = center_all(core, k, ep, sg, cfg)
    best = sel["best"]
    bvar, _ = split_arm(best)
    run.params["hyp"]["|".join(hyp)] = {
        "core": core,
        "h": {nm: fit_hmodel(r[nm], x, core, cfg) for nm in r},
        "cv": sel["cv"],
        "best": best,
        "z_in": {nm: lin_z(fam.v, r[nm], fam.var) for nm in r},
        "gb": fit_gb(k[bvar], x, fam.st, core) if cfg.gb else None,
    }
    if run.log:
        cvs = {k2: round(v2, 2) for k2, v2 in sel["cv"].items()}
        name = "|".join(hyp)
        run.log(
            f"fit {name}: n {len(fam.rows)} lam {core['lam']:g} lw "
            f"{core['lw_shrink']:.2f} z_plain {core['z_plain']:.2f} best {best} cv {cvs} "
            f"({time.time() - t:.1f} s)"
        )


def _fit_family(pd: PmrtData, family: str, run: _FitRun) -> None:
    ud = pd.ud
    rows = unit_order(ud, ud.rows_of(family))
    if len(rows) < run.cfg.min_units:
        return
    fam = _Family(
        rows=rows,
        v=v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows], run.levels),
        var=v_var(ud.probs[rows], run.levels),
        st=strata(ud, rows),
        load_a={},
    )
    for rel in ud.relations:
        for kpi in (LOAD, *(k for k in ud.kpis if k != LOAD)):
            _fit_hyp(pd, fam, (family, rel, kpi), run)


def fit_pmrt(
    pd: PmrtData,
    levels: np.ndarray,
    cfg: PmrtConfig | None = None,
    families: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Learns every hypothesis' frozen parameters on training units.

    Args:
        pd: Training unit features (episodes never used for evaluation).
        levels: Design dose of every mode, for the assignment score.
        cfg: Statistic settings; None uses the defaults.
        families: Families to fit; None fits every family of the unit table.
        log: Progress callback.

    Returns:
        Params: `version`, `config`, receiver tables `rho`, training `ctx_keys` and unit
        counts, and per hypothesis `"f|rel|kpi"` the fitted `core`, log-variance models `h`,
        CV scores `cv`, the chosen arm `best`, in-sample z `z_in` and the boosted correction
        `gb`.
    """
    cfg = cfg or PmrtConfig()
    ud = pd.ud
    families = ud.families if families is None else families
    params: dict[str, Any] = {
        "version": PMRT_VERSION,
        "legacy_version": LEGACY_VERSION,
        "config": dataclasses.asdict(cfg),
        "rho": {},
        "hyp": {},
        "n_train_units": int(ud.n),
        "n_train_eps": len(np.unique(ud.episode)),
        "ctx_keys": pd.ctx_keys,
    }
    for f in families:
        params["rho"][f] = receiver_table(pd, f, levels).tolist()
    rho = {f: np.array(params["rho"][f]) for f in params["rho"]}
    run = _FitRun(params, rho, levels, cfg, log)
    for f in families:
        _fit_family(pd, f, run)
    return params


def arm_direction(variant: str, core: Mapping[str, Any]) -> float:
    """Orientation of an arm's sign: 1 for plain-type kernels, else the training kernel sign.

    Returns 0 (use the plain arm's sign) when the training plain effect has |z| < 2.
    """
    if variant in ("pred", *SIGN_FROM_S):
        return 1.0
    if abs(core["z_plain"]) < 2.0:
        return 0.0
    return float(core["sdir"][variant] or 0.0)


def pred_weights(y: np.ndarray, pre: np.ndarray, ep: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """Weights of arm `pred`: the predictable residual of y, zero over the burn-in units."""
    rc = ResidualConfig()
    rp, _ = predictable_residuals(y, pre if np.all(np.isfinite(pre)) else None, ep, sg, rc)
    rp = rp.copy()
    rp[: rc.burn] = 0.0
    return rp


@dataclasses.dataclass(frozen=True)
class _Columns:
    params: Mapping[str, Any]
    rho: Mapping[str, np.ndarray]
    cfg: PmrtConfig
    arms: Sequence[str]


def _gb_weights(
    hp: Mapping[str, Any], k: np.ndarray, x: np.ndarray, ep_sg: tuple, cfg: PmrtConfig
) -> np.ndarray:
    core = hp["core"]
    bvar, bmod = split_arm(hp["best"])
    kg = k - hp["gb"].predict(xs(core, x))
    prior = {1: [core["prior"][bvar][1]], -1: [core["prior"][bvar][-1]]}
    rg = running_center(kg, *ep_sg, Shrinkage(cfg.n0, cfg.n1, prior))
    return modified(rg, bmod, Fitted(core, hp["h"][bvar], cfg), x)


def _target_columns(
    pd: PmrtData, rows: np.ndarray, hyp: Hyp, ctx: _Columns
) -> tuple[np.ndarray, dict[str, Any] | None]:
    ud, cfg = pd.ud, ctx.cfg
    _, rel, kpi = hyp
    hp = ctx.params["hyp"].get("|".join(hyp))
    y, pre = ud.y[(rel, kpi)][rows], ud.pre[(rel, kpi)][rows]
    if hp is None or not np.all(np.isfinite(y)):
        return np.zeros((len(rows), len(ctx.arms))), None
    core = hp["core"]
    ep, sg = ud.episode[rows], ud.sgn[rows]
    p, x, okp = hyp_design(pd, rows, hyp, ctx.rho, ctx.params["ctx_keys"])
    k = variant_scores(core, p, x, okp)
    r = center_all(core, k, ep, sg, cfg)
    best = hp["best"]
    bvar, _ = split_arm(best)
    w = {"pred": pred_weights(y, pre, ep, sg)}
    for nm in VARIANTS:
        for mod in MODS:
            w[nm + mod] = modified(r[nm], mod, Fitted(core, hp["h"][nm], cfg), x)
    w["best"] = w[best]
    w["best_gb"] = (
        _gb_weights(hp, k[bvar], x, (ep, sg), cfg) if hp.get("gb") is not None else w["best"]
    )
    sdir = {
        a: arm_direction(bvar if a in ("best", "best_gb") else split_arm(a)[0], core)
        for a in ctx.arms
    }
    meta = {"best": best, "sdir": sdir, "z_plain_train": core["z_plain"]}
    return np.column_stack([w[a] for a in ctx.arms]), meta


def family_columns(
    pd: PmrtData,
    family: str,
    params: Mapping[str, Any],
    cfg: PmrtConfig,
    arms: Sequence[str] = ARMS,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], list[dict[str, Any] | None]]:
    """Predictable weights of every (target, arm) of one family.

    Returns:
        `rows` in information order; `W`, shape `[n, n_targets * n_arms]`, column
        `t * n_arms + a`; the targets (relation, KPI); per target its meta (`best`, arm sign
        directions `sdir`, `z_plain_train`), or None when it is not testable.
    """
    ud = pd.ud
    rows = unit_order(ud, ud.rows_of(family))
    rho = {g: np.array(params["rho"][g]) for g in params["rho"]}
    ctx = _Columns(params, rho, cfg, tuple(arms))
    targets = [(r, k) for r in ud.relations for k in ud.kpis]
    cols, meta = [], []
    for rel, kpi in targets:
        col, m = _target_columns(pd, rows, (family, rel, kpi), ctx)
        cols.append(col)
        meta.append(m)
    return rows, np.column_stack(cols), targets, meta

"""Nonlinear PMRT statistics on row corpora: the row engine with a statistic T(v).

Statistics: `poly` (whitened quadratic form of polynomial features of v times setpoint
features and w), `rff` (diagonal-standardised random Fourier features of v and the state)
and `gbm` (a matched filter h_t(v) learned on past rows by gradient boosting). Every
statistic is one-sided; the declared sign is sign(sum_t v_t w_t). Adjustment `ridge` uses
the row ridge residual; `gbm` integrates the focal draw out of a past-only boosted model.
"""

import dataclasses
import itertools
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import xgboost as xgb
from scipy.stats import norm

from oran_causal.core.covariates import DESIGNED_KINDS, design_covariates, info_order
from oran_causal.core.data import Dataset, Design, Result
from oran_causal.core.design import Draw, assignment, rowparam
from oran_causal.core.pmrt.candidates import (
    NOT_APPLICABLE,
    RNG_TAG,
    bad_targets,
    build_result,
    candidate_rows,
    concurrent_columns,
    covariate_spec,
    declare_families,
    primary_family,
    tune_tau,
)
from oran_causal.core.pmrt.engine import RedrawConfig, crt_stat
from oran_causal.core.ridge import LAMS, AdjustConfig, burn_rows, predictable_weights, refit_bounds

PMRT_NL_VERSION = "pmrt-nl-v1"
STATISTICS = ("poly", "rff", "gbm")
ADJUSTMENTS = ("ridge", "gbm")
RFF_TAG = 78011


@dataclasses.dataclass(frozen=True)
class PmrtNlConfig:
    """Settings of the nonlinear statistics; the row settings keep their row meaning.

    Attributes:
        statistic: `"poly"`, `"rff"` or `"gbm"`.
        adjust: `"ridge"` or `"gbm"`; None picks gbm for statistic gbm, else ridge.
        B: Maximum number of redraws.
        seq_h: Besag-Clifford stopping count; None for fixed B.
        q: BY level.
        procedure: Declaration procedure.
        hist_lags: Lagged action rows among the covariates.
        concurrent_actions: Adjust for the other designed actions at t.
        covariates: `"eq"` or `"r3"`.
        min_burn: Minimum leading rows with weight 0.
        burn_per_cov: Burn-in rows per covariate.
        growth: Geometric growth of the refit boundaries.
        lams: Ridge lambda grid.
        chunk: Redraws per chunk.
        max_chunk_bytes: Memory cap of one chunk.
        split: RNG split index.
        tau: Secondary score threshold.
        poly_degree: Degree of the polynomial features.
        rff_dv: Random Fourier features of v.
        rff_dx: Random Fourier features of the state.
        gbm_trees: Boosting rounds.
        gbm_depth: Tree depth.
        gbm_eta: Learning rate.
        gbm_grid: Grid points of the learned profile.
        n_quad: Quadrature nodes of a continuous design.
    """

    statistic: str = "gbm"
    adjust: str | None = None
    B: int = 9999
    seq_h: int | None = 20
    q: float = 0.05
    procedure: str = "by"
    hist_lags: int = 2
    concurrent_actions: bool = True
    covariates: str = "eq"
    min_burn: int = 30
    burn_per_cov: float = 1.0
    growth: float = 1.25
    lams: tuple[float, ...] = LAMS
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    split: int = 0
    tau: float | None = None
    poly_degree: int = 3
    rff_dv: int = 10
    rff_dx: int = 20
    gbm_trees: int = 200
    gbm_depth: int = 4
    gbm_eta: float = 0.1
    gbm_grid: int = 25
    n_quad: int = 48

    @property
    def adjustment(self) -> str:
        """Effective adjustment."""
        return self.adjust or ("gbm" if self.statistic == "gbm" else "ridge")


def config_from_dict(cfg: Mapping[str, Any] | None) -> PmrtNlConfig:
    """Config from a dict; unknown keys are ignored and `lams` becomes a tuple."""
    fields = {f.name for f in dataclasses.fields(PmrtNlConfig)}
    kw: dict[str, Any] = {
        k: (tuple(v) if k == "lams" else v) for k, v in dict(cfg or {}).items() if k in fields
    }
    return PmrtNlConfig(**kw)


@dataclasses.dataclass
class Law:
    """Design law of one action with quadrature, every array in information order.

    The raw action value is `off + v`. `nodes` and `wq` (shape `[R, Q]`, R in {1, n}) are a
    quadrature of the centred design. The gbm profile grid is continuous (`glo` to `ghi`)
    or the categorical `support` with `probs`.
    """

    v: np.ndarray
    var: np.ndarray
    draw: Draw | None
    off: np.ndarray
    nodes: np.ndarray
    wq: np.ndarray
    dist: str
    glo: np.ndarray | None = None
    ghi: np.ndarray | None = None
    support: np.ndarray | None = None
    probs: np.ndarray | None = None

    def rows(self, a: int, b: int) -> Law:
        """The law restricted to rows [a, b)."""
        return Law(
            self.v[a:b],
            self.var[a:b],
            None,
            self.off[a:b],
            self.nodes if self.nodes.shape[0] == 1 else self.nodes[a:b],
            self.wq if self.wq.shape[0] == 1 else self.wq[a:b],
            self.dist,
            None if self.glo is None else self.glo[a:b],
            None if self.ghi is None else self.ghi[a:b],
            self.support,
            None
            if self.probs is None
            else (self.probs if self.probs.shape[0] == 1 else self.probs[a:b]),
        )


@dataclasses.dataclass(frozen=True)
class _Ctx:
    v: np.ndarray
    var: np.ndarray
    draw: Draw
    off: np.ndarray
    order: np.ndarray
    n: int
    n_quad: int


def _categorical(ctx: _Ctx, design: Design, dist: dict[str, Any]) -> Law | str:
    o = ctx.order
    vals = np.asarray(dist["values"], float)
    name = dist["name"]
    p = np.asarray(design.propensity if name == "categorical_rows" else dist["p"], float)
    p = p[None, :] if p.ndim == 1 else p[o]
    mean = ctx.off[o]
    if vals.ndim != 1:
        return "per-row categorical values are not supported by pmrt_nl"
    srt = np.argsort(vals, kind="stable")
    per_row = p.shape[0] > 1 or np.ptp(mean) > 0
    nodes = vals[None, :] - mean[:, None] if per_row else vals[None, :] - mean[:1, None]
    wq = np.broadcast_to(p, (nodes.shape[0], len(vals)))
    return Law(
        ctx.v[o],
        ctx.var[o],
        ctx.draw,
        ctx.off[o],
        nodes,
        wq,
        name,
        support=vals[srt],
        probs=p[:, srt],
    )


def _uniform(ctx: _Ctx, dist: dict[str, Any]) -> Law:
    o = ctx.order
    lo, hi = rowparam(dist["lo"], ctx.n)[o], rowparam(dist["hi"], ctx.n)[o]
    xi, w = np.polynomial.legendre.leggauss(ctx.n_quad)
    half = (hi - lo) / 2.0
    r = 1 if np.ptp(half) == 0 else ctx.n
    nodes = half[:r, None] * xi[None, :]
    wq = np.broadcast_to(w / 2.0, nodes.shape)
    return Law(
        ctx.v[o], ctx.var[o], ctx.draw, ctx.off[o], nodes, wq, "uniform", glo=-half, ghi=half
    )


def _normal(ctx: _Ctx) -> Law:
    o = ctx.order
    sc = np.sqrt(ctx.var[o])
    xi, w = np.polynomial.hermite_e.hermegauss(ctx.n_quad)
    r = 1 if np.ptp(sc) == 0 else ctx.n
    nodes = sc[:r, None] * xi[None, :]
    wq = np.broadcast_to(w / w.sum(), nodes.shape)
    return Law(
        ctx.v[o], ctx.var[o], ctx.draw, ctx.off[o], nodes, wq, "normal", glo=-4 * sc, ghi=4 * sc
    )


def _clipped_normal(ctx: _Ctx, dist: dict[str, Any]) -> Law:
    o, n = ctx.order, ctx.n
    m, sd = rowparam(dist["mean"], n)[o], rowparam(dist["sd"], n)[o]
    lo = rowparam(dist.get("lo", -np.inf), n)[o]
    hi = rowparam(dist.get("hi", np.inf), n)[o]
    lo_f, hi_f = np.maximum(lo, m - 8 * sd), np.minimum(hi, m + 8 * sd)
    xi, w = np.polynomial.legendre.leggauss(ctx.n_quad)
    mid, half = (lo_f + hi_f) / 2, (hi_f - lo_f) / 2
    x = mid[:, None] + half[:, None] * xi[None, :]
    dens = norm.pdf((x - m[:, None]) / sd[:, None]) / sd[:, None] * half[:, None] * w[None, :]
    fa, fb = norm.cdf((lo - m) / sd), norm.cdf((hi - m) / sd)
    xs = np.column_stack([lo_f, x, hi_f])
    ws = np.column_stack(
        [np.where(np.isfinite(lo), fa, 0.0), dens, np.where(np.isfinite(hi), 1 - fb, 0.0)]
    )
    ws = ws / ws.sum(1, keepdims=True)
    mean = ctx.off[o]
    r = 1 if (np.ptp(m) == 0 and np.ptp(sd) == 0) else n
    nodes = (xs - mean[:, None])[:r]
    return Law(
        ctx.v[o],
        ctx.var[o],
        ctx.draw,
        ctx.off[o],
        nodes,
        ws[:r],
        "clipped_normal",
        glo=lo_f - mean,
        ghi=hi_f - mean,
    )


def design_law(design: Design, col: np.ndarray, order: np.ndarray, n_quad: int) -> Law | str:
    """Design law of one action column in information order, or the reason it is unsupported."""
    asg = assignment(design, col)
    if isinstance(asg, str):
        return asg
    dist = dict(design.dist or {})
    name = dist.get("name")
    off = np.asarray(col, float) - asg.v
    ctx = _Ctx(asg.v, asg.var, asg.draw, off, order, len(col), n_quad)
    if name in ("categorical", "categorical_rows"):
        return _categorical(ctx, design, dist)
    if name == "uniform":
        return _uniform(ctx, dist)
    if name == "normal":
        return _normal(ctx)
    if name == "clipped_normal":
        return _clipped_normal(ctx, dist)
    return f"design distribution {name!r} not supported by pmrt_nl"


def _expect(law: Law, f_nodes: np.ndarray) -> np.ndarray:
    wq = law.wq.reshape(law.wq.shape + (1,) * (f_nodes.ndim - 2))
    return (wq * f_nodes).sum(1)


class Features:
    """phi(v) of the poly and rff statistics with design means and covariances."""

    def __init__(self, law: Law, kind: str, deg: int, dv: int, ai: int):
        """Features of `kind` (`"poly"` of degree `deg`, or `"rff"` with `dv` features)."""
        self.law, self.kind, self.deg = law, kind, deg
        self.sd = np.sqrt(np.maximum(law.var, 0.0))
        self.sd_safe = np.where(self.sd > 0, self.sd, 1.0)
        if kind == "rff":
            r = np.random.default_rng([RFF_TAG, 1, ai])
            self.om, self.ph = r.standard_normal(dv), r.uniform(0, 2 * np.pi, dv)
        rr = law.nodes.shape[0]
        fn = self.raw(law.nodes, self.sd_safe[:rr, None])
        self.mean = _expect(law, fn)
        outer = fn[..., :, None] * fn[..., None, :]
        self.cov = _expect(law, outer) - self.mean[:, :, None] * self.mean[:, None, :]

    def raw(self, v: np.ndarray, sd: np.ndarray) -> np.ndarray:
        """Uncentred features of v / sd."""
        u = v / sd
        if self.kind == "rff":
            return np.sqrt(2.0) * np.cos(u[..., None] * self.om + self.ph)
        return np.stack([u**j for j in range(1, self.deg + 1)], -1)

    def centred(self, v: np.ndarray) -> np.ndarray:
        """Design-centred features of draws `v` of shape `[..., n]`."""
        f = self.raw(v, self.sd_safe)
        centred = f - self.mean[None] if f.ndim == 3 else f - self.mean
        return centred * (self.sd > 0)[..., None]


def psi_setpoint(design: Design, order: np.ndarray) -> np.ndarray:
    """State features of poly: [1, s, s^2 - mean] for a dither setpoint s, else [1]."""
    n = len(order)
    if design.kind != "dither" or design.fixed_part is None:
        return np.ones((n, 1))
    s = np.asarray(design.fixed_part, float)[order]
    sd = s.std()
    if not np.isfinite(sd) or sd <= 0:
        return np.ones((n, 1))
    s = (s - s.mean()) / sd
    return np.column_stack([np.ones(n), s, s**2 - np.mean(s**2)])


def psi_rff(state: np.ndarray, dx: int, ai: int) -> np.ndarray:
    """State features of the rff statistic: [1, `dx` random Fourier features of the state]."""
    n = state.shape[0]
    if state.shape[1] == 0:
        return np.ones((n, 1))
    sd = state.std(0)
    keep = sd > 0
    x = (state[:, keep] - state[:, keep].mean(0)) / sd[keep]
    if x.shape[1] == 0:
        return np.ones((n, 1))
    r = np.random.default_rng([RFF_TAG, 2, ai, x.shape[1]])
    om = r.standard_normal((x.shape[1], dx)) / math.sqrt(x.shape[1])
    c = r.uniform(0, 2 * np.pi, dx)
    return np.column_stack([np.ones(n), np.sqrt(2.0) * np.cos(x @ om + c)])


def _whitened(cov: np.ndarray, psi: np.ndarray, w2: np.ndarray, k: int) -> list[np.ndarray]:
    if cov.shape[0] == 1:
        p = np.einsum("tl,tm,tk->klm", psi, psi, w2)
        sig = np.einsum("ij,klm->kiljm", cov[0], p)
    else:
        sig = np.einsum("tij,tl,tm,tk->kiljm", cov, psi, psi, w2)
    jl = cov.shape[1] * psi.shape[1]
    sig = sig.reshape(k, jl, jl)
    out = []
    for kk in range(k):
        s = sig[kk]
        ev, q = np.linalg.eigh((s + s.T) / 2)
        ok = ev > 1e-10 * max(ev.max(), 1e-300)
        out.append((q[:, ok] / np.sqrt(ev[ok])).T)
    return out


def _diagonal(cov: np.ndarray, psi: np.ndarray, w2: np.ndarray, k: int) -> list[np.ndarray]:
    cd = np.einsum("tjj->tj", cov)
    if cov.shape[0] == 1:
        d = np.einsum("j,tl,tk->kjl", cd[0], psi**2, w2)
    else:
        d = np.einsum("tj,tl,tk->kjl", cd, psi**2, w2)
    d = d.reshape(k, cov.shape[1] * psi.shape[1])
    return [np.where(dk > 0, 1.0 / np.sqrt(np.where(dk > 0, dk, 1.0)), 0.0) for dk in d]


class FeatureStat:
    """T(V) of poly (whitened quadratic form) and rff (diagonal-standardised sum of squares)."""

    def __init__(self, feats: Features, psi: np.ndarray, w: np.ndarray, whiten: bool):
        """Statistic over weights `w` of shape `[n, k]` and state features `psi`."""
        self.f, self.whiten = feats, whiten
        n, nl = psi.shape
        k = w.shape[1]
        self.k, self.L, self.J = k, nl, feats.mean.shape[1]
        self.M = (psi[:, :, None] * w[:, None, :]).reshape(n, nl * k)
        maker = _whitened if whiten else _diagonal
        self.A = maker(feats.cov, psi, w**2, k)
        self.df = [a.shape[0] if whiten else int((a > 0).sum()) for a in self.A]

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Statistics of draws `v` of shape `[b, n]`, shape `[b, k]`."""
        f = self.f.centred(v)
        u = np.matmul(f.transpose(0, 2, 1), self.M)
        u = (
            u.reshape(len(v), self.J, self.L, self.k)
            .transpose(0, 3, 1, 2)
            .reshape(len(v), self.k, self.J * self.L)
        )
        out = np.empty((len(v), self.k))
        for kk in range(self.k):
            z = u[:, kk] @ self.A[kk].T if self.whiten else u[:, kk] * self.A[kk]
            out[:, kk] = (z * z).sum(1)
        return out


def gbm_fit(x: np.ndarray, y: np.ndarray, cfg: PmrtNlConfig) -> xgb.Booster:
    """Single-thread histogram boosted trees with a fixed seed."""
    params = {
        "max_depth": cfg.gbm_depth,
        "eta": cfg.gbm_eta,
        "nthread": 1,
        "tree_method": "hist",
        "seed": 0,
        "objective": "reg:squarederror",
        "verbosity": 0,
    }
    return xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=cfg.gbm_trees)


def profile_grid(law: Law, g: int) -> np.ndarray:
    """Centred-v grid per row: the categorical support minus the row mean, or `g` points."""
    if law.support is not None:
        return law.support[None, :] - (law.off[:, None])
    if law.glo is None or law.ghi is None:
        raise ValueError("continuous law without a grid range")
    s = np.linspace(0.0, 1.0, g)
    return law.glo[:, None] + (law.ghi - law.glo)[:, None] * s[None, :]


def interp(law: Law, h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Profile values at centred draws: exact support lookup, else linear and clamped."""
    n, g = h.shape
    rows = np.arange(n)
    if law.support is not None:
        mids = (law.support[1:] + law.support[:-1]) / 2
        return h[rows, np.searchsorted(mids, v + law.off)]
    if law.glo is None or law.ghi is None:
        raise ValueError("continuous law without a grid range")
    span = law.ghi - law.glo
    s = np.clip((v - law.glo) / np.where(span > 0, span, 1.0), 0.0, 1.0) * (g - 1)
    i = np.minimum(s.astype(np.int64), g - 2)
    f = s - i
    i += rows * g
    h0 = np.take(h, i)
    return h0 + f * (np.take(h, i + 1) - h0)


def grid_moments(law: Law, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Design mean and variance per row of the interpolated profile `h`."""
    if law.support is not None and law.probs is not None:
        p = np.broadcast_to(law.probs, h.shape)
        m = (p * h).sum(1)
        return m, np.maximum((p * h * h).sum(1) - m * m, 0.0)
    if law.dist == "uniform":
        h0, h1 = h[:, :-1], h[:, 1:]
        m = ((h0 + h1) / 2).mean(1)
        e2 = ((h0 * h0 + h0 * h1 + h1 * h1) / 3).mean(1)
        return m, np.maximum(e2 - m * m, 0.0)
    nodes = (
        law.nodes
        if law.nodes.shape[0] != 1
        else np.broadcast_to(law.nodes, (len(h), law.nodes.shape[1]))
    )
    hn = interp(law, h, nodes.T).T
    wq = np.broadcast_to(law.wq, hn.shape)
    m = (wq * hn).sum(1)
    return m, np.maximum((wq * hn * hn).sum(1) - m * m, 0.0)


@dataclasses.dataclass
class GbmProfiles:
    """Per focal action: centred profile `h` (`[n, G, k]`), design `mean` and `var`."""

    h: dict[int, np.ndarray]
    mean: dict[int, np.ndarray]
    var: dict[int, np.ndarray]
    info: dict[str, int]


@dataclasses.dataclass(frozen=True)
class Focal:
    """A focal action of the gbm profiles: its law and its column in the model input."""

    law: Law
    col: int


def _profile_block(
    model: xgb.Booster, xfull: np.ndarray, focal: Focal, grid: np.ndarray, span: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = span
    law = focal.law
    g = grid[a:b]
    m_, g_ = g.shape
    xr = np.repeat(xfull[a:b], g_, axis=0)
    xr[:, focal.col] = (law.off[a:b, None] + g).reshape(-1)
    hb = model.inplace_predict(xr).reshape(m_, g_)
    mu, var = grid_moments(law.rows(a, b), hb)
    return hb - mu[:, None], mu, var


def gbm_profiles(
    xfull: np.ndarray,
    y: np.ndarray,
    focal: Mapping[int, Focal],
    cfg: PmrtNlConfig,
    bad_t: np.ndarray,
) -> GbmProfiles:
    """Past-only boosted model per target, refitted at the geometric block boundaries."""
    n, d = xfull.shape
    k = y.shape[1]
    burn = burn_rows(AdjustConfig(cfg.min_burn, cfg.burn_per_cov), d)
    grids = {ai: profile_grid(f.law, cfg.gbm_grid) for ai, f in focal.items()}
    out = GbmProfiles(
        {ai: np.zeros((n, g.shape[1], k)) for ai, g in grids.items()},
        {ai: np.zeros((n, k)) for ai in focal},
        {ai: np.zeros((n, k)) for ai in focal},
        {"burn": burn, "n_refits": 0, "d": d},
    )
    if burn >= n:
        return out
    bounds = refit_bounds(burn, n, cfg.growth)
    out.info["n_refits"] = len(bounds) - 1
    for kk in range(k):
        if bad_t[kk]:
            continue
        for a, b in itertools.pairwise(bounds):
            model = gbm_fit(xfull[:a], y[:a, kk], cfg)
            for ai, f in focal.items():
                h, mu, var = _profile_block(model, xfull, f, grids[ai], (a, b))
                out.h[ai][a:b, :, kk], out.mean[ai][a:b, kk], out.var[ai][a:b, kk] = h, mu, var
    return out


class GbmStat:
    """T(V) = sum_t h_t(v_t) w_t / sqrt(sum_t Var(h_t) w_t^2) per target."""

    def __init__(self, law: Law, h: np.ndarray, var: np.ndarray, w: np.ndarray):
        """Statistic of the profiles `h` (`[n, G, k]`) with design variances `var`."""
        self.law, self.w = law, w
        sd = np.sqrt((var * w * w).sum(0))
        self.sd = np.where(sd > 0, sd, 1.0)
        self.k = w.shape[1]
        self.hk = [np.ascontiguousarray(h[:, :, kk]) for kk in range(self.k)]

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Statistics of draws `v` of shape `[b, n]`, shape `[b, k]`."""
        out = np.empty((len(v), self.k))
        for kk in range(self.k):
            out[:, kk] = (interp(self.law, self.hk[kk], v) @ self.w[:, kk]) / self.sd[kk]
        return out


class _NlRun:
    def __init__(self, data: Dataset, cfg: PmrtNlConfig):
        self.data, self.cfg = data, cfg
        self.order = info_order(data)
        y = np.asarray(data.Y, float)[self.order]
        self.bad_t = bad_targets(y)
        self.yf = np.where(np.isfinite(y), y, 0.0)
        self.cov_names, m, _ = design_covariates(
            data, covariate_spec(cfg.covariates, cfg.hist_lags)
        )
        self.xc = m[self.order]
        self.a_ord = np.asarray(data.X_action, float)[self.order]
        self.laws, self.na = self._laws()
        self.gb = self._profiles()
        self.info: dict[str, dict] = {}

    def _laws(self) -> tuple[dict[int, Law], dict[str, str]]:
        laws, na = {}, {}
        for ai, a in enumerate(self.data.action_names):
            col = np.asarray(self.data.X_action, float)[:, ai]
            law = design_law(self.data.designs[ai], col, self.order, self.cfg.n_quad)
            if isinstance(law, str):
                na[a] = law
            elif not np.any(law.var > 0):
                na[a] = "zero design variance"
            else:
                laws[ai] = law
        return laws, na

    def _profiles(self) -> GbmProfiles | None:
        if not self.laws or (self.cfg.adjustment != "gbm" and self.cfg.statistic != "gbm"):
            return None
        designed = [j for j, d in enumerate(self.data.designs) if d.kind in DESIGNED_KINDS]
        use = sorted(set(designed if self.cfg.concurrent_actions else []) | set(self.laws))
        xfull = np.column_stack([self.xc, self.a_ord[:, use]])
        focal = {ai: Focal(law, self.xc.shape[1] + use.index(ai)) for ai, law in self.laws.items()}
        return gbm_profiles(xfull, self.yf, focal, self.cfg, self.bad_t)

    def weights(self, ai: int, a: str) -> np.ndarray:
        if self.cfg.adjustment == "ridge":
            conc = concurrent_columns(self.data, ai) if self.cfg.concurrent_actions else []
            x = np.column_stack([self.xc, self.a_ord[:, conc]]) if conc else self.xc
            adj = AdjustConfig(
                self.cfg.min_burn, self.cfg.burn_per_cov, self.cfg.growth, self.cfg.lams
            )
            w, self.info[a] = predictable_weights(x, self.yf, adj)
        else:
            gb = _require(self.gb)
            w = self.yf - gb.mean[ai]
            w[: gb.info["burn"]] = 0.0
            self.info[a] = dict(gb.info)
        return np.where(self.bad_t[None, :], 0.0, w)

    def statistic(
        self, ai: int, law: Law, w: np.ndarray
    ) -> tuple[Callable[[np.ndarray], np.ndarray], Any]:
        if self.cfg.statistic == "gbm":
            gb = _require(self.gb)
            return GbmStat(law, gb.h[ai], gb.var[ai], w), None
        feats = Features(law, self.cfg.statistic, self.cfg.poly_degree, self.cfg.rff_dv, ai)
        psi = (
            psi_setpoint(self.data.designs[ai], self.order)
            if self.cfg.statistic == "poly"
            else self._rff_psi(ai)
        )
        stat = FeatureStat(feats, psi, w, whiten=self.cfg.statistic == "poly")
        return stat, stat.df

    def _rff_psi(self, ai: int) -> np.ndarray:
        conc = concurrent_columns(self.data, ai) if self.cfg.concurrent_actions else []
        parts = [self.a_ord[:, conc]]
        d = self.data.designs[ai]
        if d.kind == "dither" and d.fixed_part is not None:
            parts.append(np.asarray(d.fixed_part, float)[self.order][:, None])
        if self.data.context is not None:
            parts.append(np.asarray(self.data.context, float).reshape(self.data.n, -1)[self.order])
        return psi_rff(np.column_stack(parts), self.cfg.rff_dx, ai)

    def test_action(self, ai: int, law: Law) -> dict:
        w = self.weights(ai, self.data.action_names[ai])
        stat, df = self.statistic(ai, law, w)
        per_byte = 8 * self.data.n * max(getattr(stat, "J", 1) * 2, len(self.data.kpi_names) + 2)
        chunk = int(max(1, min(self.cfg.chunk, self.cfg.max_chunk_bytes // per_byte)))
        rng = np.random.default_rng([int(self.data.seed), RNG_TAG, ai, int(self.cfg.split)])
        draw, order = _require(law.draw), self.order

        def ordered(g: np.random.Generator, b: int) -> np.ndarray:
            return draw(g, b)[:, order]

        config = RedrawConfig(self.cfg.B, chunk, self.cfg.max_chunk_bytes, self.cfg.seq_h)
        r = crt_stat(stat, law.v, ordered, rng, config)
        r["S"] = law.v @ w
        r["df"] = df
        return r


def _require[T](value: T | None) -> T:
    if value is None:
        raise ValueError("missing gbm profiles or redraw law")
    return value


class PmrtNl:
    """PMRT with a nonlinear statistic on row corpora (`oran_causal.core.data.Method`)."""

    name = "pmrt_nl"
    version = PMRT_NL_VERSION

    def __init__(self, cfg: PmrtNlConfig | None = None):
        """Uses `cfg`, or the defaults."""
        self.cfg = cfg or PmrtNlConfig()

    def tune(self, dev: list[Dataset], truth_free: bool = True) -> dict[str, Any]:
        """Default config plus the placebo-conformal threshold tau over `dev`."""
        return tune_tau(self, dev)[0]

    def run(self, data: Dataset, config: Mapping[str, Any] | None = None) -> Result:
        """Scores every candidate of `data`."""
        t_cpu = time.process_time()
        cfg = config_from_dict(config) if config is not None else self.cfg
        if cfg.statistic not in STATISTICS or cfg.adjustment not in ADJUSTMENTS:
            raise ValueError(f"statistic {cfg.statistic!r} / adjust {cfg.adjustment!r} unknown")
        run = _NlRun(data, cfg)
        per_action = {
            data.action_names[ai]: run.test_action(ai, law) for ai, law in run.laws.items()
        }
        rows, stats = candidate_rows(data, per_action, run.na, run.bad_t, two_sided=False)
        dec = declare_families(stats, data, cfg.procedure, cfg.q)
        edges, reasons, tau_decl = build_result(data, rows, dec, cfg.tau)
        cpu = time.process_time() - t_cpu
        notes = _notes(data, cfg, run, per_action, (reasons, tau_decl, len(stats)))
        return Result(self.name, self.version, edges, cpu, dataclasses.asdict(cfg), notes)


def _notes(
    data: Dataset, cfg: PmrtNlConfig, run: _NlRun, per_action: Mapping[str, dict], tail: tuple
) -> dict[str, Any]:
    reasons, tau_decl, n_tested = tail
    kpis = list(data.kpi_names)
    family, diag = primary_family(data)
    t = {
        f"{s}->{u}": float(per_action[s]["t"][kpis.index(u)])
        for s, u in data.candidates
        if s in per_action
    }
    return {
        NOT_APPLICABLE: reasons,
        "statistic": cfg.statistic,
        "adjust": cfg.adjustment,
        "one_sided": True,
        "sign_rule": "sign(sum_t v_t w_t) (linear PMRT sign on the same w)",
        "T": t,
        "df": {s: per_action[s]["df"] for s in per_action},
        "draws": {s: int(per_action[s]["draws_total"]) for s in per_action},
        "adjust_info": run.info,
        "covariates": {"set": cfg.covariates, "base": list(run.cov_names)},
        "n_tested": n_tested,
        "by_family_m": len(family),
        "by_diag_m": len(diag),
        "declared_tau": tau_decl if cfg.tau is not None else None,
        "rng": f"default_rng([seed, {RNG_TAG}, action_index, {cfg.split}])",
    }

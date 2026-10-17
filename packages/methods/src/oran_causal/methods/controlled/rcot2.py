"""RCoT-v2 conditional independence test (frozen E2 numerics, Strobl et al. null).

Per candidate: RFF residualization on Z (median-heuristic bandwidths, ridge 1e-6), T = n
||Cxy||_F^2, relative residual-collapse guard; columns z-standardized once; per-pair RFF
seed from the citests stream. Primary null (`"lpd4"`, the authors'): 1 - the Lindsay-Pilla-
Basak cdf of the weighted chi^2 with the eigenvalues of the residual product covariance,
falling back to Hall-Buckley-Eagleson, clipped at 0, as RCIT `RCoT.R` (github
ericstrobl/RCIT 7a7fb2b); LPB4 and HBE from `momentchi2` 0.1.8; num_f = 100 conditioning
features. Cross-checks: `block_perm_config()` (block-conditional permutation null, dz 25,
Besag-Clifford up to `B_MAX`) and `native_config()` (block null, B = 299 fixed). A guarded
fraction >= ceil(.05 m) halts the run. Native rule: per-target BH at q = .05.
"""

import math
import types
import warnings
from collections.abc import Callable
from typing import Any

import momentchi2
import momentchi2.utilities as mu
import numpy as np

from oran_causal.core.data import Dataset
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.core.pvalues import BC_H, SequentialP
from oran_causal.methods.controlled._citests import (
    B_MAX,
    CITestBase,
    Prepared,
    method_seed,
    progress,
)
from oran_causal.methods.controlled._rcot import (
    NULL_ANALYTIC,
    RCoTDiscoveryConfig,
    frozen_config_v2,
    pair_seed,
    rcot_pvalue_analytic,
    rcot_statistic,
    residual_collapse,
    rff_residuals,
    standardize,
    z_blocks,
)


class _NumpyWithMath(types.ModuleType):
    """numpy plus the stdlib `math` as `np.math`, which numpy 2 removed."""

    math = math

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


def _momentchi2() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """momentchi2 0.1.8 calls `np.math.factorial`; its module-level `np` gets the proxy."""
    if not hasattr(mu.np, "math"):
        vars(mu)["np"] = _NumpyWithMath("numpy_with_math")
    return momentchi2.hbe, momentchi2.lpb4


def rcot_pvalue_lpd4(
    cand: np.ndarray, target: np.ndarray, z: np.ndarray, cfg: RCoTDiscoveryConfig, seed: int
) -> tuple[float, float, bool, str]:
    """RCoT with RCIT's "lpd4" null (LPB4, HBE fallback).

    Returns:
        (statistic, p, guarded, approximation used).
    """
    hbe, lpb4 = _momentchi2()
    rng = np.random.default_rng(seed)
    fx, fy, rx, ry = rff_residuals(cand, target, z, cfg, rng)
    if residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon):
        return math.nan, 1.0, True, "guard"
    stat, _, prodc = rcot_statistic(rx, ry)
    n = prodc.shape[0]
    w = np.linalg.eigvalsh((prodc.T @ prodc) / n)
    w = w[w > 1e-12]
    if w.size == 0:
        return stat, 1.0, False, "empty"
    how = "lpb4"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = 1.0 - float(lpb4(w, float(stat)))
        if not np.isfinite(p):
            raise ValueError("lpb4 nan")
    except Exception:
        how = "hbe_fallback"
        p = 1.0 - float(hbe(w, float(stat)))
    return stat, max(p, 0.0), False, how


def block_perm_sequential(
    cand: np.ndarray,
    target: np.ndarray,
    z: np.ndarray,
    cfg: RCoTDiscoveryConfig,
    seed: int,
    *,
    h: int | None,
) -> tuple[float, float, bool, int]:
    """Block-conditional permutation null with Besag-Clifford stopping (`h` None: fixed B).

    The candidate is permuted within Z-neighbour blocks and the statistic recomputed with
    the RFF basis fixed by `seed`.

    Returns:
        (statistic, p, guarded, draws used).
    """
    cand = np.asarray(cand, dtype=float)
    target = np.asarray(target, dtype=float)
    n = cand.shape[0]
    zs = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1.0)
    blocks = z_blocks(zs, block=cfg.block_size)

    def _eval(cand_vec: np.ndarray) -> tuple[float, bool]:
        rng = np.random.default_rng(seed)
        fx, fy, rx, ry = rff_residuals(cand_vec, target, z, cfg, rng)
        collapse = residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon)
        return rcot_statistic(rx, ry)[0], collapse

    stat0, guard0 = _eval(cand)
    if guard0:
        return math.nan, 1.0, True, 0
    perm_rng = np.random.default_rng(seed * 7 + 99991)
    seq = SequentialP(stat0, cfg.block_perm_reps, h)
    while not seq.done:
        perm = np.arange(n)
        for blk in blocks:
            perm[blk] = blk[perm_rng.permutation(blk.size)]
        seq.add(_eval(cand[perm])[0])
    return stat0, seq.p, False, seq.k


def bh_per_target(p: np.ndarray, pairs: tuple[tuple[int, int], ...], q: float) -> np.ndarray:
    """BH at `q` within each target's candidates."""
    native = np.zeros(len(pairs), dtype=bool)
    for j in {j for _, j in pairs}:
        es = [e for e, (_, jj) in enumerate(pairs) if jj == j]
        native[es] = weighted_step_up(p[es], None, q, reshape=False)
    return native


def _rcot_config(cfg: dict[str, Any], data: Dataset, method_key: int) -> RCoTDiscoveryConfig:
    base = frozen_config_v2()
    seed = (
        cfg["rng_seed"] if cfg.get("rng_seed") is not None else method_seed(data.seed, method_key)
    )
    rcfg = RCoTDiscoveryConfig(
        dz=cfg["dz"],
        dxy=cfg["dxy"],
        ridge=base.ridge,
        q=base.q,
        null_method=NULL_ANALYTIC if cfg["null_method"] == "lpd4" else cfg["null_method"],
        block_perm_reps=cfg["block_perm_reps"] or base.block_perm_reps,
        block_size=cfg["block_size"],
        residual_epsilon=base.residual_epsilon,
        max_guard_fraction=base.max_guard_fraction,
        rng_seed=seed,
    )
    rcfg.validate()
    return rcfg


class RCoT2Method(CITestBase):
    """RCoT-v2 arm."""

    name = "rcot2"
    version = "xm-rcot-v2-lpd4"
    method_key = 4

    def default_config(self) -> dict[str, Any]:
        """Primary configuration: lpd4 null, dz 100."""
        c = frozen_config_v2()
        return {
            "null_method": "lpd4",
            "block_perm_reps": None,
            "bc_h": None,
            "dz": 100,
            "dxy": c.dxy,
            "block_size": c.block_size,
            "rng_seed": None,
        }

    def block_perm_config(self) -> dict[str, Any]:
        """Cross-check: frozen block permutation null, dz 25, sequential B up to `B_MAX`."""
        c = frozen_config_v2()
        return {
            **self.default_config(),
            "null_method": c.null_method,
            "dz": c.dz,
            "block_perm_reps": B_MAX,
            "bc_h": BC_H,
        }

    def native_config(self) -> dict[str, Any]:
        """Frozen E2 configuration: block null, B = 299 fixed, arm native."""
        reps = frozen_config_v2().block_perm_reps
        return {**self.block_perm_config(), "block_perm_reps": reps, "bc_h": None, "arm": "native"}

    def _one(
        self,
        args: tuple[Any, ...],
        cfg: dict[str, Any],
        rcfg: RCoTDiscoveryConfig,
        approx: dict[str, int],
    ) -> tuple[float, float, bool, int]:
        if cfg["null_method"] == "lpd4":
            st, pv, g, how = rcot_pvalue_lpd4(*args)
            approx[how] = approx.get(how, 0) + 1
            return st, pv, g, 0
        if rcfg.null_method == NULL_ANALYTIC:
            return (*rcot_pvalue_analytic(*args), 0)
        return block_perm_sequential(*args, h=cfg["bc_h"])

    def _test(self, prep: Prepared, data: Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
        rcfg = _rcot_config(cfg, data, self.method_key)
        xs, _, _ = standardize(prep.S)
        ys, _, _ = standardize(prep.Y)
        m = len(prep.pairs)
        stat, p, guarded = np.full(m, np.nan), np.ones(m), np.zeros(m, dtype=bool)
        k_used = np.zeros(m, dtype=int)
        approx: dict[str, int] = {}
        for e, (i, j) in enumerate(prep.pairs):
            args = (xs[:, i], ys[:, j], prep.z_of(xs, i), rcfg, pair_seed(rcfg.rng_seed, j, i))
            st, pv, g, k_used[e] = self._one(args, cfg, rcfg, approx)
            progress(cfg, e + 1, m)
            if g:
                guarded[e] = True
                continue
            stat[e], p[e] = st, pv
        halt = math.ceil(rcfg.max_guard_fraction * m)
        if int(guarded.sum()) >= halt:
            raise ValueError(
                f"rcot2: residual-collapse guard fired on {int(guarded.sum())} of {m} >= {halt} "
                "(HALT)"
            )
        native = bh_per_target(p, prep.pairs, rcfg.q)
        native[guarded] = False
        p[guarded] = np.nan
        return {
            "score": stat,
            "p": p,
            "declared_native": native,
            "not_testable": {int(e): "guard: residual collapse" for e in np.nonzero(guarded)[0]},
            "notes": {
                "n_guarded": int(guarded.sum()),
                "rng_seed": rcfg.rng_seed,
                "mc_draws": k_used.tolist(),
                "null": cfg["null_method"],
                "approximations": approx,
                "native_rule": "per-target BH q=.05 over the target's candidates",
            },
        }

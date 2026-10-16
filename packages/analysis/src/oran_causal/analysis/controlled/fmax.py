"""Simulated null distribution of the number of INVALID cells, `F_max`.

For one arm whose tests all have exactly level .05, under the shared-seed structure of the
campaign: S seeds shared by every cell; data nested in n (null statistics Brownian in n,
correlation `sqrt(n1 / n2)`); the E4 lambdas share the seed's streams (`lambda_corr` of the
dependence table); candidates of one dataset correlated by pair type (dependence table);
worlds and regimes independent. Per cell and rate the analysis' own seed-cluster bootstrap
(the same draws) is applied, so the result includes its coverage error. `F_max` is the
smallest f with `P(#INVALID <= f) >= .95`.

The dependence table is read as data (`load_dependence`).
"""

import dataclasses
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from oran_causal.analysis.controlled.protocol import (
    ALPHA,
    BOOT_REPS,
    BOOT_SEED,
    FMAX_NSIM,
    FMAX_Q,
    FMAX_SEED,
    VALID_UB,
)
from oran_causal.analysis.records import PathLike, file_sha256, lam_of
from oran_causal.core.scoring import PLACEBO, PLACEBO_CONF
from oran_causal.sim.controlled.datasets.truth import truth_for

Dependence = dict[str, Any]
FmaxCell = tuple[str, str, float | None, int]
BATCH = 200
_CACHE: dict[tuple, dict[str, Any]] = {}


def load_dependence(path: PathLike | None) -> Dependence | None:
    """Reads the dependence table and stamps it with its LF sha256 (`_sha`); None if absent."""
    if not path or not Path(path).exists():
        return None
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    table["_sha"] = file_sha256(path)
    return table


def boot_mult(k: int) -> np.ndarray:
    """Returns the `reps x k` multiplicities of the draws `cluster_ci` makes for k clusters."""
    idx = np.random.default_rng(BOOT_SEED).integers(0, k, (BOOT_REPS, k))
    mult = np.zeros((BOOT_REPS, k))
    np.add.at(mult, (np.repeat(np.arange(BOOT_REPS), k), idx.ravel()), 1.0)
    return mult


def _kinds(world: str, regime: str) -> list[tuple[str, str]]:
    truth = truth_for(world, regime)
    out = []
    for source, target in sorted(e for e in truth.null_edges if not e[0].startswith("K")):
        kind = "plac" if source == PLACEBO else "conf" if source == PLACEBO_CONF else "null"
        out.append((f"{source}->{target}", kind))
    return out


def _psd_unit(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    c = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def _candidate_corr(cands: list[tuple[str, str]], row: dict[str, Any] | None) -> np.ndarray:
    m = len(cands)
    c = np.eye(m)
    if row:
        for i, j in itertools.combinations(range(m), 2):
            (si, ti), (sj, tj) = cands[i][0].split("->"), cands[j][0].split("->")
            pair = "same_source" if si == sj else "same_target" if ti == tj else "other"
            c[i, j] = c[j, i] = float(row.get(pair, 0.0))
    return _psd_unit(c)


@dataclasses.dataclass(frozen=True)
class _Group:
    world: str
    regime: str
    cands: list[tuple[str, str]]
    grid: list[tuple[float | None, int]]
    cells: list[int]
    cand_chol: np.ndarray
    grid_chol: np.ndarray


def _lambda_corr(row: dict[str, Any] | None, lams: list[float | None]) -> np.ndarray:
    if row and row.get("lambda_corr") and lams != [None]:
        li = [row["lambdas"].index(x) for x in lams]
        return np.asarray(row["lambda_corr"])[np.ix_(li, li)]
    return np.eye(len(lams))


def _group_spec(
    world: str, regime: str, cells: list[FmaxCell], kinds_of: list[str], dep: Dependence
) -> _Group | None:
    cands = [c for c in _kinds(world, regime) if c[1] in kinds_of]
    if not cands:
        return None
    lams = sorted({c[2] for c in cells}, key=lambda x: -1.0 if x is None else x)
    ns = sorted({c[3] for c in cells})
    row = (dep.get("worlds") or {}).get(f"{world}|{regime}")
    corr_n = np.array([[math.sqrt(min(a, b) / max(a, b)) for b in ns] for a in ns])
    grid = [(lam, n) for lam in lams for n in ns]
    return _Group(
        world,
        regime,
        cands,
        grid,
        [grid.index((c[2], c[3])) for c in cells],
        np.linalg.cholesky(_candidate_corr(cands, row)),
        np.linalg.cholesky(_psd_unit(np.kron(_lambda_corr(row, lams), corr_n))),
    )


def _group_specs(cells: list[FmaxCell], kinds_of: list[str], dep: Dependence) -> list[_Group]:
    by_group: dict[tuple, list[FmaxCell]] = defaultdict(list)
    for c in cells:
        by_group[(c[0], c[1])].append(c)
    specs = (_group_spec(w, r, gc, kinds_of, dep) for (w, r), gc in sorted(by_group.items()))
    return [g for g in specs if g is not None]


@dataclasses.dataclass
class _Tally:
    cell_inv: np.ndarray
    cell_val: np.ndarray
    counts: list[np.ndarray] = dataclasses.field(default_factory=list)
    pooled_valid: list[np.ndarray] = dataclasses.field(default_factory=list)
    pooled_invalid: list[np.ndarray] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _Plan:
    groups: list[_Group]
    kinds_of: list[str]
    seeds: int
    mult_s: np.ndarray
    world_cnt: dict[str, dict[str, float]]

    @property
    def n_cells(self) -> int:
        return sum(len(g.cells) for g in self.groups)


def _make_plan(groups: list[_Group], kinds_of: list[str], seeds: int) -> _Plan:
    world_cnt: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for g in groups:
        for kind in kinds_of:
            m = sum(1 for c in g.cands if c[1] == kind)
            world_cnt[kind][g.world] += m * len(g.cells)
    return _Plan(groups, kinds_of, seeds, boot_mult(seeds), world_cnt)


def _cell_columns(
    plan: _Plan, group: _Group, rej: np.ndarray, pooled_hits: dict[str, dict[str, Any]]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    batch = rej.shape[0]
    inv_cols, val_cols = [], []
    for gi in group.cells:
        lo_any, hi_all = np.zeros(batch, bool), np.ones(batch, bool)
        for kind in plan.kinds_of:
            sel = [i for i, c in enumerate(group.cands) if c[1] == kind]
            if not sel:
                continue
            h = rej[:, :, sel, gi].sum(2).astype(float)
            bs = (h @ plan.mult_s.T) / (len(sel) * plan.seeds)
            lo_any |= np.quantile(bs, 0.025, axis=1) > ALPHA
            hi_all &= np.quantile(bs, 0.975, axis=1) <= VALID_UB
            acc = pooled_hits[kind]
            acc[group.world] = acc.get(group.world, 0.0) + h
        inv_cols.append(lo_any)
        val_cols.append(hi_all & ~lo_any)
    return inv_cols, val_cols


def _pooled_flags(
    plan: _Plan, pooled_hits: dict[str, dict[str, Any]], batch: int
) -> tuple[np.ndarray, np.ndarray]:
    valid, invalid = np.ones(batch, bool), np.zeros(batch, bool)
    for kind in sorted(pooled_hits):
        acc = pooled_hits[kind]
        worlds = sorted(acc)
        hp = np.concatenate([acc[w] for w in worlds], axis=1)
        cnt = np.concatenate([np.full(plan.seeds, plan.world_cnt[kind][w]) for w in worlds])
        mk = boot_mult(len(cnt))
        bs = (hp @ mk.T) / (cnt @ mk.T)
        lo, hi = np.quantile(bs, 0.025, axis=1), np.quantile(bs, 0.975, axis=1)
        invalid |= lo > ALPHA
        valid &= (hi <= VALID_UB) & ~(lo > ALPHA)
    return valid, invalid


def _simulate_batch(
    plan: _Plan, batch: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    zc = stats.norm.ppf(1 - ALPHA / 2)
    inv_cols, val_cols = [], []
    pooled_hits: dict[str, dict[str, Any]] = defaultdict(dict)
    for g in plan.groups:
        e = rng.standard_normal((batch, plan.seeds, len(g.cands), len(g.grid)))
        z = np.einsum("ij,bsjg->bsig", g.cand_chol, e)
        z = np.einsum("gh,bsih->bsig", g.grid_chol, z)
        inv, val = _cell_columns(plan, g, np.abs(z) > zc, pooled_hits)
        inv_cols += inv
        val_cols += val
    return np.column_stack(inv_cols), np.column_stack(val_cols), pooled_hits


def _smallest_f(counts: np.ndarray) -> int:
    f = 0
    while np.mean(counts <= f) < FMAX_Q:
        f += 1
    return f


def _summarise(
    tally: _Tally, order: list[tuple[str, str, int]], seeds: int, keys: Sequence[str], nsim: int
) -> dict[str, Any]:
    counts = np.concatenate(tally.counts)
    pv, pi = np.concatenate(tally.pooled_valid), np.concatenate(tally.pooled_invalid)
    f = _smallest_f(counts)
    n_cells = len(order)
    by_world = defaultdict(list)
    for (w, _r, _gi), v in zip(order, tally.cell_val / nsim, strict=True):
        by_world[w].append(v)
    return {
        "n_cells": n_cells,
        "S": int(seeds),
        "keys": list(keys),
        "nsim": nsim,
        "f_max": int(f),
        "p_count_le_fmax": float(np.mean(counts <= f)),
        "p_pooled_valid": float(pv.mean()),
        "p_pass": float(np.mean((counts <= f) & pv)),
        "p_pooled_invalid": float(pi.mean()),
        "p_cell_invalid": float(tally.cell_inv.sum() / (nsim * max(1, n_cells))),
        "p_cell_valid_by_world": {w: float(np.mean(v)) for w, v in sorted(by_world.items())},
        "count_dist": {str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())},
    }


def _run(plan: _Plan, nsim: int, seed: int) -> _Tally:
    rng = np.random.default_rng(seed)
    tally = _Tally(np.zeros(plan.n_cells), np.zeros(plan.n_cells))
    done = 0
    while done < nsim:
        batch = min(BATCH, nsim - done)
        inv, val, pooled_hits = _simulate_batch(plan, batch, rng)
        tally.counts.append(inv.sum(1))
        tally.cell_inv += inv.sum(0)
        tally.cell_val += val.sum(0)
        pv, pi = _pooled_flags(plan, pooled_hits, batch)
        tally.pooled_valid.append(pv)
        tally.pooled_invalid.append(pi)
        done += batch
    return tally


def fmax_simulate(  # noqa: PLR0913
    cells: Iterable[tuple[str, str, Any, int]],
    seeds: int,
    keys: Sequence[str],
    dep: Dependence | None,
    *,
    nsim: int = FMAX_NSIM,
    seed: int = FMAX_SEED,
) -> dict[str, Any]:
    """Simulates the number of INVALID cells of a valid arm.

    Args:
        cells: `(world, regime, lam, n)` cells of the arm.
        seeds: Seeds S shared by every cell.
        keys: Rate keys of the component; their kinds (`null`, `plac`, `conf`) select candidates.
        dep: Dependence table (`load_dependence`), or None for independent candidates.
        nsim: Simulation replicates.
        seed: Simulation seed.

    Returns:
        `f_max`, the probabilities `p_count_le_fmax`, `p_pooled_valid`, `p_pass`,
        `p_pooled_invalid`, `p_cell_invalid`, the per-world cell validity and the count
        distribution. Results are memoised per argument tuple.
    """
    dep = dep or {}
    cell_list = sorted(
        {(w, r, lam_of(lam), int(n)) for w, r, lam, n in cells},
        key=lambda c: (c[0], c[1], -1.0 if c[2] is None else c[2], c[3]),
    )
    cache_key = (tuple(cell_list), int(seeds), tuple(keys), dep.get("_sha"), nsim, seed)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    kinds_of = sorted({k.split("_")[0] for k in keys})
    groups = _group_specs(cell_list, kinds_of, dep)
    order = [(g.world, g.regime, gi) for g in groups for gi in g.cells]
    plan = _make_plan(groups, kinds_of, seeds)
    out = _summarise(_run(plan, nsim, seed), order, seeds, keys, nsim)
    _CACHE[cache_key] = out
    return out

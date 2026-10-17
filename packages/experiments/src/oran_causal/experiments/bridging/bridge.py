"""The frozen Study 1 (v4) collection as row-corpus `Dataset`s (protocol EXP_B section 2).

Read-only on everything frozen. Two steps:

1. `build_table`: the v4 analyzer's own loaders (`study1_discovery.build_caches` ->
   `pmrt.load_pmrt_pool` (H 90, H_pre 90) -> `pmrt.pmrt_data`) give the unit table of every
   stage, written as one compact npz per stage (`E6UNITS_SCHEMA`). Nothing is re-simulated.
2. `make_dataset`: one `Dataset` per (slice, family), the units of that family in the slice's
   episodes; `truth_from_gt` gives its `Truth` from the stored v4 GT cells.

Mapping (one row = one randomised unit of family f, rows in information order (seed, t0)):

- action `A_<f>` = sgn x L, L = `LEVEL_V2[logged mode]`: the signed applied change the arbiter's
  draw allows (intention to treat: the logged mode). Design `logged`, `categorical_rows` over
  (-1, 0, 1) with per-row probabilities [P(L=1) 1{sgn<0}, P(L=0), P(L=1) 1{sgn>0}].
- placebo `P_placebo` = sgn x Bernoulli(P(L=1)), the same design drawn independently
  (`default_rng([7800, 6, family index, dataset seed])`): the truth-free negative control.
- KPIs: the 15 targets `K_<rel>_<kpi>`. `X_kpi_lag` is the pre-window sum over [t0 - 90, t0),
  `Y` the post-window sum (pre + the unit outcome). A lag column that is a linear combination of
  the earlier ones is set all-NaN (`meta["lag_dropped_collinear"]`): E6 holds the UE population
  fixed, so own + nbr + far load is constant and the far-load lag is determined. Its target stays.
- context: every numeric observation-only ctx key at t0, `hist` (past design-centred modes of
  every family at the own cell and the exposure neighbours), sgn and t0 / T, encoded per
  dataset by `encode_context`.
- time: `time_index` = 0..n-1 in information order, no gap rows.
- candidates: (A_f, K) and (P_placebo, K) for the 15 targets.

Truth: GT cell TRUE -> edge with the GT sign; NULL -> null edge (an equivalence label, not a
sharp null); INDET -> unscored; P_placebo -> K null. On the placebo logs (accept applied whatever
the logged mode) every candidate is a null edge.
"""

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.bridging.plan import (
    KPI_NAMES,
    PLACEBO,
    SEEDS_V4,
    TARGETS,
    WORLD,
    Hyp,
    action_name,
    dataset_seed,
)
from oran_causal.core.data import Dataset, Design, Truth
from oran_causal.core.pmrt.units import PmrtData
from oran_causal.experiments.system.study1_discovery import build_caches, find_records
from oran_causal.methods.system import pmrt
from oran_causal.methods.system.mscr_crt import LEVEL_V2_ARR
from oran_causal.sim.system.decision.vocab import FAMILIES

E6UNITS_SCHEMA = "xm-e6units/1"
VALUES = (-1.0, 0.0, 1.0)
RNG_TAG_HARNESS = 7800
INFO_JSON = "e6units_info.json"


def unit_table(pd: PmrtData) -> dict[str, np.ndarray]:
    """Compact arrays of a `PmrtData` (all families, every unit of the H 90 / H_pre 90 set)."""
    ud = pd.ud
    if tuple(ud.families) != FAMILIES:
        raise ValueError(f"unit-table families {ud.families} != {FAMILIES}")
    p1 = np.asarray(ud.probs, float) @ LEVEL_V2_ARR
    keep_ctx = [j for j in range(pd.ctx.shape[1]) if np.isfinite(pd.ctx[:, j]).any()]
    return {
        "schema": np.array(E6UNITS_SCHEMA),
        "seed": np.asarray(ud.seed, np.int64),
        "episode": np.asarray(ud.episode, np.int64),
        "family": np.asarray(ud.family, np.int8),
        "t0": np.asarray(ud.t0, np.int32),
        "c": np.asarray(ud.c, np.int16),
        "sgn": np.asarray(ud.sgn, np.float32),
        "mode": np.asarray(ud.mode, np.int8),
        "level": LEVEL_V2_ARR[np.asarray(ud.mode, int)].astype(np.float32),
        "p1": p1.astype(np.float64),
        "probs": np.asarray(ud.probs, np.float64),
        "pre": np.column_stack([ud.pre[t] for t in TARGETS]).astype(np.float64),
        "y": np.column_stack([ud.y[t] for t in TARGETS]).astype(np.float64),
        "ctx": pd.ctx[:, keep_ctx].astype(np.float64),
        "ctx_keys": np.array([pd.ctx_keys[j] for j in keep_ctx]),
        "hist": np.asarray(pd.hist, np.float64),
        "tfrac": np.asarray(pd.tfrac, np.float64),
        "families": np.array(FAMILIES),
        "targets": np.array([f"{r}|{k}" for r, k in TARGETS]),
    }


def _files(src: str | Sequence[str | Path]) -> list[Path]:
    specs = [s for s in src.split(",") if s] if isinstance(src, str) else list(src)
    return find_records(specs)


def build_table(
    sources: Mapping[str, str | Sequence[str | Path]], out_dir: str | Path, workers: int = 4
) -> dict[str, Any]:
    """Writes `<out_dir>/e6units_<stage>.npz` per stage and `e6units_info.json`.

    Args:
        sources: {stage: v4 collection dirs or JSONL files, a list or a comma-separated string}.
        out_dir: Output directory; the per-file caches go to `<out_dir>/cache/<stage>`.
        workers: Cache-building processes.

    Returns:
        {stage: {n_units, n_episodes, seed_range, complete, subs, by_family, pi0_rows, file}}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for stage, src in sources.items():
        cache = build_caches(_files(src), stage, out / "cache" / stage, stage, workers)
        pool = pmrt.load_pmrt_pool([str(p) for p in cache], stages=[stage])
        pd = pmrt.pmrt_data(pool)
        tab: dict[str, Any] = unit_table(pd)
        tab["stage"] = np.array(stage)
        path = out / f"e6units_{stage}.npz"
        np.savez_compressed(path, **tab)
        lo, n = SEEDS_V4[stage]
        seeds = sorted({int(s) for s in pool.pool.eps["seed"]})
        info[stage] = {
            "n_units": int(pd.n),
            "n_episodes": int(pool.pool.n_eps),
            "seed_range": [seeds[0], seeds[-1]],
            "complete": seeds == list(range(lo, lo + n)),
            "subs": sorted(set(map(str, pool.pool.eps["sub"]))),
            "by_family": {f: int((tab["family"] == i).sum()) for i, f in enumerate(FAMILIES)},
            "pi0_rows": np.unique(tab["probs"], axis=0).tolist(),
            "file": path.name,
        }
    with (out / INFO_JSON).open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    return info


@dataclasses.dataclass(frozen=True)
class Table:
    """A loaded unit table of one stage.

    Attributes:
        a: The npz arrays by name.
        stage: Record stage, `eval` or `placebo`.
        ctx_keys: Context column names.
    """

    a: dict[str, np.ndarray]
    stage: str
    ctx_keys: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> Table:
        """Reads an `e6units_<stage>.npz`.

        Raises:
            ValueError: The file's schema is not `E6UNITS_SCHEMA`.
        """
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        if str(a["schema"]) != E6UNITS_SCHEMA:
            raise ValueError(f"{path}: schema {a['schema']} != {E6UNITS_SCHEMA}")
        return cls(a, str(a["stage"]), tuple(str(x) for x in a["ctx_keys"]))

    def rows(self, family: str, seeds: Iterable[int]) -> np.ndarray:
        """Rows of `family` in the episodes `seeds`, in information order (seed, t0, row)."""
        a = self.a
        m = (a["family"] == FAMILIES.index(family)) & np.isin(
            a["seed"], np.asarray(list(seeds), np.int64)
        )
        idx = np.nonzero(m)[0]
        return idx[np.lexsort((idx, a["t0"][idx], a["seed"][idx]))]


def full_rank_columns(cols: Sequence[np.ndarray], tol: float = 1e-7) -> list[int]:
    """Indices of a full-rank subset (with an intercept), greedy in the given order.

    A column is dropped if it is constant or a linear combination of the intercept and the
    columns kept before it (relative residual variance <= `tol`; 1e-7 sits above pcorr's own
    near-collinearity stop, 1e-8 on standardised columns). Truth-free, fixed order.
    """
    if not len(cols):
        return []
    n = len(cols[0])
    q = np.ones((n, 1)) / math.sqrt(n)
    keep = []
    for i, c in enumerate(cols):
        x = np.asarray(c, float) - np.mean(c)
        v = float(x @ x)
        if v <= 0.0:
            continue
        r = x - q @ (q.T @ x)
        r = r - q @ (q.T @ r)
        if float(r @ r) <= tol * v:
            continue
        q = np.column_stack([q, r / math.sqrt(float(r @ r))])
        keep.append(i)
    return keep


def drop_near_collinear(
    cols: Sequence[np.ndarray], fixed: Sequence[int], cand: Sequence[int], tol: float = 1e-6
) -> list[int]:
    """Second pass after `full_rank_columns`.

    While some column of fixed + cand has standardised residual variance given all the others,
    1 / diag(R^-1), below `tol`, drops the `cand` column with the smallest such value (ties:
    the later one). Stops when the dependence is among fixed columns only.

    Returns:
        The kept `cand` indices.
    """
    cand = list(cand)
    while cand:
        idx = [*fixed, *cand]
        x = np.column_stack([np.asarray(cols[i], float) for i in idx])
        x = (x - x.mean(0)) / x.std(0)
        r = (x.T @ x) / len(x)
        d = 1.0 / np.clip(np.diag(np.linalg.pinv(r, hermitian=True)), 1e-300, None)
        if d.min() >= tol:
            break
        dc = d[len(fixed) :]
        j = int(np.flatnonzero(dc == dc.min())[-1])
        if dc[j] >= tol:
            break
        cand.pop(j)
    return cand


def encode_context(
    m: np.ndarray, names: Sequence[str], base: Sequence[np.ndarray] = ()
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Per-dataset context encoding.

    An all-NaN column is dropped; a partly missing one is 0-filled and followed by its
    indicator `<name>:missing`. Then only the columns that add rank to (intercept, `base`, the
    earlier context columns) are kept (`full_rank_columns`, then `drop_near_collinear`), so
    every method's conditioning design is non-singular.

    Args:
        m: Raw context, shape `[n, len(names)]`.
        names: Column names.
        base: Columns that come first and are always kept (actions, kept lags).

    Returns:
        The encoded context `[n, c]` and its column names.
    """
    cols: list[np.ndarray] = []
    out: list[str] = []
    for j, nm in enumerate(names):
        x = np.asarray(m[:, j], float)
        bad = ~np.isfinite(x)
        if bad.all():
            continue
        if bad.any():
            cols += [np.where(bad, 0.0, x), bad.astype(float)]
            out += [nm, f"{nm}:missing"]
        else:
            cols.append(x)
            out.append(nm)
    nb = len(base)
    allc = [*base, *cols]
    keep = [i for i in full_rank_columns(allc) if i >= nb]
    keep = [i - nb for i in drop_near_collinear(allc, list(range(nb)), keep)]
    enc = np.column_stack([cols[i] for i in keep]) if keep else np.zeros((len(m), 0))
    return enc, tuple(out[i] for i in keep)


def context_names(ctx_keys: Sequence[str]) -> list[str]:
    """Raw context column names: ctx keys, hist (own, nbr) x family, sgn, tfrac."""
    return (
        [f"ctx_{k}" for k in ctx_keys]
        + [f"hist_{w}_{f}" for w in ("own", "nbr") for f in FAMILIES]
        + ["sgn", "tfrac"]
    )


def make_dataset(
    tab: Table, family: str, seeds: Iterable[int], split: int, slice_name: str
) -> Dataset:
    """The dataset of `family` in the episodes `seeds` (see the module docstring)."""
    a = tab.a
    r = tab.rows(family, seeds)
    n = len(r)
    sgn = a["sgn"][r].astype(float)
    p1 = a["p1"][r]
    act = sgn * a["level"][r].astype(float) + 0.0
    prop = np.column_stack([p1 * (sgn < 0), 1.0 - p1, p1 * (sgn > 0)])
    dseed = dataset_seed(split, family)
    rng = np.random.default_rng([RNG_TAG_HARNESS, 6, FAMILIES.index(family), dseed])
    plac = sgn * (rng.random(n) < p1).astype(float) + 0.0
    design = Design("logged", {"name": "categorical_rows", "values": list(VALUES)}, propensity=prop)
    ep = a["seed"][r]
    post = a["pre"][r] + a["y"][r]
    pre = a["pre"][r].copy()
    keep_lag = full_rank_columns([pre[:, j] for j in range(pre.shape[1])])
    dropped = [j for j in range(pre.shape[1]) if j not in keep_lag]
    pre[:, dropped] = np.nan
    ctx, cnames = encode_context(
        np.column_stack([a["ctx"][r], a["hist"][r], sgn, a["tfrac"][r]]),
        context_names(tab.ctx_keys),
        base=[act, plac] + [pre[:, j] for j in keep_lag],
    )
    an = action_name(family)
    cands = tuple((s, k) for s in (an, PLACEBO) for k in KPI_NAMES)
    meta = {
        "source": "e6p-v4",
        "stage": tab.stage,
        "slice": slice_name,
        "split": int(split),
        "family": family,
        "n_episodes": len(np.unique(ep)),
        "seed_range": [int(ep.min()), int(ep.max())] if n else None,
        "context_names": cnames,
        "lag_dropped_collinear": [KPI_NAMES[j] for j in dropped],
        "primary_candidates": cands,
        "secondary_candidates": (),
        "diagnostic_candidates": (),
        "placebo_conf": None,
    }
    return Dataset(
        world=WORLD,
        regime=family,
        n=n,
        seed=dseed,
        action_names=(an, PLACEBO),
        kpi_names=KPI_NAMES,
        X_action=np.column_stack([act, plac]),
        X_kpi_lag=pre,
        Y=post,
        designs=(design, design),
        candidates=cands,
        time_index=np.arange(n, dtype=np.int64),
        context=ctx,
        meta=meta,
    )


def truth_from_gt(
    gt: Mapping[Hyp, Mapping[str, Any]], family: str, placebo_log: bool = False
) -> Truth:
    """The `Truth` of a family's dataset from the v4 GT cells (all null on the placebo logs)."""
    an = action_name(family)
    edges: set[tuple[str, str]] = set()
    signs: dict[tuple[str, str], int] = {}
    nulls = {(PLACEBO, k) for k in KPI_NAMES}
    for (rel, kpi), name in zip(TARGETS, KPI_NAMES, strict=True):
        e = (an, name)
        if placebo_log:
            nulls.add(e)
            continue
        c = gt[(family, rel, kpi)]
        if c["status"] == "TRUE":
            edges.add(e)
            if c.get("sign") in (1, -1):
                signs[e] = int(c["sign"])
        elif c["status"] == "NULL":
            nulls.add(e)
    return Truth(
        world=WORLD, regime=family, edges=frozenset(edges), signs=signs, null_edges=frozenset(nulls)
    )

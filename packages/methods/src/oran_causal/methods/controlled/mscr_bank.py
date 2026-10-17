"""MSCR-v2: max-stratified correlation-ratio test and its shared permutation bank.

For a target y, candidate column i and each other column g: split rows into `nc` equal-count
strata of g (strata below `min_stratum` rows dropped) and each stratum into `nb` equal-count
bins of i; S_g = sum_s BSS_s / sum_s TSS_s of y and S* = max_g S_g. Null: within each
g-stratum, `n_perm` random equal-size partitions of y; under equal-count binning these draws
depend on the candidate only through the bin sizes, so one bank per (target, g) is shared
across candidates, and S*_null = max_{g != i} of the bank draws. Selection (v2): per target
BY at q over the parameter candidates.

Scope (from the v2 review): validated on randomized designs only (exogenous, independently
randomized parameters, noiseless E2, n = 4000, the frozen constants); no general FDR or
conditional-independence guarantee; fails on single-parameter-actuation designs; one
conditioner at a time; mean-shift dependence only; weak co-parent edges can be missed.

Threads, device and block size never change a result bit: each target consumes its own
stream exactly as v2 did. Verified with numpy 2.4.2 (argsort tie order and reduceat
summation order fix the bits). `MSCR_NUM_THREADS` (then `OMP_NUM_THREADS`) caps threads and
`MSCR_DEVICE` selects the bank device.
"""

import dataclasses
import os
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast

import numpy as np

from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.core.pvalues import sequential_p_from_draws

MSCR_VERSION = "mscr-v2"
BLOCK_ELEMS = 1 << 18

Plan = list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
Task = tuple[int, int, int, int, int]
Bank = tuple[np.ndarray, np.ndarray, list[list[tuple[np.ndarray, np.ndarray]]]]
RowSums = list[list[list[np.ndarray]]]


@dataclasses.dataclass(frozen=True)
class MSCRConfig:
    """MSCR constants (frozen v2 values).

    Attributes:
        nc: Equal-count strata of the conditioner.
        nb: Equal-count bins of the candidate within a stratum.
        min_stratum: Strata with fewer rows are dropped.
        n_perm: Permutations per (target, conditioner) bank.
        q: BY level per target.
    """

    nc: int = 6
    nb: int = 8
    min_stratum: int = 40
    n_perm: int = 2999
    q: float = 0.05


def frozen_config() -> MSCRConfig:
    """The confirmed v2 constants."""
    return MSCRConfig()


@dataclasses.dataclass(frozen=True)
class MSCRResult:
    """Output of `discover_mscr`.

    Attributes:
        pvals: Permutation p-values, `[n_targets, n_params]`.
        s_star: Observed S*, `[n_targets, n_params]`.
        declared: Per-target BY selection.
        config: Constants used.
        version: Method version.
    """

    pvals: np.ndarray
    s_star: np.ndarray
    declared: np.ndarray
    config: MSCRConfig
    version: str = MSCR_VERSION

    def param_edges(self) -> set[tuple[int, int]]:
        """Declared edges as (kpi index, param index)."""
        return {(int(k), int(p)) for k, p in zip(*np.nonzero(self.declared), strict=True)}


def equal_count_labels(values: np.ndarray, nbins: int) -> np.ndarray:
    """Rank-based equal-count bin labels 0..nbins-1 (sizes differ by at most 1, stable ties)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return (ranks * nbins) // len(values)


def _between_ss(y: np.ndarray, labels: np.ndarray, nbins: int) -> float:
    total = y.sum()
    tb = np.bincount(labels, weights=y, minlength=nbins)
    nb = np.bincount(labels, minlength=nbins).astype(np.float64)
    mask = nb > 0
    return float(np.sum(tb[mask] ** 2 / nb[mask]) - total * total / len(y))


def resolve_n_jobs(n_jobs: int | None = None) -> int:
    """Worker threads: explicit, else `MSCR_NUM_THREADS`, `OMP_NUM_THREADS`, the usable CPUs."""
    if n_jobs is not None:
        return max(1, int(n_jobs))
    for var in ("MSCR_NUM_THREADS", "OMP_NUM_THREADS"):
        val = os.environ.get(var, "").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    return max(1, os.process_cpu_count() or 1)


def resolve_device(device: str | None = None) -> str:
    """Bank device: explicit, else `MSCR_DEVICE`, else a self-checked CUDA device, else `"cpu"`.

    Accepted: `"auto"`, `"cpu"` (numpy), `"cuda"`, `"cuda:<k>"`, `"torch:<torch device>"`.

    Raises:
        ValueError: Unknown device, or a torch device that is unavailable.
    """
    if device is None or device == "auto":
        device = os.environ.get("MSCR_DEVICE", "auto").strip() or "auto"
    if device == "cpu":
        return device
    if device != "auto":
        return _validate_torch_device(device)
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        from oran_causal.methods.controlled.mscr_torch import device_selfcheck

        if device_selfcheck("cuda"):
            return "cuda"
    return "cpu"


def _validate_torch_device(device: str) -> str:
    if not (device.startswith("cuda") or device.startswith("torch:")):
        raise ValueError(
            f"MSCR device must be 'auto', 'cpu', 'cuda[:k]' or 'torch:<device>', got {device!r}"
        )
    try:
        import torch
    except ImportError as e:
        raise ValueError(f"MSCR device {device!r} needs torch, which is not installed") from e
    try:
        dev = torch.device(device.removeprefix("torch:"))
    except RuntimeError as e:
        raise ValueError(f"invalid MSCR torch device {device!r}: {e}") from e
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"MSCR device {device!r} requested but CUDA is not available")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise ValueError(
                f"MSCR device {device!r}: only {torch.cuda.device_count()} CUDA device(s)"
            )
    return device


def strata_plan(x: np.ndarray, cfg: MSCRConfig) -> Plan:
    """Per conditioner g, the kept strata as (rows, bin sizes, bin offsets); depends on x only."""
    plan = []
    for g in range(x.shape[1]):
        lab = equal_count_labels(x[:, g], cfg.nc)
        kept = []
        for s in range(cfg.nc):
            rows = np.nonzero(lab == s)[0]
            if len(rows) < cfg.min_stratum:
                continue
            ns = len(rows)
            sizes = np.bincount((np.arange(ns) * cfg.nb) // ns, minlength=cfg.nb).astype(np.int64)
            kept.append((rows, sizes, np.concatenate(([0], np.cumsum(sizes)[:-1]))))
        plan.append(kept)
    return plan


def stream_tasks(plan: Plan, n_perm: int, block_elems: int) -> tuple[list[Task], int]:
    """Bank tasks (g, kept stratum, lo, hi, first draw index) in v2 stream order, and total draws.

    v2 draws `rng.random((n_perm, ns))` per (g, kept stratum) in that order, one 64-bit output
    per double in C order, so a row block of a stratum is a contiguous stream segment.
    """
    tasks, pos = [], 0
    for g, kept in enumerate(plan):
        for k, (rows, _, _) in enumerate(kept):
            ns = len(rows)
            step = max(1, block_elems // ns)
            for lo in range(0, n_perm, step):
                hi = min(lo + step, n_perm)
                tasks.append((g, k, lo, hi, pos))
                pos += (hi - lo) * ns
    return tasks, pos


def jumpable(rng: np.random.Generator) -> bool:
    """True for a generator with jump-ahead (PCG64, PCG64DXSM)."""
    return type(rng.bit_generator) in (np.random.PCG64, np.random.PCG64DXSM)


def uniforms_at(state: Any, bg_type: type, start: int, shape: tuple[int, int]) -> np.ndarray:
    """`shape` uniforms from draw index `start` of the stream with state `state`."""
    bg: np.random.PCG64 = bg_type()
    bg.state = state
    bg.advance(start)
    return np.random.Generator(bg).random(shape)


def finish_stream(rng: np.random.Generator, state: Any, total: int) -> None:
    """Leaves `rng` where `total` sequential draws from `state` would, keeping its 32-bit buffer."""
    bg = cast("np.random.PCG64", type(rng.bit_generator)())
    bg.state = state
    bg.advance(total)
    end = dict(bg.state)
    end["has_uint32"], end["uinteger"] = state["has_uint32"], state["uinteger"]
    rng.bit_generator.state = end


def _null_rows(
    y_s: np.ndarray, u: np.ndarray, offsets: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    perm = np.argsort(u, axis=1)
    groupsums = np.add.reduceat(y_s[perm], offsets, axis=1)
    return np.sum(groupsums**2 / sizes[None, :], axis=1)


def _task_shape(plan: Plan, t: Task) -> tuple[int, int]:
    return (t[3] - t[2], len(plan[t[0]][t[1]][0]))


@dataclasses.dataclass(frozen=True)
class Execution:
    """Execution settings of the bank; they never change a result bit.

    Attributes:
        n_jobs: Worker threads; None resolves through `resolve_n_jobs`.
        device: Bank device; None resolves through `resolve_device`.
        block_elems: Uniforms per task block; None uses the device default.
    """

    n_jobs: int | None = None
    device: str | None = None
    block_elems: int | None = None


def _parallel_rowsums(
    rngs: list[np.random.Generator],
    work: tuple[list[Task], int],
    n_jobs: int,
    run: Callable[[int, Task, np.ndarray], None],
    shape: Callable[[Task], tuple[int, int]],
) -> None:
    tasks, total = work

    def run_jump(j: int, t: Task, state: Mapping[str, Any], bgt: type) -> None:
        run(j, t, uniforms_at(state, bgt, t[4], shape(t)))

    with ThreadPoolExecutor(n_jobs) as pool:
        futs: list[Future[None]] = []
        window: deque[Future[None]] = deque()
        for j, rng in enumerate(rngs):
            if jumpable(rng):
                state, bgt = rng.bit_generator.state, type(rng.bit_generator)
                futs += [pool.submit(run_jump, j, t, state, bgt) for t in tasks]
                finish_stream(rng, state, total)
                continue
            for t in tasks:
                window.append(pool.submit(run, j, t, rng.random(shape(t))))
                futs.append(window[-1])
                if len(window) >= 2 * n_jobs:
                    window.popleft().result()
        for f in futs:
            f.result()


def rowsums_cpu(
    ys: list[list[list[np.ndarray]]],
    plan: Plan,
    cfg: MSCRConfig,
    rngs: list[np.random.Generator],
    run: Execution,
) -> RowSums:
    """rowsums[j][g][k]: per-draw sum_k (group sum)^2 / size for target j, numpy kernel.

    Serial runs draw in stream order; parallel runs regenerate each task's stream segment by
    PCG64 jump-ahead, or draw on the caller for generators without it.
    """
    n_jobs = resolve_n_jobs(run.n_jobs)
    tasks, total = stream_tasks(plan, cfg.n_perm, run.block_elems or BLOCK_ELEMS)
    rowsums = [[[np.empty(cfg.n_perm) for _ in kept] for kept in plan] for _ in rngs]

    def task(j: int, t: Task, u: np.ndarray) -> None:
        g, k, lo, hi, _ = t
        _, sizes, offsets = plan[g][k]
        rowsums[j][g][k][lo:hi] = _null_rows(ys[j][g][k], u, offsets, sizes)

    def shape(t: Task) -> tuple[int, int]:
        return _task_shape(plan, t)

    if n_jobs <= 1:
        for j, rng in enumerate(rngs):
            for t in tasks:
                task(j, t, rng.random(shape(t)))
        return rowsums
    _parallel_rowsums(rngs, (tasks, total), n_jobs, task, shape)
    return rowsums


def _rowsums(
    ys: list[list[list[np.ndarray]]],
    plan: Plan,
    cfg: MSCRConfig,
    rngs: list[np.random.Generator],
    run: Execution,
) -> RowSums:
    dev = resolve_device(run.device)
    if dev == "cpu":
        return rowsums_cpu(ys, plan, cfg, rngs, run)
    from oran_causal.methods.controlled.mscr_torch import (
        device_selfcheck,
        rowsums_torch,
    )

    if not device_selfcheck(dev):
        raise RuntimeError(
            f"MSCR bank on {dev!r} is not bit-identical to the numpy bank; use device='cpu'"
        )
    return rowsums_torch(ys, plan, cfg, rngs, dataclasses.replace(run, device=dev))


def _bank(
    ys_j: list[list[np.ndarray]], plan: Plan, rowsums_j: list[list[np.ndarray]], cfg: MSCRConfig
) -> Bank:
    denom = np.zeros(len(plan))
    null = np.zeros((len(plan), cfg.n_perm))
    strata = []
    for g, kept in enumerate(plan):
        bss_null = np.zeros(cfg.n_perm)
        tss = 0.0
        info = []
        for k, (rows, _, _) in enumerate(kept):
            y_s = ys_j[g][k]
            ns = len(y_s)
            t_s = y_s.sum()
            tss += float(np.dot(y_s, y_s)) - t_s * t_s / ns
            bss_null += rowsums_j[g][k] - t_s * t_s / ns
            info.append((rows, y_s))
        denom[g] = tss
        if tss > 0:
            null[g] = bss_null / tss
        strata.append(info)
    return denom, null, strata


def build_banks(
    x: np.ndarray,
    y: np.ndarray,
    cfg: MSCRConfig,
    rngs: list[np.random.Generator],
    run: Execution | None = None,
) -> list[Bank]:
    """(denom, null, strata) for every target column of `y`; target j uses `rngs[j]`.

    Bit-identical to v2's per-target bank for any execution settings.
    """
    run = run or Execution()
    resolve_device(run.device)
    plan = strata_plan(x, cfg)
    ys = [[[y[rows, j] for rows, _, _ in kept] for kept in plan] for j in range(y.shape[1])]
    rowsums = _rowsums(ys, plan, cfg, rngs, run)
    return [_bank(ys[j], plan, rowsums[j], cfg) for j in range(y.shape[1])]


def s_star(col: np.ndarray, i: int, denom: np.ndarray, strata: list, cfg: MSCRConfig) -> float:
    """Observed S* of candidate column `i` (values `col`) over every conditioner g != i."""
    best = -np.inf
    for g, info in enumerate(strata):
        if g == i or denom[g] <= 0:
            continue
        bss = sum(
            _between_ss(y_s, equal_count_labels(col[rows], cfg.nb), cfg.nb) for rows, y_s in info
        )
        best = max(best, bss / denom[g])
    return best


def null_s_star(null: np.ndarray, i: int) -> np.ndarray:
    """S*_null of candidate `i`: the max over conditioners g != i of the bank draws."""
    keep = np.ones(null.shape[0], dtype=bool)
    keep[i] = False
    return np.max(null[keep], axis=0)


def _check_inputs(x: np.ndarray, y: np.ndarray, n_params: int) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row mismatch: x {x.shape} vs y {y.shape}")
    if not 1 <= n_params <= x.shape[1] or x.shape[1] < 2:
        raise ValueError(
            f"need 1 <= n_params <= n_cols and n_cols >= 2, got {n_params}, {x.shape[1]}"
        )


def discover_mscr(
    x: np.ndarray,
    y: np.ndarray,
    n_params: int,
    seed: int,
    config: MSCRConfig | None = None,
    *,
    run: Execution | None = None,
) -> MSCRResult:
    """MSCR-v2 parameter-to-KPI discovery.

    Args:
        x: `[n, n_cols]` matrix whose first `n_params` columns are the tested parameters;
            further columns are conditioners only.
        y: `[n, n_targets]` target KPIs.
        n_params: Number of tested columns.
        seed: Dataset seed; target j's bank uses `default_rng([seed, j, n_perm])`.
        config: Constants; default the frozen v2 values.
        run: Execution settings (threads, device); never the result.

    Raises:
        ValueError: Inconsistent shapes or `n_params`.
    """
    cfg = config or frozen_config()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    _check_inputs(x, y, n_params)
    n_t = y.shape[1]
    run = run or Execution()
    jobs = resolve_n_jobs(run.n_jobs)
    rngs = [np.random.default_rng([int(seed), j, cfg.n_perm]) for j in range(n_t)]
    banks = build_banks(
        x, y, cfg, rngs, Execution(jobs, resolve_device(run.device), run.block_elems)
    )

    def observed(ji: tuple[int, int]) -> float:
        j, i = ji
        denom, _, strata = banks[j]
        return s_star(x[:, i], i, denom, strata, cfg)

    pairs = [(j, i) for j in range(n_t) for i in range(n_params)]
    if jobs > 1:
        with ThreadPoolExecutor(jobs) as pool:
            obs = list(pool.map(observed, pairs))
    else:
        obs = [observed(ji) for ji in pairs]
    s = np.array(obs, dtype=np.float64).reshape(n_t, n_params)
    pvals = np.zeros((n_t, n_params))
    declared = np.zeros((n_t, n_params), dtype=bool)
    for j in range(n_t):
        for i in range(n_params):
            pvals[j, i] = sequential_p_from_draws(s[j, i], null_s_star(banks[j][1], i), None)[0]
        declared[j] = weighted_step_up(pvals[j], None, cfg.q, reshape=True)
    return MSCRResult(pvals=pvals, s_star=s, declared=declared, config=cfg)

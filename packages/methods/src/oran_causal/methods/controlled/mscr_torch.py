"""MSCR-v2 permutation bank on torch (CUDA or any torch device), bit-identical to the numpy bank.

The uniforms are the same numpy streams (`mscr_bank.stream_tasks`). `torch.sort` of distinct
keys equals `np.argsort`; a row with a tie is re-sorted with `np.argsort`. `np.add.reduceat`
of a contiguous segment is `a[lo] + pairwise_sum(a[lo+1:hi])` with numpy's 8-accumulator
pairwise summation (block 128); `_np_pairwise` replays that association with elementwise
float64 adds. The per-draw tail runs in numpy. Verified with numpy 2.4.2 and torch 2.10 on
torch:cpu and one CUDA device; any other combination must pass `device_selfcheck`.
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from oran_causal.methods.controlled.mscr_bank import (
    Execution,
    MSCRConfig,
    Plan,
    RowSums,
    Task,
    finish_stream,
    jumpable,
    resolve_n_jobs,
    rowsums_cpu,
    strata_plan,
    stream_tasks,
    uniforms_at,
)

BYTES_PER_ELEM = 56
MAX_BLOCK_ELEMS = 1 << 23
_SELFCHECK: dict[str, bool] = {}
CHECK_CPU = Execution(1, "cpu", 1 << 12)


def _np_pairwise(t: torch.Tensor) -> torch.Tensor:
    n = t.shape[-1]
    if n < 8:
        res = torch.full(t.shape[:-1], -0.0, dtype=t.dtype, device=t.device)
        for i in range(n):
            res = res + t[..., i]
        return res
    if n <= 128:
        r = t[..., 0:8]
        for i in range(8, n - n % 8, 8):
            r = r + t[..., i : i + 8]
        res = ((r[..., 0] + r[..., 1]) + (r[..., 2] + r[..., 3])) + (
            (r[..., 4] + r[..., 5]) + (r[..., 6] + r[..., 7])
        )
        for i in range(n - n % 8, n):
            res = res + t[..., i]
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _np_pairwise(t[..., :n2]) + _np_pairwise(t[..., n2:])


def _reduceat_last(yp: torch.Tensor, offsets: np.ndarray) -> torch.Tensor:
    ns = yp.shape[-1]
    bounds = [int(o) for o in offsets] + [ns]
    out = torch.empty((*yp.shape[:-1], len(offsets)), dtype=yp.dtype, device=yp.device)
    by_len: dict[int, list[int]] = {}
    for k in range(len(offsets)):
        by_len.setdefault(bounds[k + 1] - bounds[k], []).append(k)
    for length, ks in by_len.items():
        idx = torch.as_tensor(
            np.array([np.arange(bounds[k], bounds[k] + length) for k in ks]), device=yp.device
        )
        seg = yp[..., idx]
        if length == 1:
            out[..., ks] = seg[..., 0]
        else:
            out[..., ks] = seg[..., 0] + _np_pairwise(seg[..., 1:])
    return out


def _auto_block_elems(device: torch.device, n_targets: int) -> int:
    budget = MAX_BLOCK_ELEMS * BYTES_PER_ELEM
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(budget, free // 4)
    return int(max(budget // (BYTES_PER_ELEM * max(1, n_targets)), 1 << 12))


def _groupsums(
    u: np.ndarray, y_dev: torch.Tensor, offsets: np.ndarray, dev: torch.device
) -> np.ndarray:
    ut = torch.from_numpy(u).to(dev)
    vals, perm = torch.sort(ut, dim=-1)
    tie = (vals[..., 1:] == vals[..., :-1]).any(dim=-1)
    if bool(tie.any()):
        for j, r in tie.nonzero().cpu().tolist():
            perm[j, r] = torch.from_numpy(np.argsort(u[j, r])).to(dev)
    del vals, ut
    yp = torch.gather(y_dev[:, None, :].expand(-1, u.shape[1], -1), 2, perm)
    del perm
    return _reduceat_last(yp, offsets).cpu().numpy()


def _groupsums_split(
    u: np.ndarray, y_dev: torch.Tensor, offsets: np.ndarray, dev: torch.device
) -> np.ndarray:
    try:
        return _groupsums(u, y_dev, offsets, dev)
    except torch.OutOfMemoryError:
        if u.shape[1] <= 1:
            raise
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        h = u.shape[1] // 2
        return np.concatenate(
            [
                _groupsums_split(u[:, :h], y_dev, offsets, dev),
                _groupsums_split(u[:, h:], y_dev, offsets, dev),
            ],
            axis=1,
        )


class _HostUniforms:
    """Host uniforms `[n_t, rows, ns]` per task, prefetched two tasks ahead with jump-ahead."""

    def __init__(self, rngs: list[np.random.Generator], plan: Plan, tasks: list[Task], n_jobs: int):
        self.rngs, self.plan, self.tasks = rngs, plan, tasks
        self.jump = n_jobs > 1 and all(jumpable(r) for r in rngs)
        self.states = (
            [(r.bit_generator.state, type(r.bit_generator)) for r in rngs] if self.jump else []
        )
        self.pool = ThreadPoolExecutor(n_jobs) if self.jump else None
        self.feeder = ThreadPoolExecutor(1) if self.jump else None

    def _shape(self, t: Task) -> tuple[int, int]:
        return (t[3] - t[2], len(self.plan[t[0]][t[1]][0]))

    def _draw(self, t: Task) -> np.ndarray:
        if self.pool is not None:
            draws = self.pool.map(
                lambda sb: uniforms_at(sb[0], sb[1], t[4], self._shape(t)), self.states
            )
            return np.stack(list(draws))
        return np.stack([r.random(self._shape(t)) for r in self.rngs])

    def __iter__(self) -> Any:
        if self.feeder is None:
            for t in self.tasks:
                yield t, self._draw(t)
            return
        ahead: deque[tuple[Task, Future[np.ndarray]]] = deque()
        it = iter(self.tasks)
        for _ in range(len(self.tasks)):
            while len(ahead) < 2:
                nxt = next(it, None)
                if nxt is None:
                    break
                ahead.append((nxt, self.feeder.submit(self._draw, nxt)))
            t, fut = ahead.popleft()
            yield t, fut.result()

    def close(self, total: int) -> None:
        """Shuts the pools and leaves every generator where sequential draws would."""
        for p in (self.feeder, self.pool):
            if p is not None:
                p.shutdown()
        for r, (state, _) in zip(self.rngs, self.states, strict=False):
            finish_stream(r, state, total)


def rowsums_torch(
    ys: list[list[list[np.ndarray]]],
    plan: Plan,
    cfg: MSCRConfig,
    rngs: list[np.random.Generator],
    run: Execution,
) -> RowSums:
    """Same contract as `mscr_bank.rowsums_cpu`, computed on the torch device `run.device`."""
    dev = torch.device((run.device or "cpu").removeprefix("torch:"))
    n_t = len(rngs)
    block = run.block_elems or _auto_block_elems(dev, n_t)
    tasks, total = stream_tasks(plan, cfg.n_perm, block)
    rowsums = [[[np.empty(cfg.n_perm) for _ in kept] for kept in plan] for _ in rngs]
    host = _HostUniforms(rngs, plan, tasks, resolve_n_jobs(run.n_jobs))
    cached_gk: tuple[int, int] | None = None
    y_dev = torch.empty(0)
    try:
        for t, u in host:
            g, k, lo, hi, _ = t
            _, sizes, offsets = plan[g][k]
            if cached_gk != (g, k):
                y_dev = torch.from_numpy(np.stack([ys[j][g][k] for j in range(n_t)])).to(dev)
                cached_gk = (g, k)
            gs = _groupsums_split(u, y_dev, offsets, dev)
            for j in range(n_t):
                rowsums[j][g][k][lo:hi] = np.sum(gs[j] ** 2 / sizes[None, :], axis=1)
    finally:
        host.close(total)
    return rowsums


def _selfcheck_case(n: int, cfg: MSCRConfig, rng: np.random.Generator, device: str) -> bool:
    x = rng.uniform(size=(n, 3))
    y = rng.normal(size=(n, 2))
    plan = strata_plan(x, cfg)
    ys = [[[y[rows, j] for rows, _, _ in kept] for kept in plan] for j in range(2)]
    a = rowsums_cpu(ys, plan, cfg, [np.random.default_rng([7, j]) for j in range(2)], CHECK_CPU)
    run = Execution(1, device, 1 << 12)
    b = rowsums_torch(ys, plan, cfg, [np.random.default_rng([7, j]) for j in range(2)], run)
    return all(
        np.array_equal(a[j][g][k], b[j][g][k])
        for j in range(2)
        for g in range(len(plan))
        for k in range(len(plan[g]))
    )


def device_selfcheck(device: str) -> bool:
    """Bit-identity of this kernel against the numpy kernel on `device` (small, cached per process).

    The cases exercise the 8-accumulator branch, the recursive branch (> 128) and the
    singleton and sub-8 segments, with two targets batched.
    """
    if device in _SELFCHECK:
        return _SELFCHECK[device]
    cases = (
        (520, MSCRConfig(nc=6, nb=8, min_stratum=40, n_perm=40)),
        (700, MSCRConfig(nc=2, nb=2, min_stratum=10, n_perm=24)),
        (60, MSCRConfig(nc=2, nb=16, min_stratum=10, n_perm=24)),
    )
    rng = np.random.default_rng(12345)
    try:
        ok = all([_selfcheck_case(n, cfg, rng, device) for n, cfg in cases])  # noqa: C419
    except Exception:
        ok = False
    _SELFCHECK[device] = ok
    return ok

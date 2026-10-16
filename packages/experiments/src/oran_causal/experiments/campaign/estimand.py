"""Ruling R-36 estimand check: no truth-null primary candidate has a nonzero lag-1 total effect.

Row r of a dataset is (A_t, K_t, Y = K_{t+1}) and `advance` computes K_{t+1} from A_t, K_t and
the exogenous tape before the next action, so do(A_t = a) moves K_{t+1} only through one
`advance`. The tape is coordinate-keyed, so a deep copy advanced with a different A_t sees the
same exogenous draws and an exactly zero effect shows as Delta == 0.0 at every unit.
Check A (structural): random states over the identification ranges. Check B (interventional):
each regime's own trajectories on DEV seeds 3_000_190-3_000_199. A truth-null candidate passes
iff max |Delta| == 0.0 in both.
"""

import contextlib
import copy
import time
from collections.abc import Callable, Generator
from typing import Any, cast

import numpy as np

from oran_causal.sim.controlled.base import ControlledEnv
from oran_causal.sim.controlled.datasets import e4_logged
from oran_causal.sim.controlled.datasets.generate import WARMUP, dataset_hash, generate_dataset
from oran_causal.sim.controlled.datasets.truth import (
    E4_LAMBDAS,
    ENV_CLASSES,
    REGIMES_OF,
    candidates_for,
    truth_for,
)
from oran_causal.sim.controlled.e4 import E4Env
from oran_causal.sim.controlled.e5 import E5Env

DEV_SEEDS = tuple(range(3_000_190, 3_000_200))
M_VALUES = 2
S_STATES = 3000
KAPPA = 0.25
RNG_TAG = 9_300
PATCHED = (ControlledEnv, E4Env, E5Env)

type Advance = Callable[[ControlledEnv], np.ndarray]


def env_for(world: str, regime: str, lam: float) -> ControlledEnv:
    """The SCM instance the generator drives for (world, regime)."""
    if world == "E4":
        if regime == "R3":
            return e4_logged.E4LoggedEnv(env_seed=0, lam=lam, episode=0)
        return E4Env(env_seed=0, lam=lam, mode="obs" if regime == "R4" else "do", episode=0)
    return ENV_CLASSES[world](env_seed=0, episode=0)


def values(rng: np.random.Generator, world: str, j: int, size: int) -> np.ndarray:
    """Intervention values of action j: the E4 do-grid, else uniform on the identification range."""
    if world == "E4":
        grid = e4_logged.GRID
        return grid[rng.integers(0, len(grid), size)]
    lo, hi = ENV_CLASSES[world].id_ranges[j]
    return rng.uniform(lo, hi, size)


class Recorder:
    """|Delta| statistics per (source, target); sources are the actions then the lagged KPIs."""

    def __init__(self, world: str) -> None:
        cls = ENV_CLASSES[world]
        self.p, self.k = cls.num_params, cls.num_kpis
        self.maxabs = np.zeros((self.p + self.k, self.k))
        self.nonzero = np.zeros((self.p + self.k, self.k))
        self.slope_sum = np.zeros((self.p + self.k, self.k))
        self.units = np.zeros(self.p + self.k)

    def add(self, src: int, delta: np.ndarray, dv: float) -> None:
        """Adds one unit-level contrast of source `src` with intervention size `dv`."""
        self.maxabs[src] = np.maximum(self.maxabs[src], np.abs(delta))
        self.nonzero[src] += delta != 0.0
        if dv != 0.0:
            self.slope_sum[src] += delta / dv
        self.units[src] += 1


def contrast(
    env: Any, rec: Recorder, world: str, rng: np.random.Generator, advance: Advance
) -> None:
    """Delta of K_{t+1} at the current state for interventions on each action and lagged KPI."""
    base = advance(copy.deepcopy(env))
    for j in range(rec.p):
        for a in values(rng, world, j, M_VALUES):
            c = copy.deepcopy(env)
            dv = float(a - c.prev_params[j])
            c.prev_params[j] = a
            rec.add(j, advance(c) - base, dv)
    for i in range(rec.k):
        for _ in range(M_VALUES):
            c = copy.deepcopy(env)
            dv = float(rng.normal(0.0, max(abs(float(c.prev_kpis[i])), 1.0)))
            c.prev_kpis[i] = c.prev_kpis[i] + dv
            rec.add(rec.p + i, advance(c) - base, dv)


@contextlib.contextmanager
def wrapped_advance(
    rec: Recorder, world: str, rng: np.random.Generator, skip: int
) -> Generator[dict[str, int]]:
    """Wraps every SCM class's `advance`: each call after `skip` warm-up calls records a contrast.

    The wrapped call itself is unchanged, so the dataset stays byte-identical.
    """
    originals = {c: c.__dict__["advance"] for c in PATCHED}
    state = {"calls": 0}

    def make(orig: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def adv(self: Any) -> Any:
            if state["calls"] >= skip:
                contrast(self, rec, world, rng, lambda e: np.asarray(orig(e), float).copy())
            state["calls"] += 1
            return orig(self)

        return adv

    try:
        for c, o in originals.items():
            cast(Any, c).advance = make(o)
        yield state
    finally:
        for c, o in originals.items():
            cast(Any, c).advance = o


def _rng(check: int, world: str, regime: str, lam: float, *extra: int) -> np.random.Generator:
    return np.random.default_rng(
        [RNG_TAG, check, int(world[1]), int(regime[1]), int(lam * 10), *extra]
    )


def _random_state(env: Any, world: str, rng: np.random.Generator, kpool: np.ndarray) -> None:
    n_actions = len(ENV_CLASSES[world].id_ranges)
    env.time = int(rng.integers(5, 10_000))
    env.prev_params = np.array([values(rng, world, j, 1)[0] for j in range(n_actions)], float)
    env.params = np.array([values(rng, world, j, 1)[0] for j in range(n_actions)], float)
    env.prev_kpis = kpool[rng.integers(0, len(kpool))] * rng.uniform(0.5, 1.5, kpool.shape[1])
    if hasattr(env, "prev_Z"):
        env.prev_Z, env.Z = float(rng.normal()), float(rng.normal())
    if world == "E2":
        env._decoy_p0_ref = float(env.prev_params[0])


def structural(world: str, regime: str, lam: float) -> Recorder:
    """Check A: random states over the whole identification range."""
    rng = _rng(1, world, regime, lam)
    ds, _ = generate_dataset(world, regime, 500, DEV_SEEDS[0], lam=lam, kappa=0.0)
    rec = Recorder(world)
    env = env_for(world, regime, lam)
    adv = type(env).advance
    for _ in range(S_STATES):
        _random_state(env, world, rng, ds.X_kpi_lag)
        contrast(env, rec, world, rng, lambda e, adv=adv: np.asarray(adv(e), float).copy())
    return rec


def interventional(world: str, regime: str, lam: float, n: int) -> Recorder:
    """Check B: the regime's own trajectories on DEV seeds 3_000_190-3_000_199."""
    rec = Recorder(world)
    for seed in DEV_SEEDS:
        rng = _rng(2, world, regime, lam, seed)
        ref, _ = generate_dataset(world, regime, n, seed, lam=lam, kappa=KAPPA)
        with wrapped_advance(rec, world, rng, skip=WARMUP) as st:
            ds, _ = generate_dataset(world, regime, n, seed, lam=lam, kappa=KAPPA)
        if st["calls"] != WARMUP + n or dataset_hash(ds) != dataset_hash(ref):
            raise AssertionError(f"{world} {regime}: wrapper changed the data or the row count")
    return rec


def _edge_stats(rec_a: Recorder, rec_b: Recorder, s: int, k: int) -> dict[str, float]:
    return {
        "maxabs_A": float(rec_a.maxabs[s, k]),
        "maxabs_B": float(rec_b.maxabs[s, k]),
        "share_nonzero_B": float(rec_b.nonzero[s, k] / max(rec_b.units[s], 1)),
        "mean_slope_B": float(rec_b.slope_sum[s, k] / max(rec_b.units[s], 1)),
    }


def _moves(v: dict[str, float]) -> bool:
    return v["maxabs_A"] != 0 or v["maxabs_B"] != 0


def summarize(
    world: str, regime: str, lam: float | None, rec_a: Recorder, rec_b: Recorder
) -> dict[str, Any]:
    """Per-edge statistics and the pass verdicts of one cell."""
    truth = truth_for(world, regime)
    names = [f"P{j}" for j in range(rec_a.p)] + [f"K{i}" for i in range(rec_a.k)]
    out: dict[str, Any] = {
        "world": world,
        "regime": regime,
        "lam": lam,
        "units_B": int(rec_b.units[0]),
    }
    out |= {"primary_null": {}, "primary_true": {}, "secondary_null": {}, "secondary_true": {}}
    for s, nm in enumerate(names):
        for k in range(rec_a.k):
            e = (nm, f"K{k}")
            fam = ("primary" if s < rec_a.p else "secondary") + (
                "_true" if e in truth.edges else "_null"
            )
            out[fam][f"{e[0]}->{e[1]}"] = _edge_stats(rec_a, rec_b, s, k)
    kpis = {f"K{i}" for i in range(rec_a.k)}
    out["placebo_by_construction"] = sorted(
        {c[0] for c in candidates_for(world, regime)} - set(names) - kpis
    )
    bad = [e for e, v in out["primary_null"].items() if _moves(v)]
    bad2 = [e for e, v in out["secondary_null"].items() if _moves(v)]
    weak = [e for e, v in out["primary_true"].items() if v["share_nonzero_B"] < 0.5]
    dead = [
        e
        for fam in ("primary_true", "secondary_true")
        for e, v in out[fam].items()
        if v["maxabs_A"] == 0 or v["maxabs_B"] == 0
    ]
    out.update(
        pass_primary=not bad,
        violations_primary=bad,
        pass_secondary=not bad2,
        violations_secondary=bad2,
        true_edges_effect_lt_half_units=weak,
        true_edges_zero_effect=dead,
    )
    return out


def cells() -> list[tuple[str, str, float]]:
    """Every (world, regime, lam) cell checked; E4 R3 / R4 sweep the lambdas."""
    return [
        (w, r, lam)
        for w, rs in REGIMES_OF.items()
        for r in rs
        for lam in (E4_LAMBDAS if (w == "E4" and r in ("R3", "R4")) else (1.0,))
    ]


def check_cell(world: str, regime: str, lam: float, n: int) -> dict[str, Any]:
    """Checks A and B of one cell, summarised, with its CPU seconds."""
    t0 = time.process_time()
    row = summarize(
        world,
        regime,
        lam if world == "E4" else None,
        structural(world, regime, lam),
        interventional(world, regime, lam, n),
    )
    row["cpu_s"] = round(time.process_time() - t0, 1)
    return row


def run(n: int = 1000, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The full estimand check (`n` rows per seed); the frozen output used 1000."""
    rows = []
    for w, r, lam in cells():
        row = check_cell(w, r, lam, n)
        rows.append(row)
        log(f"{w} {r} lam={lam} primary {'PASS' if row['pass_primary'] else 'FAIL'}")
    return {
        "n_rows_per_seed": n,
        "seeds": list(DEV_SEEDS),
        "m_values": M_VALUES,
        "s_states": S_STATES,
        "kappa_B": KAPPA,
        "all_pass_primary": all(x["pass_primary"] for x in rows),
        "cells": rows,
    }

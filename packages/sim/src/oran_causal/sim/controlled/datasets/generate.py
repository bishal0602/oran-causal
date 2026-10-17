"""Dataset generation for the controlled environments.

`generate_dataset(world, regime, n, seed)` returns a `Dataset` of one-step rows and its
`Truth`. One trajectory is rolled per `(world, seed)` with `env_seed = seed` and
`episode = 0`. After `WARMUP` priming advances, row `r` records the committed state at env
time `t` and the KPIs one advance later::

    X_action[r]  = committed parameters at t
    X_kpi_lag[r] = committed KPIs at t
    Y[r]         = KPIs at t+1

so `Y[r] == X_kpi_lag[r+1]`.

Regimes (`Design.kind`):

* R1 `iid`: every action column is redrawn i.i.d. each row (uniform on its range; E4: uniform
  on the do-grid).
* R2 `dither`: per column, 20 setpoints drawn once per corpus within +/-25% of the range
  around the midpoint, each held for `n/20` rows (`fixed_part`), plus i.i.d. dither of
  +/-10% of the range (`random_part`).
* R3 `logged` (E4 only): the discretised behaviour policy on the observed confounder, with
  the propensity table of each row in `Design.propensity`.
* R4 `none` (E4 only): the observational mode with a latent confounder and nothing logged.

Every dataset carries the tuning placebo column `P_placebo`, never applied to the model and
drawn from its own stream. E4 R3 and R4 add the diagnostic column `P_placebo_conf`, drawn by
the confounded policy; its edges are nulls outside the primary candidate set.

Observation noise adds `eps ~ N(0, (kappa * sigma_k)^2)` i.i.d. per env time and KPI to the
observed KPI series, so a KPI value has the same noise as a lag and as a target. `kappa=None`
selects the world default `KAPPA_DEFAULT`.
"""

import dataclasses
import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from oran_causal.core.data import Dataset, Design, Truth
from oran_causal.sim.controlled.base import ControlledEnv
from oran_causal.sim.controlled.datasets import e4_logged
from oran_causal.sim.controlled.datasets.truth import (
    E4_LAMBDAS,
    ENV_CLASSES,
    PLACEBO,
    PLACEBO_CONF,
    REGIMES_OF,
    candidates_for,
    truth_for,
)
from oran_causal.sim.controlled.e4 import E4Env
from oran_causal.sim.controlled.e5 import E5Env

GENERATOR_VERSION = "xm-worlds/3"
WARMUP = 4
STREAM_TAG = 7800
DITHER_DELTA = 0.10
DITHER_BLOCKS = 20
DITHER_ENVELOPE = 0.25
KAPPA_E5 = 0.3
E5_SIGMA = (4.634914, 24.662764, 57.680719, 20.192141)
NOISE_SIGMA: dict[str, tuple[float, ...]] = {
    "E1": (0.288709, 0.288894, 0.322411, 0.322784),
    "E2": (27.672172, 34.66717, 44.7979, 32.491503, 40.552988, 4.791992),
    "E3": (0.288581, 0.288582, 0.407729, 0.407559, 0.288733),
    "E4": (2.515616,),
    "E5": E5_SIGMA,
}
KAPPA_DEFAULT = {"E1": 0.0, "E2": 0.0, "E3": 0.0, "E4": 0.0, "E5": KAPPA_E5}

_WORLD_CODE = {"E1": 1, "E2": 2, "E3": 3, "E4": 4, "E5": 5}
_REGIME_CODE = {"R1": 1, "R2": 2, "R3": 3, "R4": 4}
_PURPOSE = {
    "actions": 1,
    "setpoints": 2,
    "placebo": 3,
    "placebo_setpoints": 4,
    "obs_noise": 5,
    "placebo_conf": 6,
}

Ranges = tuple[tuple[float, float], ...]
Inject = Callable[[int], None]


@dataclasses.dataclass(frozen=True)
class _Request:
    world: str
    regime: str
    n: int
    seed: int
    lam: float


@dataclasses.dataclass
class _Corpus:
    xa: np.ndarray
    xk: np.ndarray
    y: np.ndarray
    time_index: np.ndarray
    designs: list[Design]
    placebo: np.ndarray
    placebo_design: Design
    conf: np.ndarray | None = None
    conf_design: Design | None = None
    context: np.ndarray | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


def _rng(seed: int, world: str, regime: str, purpose: str) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [STREAM_TAG, int(seed), _WORLD_CODE[world], _REGIME_CODE[regime], _PURPOSE[purpose]]
        )
    )


def _ranges(world: str) -> Ranges:
    return tuple(ENV_CLASSES[world].id_ranges)


def _grid(world: str) -> np.ndarray | None:
    return e4_logged.GRID if world == "E4" else None


def _iid_column_dist(lo: float, hi: float, grid: np.ndarray | None) -> dict[str, Any]:
    if grid is not None:
        return {"name": "categorical", "values": grid.tolist(), "p": [1.0 / len(grid)] * len(grid)}
    return {"name": "uniform", "lo": float(lo), "hi": float(hi)}


def _draw_iid(
    rng: np.random.Generator, steps: int, ranges: Ranges, grid: np.ndarray | None
) -> np.ndarray:
    if grid is not None:
        return grid[rng.integers(0, len(grid), size=(steps, len(ranges)))]
    lo, hi = np.array(ranges, dtype=float).T
    return rng.uniform(lo, hi, size=(steps, len(ranges)))


def _draw_dither(
    rng_setpoints: np.random.Generator,
    rng_dither: np.random.Generator,
    steps: int,
    n: int,
    ranges: Ranges,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.array(ranges, dtype=float).T
    mid, width = (lo + hi) / 2.0, hi - lo
    setpoints = rng_setpoints.uniform(
        mid - DITHER_ENVELOPE * width,
        mid + DITHER_ENVELOPE * width,
        size=(DITHER_BLOCKS, len(ranges)),
    )
    row = np.arange(steps) - (WARMUP - 1)
    block = np.clip(np.clip(row, 0, None) * DITHER_BLOCKS // n, 0, DITHER_BLOCKS - 1)
    fixed = setpoints[block]
    rand = rng_dither.uniform(
        -DITHER_DELTA * width, DITHER_DELTA * width, size=(steps, len(ranges))
    )
    value = fixed + rand
    if np.any(value < lo) or np.any(value > hi):
        raise AssertionError("F-dither left the ID range")
    return fixed, rand, block


def _committed_latent(env: ControlledEnv) -> float:
    if not isinstance(env, E4Env | E5Env):
        raise TypeError(f"{type(env).__name__} has no latent confounder")
    return env.prev_Z


def _roll(env: ControlledEnv, inject: Inject, n: int, latent_z: bool = False):
    for i in range(WARMUP):
        inject(i)
        env.advance()
    xa, xk, y, t, z = [], [], [], [], []
    for r in range(n):
        xa.append(env.prev_params.copy())
        xk.append(env.prev_kpis.copy())
        t.append(env.time)
        if latent_z:
            z.append(_committed_latent(env))
        inject(WARMUP + r)
        y.append(env.advance().copy())
    xa_a, xk_a, y_a = (np.array(v, dtype=float) for v in (xa, xk, y))
    if not np.array_equal(y_a[:-1], xk_a[1:]):
        raise AssertionError("row alignment broken: Y[r] != X_kpi_lag[r+1]")
    return (
        xa_a,
        xk_a,
        y_a,
        np.array(t, dtype=np.int64),
        (np.array(z, dtype=float) if latent_z else None),
    )


def _interventional_env(world: str, seed: int, lam: float) -> ControlledEnv:
    if world == "E4":
        env = E4Env(env_seed=seed, lam=float(lam), mode="do", episode=0)
    else:
        env = ENV_CLASSES[world](env_seed=seed, episode=0)
    env.reset(episode=0)
    return env


def _iid_designs(world: str, ranges: Ranges) -> list[Design]:
    grid = _grid(world)
    return [Design(kind="iid", dist=_iid_column_dist(lo, hi, grid)) for lo, hi in ranges]


def _dither_design(fixed: np.ndarray, rand: np.ndarray, lo: float, hi: float) -> Design:
    w = DITHER_DELTA * (hi - lo)
    return Design(
        kind="dither",
        dist={"name": "uniform", "lo": -w, "hi": w},
        random_part=rand,
        fixed_part=fixed,
    )


def _generate_interventional(req: _Request) -> _Corpus:
    world, regime, n, seed = req.world, req.regime, req.n, req.seed
    ranges = _ranges(world)
    p0_range = (ranges[0],)
    steps = WARMUP + n
    live = slice(WARMUP - 1, WARMUP - 1 + n)
    env = _interventional_env(world, seed, req.lam)
    meta: dict[str, Any] = {}
    if regime == "R1":
        grid = _grid(world)
        actions = _draw_iid(_rng(seed, world, regime, "actions"), steps, ranges, grid)
        placebo_all = _draw_iid(_rng(seed, world, regime, "placebo"), steps, p0_range, grid)
        designs = _iid_designs(world, ranges)
        placebo_design = Design(kind="iid", dist=_iid_column_dist(*ranges[0], grid))
    else:
        fixed, rand, block = _draw_dither(
            _rng(seed, world, regime, "setpoints"),
            _rng(seed, world, regime, "actions"),
            steps,
            n,
            ranges,
        )
        p_fixed, p_rand, _ = _draw_dither(
            _rng(seed, world, regime, "placebo_setpoints"),
            _rng(seed, world, regime, "placebo"),
            steps,
            n,
            p0_range,
        )
        actions, placebo_all = fixed + rand, p_fixed + p_rand
        designs = [
            _dither_design(fixed[live, j].copy(), rand[live, j].copy(), lo, hi)
            for j, (lo, hi) in enumerate(ranges)
        ]
        placebo_design = _dither_design(p_fixed[live, 0].copy(), p_rand[live, 0].copy(), *ranges[0])
        meta.update(
            block=block[live].copy(),
            dither={"delta": DITHER_DELTA, "n_blocks": DITHER_BLOCKS, "envelope": DITHER_ENVELOPE},
        )

    def inject(i: int) -> None:
        for j in range(len(ranges)):
            env.apply_action(j, float(actions[i, j]))

    xa, xk, y, t, _ = _roll(env, inject, n, latent_z=world in ("E4", "E5"))
    if not np.array_equal(xa, actions[live]):
        raise AssertionError("committed params differ from the design draws")
    return _Corpus(xa, xk, y, t, designs, placebo_all[live, 0].copy(), placebo_design, meta=meta)


def _generate_logged(req: _Request) -> _Corpus:
    world, regime, n, seed = req.world, req.regime, req.n, req.seed
    steps = WARMUP + n
    live = slice(WARMUP - 1, WARMUP - 1 + n)
    lam = float(req.lam)
    if regime == "R3":
        env: E4Env = e4_logged.E4LoggedEnv(env_seed=seed, lam=lam, episode=0)
    else:
        env = E4Env(env_seed=seed, lam=lam, mode="obs", episode=0)
    env.reset(episode=0)
    marginal_sd = float(np.sqrt(lam**2 + e4_logged.ETA_SD**2))
    placebo_draw = _rng(seed, world, regime, "placebo").normal(0.5, marginal_sd, size=steps)
    eta_conf = _rng(seed, world, regime, "placebo_conf").normal(0.0, e4_logged.ETA_SD, size=steps)
    applied, conf_values, zs = np.empty(steps), np.empty(steps), np.empty(steps)

    def inject(i: int) -> None:
        zs[i] = env.Z
        applied[i] = env.behavior_action()
        w = 0.5 + lam * env.Z + eta_conf[i]
        conf_values[i] = (
            e4_logged.GRID[e4_logged.grid_index(w)] if regime == "R3" else float(np.clip(w, 0, 1))
        )
        env.apply_action(0, applied[i])

    xa, xk, y, t, z_committed = _roll(env, inject, n, latent_z=True)
    if z_committed is None or not (
        np.array_equal(xa[:, 0], applied[live]) and np.array_equal(z_committed, zs[live])
    ):
        raise AssertionError("committed action / Z misaligned")
    conf = conf_values[live].copy()
    if regime == "R3":
        placebo, placebo_design = _r3_placebo(placebo_draw[live], marginal_sd)
        designs, conf_design, context, meta = _r3_logged(xa, conf, z_committed, lam)
        return _Corpus(
            xa,
            xk,
            y,
            t,
            designs,
            placebo,
            placebo_design,
            conf=conf,
            conf_design=conf_design,
            context=context,
            meta=meta,
        )
    placebo = np.clip(placebo_draw[live], 0.0, 1.0)
    placebo_design = Design(
        kind="iid",
        dist={"name": "clipped_normal", "mean": 0.5, "sd": marginal_sd, "lo": 0.0, "hi": 1.0},
    )
    return _Corpus(
        xa,
        xk,
        y,
        t,
        [Design(kind="none")],
        placebo,
        placebo_design,
        conf=conf,
        conf_design=Design(kind="none"),
    )


def _r3_placebo(placebo_draw: np.ndarray, marginal_sd: float) -> tuple[np.ndarray, Design]:
    placebo = e4_logged.GRID[e4_logged.grid_index(placebo_draw)]
    marginal = e4_logged.policy_probs(np.zeros(1), 0.0, eta_sd=marginal_sd)[0]
    design = Design(
        kind="iid",
        dist={"name": "categorical", "values": e4_logged.GRID.tolist(), "p": marginal.tolist()},
    )
    return placebo, design


def _r3_logged(
    xa: np.ndarray, conf: np.ndarray, z: np.ndarray, lam: float
) -> tuple[list[Design], Design, np.ndarray, dict[str, Any]]:
    n = len(z)
    dist = {
        "name": "categorical_rows",
        "values": e4_logged.GRID.tolist(),
        "policy": "V[floor(100*clip(0.5 + lam*Z + eta, 0, 1) + .5)], eta ~ N(0, sd^2)",
        "lam": lam,
        "sd": e4_logged.ETA_SD,
    }
    table = e4_logged.policy_probs(z, lam)
    k_action, k_conf = e4_logged.grid_index(xa[:, 0]), e4_logged.grid_index(conf)
    meta = {
        "realised_propensity": {
            "P0": table[np.arange(n), k_action].copy(),
            PLACEBO_CONF: table[np.arange(n), k_conf].copy(),
        },
        "context_names": ("Z",),
    }
    designs = [Design(kind="logged", dist=dist, propensity=table)]
    conf_design = Design(kind="logged", dist=dist, propensity=table.copy())
    return designs, conf_design, z.reshape(-1, 1).copy(), meta


def _apply_obs_noise(corpus: _Corpus, req: _Request, kappa: float) -> dict[str, Any] | None:
    if kappa <= 0.0:
        return None
    series = np.concatenate([corpus.xk, corpus.y[-1:]], axis=0)
    sigma = NOISE_SIGMA[req.world]
    eps = _rng(req.seed, req.world, req.regime, "obs_noise").standard_normal(series.shape) * (
        kappa * np.array(sigma)
    )
    series = series + eps
    corpus.xk, corpus.y = series[:-1].copy(), series[1:].copy()
    return {"kappa": kappa, "sigma": list(sigma)}


def _resolve_kappa(world: str, kappa: float | None) -> float:
    resolved = KAPPA_DEFAULT[world] if kappa is None else float(kappa)
    if not (np.isfinite(resolved) and resolved >= 0.0):
        raise ValueError(f"kappa must be a finite number >= 0, got {kappa}")
    return resolved


def _validate(world: str, regime: str, n: int, lam: float) -> None:
    if world not in ENV_CLASSES:
        raise ValueError(f"unknown world {world!r}")
    if regime not in REGIMES_OF[world]:
        raise ValueError(
            f"regime {regime} is not defined for {world} (allowed {REGIMES_OF[world]})"
        )
    if n < 2:
        raise ValueError("n must be >= 2")
    if world == "E4" and float(lam) not in E4_LAMBDAS:
        raise ValueError(f"E4 lambda must be one of {E4_LAMBDAS}, got {lam}")


def _assemble(
    corpus: _Corpus, req: _Request, meta: dict[str, Any], *, with_placebo: bool
) -> Dataset:
    world, regime = req.world, req.regime
    env_cls = ENV_CLASSES[world]
    action_names = tuple(f"P{i}" for i in range(env_cls.num_params))
    kpi_names = tuple(f"K{j}" for j in range(env_cls.num_kpis))
    extra: list[tuple[str, np.ndarray, Design]] = [(PLACEBO, corpus.placebo, corpus.placebo_design)]
    if regime in ("R3", "R4") and corpus.conf is not None and corpus.conf_design is not None:
        extra.append((PLACEBO_CONF, corpus.conf, corpus.conf_design))
    designs = list(corpus.designs)
    if with_placebo:
        x_action = np.concatenate([corpus.xa] + [c.reshape(-1, 1) for _, c, _ in extra], axis=1)
        action_names += tuple(name for name, _, _ in extra)
        designs += [d for _, _, d in extra]
        candidates = candidates_for(world, regime)
    else:
        x_action = corpus.xa
        candidates = tuple(
            c for c in candidates_for(world, regime) if c[0] not in (PLACEBO, PLACEBO_CONF)
        )
    meta["primary_candidates"] = tuple(
        c for c in candidates if c[0] in action_names and c[0] != PLACEBO_CONF
    )
    meta["secondary_candidates"] = tuple(c for c in candidates if c[0] in kpi_names)
    meta["diagnostic_candidates"] = tuple(c for c in candidates if c[0] == PLACEBO_CONF)
    meta["placebo_conf"] = PLACEBO_CONF if (regime in ("R3", "R4") and with_placebo) else None
    return Dataset(
        world=world,
        regime=regime,
        n=req.n,
        seed=req.seed,
        action_names=action_names,
        kpi_names=kpi_names,
        X_action=x_action,
        X_kpi_lag=corpus.xk,
        Y=corpus.y,
        designs=tuple(designs),
        candidates=candidates,
        time_index=corpus.time_index,
        context=corpus.context,
        meta=meta,
    )


def generate_dataset(
    world: str,
    regime: str,
    n: int,
    seed: int,
    *,
    lam: float = 1.0,
    kappa: float | None = None,
    with_placebo: bool = True,
) -> tuple[Dataset, Truth]:
    """Generates one `(world, regime, n, seed)` corpus and its truth.

    A pure function of its arguments: regeneration is byte-identical.

    Args:
        world: Environment id, `"E1"` to `"E5"`.
        regime: Data regime id allowed for the world.
        n: Number of rows, at least 2.
        seed: Generator seed, also used as the environment seed.
        lam: E4 coupling of the confounder to the behaviour action, one of `E4_LAMBDAS`; it
            changes the data only under R3 and R4.
        kappa: Observation-noise level relative to the KPI spread; `None` selects the world
            default.
        with_placebo: Whether to append the placebo columns and their candidates.

    Returns:
        The dataset and the ground truth of `(world, regime)`.

    Raises:
        ValueError: On an unknown world, a regime not defined for the world, `n < 2`, an E4
            `lam` outside the frozen sweep, or an invalid `kappa`.
    """
    _validate(world, regime, n, lam)
    resolved_kappa = _resolve_kappa(world, kappa)
    req = _Request(world, regime, int(n), int(seed), lam)
    corpus = _generate_interventional(req) if regime in ("R1", "R2") else _generate_logged(req)
    meta: dict[str, Any] = {
        "generator": GENERATOR_VERSION,
        "warmup": WARMUP,
        "env_seed": req.seed,
        "episode": 0,
        "placebo": PLACEBO,
    }
    meta.update(corpus.meta)
    noise = _apply_obs_noise(corpus, req, resolved_kappa)
    if noise is not None:
        meta["obs_noise"] = noise
    if world == "E4":
        meta["lam"] = float(lam)
    dataset = _assemble(corpus, req, meta, with_placebo=with_placebo)
    return dataset, truth_for(world, regime)


def dataset_hash(ds: Dataset) -> str:
    """Returns the sha256 over every array and name of a dataset, designs and meta included."""
    h = hashlib.sha256()

    def add_array(a: np.ndarray | None) -> None:
        if a is None:
            h.update(b"<none>")
        else:
            a = np.ascontiguousarray(a)
            h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())

    def add_value(key: str, value: Any) -> None:
        h.update(key.encode())
        if isinstance(value, np.ndarray):
            add_array(value)
        elif isinstance(value, dict):
            for sub in sorted(value):
                add_value(f"{key}.{sub}", value[sub])
        else:
            h.update(repr(value).encode())

    h.update(
        repr(
            (ds.world, ds.regime, ds.n, ds.seed, ds.action_names, ds.kpi_names, ds.candidates)
        ).encode()
    )
    for a in (ds.X_action, ds.X_kpi_lag, ds.Y, ds.time_index, ds.context):
        add_array(a)
    for design in ds.designs:
        h.update(repr((design.kind, design.dist, design.redraw_seed_tag)).encode())
        for a in (design.random_part, design.fixed_part, design.propensity):
            add_array(a)
    for key in sorted(ds.meta):
        add_value(key, ds.meta[key])
    return h.hexdigest()

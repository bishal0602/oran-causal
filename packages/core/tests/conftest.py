import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from oran_causal.core.data import Dataset, Design

POLICY = np.array([0.5, 0.2, 0.3])
LEVELS = np.array([1.0, 1.0, 0.0])


def rows_dataset(
    n: int = 600,
    *,
    seed: int = 0,
    regime: str = "R1",
    beta: float = 0.0,
    lag_effect: float = 0.0,
    rho: float = 0.6,
    n_blocks: int = 10,
    delta: float = 0.1,
) -> Dataset:
    """Two actions and a placebo, two AR(1) KPIs; K0 depends on A0 with slope `beta`."""
    rng = np.random.default_rng([seed, 99])
    p = 3
    dz = np.zeros((n, p))
    if regime == "R1":
        a = rng.uniform(0.0, 1.0, (n, p))
        designs = tuple(Design("iid", {"name": "uniform", "lo": 0.0, "hi": 1.0}) for _ in range(p))
    else:
        sp = rng.uniform(0.25, 0.75, (n_blocks, p))[
            np.minimum(np.arange(n) * n_blocks // n, n_blocks - 1)
        ]
        dz = rng.uniform(-delta, delta, (n, p))
        a = sp + dz
        designs = tuple(
            Design(
                "dither",
                {"name": "uniform", "lo": -delta, "hi": delta},
                random_part=dz[:, j],
                fixed_part=sp[:, j],
            )
            for j in range(p)
        )
    k = np.zeros(2)
    lag, y = np.zeros((n, 2)), np.zeros((n, 2))
    a_prev = np.zeros(p)
    for t in range(n):
        lag[t] = k
        drive = a[t, 0] if regime == "R1" else (a[t, 0] - 0.5)
        k = np.array(
            [
                rho * k[0] + beta * drive + lag_effect * a_prev[0] + rng.normal(0, 0.1),
                rho * k[1] + (0.0 if regime == "R1" else a[t, 1] - dz[t, 1]) + rng.normal(0, 0.1),
            ]
        )
        y[t] = k
        a_prev = a[t]
    names = ("P0", "P1", "P_placebo")
    cands = (*((x, kk) for x in names for kk in ("K0", "K1")), ("K0", "K1"), ("K1", "K0"))
    return Dataset(
        "E1",
        regime,
        n,
        seed,
        names,
        ("K0", "K1"),
        a,
        lag,
        y,
        designs,
        cands,
        time_index=np.arange(n),
    )


def bump_dataset(n: int = 500, seed: int = 0, regime: str = "R1", amp: float = 0.15) -> Dataset:
    """A null rows dataset plus a bump of A0 centred in its range on K0."""
    ds = rows_dataset(n, seed=seed, regime=regime)
    y = ds.Y.copy()
    y[:, 0] += amp * np.exp(-((ds.X_action[:, 0] - 0.5) ** 2) / (2 * 0.12**2))
    return dataclasses.replace(ds, Y=y)


def logged_dataset(n: int = 300, seed: int = 0) -> Dataset:
    """Logged action with context-dependent propensity, a placebo and an undesigned action."""
    rng = np.random.default_rng([seed, 7])
    ctx = rng.normal(size=(n, 1))
    p1 = 1 / (1 + np.exp(-ctx[:, 0]))
    prop = np.column_stack([1 - p1, p1])
    a0 = (rng.random(n) < p1).astype(float)
    pl = rng.uniform(0, 1, n)
    conf = ctx[:, 0] + 0.1 * rng.normal(size=n)
    y = np.column_stack([0.5 * a0 + ctx[:, 0] + 0.3 * rng.normal(size=n)])
    lag = np.vstack([[np.nan], y[:-1]])
    designs = (
        Design("logged", {"name": "categorical_rows", "values": [0.0, 1.0]}, propensity=prop),
        Design("iid", {"name": "uniform", "lo": 0.0, "hi": 1.0}),
        Design("none"),
    )
    names = ("P0", "P_placebo", "P_placebo_conf")
    cands = tuple((x, "K0") for x in names)
    meta = {"context_names": ["Z"]}
    return Dataset(
        "E4",
        "R3",
        n,
        seed,
        names,
        ("K0",),
        np.column_stack([a0, pl, conf]),
        lag,
        y,
        designs,
        cands,
        context=ctx,
        meta=meta,
    )


def skeleton_episode(rng: np.random.Generator, delta: float = 0.0) -> list[tuple[float, ...]]:
    """Units of one episode with a mode-dependent skeleton; the outcome is fixed beforehand."""
    big_t, h, thr, dwell, rho = 720, 90, 0.8, 150, 0.995
    d = np.zeros(big_t)
    e = rng.normal(0, np.sqrt(1 - rho**2), big_t)
    for t in range(1, big_t):
        d[t] = rho * d[t - 1] + e[t]
    y = 3.0 * d + rng.normal(0, 1, big_t)
    units, t, blocked = [], 120, -1
    while t < big_t - h:
        if t >= blocked and abs(d[t]) > thr:
            m = int(rng.choice(3, p=POLICY))
            units.append((t, 1.0 if d[t] > 0 else -1.0, m))
            if LEVELS[m] > 0:
                blocked, t = t + dwell, t + 1
            else:
                t += 60
            continue
        t += 1
    for t0, g, m in units:
        y[t0 : t0 + h] += delta * g * LEVELS[m] / h
    return [
        (t0, g, m, y[t0 : t0 + h].sum() - y[t0 - h : t0].sum(), y[t0 - h : t0].sum())
        for t0, g, m in units
    ]


def skeleton_units(
    rng: np.random.Generator, n_ep: int, delta: float = 0.0
) -> dict[str, np.ndarray]:
    """Decision units of `n_ep` episodes in information order."""
    rows = [(e, *u) for e in range(n_ep) for u in skeleton_episode(rng, delta)]
    a = np.array(rows, float).reshape(-1, 6)
    return {
        "episode": a[:, 0].astype(int),
        "sgn": a[:, 2],
        "mode": a[:, 3].astype(int),
        "y": a[:, 4],
        "pre": a[:, 5],
    }


@pytest.fixture
def make_rows() -> Callable[..., Dataset]:
    return rows_dataset


@pytest.fixture
def make_bump() -> Callable[..., Dataset]:
    return bump_dataset


@pytest.fixture
def make_logged() -> Callable[..., Dataset]:
    return logged_dataset


@pytest.fixture
def make_units() -> Callable[..., dict[str, Any]]:
    return skeleton_units


@pytest.fixture
def unit_design() -> tuple[np.ndarray, np.ndarray]:
    """Logging policy and dose per mode of the skeleton units."""
    return POLICY, LEVELS

"""PMRT redraw engine: design-based conditional randomization tests.

Only the known random part of one assignment is redrawn; every other quantity, including
the weights, stays fixed. All statistics of one call share one set of redraws.
"""

import dataclasses
from collections.abc import Callable

import numpy as np

from oran_causal.core.design import Draw
from oran_causal.core.pvalues import ExceedanceCounter

Statistic = Callable[[np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class RedrawConfig:
    """Redraw settings.

    Attributes:
        b_max: Maximum number of redraws B.
        chunk: Redraws per chunk.
        max_chunk_bytes: Memory cap of one chunk of redraws.
        seq_h: Besag-Clifford stopping count; None for fixed B.
    """

    b_max: int = 9999
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    seq_h: int | None = 20


@dataclasses.dataclass(frozen=True)
class Redraw:
    """Observed centred assignment, its design variance and its redraw law.

    Attributes:
        v: Observed centred assignment, shape `[n]`.
        var: Design variance per row, shape `[n]`.
        draw: Centred redraws, `draw(rng, b)` of shape `[b, n]`.
    """

    v: np.ndarray
    var: np.ndarray
    draw: Draw


def chunk_size(config: RedrawConfig, n: int) -> int:
    """Redraws per chunk under the memory cap for `n` rows."""
    return int(max(1, min(config.chunk, config.max_chunk_bytes // (8 * max(n, 1)))))


def _exceed(zb: np.ndarray, z_obs: np.ndarray, tol: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.abs(zb) >= (np.abs(z_obs) - tol)[None, :],
            zb >= (z_obs - tol)[None, :],
            zb <= (z_obs + tol)[None, :],
        ],
        axis=1,
    )


def crt(redraw: Redraw, w: np.ndarray, rng: np.random.Generator, config: RedrawConfig) -> dict:
    """CRT of the linear statistic sum_t v_t w_t for every column of `w`.

    z = S / sigma with S = v' w and sigma^2 = sum_t Var(v_t) w_t^2. The two-sided p counts
    |z_b| >= |z| and the one-sided p-values count z_b >= z and z_b <= z, each with a
    relative tolerance of 1e-9.

    Args:
        redraw: Assignment and its redraw law.
        w: Fixed weights, shape `[n, k]`.
        rng: Redraw stream.
        config: Redraw settings.

    Returns:
        `z`, `sd`, `p2`, `p_plus`, `p_minus`, `draws` (per column) and `draws_total`.
    """
    m = w.shape[1]
    sd = np.sqrt(redraw.var @ (w * w))
    sd_safe = np.where(sd > 0, sd, 1.0)
    z_obs = (redraw.v @ w) / sd_safe
    tol = 1e-9 * (1.0 + np.abs(z_obs))
    ch = chunk_size(config, len(redraw.v))
    counter = ExceedanceCounter(3 * m, config.seq_h)
    while counter.done < config.b_max and not counter.all_stopped():
        b = min(ch, config.b_max - counter.done)
        zb = (redraw.draw(rng, b) @ w) / sd_safe[None, :]
        counter.update(_exceed(zb, z_obs, tol))
    p, used = counter.pvalues(config.b_max)
    return {
        "z": z_obs,
        "sd": sd,
        "p2": p[:m],
        "p_plus": p[m : 2 * m],
        "p_minus": p[2 * m :],
        "draws": used[:m],
        "draws_total": counter.done,
    }


def crt_stat(
    stat: Statistic, v_obs: np.ndarray, draw: Draw, rng: np.random.Generator, config: RedrawConfig
) -> dict:
    """One-sided CRT of a statistic T(V), large values being evidence.

    `config.chunk` is used as given (no memory cap).

    Args:
        stat: Maps redraws of shape `[b, n]` to statistics of shape `[b, k]`.
        v_obs: Observed centred assignment, shape `[n]`.
        draw: Centred redraws.
        rng: Redraw stream.
        config: Redraw settings.

    Returns:
        `t`, `p`, `draws` (per statistic) and `draws_total`.
    """
    t_obs = stat(v_obs[None, :])[0]
    tol = 1e-9 * (1.0 + np.abs(t_obs))
    counter = ExceedanceCounter(len(t_obs), config.seq_h)
    while counter.done < config.b_max and not counter.all_stopped():
        b = min(config.chunk, config.b_max - counter.done)
        counter.update(stat(draw(rng, b)) >= (t_obs - tol)[None, :])
    p, used = counter.pvalues(config.b_max)
    return {"t": t_obs, "p": p, "draws": used, "draws_total": counter.done}

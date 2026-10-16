"""Known assignment designs: centred assignment scores and their redraws."""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import norm

from oran_causal.core.data import Design

Draw = Callable[[np.random.Generator, int], np.ndarray]


@dataclasses.dataclass
class Assignment:
    """Centred assignment of one action column.

    Attributes:
        v: Observed centred assignment, shape `[n]`.
        var: Design variance per row, shape `[n]`.
        draw: `draw(rng, b)` returns `b` centred redraws, shape `[b, n]`.
        kind: Design kind.
    """

    v: np.ndarray
    var: np.ndarray
    draw: Draw
    kind: str


Law = tuple[np.ndarray, np.ndarray, Draw]


def rowparam(x: Any, n: int) -> np.ndarray:
    """Broadcasts a scalar or per-row design parameter to shape `[n]`."""
    a = np.asarray(x, float)
    return np.full(n, float(a)) if a.ndim == 0 else a.reshape(n)


def _count_draw(edges: np.ndarray, shared: bool, n: int) -> Callable[[np.ndarray], np.ndarray]:
    def count(u: np.ndarray) -> np.ndarray:
        if edges.shape[1] <= 8:
            m = np.zeros(u.shape, np.int64)
            for j in range(edges.shape[1]):
                m += u >= edges[None, :, j]
            return m
        if shared:
            return np.searchsorted(edges[0], u, side="right")
        m = np.empty(u.shape, np.int64)
        for j in range(n):
            m[:, j] = np.searchsorted(edges[j], u[:, j], side="right")
        return m

    return count


def categorical_law(values: Any, probs: Any, n: int) -> Law:
    """Mean, variance and inverse-CDF redraws of a categorical design.

    Args:
        values: Support, shape `[m]` or `[n, m]`.
        probs: Probabilities, shape `[m]` or `[n, m]`.
        n: Number of rows.

    Returns:
        (mean, var, draw) with centred draws.
    """
    p = np.asarray(probs, float)
    if p.ndim == 1:
        p = np.broadcast_to(p, (n, len(p)))
    vals = np.asarray(values, float)
    if vals.ndim == 1:
        mean = p @ vals
        var = p @ vals**2 - mean**2
    else:
        mean = (p * vals).sum(1)
        var = (p * vals**2).sum(1) - mean**2
    cum = np.cumsum(p, axis=1)
    cum[:, -1] = np.inf
    edges = np.ascontiguousarray(cum[:, :-1])
    count = _count_draw(edges, bool(np.all(edges == edges[:1])), n)

    def draw(rng: np.random.Generator, b: int) -> np.ndarray:
        m = count(rng.random((int(b), n)))
        if vals.ndim == 1:
            v = vals[m]
        else:
            v = np.take_along_axis(np.broadcast_to(vals, (int(b), *vals.shape)), m[:, :, None], 2)[
                :, :, 0
            ]
        v = v.astype(float)
        v -= mean[None, :]
        return v

    return mean, np.maximum(var, 0.0), draw


def _uniform_law(dist: dict[str, Any], n: int) -> Law:
    lo, hi = rowparam(dist["lo"], n), rowparam(dist["hi"], n)
    mean, var = (lo + hi) / 2.0, (hi - lo) ** 2 / 12.0

    def draw(rng: np.random.Generator, b: int) -> np.ndarray:
        return rng.random((int(b), n)) * (hi - lo)[None, :] + (lo - mean)[None, :]

    return mean, var, draw


def _normal_law(dist: dict[str, Any], n: int) -> Law:
    loc = rowparam(dist.get("loc", dist.get("mean", 0.0)), n)
    sc = rowparam(dist.get("scale", dist.get("sd")), n)

    def draw(rng: np.random.Generator, b: int) -> np.ndarray:
        return rng.standard_normal((int(b), n)) * sc[None, :]

    return loc, sc**2, draw


def _clipped_normal_law(dist: dict[str, Any], n: int) -> Law:
    m, sd = rowparam(dist["mean"], n), rowparam(dist["sd"], n)
    lo, hi = rowparam(dist.get("lo", -np.inf), n), rowparam(dist.get("hi", np.inf), n)
    a, b = (lo - m) / sd, (hi - m) / sd
    fa_cdf, fb_cdf, fa, fb = norm.cdf(a), norm.cdf(b), norm.pdf(a), norm.pdf(b)
    lo0, hi0 = np.where(np.isfinite(lo), lo, 0.0), np.where(np.isfinite(hi), hi, 0.0)
    a0, b0 = np.where(np.isfinite(a), a, 0.0), np.where(np.isfinite(b), b, 0.0)
    mean = lo0 * fa_cdf + hi0 * (1 - fb_cdf) + m * (fb_cdf - fa_cdf) + sd * (fa - fb)
    ex2 = (
        lo0**2 * fa_cdf
        + hi0**2 * (1 - fb_cdf)
        + (m**2 + sd**2) * (fb_cdf - fa_cdf)
        + 2 * m * sd * (fa - fb)
        + sd**2 * (a0 * fa - b0 * fb)
    )

    def draw(rng: np.random.Generator, b_: int) -> np.ndarray:
        x = rng.standard_normal((int(b_), n)) * sd[None, :] + m[None, :]
        return np.clip(x, lo[None, :], hi[None, :]) - mean[None, :]

    return mean, np.maximum(ex2 - mean**2, 0.0), draw


_CONTINUOUS = {
    "uniform": _uniform_law,
    "normal": _normal_law,
    "clipped_normal": _clipped_normal_law,
}


def continuous_law(dist: dict[str, Any], n: int) -> Law:
    """Mean, variance and redraws of a uniform, normal or clipped-normal design.

    Raises:
        ValueError: For any other distribution name.
    """
    law = _CONTINUOUS.get(str(dist.get("name")))
    if law is None:
        raise ValueError(f"unsupported design distribution {dist.get('name')!r}")
    return law(dist, n)


def _logged_categorical(design: Design, x: np.ndarray, dist: dict[str, Any]) -> Assignment | str:
    if design.propensity is None:
        return "logged categorical design without propensity"
    mean, var, draw = categorical_law(dist["values"], design.propensity, len(x))
    return Assignment(x - mean, var, draw, design.kind)


def _observed(design: Design, a: np.ndarray) -> np.ndarray | str:
    if design.kind in ("iid", "logged"):
        return np.asarray(a, float)
    if design.kind == "dither":
        if design.random_part is None:
            return "dither design without random_part"
        return np.asarray(design.random_part, float)
    return f"unknown design kind {design.kind!r}"


def assignment(design: Design, a: np.ndarray) -> Assignment | str:
    """Centred assignment of one action column, or the reason it cannot be redrawn.

    Args:
        design: Known design of the column.
        a: Observed column, shape `[n]`.

    Returns:
        The `Assignment`, or a string reason (no known design, unsupported law).
    """
    if design.kind == "none" or design.dist is None:
        if design.kind == "none":
            return "no known design (kind 'none')"
        return f"kind {design.kind!r} without a distribution"
    dist = dict(design.dist)
    x = _observed(design, a)
    if isinstance(x, str):
        return x
    if design.kind == "logged" and dist.get("name") in ("categorical", "categorical_rows"):
        return _logged_categorical(design, x, dist)
    if dist.get("name") == "categorical":
        mean, var, draw = categorical_law(dist["values"], dist["p"], len(x))
    else:
        try:
            mean, var, draw = continuous_law(dist, len(x))
        except (KeyError, ValueError) as ex:
            return f"design distribution not redrawable: {ex}"
    return Assignment(x - mean, var, draw, design.kind)


def v_design(
    mode: np.ndarray, probs: np.ndarray, sgn: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Assignment score of logged decision units.

    v_u = sgn_u (L(W_u) - E_pi_u L), with the dose L of each mode and the logged mode
    probabilities pi_u.

    Args:
        mode: Logged mode index per unit.
        probs: Logged mode probabilities per unit, shape `[n, modes]`.
        sgn: Request direction per unit.
        levels: Dose of every mode.

    Returns:
        Scores, shape `[n]`.
    """
    m = probs @ levels
    return sgn * (levels[mode] - m)


def v_var(probs: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Design variance of `v_design` per unit."""
    m = probs @ levels
    return probs @ levels**2 - m**2

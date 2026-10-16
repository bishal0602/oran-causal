"""Rates with seed-cluster uncertainty and the three-way validity class.

Hits and counts are given per seed. Seeds are the clusters of the bootstrap, in the given
(sorted) order, with a fixed bootstrap seed.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from oran_causal.analysis.controlled.protocol import ALPHA, BOOT_REPS, BOOT_SEED, VALID_UB

Rate = dict[str, Any]
WILSON_Z = 1.959964


def cluster_ci(
    hits: ArrayLike, cnt: ArrayLike, reps: int = BOOT_REPS, seed: int = BOOT_SEED
) -> list[float | None]:
    """Cluster-bootstrap 95 % percentile interval of `sum(hits) / sum(cnt)`.

    Args:
        hits: Hits per cluster.
        cnt: Trials per cluster; clusters without trials are dropped.
        reps: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        The 2.5 % and 97.5 % quantiles, or `[None, None]` with no trials.
    """
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    keep = c > 0
    h, c = h[keep], c[keep]
    if not len(c):
        return [None, None]
    idx = np.random.default_rng(seed).integers(0, len(c), (reps, len(c)))
    bs = h[idx].sum(1) / c[idx].sum(1)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def design_effect(hits: ArrayLike, cnt: ArrayLike) -> float:
    """Returns the design effect of the clustering, at least 1."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    total, k = c.sum(), len(c)
    if total <= 0 or k < 2:
        return 1.0
    p = h.sum() / total
    if not 0 < p < 1:
        return 1.0
    v = k / (k - 1) * float(np.sum((h - p * c) ** 2)) / total**2
    return max(1.0, v / (p * (1 - p) / total))


def wilson_deff(hits: ArrayLike, cnt: ArrayLike, z: float = WILSON_Z) -> list[float | None]:
    """Returns the Wilson interval with the effective sample size `N / design_effect`."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    total = c.sum()
    if total <= 0:
        return [None, None]
    p = h.sum() / total
    n_eff = total / design_effect(h[c > 0], c[c > 0])
    shrink = z * z / n_eff
    mid = (p + shrink / 2) / (1 + shrink)
    half = z * math.sqrt(p * (1 - p) / n_eff + shrink / (4 * n_eff)) / (1 + shrink)
    return [max(0.0, mid - half), min(1.0, mid + half)]


def classify(ci: Sequence[float | None] | None) -> str | None:
    """Three-way validity of one rate.

    INVALID if the interval's lower bound exceeds .05 (takes precedence), VALID if the upper
    bound is at most .075, else INCONCLUSIVE. None without an interval.
    """
    if not ci or ci[0] is None or ci[1] is None:
        return None
    if ci[0] > ALPHA:
        return "INVALID"
    return "VALID" if ci[1] <= VALID_UB else "INCONCLUSIVE"


def rate(hits: ArrayLike, cnt: ArrayLike) -> Rate | None:
    """Summarises a rate over clusters, or None without trials."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    total = float(c.sum())
    if total <= 0:
        return None
    ci = cluster_ci(h, c)
    return {
        "rate": float(h.sum() / total),
        "hits": int(h.sum()),
        "n": int(total),
        "n_clusters": int(np.sum(c > 0)),
        "ci": ci,
        "ci_wilson_deff": wilson_deff(h, c),
        "deff": design_effect(h[c > 0], c[c > 0]),
        "validity": classify(ci),
    }


def combine(classes: Iterable[str | None]) -> str:
    """Combines the classes of several rates.

    INVALID if any is, VALID if all (at least one) are, NO_READ without a class, else
    INCONCLUSIVE.
    """
    cs = [c for c in classes if c is not None]
    if not cs:
        return "NO_READ"
    if "INVALID" in cs:
        return "INVALID"
    return "VALID" if all(c == "VALID" for c in cs) else "INCONCLUSIVE"


def mean_ci(values: Iterable[Any]) -> dict[str, Any] | None:
    """Mean, sd and cluster-bootstrap interval of per-seed values, ignoring None and NaN."""
    v = np.array(
        [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float
    )
    if not len(v):
        return None
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else None
    ci = cluster_ci(v, np.ones(len(v))) if len(v) > 1 else [None, None]
    return {"mean": float(v.mean()), "sd": sd, "n": len(v), "ci": ci}

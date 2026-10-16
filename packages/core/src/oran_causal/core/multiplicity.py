"""Multiple-testing procedures valid under arbitrary dependence.

Every procedure takes p-values (or e-values) in a fixed hypothesis order. A NaN p-value is
untested and never rejected; a NaN e-value counts as 0.
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

W_CAP = 5.0


def c_harm(m: int) -> float:
    """Returns the harmonic number c_m = sum_{k=1}^m 1/k used by the BY reshaping."""
    return float(sum(1.0 / k for k in range(1, int(m) + 1)))


def projected_means(z: np.ndarray, n_prior: int, n_target: int) -> np.ndarray:
    """Projects prior z-scores to the target sample size.

    Returns:
        xi_j = sqrt(max(z_j^2 - 1, 0)) * sqrt(n_target / n_prior).
    """
    z = np.nan_to_num(np.asarray(z, float))
    return np.sqrt(np.maximum(z * z - 1.0, 0.0)) * math.sqrt(float(n_target) / float(n_prior))


def _rw_total(xi: np.ndarray, pos: np.ndarray, alpha: float, c: float) -> np.ndarray:
    m = len(xi)
    w = np.zeros(m)
    w[pos] = (m / alpha) * norm.sf(xi[pos] / 2.0 + c / xi[pos])
    return w


def rw_weights(xi: np.ndarray, alpha: float) -> np.ndarray:
    """Roeder-Wasserman optimal Bonferroni weights for one-sided means.

    w_j = (m / alpha) * Phibar(xi_j / 2 + c / xi_j) for xi_j > 0, else 0, with c chosen so
    that the weights sum to m. All xi <= 0 gives unit weights.

    Args:
        xi: Projected one-sided means.
        alpha: Total Bonferroni level the weights are optimal for.

    Returns:
        Weights with sum m.
    """
    xi = np.asarray(xi, float)
    m = len(xi)
    pos = xi > 1e-9
    if not pos.any():
        return np.ones(m)
    lo, hi = -1.0, 1.0
    while _rw_total(xi, pos, alpha, lo).sum() < m:
        lo *= 2.0
        if lo < -1e6:
            return np.ones(m)
    while _rw_total(xi, pos, alpha, hi).sum() > m:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _rw_total(xi, pos, alpha, mid).sum() > m:
            lo = mid
        else:
            hi = mid
    w = _rw_total(xi, pos, alpha, hi)
    return w * (m / w.sum())


def expected_rejections(xi: np.ndarray, q: float, max_iter: int = 100) -> float:
    """Anticipated BY rejections: the fixed point k = sum_j Phi(xi_j - z_{1-qk/(m c_m)}) >= 1."""
    xi = np.asarray(xi, float)
    m = len(xi)
    cm = c_harm(m)
    k = 1.0
    for _ in range(max_iter):
        a = min(q * k / (m * cm), 0.5)
        k_new = max(1.0, float(norm.cdf(xi - norm.isf(a)).sum()))
        if abs(k_new - k) < 1e-6:
            break
        k = k_new
    return k


def cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """Mean-preserving cap: weights above `cap` are capped and the excess spread over the rest."""
    w = np.asarray(w, float).copy()
    tot = w.sum()
    if not np.isfinite(cap) or cap * len(w) < tot:
        return w
    for _ in range(len(w)):
        hi = w > cap + 1e-12
        if not hi.any():
            break
        w[hi] = cap
        free = w < cap - 1e-12
        rest = tot - w[~free].sum()
        s = w[free].sum()
        w[free] = w[free] * rest / s if s > 0 else rest / max(free.sum(), 1)
    return w


def _finite_or_inf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return np.where(np.isfinite(p), p, np.inf)


def check_weights(w: np.ndarray, m: int) -> None:
    """Raises ValueError unless the weights are nonnegative with sum m (relative tolerance 1e-6)."""
    if np.any(w < 0) or abs(w.sum() - m) > 1e-6 * m:
        raise ValueError("weights must be >= 0 with sum m")


def weighted_step_up(
    p: np.ndarray, w: np.ndarray | None = None, q: float = 0.05, reshape: bool = True
) -> np.ndarray:
    """Weighted step-up of Blanchard and Roquain with the BY or BH reshaping.

    R = max{k : #{j : p_j <= q w_j beta(k) / m} >= k} and p_j <= q w_j beta(R) / m is
    rejected, with beta(k) = k / c_m (BY, any dependence) or beta(k) = k (BH).

    Args:
        p: P-values; NaN marks an untested hypothesis.
        w: Nonnegative weights with sum m; None gives unit weights.
        q: Target FDR level.
        reshape: BY reshaping when True, BH when False.

    Returns:
        Boolean rejection vector.
    """
    p = np.asarray(p, float)
    m = len(p)
    w = np.ones(m) if w is None else np.asarray(w, float)
    check_weights(w, m)
    pp = _finite_or_inf(p)
    cm = c_harm(m) if reshape else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(w > 0, pp / w, np.inf)
    order = np.argsort(r, kind="stable")
    ks = np.arange(1, m + 1)
    ok = r[order] <= q * ks / (m * cm)
    out = np.zeros(m, bool)
    if ok.any():
        k = int(np.nonzero(ok)[0].max()) + 1
        out[order[:k]] = True
    return out


def dag_depths(parents: Sequence[Sequence[int]]) -> np.ndarray:
    """Depth of every node of a DAG given by parent lists (roots have depth 1)."""
    n = len(parents)
    depth = np.zeros(n, int)

    def visit(i: int, seen: tuple[int, ...] = ()) -> int:
        if depth[i]:
            return int(depth[i])
        if i in seen:
            raise ValueError("graph has a cycle")
        depth[i] = 1 + max((visit(j, (*seen, i)) for j in parents[i]), default=0)
        return int(depth[i])

    for i in range(n):
        visit(i)
    return depth


def _children(parents: Sequence[Sequence[int]]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in parents]
    for i, ps in enumerate(parents):
        for j in ps:
            children[j].append(i)
    return children


def dag_effective(parents: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Effective numbers of leaves and nodes below each node (DAGGER eq. 2).

    Returns:
        (l, m): a child's counts split evenly over its parents.
    """
    n = len(parents)
    children = _children(parents)
    depth = dag_depths(parents)
    ell, mm = np.zeros(n), np.zeros(n)
    for i in sorted(range(n), key=lambda i: -depth[i]):
        if not children[i]:
            ell[i] = mm[i] = 1.0
        else:
            ell[i] = sum(ell[b] / len(parents[b]) for b in children[i])
            mm[i] = 1.0 + sum(mm[b] / len(parents[b]) for b in children[i])
    return ell, mm


def _dagger_layer(
    p: np.ndarray, layer: np.ndarray, cand: np.ndarray, threshold: _DaggerThreshold
) -> int:
    for r in range(len(layer), 0, -1):
        if np.sum(p[layer] <= threshold(layer, cand, r)) >= r:
            return r
    return 0


class _DaggerThreshold:
    def __init__(self, ell: np.ndarray, mm: np.ndarray, n_leaves: float, q: float, reshape: bool):
        self.ell, self.mm, self.n_leaves, self.q, self.reshape = ell, mm, n_leaves, q, reshape
        self.n = len(ell)
        self.cn = c_harm(self.n)
        self.r_prev = 0

    def beta(self, x: float) -> float:
        return min(x, self.n) / self.cn if self.reshape else x

    def __call__(self, layer: np.ndarray, cand: np.ndarray, r: int) -> np.ndarray:
        q, ell, mm, nl = self.q, self.ell, self.mm, self.n_leaves
        return np.array(
            [
                q * ell[i] / nl * self.beta(mm[i] + r + self.r_prev - 1) / mm[i] if c else -1.0
                for i, c in zip(layer, cand, strict=True)
            ]
        )


def dagger(
    p: np.ndarray, parents: Sequence[Sequence[int]], q: float = 0.05, reshape: bool = True
) -> np.ndarray:
    """Reshaped DAGGER (Ramdas et al. 2019, Biometrika, Thm 1(b)).

    A node is testable only when all its parents are rejected. The reshaping
    beta(x) = min(x, N) / c_N keeps FDR <= q under arbitrary dependence.

    Args:
        p: P-values; NaN marks an untested hypothesis.
        parents: Parent indices of every node.
        q: Target FDR level.
        reshape: Apply the reshaping.

    Returns:
        Boolean rejection vector.
    """
    p = _finite_or_inf(p)
    depth = dag_depths(parents)
    ell, mm = dag_effective(parents)
    is_leaf = np.ones(len(p), bool)
    for ps in parents:
        for j in ps:
            is_leaf[j] = False
    threshold = _DaggerThreshold(ell, mm, float(is_leaf.sum()), q, reshape)
    rej = np.zeros(len(p), bool)
    for d in range(1, int(depth.max()) + 1):
        layer = np.nonzero(depth == d)[0]
        cand = np.array([all(rej[j] for j in parents[i]) for i in layer], bool)
        r_d = _dagger_layer(p, layer, cand, threshold)
        if r_d:
            rej[layer[p[layer] <= threshold(layer, cand, r_d)]] = True
        threshold.r_prev += int(rej[layer].sum())
    return rej


def _holm_update(
    w: np.ndarray, g: np.ndarray, j: int, act: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    wn = np.where(act, w + w[j] * g[j], 0.0)
    den = 1.0 - g[:, j] * g[j, :]
    gn = np.where(
        den[:, None] > 1e-12,
        (g + np.outer(g[:, j], g[j])) / np.maximum(den, 1e-12)[:, None],
        0.0,
    )
    gn[~act, :] = 0.0
    gn[:, ~act] = 0.0
    np.fill_diagonal(gn, 0.0)
    return wn, gn


def graphical_holm(p: np.ndarray, w0: np.ndarray, g: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Graphical Bonferroni-Holm of Bretz et al. (2009), Algorithm 1 (FWER <= alpha).

    Args:
        p: P-values; NaN marks an untested hypothesis.
        w0: Initial weights with sum <= 1.
        g: Transition matrix with rows summing to <= 1 and a zero diagonal.
        alpha: Target FWER level.

    Returns:
        Boolean rejection vector.
    """
    p = _finite_or_inf(p)
    w = np.asarray(w0, float).copy()
    g = np.asarray(g, float).copy()
    act = np.ones(len(p), bool)
    rej = np.zeros(len(p), bool)
    while True:
        cand = np.nonzero(act & (p <= w * alpha))[0]
        if not len(cand):
            break
        j = int(cand[np.argmin(p[cand] / np.maximum(w[cand], 1e-300))])
        rej[j], act[j] = True, False
        w, g = _holm_update(w, g, j, act)
    return rej


def _group_pvalues(p: np.ndarray, w: np.ndarray, gi: np.ndarray, n_groups: int) -> np.ndarray:
    pg = np.full(n_groups, np.inf)
    for grp in range(n_groups):
        sel = gi == grp
        wg = w[sel].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(w[sel] > 0, p[sel] * wg / w[sel], np.inf)
        pg[grp] = min(1.0, float(v.min())) if len(v) else np.inf
    return pg


def p_filter(
    p: np.ndarray,
    groups: Sequence[object],
    w: np.ndarray | None = None,
    q: float = 0.05,
    reshape: bool = True,
) -> np.ndarray:
    """Two-layer reshaped p-filter (Ramdas, Barber, Wainwright and Jordan 2019, AoS).

    Layer 1 is the leaves with weights `w`; layer 2 is `groups` with weighted-Bonferroni
    group p-values. The maximal fixed point is reached by monotone iteration from t = q.

    Args:
        p: Leaf p-values; NaN marks an untested hypothesis.
        groups: Group label of every leaf.
        w: Leaf weights with sum m; None gives unit weights.
        q: Target FDR level for both layers.
        reshape: Apply the BY reshaping on both layers.

    Returns:
        Boolean rejection vector over the leaves.
    """
    p = _finite_or_inf(p)
    m = len(p)
    w = np.ones(m) if w is None else np.asarray(w, float)
    labels, gi = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = len(labels)
    pg = _group_pvalues(p, w, gi, n_groups)
    c1 = c_harm(m) if reshape else 1.0
    c2 = c_harm(n_groups) if reshape else 1.0
    t1, t2 = q / c1, q / c2

    def selected(t1_: float, t2_: float) -> np.ndarray:
        return (p <= w * t1_) & (pg[gi] <= t2_)

    for _ in range(1000):
        n1 = q * int(selected(t1, t2).sum()) / (m * c1)
        n2 = q * len(np.unique(gi[selected(n1, t2)])) / (n_groups * c2)
        if n1 >= t1 - 1e-15 and n2 >= t2 - 1e-15:
            break
        t1, t2 = min(t1, n1), min(t2, n2)
    return selected(t1, t2)


def e_bh(e: np.ndarray, q: float = 0.05, w: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) e-BH of Wang and Ramdas (2022).

    Rejects the k* largest w_j e_j with k* = max{k : (w e)_(k) >= m / (q k)}.

    Args:
        e: E-values; NaN counts as 0.
        q: Target FDR level.
        w: Nonnegative weights with sum m; None gives the unweighted procedure.

    Returns:
        Boolean rejection vector.
    """
    e = np.nan_to_num(np.asarray(e, float), nan=0.0)
    m = len(e)
    if w is not None:
        w = np.asarray(w, float)
        check_weights(w, m)
        e = e * w
    out = np.zeros(m, bool)
    order = np.argsort(-e, kind="stable")
    ok = e[order] >= m / (q * np.arange(1, m + 1))
    if ok.any():
        out[order[: int(np.nonzero(ok)[0].max()) + 1]] = True
    return out

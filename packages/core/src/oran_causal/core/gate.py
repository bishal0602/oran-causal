"""Edge-certification gate of the referee.

A request class (f, d) that a map-driven referee can defer is certified when every guard
edge (f, rel, k) is either declared in the map or has a one-sided 90 % upper bound of the
deferral harm -d * beta at most tau(f, d, k). An uncertified class is never deferred.
tau(f, d, k) = (rho - 1) G_k / (max(N(f, d), 1) * n_rel), so deferring every unit of the
class moves the episode guard ratio by at most rho - 1.
"""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.core.ridge import predictable_residuals

Z90 = 1.2815515655446004
RHO = 1.05
DIRS = (1, -1)
DECLARED, CERTIFIED, UNRESOLVED = "DECLARED", "CERTIFIED", "UNRESOLVED"

Bound = Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class GateSpace:
    """Request classes and guard edges the gate checks.

    Attributes:
        families: Parameter families.
        relations: Relations of a guard edge.
        guard_kpis: Guard KPIs.
        dirs: Request directions.
    """

    families: Sequence[str]
    relations: Sequence[str]
    guard_kpis: Sequence[str]
    dirs: Sequence[int] = DIRS


def tau_table(
    g: Mapping[str, float], n: Mapping[tuple[str, int], float], space: GateSpace, rho: float = RHO
) -> dict:
    """Gate tolerances tau(f, d, k).

    Args:
        g: Mean per-episode guard total under accept-all, per guard KPI.
        n: Mean units per episode of class (f, d) under accept-all; missing is 0, floored at 1.
        space: Classes and guard edges.
        rho: Target episode guard ratio of one class.

    Returns:
        `{(f, d, k): tau}`.
    """
    out = {}
    for f in space.families:
        for d in space.dirs:
            units = max(float(n.get((f, d), 0.0)), 1.0)
            for k in space.guard_kpis:
                out[(f, d, k)] = (float(rho) - 1.0) * float(g[k]) / (units * len(space.relations))
    return out


def bound_ub(b: Bound | None, h: int, z: float = Z90) -> float:
    """One-sided upper bound of h * beta from a bound record; +inf when there is none.

    A `{"kind": "slope", "beta", "se"}` record gives h beta + z se; a
    `{"kind": "ci", "lo", "hi"}` record gives max(h lo, h hi).
    """
    if not b:
        return math.inf
    if b["kind"] == "slope":
        beta, se = b.get("beta"), b.get("se")
        if beta is None or se is None or not (np.isfinite(beta) and np.isfinite(se)):
            return math.inf
        return float(h * float(beta) + float(z) * float(se))
    if b["kind"] == "ci":
        lo, hi = b.get("lo"), b.get("hi")
        if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
            return math.inf
        return float(max(h * float(lo), h * float(hi)))
    raise ValueError(f"unknown bound kind {b.get('kind')}")


def classify_edge(
    m: Mapping[Any, Any],
    bounds: Mapping[Any, Bound] | None,
    edge: tuple[str, str, str],
    d: int,
    tau: float,
) -> tuple[str, float]:
    """Status and bound of one guard edge for a deferral of direction d (harm direction -d)."""
    if edge in m:
        return DECLARED, float("nan")
    ub = bound_ub((bounds or {}).get(edge), -int(d))
    return (CERTIFIED if ub <= float(tau) else UNRESOLVED), ub


def _class_edges(
    m: Mapping[Any, Any],
    bounds: Mapping[Any, Bound] | None,
    tau: Mapping,
    cls: tuple[str, int],
    space: GateSpace,
) -> tuple[list[list[Any]], bool]:
    f, d = cls
    edges, ok = [], True
    for rel in space.relations:
        for k in space.guard_kpis:
            t = tau[(f, d, k)]
            st, ub = classify_edge(m, bounds, (f, rel, k), d, t)
            edges.append([rel, k, st, ub if math.isfinite(ub) else None, t])
            ok &= st != UNRESOLVED
    return edges, ok


def certify(
    m: Mapping[Any, Any],
    bounds: Mapping[Any, Bound] | None,
    tau: Mapping,
    can_defer: Callable[[str, int], bool],
    space: GateSpace,
) -> dict:
    """Certifies every request class of a map.

    Args:
        m: Declared map, keyed by (f, rel, k).
        bounds: Bound record per guard edge, or None when the arm has no bounds.
        tau: Gate tolerances from `tau_table`.
        can_defer: Whether the referee defers class (f, d) in some context.
        space: Classes and guard edges.

    Returns:
        `classes`: per `"f|d"` its `can_defer`, `edges` (`[rel, k, status, ub, tau]`) and
        `certified`; `uncertified`: the deferrable classes that are not certified.
    """
    classes, unc = {}, []
    for f in space.families:
        for d in space.dirs:
            cd = can_defer(f, d)
            edges, ok = _class_edges(m, bounds, tau, (f, d), space) if cd else ([], True)
            classes[f"{f}|{d}"] = {"can_defer": cd, "edges": edges, "certified": bool(ok)}
            if cd and not ok:
                unc.append([f, d])
    return {"classes": classes, "uncertified": unc}


def design_slope_se(
    v: np.ndarray, y: np.ndarray, pre: np.ndarray | None, episode: np.ndarray, sgn: np.ndarray
) -> dict[str, Any]:
    """Design-based slope of a guard edge with its martingale (HC0) standard error.

    beta = sum v r / sum v^2 and se = sqrt(sum v^2 (r - beta v)^2) / sum v^2, where v is
    the assignment score and r the outcome minus its predictable running centre. Arrays in
    information order; a pre-window with a non-finite value is ignored.

    Returns:
        `{"kind": "slope", "beta", "se", "n", "z"}`.
    """
    n = len(v)
    if n < 2:
        return {"kind": "slope", "beta": 0.0, "se": None, "n": n, "z": None}
    pre_used = pre if pre is not None and np.all(np.isfinite(pre)) else None
    r, _ = predictable_residuals(y, pre_used, episode, sgn)
    vv = float(v @ v)
    if vv <= 0:
        return {"kind": "slope", "beta": 0.0, "se": None, "n": n, "z": None}
    beta = float(v @ r / vv)
    e = r - beta * v
    se = float(np.sqrt(np.sum(v * v * e * e)) / vv)
    return {"kind": "slope", "beta": beta, "se": se, "n": n, "z": beta / se if se > 0 else None}

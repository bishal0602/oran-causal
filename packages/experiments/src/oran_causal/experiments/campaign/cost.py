"""Cost projections and DEV seed-count calculations.

Paired recall gap: the smallest S >= 2 with
S >= ((t_{1-alpha/2,S-1} + t_{power,S-1}) sd_d / delta)^2.
Null-rate precision: S = ceil(z^2 deff p0 (1 - p0) / (h^2 m)).
"""

import dataclasses
import itertools
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from oran_causal.experiments.campaign.run import arm_budgets
from oran_causal.experiments.campaign.spec import ProtocolTexts, Unit, expand

type Agg = dict[str, Any]
type CostTable = dict[str, float]


def paired_seeds(
    sd_d: float, delta: float = 0.15, alpha: float = 0.05, power: float = 0.8, s_max: int = 100_000
) -> int:
    """Seeds for a two-sided paired t-test to detect a mean recall gap `delta`."""
    if not np.isfinite(sd_d) or sd_d <= 0:
        return 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    s = max(2, math.ceil(((za + zb) * sd_d / delta) ** 2))
    while s < s_max:
        need = ((stats.t.ppf(1 - alpha / 2, s - 1) + stats.t.ppf(power, s - 1)) * sd_d / delta) ** 2
        if s >= need:
            return s
        s += 1
    return s_max


def null_rate_seeds(
    m_per_seed: float, deff: float = 1.0, p0: float = 0.05, half: float = 0.02, z: float = 1.959964
) -> int | None:
    """Seeds so the 95 % CI half-width of a rate at `p0` is at most `half`."""
    if not m_per_seed or m_per_seed <= 0:
        return None
    return math.ceil(z * z * max(deff, 1.0) * p0 * (1 - p0) / (half * half * m_per_seed))


def design_effect(hits: Any, cnt: Any) -> float:
    """Seed-cluster design effect of a pooled rate (>= 1; 1 when undefined)."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    total, k = c.sum(), len(c)
    if total <= 0 or k < 2:
        return 1.0
    p = h.sum() / total
    if not 0 < p < 1:
        return 1.0
    v_clu = k / (k - 1) * float(np.sum((h - p * c) ** 2)) / total**2
    return max(1.0, v_clu / (p * (1 - p) / total))


@dataclasses.dataclass(frozen=True)
class PowerTarget:
    """Detectable recall gap, test level and power of the paired seed-count rule."""

    delta: float = 0.15
    alpha: float = 0.05
    power: float = 0.8


def _valid(entry: dict[str, Any]) -> bool:
    p = entry.get("primary")
    return bool(p) and p.get("valid_flag") == "ok"


def _pair(a: str, b: str, arms: dict[str, Any], target: PowerTarget) -> dict[str, Any] | None:
    ra, rb = arms[a]["primary"]["per_seed"], arms[b]["primary"]["per_seed"]
    seeds = sorted(
        s for s in set(ra) & set(rb) if ra[s]["recall"] is not None and rb[s]["recall"] is not None
    )
    if len(seeds) < 2:
        return None
    d = np.array([ra[s]["recall"] - rb[s]["recall"] for s in seeds])
    sd = float(np.std(d, ddof=1))
    return {
        "a": a,
        "b": b,
        "n_seeds": len(seeds),
        "mean_d": float(d.mean()),
        "sd_d": sd,
        "seeds_needed": paired_seeds(sd, target.delta, target.alpha, target.power),
        "sd_zero": sd == 0.0,
    }


def _by_cell(agg: Agg) -> dict[str, dict[str, Any]]:
    by_cell: dict[str, dict[str, Any]] = {}
    for k, e in agg["cells"].items():
        by_cell.setdefault(k.split("|", 1)[1], {})[e["arm"]] = e
    return by_cell


def recall_power(
    agg: Agg,
    delta: float = 0.15,
    alpha: float = 0.05,
    power: float = 0.8,
    focal: str | None = "pmrt_eq",
) -> dict[str, Any]:
    """Seeds needed per cell for every pair of valid arms; `focal` pairs are the primary ones."""
    out = {}
    for cell, arms in sorted(_by_cell(agg).items()):
        valid = sorted(a for a, e in arms.items() if _valid(e))
        pairs = [
            p
            for a, b in itertools.combinations(valid, 2)
            if (p := _pair(a, b, arms, PowerTarget(delta, alpha, power)))
        ]
        foc = [p for p in pairs if focal in (p["a"], p["b"])] if focal else pairs
        out[cell] = {
            "valid_arms": valid,
            "invalid_arms": sorted(set(arms) - set(valid)),
            "pairs": pairs,
            "max_focal": max((p["seeds_needed"] for p in foc), default=None),
            "max_all": max((p["seeds_needed"] for p in pairs), default=None),
        }
    return {
        "delta": delta,
        "alpha": alpha,
        "power": power,
        "focal": focal,
        "cells": out,
        "max_over_cells_focal": max(
            (v["max_focal"] for v in out.values() if v["max_focal"]), default=None
        ),
        "max_over_cells_all": max(
            (v["max_all"] for v in out.values() if v["max_all"]), default=None
        ),
    }


def null_precision(agg: Agg, half: float = 0.02, p0: float = 0.05) -> dict[str, Any]:
    """Seeds per cell so the truth-null and placebo rate CIs have half-width at most `half`."""
    out = {}
    for k, e in sorted(agg["cells"].items()):
        p = e.get("primary")
        if not p:
            continue
        row = {}
        for name in ("null_raw05", "null_decl", "plac_raw05", "plac_decl"):
            r = p.get(name)
            if not r or not r.get("n_seeds"):
                continue
            m = r["n"] / r["n_seeds"]
            row[name] = {
                "m_per_seed": m,
                "seeds_needed": null_rate_seeds(m, r.get("deff", 1.0), p0, half),
            }
        out[k] = row
    return out


def cost_table_from_agg(agg: Agg, wall_arms: frozenset[str] = frozenset()) -> CostTable:
    """(arm|world|regime|n) -> mean cost per dataset: CPU-s, or wall-s for the GPU arms."""
    out = {}
    for k, v in agg["cost"].items():
        x = v.get("wall_s_mean") if k.split("|", 1)[0] in wall_arms else v["cpu_s_mean"]
        if x is not None:
            out[k] = x
    return out


def _n_of(key: str) -> int:
    return int(key.rsplit("|n", 1)[1])


def _infeasible_cost(i: dict[str, Any]) -> float | None:
    if i.get("cost") is not None:
        return i["cost"]
    return i.get("cpu_s") if i.get("measured", True) else None


def measured_infeasible(agg: Agg) -> dict[str, tuple[int, float, str]]:
    """Arm -> (smallest n with a measured over-budget unit, its cost, kind)."""
    out: dict[str, tuple[int, float, str]] = {}
    for k, v in agg["cost"].items():
        arm, n = k.split("|", 1)[0], _n_of(k)
        for i in v.get("infeasible", []):
            c = _infeasible_cost(i)
            if c is not None and (arm not in out or n < out[arm][0]):
                out[arm] = (n, float(c), i.get("cost_kind") or "CPU-s")
    return out


def _points(table: Mapping[str, float], prefix: str) -> list[tuple[int, float]]:
    return [(_n_of(k), v) for k, v in table.items() if k.startswith(prefix) and v and v > 0]


def fit_cost(
    table: Mapping[str, float], arm: str, world: str, regime: str, n: int
) -> tuple[float | None, str]:
    """Measured mean, else a power law in n fit over the arm's (world, regime) or all points."""
    k = f"{arm}|{world}|{regime}|n{n}"
    if k in table:
        return table[k], "measured"
    pts = _points(table, f"{arm}|{world}|{regime}|n")
    if len(pts) < 2:
        pts = _points(table, f"{arm}|")
    if len(pts) < 2:
        return None, "unknown"
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    b, a = np.polyfit(x, y, 1)
    return float(math.exp(a + b * math.log(n))), f"extrapolated (n^{b:.2f})"


class _Projection:
    def __init__(self, agg: Agg, full: dict[str, Any]) -> None:
        self.agg, self.full = agg, full
        self.gpu_arms = frozenset(a for a, d in full["arms"].items() if d.get("budget_wall_s"))
        self.table = cost_table_from_agg(agg, self.gpu_arms)
        self.inf = measured_infeasible(agg)
        self.per_arm: dict[str, float] = defaultdict(float)
        self.per_n: dict[int, float] = defaultdict(float)
        self.per_role: dict[str, float] = defaultdict(float)
        self.gen = 0.0
        self.unknown: set[str] = set()
        self.capped: dict[str, int] = defaultdict(int)
        self.infeasible: dict[str, int] = defaultdict(int)
        self.how_used: dict[str, set[str]] = defaultdict(set)
        self.seen_ds: set[Any] = set()

    def _generation(self, u: Unit) -> float:
        g = [
            v["gen_cpu_s_mean"]
            for k, v in self.agg["cost"].items()
            if k.endswith(f"|{u.world}|{u.regime}|n{u.n}") and v["gen_cpu_s_mean"]
        ]
        return float(np.mean(g)) if g else 0.0

    def add(self, u: Unit) -> None:
        if u.arm in self.inf and u.n >= self.inf[u.arm][0]:
            self.infeasible[f"{u.arm}|n{u.n}"] += 1
            return
        c, how = fit_cost(self.table, u.arm, u.world, u.regime, u.n)
        if c is None:
            self.unknown.add(u.arm)
            return
        self.how_used[u.arm].add(how.split(" ")[0])
        b = arm_budgets(self.full, u.arm, self.full.get("budget_cpu_s"))
        cap = (b[1] if u.arm in self.gpu_arms else b[0]) or math.inf
        if c > cap:
            self.capped[f"{u.arm}|n{u.n}"] += 1
            c = cap
        self.per_arm[u.arm] += c
        if u.arm not in self.gpu_arms:
            self.per_n[u.n] += c
            self.per_role[u.role] += c
        if u.dataset not in self.seen_ds:
            self.seen_ds.add(u.dataset)
            self.gen += self._generation(u)

    def result(self, n_units: int) -> dict[str, Any]:
        cpu = sum(v for a, v in self.per_arm.items() if a not in self.gpu_arms) + self.gen
        return {
            "n_units": n_units,
            "n_datasets": len(self.seen_ds),
            "cpu_h_total": cpu / 3600,
            "gpu_wall_h_total": sum(v for a, v in self.per_arm.items() if a in self.gpu_arms)
            / 3600,
            "cpu_h_by_arm": {
                k: v / 3600 for k, v in sorted(self.per_arm.items()) if k not in self.gpu_arms
            },
            "gpu_wall_h_by_arm": {
                k: v / 3600 for k, v in sorted(self.per_arm.items()) if k in self.gpu_arms
            },
            "cpu_h_by_n": {str(k): v / 3600 for k, v in sorted(self.per_n.items())},
            "cpu_h_by_role": {k: v / 3600 for k, v in self.per_role.items()},
            "cpu_h_generation": self.gen / 3600,
            "arms_without_cost": sorted(self.unknown),
            "cost_basis_by_arm": {k: sorted(v) for k, v in sorted(self.how_used.items())},
            "measured_infeasible": {
                a: {"n": v[0], "cost": v[1], "kind": v[2]} for a, v in sorted(self.inf.items())
            },
            "units_infeasible_not_run": dict(self.infeasible),
            "units_capped_at_budget": dict(self.capped),
        }


def project(agg: Agg, full: dict[str, Any], texts: ProtocolTexts | None = None) -> dict[str, Any]:
    """Projected cost of the spec `full` from a pilot aggregate.

    CPU-h for CPU arms and GPU wall-h for arms with a wall budget, per arm, n and role. Units at
    n >= an arm's measured infeasible n are counted apart; extrapolations over the budget are
    capped at the budget and counted.
    """
    proj = _Projection(agg, full)
    units = expand(full, texts)
    for u in units:
        proj.add(u)
    return proj.result(len(units))

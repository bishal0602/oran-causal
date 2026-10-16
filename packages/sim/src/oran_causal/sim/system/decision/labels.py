"""Unit labels by paired lookahead rollouts.

A unit u = (c, x, t0) is labelled with probability `LABEL_RATE[x]`, drawn from
`default_rng([seed, LABEL_TAG, c, x_idx, int(t0)])`. For every mode m and every reseed k the env
is copied at t0 (between propose and apply, so the copy carries t0's requests) with reseed k,
m is held for u's T (`HoldPolicy`), every other unit is accepted, and the rollout runs H seconds.
Outcomes per rollout are the network totals and the per-cell sums of the KPI tap.

The label is the contrast against accept per reseed (same reseed in both arms: common random
numbers): `d[m][scope][kpi]` for scope "exp" (sum over N(c)) and "net" (whole network), with
`mean` over reseeds. The accept rollout is a direct accept-all rollout from t0, cached per
(t0, k). With `cells=True` every label also carries `cell[m]`, the per-cell contrast
(reseeds, `CELL_KPIS`, cells), used by the knockout ground truth.

A region probe holds the same mode on all of x's cells in the unit's region and on each cell
alone, and reports network contrasts against accept-all.
"""

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from oran_causal.sim.system.decision.collect import PI0_HIGH, PI0_ORDER, get_tap
from oran_causal.sim.system.decision.units import T_UNIT, X_KNOBS, HoldPolicy, Policy, UnitArbiter
from oran_causal.sim.system.decision.vocab import CELL_KPIS
from oran_causal.sim.system.env import E6Env

LABEL_TAG, REGION_TAG = 6613, 6614
LABEL_RATE = {"ES": 0.3, "PowerES": 0.3, "Coverage": 0.3, "SliceGuarantee": 0.05}
REGION_RATE = 0.2
H_LABEL = 90
KS = (1, 2)
KPIS = ("pv", "e", "v")

Rollout = dict[str, Any]


class Paused(NamedTuple):
    """An env paused between propose and apply at t0, its observation and an arbiter fork.

    Attributes:
        env: Env carrying t0's pending requests.
        obs: Observation of t0.
        snap: Arbiter fork taken before t0's call.
    """

    env: E6Env
    obs: Mapping[str, Any]
    snap: UnitArbiter


def label_modes(x: str) -> tuple[str, ...]:
    """Returns pi0's support for xApp `x`, accept first."""
    return tuple(m for m in PI0_ORDER if m in PI0_HIGH[x])


def _u(seed: int, tag: int, unit: Mapping[str, Any]) -> float:
    key = [int(seed), tag, int(unit["c"]), int(unit["x_idx"]), int(unit["t0"])]
    return float(np.random.default_rng(key).random())


def sampled(seed: int, unit: Mapping[str, Any], rate: float | None = None) -> bool:
    """Returns whether the unit is labelled (rate defaults to `LABEL_RATE[x]`)."""
    return _u(seed, LABEL_TAG, unit) < (LABEL_RATE[unit["x"]] if rate is None else rate)


def region_sampled(seed: int, unit: Mapping[str, Any], rate: float = REGION_RATE) -> bool:
    """Returns whether the unit gets a region probe."""
    return _u(seed, REGION_TAG, unit) < rate


def _snapshot(sim: E6Env, ext: bool = False) -> tuple[tuple[float, float, float], tuple]:
    s, tap = sim.plant.sla, get_tap(sim)
    net = (float(s["prot_viol"]), float(s["energy_j"]), float(s["viol_ue_s"]))
    return net, tap.arrays(ext)


def _diff(sim: E6Env, s0: tuple, n_sec: int, ext: bool = False) -> Rollout:
    net1, cell1 = _snapshot(sim, ext)
    return {
        "net": np.array(net1) - np.array(s0[0]),
        "cell": np.stack([a - b for a, b in zip(cell1, s0[1], strict=True)]),
        "secs": n_sec,
    }


def rollout(at: Paused, policy: Policy, k: int, H: int = H_LABEL, ext: bool = False) -> Rollout:
    """Runs one lookahead rollout from a paused env.

    The env is copied with reseed k, t0 is decided by `at.snap.fork(policy)` on `at.obs`, and H
    more seconds run under the same arbiter.

    Returns:
        `{"net": [pv, e, v], "cell": (3, cells) or, with `ext`, (6, cells) in `CELL_KPIS`
        order, "secs"}`.
    """
    sim = at.env.copy(reseed=k)
    s0 = _snapshot(sim, ext)
    arb = at.snap.fork(policy)
    sim.step_apply(arb(at.obs))
    n = 0
    while n < H and sim.sec < sim.total_s:
        o = sim.step_propose()
        sim.step_apply(arb(o))
        n += 1
    return _diff(sim, s0, n, ext)


def aa_rollout(env: E6Env, k: int, H: int = H_LABEL, ext: bool = False) -> Rollout:
    """Runs a direct accept-all rollout with no arbiter, the reference of the accept label."""
    sim = env.copy(reseed=k)
    s0 = _snapshot(sim, ext)
    sim.step_apply({"decisions": ["accept"] * len(sim.pending_requests), "writes": []})
    n = 0
    while n < H and sim.sec < sim.total_s:
        sim.step(None)
        n += 1
    return _diff(sim, s0, n, ext)


def _vals(res: Rollout, cells: Iterable[int]) -> dict[str, list[float]]:
    idx = list(cells)
    return {
        "net": [float(x) for x in res["net"]],
        "exp": [float(res["cell"][i, idx].sum()) for i in range(3)],
    }


class Labeller:
    """Per-episode labeller with accept-all rollouts cached per (t0, k).

    `cells` keeps the per-cell contrasts (module docstring). `cpu_s` and `n_roll` account for
    the labelling cost.
    """

    def __init__(
        self, ks: Sequence[int] = KS, H: int = H_LABEL, T: float = T_UNIT, cells: bool = False
    ) -> None:
        self.ks, self.H, self.T = tuple(ks), int(H), float(T)
        self.cells = bool(cells)
        self.cache: dict[tuple[float, int], Rollout] = {}
        self.cpu_s, self.n_roll = 0.0, 0

    def aa(self, at: Paused, t0: float, k: int) -> Rollout:
        """Returns the cached accept-all rollout of (t0, k); only t0's entries are kept."""
        key = (float(t0), int(k))
        if key not in self.cache:
            self.cache = {kk: v for kk, v in self.cache.items() if kk[0] == float(t0)}
            pol = HoldPolicy((), "accept", 0, 0)
            self.cache[key] = rollout(at, pol, k, self.H, self.cells)
            self.n_roll += 1
        return self.cache[key]

    def hold(
        self, at: Paused, keys: Iterable[Any], mode: str, k: int, window: tuple[float, float]
    ) -> Rollout:
        """Returns the rollout holding `mode` for units of `keys` opening inside `window`."""
        self.n_roll += 1
        return rollout(at, HoldPolicy(keys, mode, *window), k, self.H, self.cells)

    def _arm(self, at: Paused, unit: Mapping[str, Any], m: str, k: int) -> Rollout:
        if m == "accept":
            return self.aa(at, unit["t0"], k)
        t0 = unit["t0"]
        return self.hold(at, {(unit["c"], unit["x"])}, m, k, (t0, t0 + 1.0))

    def _contrasts(self, modes: Sequence[str], raw: Mapping[str, list]) -> tuple[dict, dict]:
        d, mean = {}, {}
        n = len(self.ks)
        for m in modes:
            d[m] = {
                sc: {
                    kpi: [raw[m][i][sc][j] - raw["accept"][i][sc][j] for i in range(n)]
                    for j, kpi in enumerate(KPIS)
                }
                for sc in ("exp", "net")
            }
            mean[m] = {sc: {kpi: float(np.mean(v)) for kpi, v in d[m][sc].items()} for sc in d[m]}
        return d, mean

    def label(
        self, at: Paused, unit: Mapping[str, Any], modes: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Labels one unit (module docstring); `modes` defaults to `label_modes(x)`."""
        t_cpu = time.process_time()
        modes = tuple(modes) if modes is not None else label_modes(unit["x"])
        raw: dict[str, list] = {}
        cellres: dict[str, list] = {}
        for m in modes:
            raw[m], cellres[m] = [], []
            for k in self.ks:
                res = self._arm(at, unit, m, k)
                raw[m].append(_vals(res, unit["exp"]) | {"secs": res["secs"]})
                cellres[m].append(res["cell"])
        d, mean = self._contrasts(modes, raw)
        out: dict[str, Any] = {
            "modes": list(modes),
            "H": self.H,
            "ks": list(self.ks),
            "raw": raw,
            "d": d,
            "mean": mean,
        }
        if self.cells:
            out["cell_kpis"] = list(CELL_KPIS)
            out["cell"] = {
                m: np.stack([cellres[m][i] - cellres["accept"][i] for i in range(len(self.ks))])
                for m in modes
            }
        cpu = time.process_time() - t_cpu
        self.cpu_s += cpu
        out["cpu_s"] = round(cpu, 3)
        return out

    def _cell_contrasts(
        self, at: Paused, unit: Mapping[str, Any], cells: list[int], mode: str, label: Any
    ) -> dict[int, list]:
        x, t0 = unit["x"], unit["t0"]
        aa = [self.aa(at, t0, k)["net"] for k in self.ks]
        out = {}
        for q in cells:
            if q == unit["c"] and label is not None and mode in label["d"]:
                net = label["d"][mode]["net"]
                out[q] = [[net[kpi][i] for kpi in KPIS] for i in range(len(self.ks))]
                continue
            out[q] = [
                (self.hold(at, {(q, x)}, mode, k, (t0, t0 + self.T))["net"] - a).tolist()
                for k, a in zip(self.ks, aa, strict=True)
            ]
        return out

    def region_probe(
        self,
        at: Paused,
        unit: Mapping[str, Any],
        site: Sequence[int] | np.ndarray,
        mode: str = "reject",
        label: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Probes the unit's region; None when x acts on fewer than 2 cells of the region.

        `label` (the unit's label) supplies the own-cell rollout when it contains `mode`.
        """
        c, x, t0 = unit["c"], unit["x"], unit["t0"]
        types = X_KNOBS[x]
        acts = sorted({int(k[1]) for k in at.obs["static"]["knobs"] if k[0] in types})
        g = int(site[c])
        cells = [q for q in acts if int(site[q]) == g]
        if len(cells) < 2:
            return None
        t_cpu = time.process_time()
        aa = [self.aa(at, t0, k)["net"] for k in self.ks]
        keys = {(q, x) for q in cells}
        d_region = [
            (self.hold(at, keys, mode, k, (t0, t0 + self.T))["net"] - a).tolist()
            for k, a in zip(self.ks, aa, strict=True)
        ]
        d_cell = self._cell_contrasts(at, unit, cells, mode, label)
        sum_c = [np.sum([d_cell[q][i] for q in cells], 0).tolist() for i in range(len(self.ks))]
        cpu = time.process_time() - t_cpu
        self.cpu_s += cpu
        return {
            "region": g,
            "cells": cells,
            "mode": mode,
            "dR": d_region,
            "dC": {str(q): v for q, v in d_cell.items()},
            "sum_dC": sum_c,
            "kpis": list(KPIS),
            "cpu_s": round(cpu, 3),
        }

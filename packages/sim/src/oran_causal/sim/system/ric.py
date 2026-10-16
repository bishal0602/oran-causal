"""Near-RT RIC layer of the system-level plant.

KPM reporting (granularity, delay, drops), the knob registry with hard actuator limits, and the
request plumbing. The WG3 lock and rollback actions and the churn cap live in
`oran_causal.sim.system.env.E6Env.step_apply`; every change still passes `feasible` here.

Knob keys: `("cio", s, n)` dB, `("hys", c)` dB, `("ttt", c)` ms, `("ll_ratio", c)` fraction,
`("sleep", c)` 0/1, `("carrier", c)` active macro carriers 1..`MACRO_NTRX`. With the power and
protected-slice extension: `("ptx", c)` Tx-power offset in dB and `("prot_min", c)` protected
work-conserving minimum PRB share. Their limits come from the plant's `cfg.e6p`; `LIMITS` holds the
`E6PConfig()` defaults for callers that only need a reference range.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from oran_causal.sim.system import config as C
from oran_causal.sim.system.sim import Plant, tape_rng

Knob = tuple

LIMITS: dict[str, tuple[float, float, float | None, float]] = {
    "cio": (C.CIO_RANGE[0], C.CIO_RANGE[1], 2.0, 5.0),  # [S] range, [A] rate
    "hys": (C.HYS_RANGE[0], C.HYS_RANGE[1], 0.5, 10.0),  # [A]
    "ttt": (min(C.TTT_SET_MS), max(C.TTT_SET_MS), None, 10.0),
    "ll_ratio": (0.0, 0.5, 0.10, 1.0),  # [A] TS 28.541 ratio
    "sleep": (0, 1, None, 120.0),  # [A]
    "carrier": (1, C.MACRO_NTRX, 1.0, 120.0),  # [A]
}
_P0 = C.E6PConfig()
LIMITS.update(
    {
        "ptx": (*_P0.ptx_range_db, _P0.ptx_max_step_db, _P0.ptx_min_interval_s),
        "prot_min": (*_P0.prot_min_range, _P0.prot_min_max_step, _P0.prot_min_interval_s),
    }
)
P_TYPES = ("ptx", "prot_min")
GRAN_S = {"fast": 1, "thp": 5, "mob": 30, "energy": 60}  # [A]
MOB_KEYS = ("ho_att", "ho_succ", "too_late", "too_early", "wrong_cell", "pingpong")

_CELL_ATTR = {
    "hys": "hys",
    "ttt": "ttt",
    "ll_ratio": "ll_ratio",
    "sleep": "asleep",
    "carrier": "n_car",
    "ptx": "ptx_off",
    "prot_min": "prot_min",
}
_PLAIN_SET = ("hys", "ll_ratio", "ptx", "prot_min")
_QUANT: dict[str, Callable[[float], float]] = {
    "cio": lambda v: float(np.round(v)),
    "hys": lambda v: float(np.round(v * 2) / 2),
    "ttt": lambda v: float(min(C.TTT_SET_MS, key=lambda x: abs(float(x) - v))),
    "ll_ratio": lambda v: float(np.round(v * 20) / 20),
    "sleep": lambda v: float(v >= 0.5),
    "carrier": lambda v: float(np.round(v)),
}


def knob_get(plant: Plant, k: Knob) -> float:
    """Value of knob `k` in force on `plant`.

    Raises:
        KeyError: for an unknown knob type.
    """
    typ = k[0]
    if typ == "cio":
        return float(plant.cio[k[1], k[2]])
    if typ not in _CELL_ATTR:
        raise KeyError(k)
    return float(getattr(plant, _CELL_ATTR[typ])[k[1]])


def _limits(plant: Plant, typ: str) -> tuple[float, float, float | None, float]:
    if typ == "ptx":
        P = plant.cfg.e6p
        return (*P.ptx_range_db, P.ptx_max_step_db, P.ptx_min_interval_s)
    if typ == "prot_min":
        P = plant.cfg.e6p
        return (*P.prot_min_range, P.prot_min_max_step, P.prot_min_interval_s)
    return LIMITS[typ]


def quantise(k: Knob, v: float, P: C.E6PConfig | None = None) -> float:
    """Round a value onto the actuator grid of knob type `k[0]`.

    Args:
        k: knob key; only its type is read.
        v: requested value.
        P: extension configuration giving the `ptx` and `prot_min` grids (default `E6PConfig()`).

    Returns:
        The quantised value; unknown types pass through.
    """
    typ = k[0]
    if typ in P_TYPES:
        P = P or _P0
        g = P.ptx_grid_db if typ == "ptx" else P.prot_min_grid
        return float(np.round(v / g) * g)
    if typ in _QUANT:
        return _QUANT[typ](v)
    return float(v)


def feasible(
    plant: Plant, k: Knob, v: float, now: float, last_change: Mapping[Knob, float]
) -> tuple[float, bool]:
    """Clip a requested value to the hard actuator limits: range, max step and min interval.

    Returns:
        The value that would be applied and whether the change is allowed now; a refused change
        returns the current value.
    """
    lo, hi, mstep, mint = _limits(plant, k[0])
    cur = knob_get(plant, k)
    if now - last_change.get(k, -1e9) < mint:
        return cur, False
    v = min(max(quantise(k, v, plant.cfg.e6p if k[0] in P_TYPES else None), lo), hi)
    if k[0] == "ttt":
        i0, i1 = C.TTT_SET_MS.index(int(cur)), C.TTT_SET_MS.index(int(v))
        v = float(C.TTT_SET_MS[i0 + int(np.clip(i1 - i0, -1, 1))])
    elif mstep is not None:
        v = cur + float(np.clip(v - cur, -mstep, mstep))
    return v, True


def _set_sleep(plant: Plant, c: int, v: float, now: float) -> None:
    if v >= 0.5 and not plant.asleep[c]:
        plant.asleep[c] = True
        if plant.p_on and plant.P.pico_sleep_ho:
            plant.sleep_handover(c, now)
    elif v < 0.5 and plant.asleep[c]:
        plant.asleep[c] = False
        plant.waking_until[c] = now + C.PICO_WAKE_S


def _set_carrier(plant: Plant, c: int, v: float, now: float) -> None:
    n = int(v)
    if n > plant.n_car[c]:
        plant.car_on_at[c] = now + C.CARRIER_ON_S
    plant.n_car[c] = n


def knob_set(plant: Plant, k: Knob, v: float, now: float) -> None:
    """Write knob `k` on `plant` without any actuator check; unknown types are ignored."""
    typ = k[0]
    if typ == "cio":
        plant.cio[k[1], k[2]] = v
    elif typ == "ttt":
        plant.ttt[k[1]] = int(v)
    elif typ in _PLAIN_SET:
        getattr(plant, _CELL_ATTR[typ])[k[1]] = v
    elif typ == "sleep":
        _set_sleep(plant, k[1], v, now)
    elif typ == "carrier":
        _set_carrier(plant, k[1], v, now)


class KPM:
    """Aggregates 1 s plant counters into E2SM-KPM-like reports, delivered with delay and drops."""

    def __init__(self, cfg: C.E6Config, plant: Plant) -> None:
        self.cfg, self.plant = cfg, plant
        self.buf: dict[str, list[dict[str, Any]]] = {g: [] for g in GRAN_S}
        self.in_flight: list[tuple[float, dict[str, Any]]] = []
        self.delivered: list[dict[str, Any]] = []
        if cfg.kpm == "degraded":
            self.delay, self.drop = (2.0, 5.0), 0.05
        else:
            self.delay, self.drop = tuple(cfg.kpm_delay_s), cfg.kpm_drop

    def second(self, sec: int) -> None:
        """Close simulated second `sec` (window [sec-1, sec)) and emit the reports that end now."""
        c = self.plant.take_counters()
        for g in self.buf:
            self.buf[g].append(c)
        salt = self.plant.rng_salt
        r = tape_rng(self.cfg.seed, "kpm", sec, salt)
        for g, per in GRAN_S.items():
            if sec % per:
                continue
            rep = self._make(g, self.buf[g], sec - per, sec)
            self.buf[g] = []
            if r.uniform() < self.drop:
                continue
            self.in_flight.append((sec + r.uniform(*self.delay), rep))

    def deliver(self, now: float) -> list[dict[str, Any]]:
        """Reports whose delivery time has come, oldest first, each with its `arrived` time."""
        due = [x for x in self.in_flight if x[0] <= now]
        self.in_flight = [x for x in self.in_flight if x[0] > now]
        out = [dict(rep, arrived=t) for t, rep in sorted(due, key=lambda z: z[0])]
        self.delivered.extend(out)
        return out

    def _make(self, g: str, cs: list[dict[str, Any]], t0: int, t1: int) -> dict[str, Any]:
        ticks = max(sum(c["ticks"] for c in cs), 1)
        rep: dict[str, Any] = {"gran": g, "t0": t0, "t1": t1}
        if g == "fast":
            self._make_fast(rep, cs, ticks)
        elif g == "thp":
            bits = sum(c["bits"] for c in cs)
            rep["thp_slice"] = bits / (len(cs) * C.CONTROL_S)
            e = self._pool(cs, "embb_thp")
            rep["embb_thp_p5"] = np.array([np.percentile(x, 5) if x else np.nan for x in e])
        elif g == "mob":
            for key in MOB_KEYS:
                rep[key] = sum(c[key] for c in cs)
            rep["rlf"] = sum(c["rlf"] for c in cs)
        elif g == "energy":
            rep["energy_j"] = sum(c["energy_j"] for c in cs)
        if g == "fast" and self.plant.p_on:
            self._make_p(rep, cs, ticks)
        return rep

    def _pool(self, cs: list[dict[str, Any]], key: str) -> list[list[float]]:
        out: list[list[float]] = [[] for _ in range(self.plant.nc)]
        for c in cs:
            for i in range(self.plant.nc):
                out[i].extend(c[key][i])
        return out

    def _make_fast(self, rep: dict[str, Any], cs: list[dict[str, Any]], ticks: int) -> None:
        used = sum(c["prb_used"] for c in cs)
        rsv = sum(c["prb_rsv"] for c in cs)
        cap = sum(c["prb_cap"] for c in cs)
        with np.errstate(invalid="ignore", divide="ignore"):
            rep["prb_util"] = np.where(cap > 0, (used.sum(1) + rsv) / cap, 0.0)
            rep["prb_util_slice"] = np.where(cap[:, None] > 0, used / cap[:, None], 0.0)
            rep["prb_rsv_idle"] = np.where(cap > 0, rsv / cap, 0.0)
        rep["carriers"] = self.plant.n_car.copy()
        rep["act_ue"] = sum(c["act_ue"] for c in cs) / ticks
        d = self._pool(cs, "ll_delay")
        rep["ll_delay_p95"] = np.array([np.percentile(x, 95) if x else np.nan for x in d])
        rep["ll_samples"] = np.array([len(x) for x in d])

    def _make_p(self, rep: dict[str, Any], cs: list[dict[str, Any]], ticks: int) -> None:
        p, nc = self.plant, self.plant.nc
        rep["ptx_db"] = p.ptx_off.copy()
        rep["ptx_dbm"] = np.where(p.lay.is_macro, C.MACRO_PTX_DBM, C.PICO_PTX_DBM) + p.ptx_off
        srv = np.concatenate([x for c in cs for x in c["sinr_serv"]] or [np.zeros(0, int)])
        val = np.concatenate([x for c in cs for x in c["sinr_val"]] or [np.zeros(0)])
        edge = np.full(nc, np.nan)
        cnt = np.bincount(srv, minlength=nc)
        for i in np.nonzero(cnt)[0]:
            edge[i] = np.percentile(val[srv == i], p.P.edge_pct)
        rep["edge_sinr_p"] = edge
        rep["sinr_samples"] = cnt
        if p.p_prot:
            self._make_prot(rep, cs, ticks)

    def _make_prot(self, rep: dict[str, Any], cs: list[dict[str, Any]], ticks: int) -> None:
        p = self.plant
        cap = sum(c["prb_cap"] for c in cs)
        span = len(cs) * C.CONTROL_S
        with np.errstate(invalid="ignore", divide="ignore"):
            rep["prot_min"] = p.prot_min.copy()
            rep["prot_dem_share"] = sum(c["prot_dem"] for c in cs) / ticks
            used = sum(c["prot_used"] for c in cs)
            rep["prot_used_share"] = np.where(cap > 0, used / cap, 0.0)
            min_prb = sum(c["prot_min_prb"] for c in cs)
            rep["prot_min_share"] = np.where(cap > 0, min_prb / cap, 0.0)
            rep["prot_thp"] = sum(c["prot_bits"] for c in cs) / span
            rep["prot_act_ue"] = sum(c["prot_act"] for c in cs) / ticks
            ev = sum(c["prot_eval"] for c in cs)
            rep["prot_eval"] = ev
            below = sum(c["prot_below"] for c in cs)
            rep["prot_below_frac"] = np.where(ev > 0, below / np.maximum(ev, 1), np.nan)

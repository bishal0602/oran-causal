"""Decision units of the learned-arbiter studies.

A unit (cell c, xApp x, t0) opens at x's first request on c when no (c, x) unit is active, gets
a mode from a policy at that moment, and the mode holds for every request of x on c while
t < t0 + T. The cell of a request is the knob's own cell `knob[1]`.

Modes: accept, reject, half (the slew rule of `half_step_p`), lock (reject and lock the knob
until the unit ends, plant-identical to reject when every knob has one writer, `LOCK_EQUIV`),
and a rollback flag "+rb" for SliceGuarantee and PowerES: at unit open, x's knob on c is rolled
back if the arbiter saw it change within `RB_WINDOW` seconds and that change was not its own
rollback.

`UnitArbiter` reads only the observation: the static cell graph, the applied configuration,
delivered KPM reports and the requests. Knob change times are inferred from consecutive
`obs["config"]` snapshots. The exposure set N(c) is c plus its out- and in-neighbours.
"""

import copy
import warnings
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.sim.system.config import E6PConfig
from oran_causal.sim.system.ric import LIMITS

Unit = dict[str, Any]
Policy = Callable[[Unit], tuple[str, float]]
Knob = tuple[Any, ...]
Obs = Mapping[str, Any]
Decision = str | tuple[str, float]

XAPPS_P = ("ES", "PowerES", "SliceGuarantee", "Coverage")
X_KNOBS = {
    "ES": ("carrier", "sleep"),
    "PowerES": ("ptx",),
    "SliceGuarantee": ("prot_min",),
    "Coverage": ("ptx",),
}
RB_XAPPS = ("SliceGuarantee", "PowerES")
BASE_MODES = ("accept", "half", "reject")
T_UNIT = 60.0
RB_WINDOW = 60.0
LOCK_EQUIV = {"P1": "reject", "P3": "reject"}
QUANTUM = {"cio": 1.0, "hys": 0.5, "ll_ratio": 0.05, "sleep": 1.0, "carrier": 1.0}
_P = E6PConfig()
QUANTUM_P = dict(QUANTUM, ptx=_P.ptx_grid_db, prot_min=_P.prot_min_grid)
OPEN_RULES = ("first", "feasible")
MIN_INTERVAL = {k: v[3] for k, v in LIMITS.items()} | {
    "ptx": _P.ptx_min_interval_s,
    "prot_min": _P.prot_min_interval_s,
}

REPORT_MEDIATORS = (
    "prb_util",
    "act_ue",
    "edge_sinr_p",
    "prot_dem_share",
    "prot_used_share",
    "prot_min_share",
    "prot_thp",
    "prot_act_ue",
    "prot_below_frac",
)
CONFIG_MEDIATORS = ("carriers", "asleep", "ptx_db", "prot_min")
MEDIATORS = REPORT_MEDIATORS + CONFIG_MEDIATORS
W_REPORTS = 5
NBR_AGG = {
    "prb_util": ("mean", "max"),
    "act_ue": ("sum",),
    "edge_sinr_p": ("mean", "min"),
    "prot_dem_share": ("mean", "max"),
    "prot_used_share": ("mean",),
    "prot_min_share": ("mean",),
    "prot_thp": ("sum",),
    "prot_act_ue": ("sum",),
    "prot_below_frac": ("mean", "max"),
    "carriers": ("mean",),
    "asleep": ("sum",),
    "ptx_db": ("mean",),
    "prot_min": ("mean",),
}
_AGG = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin, "sum": np.nansum}


def knob_region(site: Sequence[int], knob: Knob) -> int:
    """Returns the region (site) of the knob's own cell."""
    return int(site[knob[1]])


def split_mode(mode: str) -> tuple[str, bool]:
    """Splits a mode into its base mode and rollback flag, e.g. "half+rb" -> ("half", True).

    Raises:
        ValueError: Unknown mode.
    """
    base, _, rb = mode.partition("+")
    if base not in (*BASE_MODES, "lock") or rb not in ("", "rb"):
        raise ValueError(f"unknown mode {mode!r}")
    return base, rb == "rb"


def x_index(x: str) -> int:
    """Returns the stable index of an xApp in RNG keys."""
    return XAPPS_P.index(x)


def half_step_p(knob: Knob, cur: float, prop: float, hs: dict[Knob, int]) -> Decision:
    """Returns the "half" decision of one request under the slew rule with the E6-P quanta.

    A step of at least two quanta becomes ("modify", cur + ceil(n / 2) quanta toward prop). A
    single-quantum step alternates accept and reject per knob, starting in the phase given by
    the parity of the knob's cell indices; `hs` holds the per-knob count and is updated.
    """
    q = QUANTUM_P.get(knob[0])
    n = abs(prop - cur) / q if q else 0.0
    if n >= 2 - 1e-6:
        k = int(np.ceil(round(n / 2, 6)))
        return ("modify", float(cur + np.sign(prop - cur) * k * q))
    c = hs.get(knob, 0)
    hs[knob] = c + 1
    phase = sum(int(x) for x in knob[1:]) % 2
    return "accept" if (c + phase) % 2 == 0 else "reject"


def exposure_sets(static: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    """Returns N(c) = {c} + out- and in-neighbours of every cell, from `obs["static"]`."""
    nb = static["neighbours"]
    n = int(static["cells"])
    out = {c: {c} | {int(v) for v in nb[c]} for c in range(n)}
    for c in range(n):
        for v in nb[c]:
            out[int(v)].add(c)
    return {c: tuple(sorted(s)) for c, s in out.items()}


def accept_all_policy(unit: Unit) -> tuple[str, float]:
    """Returns accept with propensity 1 for every unit."""
    return "accept", 1.0


class HoldPolicy:
    """Holds `mode` for units of the (c, x) pairs in `keys` that open in [t_lo, t_hi).

    Every other unit is accepted with propensity 1.
    """

    def __init__(self, keys: Any, mode: str, t_lo: float, t_hi: float) -> None:
        self.keys, self.mode = frozenset(keys), mode
        self.t_lo, self.t_hi = float(t_lo), float(t_hi)

    def __call__(self, unit: Unit) -> tuple[str, float]:
        """Returns the unit's mode and propensity."""
        if (unit["c"], unit["x"]) in self.keys and self.t_lo <= unit["t0"] < self.t_hi:
            return self.mode, 1.0
        return "accept", 1.0


class UnitArbiter:
    """Obs-only WG3 arbiter over (cell, xApp, t0) units.

    `units` keeps every opened unit when `record` is set; `opened` holds the units opened by the
    latest call. A policy returns (mode, propensity) for a unit dict holding c, x, x_idx, t0,
    knob, exp and the obs-only context `ctx`. Before `warmup_s` every request is accepted and no
    unit opens. `open_rule` "feasible" passes a request whose knob is still inside its actuator
    min interval as accept without opening a unit; "first" opens a unit at x's first request.
    """

    def __init__(
        self,
        policy: Policy | None,
        T: float = T_UNIT,
        warmup_s: float = 0.0,
        record: bool = True,
        open_rule: str = "first",
    ) -> None:
        if open_rule not in OPEN_RULES:
            raise ValueError(f"open_rule in {OPEN_RULES}")
        self.policy = policy
        self.T, self.warmup_s = float(T), float(warmup_s)
        self.record, self.open_rule = record, open_rule
        self.active: dict[tuple[int, str], Unit] = {}
        self.units: list[Unit] = []
        self.opened: list[Unit] = []
        self.hs: dict[Knob, int] = {}
        self.last_chg: dict[Knob, float] = {}
        self.rb_at: dict[Knob, float] = {}
        self.prev_cfg: Mapping[Knob, float] | None = None
        self.prev_t = 0.0
        self.prev_rb: tuple[Knob, ...] = ()
        self.fast: deque[Mapping[str, Any]] = deque(maxlen=W_REPORTS)
        self.exp: dict[int, tuple[int, ...]] = {}
        self.static: Mapping[str, Any] | None = None
        self._med_t: float | None = None
        self._med = np.zeros((0, len(MEDIATORS)))

    def _observe(self, obs: Obs) -> None:
        cfg = obs["config"]
        if self.prev_cfg is not None:
            for k, v in cfg.items():
                if abs(v - self.prev_cfg.get(k, v)) > 1e-9:
                    self.last_chg[k] = self.prev_t
                    if k in self.prev_rb:
                        self.rb_at[k] = self.prev_t
        self.prev_cfg, self.prev_t = cfg, obs["t"]
        for rep in obs["new_reports"]:
            if rep["gran"] == "fast":
                self.fast.append(rep)
        if self.static is None:
            self.static = obs["static"]
            self.exp = exposure_sets(self.static)

    def _static(self) -> Mapping[str, Any]:
        if self.static is None:
            raise RuntimeError("no observation seen yet")
        return self.static

    def _report_mediators(self, m: np.ndarray) -> None:
        for j, f in enumerate(REPORT_MEDIATORS):
            vals = [np.asarray(r[f], float) for r in self.fast if f in r]
            if not vals:
                continue
            vals = [v.sum(1) if v.ndim == 2 else v for v in vals]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m[:, j] = np.nanmean(np.stack(vals), 0)

    def mediators(self, obs: Obs) -> np.ndarray:
        """Returns the (cells, len(MEDIATORS)) mediator matrix at this second; NaN = no data."""
        if self._med_t == obs["t"]:
            return self._med
        static = self._static()
        n = int(static["cells"])
        m = np.full((n, len(MEDIATORS)), np.nan)
        if self.fast:
            self._report_mediators(m)
        cfg, macro = obs["config"], np.asarray(static["is_macro"], bool)
        j0 = len(REPORT_MEDIATORS)
        for c in range(n):
            m[c, j0] = cfg.get(("carrier", c), np.nan) if macro[c] else np.nan
            m[c, j0 + 1] = cfg.get(("sleep", c), 0.0) if not macro[c] else 0.0
            m[c, j0 + 2] = cfg.get(("ptx", c), 0.0)
            m[c, j0 + 3] = cfg.get(("prot_min", c), 0.0)
        self._med_t, self._med = obs["t"], m
        return m

    def context(self, c: int, r: Mapping[str, Any], obs: Obs) -> dict[str, Any]:
        """Returns the obs-only context of a unit opened by request `r` on cell `c`.

        The context holds the request, c's mediators and aggregates of the mediators over
        N(c) minus c.
        """
        m = self.mediators(obs)
        k = r["knob"]
        t = obs["t"]
        ctx: dict[str, Any] = {
            "knob": k[0],
            "is_macro": float(bool(self._static()["is_macro"][c])),
            "cur": float(r["cur"]),
            "prop": float(r["prop"]),
            "step": float(r["prop"] - r["cur"]),
            "since_change": float(min(t - self.last_chg[k], 1e4)) if k in self.last_chg else 1e4,
            "n_exp": len(self.exp[c]),
        }
        nb = [v for v in self.exp[c] if v != c]
        for j, f in enumerate(MEDIATORS):
            ctx["own_" + f] = float(m[c, j])
            col = m[nb, j] if nb else np.full(1, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for a in NBR_AGG[f]:
                    agg = float(_AGG[a](col)) if np.isfinite(col).any() else float("nan")
                    ctx[f"nbr_{a}_{f}"] = agg
        return ctx

    def dwell_blocked(self, k: Knob, now: float) -> bool:
        """Returns whether the knob changed less than its actuator min interval ago."""
        t = self.last_chg.get(k)
        return t is not None and now - t < MIN_INTERVAL[k[0]]

    def _open(self, r: Mapping[str, Any], obs: Obs, rb: list[Knob]) -> Unit:
        x, k = r["xapp"], r["knob"]
        c, now = int(k[1]), obs["t"]
        u: Unit = {
            "c": c,
            "x": x,
            "x_idx": x_index(x),
            "t0": float(now),
            "knob": k[0],
            "exp": list(self.exp[c]),
            "ctx": self.context(c, r, obs),
        }
        if self.policy is None:
            raise RuntimeError("arbiter has no policy")
        mode, p = self.policy(u)
        _, want_rb = split_mode(mode)
        if want_rb and x not in RB_XAPPS:
            raise ValueError(f"rollback flag not allowed for {x}")
        u.update(mode=mode, p=float(p), n_req=0, n_changed=0)
        self.active[(c, x)] = u
        self.opened.append(u)
        if self.record:
            self.units.append(u)
        if want_rb:
            t_ch = self.last_chg.get(k)
            if (
                t_ch is not None
                and now - t_ch <= RB_WINDOW
                and self.rb_at.get(k) != t_ch
                and k not in rb
            ):
                rb.append(k)
                u["rb_issued"] = True
        return u

    def _decide(self, u: Unit, r: Mapping[str, Any], now: float) -> Decision:
        base, _ = split_mode(u["mode"])
        u["n_req"] += 1
        if base == "accept":
            return "accept"
        if base == "reject":
            return "reject"
        if base == "half":
            return half_step_p(r["knob"], r["cur"], r["prop"], self.hs)
        return ("lock", float(max(u["t0"] + self.T - now, 1e-6)))

    def __call__(self, obs: Obs) -> dict[str, Any]:
        """Returns the decision dict `{"decisions", "writes": [], "rollback"}` for `obs`."""
        self._observe(obs)
        now = obs["t"]
        reqs = obs["requests"]
        self.opened = []
        if now < self.warmup_s:
            self.prev_rb = ()
            return {"decisions": ["accept"] * len(reqs), "writes": [], "rollback": []}
        for key in [key for key, u in self.active.items() if now >= u["t0"] + self.T]:
            del self.active[key]
        dec: list[Decision] = []
        rb: list[Knob] = []
        for r in reqs:
            u = self.active.get((int(r["knob"][1]), r["xapp"]))
            if u is None and self.open_rule == "feasible" and self.dwell_blocked(r["knob"], now):
                dec.append("accept")
                continue
            if u is None:
                u = self._open(r, obs, rb)
            dec.append(self._decide(u, r, now))
        self.prev_rb = tuple(rb)
        return {"decisions": dec, "writes": [], "rollback": rb}

    def fork(self, policy: Policy | None, record: bool = False) -> UnitArbiter:
        """Returns an independent copy of the observation state with a new policy.

        The copy has no active units, so every unit re-opens under `policy`. For lookahead
        rollouts: take it from a snapshot made before this second's call, then call it on the
        same obs.
        """
        pol, self.policy = self.policy, None
        units, opened, active = self.units, self.opened, self.active
        self.units, self.opened, self.active = [], [], {}
        try:
            c = copy.deepcopy(self)
        finally:
            self.policy, self.units, self.opened, self.active = pol, units, opened, active
        c.policy, c.record = policy, record
        return c

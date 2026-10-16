"""Randomised and confounded unit collection on the system-level plant.

Logging policies (`policy(unit) -> (mode, propensity)`):

- `RandomizedUnitPolicy`: context-free pi0. One uniform per unit from
  `default_rng([seed, PI0_TAG, c, x_idx, int(t0)])`, modes in `PI0_ORDER`, so a unit re-opening
  at the same second inside an `env.copy` rollout draws the same mode. Tables: `PI0_HIGH`,
  `PI0_LOW` (accept 0.8, other modes rescaled then floored at `P_MIN`), `PI0_HIGH_NO_RB` and
  `PI0_V4` (accept 0.5 / reject 0.5).
- `PlaceboPolicy`: draws and logs the pi0 mode but applies accept, a sharp null.
- `IncumbentPolicy`: context-dependent accept / reject from the unit's obs-only context only.
  Pressure s = max(own_prb_util, nbr_max_prb_util); the request class (saving, restore, SG
  raise, SG lower) and s or the protected below-floor share pick P(accept) from
  `INCUMBENT_TABLE`, clipped to [`INC_FLOOR`, 1 - `INC_FLOOR`]. The full probability row is
  logged as `unit["probs"]`.
- `PlaceboIncumbent`: the incumbent draw logged, accept applied.

`CellKPITap` wraps `plant.take_counters` and accumulates per cell, over scored seconds, the
protected violated UE-s (pv), all-UE violated UE-s (v) and energy J (e), plus RLFs, served UE-s
(load) and PRBs used, and keeps a per-second per-cell series of every second. These are
privileged plant counters: outcomes and labels only, never features.
"""

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from oran_causal.sim.system.decision.records import SERIES_FIELDS
from oran_causal.sim.system.decision.units import T_UNIT, Policy, Unit, UnitArbiter
from oran_causal.sim.system.decision.vocab import MODES
from oran_causal.sim.system.env import E6Env

PI0_TAG = 6612
P_MIN = 0.1
PI0_ORDER = MODES
PI0_HIGH = {
    "ES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
    "PowerES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
    "Coverage": {"accept": 0.5, "half": 0.2, "reject": 0.3},
    "SliceGuarantee": {"accept": 0.5, "half": 0.15, "reject": 0.25, "accept+rb": 0.10},
}
PI0_HIGH_NO_RB = {
    "ES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
    "PowerES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
    "SliceGuarantee": {"accept": 0.5, "half": 0.2, "reject": 0.3},
}
LOW_ACCEPT = 0.8
EXT_FIELDS = ("rlf", "load", "prb")

Table = Mapping[str, float]


def low_table(high: Table, accept: float = LOW_ACCEPT, p_min: float | None = P_MIN) -> dict:
    """Returns the low-epsilon table of a high-epsilon one.

    Accept becomes `accept`; the other modes are scaled by (1 - accept) / (1 - high accept),
    then floored at `p_min`, accept absorbing the difference.

    Raises:
        ValueError: The floor leaves accept below `p_min`.
    """
    s = (1 - accept) / (1 - high["accept"])
    out = {m: p * s for m, p in high.items() if m != "accept"}
    if p_min is not None:
        out = {m: max(p, p_min) for m, p in out.items()}
    out = {"accept": 1.0 - sum(out.values()), **out}
    if p_min is not None and out["accept"] < p_min:
        raise ValueError("floor leaves accept below p_min")
    return {m: out[m] for m in PI0_ORDER if m in out}


PI0_LOW = {x: low_table(t) for x, t in PI0_HIGH.items()}
PI0_V4 = {x: {"accept": 0.5, "reject": 0.5} for x in ("ES", "PowerES", "SliceGuarantee")}
PI0 = {"high": PI0_HIGH, "low": PI0_LOW, "high_no_rb": PI0_HIGH_NO_RB, "v4": PI0_V4}


def regime_for(j: int) -> str:
    """Returns the pi0 regime of episode index `j`: high on even, low on odd."""
    return "high" if int(j) % 2 == 0 else "low"


class RandomizedUnitPolicy:
    """Context-free logging policy pi0.

    `tables` overrides the per-xApp mode probabilities, e.g. `{x: {"accept": 1.0}}` for the
    all-accept check; otherwise `regime` picks one of `PI0`.
    """

    def __init__(
        self,
        seed: int,
        regime: str = "high",
        tables: Mapping[str, Table] | None = None,
        p_min: float | None = P_MIN,
    ) -> None:
        if tables is None and regime not in PI0:
            raise ValueError(f"regime in {sorted(PI0)}")
        self.seed, self.regime = int(seed), regime
        self.tables = tables if tables is not None else PI0[regime]
        for x, t in self.tables.items():
            if abs(sum(t.values()) - 1.0) > 1e-9:
                raise ValueError(f"pi0 table for {x} does not sum to 1")
            if p_min is not None and tables is None and min(t.values()) < p_min - 1e-12:
                raise ValueError(f"pi0 table for {x} has a propensity below {p_min}")

    def draw_u(self, unit: Mapping[str, Any]) -> float:
        """Returns the unit's keyed uniform draw."""
        key = [self.seed, PI0_TAG, int(unit["c"]), int(unit["x_idx"]), int(unit["t0"])]
        return float(np.random.default_rng(key).random())

    def __call__(self, unit: Unit) -> tuple[str, float]:
        """Returns the unit's mode and propensity."""
        t = self.tables[unit["x"]]
        u, acc = self.draw_u(unit), 0.0
        modes = [m for m in PI0_ORDER if m in t]
        for m in modes:
            acc += t[m]
            if u < acc:
                return m, float(t[m])
        return modes[-1], float(t[modes[-1]])


class PlaceboPolicy(RandomizedUnitPolicy):
    """Sharp-null placebo: logs the pi0 draw on the unit and applies accept."""

    def __call__(self, unit: Unit) -> tuple[str, float]:
        """Logs the pi0 draw as `pi0_mode` / `pi0_p` and returns ("accept", 1.0)."""
        m, p = super().__call__(unit)
        unit["pi0_mode"], unit["pi0_p"] = m, p
        return "accept", 1.0


INC_TAG = 6622
INC_FLOOR = 0.15
INC_S_HI = 0.6
INC_SG_RAISE_FRAC = 0.05
INC_ORDER = ("accept", "reject")
INCUMBENT_TABLE = {
    "saving": {"lo": 0.85, "hi": 0.15},
    "restore": {"lo": 0.35, "hi": 0.85},
    "sg_raise": {"met": 0.85, "else": 0.30},
    "sg_lower": {"met": 0.85, "else": 0.20},
}
INC_SAVING_DIR = {"carrier": -1, "sleep": 1, "ptx": -1}


def _nz(x: Any) -> float:
    try:
        v = float(x)
    except TypeError, ValueError:
        return 0.0
    return 0.0 if not np.isfinite(v) else v


def incumbent_class(ctx: Mapping[str, Any]) -> str:
    """Returns "saving", "restore", "sg_raise" or "sg_lower" from the context's knob and step.

    Raises:
        ValueError: No rule for the knob family.
    """
    f = str(ctx["knob"])
    d = 1 if float(ctx["step"]) >= 0 else -1
    if f == "prot_min":
        return "sg_raise" if d > 0 else "sg_lower"
    if f in INC_SAVING_DIR:
        return "saving" if d == INC_SAVING_DIR[f] else "restore"
    raise ValueError(f"IncumbentPolicy: no rule for knob family {f!r}")


def incumbent_pressure(ctx: Mapping[str, Any]) -> float:
    """Returns s = max(own_prb_util, nbr_max_prb_util) over the finite ones, 0 if none."""
    vals = [float(ctx.get(k, np.nan)) for k in ("own_prb_util", "nbr_max_prb_util")]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else 0.0


def incumbent_consts() -> dict[str, Any]:
    """Returns the incumbent policy constants logged in collection headers."""
    return {
        "table": INCUMBENT_TABLE,
        "floor": INC_FLOOR,
        "s_hi": INC_S_HI,
        "sg_raise_frac": INC_SG_RAISE_FRAC,
        "tag": INC_TAG,
        "saving_dir": INC_SAVING_DIR,
        "pressure": "max(ctx own_prb_util, ctx nbr_max_prb_util)",
    }


class IncumbentPolicy:
    """Context-dependent confounded logging policy (module docstring).

    Reads only `unit["ctx"]` and the RNG key fields (c, x_idx, t0); logs `unit["probs"]` and
    `unit["inc"]`.
    """

    def __init__(
        self,
        seed: int,
        table: Mapping[str, Mapping[str, float]] | None = None,
        floor: float = INC_FLOOR,
        s_hi: float = INC_S_HI,
        sg_raise_frac: float = INC_SG_RAISE_FRAC,
    ) -> None:
        self.seed, self.table = int(seed), dict(table or INCUMBENT_TABLE)
        self.floor, self.s_hi, self.sg_raise_frac = float(floor), float(s_hi), float(sg_raise_frac)
        if not 0.0 < self.floor <= 0.5:
            raise ValueError("floor in (0, .5]")

    def accept_prob(self, ctx: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        """Returns P(accept) and the class, key, pressure and below-floor share behind it."""
        cls = incumbent_class(ctx)
        s = incumbent_pressure(ctx)
        below = _nz(ctx.get("own_prot_below_frac", np.nan))
        t = self.table[cls]
        if cls in ("saving", "restore"):
            key = "hi" if s >= self.s_hi else "lo"
        elif cls == "sg_raise":
            key = "met" if below > self.sg_raise_frac else "else"
        else:
            key = "met" if below == 0.0 else "else"
        pa = float(min(max(float(t[key]), self.floor), 1.0 - self.floor))
        return pa, {"cls": cls, "key": key, "s": s, "below": below}

    def row(self, unit: Mapping[str, Any]) -> dict[str, float]:
        """Returns the unit's mode-probability row."""
        pa, _ = self.accept_prob(unit["ctx"])
        return {"accept": pa, "reject": 1.0 - pa}

    def draw_u(self, unit: Mapping[str, Any]) -> float:
        """Returns the unit's keyed uniform draw."""
        key = [self.seed, INC_TAG, int(unit["c"]), int(unit["x_idx"]), int(unit["t0"])]
        return float(np.random.default_rng(key).random())

    def __call__(self, unit: Unit) -> tuple[str, float]:
        """Returns the unit's mode and propensity and logs its row and class."""
        pa, info = self.accept_prob(unit["ctx"])
        row = {"accept": pa, "reject": 1.0 - pa}
        unit["probs"], unit["inc"] = row, info
        m = "accept" if self.draw_u(unit) < pa else "reject"
        return m, float(row[m])


class PlaceboIncumbent(IncumbentPolicy):
    """Sharp-null placebo of `IncumbentPolicy`: same draw and logged row, accept applied."""

    def __call__(self, unit: Unit) -> tuple[str, float]:
        """Logs the incumbent draw as `pi0_mode` / `pi0_p` and returns ("accept", 1.0)."""
        m, p = super().__call__(unit)
        unit["pi0_mode"], unit["pi0_p"] = m, p
        return "accept", 1.0


def finalize_units(units: list[Unit]) -> None:
    """Relabels placebo units in place: mode / p = the logged draw, applied_mode = applied."""
    for u in units:
        if "pi0_mode" in u:
            u["applied_mode"] = u["mode"]
            u["mode"], u["p"] = u.pop("pi0_mode"), u.pop("pi0_p")


class CellKPITap:
    """Per-cell KPI tap installed as `plant.take_counters` (module docstring).

    `count_all` makes the cumulative arrays count warm-up seconds too; the series always holds
    every second. Row i of the series is env second t = i + 1.
    """

    def __init__(self, plant: Any, n_sec: int | None = None, count_all: bool = False) -> None:
        self.plant = plant
        self.count_all = bool(count_all)
        n = plant.nc
        self.pv, self.v, self.e = np.zeros(n), np.zeros(n), np.zeros(n)
        self.rlf, self.load, self.prb = np.zeros(n), np.zeros(n), np.zeros(n)
        cfg = plant.cfg
        T = int(n_sec) if n_sec is not None else round(cfg.warmup_s + cfg.scored_s)
        self.ser = np.zeros((max(T, 1), len(SERIES_FIELDS), n), np.float32)
        self.ser_scored = np.zeros(max(T, 1), bool)
        self.n_sec = 0

    def _second(self, c: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
        p = self.plant
        n = p.nc
        serv = p.serv
        vp = getattr(p, "sec_viol_prot", None)
        pv_s = np.bincount(serv[vp], minlength=n) if vp is not None else np.zeros(n, np.int64)
        v_s = (
            np.bincount(serv[p.sec_viol_ll | p.sec_viol_embb | p.sec_viol_be], minlength=n)
            if hasattr(p, "sec_viol_ll")
            else np.zeros(n, np.int64)
        )
        ticks = max(int(c["ticks"]), 1)
        rlf_s = np.asarray(c["rlf"], float)
        used_s = np.asarray(c["prb_used"], float).sum(1) / ticks
        ue_s = np.bincount(serv, minlength=n)
        return pv_s, v_s, rlf_s, used_s, ue_s

    def _grow(self) -> None:
        self.ser = np.concatenate([self.ser, np.zeros_like(self.ser)])
        self.ser_scored = np.concatenate([self.ser_scored, np.zeros_like(self.ser_scored)])

    def __call__(self) -> Mapping[str, Any]:
        """Takes the plant's counters, accumulates the tap and returns the counters."""
        p = self.plant
        c = type(p).take_counters(p)
        scored = bool(getattr(p, "sec_scored", False))
        pv_s, v_s, rlf_s, used_s, ue_s = self._second(c)
        if scored or self.count_all:
            self.pv += pv_s
            self.v += v_s
            self.e += c["energy_j"]
            self.rlf += rlf_s
            self.load += ue_s
            self.prb += used_s
        if self.n_sec >= len(self.ser):
            self._grow()
        ticks = max(int(c["ticks"]), 1)
        row = self.ser[self.n_sec]
        row[0], row[1], row[2], row[3] = pv_s, v_s, c["energy_j"], rlf_s
        row[4], row[5] = used_s, np.asarray(c["prb_cap"], float) / ticks
        row[6] = ue_s
        row[7] = np.bincount(p.serv[p.prot], minlength=p.nc) if p.prot.any() else 0.0
        self.ser_scored[self.n_sec] = scored
        self.n_sec += 1
        return c

    def sums(self, cells: Any) -> tuple[float, float, float]:
        """Returns the cumulative (pv, e, v) summed over `cells`."""
        idx = list(cells)
        return float(self.pv[idx].sum()), float(self.e[idx].sum()), float(self.v[idx].sum())

    def arrays(self, ext: bool = False) -> tuple[np.ndarray, ...]:
        """Returns copies of the cumulative (pv, e, v) arrays, plus (rlf, load, prb) if `ext`."""
        base = (self.pv.copy(), self.e.copy(), self.v.copy())
        return (*base, self.rlf.copy(), self.load.copy(), self.prb.copy()) if ext else base

    def series(self) -> dict[str, Any]:
        """Returns `{"fields", "t", "scored", "data" (T, fields, cells) float32}`."""
        n = self.n_sec
        return {
            "fields": list(SERIES_FIELDS),
            "t": np.arange(1, n + 1),
            "scored": self.ser_scored[:n].copy(),
            "data": self.ser[:n].copy(),
        }


def install_tap(env: E6Env, count_all: bool = False) -> CellKPITap:
    """Installs a tap on the env's plant, or returns the installed one."""
    tap = env.plant.__dict__.get("take_counters")
    if isinstance(tap, CellKPITap):
        return tap
    tap = CellKPITap(env.plant, count_all=count_all)
    env.plant.__dict__["take_counters"] = tap
    return tap


def get_tap(env: E6Env) -> CellKPITap:
    """Returns the installed tap.

    Raises:
        RuntimeError: No tap is installed.
    """
    tap = env.plant.__dict__.get("take_counters")
    if not isinstance(tap, CellKPITap):
        raise RuntimeError("no CellKPITap installed on this env (install_tap)")
    return tap


def _close_kpi(tap: CellKPITap, u: Unit, t_end: float, trunc: bool) -> None:
    pv, e, v = tap.sums(u["exp"])
    pv_c, e_c, v_c = tap.sums([u["c"]])
    o = u.pop("_open")
    u["kpi"] = {
        "pv": pv - o[0],
        "e": e - o[1],
        "v": v - o[2],
        "pv_own": pv_c - o[3],
        "e_own": e_c - o[4],
        "v_own": v_c - o[5],
        "secs": float(t_end - u["t0"]),
        "trunc": bool(trunc),
    }


Labeller = Callable[[E6Env, Mapping[str, Any], UnitArbiter | None, list[Unit]], None]


@dataclasses.dataclass(frozen=True)
class Collection:
    """Settings of one collection episode.

    Attributes:
        T: Unit hold time in seconds.
        open_rule: Unit open rule of the arbiter.
        arb_warmup_s: The arbiter's all-accept, no-unit period; None = the config's warm-up.
        count_all: Tap flag: cumulative arrays count warm-up seconds.
        churn_cap: Env churn cap of a new env.
        arbiter_cls: `UnitArbiter` subclass with the same constructor.
    """

    T: float = T_UNIT
    open_rule: str = "first"
    arb_warmup_s: float | None = None
    count_all: bool = False
    churn_cap: Any = None
    arbiter_cls: type[UnitArbiter] = UnitArbiter


COLLECTION = Collection()


def run_collection(
    cfg: Any,
    policy: Policy,
    options: Collection = COLLECTION,
    labeller: Labeller | None = None,
    env: E6Env | None = None,
) -> dict[str, Any]:
    """Runs one collection episode under `wg3=True` with a unit arbiter and the KPI tap.

    Args:
        cfg: Plant config.
        policy: Logging policy.
        options: Collection settings.
        labeller: Called as `labeller(env, obs, snap, opened)` every second in which units
            opened, after the arbiter decided and before `env.step_apply`; `snap` is an arbiter
            fork taken before this second's call.
        env: Env to run; a new `E6Env(cfg, log=False, wg3=True)` if None.

    Returns:
        `{"env", "arb", "tap", "units", "cpu_s"}`; each unit carries its window KPIs `kpi`.
    """
    t_cpu = time.process_time()
    env = env if env is not None else E6Env(cfg, log=False, wg3=True, churn_cap=options.churn_cap)
    tap = install_tap(env, count_all=options.count_all)
    wu = float(cfg.warmup_s) if options.arb_warmup_s is None else float(options.arb_warmup_s)
    arb = options.arbiter_cls(policy, T=options.T, warmup_s=wu, open_rule=options.open_rule)
    T = options.T
    live: list[Unit] = []
    while env.sec < env.total_s:
        obs = env.step_propose()
        now = obs["t"]
        still = []
        for u in live:
            if now >= u["t0"] + T:
                _close_kpi(tap, u, u["t0"] + T, False)
            else:
                still.append(u)
        live = still
        want_snap = labeller is not None and obs["requests"] and now >= arb.warmup_s
        snap = arb.fork(None) if want_snap else None
        dec = arb(obs)
        for u in arb.opened:
            u["_open"] = tap.sums(u["exp"]) + tap.sums([u["c"]])
            live.append(u)
        if labeller is not None and arb.opened:
            labeller(env, obs, snap, list(arb.opened))
        env.step_apply(dec)
    for u in live:
        _close_kpi(tap, u, float(env.sec), True)
    finalize_units(arb.units)
    return {
        "env": env,
        "arb": arb,
        "tap": tap,
        "units": arb.units,
        "cpu_s": time.process_time() - t_cpu,
    }

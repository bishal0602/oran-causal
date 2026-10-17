"""Episode loop of the system-level simulator and the common arbiter interface.

Per control second, after the plant's 25 ticks:

1. KPM: close the second's counters, emit reports whose granularity ends now, deliver reports
   whose delay elapsed;
2. xApps observe delivered reports; due xApps propose requests, reading the applied
   configuration;
3. the arbiter sees `obs` and returns decisions (no arbiter: accept everything, last writer
   wins);
4. the RIC applies accepted or modified values through the hard actuator limits; xApps get ACK or
   NACK.

`step_propose` runs 1-2 and parks the requests on the env; `step_apply(dec)` runs 4. A copy taken
between the two phases carries the pending requests, so `env.copy().step_apply(dec)` replays
exactly what `env.step_apply(dec)` does.

Arbiter contract: `arbiter(obs) -> {"decisions": [...], "writes": [(knob, value), ...],
"rollback": [knob, ...]}`. `decisions[i]` answers `obs["requests"][i]` with an O-RAN WG3
conflict-mitigation action: `"accept"`, `"reject"`, `("modify", value)`, `"defer"` (re-offered
next second, rejected after 10 s) or `("lock", secs)` (reject and lock the knob while
`t < t_lock + secs`). Requests on a locked knob still appear in `obs` but are force-rejected.
`rollback` restores each knob to the value in force before its most recent change, before the
decisions and through the same actuator limits. `writes` are arbiter-originated settings, limited
by the actuator limits and `write_budget` per scored hour; with `wg3=True` a non-empty list
raises `ValueError`.

Churn: every applied knob change counts in `stats["changes"]` (warm-up included). With
`churn_cap` set, once the cap is reached every further change is refused: accept and modify NACK
(counted as rejections and `churn_blocked`), rollbacks and writes are skipped. No-op accepts are
not changes.

`obs = {"t", "new_reports", "config", "requests", "locked", "changes", "churn_cap", "static"}`;
arbiters get only `obs`.
"""

import copy
from collections.abc import Callable
from typing import Any

from oran_causal.sim.system import config as C
from oran_causal.sim.system.ric import KPM, Knob, feasible, knob_get, knob_set
from oran_causal.sim.system.sim import Plant, tape_rng
from oran_causal.sim.system.trace import Trace, TraceRecorder
from oran_causal.sim.system.xapps import ES, MIXES, Request, XApp
from oran_causal.sim.system.xapps_p import ESPico, build_p_xapps

Obs = dict[str, Any]
Decision = dict[str, Any]
Arbiter = Callable[[Obs], Decision]
MAX_DEFER_S = 10
STAT_KEYS = (
    "req",
    "acc",
    "rej",
    "mod",
    "def",
    "writes",
    "changes",
    "churn_blocked",
    "lock_blocked",
    "locks",
    "rollbacks",
)
_RB_OUT = {"ok": "applied", "nack": "actuator"}
_WR_OUT = {"nack": "actuator"}


class E6Env:
    """One episode of the plant, its RIC, KPM and xApps.

    Args:
        cfg: episode configuration.
        write_budget: arbiter writes allowed per scored hour.
        log: keep a per-second log of configuration, reports, requests and decisions.
        wg3: refuse free-form arbiter writes (WG3 actions only).
        churn_cap: maximum number of applied knob changes, `None` for no cap.
        trace: attach a `TraceRecorder`; `get_trace()` returns the episode trace.
        trace_snapshot_s: seconds between full configuration snapshots in the trace.
    """

    def __init__(
        self,
        cfg: C.E6Config,
        *,
        write_budget: float = 300.0,
        log: bool = True,
        wg3: bool = False,
        churn_cap: int | None = None,
        trace: bool = False,
        trace_snapshot_s: int = 60,
    ) -> None:
        self.cfg = cfg
        self.plant = Plant(cfg)
        self.kpm = KPM(cfg, self.plant)
        self.xapps: list[XApp] = [cls(self, i) for i, cls in enumerate(MIXES[cfg.mix])]
        self.last_change: dict[Knob, float] = {}
        self.prev_val: dict[Knob, float] = {}
        self.plant.oam_hook = self._on_oam
        self.lock_until: dict[Knob, float] = {}
        self.deferred: list[dict[str, Any]] = []
        self.write_budget = write_budget
        self.writes_used = 0
        self.wg3, self.churn_cap = wg3, churn_cap
        self.log_on = log
        self.log: list[dict[str, Any]] = []
        self.knobs = self._base_knobs()
        r = tape_rng(cfg.seed, "xapp", 999)
        total_s = cfg.warmup_s + cfg.scored_s
        update = cfg.update and self.xapps
        self.update_at = (cfg.warmup_s + r.uniform(0.15, 0.6) * cfg.scored_s) if update else None
        self.update_xapp = int(r.integers(max(len(self.xapps), 1)))
        self._extend_e6p()
        self.total_s = int(total_s)
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        self.sec = 0
        self._static: dict[str, Any] | None = None
        self._pending: tuple[float, list, list[Request]] | None = None
        self._tr = TraceRecorder(self, trace_snapshot_s) if trace else None

    def _base_knobs(self) -> list[Knob]:
        lay = self.plant.lay
        nc = lay.n_cells
        knobs: list[Knob] = [("cio", s, n) for s in range(nc) for n in lay.neighbours[s]]
        knobs += [("hys", c) for c in range(nc)] + [("ttt", c) for c in range(nc)]
        if self.cfg.mix == "M4":
            knobs += [("ll_ratio", c) for c in range(nc)]
        knobs += [("sleep", c) for c in range(nc) if not lay.is_macro[c]]
        knobs += [("carrier", c) for c in range(nc) if lay.is_macro[c]]
        fwd = {(s, n) for s in range(nc) for n in lay.neighbours[s]}
        knobs += [("cio", n, s) for s in range(nc) for n in lay.neighbours[s] if (n, s) not in fwd]
        return knobs

    def _extend_e6p(self) -> None:
        P, lay = self.cfg.e6p, self.plant.lay
        if P.ptx_on:
            self.knobs += [("ptx", c) for c in range(lay.n_cells) if self.plant.ptx_cells[c]]
        if P.prot_on:
            self.knobs += [("prot_min", c) for c in range(lay.n_cells)]
        if P.enabled and P.es_pico:
            self.xapps = [ESPico(self, i) if type(x) is ES else x for i, x in enumerate(self.xapps)]
        if P.xapps:
            self.xapps += build_p_xapps(self, P.xapps, start_idx=len(self.xapps))

    def static(self) -> dict[str, Any]:
        """Static network description handed to arbiters in `obs["static"]`."""
        lay = self.plant.lay
        return {
            "cells": lay.n_cells,
            "neighbours": lay.neighbours,
            "is_macro": lay.is_macro.copy(),
            "knobs": list(self.knobs),
            "limits": "see ric.LIMITS",
            "xapps": {x.name: sorted({k[0] for k in self.knobs}) for x in self.xapps},
        }

    def config(self) -> dict[Knob, float]:
        """Applied value of every registered knob."""
        return {k: knob_get(self.plant, k) for k in self.knobs}

    def apply(self, k: Knob, v: float, now: float) -> tuple[str, float]:
        """Apply a value through the actuator limits and the churn cap.

        Returns:
            `("ok", value)` when applied or already in force, `("nack", current)` when the
            actuator limits refuse it, `("churn", current)` when the churn cap is reached.
        """
        v2, ok = feasible(self.plant, k, v, now, self.last_change)
        cur = knob_get(self.plant, k)
        if not ok:
            return "nack", cur
        if abs(v2 - cur) > 1e-9:
            if self.churn_cap is not None and self.stats["changes"] >= self.churn_cap:
                self.stats["churn_blocked"] += 1
                return "churn", cur
            knob_set(self.plant, k, v2, now)
            self.prev_val[k] = cur
            self.last_change[k] = now
            self.stats["changes"] += 1
        return "ok", v2

    def _on_oam(self, k: Knob, v: float, now: float) -> None:
        self.prev_val[k] = float(v)

    def locked(self, k: Knob, now: float) -> bool:
        """Whether a WG3 lock on `k` is active at `now`."""
        return now < self.lock_until.get(k, -1e9)

    def run(self, arbiter: Arbiter | None = None) -> dict[str, Any]:
        """Run the episode to its end and return `score()`."""
        while self.sec < self.total_s:
            self.step(arbiter)
        return self.score()

    def copy(self, reseed: int | None = None) -> E6Env:
        """Independent copy for lookahead rollouts, sharing only the layout and gain maps.

        Taken between `step_propose` and `step_apply`, it carries the pending requests. Copies
        are untraced.

        Args:
            reseed: `None` continues the same exogenous tape (bit-identical replay). An integer
                keeps the state at the copy point but re-draws every future exogenous draw (plant
                load, mobility, traffic and measurement noise per tick, KPM delay and drop per
                second) from an independent stream; the same integer gives the same future.
                Draws made at construction (layout, UEs, scenario, xApp variants) are kept.

        Returns:
            The copy.
        """
        memo: dict[int, Any] = {
            id(self.plant.gm): self.plant.gm,
            id(self.plant.lay): self.plant.lay,
        }
        if self._tr is not None:
            memo[id(self._tr)] = None
        c = copy.deepcopy(self, memo)
        if reseed is not None:
            c.plant.rng_salt = int(reseed)
        return c

    def get_trace(self) -> Trace:
        """The episode trace.

        Raises:
            RuntimeError: if the env was built with `trace=False`.
        """
        if self._tr is None:
            raise RuntimeError("E6Env was built with trace=False")
        return self._tr.finish()

    @property
    def pending_requests(self) -> list[Request]:
        """Requests parked by `step_propose`, empty when no second is pending."""
        return [] if self._pending is None else self._pending[2]

    def step(self, arbiter: Arbiter | None = None) -> None:
        """Advance one control second: plant ticks, KPM, xApps, arbiter, RIC apply."""
        obs = self._propose(arbiter is not None)
        if arbiter is None or obs is None:
            self.step_apply({"decisions": ["accept"] * len(self.pending_requests), "writes": []})
        else:
            self.step_apply(arbiter(obs))

    def step_propose(self) -> Obs:
        """Phase 1: ticks, KPM, the hidden xApp update, xApp proposals.

        Returns:
            The arbiter observation; the requests stay pending until `step_apply`.
        """
        obs = self._propose(True)
        if obs is None:
            raise RuntimeError("step_propose built no observation")
        return obs

    def _propose(self, build_obs: bool) -> Obs | None:
        if self._pending is not None:
            raise RuntimeError("step_propose called twice without step_apply")
        p = self.plant
        if self._static is None:
            self._static = self.static()
        for _ in range(C.TICKS_PER_CONTROL):
            p.tick()
        if self._tr is not None:
            self._tr.plant_second()
        self.sec += 1
        now = float(self.sec)
        self.kpm.second(self.sec)
        new = self.kpm.deliver(now)
        if self.update_at is not None and now >= self.update_at:
            self.xapps[self.update_xapp].update_version()
            self.update_at = None
        reqs = [d["req"] for d in self.deferred]
        for x in self.xapps:
            x.observe(new)
            if x.due(now):
                reqs.extend(x.propose(now))
        self._pending = (now, new, reqs)
        if not build_obs:
            return None
        return {
            "t": now,
            "new_reports": new,
            "config": self.config(),
            "requests": reqs,
            "static": self._static,
            "locked": {k: u for k, u in self.lock_until.items() if now < u},
            "changes": self.stats["changes"],
            "churn_cap": self.churn_cap,
        }

    def step_apply(self, dec: Decision) -> None:
        """Phase 2: rollbacks, per-request decisions with ACK/NACK, writes, log.

        Raises:
            RuntimeError: without a pending `step_propose`.
            ValueError: for free-form writes under `wg3=True`.
        """
        if self._pending is None:
            raise RuntimeError("step_apply called without a pending step_propose")
        if self.wg3 and dec.get("writes"):
            raise ValueError(
                "wg3=True: free-form arbiter writes are not a WG3 action (use decisions/rollback)"
            )
        now, new, reqs = self._pending
        self._pending = None
        if self._tr is not None:
            self._tr.begin(now, new, reqs)
        for k in dec.get("rollback", ()):
            self._rollback(k, now)
        old_def = {id(d["req"]): d["age"] for d in self.deferred}
        self.deferred = []
        for r, d in zip(reqs, dec["decisions"], strict=True):
            self._decide(r, d, now, old_def)
        for k, v in dec.get("writes", []):
            if self.writes_used >= self.write_budget * max(self.cfg.scored_s, 1) / 3600 + 1e-9:
                break
            self._write(k, v, now)
        if self.log_on:
            self.log.append(
                {
                    "t": now,
                    "config": self.config(),
                    "reports": new,
                    "n_req": len(reqs),
                    "requests": reqs,
                    "decisions": list(dec["decisions"]),
                    "rollback": list(dec.get("rollback", ())),
                }
            )
        if self._tr is not None:
            self._tr.end(now)

    def _rollback(self, k: Knob, now: float) -> None:
        st = "noop"
        if k in self.prev_val and abs(self.prev_val[k] - knob_get(self.plant, k)) > 1e-9:
            st = self.apply(k, self.prev_val[k], now)[0]
            if st == "ok":
                self.stats["rollbacks"] += 1
        if self._tr is not None:
            self._tr.rollback(now, k, _RB_OUT.get(st, st))

    def _write(self, k: Knob, v: float, now: float) -> None:
        st = self.apply(k, v, now)[0]
        if st == "ok":
            self.writes_used += 1
            self.stats["writes"] += 1
        if self._tr is not None:
            self._tr.write(now, k, v, _WR_OUT.get(st, st))

    def _decide(self, r: Request, d: Any, now: float, old_def: dict[int, int]) -> None:
        self.stats["req"] += 1
        x = next(a for a in self.xapps if a.name == r["xapp"])
        tr = self._tr
        d0, why = d, "reject"
        if self.locked(r["knob"], now):
            d, why = "reject", "locked"
            self.stats["lock_blocked"] += 1
        elif isinstance(d, tuple) and d[0] == "lock":
            self.lock_until[r["knob"]] = now + float(d[1])
            self.stats["locks"] += 1
            d, why = "reject", "lock_set"
            if tr is not None:
                tr.lock(now, r["knob"], self.lock_until[r["knob"]])
        if d == "defer":
            age = old_def.get(id(r), 0) + 1
            if age <= MAX_DEFER_S:
                self.deferred.append({"req": r, "age": age})
                self.stats["def"] += 1
                if tr is not None:
                    tr.request(now, r, d0, "deferred", knob_get(self.plant, r["knob"]), age - 1)
                return
            d, why = "reject", "expired"
        if d == "reject":
            self.stats["rej"] += 1
            x.result(r, False, knob_get(self.plant, r["knob"]), now)
            if tr is not None:
                cur = knob_get(self.plant, r["knob"])
                tr.request(now, r, d0, why, cur, old_def.get(id(r), 0))
            return
        n_ch = self.stats["changes"]
        st, applied = self._accept(x, r, d, now)
        if tr is not None:
            ok_why = "ok" if self.stats["changes"] > n_ch else "noop"
            why = ok_why if st == "ok" else ("churn" if st == "churn" else "actuator")
            tr.request(now, r, d0, why, applied, old_def.get(id(r), 0))

    def _accept(self, x: XApp, r: Request, d: Any, now: float) -> tuple[str, float]:
        v = r["prop"] if d == "accept" else float(d[1])
        st, applied = self.apply(r["knob"], v, now)
        ok = st == "ok"
        self.stats["acc" if d == "accept" else "mod"] += int(ok)
        self.stats["rej"] += int(st == "churn")
        x.result(r, ok, applied, now)
        return st, applied

    def score(self) -> dict[str, Any]:
        """Episode score: SLA rates over the scored window plus the RIC statistics."""
        S = self.plant.sla
        ue_h = S["ue_s"] / 3600.0
        out = {
            "svr": S["viol_ue_s"] / max(ue_h, 1e-9),
            "viol_frac": S["viol_ue_s"] / max(S["ue_s"], 1),
            "ll_viol": S["ll_viol"],
            "embb_viol": S["embb_viol"],
            "outage_viol": S["outage_viol"],
            "severe": S["severe"],
            "energy_kwh": S["energy_j"] / 3.6e6,
            "rlf_per_ue_h": S["rlf"] / max(ue_h, 1e-9),
            "ho_per_ue_h": S["ho"] / max(ue_h, 1e-9),
            "pingpong": S["pingpong"],
        }
        if "prot_viol" in S:
            out["prot_viol"] = S["prot_viol"]
            out["prot_viol_frac"] = S["prot_viol"] / max(S["prot_ue_s"], 1)
        return {**out, **self.stats}


def run_episode(
    cfg: C.E6Config, arbiter: Arbiter | None = None, **kw: Any
) -> tuple[dict[str, Any], E6Env]:
    """Build an env with `kw`, run it with `arbiter`, and return its score and the env."""
    env = E6Env(cfg, **kw)
    return env.run(arbiter), env

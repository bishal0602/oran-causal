"""Rule-based SON xApps (TS 32.522 / TS 36.902 style) of the system-level plant.

The xApps act only on delivered KPM reports and the applied configuration. Each draws an
implementation variant per seed: thresholds scaled within +-25 % (and by
`cfg.xapp_threshold_scale`), a cadence phase, and a blocked-action behaviour: `retry` re-proposes
the same value next cycle, `escalate` keeps integrating its own target so a later acceptance
jumps, `hold` skips one cycle. All variants read the applied configuration before proposing.

A request is `{"xapp", "ver", "knob", "cur", "prop", "t"}`.
"""

import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from oran_causal.sim.system import config as C
from oran_causal.sim.system.ric import Knob, knob_get
from oran_causal.sim.system.sim import tape_rng

if TYPE_CHECKING:
    from oran_causal.sim.system.env import E6Env

BEHAVIOURS = ("retry", "escalate", "hold")
Request = dict[str, Any]
Report = dict[str, Any]


def _kept(out: list[Request | None]) -> list[Request]:
    return [r for r in out if r]


class XApp:
    """Base xApp: variant draw, report intake, cadence and NACK handling.

    Attributes:
        name: role name used in requests and arbiter rules.
        cadence: seconds between proposals.
    """

    name = "base"
    cadence = 10.0

    def __init__(self, env: E6Env, idx: int) -> None:
        self.env, self.p = env, env.plant
        r = tape_rng(env.cfg.seed, "xapp", 1000 + idx)
        self.scale = r.uniform(0.75, 1.25) * env.cfg.xapp_threshold_scale
        self.behaviour = BEHAVIOURS[int(r.integers(3))]
        self.next_t = float(r.uniform(0, self.cadence))
        self.ver = 1
        self.pending_target: dict[Knob, float] = {}
        self.hold_until = -1.0
        self.reports: dict[str, Report] = {}

    def observe(self, reps: list[Report]) -> None:
        """Take in newly delivered KPM reports."""
        for rep in reps:
            self.reports[rep["gran"]] = rep

    def due(self, now: float) -> bool:
        """Whether the xApp proposes this second; advances the cadence clock when it is time."""
        if now + 1e-9 >= self.next_t:
            self.next_t += self.cadence
            return now >= self.hold_until
        return False

    def req(self, knob: Knob, prop: float, now: float) -> Request | None:
        """Build a request for `knob`, or `None` when it would not change the knob."""
        cur = knob_get(self.p, knob)
        if self.behaviour == "escalate" and knob in self.pending_target:
            prop = self.pending_target[knob] + (prop - cur)
        if abs(prop - cur) < 1e-9:
            return None
        return {
            "xapp": self.name,
            "ver": self.ver,
            "knob": knob,
            "cur": cur,
            "prop": float(prop),
            "t": now,
        }

    def result(self, r: Request, accepted: bool, applied: float, now: float) -> None:
        """ACK or NACK of request `r`."""
        k = r["knob"]
        if accepted:
            self.pending_target.pop(k, None)
        elif self.behaviour == "escalate":
            self.pending_target[k] = r["prop"]
        elif self.behaviour == "hold":
            self.hold_until = now + self.cadence

    def propose(self, now: float) -> list[Request]:
        """Requests of this cycle."""
        return []

    def update_version(self) -> None:
        """Apply the hidden version update."""
        self.ver += 1


class MRO(XApp):
    """Mobility robustness optimisation on CIO, TTT and Hys from the mobility reports."""

    name, cadence = "MRO", 30.0
    WINDOW_KEYS: ClassVar[tuple[str, ...]] = (
        "ho_att",
        "too_late",
        "too_early",
        "pingpong",
        "wrong_cell",
    )

    def __init__(self, env: E6Env, idx: int) -> None:
        super().__init__(env, idx)
        self.tl_thr, self.te_thr, self.step = 0.02 * self.scale, 0.10 * self.scale, 1.0
        self.mob_hist: list[Report] = []

    def observe(self, reps: list[Report]) -> None:
        """Take in reports and keep the last two mobility reports (a 60 s window)."""
        super().observe(reps)
        for rep in reps:
            if rep["gran"] == "mob":
                self.mob_hist = [*self.mob_hist, rep][-2:]

    def propose(self, now: float) -> list[Request]:
        """CIO up (or TTT down) on too-late HOs, CIO down and Hys up on too-early HOs."""
        if not self.mob_hist:
            return []
        rep = {k: sum(h[k] for h in self.mob_hist) for k in self.WINDOW_KEYS}
        out: list[Request | None] = []
        for s in range(self.p.nc):
            hys_up = False
            for n in self.p.lay.neighbours[s]:
                hys_up |= self._pair(rep, s, n, now, out)
            if hys_up and self.p.hys[s] < C.HYS_RANGE[1]:
                out.append(self.req(("hys", s), self.p.hys[s] + 0.5, now))
        return _kept(out)

    def _pair(self, rep: Report, s: int, n: int, now: float, out: list[Request | None]) -> bool:
        att = rep["ho_att"][s, n]
        tl = rep["too_late"][s, n]
        te = rep["too_early"][s, n] + rep["pingpong"][s, n] + rep["wrong_cell"][s, n]
        den = att + tl
        if den < 3:
            return False
        cio = self.p.cio[s, n]
        if tl >= 2 and tl / den > self.tl_thr:
            if cio < C.CIO_RANGE[1]:
                out.append(self.req(("cio", s, n), cio + self.step, now))
            else:
                i = C.TTT_SET_MS.index(int(self.p.ttt[s]))
                if i > 0:
                    out.append(self.req(("ttt", s), C.TTT_SET_MS[i - 1], now))
        elif te >= 3 and te / max(att, 1) > self.te_thr and tl == 0:
            out.append(self.req(("cio", s, n), cio - self.step, now))
            return True
        return False

    def update_version(self) -> None:
        """Version 2 doubles the CIO step."""
        super().update_version()
        self.step = 2.0


class TS(XApp):
    """Traffic steering: cell range extension on CIO toward less-loaded neighbours."""

    name, cadence = "TS", 10.0

    def __init__(self, env: E6Env, idx: int) -> None:
        super().__init__(env, idx)
        self.u_hi, self.gap = 0.6 * self.scale, 0.15 * self.scale
        self.u_tgt, self.cio_max = 0.5 * self.scale, 4.0
        self.u: np.ndarray | None = None

    def propose(self, now: float) -> list[Request]:
        """Offload from busy cells whose eMBB users suffer; relax CIO between balanced cells."""
        rep = self.reports.get("fast")
        if rep is None:
            return []
        u = rep["prb_util"]
        ewma = u if self.u is None else 0.7 * self.u + 0.3 * u
        self.u = ewma
        thp = self.reports.get("thp")
        p5 = thp["embb_thp_p5"] if thp is not None else np.full(self.p.nc, np.nan)
        suffer = np.nan_to_num(p5, nan=np.inf) < C.EMBB_THP_TARGET_BPS * 1.2
        out: list[Request | None] = []
        for s in range(self.p.nc):
            if self.p.asleep[s]:
                continue
            for n in self.p.lay.neighbours[s]:
                if not self.p.asleep[n]:
                    out.extend(self._pair(ewma, bool(suffer[s]), (s, n), now))
        return _kept(out)

    def _pair(
        self, u: np.ndarray, suffer: bool, pair: tuple[int, int], now: float
    ) -> list[Request | None]:
        s, n = pair
        out: list[Request | None] = []
        cio = self.p.cio[s, n]
        d = u[s] - u[n]
        busy = u[s] > self.u_hi and u[n] < self.u_tgt and d > self.gap
        if suffer and busy and cio < self.cio_max:
            out.append(self.req(("cio", s, n), cio + 1, now))
            if self.p.cio[n, s] > C.CIO_RANGE[0]:
                out.append(self.req(("cio", n, s), self.p.cio[n, s] - 1, now))
        elif abs(d) < 0.05 and cio != 0:
            out.append(self.req(("cio", s, n), cio - np.sign(cio), now))
        return out

    def update_version(self) -> None:
        """Version 2 narrows the load gap."""
        super().update_version()
        self.gap = 0.08 * self.scale


class ES(XApp):
    """Energy saving by macro capacity-carrier shutdown; the coverage carrier stays on.

    A carrier is switched off when the sector's full-band load stays low for 60 s and its macro
    neighbours have headroom, and back on when the remaining carrier gets busy. Reported
    utilisation is relative to active capacity, so shutting a carrier doubles it.
    """

    name, cadence = "ES", 10.0

    def __init__(self, env: E6Env, idx: int) -> None:
        super().__init__(env, idx)
        lay = self.p.lay
        self.macros = [c for c in range(lay.n_cells) if lay.is_macro[c]]
        self.nbr = {c: [n for n in lay.neighbours[c] if lay.is_macro[n]] for c in self.macros}
        self.low_since: dict[int, float | None] = dict.fromkeys(self.macros)
        self.u_off, self.u_mac, self.u_on = 0.30 * self.scale, 0.70 * self.scale, 0.80 * self.scale
        self.u: np.ndarray | None = None

    def propose(self, now: float) -> list[Request]:
        """Carrier off or on per macro sector from a ~20 s EWMA of the utilisation."""
        rep = self.reports.get("fast")
        if rep is None:
            return []
        self.u = rep["prb_util"] if self.u is None else 0.6 * self.u + 0.4 * rep["prb_util"]
        full = self.u * self.p.n_car / self.p.n_trx
        out: list[Request | None] = []
        for c in self.macros:
            ncar, ntrx = self.p.n_car[c], self.p.n_trx[c]
            if ncar < ntrx:
                self.low_since[c] = None
                if self.u[c] > self.u_on:
                    out.append(self.req(("carrier", c), float(ntrx), now))
                continue
            nb = float(np.mean(full[self.nbr[c]])) if self.nbr[c] else 0.0
            if full[c] < self.u_off and nb < self.u_mac:
                since = self.low_since[c] or now
                self.low_since[c] = since
                if now - since >= 60:
                    out.append(self.req(("carrier", c), float(ncar - 1), now))
            else:
                self.low_since[c] = None
        return _kept(out)

    def update_version(self) -> None:
        """Version 2 raises the switch-on threshold."""
        super().update_version()
        self.u_on = 0.90 * self.scale


class SliceSLA(XApp):
    """LL slice SLA assurance on the dedicated PRB ratio from the last 5 s of fast reports."""

    name, cadence = "SLICE", 1.0

    def __init__(self, env: E6Env, idx: int) -> None:
        super().__init__(env, idx)
        self.hi, self.lo = 0.8 * self.scale, 0.4 * self.scale
        self.good_since: dict[int, float] = {}
        self.fast_hist: list[Report] = []

    def observe(self, reps: list[Report]) -> None:
        """Take in reports and keep the last five fast reports."""
        super().observe(reps)
        for rep in reps:
            if rep["gran"] == "fast":
                self.fast_hist = [*self.fast_hist, rep][-5:]

    def propose(self, now: float) -> list[Request]:
        """Raise `ll_ratio` on high LL delay, return idle or unneeded dedicated PRBs."""
        hist = self.fast_hist
        if not hist:
            return []
        n_s = sum(h["ll_samples"] for h in hist)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d_all = np.nanmax(np.array([h["ll_delay_p95"] for h in hist]), 0)
        out: list[Request | None] = []
        for c in range(self.p.nc):
            d = d_all[c]
            if np.isnan(d) or n_s[c] < 5:
                continue
            out.append(self._cell(hist, c, d, now))
        return _kept(out)

    def _cell(self, hist: list[Report], c: int, d: float, now: float) -> Request | None:
        tgt = C.LL_DELAY_TARGET_S
        r = self.p.ll_ratio[c]
        idle = float(np.mean([h["prb_rsv_idle"][c] for h in hist])) / r if r > 0 else 0.0
        if idle > 0.5 and r > 0:
            self.good_since.pop(c, None)
            return self.req(("ll_ratio", c), r - 0.05, now)
        if d > self.hi * tgt and r < 0.5 and idle < 0.2:
            self.good_since.pop(c, None)
            return self.req(("ll_ratio", c), r + 0.05, now)
        if d < self.lo * tgt and r > 0:
            self.good_since.setdefault(c, now)
            if now - self.good_since[c] >= 10:
                self.good_since[c] = now
                return self.req(("ll_ratio", c), r - 0.05, now)
            return None
        self.good_since.pop(c, None)
        return None

    def update_version(self) -> None:
        """Version 2 lowers the delay trigger."""
        super().update_version()
        self.hi = 0.6 * self.scale


MIXES: dict[str, tuple[type[XApp], ...]] = {
    "none": (),
    "MRO": (MRO,),
    "TS": (TS,),
    "ES": (ES,),
    "SLICE": (SliceSLA,),
    "M_TS_MRO": (MRO, TS),
    "M2": (MRO, TS, ES),
    "M4": (MRO, TS, ES, SliceSLA),
}

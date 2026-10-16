"""xApps of the power and protected-slice extension (`cfg.e6p`, off by default).

Same interface as the base xApps (`XApp`): they act only on delivered KPM reports (the extension
fields of the fast report) and on the applied configuration, and inherit the per-seed
implementation variant. The env appends them after the base mix, so the base xApps keep their
indices and RNG draws. The threshold scale multiplies only the dimensionless thresholds (load and
violation shares), not the dB thresholds.

- `ESPico` (role name "ES"): the base `ES` for macro carriers plus pico sleep (TS 28.541
  DESManagementFunction roles). A pico sleeps when its load is below `u_off` and its candidate
  macro (strongest gain at the pico site) has full-band load below `u_mac` for 60 s; it wakes when
  the candidate's active-capacity utilisation exceeds `u_on`.
- `PowerES`: lowers the Tx-power offset of a lightly loaded macro by `pes_step_db` and restores it
  toward nominal when the active-capacity utilisation exceeds `pes_u_high`; never above 0 dB.
- `Coverage`: raises the Tx-power offset where the cell's edge-SINR percentile is weak both in
  absolute terms and relative to its neighbours (CCO-style), or where its protected floors fail;
  releases a positive offset once the edge is comfortable.
- `SliceGuarantee`: raises the protected minimum PRB share where protected floors are violated and
  lowers it after a slack spell.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np

from oran_causal.sim.system.ric import knob_get
from oran_causal.sim.system.xapps import ES, Report, Request, XApp

if TYPE_CHECKING:
    from oran_causal.sim.system.env import E6Env


def _nanmean(stack: list[np.ndarray]) -> np.ndarray:
    a = np.array(stack, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, 0)


def _kept(out: list[Request | None]) -> list[Request]:
    return [r for r in out if r]


class _PXApp(XApp):
    def __init__(self, env: E6Env, idx: int) -> None:
        super().__init__(env, idx)
        self.P = env.cfg.e6p
        self.hist: list[Report] = []

    def observe(self, reps: list[Report]) -> None:
        super().observe(reps)
        for rep in reps:
            if rep["gran"] == "fast":
                self.hist = [*self.hist, rep][-int(max(self.cadence, 1)) :]


class ESPico(ES):
    """Base `ES` for macro carriers, unchanged, plus pico sleep and wake."""

    name = "ES"

    def __init__(self, env: E6Env, idx: int) -> None:
        super().__init__(env, idx)
        lay = self.p.lay
        self.picos = [c for c in range(lay.n_cells) if not lay.is_macro[c]]
        g = self.p.gm.lookup(lay.cell_pos[self.picos])
        mac = np.nonzero(lay.is_macro)[0]
        self.cand = {pc: int(mac[np.argmax(g[i, mac])]) for i, pc in enumerate(self.picos)}
        self.pico_low_since: dict[int, float | None] = dict.fromkeys(self.picos)

    def propose(self, now: float) -> list[Request]:
        """Macro carrier requests of `ES`, then pico sleep and wake requests."""
        out = super().propose(now)
        u = self.u
        if u is None:
            return out
        full = u * self.p.n_car / self.p.n_trx
        for pc in self.picos:
            r = self._pico(pc, u, full, now)
            if r:
                out.append(r)
        return out

    def _pico(self, pc: int, u: np.ndarray, full: np.ndarray, now: float) -> Request | None:
        m = self.cand[pc]
        if self.p.asleep[pc]:
            self.pico_low_since[pc] = None
            return self.req(("sleep", pc), 0.0, now) if u[m] > self.u_on else None
        if self.p.waking_until[pc] > now:
            return None
        if full[pc] < self.u_off and full[m] < self.u_mac:
            since = self.pico_low_since[pc]
            since = since if since is not None else now
            self.pico_low_since[pc] = since
            return self.req(("sleep", pc), 1.0, now) if now - since >= 60 else None
        self.pico_low_since[pc] = None
        return None


class PowerES(_PXApp):
    """Tx-power energy saving on lightly loaded macros."""

    name = "PowerES"

    def __init__(self, env: E6Env, idx: int) -> None:
        self.cadence = env.cfg.e6p.pes_cadence_s
        super().__init__(env, idx)
        P = self.P
        self.u_low, self.u_high = P.pes_u_low * self.scale, P.pes_u_high * self.scale
        self.cells = [c for c in range(self.p.nc) if self.p.ptx_cells[c]]
        self.low_since: dict[int, float | None] = dict.fromkeys(self.cells)
        self.u: np.ndarray | None = None

    def propose(self, now: float) -> list[Request]:
        """Step the offset down after a light spell, back up toward 0 dB when busy."""
        rep = self.reports.get("fast")
        if rep is None:
            return []
        u = rep["prb_util"] if self.u is None else 0.6 * self.u + 0.4 * rep["prb_util"]
        self.u = u
        full = u * self.p.n_car / self.p.n_trx
        lo = self.P.ptx_range_db[0]
        out: list[Request | None] = []
        for c in self.cells:
            if self.p.asleep[c]:
                continue
            k = ("ptx", c)
            cur = knob_get(self.p, k)
            if full[c] < self.u_low:
                since = self.low_since[c]
                since = since if since is not None else now
                self.low_since[c] = since
                if now - since >= self.P.pes_hold_s and cur > lo + 1e-9:
                    out.append(self.req(k, max(lo, cur - self.P.pes_step_db), now))
            else:
                self.low_since[c] = None
                if u[c] > self.u_high and cur < -1e-9:
                    out.append(self.req(k, min(0.0, cur + self.P.pes_step_db), now))
        return _kept(out)


class Coverage(_PXApp):
    """Coverage and capacity optimisation on the Tx-power offset."""

    name = "Coverage"

    def __init__(self, env: E6Env, idx: int) -> None:
        self.cadence = env.cfg.e6p.cov_cadence_s
        super().__init__(env, idx)
        self.floor_frac = self.P.cov_floor_frac * self.scale
        self.cells = [c for c in range(self.p.nc) if self.p.ptx_cells[c]]

    def propose(self, now: float) -> list[Request]:
        """Raise power where the edge is weak or floors fail, else release a positive offset."""
        if not self.hist:
            return []
        edge = _nanmean([h["edge_sinr_p"] for h in self.hist])
        if "prot_below_frac" in self.hist[-1]:
            below = _nanmean([h["prot_below_frac"] for h in self.hist])
        else:
            below = np.full(self.p.nc, np.nan)
        out: list[Request | None] = []
        for c in self.cells:
            if self.p.asleep[c] or np.isnan(edge[c]):
                continue
            out.append(self._cell(c, edge, below, now))
        return _kept(out)

    def _cell(self, c: int, edge: np.ndarray, below: np.ndarray, now: float) -> Request | None:
        P = self.P
        hi = P.ptx_range_db[1]
        k = ("ptx", c)
        cur = knob_get(self.p, k)
        nb = [n for n in self.p.lay.neighbours[c] if not self.p.asleep[n] and not np.isnan(edge[n])]
        nb_edge = float(np.mean(edge[nb])) if nb else np.inf
        weak = edge[c] < P.cov_sinr_low_db and edge[c] < nb_edge - P.cov_margin_db
        fail = weak or (not np.isnan(below[c]) and below[c] > self.floor_frac)
        if fail and cur < hi - 1e-9:
            return self.req(k, min(hi, cur + P.cov_step_db), now)
        comfortable = edge[c] > P.cov_sinr_ok_db or edge[c] >= nb_edge
        if not fail and comfortable and cur > 1e-9:
            return self.req(k, max(0.0, cur - P.cov_step_db), now)
        return None


class SliceGuarantee(_PXApp):
    """Protected-slice guarantee on the work-conserving minimum PRB share."""

    name = "SliceGuarantee"

    def __init__(self, env: E6Env, idx: int) -> None:
        self.cadence = env.cfg.e6p.sg_cadence_s
        super().__init__(env, idx)
        P = self.P
        self.hi, self.lo = P.sg_viol_hi * self.scale, P.sg_viol_lo * self.scale
        self.good_since: dict[int, float] = {}

    def propose(self, now: float) -> list[Request]:
        """Raise `prot_min` on violated floors, lower it after a slack spell."""
        if not self.hist:
            return []
        below = _nanmean([h["prot_below_frac"] for h in self.hist])
        used = np.mean([h["prot_used_share"] for h in self.hist], 0)
        out: list[Request | None] = [
            self._cell(c, below[c], used[c], now) for c in range(self.p.nc)
        ]
        return _kept(out)

    def _cell(self, c: int, b: float, used: float, now: float) -> Request | None:
        P = self.P
        mlo, mhi = P.prot_min_range
        k = ("prot_min", c)
        cur = knob_get(self.p, k)
        if not np.isnan(b) and b > self.hi:
            self.good_since.pop(c, None)
            return self.req(k, min(mhi, cur + P.sg_step), now) if cur < mhi - 1e-9 else None
        if cur > mlo + 1e-9 and (np.isnan(b) or b < self.lo) and used < P.sg_slack * cur:
            self.good_since.setdefault(c, now)
            if now - self.good_since[c] >= P.sg_hold_s:
                self.good_since[c] = now
                return self.req(k, max(mlo, cur - P.sg_step), now)
            return None
        self.good_since.pop(c, None)
        return None


P_XAPPS: dict[str, type[_PXApp]] = {
    "PowerES": PowerES,
    "Coverage": Coverage,
    "SliceGuarantee": SliceGuarantee,
}
_NEEDS = {"PowerES": "ptx_on", "Coverage": "ptx_on", "SliceGuarantee": "prot_on"}


def build_p_xapps(env: E6Env, names: tuple[str, ...], start_idx: int) -> list[XApp]:
    """Instantiate the extension xApps named in `cfg.e6p.xapps`, indexed after the base mix.

    Raises:
        ValueError: for an unknown xApp, a missing knob or an xApp deployed twice.
    """
    P = env.cfg.e6p
    out: list[XApp] = []
    for j, nm in enumerate(names):
        if nm not in P_XAPPS:
            raise ValueError(f"unknown E6-P xApp {nm!r} (known: {sorted(P_XAPPS)})")
        if not getattr(P, _NEEDS[nm]):
            raise ValueError(f"E6-P xApp {nm} needs e6p.{_NEEDS[nm]}=True")
        if any(x.name == nm for x in env.xapps) or nm in names[:j]:
            raise ValueError(f"E6-P xApp {nm} deployed twice")
        out.append(P_XAPPS[nm](env, start_idx + j))
    return out

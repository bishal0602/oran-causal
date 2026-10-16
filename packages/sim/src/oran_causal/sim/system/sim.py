"""The system-level plant of the simulator.

UEs, traffic, mobility, radio, handover and RLF, slice-aware scheduler, queues, energy and KPM
counters.

Every tick (`TICK_S` = 40 ms) runs, in this order:

1. hidden processes: per-site log-OU load m(t), mobility, traffic arrivals;
2. radio: large-scale gains, measured RSRP with error, L3 filter; SINR from the previous tick's
   physical PRB occupancy of every other cell (load-coupled interference). Cell sleep removes a
   cell; macro carrier shutdown removes capacity, not coverage;
3. mobility management: RLF timers (T310) on EWMA SINR, A3 entering condition with TTT, HO
   execution or failure, MRO event classification (too late, too early, wrong cell, ping-pong);
4. scheduler: LL dedicated pool, then LL in the shared pool, then eMBB and BE equal share
   (water-filled twice); UEs in HO interruption or RLF outage are not served;
5. queues, SLA bookkeeping per UE-second, energy, KPM counters.

Configuration changes applied by the RIC take effect at the start of the next tick. Exogenous
randomness is keyed by (seed, stream, tick), so arms with different actions see the same tape.
Stress scenarios are drawn once from the (seed, "scenario") stream and are otherwise
deterministic in time; `base` builds no scenario object.

With the power and protected-slice extension (`E6PConfig`) enabled, the scheduler pools per cell
are: the LL dedicated pool `cap * ll_ratio` (LL only, not work-conserving); the protected
minimum share `min(cap * prot_min, cap - dedicated)`, served to protected UEs first with
whatever they leave returned to the shared pool (work-conserving); and the shared pool for the
rest. A protected UE-second violates when it is backlogged for at least `prot_min_backlog_s`
with throughput below `prot_floor_bps`, or in outage.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from oran_causal.sim.system import config as C
from oran_causal.sim.system.geometry import GainMaps, Layout

STREAMS = {
    "layout": 1,
    "ue": 2,
    "mob": 3,
    "traffic": 4,
    "meas": 5,
    "load": 6,
    "kpm": 7,
    "xapp": 8,
    "scenario": 9,
    "e6p": 10,
}
LL, EMBB, BE = 0, 1, 2
A_L3 = 0.5 ** (C.L3_K / 4.0)
NOISE_W = 10 ** ((C.NOISE_DBM_HZ + 10 * np.log10(C.BW_HZ) + C.UE_NF_DB) / 10) / 1000.0
TAPE_TAG = 6600
RESEED_TAG = 6620
LL_RING = 64


def tape_rng(seed: int, stream: str, t: int = 0, salt: int | None = None) -> np.random.Generator:
    """Generator of one (seed, stream, t) cell of the exogenous tape.

    Args:
        seed: episode seed.
        stream: a key of `STREAMS`.
        t: tick, second or draw index within the stream.
        salt: `None` for the episode tape; an integer re-draws the future (`E6Env.copy(reseed=)`).

    Returns:
        A fresh `numpy` generator.
    """
    if salt is None:
        return np.random.default_rng([int(seed), TAPE_TAG, STREAMS[stream], int(t)])
    return np.random.default_rng(
        [int(seed), TAPE_TAG, STREAMS[stream], int(t), RESEED_TAG, int(salt)]
    )


class _Sched(NamedTuple):
    in_out: np.ndarray
    prb_rate: np.ndarray
    car: np.ndarray
    cap_prb: np.ndarray
    demand: np.ndarray
    alloc: np.ndarray
    served: np.ndarray
    ded: np.ndarray
    used_ded: np.ndarray
    m_share: np.ndarray | None


class Plant:
    """One episode of the radio access network, advanced tick by tick.

    The RIC-controlled configuration (`cio`, `hys`, `ttt`, `ll_ratio`, `asleep`, `n_car`,
    `ptx_off`, `prot_min`) is read at the start of every tick. `sla` holds the episode SLA
    accumulators and `ctr` the KPM counters since the last `take_counters`.
    """

    def __init__(self, cfg: C.E6Config) -> None:
        self.cfg = cfg
        s = cfg.seed
        r = tape_rng(s, "layout")
        self.lay = Layout(cfg, r)
        self.gm = GainMaps(self.lay, cfg, r)
        nc = self.lay.n_cells
        self.nc = nc
        self.site_of = np.minimum(self.lay.cell_site, 6)
        self._init_ues(cfg, tape_rng(s, "ue"))
        self.oam_hook: Callable[[tuple, float, float], None] | None = None
        self.scn = None if cfg.scenario == "base" else make_scenario(cfg, self)
        self.rng_salt: int | None = None
        self._init_e6p(cfg)
        self._init_knobs(nc)
        self._init_radio_state()
        self._init_traffic_state()

    def _init_ues(self, cfg: C.E6Config, ru: np.random.Generator) -> None:
        n = cfg.ue_count()
        self.n = n
        hot = ru.uniform(size=n) < 0.35
        pos = ru.uniform(-900, 900, (n, 2))
        k = ru.integers(len(self.lay.hotspots), size=n)
        pos[hot] = self.lay.hotspots[k[hot]] + ru.normal(0, cfg.hotspot_sigma_m, (hot.sum(), 2))
        self.pos = self.lay.wrap(pos)
        self.indoor = ru.uniform(size=n) < 0.7
        mobile = ~self.indoor
        vfrac = 0.1 if cfg.mobility == "ped" else 0.4
        veh = mobile & (ru.uniform(size=n) < vfrac)
        self.speed = np.where(veh, ru.uniform(30, 60, n) / 3.6, np.where(mobile, 3.0 / 3.6, 0.0))
        self.heading = ru.uniform(0, 2 * np.pi, n)
        self.mobile_idx = np.nonzero(self.speed > 0)[0]
        u = ru.uniform(size=n)
        self.sl = np.where(u < cfg.frac_ll, LL, np.where(u < cfg.frac_ll + cfg.frac_embb, EMBB, BE))
        self.ll_pps = ru.uniform(*C.LL_RATE_PPS, n) * cfg.lf()

    def _init_e6p(self, cfg: C.E6Config) -> None:
        P = cfg.e6p
        self.P, self.p_on, self.p_ptx, self.p_prot = P, P.enabled, bool(P.ptx_on), bool(P.prot_on)
        if self.p_on:
            validate_e6p(P)
        nc, n = self.nc, self.n
        self.ptx_cells = self.lay.is_macro.copy() if P.ptx_scope == "macro" else np.ones(nc, bool)
        self.ptx_off = np.where(self.ptx_cells & self.p_ptx, float(P.ptx_init_db), 0.0)
        self.prot = np.zeros(n, bool)
        if self.p_prot:
            draw = tape_rng(cfg.seed, "e6p", 1).uniform(size=n)
            self.prot = (self.sl == EMBB) & (draw < P.prot_frac)
        self.prot_min = np.full(nc, float(P.prot_min_init) if self.p_prot else 0.0)

    def _init_knobs(self, nc: int) -> None:
        self.cio = np.zeros((nc, nc))
        self.hys = np.full(nc, 2.0)
        self.ttt = np.full(nc, 320)
        self.ll_ratio = np.zeros(nc)
        self.asleep = np.zeros(nc, bool)
        self.waking_until = np.full(nc, -1.0)
        self.n_trx = np.where(self.lay.is_macro, C.MACRO_NTRX, 1)
        self.n_car = self.n_trx.copy()
        self.car_on_at = np.full(nc, -1.0)

    def _init_radio_state(self) -> None:
        n, nc = self.n, self.nc
        self.t = 0
        g = self.gains()
        self.l3 = g.copy()
        self.serv = np.argmax(g, 1)
        self.sinr_ewma = np.zeros(n)
        self.ttt_cand = np.full(n, -1)
        self.ttt_acc = np.zeros(n)
        self.int_until = np.zeros(n)
        self.t310 = np.full(n, -1.0)
        self.last_ho_t = np.full(n, -1e9)
        self.last_ho_src = np.full(n, -1)
        self.rho = np.full(nc, 0.3)
        self.occ = np.full(nc, 0.3)

    def _init_traffic_state(self) -> None:
        n, nc = self.n, self.nc
        self.m = np.ones(7)
        self.q = np.zeros(n)
        self.ll_arr = np.zeros((n, LL_RING))
        self.ll_cum_a = np.zeros(n)
        self.ll_cum_s = np.zeros(n)
        self.sec_backlog_t = np.zeros(n)
        self.sec_bits = np.zeros(n)
        self.sec_maxdelay = np.zeros(n)
        self.sec_outage = np.zeros(n, bool)
        self.sla: dict[str, Any] = {
            "viol_ue_s": 0.0,
            "ue_s": 0.0,
            "ll_viol": 0.0,
            "embb_viol": 0.0,
            "outage_viol": 0.0,
            "severe": 0,
            "energy_j": 0.0,
            "rlf": 0,
            "ho": 0,
            "pingpong": 0,
        }
        if self.p_on:
            self.sla["lowsinr_ue_s"] = 0.0
        if self.p_prot:
            self.sla.update({"prot_viol": 0.0, "prot_ue_s": 0.0})
        self.cell_viol_run = np.zeros((nc, 3))
        self.ctr = self.new_counters()

    def gains(self) -> np.ndarray:
        """Current large-scale gains in dBm, shape (UEs, cells); sleeping cells read -300."""
        g = self.gm.lookup(self.pos).astype(float)
        if self.p_ptx:
            g += self.ptx_off[None, :]
        g[self.indoor] -= C.O2I_DB
        g[:, self.asleep | (self.waking_until > self.t * C.TICK_S)] = -300.0
        return g

    def _tape(self, stream: str, t: int) -> np.random.Generator:
        if self.rng_salt is None:
            return tape_rng(self.cfg.seed, stream, t)
        return tape_rng(self.cfg.seed, stream, t, self.rng_salt)

    def new_counters(self) -> dict[str, Any]:
        """A fresh, zeroed KPM counter set."""
        nc = self.nc
        c: dict[str, Any] = {
            "prb_used": np.zeros((nc, 3)),
            "prb_rsv": np.zeros(nc),
            "prb_cap": np.zeros(nc),
            "ticks": 0,
            "bits": np.zeros((nc, 3)),
            "ho_att": np.zeros((nc, nc)),
            "ho_succ": np.zeros((nc, nc)),
            "too_late": np.zeros((nc, nc)),
            "too_early": np.zeros((nc, nc)),
            "wrong_cell": np.zeros((nc, nc)),
            "pingpong": np.zeros((nc, nc)),
            "rlf": np.zeros(nc),
            "energy_j": np.zeros(nc),
            "ll_delay": [[] for _ in range(nc)],
            "embb_thp": [[] for _ in range(nc)],
            "act_ue": np.zeros((nc, 3)),
        }
        if self.p_on:
            c.update({"sinr_serv": [], "sinr_val": []})
            if self.p_prot:
                for key in (
                    "prot_dem",
                    "prot_used",
                    "prot_min_prb",
                    "prot_bits",
                    "prot_act",
                    "prot_eval",
                    "prot_below",
                ):
                    c[key] = np.zeros(nc)
        return c

    def take_counters(self) -> dict[str, Any]:
        """Return the counters accumulated since the last call and start new ones."""
        c, self.ctr = self.ctr, self.new_counters()
        return c

    @staticmethod
    def se(sinr_db: np.ndarray) -> np.ndarray:
        """Truncated-Shannon spectral efficiency (TR 36.942 A.2) in bit/s/Hz."""
        s = C.SE_ALPHA * np.log2(1 + 10 ** (sinr_db / 10))
        return np.where(sinr_db < C.SINR_MIN_DB, 0.0, np.minimum(s, C.SE_MAX))

    def tick(self) -> None:
        """Advance the plant by one tick."""
        dt, t = C.TICK_S, self.t
        now = t * dt
        ll = self._hidden(t, now, dt)
        sinr, pw = self._radio(t)
        self._mobility(now, dt, pw)
        sc = self._schedule(now, dt, sinr, ll)
        backlogged = self._queues(t, dt, now, ll, sc)
        self._count(sc, backlogged, ll)
        self._energy(sc.car, now, dt)
        self.t += 1
        if self.t % C.TICKS_PER_CONTROL == 0:
            self._close_second()

    def _hidden(self, t: int, now: float, dt: float) -> np.ndarray:
        cfg, n = self.cfg, self.n
        rl = self._tape("load", t)
        a = np.exp(-dt / cfg.m_tau_s)
        logm = np.log(self.m) * a + cfg.m_sigma * np.sqrt(1 - a * a) * rl.normal(size=7)
        self.m = np.exp(logm)
        if self.scn is not None:
            self.scn.on_tick(self, now)
        rm = self._tape("mob", t)
        turn = rm.uniform(size=n) < dt / 20.0
        self.heading = np.where(turn, rm.uniform(0, 2 * np.pi, n), self.heading)
        mv = self.mobile_idx
        if len(mv):
            step = self.speed[mv] * dt
            heading = np.c_[np.cos(self.heading[mv]), np.sin(self.heading[mv])]
            self.pos[mv] = self.lay.wrap_near(self.pos[mv] + step[:, None] * heading)
        if self.scn is not None:
            self.scn.move(self, dt)
        return self._arrivals(t, now, dt)

    def _arrivals(self, t: int, now: float, dt: float) -> np.ndarray:
        cfg, n = self.cfg, self.n
        rt = self._tape("traffic", t)
        site = self.site_of[self.serv]
        total = cfg.warmup_s + cfg.scored_s
        ramp = cfg.ramp[0] + (cfg.ramp[1] - cfg.ramp[0]) * min(now / max(total, 1e-9), 1.0)
        mult = self.m[site] * cfg.lf() * ramp
        ll = self.sl == LL
        arr = np.zeros(n)
        arr[ll] = rt.poisson(self.ll_pps[ll] * dt * ramp) * C.LL_PKT_BYTES * 8
        rate = np.where(self.sl == EMBB, cfg.embb_files_per_s, cfg.be_files_per_s) * mult
        if self.scn is not None:
            rate = rate * self.scn.traffic_mult(now, self.pos)
        files = rt.poisson(np.where(ll, 0.0, rate * dt))
        arr[~ll] = files[~ll] * C.EMBB_FILE_BYTES * 8
        self.q += arr
        self.ll_cum_a += np.where(ll, arr, 0.0)
        self.ll_arr[:, t % LL_RING] = self.ll_cum_a
        return ll

    def _radio(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        g = self.gains()
        rq = self._tape("meas", t)
        meas = g + rq.normal(0, C.MEAS_ERR_DB, g.shape)
        self.l3 = (1 - A_L3) * self.l3 + A_L3 * meas
        pw = 10 ** (g / 10) / 1000.0
        load_w = pw * self.occ[None, :]
        tot_int = load_w.sum(1)
        idx = np.arange(self.n)
        s_w = pw[idx, self.serv]
        sinr = 10 * np.log10(s_w / (tot_int - load_w[idx, self.serv] + NOISE_W) + 1e-30)
        self.sinr_ewma = 0.8 * self.sinr_ewma + 0.2 * sinr
        return sinr, pw

    def _mobility(self, now: float, dt: float, pw: np.ndarray) -> None:
        in_out = self.int_until > now
        bad = (self.sinr_ewma < C.QOUT_DB) & ~in_out
        self.t310 = np.where(bad & (self.t310 < 0), now, self.t310)
        self.t310 = np.where(self.sinr_ewma > C.QIN_DB, -1.0, self.t310)
        rlf = (self.t310 >= 0) & (now - self.t310 >= C.T310_S) & ~in_out
        for u in np.nonzero(rlf)[0]:
            self._rlf(u, now)
        idx = np.arange(self.n)
        cand_val = self.l3 + self.cio[self.serv] - self.hys[self.serv][:, None]
        cand_val[idx, self.serv] = -1e9
        cand = np.argmax(cand_val, 1)
        enter = (cand_val[idx, cand] > self.l3[idx, self.serv]) & ~in_out
        same = enter & (cand == self.ttt_cand)
        self.ttt_acc = np.where(same, self.ttt_acc + dt * 1000, np.where(enter, dt * 1000, 0.0))
        self.ttt_cand = np.where(enter, cand, -1)
        fire = enter & (self.ttt_acc >= self.ttt[self.serv])
        for u in np.nonzero(fire)[0]:
            self._handover(u, int(cand[u]), now, pw)

    def _schedule(self, now: float, dt: float, sinr: np.ndarray, ll: np.ndarray) -> _Sched:
        nc = self.nc
        in_out = self.int_until > now
        prb_rate = self.se(sinr) * C.PRB_HZ * (1 - C.OVERHEAD)
        car = np.where(self.car_on_at > now, self.n_car - 1, self.n_car)
        cap_frac = car / self.n_trx
        cap_prb = np.where(self.asleep | (self.waking_until > now), 0.0, float(C.N_PRB) * cap_frac)
        demand = np.where(in_out | (prb_rate <= 0), 0.0, self.q / np.maximum(prb_rate * dt, 1e-9))
        cell = self.serv
        ded = cap_prb * self.ll_ratio
        d_ll = np.where(ll, demand, 0.0)
        tot_ll = np.bincount(cell, d_ll, nc)
        frac_ded = np.where(tot_ll > 0, np.minimum(1.0, ded / np.maximum(tot_ll, 1e-9)), 0.0)
        alloc_ded = d_ll * frac_ded[cell]
        used_ded = np.bincount(cell, alloc_ded, nc)
        left = cap_prb - ded
        alloc = alloc_ded.copy()
        d_o = np.where(ll, d_ll - alloc_ded, demand)
        m_share = None
        if self.p_prot:
            m_share, d_o, left = self._protected_share(cap_prb, left, d_o, alloc)
        for _ in range(2):
            nb = np.bincount(cell, (d_o > 1e-9).astype(float), nc)
            share = np.where(nb > 0, left / np.maximum(nb, 1), 0.0)
            give = np.minimum(d_o, share[cell])
            alloc += give
            d_o = d_o - give
            left = left - np.bincount(cell, give, nc)
        served = np.minimum(self.q, alloc * prb_rate * dt)
        self.q -= served
        used = np.bincount(cell, alloc, nc)
        self.rho = np.where(cap_prb > 0, np.minimum(1.0, used / np.maximum(cap_prb, 1)), 0.0)
        self.occ = self.rho * cap_prb / C.N_PRB
        return _Sched(in_out, prb_rate, car, cap_prb, demand, alloc, served, ded, used_ded, m_share)

    def _protected_share(
        self, cap_prb: np.ndarray, left: np.ndarray, d_o: np.ndarray, alloc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nc, cell = self.nc, self.serv
        m_share = np.minimum(cap_prb * self.prot_min, np.maximum(left, 0.0))
        d_p = np.where(self.prot, d_o, 0.0)
        left_m = m_share.copy()
        for _ in range(2):
            nbp = np.bincount(cell, (d_p > 1e-9).astype(float), nc)
            sh = np.where(nbp > 0, left_m / np.maximum(nbp, 1), 0.0)
            give = np.minimum(d_p, sh[cell])
            alloc += give
            d_p = d_p - give
            d_o = d_o - give
            left_m = left_m - np.bincount(cell, give, nc)
        left = left - (m_share - left_m)
        return m_share, d_o, left

    def _queues(self, t: int, dt: float, now: float, ll: np.ndarray, sc: _Sched) -> np.ndarray:
        served = sc.served
        self.ll_cum_s += np.where(ll, served, 0.0)
        hol = np.zeros(self.n)
        if ll.any():
            ring = self.ll_arr[ll]
            order = (t - np.arange(LL_RING)) % LL_RING
            hist = ring[:, order]
            ok = hist > self.ll_cum_s[ll][:, None] + 1e-6
            depth = ok.sum(1)
            hol[ll] = np.where(self.q[ll] > 1e-6, depth * dt + C.HOL_PROC_S, 0.0)
        self.sec_maxdelay = np.maximum(self.sec_maxdelay, hol)
        backlogged = (self.q > 1e-6) | (served > 0)
        self.sec_backlog_t += np.where(backlogged & ~ll, dt, 0.0)
        self.sec_bits += served
        self.sec_outage |= sc.in_out & (self.int_until - now > C.HO_EXEC_S + 1e-9)
        self.sec_outage |= sc.prb_rate <= 0
        return backlogged

    def _count(self, sc: _Sched, backlogged: np.ndarray, ll: np.ndarray) -> None:
        nc, cell, ctr = self.nc, self.serv, self.ctr
        alloc, served = sc.alloc, sc.served
        ctr["ticks"] += 1
        for sidx in (LL, EMBB, BE):
            m_ = self.sl == sidx
            ctr["prb_used"][:, sidx] += np.bincount(cell[m_], alloc[m_], nc)
            ctr["bits"][:, sidx] += np.bincount(cell[m_], served[m_], nc)
            ctr["act_ue"][:, sidx] += np.bincount(cell[m_], backlogged[m_].astype(float), nc)
        ctr["prb_cap"] += sc.cap_prb
        ctr["prb_rsv"] += sc.ded - sc.used_ded
        if self.p_prot:
            pm = self.prot
            cap_prb = sc.cap_prb
            capd = np.maximum(cap_prb, 1e-9)
            dem = np.bincount(cell[pm], sc.demand[pm], nc)
            ctr["prot_dem"] += np.where(cap_prb > 0, np.minimum(1.0, dem / capd), 0.0)
            ctr["prot_used"] += np.bincount(cell[pm], alloc[pm], nc)
            ctr["prot_min_prb"] += sc.m_share
            ctr["prot_bits"] += np.bincount(cell[pm], served[pm], nc)
            ctr["prot_act"] += np.bincount(cell[pm], backlogged[pm].astype(float), nc)

    def _energy(self, car: np.ndarray, now: float, dt: float) -> None:
        sleep_w = (C.MACRO_NTRX - car) * C.MACRO_SLEEP_W / C.MACRO_NTRX
        if self.p_ptx:
            lin = 10 ** (self.ptx_off / 10)
            p_mac = car * (C.MACRO_P0_W + C.MACRO_DP * self.rho * C.MACRO_PMAX_W * lin) + sleep_w
            p_pic = C.PICO_P0_W + C.PICO_DP * self.rho * C.PICO_PMAX_W * lin
        else:
            p_mac = car * (C.MACRO_P0_W + C.MACRO_DP * self.rho * C.MACRO_PMAX_W) + sleep_w
            p_pic = C.PICO_P0_W + C.PICO_DP * self.rho * C.PICO_PMAX_W
        pw_cell = np.where(self.lay.is_macro, p_mac, np.where(self.asleep, C.PICO_SLEEP_W, p_pic))
        pw_cell = np.where(~self.lay.is_macro & (self.waking_until > now), C.PICO_P0_W, pw_cell)
        self.ctr["energy_j"] += pw_cell * dt
        if now >= self.cfg.warmup_s:
            self.sla["energy_j"] += float(pw_cell.sum() * dt)

    def _handover(self, u: int, tgt: int, now: float, pw: np.ndarray) -> None:
        src = int(self.serv[u])
        self.ctr["ho_att"][src, tgt] += 1
        s = pw[u, tgt]
        i = (pw[u] * self.occ).sum() - pw[u, tgt] * self.occ[tgt]
        sinr_t = 10 * np.log10(s / (i + NOISE_W) + 1e-30)
        if sinr_t < C.HO_FAIL_SINR_DB:
            self.serv[u] = tgt
            self.last_ho_t[u], self.last_ho_src[u] = now, src
            self._rlf(u, now, ho_fail=True)
            return
        self.ctr["ho_succ"][src, tgt] += 1
        self.sla["ho"] += 1
        if self.last_ho_src[u] == tgt and now - self.last_ho_t[u] <= C.PINGPONG_S:
            self.ctr["pingpong"][tgt, src] += 1
            self.sla["pingpong"] += 1
        self.last_ho_t[u], self.last_ho_src[u] = now, src
        self.serv[u] = tgt
        self.int_until[u] = now + C.HO_EXEC_S
        self.ttt_cand[u], self.ttt_acc[u] = -1, 0.0
        self.sinr_ewma[u] = sinr_t

    def _rlf(self, u: int, now: float, ho_fail: bool = False) -> None:
        cur = int(self.serv[u])
        best = int(np.argmax(self.l3[u]))
        self.ctr["rlf"][cur] += 1
        self.sla["rlf"] += 1
        recent = now - self.last_ho_t[u] <= C.TOO_EARLY_S or ho_fail
        if recent and self.last_ho_src[u] >= 0:
            src = int(self.last_ho_src[u])
            if best == src:
                self.ctr["too_early"][src, cur] += 1
            elif best != cur:
                self.ctr["wrong_cell"][src, cur] += 1
        elif best != cur:
            self.ctr["too_late"][cur, best] += 1
        self.serv[u] = best
        self.int_until[u] = now + C.RLF_OUTAGE_S
        self.t310[u] = -1.0
        self.ttt_cand[u], self.ttt_acc[u] = -1, 0.0
        self.sinr_ewma[u] = 0.0

    def _close_second(self) -> None:
        sl = self.sl
        ll_v = (sl == LL) & ((self.sec_maxdelay > C.LL_DELAY_TARGET_S) | self.sec_outage)
        thp = np.where(
            self.sec_backlog_t > 0,
            self.sec_bits / np.maximum(self.sec_backlog_t, 1e-9),
            np.inf,
        )
        slow = (self.sec_backlog_t >= 0.2) & (thp < C.EMBB_THP_TARGET_BPS)
        em_v = (sl == EMBB) & (slow | self.sec_outage)
        be_v = (sl == BE) & self.sec_outage
        v = ll_v | em_v | be_v
        scored = self.t * C.TICK_S > self.cfg.warmup_s
        if scored:
            self._score_second(v, ll_v, em_v)
        self.sec_viol_ll, self.sec_viol_embb, self.sec_viol_be = ll_v, em_v, be_v
        self.sec_out, self.sec_scored = self.sec_outage.copy(), scored
        cell = self.serv
        if self.p_on:
            self._close_second_p(thp, scored)
        for u in np.nonzero(sl == LL)[0]:
            if not self.sec_outage[u]:
                self.ctr["ll_delay"][cell[u]].append(self.sec_maxdelay[u])
        for u in np.nonzero((sl == EMBB) & (self.sec_backlog_t >= 0.2))[0]:
            self.ctr["embb_thp"][cell[u]].append(min(thp[u], 1e9))
        self.sec_backlog_t[:] = 0
        self.sec_bits[:] = 0
        self.sec_maxdelay[:] = 0
        self.sec_outage[:] = False

    def _score_second(self, v: np.ndarray, ll_v: np.ndarray, em_v: np.ndarray) -> None:
        S, sl = self.sla, self.sl
        S["viol_ue_s"] += v.sum()
        S["ue_s"] += self.n
        S["ll_viol"] += ll_v.sum()
        S["embb_viol"] += em_v.sum()
        S["outage_viol"] += self.sec_outage.sum()
        for s_ in (LL, EMBB, BE):
            m_ = sl == s_
            tot = np.bincount(self.serv[m_], minlength=self.nc)
            bad = np.bincount(self.serv[m_ & v], minlength=self.nc)
            frac = np.where(tot > 0, bad / np.maximum(tot, 1), 0.0)
            run = np.where(frac > 0.2, self.cell_viol_run[:, s_] + 1, 0)
            S["severe"] += int((run == 10).sum())
            self.cell_viol_run[:, s_] = run

    def sleep_handover(self, c: int, now: float) -> None:
        """Hand every UE served by pico `c` over to its best remaining cell by measured L3 RSRP.

        Called by the RIC right after `asleep[c]` is set, only when the extension is enabled.
        The normal HO execution applies (a target below `HO_FAIL_SINR_DB` still fails into RLF);
        UEs already in HO interruption or RLF outage are only re-pointed.
        """
        g = self.gains()
        pw = 10 ** (g / 10) / 1000.0
        off = self.asleep | (self.waking_until > now)
        for u in np.nonzero(self.serv == c)[0]:
            l3 = np.where(off, -np.inf, self.l3[u])
            tgt = int(np.argmax(l3))
            if self.int_until[u] > now:
                self.serv[u] = tgt
                self.ttt_cand[u], self.ttt_acc[u] = -1, 0.0
            else:
                self._handover(u, tgt, now, pw)

    def _close_second_p(self, thp: np.ndarray, scored: bool) -> None:
        ctr, nc, cell = self.ctr, self.nc, self.serv
        ok = ~self.sec_outage
        ctr["sinr_serv"].append(cell[ok].copy())
        ctr["sinr_val"].append(self.sinr_ewma[ok].copy())
        if scored:
            self.sla["lowsinr_ue_s"] += float((self.sinr_ewma < C.QIN_DB).sum())
        if not self.p_prot:
            return
        P, pm = self.P, self.prot
        act = pm & (self.sec_backlog_t >= P.prot_min_backlog_s)
        pv = pm & ((act & (thp < P.prot_floor_bps)) | self.sec_outage)
        self.sec_viol_prot = pv
        if scored:
            self.sla["prot_viol"] += float(pv.sum())
            self.sla["prot_ue_s"] += float(pm.sum())
        ctr["prot_eval"] += np.bincount(cell[act | pv], minlength=nc)
        ctr["prot_below"] += np.bincount(cell[pv], minlength=nc)


def wrap_dist(lay: Layout, pos: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Wrap-around (minimum-image) distance of each point in `pos` (n, 2) to `centre` (2,)."""
    c = np.asarray(centre)[None, :] + lay.cands()
    return np.linalg.norm(np.atleast_2d(pos)[:, None, :] - c[None], axis=-1).min(1)


class Scenario:
    """Base of the stress scenarios: a calm no-op that never reads the RIC configuration."""

    def __init__(self, cfg: C.E6Config, plant: Plant) -> None:
        self.cfg, self.lay = cfg, plant.lay

    def at(self, frac: float) -> float:
        """Absolute episode time in seconds of a fraction of the scored window."""
        return self.cfg.warmup_s + frac * self.cfg.scored_s

    def on_tick(self, plant: Plant, now: float) -> None:
        """Hook run at the start of every tick."""

    def move(self, plant: Plant, dt: float) -> None:
        """Hook run after the UE random walk of every tick."""

    def traffic_mult(self, now: float, pos: np.ndarray) -> np.ndarray | float:
        """Multiplier of the eMBB and BE offered traffic per UE."""
        return 1.0

    def dist(self, pos: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Wrap-around distance of each point in `pos` to `centre`."""
        return wrap_dist(self.lay, pos, centre)

    def _anchor(self, r: np.random.Generator, mode: str) -> np.ndarray:
        d = C.ISD_M
        site = self.lay.sites[int(r.integers(7))]
        if mode == "band":
            rad, ang = r.uniform(150, 250), r.uniform(0, 2 * np.pi)
        elif mode == "edge":
            rad, ang = d / 2.0, np.pi / 3.0 * int(r.integers(6))
        elif mode == "vertex":
            rad, ang = d / np.sqrt(3.0), np.pi / 6.0 + np.pi / 3.0 * int(r.integers(6))
        else:
            raise ValueError(f"unknown scenario location {mode!r}")
        return self.lay.wrap(site + rad * np.array([np.cos(ang), np.sin(ang)]))[0]


class SurgeScenario(Scenario):
    """S1: event or moving-hotspot surge.

    The eMBB and BE offered traffic of every UE inside a disk is multiplied by
    `1 + (surge_mult - 1) * envelope(t)`, a trapezoid of onset, ramp up, hold and ramp down. The
    anchor is `band` (150-250 m from a random macro site, the hotspot rule) or `vertex` (a 3-site
    corner).
    """

    def __init__(self, cfg: C.E6Config, plant: Plant) -> None:
        super().__init__(cfg, plant)
        r = tape_rng(cfg.seed, "scenario", 1)
        self.c0 = self._anchor(r, cfg.surge_loc)
        a = r.uniform(0, 2 * np.pi)
        self.vel = cfg.surge_speed_mps * np.array([np.cos(a), np.sin(a)])
        self.t0 = self.at(cfg.surge_onset_frac)
        self.ramp = cfg.surge_ramp_frac * cfg.scored_s
        self.t_end = self.t0 + 2 * self.ramp + cfg.surge_hold_frac * cfg.scored_s

    def envelope(self, now: float) -> float:
        """Surge intensity in [0, 1] at time `now`."""
        if now < self.t0 or now > self.t_end:
            return 0.0
        if self.ramp <= 0:
            return 1.0
        return float(min(1.0, (now - self.t0) / self.ramp, (self.t_end - now) / self.ramp))

    def centre(self, now: float) -> np.ndarray:
        """Disk centre at time `now`; it drifts only while the event lasts."""
        el = min(max(now, self.t0), self.t_end) - self.t0
        return self.lay.wrap(self.c0 + self.vel * el)[0]

    def inside(self, now: float, pos: np.ndarray) -> np.ndarray:
        """Mask of the points inside the surge disk at time `now`."""
        return self.dist(pos, self.centre(now)) <= self.cfg.surge_radius_m

    def traffic_mult(self, now: float, pos: np.ndarray) -> np.ndarray | float:
        """Per-UE multiplier of the offered eMBB and BE traffic."""
        e = self.envelope(now)
        if e <= 0.0:
            return 1.0
        return 1.0 + (self.cfg.surge_mult - 1.0) * e * self.inside(now, pos)


class MistuneScenario(Scenario):
    """S2, the too-late-HO hypothesis: a high-speed road and an OAM Hys/TTT rollout.

    The road (present from t = 0) crosses a cluster whose Hys and TTT are overwritten by an OAM
    parameter rollout at `mis_onset_frac`. Corridor UEs drive the section at `corr_speed_kmh` with
    a U-turn at each end; the mis-set cluster is every cell that is strongest (outdoor, all cells
    on) anywhere on the section. The anchor is `edge` (midway between two adjacent macro sites) or
    `vertex`.

    OAM writes (the rollout, and an SMO restore via `oam_set`) bypass the RIC: no request, NACK,
    churn or actuator dwell. Each written knob is reported to `plant.oam_hook`, which records it as
    the knob's last-known-good, so a RIC rollback cannot undo an OAM write.
    """

    def __init__(self, cfg: C.E6Config, plant: Plant) -> None:
        super().__init__(cfg, plant)
        r = tape_rng(cfg.seed, "scenario", 2)
        self.centre = self._anchor(r, cfg.corr_loc)
        a = r.uniform(0, np.pi)
        self.u = np.array([np.cos(a), np.sin(a)])
        self.half = cfg.corr_len_m / 2.0
        n = plant.n
        k = round(cfg.corr_frac_ue * n)
        self.idx = np.sort(r.choice(n, k, replace=False))
        self.s = r.uniform(-self.half, self.half, k)
        self.dir = np.where(r.uniform(size=k) < 0.5, 1.0, -1.0)
        self.v = cfg.corr_speed_kmh / 3.6
        plant.indoor[self.idx] = False
        plant.speed[self.idx] = self.v
        self._place(plant)
        plant.mobile_idx = np.setdiff1d(plant.mobile_idx, self.idx)
        road = np.arange(-self.half, self.half + 1e-9, C.GRID_M)[:, None] * self.u
        pts = self.lay.wrap(self.centre + road)
        self.cluster = np.unique(np.argmax(plant.gm.lookup(pts), 1))
        self.t_mis = self.at(cfg.mis_onset_frac)
        self.applied = False
        self.push_t: float | None = None
        self.pre_hys: np.ndarray | None = None
        self.pre_ttt: np.ndarray | None = None
        self.oam_log: list[tuple[float, str]] = []

    def _place(self, plant: Plant) -> None:
        plant.pos[self.idx] = self.lay.wrap(self.centre + self.s[:, None] * self.u)
        back = np.where(self.dir > 0, 0.0, np.pi)
        plant.heading[self.idx] = np.arctan2(self.u[1], self.u[0]) + back

    def oam_set(self, plant: Plant, hys: Any, ttt: Any, now: float, tag: str) -> None:
        """OAM/SMO write of per-cluster-cell Hys (dB) and TTT (ms) outside the RIC."""
        hys = np.broadcast_to(np.asarray(hys, float), self.cluster.shape)
        ttt = np.broadcast_to(np.asarray(ttt, int), self.cluster.shape)
        plant.hys[self.cluster] = hys
        plant.ttt[self.cluster] = ttt
        if plant.oam_hook is not None:
            for c, h, m in zip(self.cluster, hys, ttt, strict=True):
                plant.oam_hook(("hys", int(c)), float(h), now)
                plant.oam_hook(("ttt", int(c)), float(m), now)
        self.oam_log.append((now, tag))

    def on_tick(self, plant: Plant, now: float) -> None:
        """Apply the one-time OAM rollout once its onset is reached."""
        if not self.applied and now >= self.t_mis:
            self.pre_hys, self.pre_ttt = (
                plant.hys[self.cluster].copy(),
                plant.ttt[self.cluster].copy(),
            )
            self.oam_set(plant, self.cfg.mis_hys_db, int(self.cfg.mis_ttt_ms), now, "push")
            self.applied, self.push_t = True, now

    def move(self, plant: Plant, dt: float) -> None:
        """Drive the corridor UEs along the road, reflecting at both ends."""
        s = self.s + self.dir * self.v * dt
        hi, lo = s > self.half, s < -self.half
        s = np.where(hi, 2 * self.half - s, np.where(lo, -2 * self.half - s, s))
        self.dir = np.where(hi, -1.0, np.where(lo, 1.0, self.dir))
        self.s = s
        self._place(plant)


SCENARIOS: dict[str, type[Scenario]] = {"surge": SurgeScenario, "mistune": MistuneScenario}


def validate_e6p(P: C.E6PConfig) -> None:
    """Reject invalid extension parameters.

    Raises:
        ValueError: if a parameter is out of range.
    """

    def need(ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(f"E6-P: {msg}")

    need(P.ptx_scope in {"macro", "all"}, "ptx_scope must be 'macro' or 'all'")
    lo, hi = P.ptx_range_db
    need(lo <= P.ptx_init_db <= hi, "ptx_init_db must lie in ptx_range_db")
    need(P.ptx_grid_db > 0 and P.ptx_max_step_db > 0, "ptx grid/step must be > 0")
    need(
        0.0 <= P.prot_frac <= 1.0 and P.prot_floor_bps > 0,
        "prot_frac in [0, 1] and prot_floor_bps > 0",
    )
    mlo, mhi = P.prot_min_range
    need(0.0 <= mlo <= P.prot_min_init <= mhi <= 1.0, "need 0 <= prot_min lo <= init <= hi <= 1")
    need(P.prot_min_grid > 0 and P.prot_min_max_step > 0, "prot_min grid/step must be > 0")


def make_scenario(cfg: C.E6Config, plant: Plant | None) -> Scenario:
    """Validate the scenario parameters and build the scenario of `cfg` for `plant`.

    Raises:
        ValueError: if the scenario or one of its parameters is invalid.
    """
    cfg.validate_scenario()
    if cfg.scenario not in SCENARIOS:
        raise ValueError(f"unknown E6 scenario {cfg.scenario!r}")
    if plant is None:
        raise ValueError("a scenario needs a plant")
    return SCENARIOS[cfg.scenario](cfg, plant)

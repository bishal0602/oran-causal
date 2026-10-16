"""System-level simulator configuration and parameter registry.

Each numeric default carries a provenance tag: `[S]` sourced (document or table named), `[A]`
modelling assumption, `[V]` sourced but still to verify against the document, `[I]` inherited
from the base plant, `[S->]` derived from a sourced value.

`SCENARIO_HOLDOUT` holds the reserved stress-scenario variants. Each changes one factor
(timing, location or intensity) of the DEV default and is never used for development runs.
"""

from dataclasses import dataclass, field
from typing import Any

TICK_S = 0.04  # [A]
CONTROL_S = 1.0  # [S] O-RAN WG3 RICARCH near-RT interval, upper end of 10 ms-1 s
TICKS_PER_CONTROL = round(CONTROL_S / TICK_S)

FREQ_GHZ = 2.0  # [V] TS 38.101-1 band n1-like, co-channel macro + pico
BW_HZ = 20e6
N_PRB = 106  # [S] TS 38.101-1 Tab 5.3.2-1 (20 MHz, 15 kHz)
PRB_HZ = 180e3
OVERHEAD = 0.25  # [A]
SE_MAX = 4.4  # [S] TR 36.942 A.2
SE_ALPHA = 0.6  # [S] TR 36.942 A.2
SINR_MIN_DB = -10.0  # [S] TR 36.942 A.2
NOISE_DBM_HZ = -174.0  # [S]
UE_NF_DB = 9.0  # [S] TR 36.814
MACRO_PTX_DBM = 46.0  # [V] TR 36.814 Tab A.2.1.1-2
PICO_PTX_DBM = 30.0  # [V] TR 36.872
MACRO_GAIN_DBI = 14.0  # [S] TR 36.814 Case 1, cable loss folded in [A]
PICO_GAIN_DBI = 5.0  # [A]
MACRO_H, PICO_H, UE_H = 25.0, 10.0, 1.5  # [S] TR 38.901 Tab 7.2-1
HPBW_DEG, AM_DB = 70.0, 20.0  # [S] TR 36.814 Case 1 2D sector pattern
VERTICAL = False  # [S] TR 36.814 Case 1 has no vertical pattern
VBW_DEG, SLAV_DB, TILT_DEG = 10.0, 30.0, 12.0  # [A]
ISD_M = 500.0  # [S] TR 36.814 Case 1
SF_SIGMA_MACRO = (4.0, 6.0)  # [S] TR 38.901 Tab 7.5-6 UMa LOS/NLOS
SF_SIGMA_PICO = (4.0, 7.82)  # [S] TR 38.901 UMi-SC LOS/NLOS
SF_DECORR_M = 50.0  # [S] ~37/50 m, one value for the grid filter [A]
O2I_DB = 20.0  # [A]
GRID_M = 10.0  # [A]

L3_K = 4  # [S] TS 38.331 filterCoefficient
MEAS_ERR_DB = 2.0  # [A] within TS 38.133 relative accuracy
HO_EXEC_S = 0.05  # [A]
HO_FAIL_SINR_DB = -8.0  # [A]
QOUT_DB, QIN_DB = -8.0, -6.0  # [V] TR 36.839
T310_S = 1.0  # [V]
RLF_OUTAGE_S = 1.0  # [A]
PINGPONG_S = 1.0  # [S] TR 36.839 MTS
TOO_EARLY_S = 1.0  # [S]
TTT_SET_MS = (40, 80, 160, 256, 320, 480, 640)  # [S] TS 38.331 subset
CIO_RANGE = (-6.0, 6.0)  # [S] subset of Q-OffsetRange
HYS_RANGE = (0.0, 5.0)

MACRO_P0_W, MACRO_DP, MACRO_PMAX_W, MACRO_SLEEP_W, MACRO_NTRX = (
    130.0,
    4.7,
    20.0,
    75.0,
    2,
)  # [S] Auer 2011
PICO_P0_W, PICO_DP, PICO_PMAX_W, PICO_SLEEP_W = 6.8, 4.0, 0.13, 4.3  # [S] Auer 2011
PICO_WAKE_S = 5.0  # [A]
CARRIER_ON_S = 2.0  # [A]

LL_PKT_BYTES = 200  # [A]
LL_RATE_PPS = (50.0, 200.0)  # [A]
LL_DELAY_TARGET_S = 0.10  # [A] coarse RAN delay target, not a 3GPP URLLC guarantee
EMBB_FILE_BYTES = 0.5e6  # [S] FTP model 3 (TR 36.814)
EMBB_THP_TARGET_BPS = 2e6  # [A]
HOL_PROC_S = 0.004  # [A]

SCENARIO_VERSION = "E6-scn-v1"
SCENARIO_NAMES = ("base", "surge", "mistune")
SCENARIO_HOLDOUT = {
    "surge": {
        "H-time": {"surge_onset_frac": 0.45, "surge_ramp_frac": 0.05},
        "H-loc": {"surge_loc": "vertex"},
        "H-int": {"surge_mult": 4.0},  # [A]
        "H-move": {"surge_speed_mps": 3.0 / 3.6},  # [S] pedestrian 3 km/h
    },
    "mistune": {
        "H-time": {"mis_onset_frac": 0.35},
        "H-loc": {"corr_loc": "vertex"},
        "H-int": {"mis_hys_db": 5.0, "mis_ttt_ms": 640},  # [A]
        "H-dir": {"mis_hys_db": 0.0, "mis_ttt_ms": 40},  # [A] TR 36.839 Set 5 with Hys clipped to 0
    },
}


@dataclass(frozen=True)
class E6PConfig:
    """Power and protected-slice extension of the plant, off by default.

    With every flag off and no extension xApps, the plant, the RIC, the knob list and every RNG
    stream are bit-identical to the base plant `E6-scn-v1`.

    The Tx-power knob `("ptx", c)` is a total-carrier offset in dB on cell `c`'s gain column, so
    serving RSRP, SINR, measured L3 and the interference the cell causes all follow it. EARTH
    power per active carrier is `P0 + DP * rho * PMAX * 10^(offset/10)`. A share of eMBB UEs is
    protected by a per-UE rate floor, and `("prot_min", c)` is a work-conserving minimum PRB
    share for them (TS 28.541 rRMPolicyMinRatio). When the extension is enabled, a sleeping pico
    hands its UEs over (`pico_sleep_ho`) and the base ES xApp also sleeps picos (`es_pico`).
    """

    ptx_on: bool = False
    ptx_scope: str = "macro"  # [S->] macro | all
    ptx_range_db: tuple[float, float] = (-9.0, 3.0)  # [S] TR 38.864 Tab 6.4.1.2-1, Tab 5.1-1 Set 2
    ptx_init_db: float = 0.0  # [I]
    ptx_grid_db: float = 1.0  # [S] TS 38.331 integer-dB power parameters
    ptx_max_step_db: float = 3.0  # [S->] one TR 38.864 evaluation step
    ptx_min_interval_s: float = 10.0  # [A]
    prot_on: bool = False
    prot_frac: float = 0.2  # [A]
    prot_floor_bps: float = 2e6  # [S]+[I] TR 26.925 720p lower end
    prot_min_backlog_s: float = 0.2  # [I]
    prot_min_range: tuple[float, float] = (0.0, 0.5)  # [I]
    prot_min_init: float = 0.0  # [S] TS 28.541 rRMPolicyMinRatio default
    prot_min_grid: float = 0.05  # [I]
    prot_min_max_step: float = 0.10  # [I]
    prot_min_interval_s: float = 5.0  # [A]
    edge_pct: float = 5.0  # [S] ITU-R M.2410 5th-percentile user
    pico_sleep_ho: bool = True
    es_pico: bool = True
    xapps: tuple[str, ...] = ()
    pes_cadence_s: float = 10.0  # [I]
    pes_u_low: float = 0.30  # [I]/[S->]
    pes_u_high: float = 0.80  # [I]
    pes_hold_s: float = 60.0  # [I]
    pes_step_db: float = 3.0  # [S->]
    cov_cadence_s: float = 10.0  # [A]
    cov_sinr_low_db: float = -6.0  # [S->] Q_in
    cov_sinr_ok_db: float = -3.0  # [S->]
    cov_margin_db: float = 3.0  # [S->]
    cov_floor_frac: float = 0.05  # [S->]
    cov_step_db: float = 3.0  # [S->]
    sg_cadence_s: float = 5.0  # [I]
    sg_viol_hi: float = 0.05  # [S->] 95 % availability rule
    sg_viol_lo: float = 0.01  # [A]
    sg_slack: float = 0.5  # [I]
    sg_hold_s: float = 10.0  # [I]
    sg_step: float = 0.05  # [I]

    @property
    def enabled(self) -> bool:
        """Whether any part of the extension is on."""
        return bool(self.ptx_on or self.prot_on or self.xapps)


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


@dataclass(frozen=True)
class E6Config:
    """Episode configuration of the system-level plant.

    Scenario times are fractions of the scored window (onset = `warmup_s + frac * scored_s`), so
    every event falls inside scoring. `scenario` is `base`, `surge` (S1, TR 36.902 cl. 4.6 MLB) or
    `mistune` (S2, TR 36.902 cl. 4.5 MRO). `load_factor` 0 means 1.8 for medium and 2.4 for high
    load.
    """

    seed: int = 0
    load: str = "medium"
    mobility: str = "ped"
    mix: str = "M2"
    update: bool = False
    kpm: str = "nominal"
    n_ue: int = 300  # [A]
    n_pico: int = 3  # [A]
    warmup_s: float = 300.0  # [A]
    scored_s: float = 1800.0  # [A]
    frac_ll: float = 0.15  # [A]
    frac_embb: float = 0.55  # [A]
    embb_files_per_s: float = 0.18  # [A]
    be_files_per_s: float = 0.11  # [A]
    load_factor: float = 0.0  # [A] DEV-calibrated
    ramp: tuple[float, float] = (0.4, 1.0)  # [A]
    m_tau_s: float = 300.0  # [A]
    m_sigma: float = 0.3  # [A]
    n_hotspots: int = 3  # [A]
    hotspot_sigma_m: float = 60.0  # [A]
    kpm_delay_s: tuple[float, float] = (0.1, 0.5)  # [A]
    kpm_drop: float = 0.0
    pl_offset_db: float = 0.0  # [A]
    xapp_threshold_scale: float = 1.0
    scenario: str = "base"
    surge_mult: float = 3.0  # [A] informed by Shafiq et al. SIGMETRICS'13 aggregate ratio [S]
    surge_radius_m: float = 250.0  # [A]
    surge_loc: str = "band"  # [A] band | vertex
    surge_onset_frac: float = 0.2  # [A]
    surge_ramp_frac: float = 0.1  # [A]
    surge_hold_frac: float = 0.3  # [A]
    surge_speed_mps: float = 0.0  # [A]
    corr_frac_ue: float = 0.10  # [A]
    corr_speed_kmh: float = 120.0  # [S] TR 36.839 Tab 5.2.4.1, ITU-R M.2410 cl. 4.11
    corr_len_m: float = 1000.0  # [A]
    corr_loc: str = "edge"  # [A] edge | vertex
    mis_onset_frac: float = 0.1  # [A]
    mis_hys_db: float = 3.0  # [A] TR 36.839 Tab 5.3.2.1 Set 1 A3 offset carried by Hys
    mis_ttt_ms: int = 480  # [S] TR 36.839 Tab 5.3.2.1 Set 1
    extra: dict[str, Any] = field(default_factory=dict)
    e6p: E6PConfig = field(default_factory=E6PConfig)

    def ue_count(self) -> int:
        """Number of UEs in the episode."""
        return self.n_ue

    def lf(self) -> float:
        """Offered-traffic load factor."""
        return self.load_factor or (1.8 if self.load == "medium" else 2.4)

    def validate_scenario(self) -> None:
        """Reject invalid stress-scenario parameters.

        Raises:
            ValueError: if the scenario is unknown or one of its parameters is out of range.
        """
        prefix = f"E6 scenario {self.scenario!r}: "
        _need(
            self.scenario in SCENARIO_NAMES, f"{prefix}unknown scenario (known: {SCENARIO_NAMES})"
        )
        _need(
            self.warmup_s >= 0 and self.scored_s > 0,
            f"{prefix}need warmup_s >= 0 and scored_s > 0",
        )
        if self.scenario == "surge":
            self._validate_surge(prefix)
        elif self.scenario == "mistune":
            self._validate_mistune(prefix)

    def _validate_surge(self, prefix: str) -> None:
        _need(self.surge_mult >= 1.0, f"{prefix}surge_mult must be >= 1")
        _need(self.surge_radius_m > 0, f"{prefix}surge_radius_m must be > 0")
        _need(self.surge_speed_mps >= 0, f"{prefix}surge_speed_mps must be >= 0")
        _need(self.surge_loc in {"band", "vertex"}, f"{prefix}surge_loc must be 'band' or 'vertex'")
        for f in ("surge_onset_frac", "surge_ramp_frac", "surge_hold_frac"):
            _need(0.0 <= getattr(self, f) <= 1.0, f"{prefix}{f} must be in [0, 1]")
        _need(
            self.surge_onset_frac + 2 * self.surge_ramp_frac + self.surge_hold_frac <= 1.0 + 1e-12,
            f"{prefix}the surge must end by the end of the scored window "
            "(onset + 2 ramp + hold <= 1)",
        )

    def _validate_mistune(self, prefix: str) -> None:
        _need(
            0.0 < self.corr_frac_ue <= 1.0 and round(self.corr_frac_ue * self.n_ue) >= 1,
            f"{prefix}corr_frac_ue must be in (0, 1] and put >= 1 UE on the corridor",
        )
        _need(self.corr_speed_kmh > 0, f"{prefix}corr_speed_kmh must be > 0")
        _need(0 < self.corr_len_m <= 7 * ISD_M, f"{prefix}corr_len_m must be in (0, 7 ISD]")
        _need(self.corr_loc in {"edge", "vertex"}, f"{prefix}corr_loc must be 'edge' or 'vertex'")
        _need(0.0 <= self.mis_onset_frac < 1.0, f"{prefix}mis_onset_frac must be in [0, 1)")
        on_grid = abs(self.mis_hys_db * 2 - round(self.mis_hys_db * 2)) < 1e-9
        _need(
            HYS_RANGE[0] <= self.mis_hys_db <= HYS_RANGE[1] and on_grid,
            f"{prefix}mis_hys_db must be on the 0.5 dB grid within {HYS_RANGE}",
        )
        _need(
            int(self.mis_ttt_ms) == self.mis_ttt_ms and int(self.mis_ttt_ms) in TTT_SET_MS,
            f"{prefix}mis_ttt_ms must be one of {TTT_SET_MS}",
        )

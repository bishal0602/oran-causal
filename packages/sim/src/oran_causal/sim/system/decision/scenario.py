"""Frozen E6-P scenario of the system-level studies.

Every study runs the P3 pair (ES with PowerES and SliceGuarantee) in the surge-L40 stratum:
`make_cfg(PAIR, STRATUM, seed, LOAD_FACTOR["L40"])`, 120 s warm-up plus 600 s scored time. The
load factors were calibrated so that the target load share is reached (`TARGET_L`).
"""

from typing import Any

from oran_causal.sim.system import config as C

FROZEN_E6P: dict[str, Any] = {
    "ptx_scope": "macro",
    "ptx_range_db": (-9.0, 3.0),
    "ptx_init_db": 0.0,
    "ptx_grid_db": 1.0,
    "ptx_max_step_db": 3.0,
    "ptx_min_interval_s": 10.0,
    "prot_frac": 0.2,
    "prot_floor_bps": 2e6,
    "prot_min_backlog_s": 0.2,
    "prot_min_range": (0.0, 0.5),
    "prot_min_init": 0.0,
    "prot_min_grid": 0.05,
    "prot_min_max_step": 0.10,
    "prot_min_interval_s": 5.0,
    "edge_pct": 5.0,
    "pico_sleep_ho": True,
    "es_pico": True,
    "pes_cadence_s": 10.0,
    "pes_u_low": 0.30,
    "pes_u_high": 0.80,
    "pes_hold_s": 60.0,
    "pes_step_db": 3.0,
    "cov_cadence_s": 10.0,
    "cov_sinr_low_db": -6.0,
    "cov_sinr_ok_db": -3.0,
    "cov_margin_db": 3.0,
    "cov_floor_frac": 0.05,
    "cov_step_db": 3.0,
    "sg_cadence_s": 5.0,
    "sg_viol_hi": 0.05,
    "sg_viol_lo": 0.01,
    "sg_slack": 0.5,
    "sg_hold_s": 10.0,
    "sg_step": 0.05,
}
E6_FROZEN: dict[str, Any] = {
    "n_ue": 300,
    "n_pico": 3,
    "mobility": "ped",
    "kpm": "nominal",
    "update": False,
    "warmup_s": 120.0,
    "scored_s": 600.0,
}
DEFAULT_FIELDS = (
    "surge_mult",
    "surge_radius_m",
    "surge_loc",
    "surge_onset_frac",
    "surge_ramp_frac",
    "surge_hold_frac",
    "surge_speed_mps",
    "frac_ll",
    "frac_embb",
    "embb_files_per_s",
    "be_files_per_s",
    "ramp",
    "m_tau_s",
    "m_sigma",
    "kpm_delay_s",
    "kpm_drop",
    "pl_offset_db",
    "xapp_threshold_scale",
)
PAIRS: dict[str, dict[str, Any]] = {
    "P1": {
        "idx": 1,
        "mix": "ES",
        "p_xapps": ("SliceGuarantee",),
        "A": ("ES",),
        "B": ("SliceGuarantee",),
    },
    "P2": {
        "idx": 2,
        "mix": "none",
        "p_xapps": ("PowerES", "Coverage"),
        "A": ("PowerES",),
        "B": ("Coverage",),
    },
    "P3": {
        "idx": 3,
        "mix": "ES",
        "p_xapps": ("PowerES", "SliceGuarantee"),
        "A": ("ES", "PowerES"),
        "B": ("SliceGuarantee",),
    },
}
STRATA = (("base", "L10"), ("base", "L40"), ("surge", "L10"), ("surge", "L40"))
TARGET_L = {"L10": 0.10, "L40": 0.40}
LOAD_FACTOR = {"L10": 0.7875, "L40": 1.7554687500000001}
PAIR, STRATUM = "P3", 3
SMOKE_S = 20.0


def deployed(pair: str) -> tuple[str, ...]:
    """Returns the xApps deployed in a pair."""
    p = PAIRS[pair]
    return (("ES",) if p["mix"] == "ES" else ()) + p["p_xapps"]


def _e6p_bad(p: Any) -> dict[str, tuple[Any, Any]]:
    bad = {k: (getattr(p, k), v) for k, v in FROZEN_E6P.items() if getattr(p, k) != v}
    if not (p.ptx_on and p.prot_on):
        bad["ptx_on/prot_on"] = (p.ptx_on, p.prot_on)
    if p.prot_floor_bps != C.EMBB_THP_TARGET_BPS or p.prot_min_backlog_s != 0.2:
        bad["nonprot_embb_identity"] = (p.prot_floor_bps, C.EMBB_THP_TARGET_BPS)
    return bad


def check_frozen(cfg: C.E6Config, smoke: bool = False) -> None:
    """Checks a plant config against the frozen E6-P values.

    Raises:
        ValueError: A field differs from the frozen value (warm-up and scored time are free
            under `smoke`).
    """
    bad = _e6p_bad(cfg.e6p)
    for k, v in E6_FROZEN.items():
        if smoke and k in ("warmup_s", "scored_s"):
            continue
        if getattr(cfg, k) != v:
            bad[k] = (getattr(cfg, k), v)
    ref = C.E6Config()
    for f in DEFAULT_FIELDS:
        if getattr(cfg, f) != getattr(ref, f):
            bad[f] = (getattr(cfg, f), getattr(ref, f))
    if bad:
        raise ValueError(f"config differs from the frozen E6P values: {bad}")


def make_cfg(pair: str, stratum: int, seed: int, lf: float, smoke: bool = False) -> C.E6Config:
    """Returns the frozen plant config of a pair ("*" for no xApps) and stratum.

    Args:
        pair: "P1", "P2", "P3" or "*".
        stratum: Index into `STRATA`.
        seed: Plant seed.
        lf: Load factor.
        smoke: Shorten warm-up and scored time to `SMOKE_S` each.
    """
    if pair == "*":
        mix, px = "none", ()
    else:
        mix, px = PAIRS[pair]["mix"], PAIRS[pair]["p_xapps"]
    kw = dict(E6_FROZEN)
    if smoke:
        kw.update(warmup_s=SMOKE_S, scored_s=SMOKE_S)
    cfg = C.E6Config(
        seed=int(seed),
        mix=mix,
        scenario=STRATA[stratum][0],
        load_factor=float(lf),
        e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=px),
        **kw,
    )
    check_frozen(cfg, smoke)
    return cfg

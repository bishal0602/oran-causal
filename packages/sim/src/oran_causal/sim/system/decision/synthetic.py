"""Synthetic episode records with a planted unit effect, for mechanism tests.

Cells sit on a ring with N(c) = {c - 1, c, c + 1}. The per-cell series are an AR(1) load around
20 UEs, Poisson pv, v and rlf, and energy 1000 J plus noise. Units of the four families open on
random cells and seconds in [120, T - H); modes are drawn from `PI0_HIGH_NO_RB` (accept applied
on placebo records). A unit of `family` adds effect x level x sgn / H per second to the load and
pv of each neighbour, and subtracts it from its own load, over its window; `far_effect` does the
same on the far cells. ptx units carry a fixed nuisance effect on their own energy. The records
have no privileged ground-truth fields.
"""

import dataclasses
from typing import Any

import numpy as np

from oran_causal.sim.system.decision.collect import PI0_HIGH_NO_RB
from oran_causal.sim.system.decision.records import H_UNIT, SCHEMA, SERIES_FIELDS, enc
from oran_causal.sim.system.decision.vocab import LEVEL, MODES

XAPP_OF = {"carrier": "ES", "sleep": "ES", "ptx": "PowerES", "prot_min": "SliceGuarantee"}
FAMILY_P = {"carrier": 0.2, "sleep": 0.35, "ptx": 0.2, "prot_min": 0.25}
_FI = {f: SERIES_FIELDS.index(f) for f in SERIES_FIELDS}


@dataclasses.dataclass(frozen=True)
class SyntheticSpec:
    """Shape and planted effects of synthetic records.

    Attributes:
        n_ep: Episodes.
        units_per_ep: Units per episode.
        n_cells: Cells on the ring.
        effect: Planted neighbour effect of `family` units.
        family: Knob family carrying the effect.
        placebo: Apply accept to every unit while logging the drawn mode.
        T: Episode length in seconds.
        far_effect: Planted far-cell effect of `family` units.
        seed0: Seed of the first episode.
        strong_nuisance: Scale of the ptx nuisance effect on own energy.
    """

    n_ep: int = 8
    units_per_ep: int = 60
    n_cells: int = 8
    effect: float = 0.0
    family: str = "sleep"
    placebo: bool = False
    T: int = 720
    far_effect: float = 0.0
    seed0: int = 1000
    strong_nuisance: float = 1.0


DEFAULT_SPEC = SyntheticSpec()


def _series(rng: np.random.Generator, spec: SyntheticSpec) -> np.ndarray:
    T, n = spec.T, spec.n_cells
    data = np.zeros((T, len(SERIES_FIELDS), n))
    lvl = 20.0 + rng.normal(0, 2, n)
    ar = np.zeros(n)
    for t in range(T):
        ar = 0.95 * ar + rng.normal(0, 0.5, n)
        data[t, _FI["ue"]] = lvl + ar
    data[:, _FI["pv"]] = rng.poisson(0.5, (T, n))
    data[:, _FI["v"]] = rng.poisson(1.0, (T, n))
    data[:, _FI["e"]] = 1000.0 + rng.normal(0, 5.0, (T, n))
    data[:, _FI["rlf"]] = rng.poisson(0.02, (T, n))
    return data


def _plant(data: np.ndarray, spec: SyntheticSpec, unit: dict[str, Any], lev: float) -> None:
    n, c, f = spec.n_cells, unit["c"], unit["knob"]
    exp = unit["exp"]
    w = slice(int(unit["t0"]), int(unit["t0"]) + H_UNIT)
    nb = [q for q in exp if q != c]
    far = [q for q in range(n) if q not in exp]
    if f == spec.family and spec.effect:
        for q in nb:
            data[w, _FI["ue"], q] += spec.effect * lev / H_UNIT
            data[w, _FI["pv"], q] += spec.effect * lev / H_UNIT
        data[w, _FI["ue"], c] -= spec.effect * lev / H_UNIT
    if f == spec.family and spec.far_effect:
        for q in far:
            data[w, _FI["ue"], q] += spec.far_effect * lev / H_UNIT
    if f == "ptx" and spec.strong_nuisance:
        data[w, _FI["e"], c] += spec.strong_nuisance * 50.0 * lev / H_UNIT


def _unit(rng: np.random.Generator, spec: SyntheticSpec, i: int, t0: int) -> dict[str, Any]:
    fams = list(FAMILY_P)
    f = fams[int(rng.choice(len(fams), p=[FAMILY_P[k] for k in fams]))]
    x = XAPP_OF[f]
    n = spec.n_cells
    c = int(rng.integers(n))
    exp = sorted({(c - 1) % n, c, (c + 1) % n})
    tab = PI0_HIGH_NO_RB[x]
    ms = [m for m in MODES if m in tab]
    m = ms[int(rng.choice(len(ms), p=[tab[k] for k in ms]))]
    step = float(rng.choice([-1.0, 1.0]))
    ctx = {
        "knob": f,
        "is_macro": float(f in ("carrier", "ptx")),
        "cur": 0.0,
        "prop": step,
        "step": step,
        "since_change": float(rng.integers(0, 300)),
        "n_exp": len(exp),
        "own_prb_util": float(rng.uniform(0.2, 0.9)),
        "nbr_mean_prb_util": float(rng.uniform(0.2, 0.9)),
    }
    return {
        "i": i,
        "c": c,
        "x": x,
        "x_idx": ("ES", "PowerES", "SliceGuarantee").index(x),
        "knob": f,
        "t0": float(t0),
        "mode": m,
        "p": float(tab[m]),
        "applied_mode": "accept" if spec.placebo else m,
        "cur": 0.0,
        "prop": step,
        "step": step,
        "exp": exp,
        "n_req": 1,
        "n_applied": 1,
        "rb_issued": False,
        "ctx": ctx,
        "lab_kpi": None,
    }


def _episode(rng: np.random.Generator, spec: SyntheticSpec, e: int) -> dict[str, Any]:
    data = _series(rng, spec)
    units = []
    t0s = np.sort(rng.integers(120, spec.T - H_UNIT, spec.units_per_ep))
    for i, t0 in enumerate(t0s):
        u = _unit(rng, spec, i, int(t0))
        lev = LEVEL[u["applied_mode"]] * (np.sign(u["step"]) or 1.0)
        _plant(data, spec, u, lev)
        units.append(u)
    seed = spec.seed0 + e
    return {
        "kind": "episode",
        "schema": SCHEMA,
        "key": ["dev", "", seed],
        "stage": "dev",
        "sub": "",
        "seed": seed,
        "cfg_seed": seed,
        "j": e,
        "fold": e % 3,
        "smoke": False,
        "policy": "placebo" if spec.placebo else "pi0",
        "pi0_table": PI0_HIGH_NO_RB,
        "T": 60.0,
        "n_cells": spec.n_cells,
        "units": units,
        "lab_series": {
            "fields": list(SERIES_FIELDS),
            "t_first": 1,
            "scored": enc((np.arange(spec.T) >= 120).astype(float)),
            "data": enc(data),
        },
    }


def synthetic_records(
    rng: np.random.Generator, spec: SyntheticSpec = DEFAULT_SPEC
) -> list[dict[str, Any]]:
    """Returns synthetic records of schema `SCHEMA` (module docstring)."""
    return [_episode(rng, spec, e) for e in range(spec.n_ep)]

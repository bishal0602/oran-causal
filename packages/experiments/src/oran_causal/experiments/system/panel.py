"""Per-cell observation panel of one system-level episode, as stored in its record (`panel`).

One row per (10 s step, cell), step-major. Columns read only runtime-observable trace arrays
(delivered KPM reports, logged requests, the configuration in force); the `lab_*` labels never
enter. Kinds:

- `knob_own`: the cell's configuration at the start of the step (hys, ttt, ll_ratio, carrier
  share, sleep, outgoing CIO mean / max, ptx, prot_min).
- `knob_nbr`: neighbours' knobs acting on the cell (incoming CIO mean; carrier, sleep, ptx,
  prot_min means over CIO neighbours).
- `state_nbr`: neighbour PRB utilisation and active UEs over the previous step.
- `kpi`, `kpi_lag`: KPM families over the step (finer reports averaged, coarser ones repeated)
  and the same family one report period earlier.
- `context`: `is_macro`.

Neighbours are the cells linked by any CIO knob. Columns are never dropped here (the records
keep every column; degeneracy is judged by consumers on pooled rows).
"""

import dataclasses
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.sim.system.config import MACRO_NTRX
from oran_causal.sim.system.decision.records import enc
from oran_causal.sim.system.trace import LABEL_PREFIX, Trace

PANEL_SCHEMA = "e6p-panel/1"
KPI_FAMILIES = (
    "prb_util",
    "ll_delay_p95",
    "embb_thp_p5",
    "rlf",
    "too_late",
    "too_early",
    "energy_w",
)
KPI_FAMILIES_PX = ("prot_viol", "prot_act_ue", "edge_sinr_p")
KPI_GRAN_S = {
    "prb_util": 1,
    "ll_delay_p95": 1,
    "embb_thp_p5": 5,
    "rlf": 30,
    "too_late": 30,
    "too_early": 30,
    "energy_w": 60,
    "prot_viol": 1,
    "prot_act_ue": 1,
    "edge_sinr_p": 1,
}
KNOB_COLUMNS = {
    "own_hys": ("knob_own", "hys"),
    "own_ttt": ("knob_own", "ttt"),
    "own_ll_ratio": ("knob_own", "ll_ratio"),
    "own_carrier": ("knob_own", "carrier"),
    "own_sleep": ("knob_own", "sleep"),
    "cio_out_mean": ("knob_own", "cio"),
    "cio_out_max": ("knob_own", "cio"),
    "cio_in_mean": ("knob_nbr", "cio"),
    "nbr_carrier": ("knob_nbr", "carrier"),
}
KNOB_COLUMNS_P = {
    "own_ptx": ("knob_own", "ptx"),
    "own_prot_min": ("knob_own", "prot_min"),
    "nbr_sleep": ("knob_nbr", "sleep"),
    "nbr_ptx": ("knob_nbr", "ptx"),
    "nbr_prot_min": ("knob_nbr", "prot_min"),
}

type Arrays = Mapping[str, Any]
type Field = Callable[[int], np.ndarray]


@dataclasses.dataclass
class Panel:
    """One episode's panel; `data[col]` is (n,) float64 in step-major, then cell, order."""

    data: dict[str, np.ndarray]
    kind: dict[str, str]
    family: dict[str, str]
    episode: np.ndarray
    t: np.ndarray
    cell: np.ndarray
    step_s: int
    neighbours: list[list[int]]
    ownership: dict[str, dict[str, dict[str, int]]]
    xapps: list[str]
    dropped: dict[str, str] = dataclasses.field(default_factory=dict)
    notes: dict[str, str] = dataclasses.field(default_factory=dict)

    def put(self, name: str, arr: np.ndarray, kind: str, family: str) -> None:
        """Adds column `name` from a (steps, cells) array."""
        self.data[name] = np.asarray(arr, float).reshape(-1)
        self.kind[name], self.family[name] = kind, family


def neighbours_from_knobs(knobs: Sequence[tuple], n_cells: int) -> list[list[int]]:
    """Returns the cells linked to each cell by a CIO knob in either direction."""
    nb: list[set[int]] = [set() for _ in range(n_cells)]
    for k in knobs:
        if k[0] == "cio":
            s, n = int(k[1]), int(k[2])
            nb[s].add(n)
            nb[n].add(s)
    return [sorted(x) for x in nb]


def knob_ownership(trace: Trace) -> dict[str, dict[str, dict[str, int]]]:
    """Returns xApp -> knob family -> requests offered and applied, from the request log."""
    a, meta = trace.arrays, trace.meta
    ok = meta["out_codes"].index("ok")
    own: dict[str, dict[str, dict[str, int]]] = {x: {} for x in meta["xapps"]}
    for x, k, o in zip(a["rq_xapp"], a["rq_knob"], a["rq_out"], strict=True):
        fam = meta["knobs"][int(k)][0]
        d = own[meta["xapps"][int(x)]].setdefault(fam, {"req": 0, "applied": 0})
        d["req"] += 1
        d["applied"] += int(o == ok)
    return own


def _kpi_windows(
    a: Arrays, gran: str, tk: np.ndarray, step: int, n_cells: int, fields: Mapping[str, Field]
) -> dict[str, np.ndarray]:
    out = {name: np.full((len(tk), n_cells), np.nan) for name in fields}
    key = f"kpm_{gran}_t0"
    if key not in a or len(a[key]) == 0:
        return out
    t0, t1 = a[key], a[f"kpm_{gran}_t1"]
    for i, tt in enumerate(tk):
        inside = np.nonzero((t0 >= tt) & (t1 <= tt + step))[0]
        if len(inside) == 0:
            inside = np.nonzero((t0 <= tt) & (t1 >= tt + step))[0][:1]
        if len(inside) == 0:
            continue
        for name, fn in fields.items():
            vals = np.stack([fn(j) for j in inside])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[name][i] = np.nanmean(vals, 0)
    return out


@dataclasses.dataclass(frozen=True)
class _Grid:
    a: Arrays
    kidx: dict[tuple, int]
    tk: np.ndarray
    cfg: np.ndarray
    n_cells: int
    step: int
    nb: list[list[int]]

    @property
    def K(self) -> int:
        return len(self.tk)

    def per_cell(self, fam: str, default: float) -> tuple[np.ndarray, bool]:
        idx = [self.kidx.get((fam, c), -1) for c in range(self.n_cells)]
        out = np.full((self.K, self.n_cells), float(default))
        for c, i in enumerate(idx):
            if i >= 0:
                out[:, c] = self.cfg[:, i]
        return out, any(i >= 0 for i in idx)

    def nbr_mean(self, v: np.ndarray) -> np.ndarray:
        if not self.K:
            return np.zeros((0, self.n_cells))
        nan = np.full(self.K, np.nan)
        return np.stack([v[:, n].mean(1) if n else nan for n in self.nb], 1)

    def lag(self, v: np.ndarray, L: int) -> np.ndarray:
        out = np.full((self.K, self.n_cells), np.nan)
        if self.K > L:
            out[L:] = v[:-L]
        return out

    def windows(self, gran: str, fields: Mapping[str, Field]) -> dict[str, np.ndarray]:
        return _kpi_windows(self.a, gran, self.tk, self.step, self.n_cells, fields)


def _grid(trace: Trace, step_s: int, t_min: int) -> _Grid:
    a = trace.features()
    if any(k.startswith(LABEL_PREFIX) for k in a):
        raise ValueError("trace features carry label arrays")
    knobs = [tuple(k) for k in trace.meta["knobs"]]
    n_cells = len(trace.meta["cell_region"])
    tk = np.arange(int(t_min), int(a["t"][-1]) - int(step_s) + 1, int(step_s))
    cfg = (
        np.stack([trace.config_at(int(t) + 1) for t in tk])
        if len(tk)
        else np.zeros((0, len(knobs)))
    )
    return _Grid(
        a=a,
        kidx={k: i for i, k in enumerate(knobs)},
        tk=tk,
        cfg=cfg,
        n_cells=n_cells,
        step=int(step_s),
        nb=neighbours_from_knobs(knobs, n_cells),
    )


def _knob_columns(g: _Grid, is_macro: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    cols, have = {}, {}
    cols["own_hys"], have["own_hys"] = g.per_cell("hys", np.nan)
    cols["own_ttt"], have["own_ttt"] = g.per_cell("ttt", np.nan)
    cols["own_ll_ratio"], have["own_ll_ratio"] = g.per_cell("ll_ratio", np.nan)
    car, have["own_carrier"] = g.per_cell("carrier", MACRO_NTRX)
    cols["own_carrier"] = np.where(is_macro[None, :], car / MACRO_NTRX, 1.0)
    cols["own_sleep"], have["own_sleep"] = g.per_cell("sleep", 0.0)
    C = g.n_cells
    k = g.kidx
    cio_out = [[k[("cio", c, n)] for n in range(C) if ("cio", c, n) in k] for c in range(C)]
    cio_in = [[k[("cio", s, c)] for s in range(C) if ("cio", s, c) in k] for c in range(C)]
    nan = np.full(g.K, np.nan)
    cols["cio_out_mean"] = np.stack([g.cfg[:, i].mean(1) if i else nan for i in cio_out], 1)
    cols["cio_out_max"] = np.stack([g.cfg[:, i].max(1) if i else nan for i in cio_out], 1)
    cols["cio_in_mean"] = np.stack([g.cfg[:, i].mean(1) if i else nan for i in cio_in], 1)
    have.update(cio_out_mean=True, cio_out_max=True, cio_in_mean=True)
    cols["nbr_carrier"] = np.stack(
        [cols["own_carrier"][:, n].mean(1) if n else nan for n in g.nb], 1
    )
    have["nbr_carrier"] = have["own_carrier"]
    return cols, have


def _mob_rate(a: Arrays, names: Sequence[str]) -> Field:
    def fn(j: int) -> np.ndarray:
        win = float(a["kpm_mob_t1"][j] - a["kpm_mob_t0"][j])
        v = sum(np.asarray(a[f"kpm_mob_{x}"][j], float) for x in names)
        v = np.asarray(v)
        return (v.sum(1) if v.ndim == 2 else v) / win

    return fn


def _base_kpis(g: _Grid) -> dict[str, np.ndarray]:
    a = g.a
    fast = g.windows(
        "fast",
        {
            "prb_util": lambda j: a["kpm_fast_prb_util"][j],
            "ll_delay_p95": lambda j: a["kpm_fast_ll_delay_p95"][j],
        },
    )
    thp = g.windows("thp", {"embb_thp_p5": lambda j: a["kpm_thp_embb_thp_p5"][j]})
    mob = g.windows(
        "mob",
        {
            "rlf": _mob_rate(a, ["rlf"]),
            "too_late": _mob_rate(a, ["too_late"]),
            "too_early": _mob_rate(a, ["too_early", "wrong_cell", "pingpong"]),
        },
    )
    en = g.windows(
        "energy",
        {
            "energy_w": lambda j: (
                np.asarray(a["kpm_energy_energy_j"][j], float)
                / float(a["kpm_energy_t1"][j] - a["kpm_energy_t0"][j])
            )
        },
    )
    return {**fast, **thp, **mob, **en}


def _lag_steps(fam: str, step: int) -> int:
    return max(1, math.ceil(KPI_GRAN_S[fam] / step))


def _base_panel(trace: Trace, g: _Grid, episode: int) -> Panel:
    C, K = g.n_cells, g.K
    is_macro = np.array([("carrier", c) in g.kidx for c in range(C)])
    cols, have = _knob_columns(g, is_macro)
    kpis = _base_kpis(g)
    lags = {f + "_lag": g.lag(kpis[f], _lag_steps(f, g.step)) for f in KPI_FAMILIES}
    nan = np.full(K, np.nan)
    util_lag = lags["prb_util_lag"]
    nbr_util = np.stack([util_lag[:, n].mean(1) if n else nan for n in g.nb], 1)
    p = Panel(
        data={},
        kind={},
        family={},
        episode=np.full(K * C, int(episode)),
        t=np.repeat(g.tk, C).astype(np.int64),
        cell=np.tile(np.arange(C), K),
        step_s=g.step,
        neighbours=g.nb,
        ownership=knob_ownership(trace),
        xapps=list(trace.meta["xapps"]),
    )
    for name, (k, fam) in KNOB_COLUMNS.items():
        if have.get(name, True):
            p.put(name, cols[name], k, fam)
        else:
            p.dropped[name] = "knob_absent_in_mix"
    p.put("nbr_util_lag", nbr_util, "state_nbr", "prb_util")
    for fam in KPI_FAMILIES:
        p.put(fam, kpis[fam], "kpi", fam)
    for name, v in lags.items():
        p.put(name, v, "kpi_lag", name[:-4])
    p.put("is_macro", np.broadcast_to(is_macro, (K, C)).astype(float), "context", "is_macro")
    return p


def _p_fields(a: Arrays) -> dict[str, Field]:
    def act_ue(j: int) -> np.ndarray:
        v = np.asarray(a["kpm_fast_act_ue"][j], float)
        return v.sum(1) if v.ndim == 2 else v

    def prot_viol(j: int) -> np.ndarray:
        ev = np.asarray(a["kpm_fast_prot_eval"][j], float)
        fr = np.asarray(a["kpm_fast_prot_below_frac"][j], float)
        return np.where(ev > 0, np.rint(np.nan_to_num(fr) * ev), 0.0)

    def field(name: str) -> Field:
        return lambda j: np.asarray(a[f"kpm_fast_{name}"][j], float)

    fields: dict[str, Field] = {"act_ue": act_ue}
    if "kpm_fast_prot_eval" in a and "kpm_fast_prot_below_frac" in a:
        fields["prot_viol"] = prot_viol
    for name in ("prot_act_ue", "edge_sinr_p"):
        if f"kpm_fast_{name}" in a:
            fields[name] = field(name)
    return fields


def build_panel(trace: Trace, step_s: int = 10, t_min: int = 0, episode: int = 0) -> Panel:
    """Returns the panel of one trace (module docstring); no column is dropped."""
    g = _grid(trace, step_s, t_min)
    p = _base_panel(trace, g, episode)
    own_ptx, have_ptx = g.per_cell("ptx", 0.0)
    own_prot, have_prot = g.per_cell("prot_min", 0.0)
    sleep, have_sleep = g.per_cell("sleep", 0.0)
    cols = {
        "own_ptx": (own_ptx, have_ptx),
        "own_prot_min": (own_prot, have_prot),
        "nbr_sleep": (g.nbr_mean(sleep), have_sleep),
        "nbr_ptx": (g.nbr_mean(own_ptx), have_ptx),
        "nbr_prot_min": (g.nbr_mean(own_prot), have_prot),
    }
    fast = g.windows("fast", _p_fields(g.a))
    for name, (k, fam) in KNOB_COLUMNS_P.items():
        arr, have = cols[name]
        if have:
            p.put(name, arr, k, fam)
        else:
            p.dropped[name] = "knob_absent_in_mix"
    p.put("nbr_act_ue_lag", g.nbr_mean(g.lag(fast["act_ue"], 1)), "state_nbr", "act_ue")
    for fam in KPI_FAMILIES_PX:
        if fam not in fast:
            p.dropped[fam] = "report_field_absent"
            continue
        p.put(fam, fast[fam], "kpi", fam)
        p.put(fam + "_lag", g.lag(fast[fam], _lag_steps(fam, g.step)), "kpi_lag", fam)
    return p


def panel_to_rec(panel: Panel) -> dict[str, Any]:
    """Returns the JSON-able record of a panel (float32 arrays via `enc`), without the site map."""
    return {
        "schema": PANEL_SCHEMA,
        "step_s": int(panel.step_s),
        "n_rows": len(panel.t),
        "n_cells": len(panel.neighbours),
        "episode": enc(panel.episode),
        "t": enc(panel.t),
        "cell": enc(panel.cell),
        "columns": list(panel.data),
        "kind": dict(panel.kind),
        "family": dict(panel.family),
        "data": {c: enc(v) for c, v in panel.data.items()},
        "neighbours": [[int(x) for x in n] for n in panel.neighbours],
        "ownership": panel.ownership,
        "xapps": list(panel.xapps),
        "dropped": dict(panel.dropped),
        "notes": dict(panel.notes),
    }

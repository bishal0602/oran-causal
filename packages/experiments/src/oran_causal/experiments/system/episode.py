"""One collection episode of Study 1 or 2: plant, logging arbiter, knockout labels, record body.

The record format is schema `e6p-disc-rec/1` (`sim.system.decision.records.SCHEMA`): JSON lines
whose arrays are `enc` dicts. Its body holds the obs-only parts (`obs_static`, `units`, `panel`)
and the privileged ones (`lab_series`, `lab_outcome`, `gt_static`, `gt_labels`).
"""

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from oran_causal.experiments.system.common import LOAD_FACTOR_L40, Record, cfg_seed, outcome
from oran_causal.experiments.system.panel import build_panel, panel_to_rec
from oran_causal.sim.system.config import E6Config
from oran_causal.sim.system.decision import gt as G
from oran_causal.sim.system.decision.collect import Collection, run_collection
from oran_causal.sim.system.decision.labels import Labeller, Paused, sampled
from oran_causal.sim.system.decision.records import enc
from oran_causal.sim.system.decision.scenario import PAIR, STRATUM, make_cfg
from oran_causal.sim.system.decision.units import Policy, Unit
from oran_causal.sim.system.decision.vocab import CELL_KPIS
from oran_causal.sim.system.env import E6Env
from oran_causal.sim.system.trace import Trace

STEP_S = 10
OPEN_RULE = "feasible"
ARB_WARMUP_S = 0.0
COUNT_ALL = True
GT_T0_MIN = 90

type UnitRecord = Callable[[int, Mapping[str, Any], int], Record]


def gt_consts() -> dict[str, Any]:
    """The knockout-label settings recorded in run headers."""
    return {
        "ks": G.GT_KS,
        "H": G.GT_H,
        "modes": G.GT_MODES,
        "rate": G.GT_RATE,
        "t0_min": GT_T0_MIN,
        "kpis": CELL_KPIS,
    }


@dataclasses.dataclass
class GtHook:
    """Labels the sampled units of a GT episode as they open (`run_collection`'s labeller).

    Units of a family in `GT_RATE` are sampled with tag 6613; units opening before `GT_T0_MIN`,
    too late for a full window, or over `max_labels` are counted and skipped.
    """

    seed: int
    labeller: Labeller
    max_labels: int | None = None
    counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict.fromkeys(
            ("labels", "skipped_end", "skipped_cap", "skipped_early"), 0
        )
    )

    def __call__(self, env: E6Env, obs: Mapping[str, Any], snap: Any, opened: list[Unit]) -> None:
        """Labels the sampled units among those opened this second."""
        for u in opened:
            rate = G.GT_RATE.get(u["knob"])
            if rate is None or not sampled(self.seed, u, rate):
                continue
            if u["t0"] < GT_T0_MIN:
                self.counts["skipped_early"] += 1
            elif u["t0"] + G.GT_H > env.total_s:
                self.counts["skipped_end"] += 1
            elif self.max_labels is not None and self.counts["labels"] >= self.max_labels:
                self.counts["skipped_cap"] += 1
            else:
                u["_gt"] = self.labeller.label(Paused(env, obs, snap), u, modes=G.GT_MODES)
                self.counts["labels"] += 1


def n_applied(trace: Trace, units: Sequence[Mapping[str, Any]], T: float) -> list[int]:
    """Per unit: its xApp's requests on its knob inside the unit window that changed the knob."""
    a, meta = trace.arrays, trace.meta
    kidx = {tuple(k): i for i, k in enumerate(meta["knobs"])}
    xidx = {x: i for i, x in enumerate(meta["xapps"])}
    ok = meta["out_codes"].index("ok")
    t, xs, ks, out = a["rq_t"], a["rq_xapp"], a["rq_knob"], a["rq_out"]
    res = []
    for u in units:
        k = kidx.get((u["knob"], int(u["c"])), -1)
        m = (
            (xs == xidx.get(u["x"], -1))
            & (ks == k)
            & (t >= u["t0"])
            & (t < u["t0"] + T)
            & (out == ok)
        )
        res.append(int(m.sum()))
    return res


def gt_label_record(i: int, u: Mapping[str, Any]) -> Record:
    """The `gt_labels` entry of labelled unit `u` at position `i`."""
    L = u["_gt"]
    return {
        "i": i,
        "c": int(u["c"]),
        "x": u["x"],
        "knob": u["knob"],
        "t0": float(u["t0"]),
        "step": u["ctx"]["step"],
        "exp": [int(v) for v in u["exp"]],
        "ks": L["ks"],
        "H": L["H"],
        "kpis": L["cell_kpis"],
        "delta": enc(G.unit_delta(L)),
        "d_net_reject": L["d"]["reject"]["net"],
        "secs": [r["secs"] for r in L["raw"]["reject"]],
        "cpu_s": L["cpu_s"],
    }


def family_mode_counts(urecs: Sequence[Record], key: Callable[[Record], str]) -> dict:
    """Unit counts by `key(unit)` and logged mode."""
    by: dict[str, dict[str, int]] = {}
    for u in urecs:
        b = by.setdefault(key(u), {})
        b[u["mode"]] = b.get(u["mode"], 0) + 1
    return by


def sleep_counts(urecs: Sequence[Record]) -> dict[str, int]:
    """Sleep units and their logged rejects."""
    sl = [u for u in urecs if u["knob"] == "sleep"]
    return {"sleep_units": len(sl), "sleep_rejects": sum(u["mode"] == "reject" for u in sl)}


def gt_static(env: E6Env) -> Record:
    """Privileged layout truth: ES candidate macros, cell sites, plant neighbours, cell class."""
    lay = env.plant.lay
    es = next((x for x in env.xapps if x.name == "ES"), None)
    cand = getattr(es, "cand", {}) if es is not None else {}
    return {
        "cand": {str(k): int(v) for k, v in cand.items()},
        "cell_site": [int(v) for v in lay.cell_site],
        "neighbours": [[int(v) for v in n] for n in lay.neighbours],
        "is_macro": [bool(v) for v in lay.is_macro],
    }


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """What one collection episode runs: plant seed, policy, arbiter class, labels.

    Attributes:
        seed: Protocol seed.
        smoke: Plumbing run (DEV plant seed).
        policy: Logging policy factory, called with the plant seed.
        collection: Arbiter settings.
        gt: Label units (knockout ground truth).
        short: Scored seconds under smoke.
        max_labels: Label cap under smoke.
    """

    seed: int
    smoke: bool
    policy: Callable[[int], Policy]
    collection: Collection
    gt: bool = False
    short: float | None = None
    max_labels: int | None = None

    @property
    def cfg_seed(self) -> int:
        """The plant seed."""
        return cfg_seed(self.seed, self.smoke)

    def cfg(self) -> E6Config:
        """The frozen study plant config (scored time shortened under smoke)."""
        cfg = make_cfg(PAIR, STRATUM, self.cfg_seed, LOAD_FACTOR_L40)
        if self.smoke and self.short:
            cfg = dataclasses.replace(cfg, scored_s=float(self.short))
        return cfg


@dataclasses.dataclass
class Collected:
    """A finished collection episode: its env, cfg, raw result, label hook and timings."""

    env: E6Env
    cfg: E6Config
    collection: Collection
    res: dict[str, Any]
    hook: GtHook | None
    panel: Record
    panel_cpu: float
    urecs: list[Record]


def collect(spec: EpisodeSpec, unit_record: UnitRecord) -> Collected:
    """Runs one collection episode and builds its unit records and panel."""
    cfg = spec.cfg()
    env = E6Env(cfg, log=False, wg3=True, trace=True)
    hook = None
    if spec.gt:
        lab = Labeller(ks=G.GT_KS, H=G.GT_H, T=spec.collection.T, cells=True)
        hook = GtHook(spec.cfg_seed, lab, spec.max_labels)
    res = run_collection(cfg, spec.policy(spec.cfg_seed), spec.collection, hook, env)
    t_p = time.process_time()
    trace = env.get_trace()
    panel = panel_to_rec(build_panel(trace, step_s=STEP_S, episode=spec.seed))
    panel_cpu = time.process_time() - t_p
    units = res["units"]
    n_app = n_applied(trace, units, spec.collection.T)
    urecs = [unit_record(i, u, n_app[i]) for i, u in enumerate(units)]
    return Collected(env, cfg, spec.collection, res, hook, panel, panel_cpu, urecs)


def body(c: Collected) -> Record:
    """The record fields shared by every collection episode (plant, panel, labels, CPU)."""
    ser = c.res["tap"].series()
    st = c.env.static()
    label_cpu = c.hook.labeller.cpu_s if c.hook is not None else 0.0
    cpu = c.res["cpu_s"]
    return {
        "T": c.collection.T,
        "open_rule": OPEN_RULE,
        "load_factor": LOAD_FACTOR_L40,
        "episode_s": c.env.total_s,
        "warmup_s": c.cfg.warmup_s,
        "n_cells": int(c.env.plant.nc),
        "step_s": STEP_S,
        "obs_static": {
            "is_macro": [bool(v) for v in st["is_macro"]],
            "neighbours": [[int(v) for v in n] for n in st["neighbours"]],
        },
        "units": c.urecs,
        "panel": c.panel,
        "lab_series": {
            "fields": ser["fields"],
            "t_first": 1,
            "scored": enc(ser["scored"]),
            "data": enc(ser["data"]),
        },
        "lab_outcome": outcome(c.env),
        "gt_static": gt_static(c.env),
        "gt_labels": [
            gt_label_record(i, u) for i, u in enumerate(c.res["units"]) if u.get("_gt") is not None
        ],
        "cpu_s": round(cpu + c.panel_cpu, 2),
        "collect_cpu_s": round(cpu - label_cpu, 2),
        "label_cpu_s": round(label_cpu, 2),
        "panel_cpu_s": round(c.panel_cpu, 2),
        "n_roll": c.hook.labeller.n_roll if c.hook is not None else 0,
    }


def label_counts(c: Collected) -> dict[str, int]:
    """The label hook's counters (zeros without labels)."""
    if c.hook is not None:
        return dict(c.hook.counts)
    return dict.fromkeys(("labels", "skipped_end", "skipped_cap", "skipped_early"), 0)

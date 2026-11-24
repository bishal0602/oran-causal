"""X19: compute and logging cost of the simulator PMRT and the referee (posthoc PROTOCOL, X19).

Timed on stored records only; nothing simulates the plant. Every timing unit runs in a worker
process with one thread per numerical library, one worker per vCPU (as X1), and is stamped with
the host, load, code commit and package versions, so the records convert with X1's speed factors.

* `disc`: the frozen simulator PMRT (artifact `study1_pmrt.json`, B 9999) on a stored discovery
  unit table: feature build (`pmrt_data`), test (`run_integrated`) and declaration (`declare_all`),
  each timed apart, after a separately timed pool load. Study 1 (uniform-randomised): the pooled
  600 episodes and the first 60-, 120- and 300-episode slices; Study 2 (state-dependent): the
  pooled 600 DISC episodes. The primary hypothesis table must equal the stored one exactly.
* `fit`: the training fit (`fit_pmrt`, steps 1-7, on the 480 ev2 plus 20 DEV episodes) and the
  step-8 prior run (`run_integrated` of the frozen parameters on the 480 ev2 episodes). The fitted
  parameters and the prior z are compared with the frozen artifact, for information.
* `referee`: per-decision latency of MapGateV2(MG:PMRT) and CertSafeMapGateV2(CS:PMRT) on every
  stored Study 2 DISC unit context, and the time of one gate certification.
* `logging`: units per episode-hour by class and policy, the size of a unit record and of the
  KPI series, and a per-cell-day projection for the simulated scenario.

    python -m oran_causal.experiments.posthoc.x19 units
    python -m oran_causal.experiments.posthoc.x19 run --out-dir D --s1 DIR --s2 DIR --train DIR
        [--raw DIR ...] [--procs N] [--reps 3] [--only KIND ...]
    python -m oran_causal.experiments.posthoc.x19 launch --name NAME [--dry-run]

`--s1`, `--s2` and `--train` are the stored job directories of the Study 1 discovery analysis
(`e6p-disc-v4an-pmrt-1`), the Study 2 discovery analysis (`e6p-conf-discan-2`) and the training
bench (`disc-bench-1`); each holds `out/cache/<stage>/*.npz` (training: `out/cache/*.npz`) and,
for the analyses, the stored `out/analysis_v4.json` / `out/disc_conf.json`. `--raw` adds raw
episode record directories for the series volume. The summary is
`oran_causal.analysis.posthoc.x19`.
"""

import argparse
import contextlib
import dataclasses
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.pmrt.units import PmrtConfig, fit_pmrt
from oran_causal.experiments.campaign.host import PLATFORM_ENV, cpu_quota, host_info, load_info
from oran_causal.experiments.campaign.provenance import (
    COMMIT_ENV,
    DIRTY_ENV,
    LOCK_ENV,
    code_info,
    pkg_versions,
)
from oran_causal.experiments.harness.isolation import PeakRSS
from oran_causal.experiments.harness.records import append_record, clean
from oran_causal.experiments.launch import kaggle
from oran_causal.experiments.launch.commands import LOCK_REL, THREADS, SetupPlan, setup_cmd
from oran_causal.experiments.launch.runner import Dirs
from oran_causal.experiments.paths import ARTIFACTS
from oran_causal.experiments.posthoc.x1_timing import THREAD_ENV
from oran_causal.experiments.system import study1_discovery as s1
from oran_causal.methods.system import artifact as pmrt_artifact
from oran_causal.methods.system import pmrt as pm
from oran_causal.methods.system import referees as rf
from oran_causal.methods.system.mscr_crt import SPLIT_POOLED
from oran_causal.sim.system.decision import records
from oran_causal.sim.system.decision.vocab import FAMILIES, KPIS

VERSION = "x19-cost/1"
PROTOCOL = "protocols/posthoc/PROTOCOL.md 2026-11-24: X19"
B = 9999
REPS = 3
KINDS = ("disc", "fit", "referee", "logging")
S1_SLICES = ("pooled", "s60_0", "s120_0", "s300_0")
S2_SLICES = ("pooled",)
TRAIN_SUBS = ("", "v2")
PRIOR_SUB = "v2"
CERTIFY_CALLS = 200
SERIES_EPISODES = 5
SOURCES = {
    "s1": "e6p-disc-v4an-pmrt-1",
    "s2": "e6p-conf-discan-2",
    "train": "disc-bench-1",
}
RAW_SOURCES = ("e6p-disc-v4ev-1-a", "e6p-conf-disc-1-a")
EST_COST_S = {
    "disc|s1|pooled": 150.0,
    "disc|s1|s300_0": 90.0,
    "disc|s1|s120_0": 60.0,
    "disc|s1|s60_0": 50.0,
    "disc|s2|pooled": 150.0,
    "fit|train|all": 400.0,
    "referee|s2|pooled": 120.0,
    "logging|all|all": 120.0,
}
UNIT_LAYOUT = (
    ("cell", 2),
    ("xapp", 1),
    ("family", 1),
    ("t0_s", 4),
    ("mode", 1),
    ("propensity", 4),
    ("direction", 1),
)
F32 = 4
WORKER_MODULE = "oran_causal.experiments.posthoc.x19"
WORK_DIR = Path(tempfile.gettempdir()) / "oran-causal-launch"
WALL_S = 4 * 3600

type Record = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Inputs:
    """Stored job directories read by the measurements.

    Attributes:
        s1: Study 1 discovery analysis job (`out/cache/{eval,placebo}`, `out/analysis_v4.json`).
        s2: Study 2 discovery analysis job (`out/cache/{disc,placebo}`, `out/disc_conf.json`).
        train: Training bench job (`out/cache/*.npz`).
        raw: Raw episode record directories or files for the series volume.
    """

    s1: Path
    s2: Path
    train: Path
    raw: tuple[Path, ...] = ()


@dataclasses.dataclass(frozen=True)
class Frozen:
    """The frozen PMRT: parameters, layer priors and statistic settings.

    Attributes:
        params: Parameters usable by `run_integrated`.
        priors: Layer prior per statistic.
        cfg: Statistic settings with the timed B.
        artifact: Artifact status (path, sha256, verification), or a test label.
    """

    params: Mapping[str, Any]
    priors: Mapping[str, Any]
    cfg: PmrtConfig
    artifact: Mapping[str, Any]


def frozen_artifact(path: Path = s1.ARTIFACT, b: int = B) -> Frozen:
    """The frozen Study 1 artifact with `b` redraws."""
    params, priors = pmrt_artifact.load_artifact(path)
    return Frozen(params, priors, pmrt_artifact.pmrt_config(params, b), s1.artifact_status(path))


@dataclasses.dataclass(frozen=True)
class Unit:
    """One timing unit.

    Attributes:
        kind: One of `KINDS`.
        study: `s1`, `s2`, `train` or `all`.
        slice: Slice name, or `all`.
        rep: Repetition index.
    """

    kind: str
    study: str
    slice: str
    rep: int

    @property
    def cell(self) -> str:
        """The unit without its repetition."""
        return f"{self.kind}|{self.study}|{self.slice}"

    @property
    def key(self) -> str:
        """The record key."""
        return f"{self.cell}|r{self.rep}"

    @property
    def est_cost_s(self) -> float:
        """Estimated cost in reference seconds, used to balance the processes."""
        return EST_COST_S.get(self.cell, 60.0)

    def to_json(self) -> dict[str, Any]:
        """The unit as a JSON object."""
        return dataclasses.asdict(self)


def units(reps: int = REPS, kinds: Sequence[str] = KINDS) -> list[Unit]:
    """Every timing unit: `reps` repetitions of each timed cell, logging once."""
    cells: list[tuple[str, str, str]] = []
    if "disc" in kinds:
        cells += [("disc", "s1", s) for s in S1_SLICES] + [("disc", "s2", s) for s in S2_SLICES]
    if "fit" in kinds:
        cells.append(("fit", "train", "all"))
    if "referee" in kinds:
        cells.append(("referee", "s2", "pooled"))
    out = [Unit(k, st, sl, r) for k, st, sl in cells for r in range(reps)]
    if "logging" in kinds:
        out.append(Unit("logging", "all", "all", 0))
    return out


def assign(us: Sequence[Unit], procs: int) -> list[list[Unit]]:
    """Longest-processing-time first: the costliest unit to the least-loaded process."""
    bins: list[list[Unit]] = [[] for _ in range(max(procs, 1))]
    load = [0.0] * len(bins)
    for u in sorted(us, key=lambda u: (-u.est_cost_s, u.key)):
        i = load.index(min(load))
        bins[i].append(u)
        load[i] += u.est_cost_s
    return bins


@dataclasses.dataclass
class Phases:
    """CPU and wall seconds of the named phases of one unit."""

    cpu_s: dict[str, float] = dataclasses.field(default_factory=dict)
    wall_s: dict[str, float] = dataclasses.field(default_factory=dict)

    @contextlib.contextmanager
    def timed(self, name: str) -> Generator[None]:
        """Adds the CPU and wall time of the block to phase `name`."""
        c0, w0 = time.process_time(), time.perf_counter()
        try:
            yield
        finally:
            self.cpu_s[name] = self.cpu_s.get(name, 0.0) + time.process_time() - c0
            self.wall_s[name] = self.wall_s.get(name, 0.0) + time.perf_counter() - w0

    def to_json(self, compute: Sequence[str]) -> dict[str, Any]:
        """Phases plus the compute totals over `compute`."""
        return {
            "phase_cpu_s": dict(self.cpu_s),
            "phase_wall_s": dict(self.wall_s),
            "cpu_s": sum(self.cpu_s.get(p, 0.0) for p in compute),
            "wall_s": sum(self.wall_s.get(p, 0.0) for p in compute),
        }


def cache_files(job: Path, stage: str | None) -> list[str]:
    """Sorted cache files of a stored job: `out/cache/<stage>/*.npz`, or `out/cache/*.npz`."""
    d = job / "out" / "cache"
    d = d / stage if stage else d
    files = sorted(str(p) for p in d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cache files in {d}")
    return files


def _json(x: Any) -> Any:
    return json.loads(json.dumps(clean(x)))


def _numbers(a: Any, b: Any, path: str, out: list[tuple[str, Any, Any]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _numbers(a.get(k), b.get(k), f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _numbers(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append((path, a, b))


def compare(got: Any, stored: Any, rtol: float = 1e-9) -> dict[str, Any]:
    """Exact and close agreement of two JSON values.

    Returns:
        `exact` (every leaf equal), `close` (numbers within `rtol`, everything else equal),
        `n_diff`, `max_abs` over the differing numbers and the first five differences.
    """
    diffs: list[tuple[str, Any, Any]] = []
    _numbers(_json(got), _json(stored), "", diffs)
    close, max_abs = True, 0.0
    for _, a, b in diffs:
        num = isinstance(a, int | float) and isinstance(b, int | float)
        num = num and not isinstance(a, bool) and not isinstance(b, bool)
        if num and math.isfinite(a) and math.isfinite(b):
            d = abs(float(a) - float(b))
            max_abs = max(max_abs, d)
            close &= d <= rtol * max(abs(float(a)), abs(float(b)), 1e-300)
        else:
            close = False
    return {
        "exact": not diffs,
        "close": close,
        "n_diff": len(diffs),
        "max_abs": max_abs,
        "first": [list(d) for d in diffs[:5]],
    }


def declared(hyp: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    """Sorted primary declarations `[hypothesis, sign]` of a hypothesis table."""
    return sorted(
        [h, (r.get("primary") or {}).get("sign")]
        for h, r in hyp.items()
        if (r.get("primary") or {}).get("declared")
    )


def _stored_s1(job: Path) -> dict[str, Any]:
    a = json.loads((job / "out" / "analysis_v4.json").read_text(encoding="utf-8"))
    return {k: v["hyp"] for k, v in a["slices_result"].items()}


def _stored_s2(job: Path) -> dict[str, Any]:
    a = json.loads((job / "out" / "disc_conf.json").read_text(encoding="utf-8"))
    return {"pooled": a["disc_hyp"]}


def _s1_slices(pool: records.Pool) -> dict[str, s1.Slice]:
    seeds = sorted(int(x) for x in pool.eps["seed"])
    e0, en = s1.SEEDS["eval"]
    pseudo = not (seeds and all(e0 <= x < e0 + en for x in seeds))
    return {sl.name: sl for sl in s1.slice_plan(seeds, pseudo=pseudo)}


def measure_disc(inputs: Inputs, study: str, name: str, frozen: Frozen) -> Record:
    """Times the frozen PMRT on one stored discovery slice and checks it against the record.

    Raises:
        KeyError: Unknown study or slice.
    """
    ph = Phases()
    if study == "s1":
        job, stage, stored = inputs.s1, "eval", _stored_s1(inputs.s1)
    elif study == "s2":
        job, stage, stored = inputs.s2, "disc", _stored_s2(inputs.s2)
    else:
        raise KeyError(study)
    with ph.timed("load"):
        pool = pm.load_pmrt_pool(cache_files(job, stage), stages=["eval"])
    if study == "s1":
        sl = _s1_slices(pool.pool)[name]
        episodes = s1.episodes_of(pool.pool, sl.seeds).tolist()
        split, n_target = sl.split, sl.n_target
    else:
        if name != "pooled":
            raise KeyError(name)
        episodes, split, n_target = None, SPLIT_POOLED, int(pool.pool.n_eps)
    with ph.timed("features"):
        pd = pm.pmrt_data(pool, episodes)
    with ph.timed("test"):
        run = pm.run_integrated(pd, frozen.params, frozen.cfg, split)
    with ph.timed("declare"):
        decls = pm.declare_all(run, frozen.priors, n_target)
    hyp = _json(s1.hyp_table(run, decls[s1.PRIMARY]))
    ref = stored.get(name)
    return {
        **ph.to_json(("features", "test", "declare")),
        "n_episodes": len(np.unique(pd.ud.episode)),
        "n_units": int(pd.n),
        "n_tested": sum(1 for r in hyp.values() if r.get("status") == "tested"),
        "split": int(split),
        "n_target": int(n_target),
        "B": int(frozen.cfg.B),
        "declared": declared(hyp),
        "repro": compare(hyp, ref) if ref is not None else {"exact": False, "missing": True},
        "repro_declared": ref is not None and declared(hyp) == declared(ref),
        "hyp": hyp,
    }


def _subs_of(path: str) -> set[str]:
    with np.load(path, allow_pickle=False) as z:
        return {str(s) for s in z["ep_sub"].tolist()}


def _select(pool: records.Pool, subs: Iterable[str]) -> list[int]:
    want = set(subs)
    return [e for e in range(pool.n_eps) if str(pool.eps["sub"][e]) in want]


def fit_check(fitted: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Agreement of fitted parameters with frozen ones on the stored core fields, per hypothesis."""
    keys = ("lam", "S", "z_plain", "sdir", "B", "prior")
    exact = close = 0
    worst: list[list[Any]] = []
    hyps = sorted(set(frozen["hyp"]) & set(fitted["hyp"]))
    for h in hyps:
        a = {k: fitted["hyp"][h]["core"].get(k) for k in keys}
        b = {k: frozen["hyp"][h]["core"].get(k) for k in keys}
        c = compare(a, b, rtol=1e-6)
        exact += int(c["exact"])
        close += int(c["close"])
        if not c["close"]:
            worst.append([h, c["n_diff"], c["max_abs"]])
    return {
        "hypotheses": len(hyps),
        "missing": sorted(set(frozen["hyp"]) ^ set(fitted["hyp"])),
        "exact": exact,
        "close_1e-6": close,
        "not_close": worst[:10],
    }


def _z(z: Mapping[Any, float]) -> dict[str, float]:
    return {"|".join(k): float(v) for k, v in sorted(z.items())}


def measure_fit(inputs: Inputs, frozen: Frozen) -> Record:
    """Times the training fit and the step-8 prior run on the stored training episodes."""
    ph = Phases()
    files = [f for f in cache_files(inputs.train, None) if _subs_of(f) & set(TRAIN_SUBS)]
    with ph.timed("load"):
        pool = pm.load_pmrt_pool(files)
    train = _select(pool.pool, TRAIN_SUBS)
    prior_eps = _select(pool.pool, (PRIOR_SUB,))
    with ph.timed("features"):
        pd = pm.pmrt_data(pool, train)
    with ph.timed("fit"):
        fitted = fit_pmrt(pd, pm.LEVEL_V2_ARR, frozen.cfg)
    with ph.timed("prior_features"):
        pd2 = pm.pmrt_data(pool, prior_eps)
    with ph.timed("prior_test"):
        run = pm.run_integrated(pd2, frozen.params, frozen.cfg, SPLIT_POOLED)
    prior_z = {s: _z(pm.prior_from_run(run, s).z) for s in pm.STATS}
    stored_z = {s: _z(frozen.priors[s].z) for s in pm.STATS}
    return {
        **ph.to_json(("features", "fit", "prior_features", "prior_test")),
        "n_train_episodes": len(train),
        "n_train_units": int(pd.n),
        "n_prior_episodes": len(prior_eps),
        "n_prior_units": int(pd2.n),
        "fit_check": fit_check(fitted, frozen.params),
        "prior_check": compare(prior_z, stored_z, rtol=1e-9),
    }


def unit_contexts(pool: records.Pool) -> list[dict[str, Any]]:
    """The opening context of every stored unit, as the referee reads it (`{"c", "ctx"}`)."""
    u, keys = pool.u, list(pool.ctx_keys)
    out = []
    for i in range(len(u["gep"])):
        ctx: dict[str, Any] = {
            keys[j]: float(u["ctx"][i, j]) for j in range(len(keys)) if u["ctx_num"][i, j]
        }
        ctx["knob"] = FAMILIES[int(u["family"][i])]
        ctx["step"] = float(u["step"][i])
        out.append({"c": int(u["c"][i]), "ctx": ctx})
    return out


def latency(call: Callable[[Mapping[str, Any]], Any], us: Sequence[Mapping[str, Any]]) -> dict:
    """Median, 99th percentile, mean and maximum microseconds of `call` per unit."""
    ns = np.empty(len(us))
    clock = time.perf_counter_ns
    for i, u in enumerate(us):
        t = clock()
        call(u)
        ns[i] = clock() - t
    us_ = ns / 1e3
    if not len(us_):
        return {"n": 0}
    return {
        "n": len(us_),
        "median_us": float(np.median(us_)),
        "p99_us": float(np.quantile(us_, 0.99)),
        "mean_us": float(us_.mean()),
        "max_us": float(us_.max()),
    }


def _artifact(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def measure_referee(
    inputs: Inputs, maps_doc: Mapping[str, Any], cs_doc: Mapping[str, Any]
) -> Record:
    """Times MapGateV2(MG:PMRT) and CertSafeMapGateV2(CS:PMRT) on every stored DISC context."""
    ph = Phases()
    with ph.timed("load"):
        pool = records.load_pool(cache_files(inputs.s2, "disc"), H=90, H_pre=90, stages={"eval"})
        ctxs = unit_contexts(pool)
    mg = rf.load_conf_maps(maps_doc)["maps"]["MG:PMRT"]
    cs = rf.load_certsafe(cs_doc)["tables"]["CS:PMRT"]
    bounds = rf.bounds_from_json(cs_doc["bounds"]["pmrt"])
    tau = rf.tau_from_json(cs_doc["tau"])
    theta = float(cs_doc["mapgate"]["theta"])
    with ph.timed("mapgate"):
        mg_lat = latency(rf.MapGateV2(mg, theta), ctxs)
    with ph.timed("certsafe"):
        cs_lat = latency(rf.CertSafeMapGateV2(cs["map"], cs["uncertified"], theta), ctxs)
    calls = np.empty(CERTIFY_CALLS)
    cert: dict[str, Any] = {}
    with ph.timed("certify"):
        for i in range(CERTIFY_CALLS):
            t = time.perf_counter_ns()
            cert = rf.certify_map(cs["map"], bounds, tau, theta)
            calls[i] = time.perf_counter_ns() - t
    return {
        **ph.to_json(("mapgate", "certsafe", "certify")),
        "n_contexts": len(ctxs),
        "MG:PMRT": mg_lat,
        "CS:PMRT": cs_lat,
        "certify": {
            "calls": CERTIFY_CALLS,
            "median_us": float(np.median(calls) / 1e3),
            "p99_us": float(np.quantile(calls, 0.99) / 1e3),
            "uncertified": cert.get("uncertified"),
            "repro": [list(x) for x in cert.get("uncertified", [])]
            == [list(x) for x in cs["uncertified"]],
        },
    }


def unit_rates(pool: records.Pool) -> dict[str, Any]:
    """Units per episode-hour, overall, per cell, by request class and by logging policy.

    An episode lasts its series length `T` in seconds.
    """
    u, e = pool.u, pool.eps
    hours = float(np.sum(e["T"].astype(float))) / 3600.0
    fam = np.asarray(u["family"], int)
    sgn = np.where(np.asarray(u["step"], float) < 0, -1, 1)
    pol = np.asarray(e["policy"])[np.asarray(u["gep"], int)]
    n = len(fam)
    by_class = {
        f"{FAMILIES[f]}|{d:+d}": int(np.sum((fam == f) & (sgn == d)))
        for f in range(len(FAMILIES))
        for d in (1, -1)
    }
    by_policy = {str(p): int(np.sum(pol == p)) for p in sorted(set(pol.tolist()))}
    rate = n / hours if hours else float("nan")
    return {
        "episodes": int(pool.n_eps),
        "hours": hours,
        "n_cells": int(pool.n_cells),
        "units": n,
        "units_per_ep_hour": rate,
        "units_per_cell_hour": rate / pool.n_cells,
        "by_class_per_ep_hour": {k: v / hours for k, v in by_class.items()} if hours else {},
        "by_policy_per_ep_hour": {k: v / hours for k, v in by_policy.items()} if hours else {},
        "by_class": by_class,
        "by_policy": by_policy,
    }


def unit_record_size(pool: records.Pool) -> dict[str, Any]:
    """Bytes of one logged unit: a fixed binary layout, and the measured compressed table."""
    keys = list(pool.ctx_keys)
    u = pool.u
    fixed = sum(b for _, b in UNIT_LAYOUT) + F32 * len(keys)
    n = len(u["gep"])
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        cell=np.asarray(u["c"], np.uint16),
        xapp=np.unique(np.asarray(u["xapp"]), return_inverse=True)[1].astype(np.uint8),
        family=np.asarray(u["family"], np.uint8),
        t0=np.asarray(u["t0"], np.uint32),
        mode=np.asarray(u["mode"], np.uint8),
        propensity=np.asarray(u["p"], np.float32),
        direction=np.where(np.asarray(u["step"], float) < 0, -1, 1).astype(np.int8),
        ctx=np.where(u["ctx_num"], u["ctx"], np.nan).astype(np.float32),
    )
    return {
        "fields": [f for f, _ in UNIT_LAYOUT] + [f"ctx:{k}" for k in keys],
        "n_ctx": len(keys),
        "fixed_bytes": fixed,
        "compressed_bytes_per_unit": buf.getbuffer().nbytes / n if n else float("nan"),
    }


def series_volume(paths: Sequence[Path], max_eps: int = SERIES_EPISODES) -> dict[str, Any]:
    """KPI series bytes per cell-second of stored raw episodes (the first `max_eps` per path)."""
    rows = []
    for p in paths:
        files = sorted(str(f) for f in Path(p).rglob("*.jsonl") if "bundle" not in f.parts)
        files = files if Path(p).is_dir() else [str(p)]
        seen = 0
        for rec in records.iter_records(files):
            ls = rec.get("lab_series")
            if rec.get("kind") != "episode" or not ls:
                continue
            data = records.dec(ls["data"])
            T, F, C = data.shape
            rows.append(
                {
                    "src": Path(p).name,
                    "seed": rec.get("seed"),
                    "shape": [T, F, C],
                    "dtype": str(data.dtype),
                    "raw_bytes_per_cell_s": data.nbytes / (T * C),
                    "stored_bytes_per_cell_s": len(json.dumps(ls["data"])) / (T * C),
                }
            )
            seen += 1
            if seen >= max_eps:
                break
    needed = len(KPIS) * F32
    return {
        "fields_all": list(records.SERIES_FIELDS),
        "kpis_needed": list(KPIS),
        "needed_f32_bytes_per_cell_s": needed,
        "all_f32_bytes_per_cell_s": len(records.SERIES_FIELDS) * F32,
        "episodes": rows,
    }


def projection(rates: Mapping[str, Any], size: Mapping[str, Any], bytes_cell_s: float) -> dict:
    """Per-cell-day volume of the simulated scenario's rates (a projection, not a measurement)."""
    upd = float(rates["units_per_cell_hour"]) * 24.0
    return {
        "note": "arithmetic on the simulated scenario's rates only",
        "units_per_cell_day": upd,
        "unit_bytes_per_cell_day": upd * float(size["fixed_bytes"]),
        "series_bytes_per_cell_day": 86400.0 * bytes_cell_s,
    }


def measure_logging(inputs: Inputs) -> Record:
    """Logging volume of the stored Study 1 and Study 2 discovery and placebo episodes."""
    ph = Phases()
    out: dict[str, Any] = {}
    with ph.timed("logging"):
        for name, job, stage, st in (
            ("s1_eval", inputs.s1, "eval", "eval"),
            ("s1_placebo", inputs.s1, "placebo", "placebo"),
            ("s2_disc", inputs.s2, "disc", "eval"),
            ("s2_placebo", inputs.s2, "placebo", "placebo"),
        ):
            pool = records.load_pool(cache_files(job, stage), H=90, H_pre=90, stages={st})
            rates, size = unit_rates(pool), unit_record_size(pool)
            out[name] = {
                "rates": rates,
                "unit_record": size,
                "projection": projection(rates, size, len(KPIS) * F32),
            }
        out["series"] = (
            series_volume(inputs.raw) if inputs.raw else {"skipped": "no raw records given"}
        )
    return {**ph.to_json(("logging",)), "volume": out}


@dataclasses.dataclass(frozen=True)
class Providers:
    """Environment queries and loaders of a worker; tests replace them for hermetic runs."""

    code: Callable[[], dict[str, Any]] = code_info
    host: Callable[[], dict[str, Any]] = lambda: {**host_info(), "cpu_quota": cpu_quota()}
    load: Callable[[], dict[str, Any]] = load_info
    pkgs: Callable[[], dict[str, Any]] = pkg_versions
    frozen: Callable[[], Frozen] = frozen_artifact
    maps_doc: Callable[[], dict[str, Any]] = lambda: _artifact("study2_maps.json")
    cs_doc: Callable[[], dict[str, Any]] = lambda: _artifact("study3_certsafe.json")
    log: Callable[[str], None] = print


def measure(u: Unit, inputs: Inputs, p: Providers) -> Record:
    """Runs one unit's measurement inside the peak-RSS scope."""
    with PeakRSS() as mem:
        if u.kind == "disc":
            rec = measure_disc(inputs, u.study, u.slice, p.frozen())
        elif u.kind == "fit":
            rec = measure_fit(inputs, p.frozen())
        elif u.kind == "referee":
            rec = measure_referee(inputs, p.maps_doc(), p.cs_doc())
        elif u.kind == "logging":
            rec = measure_logging(inputs)
        else:
            raise KeyError(u.kind)
    return {**rec, "peak_rss_mb": mem.mb, "rss_scope": mem.scope}


def done_keys(path: Path) -> set[str]:
    """Keys of the ok records already in a worker's output."""
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add(r["key"])
    return done


def worker(
    us: Sequence[Unit], inputs: Inputs, out: Path, providers: Providers | None = None
) -> int:
    """Measures the units one after the other and appends one record per unit.

    A unit whose ok record is already in `out` is skipped, so a cut worker resumes.

    Returns:
        The number of units run.
    """
    p = providers or Providers()
    done = done_keys(out)
    host, pkgs, code = p.host(), p.pkgs(), p.code()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_run = 0
    with out.open("a", encoding="utf-8") as fh:
        for u in us:
            if u.key in done:
                continue
            load0 = p.load()
            rec: Record
            try:
                rec = measure(u, inputs, p)
                rec["error"] = None
            except Exception:
                rec = {"error": traceback.format_exc()[-4000:]}
            rec.update(
                key=u.key,
                unit=u.to_json(),
                status="ok" if rec["error"] is None else "error",
                version=VERSION,
                protocol=PROTOCOL,
                host=host,
                pkgs=pkgs,
                code=code,
                load={"start": load0, "end": p.load()},
                threads={k: os.environ.get(k) for k in THREAD_ENV},
            )
            append_record(fh, clean(rec))
            n_run += 1
            p.log(
                f"[x19] {u.key} cpu {rec.get('cpu_s') or float('nan'):.1f}s wall "
                f"{rec.get('wall_s') or float('nan'):.1f}s {rec['status']}"
            )
    return n_run


Spawn = Callable[[Sequence[str], Mapping[str, str], Path], subprocess.Popen]


def spawn_process(argv: Sequence[str], env: Mapping[str, str], log: Path) -> subprocess.Popen:
    """Starts a worker process with its output appended to `log`."""
    return subprocess.Popen(argv, env=dict(env), stdout=log.open("a"), stderr=subprocess.STDOUT)


def worker_env(base: Mapping[str, str]) -> dict[str, str]:
    """The environment of a worker: one thread per numerical library."""
    return {**base, **dict.fromkeys(THREAD_ENV, "1")}


def input_args(s1: str, s2: str, train: str, raw: Sequence[str] = ()) -> list[str]:
    """The `run` / `worker` arguments naming the stored jobs."""
    args = ["--s1", s1, "--s2", s2, "--train", train]
    for r in raw:
        args += ["--raw", r]
    return args


def _input_args(inputs: Inputs) -> list[str]:
    return input_args(
        str(inputs.s1), str(inputs.s2), str(inputs.train), [str(r) for r in inputs.raw]
    )


def run(
    us: Sequence[Unit],
    inputs: Inputs,
    out_dir: Path,
    procs: int | None,
    *,
    environ: Mapping[str, str],
    spawn: Spawn = spawn_process,
    quota: Callable[[], int] = cpu_quota,
    host: Callable[[], dict[str, Any]] = host_info,
    python: str = sys.executable,
) -> int:
    """One worker process per vCPU, balanced by estimated cost; writes `run.json`.

    Returns:
        The largest worker exit code.
    """
    p = procs or quota()
    bins = [b for b in assign(us, p) if b]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = worker_env(environ)
    t0 = time.time()
    ps = []
    for i, b in enumerate(bins):
        units_file = out_dir / f"units_{i}.json"
        units_file.write_text(json.dumps([u.to_json() for u in b], indent=0), encoding="utf-8")
        argv = [
            python,
            "-u",
            "-m",
            WORKER_MODULE,
            "worker",
            "--units",
            str(units_file),
            "--out",
            str(out_dir / f"res_{i}.jsonl"),
            *_input_args(inputs),
        ]
        ps.append(spawn(argv, env, out_dir / f"log_{i}.txt"))
    rcs = [q.wait() for q in ps]
    info = {
        "version": VERSION,
        "protocol": PROTOCOL,
        "procs": len(bins),
        "cpu_quota": quota(),
        "units": len(us),
        "wall_s": time.time() - t0,
        "rcs": rcs,
        "host": host(),
        "inputs": {k: str(v) for k, v in dataclasses.asdict(inputs).items()},
        "est_cost_s_per_proc": [sum(u.est_cost_s for u in b) for b in bins],
    }
    (out_dir / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info), flush=True)
    return max(rcs, default=0)


def session_command(
    out_dir: str, platform: str, code: Mapping[str, Any], src: str, wall_s: int = WALL_S
) -> str:
    """The shell command of a cloud session: environment setup, then `run` on the sources.

    The sources are the stored jobs mounted at `<src>/<slug>` (`SOURCES`, `RAW_SOURCES`).
    """
    env = (
        f"{PLATFORM_ENV}={platform} {COMMIT_ENV}={code['commit']} "
        f"{DIRTY_ENV}={int(bool(code['dirty']))} {LOCK_ENV}={LOCK_REL} {THREADS}"
    )
    args = " ".join(
        input_args(
            f"{src}/{SOURCES['s1']}",
            f"{src}/{SOURCES['s2']}",
            f"{src}/{SOURCES['train']}",
            [f"{src}/{s}" for s in RAW_SOURCES],
        )
    )
    return (
        setup_cmd(SetupPlan(out_dir=out_dir))
        + f"{env} timeout {wall_s} python -u -m {WORKER_MODULE} run --out-dir {out_dir} {args}"
    )


def kaggle_job(name: str, owner: str, code: Mapping[str, Any]) -> kaggle.KaggleJob:
    """The Kaggle session of X19: the stored jobs as kernel-output sources."""
    sources = [f"{owner}/{s}" for s in (*SOURCES.values(), *RAW_SOURCES)]
    cmd = session_command("$JOB_OUT", "kaggle", code, "$JOB_SRC")
    return kaggle.KaggleJob(name, cmd, (), sources)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("units")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--reps", type=int, default=REPS)
    for name in ("run", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--s1", required=True)
        p.add_argument("--s2", required=True)
        p.add_argument("--train", required=True)
        p.add_argument("--raw", action="append", default=[])
        if name == "run":
            p.add_argument("--out-dir", required=True)
            p.add_argument("--procs", type=int, default=None)
            p.add_argument("--reps", type=int, default=REPS)
            p.add_argument("--only", nargs="+", choices=KINDS, default=list(KINDS))
        else:
            p.add_argument("--units", required=True)
            p.add_argument("--out", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--name", required=True)
    p.add_argument("--owner", default=None)
    p.add_argument("--dry-run", action="store_true")
    return ap


def _inputs(a: argparse.Namespace) -> Inputs:
    return Inputs(Path(a.s1), Path(a.s2), Path(a.train), tuple(Path(r) for r in a.raw))


def _launch(a: argparse.Namespace) -> int:
    client = kaggle.KaggleClient(owner=a.owner)
    job = kaggle_job(a.name, client.account, code_info())
    print(f"[x19] kaggle {job.name}: sources {list(job.sources)}")
    print(f"[x19] cmd: {job.cmd}")
    if a.dry_run:
        return 0
    print(json.dumps(kaggle.launch(job, client, Dirs(WORK_DIR)), indent=1, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: `units`, `run`, `worker` and `launch`."""
    a = _parser().parse_args(argv)
    if a.cmd == "units":
        us = units(a.reps)
        bins = assign(us, a.procs)
        tot = sum(u.est_cost_s for u in us) / 3600
        wall = max(sum(u.est_cost_s for u in b) for b in bins) / 3600
        print(f"[x19] {len(us)} units, est. {tot:.2f} CPU-h, wall at {a.procs} procs ~{wall:.2f} h")
        for u in us:
            print(f"  {u.key}  est {u.est_cost_s:.0f} s")
    elif a.cmd == "run":
        return run(units(a.reps, a.only), _inputs(a), Path(a.out_dir), a.procs, environ=os.environ)
    elif a.cmd == "worker":
        raw = json.loads(Path(a.units).read_text(encoding="utf-8"))
        worker([Unit(**u) for u in raw], _inputs(a), Path(a.out))
    else:
        return _launch(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

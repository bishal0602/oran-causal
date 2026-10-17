"""Study 1 discovery tests over the v4 episodes: PMRT and the baselines on fresh slices.

The frozen PMRT artifact (`E6P_PMRT_V4.json`) is run on the EVAL episodes pooled (split 0,
n_target 600) and on the 60-, 120- and 300-episode slices (splits 20 + i, 40 + k, 60 + m), with
every (statistic, layer) combination; the primary is loadsp_c + wby1s. The placebo episodes run
at split 9 (n_target 40), and the null-outcome variants (cyclic `lab_series` shifts 1-4 inside
each group of 60 EVAL episodes, B 999) at split 0. The baselines are corr and granger (DEV
far-FPR tau), granger_by, MSCR-CRT v2 + BY and the descriptive shap_gbdt, int, qacm and
two_tower. Every declaration is scored against the knockout GT reference of the gt episodes.

The output holds the test parts of the study's analysis; the criteria (K0, K0n, K1, G, P1-P3,
S) and the verdict are `analysis.system.study1`'s.
"""

import dataclasses
import hashlib
import json
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.system import study1 as crit
from oran_causal.core.data import UnitTable
from oran_causal.experiments.paths import ARTIFACTS
from oran_causal.experiments.system import study1
from oran_causal.experiments.system.common import sha_lf, to_json
from oran_causal.methods.system import artifact as pmrt_artifact
from oran_causal.methods.system import baselines as bl
from oran_causal.methods.system import mscr_crt
from oran_causal.methods.system import pmrt as pm
from oran_causal.sim.system.decision import gt as G
from oran_causal.sim.system.decision import records
from oran_causal.sim.system.decision.vocab import FAMILIES, MODES, PRIMARY_ORIENT

SCHEMA = "e6p-disc-v4-tests/1"
ARTIFACT = ARTIFACTS / pmrt_artifact.ARTIFACT_NAME
PRIMARY = ("loadsp_c", "wby1s")
SEEDS = {
    "dev": (188000, 20),
    "eval": (188100, 600),
    "placebo": (188700, 40),
    "gt": (188800, 40),
}
N60, N120, N300 = 60, 120, 300
SPLIT = {"pooled": 0, "60": 20, "120": 40, "300": 60, "placebo": 9, "null": 0}
H = H_PRE = 90
BASELINES = ("corr", "granger", "granger_by")
DESC_METHODS = ("shap_gbdt", "int", "qacm", "two_tower")
KEEP = (
    "kind",
    "schema",
    "key",
    "stage",
    "sub",
    "seed",
    "j",
    "fold",
    "smoke",
    "policy",
    "pi0_table",
    "n_cells",
    "units",
    "lab_series",
)
ALPHA = 0.05
SECTIONS = ("placebo", "slices", "null")

type Decl = Mapping[tuple, Mapping[str, Any]]
type Method = Callable[[UnitTable, int], dict]


def log(s: str = "") -> None:
    """Prints a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def keys_to_str(d: Any) -> Any:
    """Copy of `d` with tuple keys joined by "|" (JSON-able)."""
    if isinstance(d, dict):
        return {
            ("|".join(map(str, k)) if isinstance(k, tuple) else k): keys_to_str(v)
            for k, v in d.items()
        }
    if isinstance(d, list | tuple):
        return [keys_to_str(v) for v in d]
    return d


def dump(obj: Any, path: Path) -> None:
    """Writes `obj` as indented JSON with "|"-joined tuple keys."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(keys_to_str(obj), fh, indent=1, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, set | tuple):
        return list(o)
    return to_json(o)


def find_records(specs: Iterable[str | Path]) -> list[Path]:
    """Record files of comma-free specs: files, globs, or directories.

    A directory contributes its `res_*.jsonl` shard files (else its `all.jsonl` files),
    recursively, skipping bundle copies. Duplicates are dropped, first occurrence kept.
    """
    out: list[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            res = sorted(f for f in p.rglob("res_*.jsonl") if "bundle" not in f.parts)
            out += res or sorted(f for f in p.rglob("all.jsonl") if "bundle" not in f.parts)
        elif any(ch in str(spec) for ch in "*?["):
            out += sorted(Path(".").glob(str(spec)))
        else:
            out.append(p)
    seen: set[Path] = set()
    uniq = []
    for f in out:
        a = f.resolve()
        if a not in seen:
            seen.add(a)
            uniq.append(f)
    return uniq


def _cache_one(job: tuple[Path, str, Path]) -> Path | None:
    path, stage, dst = job
    if not dst.exists():
        try:
            records.build_cache([str(path)], dst, stages={stage}, src=path.name)
        except ValueError as e:
            log(f"[cache] skip {path}: {e}")
            return None
    return dst


def build_caches(
    files: Sequence[Path], stage: str, cache_dir: Path, tag: str, workers: int = 1
) -> list[Path]:
    """One unit-table cache per JSONL file (episodes of `stage`); npz inputs pass through.

    Files without an episode of `stage` are skipped. Existing caches are reused.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = [f for f in files if f.suffix == ".npz"]
    jobs = []
    for i, f in enumerate(f for f in files if f.suffix != ".npz"):
        h = hashlib.sha256(str(f.resolve()).encode()).hexdigest()[:10]
        jobs.append((f, stage, cache_dir / f"{tag}_{i:03d}_{h}.npz"))
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, len(jobs))) as pool:
            done = [r for r in pool.imap_unordered(_cache_one, jobs) if r]
    else:
        done = [r for r in map(_cache_one, jobs) if r]
    return npz + sorted(done)


def gt_reference(paths: Sequence[Path]) -> dict[str, Any]:
    """The knockout GT reference of every gt episode of `paths` (orientation "dir").

    Returns:
        `{"ref", "counts", "n_episodes", "n_labels", "labels_per_family", "seeds", "delta",
        "nbr_true", "cells"}`; `ref` maps each hypothesis to its status and sign.
    """
    eps = records.gt_episodes([str(p) for p in paths])
    gs = G.summarize(eps)
    table = gs["tables"][PRIMARY_ORIENT]
    return {
        "ref": crit.gt_ref(table["cells"]),
        "counts": table["counts"],
        "n_episodes": len(eps),
        "n_labels": gs["n_labels"],
        "labels_per_family": gs["labels_per_family"],
        "seeds": [int(e["seed"]) for e in eps],
        "delta": table["delta"],
        "nbr_true": sorted(
            (c["family"], c["kpi"], c["sign"])
            for c in table["cells"]
            if c["relation"] == "nbr" and c["status"] == "TRUE"
        ),
        "cells": table["cells"],
    }


def declared_list(decl: Decl) -> list[tuple[str, int]]:
    """Sorted (hypothesis, sign) pairs of the declared edges."""
    return sorted(
        ("|".join(h), int(v.get("sign", 0))) for h, v in decl.items() if v and v.get("declared")
    )


def _finite_p(decl: Decl) -> list[float]:
    return [
        float(v["p"])
        for v in decl.values()
        if v and v.get("p") is not None and np.isfinite(float(v["p"]))
    ]


def placebo_check(method: Method, data: UnitTable, split: int = SPLIT["placebo"]) -> dict:
    """A method on placebo units: its declarations and its rejection rate at 0.05 (finite p)."""
    t = time.time()
    decl = method(data, split)
    ps = _finite_p(decl)
    return {
        "n_declared": sum(1 for v in decl.values() if v and v.get("declared")),
        "declared": sorted("|".join(h) for h, v in decl.items() if v and v.get("declared")),
        "n_p": len(ps),
        "n_reject": int(sum(p <= ALPHA for p in ps)),
        "rate": float(np.mean([p <= ALPHA for p in ps])) if ps else float("nan"),
        "min_p": float(min(ps)) if ps else float("nan"),
        "units": int(data.n),
        "wall_s": round(time.time() - t, 1),
    }


def unit_counts(ud: UnitTable) -> dict[str, Any]:
    """Units per family, those with t0 in [90, 120), and the sleep units and rejects."""
    rej = MODES.index("reject")
    early = {}
    for f in FAMILIES:
        r = ud.rows_of(f)
        early[f] = int(((ud.t0[r] >= 90) & (ud.t0[r] < 120)).sum())
    sl = ud.rows_of("sleep")
    return {
        "units": int(ud.n),
        "by_family": {f: len(ud.rows_of(f)) for f in FAMILIES},
        "t0_90_120_by_family": early,
        "sleep_units": len(sl),
        "sleep_rejects": int((ud.mode[sl] == rej).sum()),
        "n_episodes": len(np.unique(ud.episode)),
    }


def hyp_table(run: Mapping[str, Mapping[str, Any]], decl: Decl) -> dict[str, Any]:
    """Per hypothesis: z and p of the three statistic arms and the primary layer's use of it."""
    out = {}
    for h, o in run.items():
        d = decl.get(tuple(h.split("|")), {})
        row: dict[str, Any] = {"status": o.get("status")}
        if o.get("status") == "tested":
            for s in pm.STATS:
                row[s] = {k: o[s].get(k) for k in ("z", "p2", "p_plus", "p_minus", "sign")}
        row["primary"] = {
            k: d.get(k) for k in ("declared", "sign", "p", "w", "one_sided", "status")
        }
        out[h] = row
    return out


def decl_record(decl: Decl) -> dict[str, Any]:
    """A declaration set as JSON: hypothesis -> declared, sign, p, w, one-sided, status, z."""
    keep = ("declared", "sign", "p", "w", "one_sided", "status", "z_approx")
    return {"|".join(h): {k: v.get(k) for k in keep if k in v} for h, v in decl.items()}


@dataclasses.dataclass(frozen=True)
class Slice:
    """An EVAL slice: name, kind (pooled / 60 / 120 / 300), index, RNG split, size, seeds."""

    name: str
    kind: str
    index: int
    split: int
    n_target: int
    seeds: tuple[int, ...]


def slice_plan(
    seeds: Iterable[int], pseudo: bool = False, sizes: Sequence[int] = (N60, N120, N300)
) -> list[Slice]:
    """The pooled set and the 60-, 120- and 300-episode slices of the EVAL seeds.

    Slice i of size n holds the seeds with j // n == i, j = seed - 188100 (`pseudo`: the seed's
    rank, for incomplete data).
    """
    seeds = sorted(int(s) for s in seeds)
    base = SEEDS["eval"][0]
    j = {s: (r if pseudo else s - base) for r, s in enumerate(seeds)}
    out = [Slice("pooled", "pooled", 0, SPLIT["pooled"], len(seeds), tuple(seeds))]
    n_total = (max(j.values()) + 1 if seeds else 0) if pseudo else SEEDS["eval"][1]
    for kind, n in zip(("60", "120", "300"), sizes, strict=True):
        for i in range(n_total // n):
            mem = tuple(s for s in seeds if j[s] // n == i)
            out.append(Slice(f"s{kind}_{i}", kind, i, SPLIT[kind] + i, n, mem))
    return out


def episodes_of(pool: records.Pool, seeds: Iterable[int]) -> np.ndarray:
    """Pool episode ids of the given seeds."""
    want = {int(s) for s in seeds}
    return np.array([e for e in range(pool.n_eps) if int(pool.eps["seed"][e]) in want], int)


def data_status(pools: Mapping[str, Any], gt_seeds: Sequence[int], dry: bool) -> dict[str, Any]:
    """Completeness of the v4 inputs (`analysis.system.study1.data_status` of the pools)."""
    stages = {
        name: crit.stage_status(
            name,
            None if pool is None else [int(s) for s in pool.eps["seed"]],
            [] if pool is None else [str(s) for s in pool.eps["sub"]],
            False if pool is None else bool(np.any(pool.eps["smoke"])),
        )
        for name, pool in pools.items()
    }
    return crit.data_status(stages, gt_seeds, dry)


def load_records_light(files: Sequence[Path], stage: str, allow_smoke: bool = False) -> list[dict]:
    """Episode records of `stage` with only the `KEEP` keys, deduplicated, sorted by seed."""
    recs = {}
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or '"episode"' not in line[:200]:
                    continue
                r = json.loads(line)
                if r.get("kind") != "episode" or r.get("stage") != stage:
                    continue
                if r.get("smoke") and not allow_smoke:
                    continue
                for u in r["units"]:
                    u.pop("lab_kpi", None)
                recs[(int(r["seed"]), str(r.get("sub") or ""))] = {k: r[k] for k in KEEP if k in r}
    return [recs[k] for k in sorted(recs)]


def shift_variants(recs: Sequence[dict], shifts: int) -> list[tuple[str, list[dict]]]:
    """Null-outcome variants: episode i takes the `lab_series` of episode (i + k) mod G."""
    n = len(recs)
    return [
        (
            f"shift{k}",
            [dict(r, lab_series=recs[(i + k) % n]["lab_series"]) for i, r in enumerate(recs)],
        )
        for k in range(1, shifts + 1)
        if n and k % n
    ]


def null_groups(seeds: Sequence[int], size: int, pseudo: bool) -> list[list[int]]:
    """Record indices of the null-outcome groups: j // size (pseudo: consecutive records)."""
    if pseudo:
        order = list(range(len(seeds)))
        return [order[i : i + size] for i in range(0, len(order) - size + 1, size)]
    base, n = SEEDS["eval"]
    return [[i for i, s in enumerate(seeds) if (s - base) // size == g] for g in range(n // size)]


def null_tests(
    recs: Sequence[dict],
    groups: Sequence[Sequence[int]],
    params: Mapping[str, Any],
    priors: Mapping[str, Any],
    cfg: Any,
    shifts: int,
) -> dict[str, Any]:
    """PMRT on every null-outcome variant.

    Returns:
        The variant names, the group count, and per combination the used p-values, their
        families and the number of declarations per variant.
    """
    tmp = Path(tempfile.mkdtemp(prefix="k0n_v4_"))
    per = {pm.cname(s, lay): {"p": [], "fam": [], "var_decl": []} for s, lay in pm.COMBOS}
    names = []
    for gi, idx in enumerate(groups):
        sub = [recs[i] for i in idx]
        for name, rr in shift_variants(sub, shifts):
            t = time.time()
            pd = pm.pmrt_data_from_records(rr, tmp)
            run = pm.run_integrated(pd, params, cfg, SPLIT["null"])
            for (s, lay), d in pm.declare_all(run, priors, len(sub)).items():
                o = per[pm.cname(s, lay)]
                o["var_decl"].append(sum(bool(v["declared"]) for v in d.values()))
                for h, v in d.items():
                    if v.get("status") != "undetermined" and np.isfinite(v.get("p", np.nan)):
                        o["p"].append(float(v["p"]))
                        o["fam"].append(h[0])
            names.append(f"g{gi}:{name}")
            n_dec = per[pm.cname(*PRIMARY)]["var_decl"][-1]
            log(f"  K0n g{gi} {name}: units {pd.n}, primary decl {n_dec} ({time.time() - t:.0f} s)")
    return {"variants": names, "groups": len(groups), "combos": per}


@dataclasses.dataclass(frozen=True)
class Settings:
    """Run settings of the Study 1 discovery tests.

    Attributes:
        B: CRT redraws of the slices and the placebo.
        B_null: Redraws of the null-outcome variants.
        null_group: Episodes per null-outcome group.
        null_shifts: Cyclic shifts per group.
        workers: Cache-building processes.
        desc_methods: Descriptive baselines.
        desc_kinds: Slice kinds on which the descriptive baselines run.
        v2: Run MSCR-CRT v2 + BY.
        sections: Parts to run (`SECTIONS`).
        slices: Slice names to run (None = all).
        dry_run: Never a full analysis (DEV data stands in for EVAL when EVAL is absent).
        allow_smoke: Accept smoke records.
    """

    B: int = 9999
    B_null: int = 999
    null_group: int = 60
    null_shifts: int = 4
    workers: int = 1
    desc_methods: tuple[str, ...] = DESC_METHODS
    desc_kinds: frozenset[str] = frozenset({"pooled", "300", "120"})
    v2: bool = True
    sections: frozenset[str] = frozenset(SECTIONS)
    slices: frozenset[str] | None = None
    dry_run: bool = False
    allow_smoke: bool = False


DEFAULT_SETTINGS = Settings()


@dataclasses.dataclass(frozen=True)
class Inputs:
    """Record locations per stage (files, directories or globs) and the cache directory."""

    eval: tuple[str, ...]
    placebo: tuple[str, ...]
    gt: tuple[str, ...]
    dev: tuple[str, ...]
    cache_dir: Path


def artifact_status(path: Path = ARTIFACT) -> dict[str, Any]:
    """sha256 of the PMRT artifact against its pin, and `artifact.verify`'s problems."""
    a = json.loads(path.read_text(encoding="utf-8"))
    sha = sha_lf(path)
    v = pmrt_artifact.verify(path)
    return {
        "path": f"docs/benchmark/artifacts/{path.name}",
        "sha256": sha,
        "sha_ok": sha == pmrt_artifact.ARTIFACT_SHA256,
        "schema": a.get("schema"),
        "version": a.get("version"),
        "legacy_version": a.get("legacy_version"),
        "problems": v["problems"],
        "code_ok": not v["problems"],
        "ok": sha == pmrt_artifact.ARTIFACT_SHA256 and not v["problems"],
    }


@dataclasses.dataclass
class _Run:
    params: Mapping[str, Any]
    priors: Mapping[str, Any]
    cfg: Any
    ref: Mapping[tuple, Mapping[str, Any]]
    bmeth: dict[str, Method]
    dmeth: dict[str, Method]
    errors: dict[str, str]
    settings: Settings


def _v2_decl(ud: UnitTable, B: int, split: int) -> dict:
    cfg = mscr_crt.UnitCRTConfigV2(B=B)
    return mscr_crt.edges_from_crt_v2(mscr_crt.run_crt_units_v2(ud, cfg, split))


def _desc_methods(dev: UnitTable, names: Sequence[str], errors: dict[str, str]) -> dict:
    out = {}
    for nm in names:
        t = time.time()
        try:
            out[nm] = bl.desc_method(nm, dev)
        except Exception as e:
            errors[nm] = repr(e)[:400]
            log(f"desc {nm}: unavailable ({errors[nm][:120]})")
            continue
        log(f"desc {nm}: tau {(out[nm].tau or {}).get('tau')} [{time.time() - t:.0f} s]")
    return out


def placebo_tests(r: _Run, ppool: pm.PmrtPool) -> dict[str, Any]:
    """PMRT, the baselines and MSCR-CRT v2 on the placebo episodes."""
    pdp = pm.pmrt_data(ppool)
    run_p = pm.run_integrated(pdp, r.params, r.cfg, SPLIT["placebo"])
    decl_p = pm.declare_all(run_p, r.priors, int(ppool.pool.n_eps))
    base: dict[str, Any] = {}
    for b, m in [*r.bmeth.items(), *r.dmeth.items()]:
        try:
            base[b] = placebo_check(m, pdp.ud, SPLIT["placebo"])
        except Exception as e:
            base[b] = {"error": repr(e)[:300]}
    if r.settings.v2:
        e2 = _v2_decl(pdp.ud, r.settings.B, SPLIT["placebo"])
        p05 = [v["p"] <= ALPHA for v in e2.values() if np.isfinite(v["p"])]
        base["v2+by"] = {
            "n_declared": sum(bool(v["declared"]) for v in e2.values()),
            "rate05": float(np.mean(p05 or [np.nan])),
        }
    return {
        "placebo_decl": {pm.cname(*k): decl_record(d) for k, d in decl_p.items()},
        "placebo_baselines": base,
        "placebo_units": unit_counts(pdp.ud),
        "placebo_hyp": hyp_table(run_p, decl_p[PRIMARY]),
    }


def slice_tests(r: _Run, pool: pm.PmrtPool, sl: Slice, episodes: np.ndarray) -> dict[str, Any]:
    """PMRT (every combination) and the baselines on one EVAL slice, scored against the GT."""
    t = time.time()
    pd = pm.pmrt_data(pool, episodes.tolist())
    run = pm.run_integrated(pd, r.params, r.cfg, sl.split)
    decls: dict[str, Any] = {
        pm.cname(*k): d for k, d in pm.declare_all(run, r.priors, sl.n_target).items()
    }
    for b, m in r.bmeth.items():
        decls[b] = m(pd.ud, sl.split)
    if r.settings.v2:
        decls["v2+by"] = _v2_decl(pd.ud, r.settings.B, sl.split)
    if sl.kind in r.settings.desc_kinds:
        for b, m in r.dmeth.items():
            try:
                decls[b] = m(pd.ud, sl.split)
            except Exception as e:
                r.errors[f"{b}@{sl.name}"] = repr(e)[:300]
    mets = {
        k: crit.subset_metrics(d, r.ref) | {"declared": declared_list(d)} for k, d in decls.items()
    }
    res = {
        "kind": sl.kind,
        "index": sl.index,
        "split": sl.split,
        "n_target": sl.n_target,
        "n_episodes": len(episodes),
        "units": unit_counts(pd.ud),
        "metrics": mets,
        "hyp": hyp_table(run, decls[pm.cname(*PRIMARY)]),
        "wall_s": round(time.time() - t, 1),
    }
    pmet = mets[pm.cname(*PRIMARY)]
    log(
        f"{sl.name:>9} ({len(episodes)} eps, {pd.n} units, {res['wall_s']} s): primary chain "
        f"{pmet['chain_hits']}/{pmet['chain_true']} indF1 {pmet['ind_f1']:.2f} "
        f"#dec {pmet['n_declared']}"
    )
    return res


def analyze(
    inputs: Inputs, settings: Settings = DEFAULT_SETTINGS, artifact: Path = ARTIFACT
) -> dict:
    """Runs the Study 1 discovery tests and returns their output (module docstring)."""
    t_all = time.time()
    timing: dict[str, float] = {}
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol": study1.PROTOCOL.status(),
        "primary": "+".join(PRIMARY),
        "artifact": artifact_status(artifact),
    }
    params, priors = pmrt_artifact.load_artifact(artifact)
    t = time.time()
    files = {k: find_records(getattr(inputs, k)) for k in ("eval", "placebo", "dev")}
    caches = {
        k: build_caches(files[k], k, inputs.cache_dir / k, k, settings.workers) for k in files
    }
    timing["caches"] = round(time.time() - t, 1)
    ev_stage = "eval"
    if not caches["eval"] and settings.dry_run:
        caches["eval"], ev_stage = caches["dev"], "dev"
    if not all(caches.values()):
        raise ValueError(f"missing inputs: { {k: len(v) for k, v in caches.items()} }")
    pool = pm.load_pmrt_pool([str(p) for p in caches["eval"]], stages=[ev_stage])
    ppool = pm.load_pmrt_pool([str(p) for p in caches["placebo"]], stages=["placebo"])
    dpool = records.load_pool([str(p) for p in caches["dev"]], H=H, H_pre=H_PRE, stages={"dev"})
    smoke = any(bool(np.any(p.eps["smoke"])) for p in (pool.pool, ppool.pool, dpool))
    if smoke and not settings.allow_smoke:
        raise ValueError("smoke records in the inputs")
    t = time.time()
    g = gt_reference(find_records(inputs.gt))
    timing["gt"] = round(time.time() - t, 1)
    out["gt"] = {k: v for k, v in g.items() if k != "ref"}
    dry = settings.dry_run or ev_stage != "eval"
    out["data"] = data_status(
        {"eval": pool.pool if ev_stage == "eval" else None, "placebo": ppool.pool, "dev": dpool},
        g["seeds"],
        dry,
    )
    ev_seeds = sorted(int(s) for s in pool.pool.eps["seed"])
    e0, en = SEEDS["eval"]
    in_range = bool(ev_seeds) and all(e0 <= x < e0 + en for x in ev_seeds)
    plan = slice_plan(ev_seeds, pseudo=dry or not in_range)
    eps = {sl.name: episodes_of(pool.pool, sl.seeds) for sl in plan}
    out["slices"] = [
        {
            "name": sl.name,
            "kind": sl.kind,
            "index": sl.index,
            "split": sl.split,
            "n_target": sl.n_target,
            "complete": len(eps[sl.name]) == sl.n_target,
            "seed_range": [min(sl.seeds), max(sl.seeds)] if sl.seeds else None,
            "n": len(eps[sl.name]),
        }
        for sl in plan
    ]
    dev = dpool.unit_data()
    errors: dict[str, str] = {}
    bmeth: dict[str, Any] = {b: bl.method_baseline(b, dev) for b in BASELINES}
    dmeth = _desc_methods(dev, settings.desc_methods, errors)
    out["baseline_tau"] = {b: m.tau for b, m in bmeth.items()}
    out["desc_tau"] = {k: m.tau for k, m in dmeth.items()}
    r = _Run(
        params,
        priors,
        pmrt_artifact.pmrt_config(params, settings.B),
        g["ref"],
        bmeth,
        dmeth,
        errors,
        settings,
    )
    if "placebo" in settings.sections:
        t = time.time()
        out |= placebo_tests(r, ppool)
        timing["placebo"] = round(time.time() - t, 1)
    if "slices" in settings.sections:
        res = {}
        for sl in plan:
            if len(eps[sl.name]) and (settings.slices is None or sl.name in settings.slices):
                res[sl.name] = slice_tests(r, pool, sl, eps[sl.name])
        out["slices_result"] = res
        timing["slices"] = round(sum(v["wall_s"] for v in res.values()), 1)
    if "null" in settings.sections:
        t = time.time()
        src = files["eval"] if ev_stage == "eval" else files["dev"]
        recs = load_records_light(src, ev_stage, settings.allow_smoke)
        groups = null_groups(
            [int(x["seed"]) for x in recs], settings.null_group, dry or not in_range
        )
        log(f"K0n: {len(recs)} records, {len(groups)} groups, {settings.null_shifts} shifts")
        cfg_null = pmrt_artifact.pmrt_config(params, settings.B_null)
        out["K0n_raw"] = null_tests(recs, groups, params, priors, cfg_null, settings.null_shifts)
        timing["k0n"] = round(time.time() - t, 1)
    out["desc_errors"] = errors
    timing["total"] = round(time.time() - t_all, 1)
    out["timing_s"] = timing
    return out


def run(inputs: Inputs, out_dir: Path, settings: Settings = DEFAULT_SETTINGS) -> Path:
    """Runs `analyze` and writes its tests and their analysis into `out_dir`.

    Writes `tests_v4.json`, `analysis_v4.json` (`analysis.system.study1.from_tests`) and
    `verdict_v4.json`; returns the analysis path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = out_dir / "tests_v4.json"
    dump(analyze(inputs, settings), tests)
    rep = crit.from_tests(json.loads(tests.read_text(encoding="utf-8")))
    path = out_dir / "analysis_v4.json"
    dump(rep, path)
    dump(crit.verdict_record(rep), out_dir / "verdict_v4.json")
    log(f"wrote {tests}, {path} ({rep['verdict']['verdict']})")
    return path

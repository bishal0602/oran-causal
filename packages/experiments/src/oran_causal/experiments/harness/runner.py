"""Job planning and execution of the plain runner.

A job is (world, regime, lam, n, seed, kappa, method). A spec names `methods`, `worlds`,
`regimes`, `ns`, `seeds` ([lo, hi] inclusive or a list), `lams` (E4 only), the required
`kappas`, and optional per-cell `configs`. Job keys always carry the kappa:
`<method>|<world>|<regime>|[lam<l>|]k<kappa>|n<n>|s<seed>`.
"""

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.data import Method
from oran_causal.experiments.harness.records import Job, append_record, done_keys, run_one
from oran_causal.methods.registry import resolve
from oran_causal.sim.controlled.datasets.generate import generate_dataset
from oran_causal.sim.controlled.datasets.truth import REGIMES_OF

DEV_SEEDS = range(3_000_000, 3_000_200)
EVAL_FLOOR = 3_100_000
SCORE_MEANS = ("precision", "recall", "f1", "fdp", "null_fpr", "sign_acc")

type Spec = dict[str, Any]
type DatasetKey = tuple[str, str, float | None, int, int, float | None]
type Resolver = Callable[[str], Any]
type Log = Callable[[str], None]


def seed_range(seeds: list[Any]) -> list[int]:
    """Expands `[lo, hi]` (inclusive, hi > lo + 1) or returns the explicit list as ints."""
    if len(seeds) == 2 and seeds[1] > seeds[0] + 1:
        return list(range(int(seeds[0]), int(seeds[1]) + 1))
    return [int(x) for x in seeds]


def dev_seeds(spec: Spec) -> list[int]:
    """The spec's seeds; only the DEV block 3000000-3000199 is allowed here."""
    seeds = seed_range(spec["seeds"])
    bad = [x for x in seeds if x not in DEV_SEEDS]
    if bad:
        raise ValueError(f"seeds outside the DEV block 3000000-3000199: {bad[:5]}")
    return seeds


def _valid_kappa(k: Any) -> bool:
    if k is None or isinstance(k, bool):
        return False
    return math.isfinite(float(k)) and float(k) >= 0


def kappas(spec: Spec) -> list[float]:
    """The spec's observation-noise levels; required, finite and non-negative."""
    if "kappas" not in spec:
        raise ValueError(
            'spec must give "kappas" (R-24 observation noise, e.g. [0.3]; [0] = noiseless)'
        )
    ks = spec["kappas"]
    if isinstance(ks, str | bytes) or not hasattr(ks, "__iter__") or not len(ks):
        raise ValueError(f'"kappas" must be a non-empty list of numbers, got {ks!r}')
    for k in ks:
        if not _valid_kappa(k):
            raise ValueError(f'"kappas" entries must be finite numbers >= 0, got {k!r}')
    return [float(k) for k in ks]


def lams_for(world: str, lams: Iterable[Any]) -> list[float | None]:
    """E4 coupling values of a world: the given list for E4, `[None]` elsewhere."""
    return [float(x) for x in lams] if world == "E4" else [None]


def dataset_groups(spec: Spec) -> list[DatasetKey]:
    """Ordered dataset keys (world, regime, lam, n, seed, kappa)."""
    ks = kappas(spec)
    seeds = dev_seeds(spec)
    out: list[DatasetKey] = []
    for w in spec["worlds"]:
        for r in (r for r in spec["regimes"] if r in REGIMES_OF[w]):
            for lam in lams_for(w, spec.get("lams", [1.0])):
                out += [(w, r, lam, int(n), s, k) for k in ks for n in spec["ns"] for s in seeds]
    return out


def _kappa_part(kappa: float | None) -> str:
    return "" if kappa is None else f"k{kappa:g}|"


def _lam_part(lam: float | None) -> str:
    return "" if lam is None else f"lam{lam:g}|"


def job_key(method: str, dataset: DatasetKey) -> str:
    """`<method>|<world>|<regime>|[lam<l>|][k<kappa>|]n<n>|s<seed>` of a dataset key."""
    w, r, lam, n, s, kappa = dataset
    return f"{method}|{w}|{r}|{_lam_part(lam)}{_kappa_part(kappa)}n{n}|s{s}"


def cell_key(w: str, r: str, lam: float | None, n: int, kappa: float | None = None) -> str:
    """`<world>|<regime>|[lam<l>|][k<kappa>|]n<n>`."""
    return f"{w}|{r}|{_lam_part(lam)}{_kappa_part(kappa)}n{n}"


def all_keys(spec: Spec) -> list[str]:
    """Every job key of the spec, dataset-major."""
    return [job_key(m, g) for g in dataset_groups(spec) for m in spec["methods"]]


def method_config(spec: Spec, method: str, dataset: DatasetKey) -> dict[str, Any]:
    """Arm config of a cell: `default`, then the kappa-free cell key, then the key with kappa."""
    w, r, lam, n, _, kappa = dataset
    c = spec.get("configs", {}).get(method, {})
    out = {**c.get("default", {}), **c.get(cell_key(w, r, lam, n), {})}
    if kappa is not None:
        out.update(c.get(cell_key(w, r, lam, n, kappa), {}))
    return out


def registry_resolve(ref: str) -> Any:
    """Resolves a registry name through the methods registry."""
    return resolve(ref)


def load_method(ref: str, resolve: Resolver = registry_resolve) -> Method:
    """Returns an instance of the arm `ref` (a class is instantiated, an instance is returned)."""
    obj = resolve(ref)
    return obj() if isinstance(obj, type) else obj


def _generate(dataset: DatasetKey) -> tuple[Any, Any, float]:
    w, r, lam, n, s, kap = dataset
    c0 = time.process_time()
    ds, truth = generate_dataset(w, r, n, s, lam=1.0 if lam is None else lam, kappa=kap)
    return ds, truth, time.process_time() - c0


def run_part(
    spec: Spec,
    part: tuple[int, int],
    out: Path,
    resolve: Resolver = registry_resolve,
    log: Log = print,
) -> int:
    """Runs the dataset groups `index::count` of `part = (index, count)`, one record per job.

    Jobs already in `out` without error are skipped, so a cut run resumes.

    Returns:
        The number of jobs run.
    """
    done = done_keys(out)
    methods: dict[str, Method] = {}
    n_run = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for g in dataset_groups(spec)[part[0] :: part[1]]:
            todo = [m for m in spec["methods"] if job_key(m, g) not in done]
            if not todo:
                continue
            ds, truth, gen = _generate(g)
            for m in todo:
                if m not in methods:
                    methods[m] = load_method(m, resolve)
                job = Job(job_key(m, g), ds, truth, method_config(spec, m, g), gen)
                rec = run_one(methods[m], job)
                append_record(fh, rec)
                n_run += 1
                status = "ERROR" if rec["error"] else "ok"
                log(f"[xm] {job.key} cpu {rec.get('cpu_s', float('nan')):.2f}s {status}")
    return n_run


def _best_records(inputs: Iterable[Path]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for path in inputs:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                old = best.get(r["key"])
                if old is None or old.get("error") is not None or r.get("error") is None:
                    best[r["key"]] = r
    return best


def _mean(vals: Iterable[Any]) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _cell_summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    sc = [r["scores"] for r in rs]
    return {
        "n_jobs": len(rs),
        **{m: _mean([s[m] for s in sc]) for m in SCORE_MEANS},
        "placebo_declared_total": int(sum(s["placebo_declared"] for s in sc)),
        "placebo_conf_declared_total": int(sum(s.get("placebo_conf_declared", 0) for s in sc)),
        "n_not_testable_true_total": int(sum(s.get("n_not_testable_true", 0) for s in sc)),
        "n_not_testable_null_total": int(sum(s.get("n_not_testable_null", 0) for s in sc)),
        "cpu_s_mean": _mean([r["cpu_s"] for r in rs]),
        "peak_rss_mb_max": max((r["peak_rss_mb"] or 0) for r in rs),
    }


def _per_cell(best: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in best.values():
        if r.get("error") is None:
            j = r["job"]
            ck = cell_key(j["world"], j["regime"], j["lam"], j["n"], j.get("kappa"))
            cells[f"{r['key'].split('|')[0]}|{ck}"].append(r)
    return {c: _cell_summary(rs) for c, rs in sorted(cells.items())}


def merge(spec: Spec, inputs: Iterable[Path], out: Path) -> dict[str, Any]:
    """Merges part files: one record per key, sorted by key.

    A clean record beats an error record and later beats earlier. Writes `out` and
    `<out>.summary.json` (completeness and per-cell means).

    Returns:
        The summary.
    """
    best = _best_records(inputs)
    want = all_keys(spec)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for k in sorted(best):
            fh.write(json.dumps(best[k], allow_nan=False) + "\n")
    summary = {
        "n_expected": len(want),
        "n_records": len(best),
        "missing": sorted(set(want) - set(best)),
        "errors": sorted(k for k, r in best.items() if r.get("error") is not None),
        "unexpected": sorted(set(best) - set(want)),
        "per_cell": _per_cell(best),
    }
    Path(f"{out}.summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8", newline="\n"
    )
    return summary

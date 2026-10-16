"""One job record: run a method on a dataset, score it, make it JSON-safe."""

import dataclasses
import json
import math
import platform
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.data import Dataset, Method, Result, Truth
from oran_causal.core.scoring import score
from oran_causal.experiments.harness.isolation import PeakRSS
from oran_causal.sim.controlled.datasets.generate import dataset_hash

RUNNER_VERSION = "oran-causal-runner/1"

type Record = dict[str, Any]


def clean(x: Any) -> Any:
    """JSON-safe copy: NaN and inf become None, numpy scalars and arrays become Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def versions() -> dict[str, str]:
    """Interpreter, numpy, platform and runner versions stamped on every record."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "runner": RUNNER_VERSION,
    }


def job_fields(ds: Dataset) -> dict[str, Any]:
    """The `job` block of a record: the dataset coordinates."""
    noise = ds.meta.get("obs_noise") or {}
    return {
        "world": ds.world,
        "regime": ds.regime,
        "lam": ds.meta.get("lam"),
        "n": ds.n,
        "seed": ds.seed,
        "kappa": float(noise.get("kappa", 0.0)),
    }


def edge_dicts(result: Result) -> list[dict[str, Any]]:
    """The `edges` list of a record."""
    return [
        {
            "source": e.source,
            "target": e.target,
            "score": e.score,
            "p": e.p,
            "sign": e.sign,
            "declared": e.declared,
        }
        for e in result.edges
    ]


def _result_fields(result: Result, ds: Dataset, truth: Truth) -> Record:
    return {
        "version": result.version,
        "method": result.method,
        "method_cpu_s": result.cpu_s,
        "result_config": result.config,
        "notes": result.notes,
        "edges": edge_dicts(result),
        "scores": score(result, truth, candidates=ds.candidates, kpi_sources=ds.kpi_names),
    }


@dataclasses.dataclass(frozen=True)
class Job:
    """One arm on one dataset.

    Attributes:
        key: The job key.
        ds: The dataset.
        truth: Ground truth of the dataset's (world, regime), read only by the scorer.
        config: The arm config of this cell.
        gen_cpu_s: CPU seconds spent generating the dataset.
    """

    key: str
    ds: Dataset
    truth: Truth
    config: dict[str, Any]
    gen_cpu_s: float


def run_one(method: Method, job: Job) -> Record:
    """Runs one job and returns its JSON-safe record.

    A failing method gives a record whose `error` holds the traceback text.
    """
    ds, truth, config = job.ds, job.truth, job.config
    rec: Record = {
        "key": job.key,
        "method": getattr(method, "name", None),
        "job": job_fields(ds),
        "dataset_sha256": dataset_hash(ds),
        "gen_cpu_s": job.gen_cpu_s,
        "config": config,
        "versions": versions(),
        "error": None,
    }
    try:
        with PeakRSS() as mem:
            c0, w0 = time.process_time(), time.perf_counter()
            result = method.run(ds, config)
            rec["cpu_s"], rec["wall_s"] = time.process_time() - c0, time.perf_counter() - w0
        rec.update(peak_rss_mb=mem.mb, rss_scope=mem.scope)
        rec.update(_result_fields(result, ds, truth))
    except Exception:
        rec["error"] = traceback.format_exc()
    return clean(rec)


def read_records(path: Path) -> Iterable[Record]:
    """Parseable records of a JSONL file; a partial last line is skipped."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def done_keys(path: Path) -> set[str]:
    """Keys of the records in `path` that ran without error."""
    return {r["key"] for r in read_records(path) if r.get("error") is None}


def append_record(fh: Any, record: Record) -> None:
    """Writes one record line and flushes it."""
    fh.write(json.dumps(record, allow_nan=False) + "\n")
    fh.flush()

"""Reading merged campaign record sets and the per-record helpers shared by the analyses.

A record is one JSON object per campaign unit `(arm, world, regime, lam, n, kappa, seed)` with
the fields `key`, `job`, `role`, `status`, `edges`, `notes`, `cpu_s`, `code`, `run_mode`,
`integrity`, `pkgs`, `host` and `dataset_sha256`. Record sets are `.jsonl` or `.jsonl.gz`
files, read as streams.
"""

import gzip
import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from oran_causal.core.data import EdgeResult, Result

Record = dict[str, Any]
Unit = dict[str, Any]
PathLike = str | Path


def iter_lines(path: PathLike) -> Iterator[str]:
    """Yields the non-blank lines of a plain or gzip-compressed text file.

    Args:
        path: File path; a `.gz` suffix selects gzip.

    Yields:
        One line per call, including its newline.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def iter_records(path: PathLike) -> Iterator[Record]:
    """Streams the records of one file, skipping lines that are not valid JSON."""
    for line in iter_lines(path):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def record_order(record: Record) -> tuple[str, str]:
    """Sort key that makes record order independent of file order."""
    return record.get("key", ""), json.dumps(record, sort_keys=True)


def load_records(path: PathLike) -> list[Record]:
    """Loads one record file, sorted by key."""
    return sorted(iter_records(path), key=record_order)


def load_many(paths: Iterable[PathLike]) -> list[Record]:
    """Loads several record files into one list sorted by key."""
    return sorted(itertools.chain.from_iterable(iter_records(p) for p in paths), key=record_order)


def file_sha256(path: PathLike) -> str:
    """Returns the sha256 of a file with CRLF normalised to LF, the freeze-commit hash rule."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def kappa_part(kappa: float | None) -> str:
    """Returns the kappa segment of a key, empty for None."""
    return "" if kappa is None else f"k{kappa:g}|"


def cell_key(world: str, regime: str, lam: float | None, n: int, kappa: float | None = None) -> str:
    """Returns the key of a measurement cell, for example `E4|R3|lam1|k0.25|n1000`."""
    lam_part = "" if lam is None else f"lam{lam:g}|"
    return f"{world}|{regime}|{lam_part}{kappa_part(kappa)}n{n}"


def job_key(
    arm: str,
    world: str,
    regime: str,
    lam: float | None,
    n: int,
    seed: int,
    kappa: float | None = None,
) -> str:
    """Returns the key of one campaign unit: the cell key with the arm in front, the seed behind."""
    lam_part = "" if lam is None else f"lam{lam:g}|"
    return f"{arm}|{world}|{regime}|{lam_part}{kappa_part(kappa)}n{n}|s{seed}"


def lam_of(value: Any) -> float | None:
    """Converts a recorded lambda to float, keeping None."""
    return None if value is None else float(value)


def cell_of(world: str, regime: str, lam: Any, n: int, kappa: float) -> str:
    """Returns the cell key of a unit, normalising the types read from JSON."""
    return cell_key(world, regime, lam_of(lam), int(n), float(kappa))


def status_of(record: Record) -> str:
    """Returns the record status, deriving `ok` or `error` from `error` when absent."""
    return record.get("status") or ("ok" if record.get("error") is None else "error")


def arm_of(record: Record) -> str:
    """Returns the arm name of a record, falling back to the first segment of its key."""
    return record.get("arm") or str(record.get("key", "")).split("|")[0]


def unit_of(record: Record) -> Unit:
    """Returns the `(arm, world, regime, lam, n, kappa, seed, role)` fields of a record."""
    job = record["job"]
    return {
        "arm": arm_of(record),
        "world": job["world"],
        "regime": job["regime"],
        "lam": lam_of(job.get("lam")),
        "n": int(job["n"]),
        "kappa": float(job.get("kappa") or 0.0),
        "seed": int(job["seed"]),
        "role": record.get("role") or "measure",
    }


def to_result(record: Record) -> Result:
    """Rebuilds the method `Result` stored in a record."""
    edges = tuple(
        EdgeResult(
            e["source"],
            e["target"],
            math.nan if e.get("score") is None else float(e["score"]),
            e.get("p"),
            int(e.get("sign") or 0),
            bool(e.get("declared")),
        )
        for e in record.get("edges", [])
    )
    return Result(
        record.get("method") or "?",
        record.get("version") or "?",
        edges,
        float(record.get("method_cpu_s") or 0.0),
        dict(record.get("config") or {}),
        dict(record.get("notes") or {}),
    )

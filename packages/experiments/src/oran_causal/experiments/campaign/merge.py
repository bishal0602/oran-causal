"""Streaming merge of campaign shard files.

One record per key: ok > infeasible > error, later wins within a class; duplicates and conflicts
(two ok records of one key with different declarations on the same dataset hash) are counted;
each dataset's arms are kept on one platform where possible.
"""

import dataclasses
import gzip
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO, Any

from oran_causal.experiments.campaign.integrity import MergeTally, accept_rule, record_status
from oran_causal.experiments.campaign.spec import ProtocolTexts, expand, is_eval

RANK = {"ok": 2, "infeasible": 1, "error": 0}

type Record = dict[str, Any]
type Accept = Callable[[Record], bool]


def dataset_of(r: Record) -> tuple[Any, ...]:
    """Dataset coordinates of a record's job."""
    j = r.get("job") or {}
    return (
        j.get("world"),
        j.get("regime"),
        j.get("lam"),
        j.get("n"),
        j.get("seed"),
        j.get("kappa"),
    )


def platform_of(r: Record) -> str | None:
    """Host platform tag of a record."""
    return (r.get("host") or {}).get("platform")


@dataclasses.dataclass
class Entry:
    """Index entry of one record line; the record itself stays on disk."""

    path: int
    offset: int
    length: int
    key: str
    status: str
    platform: str | None
    dataset: tuple[Any, ...]
    sha: str | None
    dhash: int | None


def _entry(pi: int, start: int, raw: bytes, r: Record) -> Entry:
    edges = r.get("edges")
    dhash = hash(tuple(bool(e.get("declared")) for e in edges)) if edges else None
    return Entry(
        pi,
        start,
        len(raw),
        r["key"],
        record_status(r),
        platform_of(r),
        dataset_of(r),
        r.get("dataset_sha256"),
        dhash,
    )


def _parse(raw: bytes, stats: dict[str, Any]) -> Record | None:
    try:
        r = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        stats["bad_lines"] += 1
        return None
    if not isinstance(r, dict) or r.get("key") is None:
        stats["bad_lines"] += 1
        return None
    return r


def _index(paths: Sequence[Path], accept: Accept | None, stats: dict[str, Any]) -> Iterator[Entry]:
    for pi, path in enumerate(paths):
        offset = 0
        with path.open("rb") as fh:
            for raw in fh:
                start, offset = offset, offset + len(raw)
                if not raw.strip():
                    continue
                stats["lines"] += 1
                r = _parse(raw, stats)
                if r is None:
                    continue
                if accept is not None and not accept(r):
                    stats["foreign"] += 1
                    continue
                yield _entry(pi, start, raw, r)


def _new_stats() -> dict[str, Any]:
    return {
        "lines": 0,
        "bad_lines": 0,
        "duplicates": 0,
        "foreign": 0,
        "conflicts": [],
        "platform_switched_datasets": 0,
        "mixed_platform_datasets": [],
    }


def _keep_best(
    entries: Iterator[Entry], stats: dict[str, Any]
) -> tuple[dict[str, Entry], dict[str, list[Entry]]]:
    best: dict[str, Entry] = {}
    alts: dict[str, list[Entry]] = defaultdict(list)
    for e in entries:
        old = best.get(e.key)
        if old is None:
            best[e.key] = e
            continue
        stats["duplicates"] += 1
        if old.status == e.status == "ok" and old.sha == e.sha and old.dhash != e.dhash:
            stats["conflicts"].append(e.key)
        if RANK[e.status] >= RANK[old.status]:
            best[e.key] = e
            alts[e.key].append(old)
        else:
            alts[e.key].append(e)
    return best, alts


def _one_platform(
    best: dict[str, Entry], alts: dict[str, list[Entry]], keys: list[str]
) -> dict[str, Entry] | None:
    plats = {best[k].platform for k in keys}
    candidates = {k: [best[k], *alts.get(k, [])] for k in keys}
    counts = {p: sum(best[k].platform == p for k in keys) for p in plats}
    for p in sorted(plats, key=lambda q: (-counts[q], str(q))):
        pick = {
            k: next((x for x in candidates[k] if x.platform == p and x.status != "error"), None)
            for k in keys
        }
        if all(pick.values()):
            return {k: v for k, v in pick.items() if v is not None}
    return None


def _apply_platform_rule(
    best: dict[str, Entry], alts: dict[str, list[Entry]], stats: dict[str, Any]
) -> None:
    by_ds: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for k, e in best.items():
        by_ds[e.dataset].append(k)
    for dsk, keys in sorted(by_ds.items(), key=lambda t: str(t[0])):
        if len({best[k].platform for k in keys}) <= 1:
            continue
        pick = _one_platform(best, alts, keys)
        if pick is None:
            stats["mixed_platform_datasets"].append("|".join(str(x) for x in dsk))
        else:
            best.update(pick)
            stats["platform_switched_datasets"] += 1


def merge_index(
    paths: Sequence[Path], accept: Accept | None = None
) -> tuple[dict[str, Entry], dict[str, Any]]:
    """Applies the merge rules on index entries only (a few hundred bytes per record).

    Returns:
        The chosen entry per key and the merge statistics.
    """
    stats = _new_stats()
    best, alts = _keep_best(_index(paths, accept, stats), stats)
    _apply_platform_rule(best, alts, stats)
    return best, stats


def read_entry(fhs: Sequence[IO[bytes]], e: Entry) -> Record:
    """Loads the record an index entry points at."""
    fhs[e.path].seek(e.offset)
    return json.loads(fhs[e.path].read(e.length))


def merge_records(
    paths: Sequence[Path], accept: Accept | None = None
) -> tuple[dict[str, Record], dict[str, Any]]:
    """`merge_index` with the chosen records loaded (small inputs)."""
    best, stats = merge_index(paths, accept)
    fhs = [p.open("rb") for p in paths]
    try:
        return {k: read_entry(fhs, e) for k, e in best.items()}, stats
    finally:
        for fh in fhs:
            fh.close()


def _open_text_out(out: Path) -> IO[str]:
    if out.suffix == ".gz":
        return gzip.open(out, "wt", encoding="utf-8", newline="\n")
    return out.open("w", encoding="utf-8", newline="\n")


def summary_path(out: Path) -> Path:
    """`<out without .gz>.summary.json`."""
    base = out.with_suffix("") if out.suffix == ".gz" else out
    return Path(f"{base}.summary.json")


def _write_merged(
    paths: Sequence[Path], best: dict[str, Entry], out: Path, tally: MergeTally
) -> None:
    fhs = [p.open("rb") for p in paths]
    try:
        with _open_text_out(out) as fo:
            for k in sorted(best):
                r = read_entry(fhs, best[k])
                fo.write(json.dumps(r, allow_nan=False) + "\n")
                tally.add(r)
    finally:
        for fh in fhs:
            fh.close()


def merge(
    spec: dict[str, Any], inputs: Sequence[Path], out: Path, texts: ProtocolTexts | None = None
) -> dict[str, Any]:
    """Merges shard files for `spec` into `out` (key-sorted; `.gz` writes gzip).

    EVAL specs keep only EVAL records of the spec's protocol and spec; DEV specs drop EVAL
    records. An EVAL merge mixing cost tables or part counts is refused.

    Returns:
        The summary, also written to `summary_path(out)`.
    """
    best, stats = merge_index(inputs, accept_rule(spec))
    want = {u.key for u in expand(spec, texts)}
    tally = MergeTally()
    _write_merged(inputs, best, out, tally)
    summary = tally.summary(want, stats, "eval" if is_eval(spec) else "dev")
    sp = summary_path(out)
    sp.write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    mixed = [f for f in ("cost_table_sha256", "parts") if len(summary[f]) > 1]
    if is_eval(spec) and mixed:
        out.unlink()
        mix = ", ".join(f"{f} {summary[f]}" for f in mixed)
        raise ValueError(
            f"merge refused: EVAL records mix {mix} (one cost table and part count per EVAL run; "
            f"summary kept at {sp})"
        )
    return summary


def iter_jsonl(path: Path) -> Iterator[Record]:
    """Records of a (possibly gzipped) JSONL file, one at a time; broken lines skipped."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

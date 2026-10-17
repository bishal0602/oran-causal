"""Shared pieces of the system-level study drivers.

Frozen-document checks, the seed registry and seed blocks, JSONL record I/O, run headers, the
resumable shard loop and the episode outcome fields every job record carries.
"""

import dataclasses
import hashlib
import itertools
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.experiments.campaign.provenance import code_info
from oran_causal.experiments.harness.isolation import PeakRSS
from oran_causal.experiments.launch.fingerprint import numeric_env
from oran_causal.experiments.paths import MANIFESTS, PROTOCOLS
from oran_causal.sim.system import config as C
from oran_causal.sim.system.decision.scenario import LOAD_FACTOR
from oran_causal.sim.system.env import E6Env

type Record = dict[str, Any]

SEED_REGISTRY = MANIFESTS / "seeds.json"
LOAD_FACTOR_L40 = LOAD_FACTOR["L40"]
OUTCOME_FIELDS = (
    "viol_ue_s",
    "ue_s",
    "ll_viol",
    "embb_viol",
    "outage_viol",
    "severe",
    "energy_j",
    "rlf",
    "ho",
    "pingpong",
    "prot_viol",
    "prot_ue_s",
    "lowsinr_ue_s",
)
SUM_FIELDS = (
    "prot_viol",
    "prot_ue_s",
    "energy_j",
    "viol_ue_s",
    "ue_s",
    "nonprot_embb_viol",
    "ll_viol",
    "rlf",
)
DEV_SEEDS = 31
MAPGATE = "cdd_oran/decision/mapgate.py"
CERTSAFE_CODE = "cdd_oran/decision/certsafe.py"
CONF_ANALYZER = "scratchpad/e6_dev/e6p_conf_analyze.py"
CONF_DRIVER = "scratchpad/e6_dev/e6p_conf.py"
CS_ANALYZER = "scratchpad/e6_dev/e6p_certsafe_analyze.py"
CS_DRIVER = "scratchpad/e6_dev/e6p_certsafe.py"
LEGACY_SHA256 = {
    MAPGATE: "64ad9d1beb336681d3f384b5d9b1871615bc317f78c0eaa6ca0d9b9dcd426dd9",
    CONF_ANALYZER: "3d1a0965ce986889bb0dd19a228f0469c29cafd0884e23fe4ed9ffab17e8e000",
    CONF_DRIVER: "74ae5309b9373632a03a17076ce5ba1f465be32de9b3e32d8cdb0131e786ac9d",
    CERTSAFE_CODE: "f838af6a39691d5882c9cc8109db25866d40c1db727370fdb980872befbf5daf",
    CS_ANALYZER: "0adde7aa181e2874b3394636932bb0df8ef71ae58c14aa628dbe9afbe9ee6a33",
    CS_DRIVER: "41d6d173296345fadf72e49d3e36cf1b49b25168b14966c93943fc83144b7e84",
}


def sha_lf(path: Path) -> str:
    """Returns the sha256 of a file with CRLF normalised to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclasses.dataclass(frozen=True)
class FrozenDoc:
    """A frozen document: its legacy name, its location here and its pinned LF sha256.

    Attributes:
        name: The document's path in the original repository (recorded in statuses).
        path: Its location in this workspace.
        want: The pinned LF sha256.
        needs_line: The document must also carry a "FROZEN: yes" line.
    """

    name: str
    path: Path
    want: str
    needs_line: bool = True

    def status(self) -> dict[str, Any]:
        """Returns `{doc, exists, line, sha256, want, frozen}` of the document."""
        if not self.path.exists():
            return {
                "doc": self.name,
                "exists": False,
                "sha256": None,
                "want": self.want,
                "frozen": False,
            }
        text = self.path.read_bytes().decode("utf-8", "replace")
        line = next(
            (s.strip() for s in text.splitlines() if s.strip().startswith("FROZEN:")),
            "FROZEN: ?",
        )
        sha = sha_lf(self.path)
        line_ok = line.lower().startswith("frozen: yes") or not self.needs_line
        return {
            "doc": self.name,
            "exists": True,
            "line": line,
            "sha256": sha,
            "want": self.want,
            "frozen": bool(sha == self.want and line_ok),
        }


def protocol_doc(study: str, name: str, want: str, needs_line: bool = True) -> FrozenDoc:
    """Returns the `FrozenDoc` of `protocols/<study>/<name>` (legacy `docs/benchmark/<name>`)."""
    return FrozenDoc(f"docs/benchmark/{name}", PROTOCOLS / study / name, want, needs_line)


@dataclasses.dataclass(frozen=True)
class Block:
    """An inclusive seed range."""

    lo: int
    hi: int

    def __contains__(self, seed: object) -> bool:
        return isinstance(seed, int) and self.lo <= seed <= self.hi

    def overlaps(self, other: Block) -> bool:
        """Returns whether two ranges share a seed."""
        return not (self.hi < other.lo or self.lo > other.hi)

    def within(self, other: Block) -> bool:
        """Returns whether this range lies inside `other`."""
        return other.lo <= self.lo and self.hi <= other.hi

    @classmethod
    def sized(cls, base: int, n: int) -> Block:
        """Returns the range of `n` seeds from `base`."""
        return cls(base, base + n - 1)


def check_seed(seed: int, allowed: Block, forbidden: Iterable[Block] = ()) -> int:
    """Returns `seed` after checking it lies in `allowed` and outside every forbidden block.

    Raises:
        ValueError: The seed is outside `allowed` or inside a forbidden block.
    """
    seed = int(seed)
    if seed not in allowed:
        raise ValueError(f"seed {seed} outside the block [{allowed.lo}, {allowed.hi}]")
    for b in forbidden:
        if seed in b:
            raise ValueError(f"seed {seed} inside a forbidden block [{b.lo}, {b.hi}]")
    return seed


def cfg_seed(seed: int, smoke: bool) -> int:
    """Returns the plant seed: the protocol seed, or a DEV seed (seed % 31) under smoke."""
    return seed % DEV_SEEDS if smoke else seed


def _int_pairs(o: Any) -> Iterator[tuple[int, int]]:
    if isinstance(o, list) and len(o) == 2 and all(isinstance(v, int) for v in o):
        yield o[0], o[1]
    elif isinstance(o, list):
        for v in o:
            yield from _int_pairs(v)
    elif isinstance(o, dict):
        for v in o.values():
            yield from _int_pairs(v)


@dataclasses.dataclass(frozen=True)
class SeedRegistry:
    """The seed registry (`manifests/seeds.json`, the old `SEED_REGISTRY.json`)."""

    path: Path = SEED_REGISTRY

    @cached_property
    def doc(self) -> dict[str, Any]:
        """The parsed registry."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    def ranges(self, world: str = "E6") -> list[tuple[int, int]]:
        """Every `[lo, hi]` range registered anywhere under `world`."""
        return list(_int_pairs(self.doc.get(world, {})))

    def covers(self, block: Block, world: str = "E6") -> bool:
        """Returns whether one registered range of `world` holds `block`."""
        return any(lo <= block.lo and hi >= block.hi for lo, hi in self.ranges(world))

    def has_tags(self, tags: Iterable[int]) -> bool:
        """Returns whether every RNG stream tag is registered."""
        known = self.doc.get("rng_stream_tags", [])
        return all(t in known for t in tags)

    def e6(self, key: str, default: Any = None) -> Any:
        """Returns one E6 entry."""
        return (self.doc.get("E6") or {}).get(key, default)


@dataclasses.dataclass(frozen=True)
class Part:
    """Shard `index` of `count`: job `u` belongs to it iff `u % count == index`."""

    index: int = 0
    count: int = 1

    @classmethod
    def parse(cls, spec: str | None) -> Part:
        """Parses "i/k" (None = the whole run)."""
        if spec is None:
            return cls()
        i, k = (int(v) for v in spec.split("/"))
        if not 0 <= i < k:
            raise ValueError(f"part {spec}: need 0 <= i < k")
        return cls(i, k)

    def owns(self, u: int) -> bool:
        """Returns whether job number `u` belongs to this shard."""
        return u % self.count == self.index

    def __str__(self) -> str:
        return f"{self.index}/{self.count}"


def to_json(x: Any) -> Any:
    """`json.dumps` default: numpy scalars and arrays to Python, anything else to its str."""
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def read_records(path: Path) -> list[Record]:
    """Returns the JSON lines of `path` (missing file: none; unparsable lines are skipped)."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_record(path: Path, rec: Record) -> None:
    """Appends one JSON line to `path`."""
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, default=to_json) + "\n")


def header(driver: str, stage: str, part: Part, smoke: bool, **extra: Any) -> Record:
    """Returns the header line of one run call.

    `code` is the commit of `campaign.provenance.code_info` (launcher environment, git, bundle
    manifest); `platform` repeats `numeric_env` under the key the referee analyses read.
    """
    code = code_info()
    env = numeric_env()
    return {
        "kind": "header",
        "driver": driver,
        "stage": stage,
        "part": str(part),
        "smoke": smoke,
        "code": {"git_head": code["commit"], "e6_dirty": code["dirty"]},
        "numeric_env": env,
        "platform": env,
        "scenario_version": C.SCENARIO_VERSION,
        "load_factor": LOAD_FACTOR_L40,
        "cpus": os.cpu_count(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }


def outcome(env: E6Env) -> dict[str, Any]:
    """Returns the episode's SLA totals, derived rates and the env's `st_*` counters."""
    sla, st = env.plant.sla, env.stats
    o: dict[str, Any] = {k: float(sla[k]) for k in OUTCOME_FIELDS}
    o.update(
        psvr=3600.0 * sla["prot_viol"] / max(sla["prot_ue_s"], 1e-9),
        svr=3600.0 * sla["viol_ue_s"] / max(sla["ue_s"], 1e-9),
        energy_kwh=sla["energy_j"] / 3.6e6,
        nonprot_embb_viol=float(sla["embb_viol"] - sla["prot_viol"]),
    )
    o.update({"st_" + k: v for k, v in st.items()})
    return o


def run_env(cfg: C.E6Config, arbiter: Any) -> E6Env:
    """Runs one `wg3` episode under `arbiter` (None = accept-all) and returns the env."""
    env = E6Env(cfg, log=False, wg3=True)
    while env.sec < env.total_s:
        env.step(arbiter)
    return env


def repro_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Max absolute difference of two outcomes over `SUM_FIELDS`."""
    return max(abs(float(a[f]) - float(b[f])) for f in SUM_FIELDS)


@dataclasses.dataclass(frozen=True)
class Job:
    """One resumable unit of a stage: its record key, protocol seed and runner."""

    key: tuple
    seed: int
    run: Callable[[], Record]


def timed(fn: Callable[[], Record]) -> Record:
    """Runs a job and adds `secs` (wall) and `rss_mb` (peak memory) to its record."""
    t = time.time()
    with PeakRSS() as rss:
        rec = fn()
    return rec | {"secs": round(time.time() - t, 1), "rss_mb": rss.mb}


def run_jobs(
    out: Path,
    head: Record,
    jobs: Sequence[Job],
    part: Part,
    smoke: bool,
    show: Sequence[str] = ("seed", "cpu_s", "secs"),
) -> int:
    """Runs a stage's jobs into `out` (JSONL) and returns how many ran.

    Writes the header, skips jobs whose key is already in `out` with the same smoke flag,
    runs the jobs of shard `part` (under `smoke`: only the first seed's jobs, any shard) and
    closes with a `close` line.
    """
    append_record(out, head)
    done = {
        tuple(r["key"])
        for r in read_records(out)
        if r.get("kind") in ("episode", "job") and r.get("smoke") == smoke
    }
    t0, n = time.time(), 0
    first = jobs[0].seed if jobs else None
    for u, job in enumerate(jobs):
        if smoke and job.seed != first:
            break
        if (not smoke and not part.owns(u)) or job.key in done:
            continue
        rec = timed(job.run)
        append_record(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in show}, default=to_json), flush=True)
    append_record(
        out,
        {
            "kind": "close",
            "stage": head["stage"],
            "part": str(part),
            "n_jobs": n,
            "n_total": len(jobs),
            "secs": round(time.time() - t0, 1),
        },
    )
    return n


def check_layout(stages: Mapping[str, Block], block: Block, forbidden: Iterable[Block]) -> None:
    """Checks that every stage range lies in `block`, outside the forbidden ones, and disjoint.

    Raises:
        ValueError: A stage range is out of place or two overlap.
    """
    forbidden = tuple(forbidden)
    for name, b in stages.items():
        if not b.within(block) or any(b.overlaps(f) for f in forbidden):
            raise ValueError(f"stage {name} {b} outside {block} or inside a forbidden block")
    rs = sorted(stages.values(), key=lambda b: b.lo)
    if any(a.overlaps(b) for a, b in itertools.pairwise(rs)):
        raise ValueError(f"stage ranges overlap: {rs}")

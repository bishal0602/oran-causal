"""Study 3 episode stages: the certified-safe referee.

Plant, referee constants and arm runner are Study 2's; the maps are Study 2's frozen maps
artifact (nothing is re-discovered). Stages and seeds (block 191000-191999, protocol
`E6P_CERTSAFE_PROTOCOL.md`):

| stage | seeds | jobs |
|---|---|---|
| dev | 191000-191039 | `CAL:allaccept` on every seed; `noarb` repro on 191000-191001 |
| eval | 191100-191259 | anchors, references, one job per certified-safe signature |

Records are schema `e6p-optaka2-rec/1`, sub "certsafe".
"""

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from oran_causal.analysis.system import study3 as astudy3
from oran_causal.experiments.paths import ARTIFACTS
from oran_causal.experiments.system import study2
from oran_causal.experiments.system.common import (
    CERTSAFE_CODE,
    LEGACY_SHA256,
    MAPGATE,
    Block,
    FrozenDoc,
    Job,
    Part,
    Record,
    SeedRegistry,
    check_layout,
    check_seed,
    header,
    protocol_doc,
    run_jobs,
)
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.scenario import PAIR, STRATUM

SUB = astudy3.SUB
BLOCK = Block(191000, 191999)
LAYOUT = {stage: Block.sized(*base_n) for stage, base_n in astudy3.LAYOUT.items()}
RESERVES = (Block(191040, 191099), Block(191260, 191999))
FORBIDDEN = (*study2.FORBIDDEN, Block(186000, 187999), Block(960000, 10**9), *RESERVES)
REPRO_SEEDS = (191000, 191001)
REFS = astudy3.REFS
PRIMARY_ARM = astudy3.PRIMARY_ARM
ALL_ARMS = astudy3.ALL_ARMS
PROTOCOL = protocol_doc(
    "study3",
    "E6P_CERTSAFE_PROTOCOL.md",
    "cdd2c7e63fe3b81efe07c6cedea331b81124eca078f056e87f3008e6d2850b4a",
)
CERTSAFE = FrozenDoc(
    "docs/benchmark/artifacts/E6P_CERTSAFE.json",
    ARTIFACTS / "E6P_CERTSAFE.json",
    "303ac557be16ecd49723ee92850fde0e4e6d8c4c1cecde62aa0d0c48ec2da210",
    needs_line=False,
)

check_layout(LAYOUT, BLOCK, FORBIDDEN)
if len(rf.CS_ARMS) != 17 or len(ALL_ARMS) != 26 or rf.CAL_ARM in ALL_ARMS:
    raise ValueError("the 17 certified-safe arms and 26 named arms of the protocol")


def stage_seeds(stage: str) -> list[int]:
    """The protocol seeds of a stage."""
    b = LAYOUT[stage]
    return list(range(b.lo, b.hi + 1))


def check_stage_seed(seed: int, stage: str) -> int:
    """Returns `seed` after checking it lies in `stage`'s range and outside every forbidden block.

    Raises:
        ValueError: Unknown stage or a seed out of place.
    """
    if stage not in LAYOUT:
        raise ValueError(f"stage in {tuple(LAYOUT)}")
    check_seed(seed, BLOCK, FORBIDDEN)
    return check_seed(seed, LAYOUT[stage])


def load_artifact(path: Path = CERTSAFE.path) -> dict[str, Any]:
    """Tables, alias table and source maps of the frozen certsafe artifact (re-verified)."""
    return rf.load_certsafe(json.loads(path.read_text(encoding="utf-8")))


def cs_job(
    seed: int,
    arm: str,
    stage: str,
    art: Mapping[str, Any] | None,
    smoke: bool = False,
    short: float | None = None,
) -> Record:
    """One Study 3 referee job."""
    check_stage_seed(seed, stage)

    def arbiter(a: str, sd: int, w: float) -> Any:
        return rf.cs_arbiter(a, sd, w, art)

    alias = art["alias"] if art else None
    return study2.arm_job(seed, arm, arbiter, alias, smoke, short, sub=SUB, cs_stage=stage)


def eval_arms(alias: Mapping[str, Any]) -> list[str]:
    """Distinct EVAL arms: anchors, references and the artifact's certified-safe jobs."""
    return [*rf.ANCHORS, *REFS, *(a for a in alias["jobs"] if a not in rf.ANCHORS)]


def jobs(
    stage: str,
    art: Mapping[str, Any] | None,
    smoke: bool = False,
    short: float | None = None,
    arms: frozenset[str] | None = None,
) -> list[Job]:
    """The (seed, arm) jobs of a stage in shard order (seed-major).

    Raises:
        ValueError: Unknown stage, or eval without the artifact.
    """
    if stage == "dev":
        pairs = [
            (sd, a)
            for sd in stage_seeds("dev")
            for a in ((rf.CAL_ARM, "noarb") if sd in REPRO_SEEDS else (rf.CAL_ARM,))
        ]
    elif stage == "eval":
        if art is None:
            raise ValueError("eval jobs need the certsafe artifact")
        pairs = [(sd, a) for sd in stage_seeds("eval") for a in eval_arms(art["alias"])]
    else:
        raise ValueError(f"stage in {tuple(LAYOUT)}")
    return [
        Job((sd, a), sd, lambda sd=sd, a=a: cs_job(sd, a, stage, art, smoke, short))
        for sd, a in pairs
        if arms is None or a in arms
    ]


def registry_status(reg: SeedRegistry | None = None) -> dict[str, bool]:
    """Whether the registry holds the block, tag 6624 and both stage ranges."""
    reg = reg or SeedRegistry()
    blk = reg.e6("e6p_certsafe_episodes", [])
    note = str(reg.e6("e6p_certsafe_note", ""))
    return {
        "block_191000_191999": any(lo <= BLOCK.lo and hi >= BLOCK.hi for lo, hi in blk),
        "tag_6624": reg.has_tags([6624]),
        "layout": all(f"{b.lo}-{b.hi}" in note for b in LAYOUT.values()),
    }


def guard(
    stage: str, smoke: bool, art_doc: FrozenDoc = CERTSAFE, maps_doc: FrozenDoc = study2.MAPS
) -> dict[str, Any]:
    """Checks the protocol's refusals (smoke runs are exempt) and returns the statuses.

    Raises:
        RuntimeError: Unfrozen protocol, changed maps artifact, eval without the frozen
            certsafe artifact, an unregistered layout, or a run off Linux.
    """
    if stage not in LAYOUT:
        raise ValueError(f"stage in {tuple(LAYOUT)}")
    fz, reg, ms = PROTOCOL.status(), registry_status(), maps_doc.status()
    ar = art_doc.status() if stage == "eval" else None
    st = {"freeze": fz, "registry": reg, "maps": ms, "artifact": ar}
    if smoke:
        return st
    if not fz["frozen"]:
        raise RuntimeError(f"stage {stage} needs the frozen protocol: {fz}")
    if not ms["frozen"]:
        raise RuntimeError(f"the Study 2 maps artifact must be the frozen one: {ms}")
    if ar is not None and not ar["frozen"]:
        raise RuntimeError(f"stage eval needs the frozen certsafe artifact: {ar}")
    if not all(reg.values()):
        raise RuntimeError(f"seed block / tag / layout not registered: {reg}")
    if not sys.platform.startswith("linux"):
        raise RuntimeError(f"stage {stage}: Linux numerics only; platform {sys.platform}")
    return st


def run(
    stage: str,
    out: Path,
    part: Part | None = None,
    smoke: bool = False,
    short: float | None = None,
    arms: frozenset[str] | None = None,
    art_doc: FrozenDoc = CERTSAFE,
    maps_doc: FrozenDoc = study2.MAPS,
) -> int:
    """Runs a Study 3 stage (shard `part`) into `out` and returns the number of jobs run.

    `art_doc` and `maps_doc` are the pinned certsafe and maps artifacts (the frozen ones by
    default).

    Raises:
        ValueError: `short` without `smoke`.
    """
    if short and not smoke:
        raise ValueError("short is smoke-only")
    part = part or Part()
    st = guard(stage, smoke, art_doc, maps_doc)
    art = load_artifact(art_doc.path) if stage == "eval" else None
    head = header(
        astudy3.DRIVER,
        stage,
        part,
        smoke,
        cs_stage=stage,
        mapgate_sha256=LEGACY_SHA256[MAPGATE],
        certsafe_sha256=LEGACY_SHA256[CERTSAFE_CODE],
        schema=study2.SCHEMA_EVAL,
        protocol=PROTOCOL.name,
        **st,
        artifact_source={"jobs": art["alias"]["jobs"], "alias_of": art["alias"]["alias_of"]}
        if art
        else None,
        consts={
            "pair": PAIR,
            "stratum": STRATUM,
            "T": study2.COLLECTION.T,
            "open_rule": rf.OPEN_RULE,
            "theta": rf.THETA,
            "k_conf": rf.K_CONF,
            "layout": {s: [b.lo, b.hi - b.lo + 1] for s, b in LAYOUT.items()},
            "repro_seeds": REPRO_SEEDS,
            "smoke": {"short": short} if smoke else None,
            "arms_filter": sorted(arms) if arms else None,
        },
    )
    js = jobs(stage, art, smoke, short, arms)
    show = ("cs_stage", "seed", "arm", "psvr", "rlf", "energy_j", "cpu_s", "secs")
    return run_jobs(out, head, js, part, smoke, show)

"""Study 1 episode stages: randomized unit collection and knockout ground truth (discovery v4).

Plant P3 surge-L40 (`scenario.make_cfg`), 120 s warm-up + 600 s scored. Each unit's mode is drawn
by the logging policy pi0 = `PI0_V4` (accept / reject 0.5 for ES, PowerES and SliceGuarantee) from
t = 0 (`arb_warmup_s` 0); the KPI tap counts warm-up seconds. Stages and seeds (block
188000-189999, protocol `E6P_DISCOVERY_PROTOCOL_V4.md` section 4):

| stage | seeds | records |
|---|---|---|
| dev | 188000-188019 | allowed before the freeze |
| eval | 188100-188699 | fold j // 120, slice60 j // 60, slice120 j // 120 |
| placebo | 188700-188739 | `PlaceboPolicy`: pi0 drawn and logged, accept applied |
| gt | 188800-188839 | knockout labels for units with t0 >= 90 |

Records are schema `e6p-disc-rec/1`, key `[stage, "v4", seed]`.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oran_causal.experiments.system.common import (
    Block,
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
from oran_causal.experiments.system.episode import (
    ARB_WARMUP_S,
    COUNT_ALL,
    OPEN_RULE,
    STEP_S,
    EpisodeSpec,
    body,
    collect,
    family_mode_counts,
    gt_consts,
    label_counts,
    sleep_counts,
)
from oran_causal.sim.system.decision.collect import (
    PI0_V4,
    Collection,
    PlaceboPolicy,
    RandomizedUnitPolicy,
)
from oran_causal.sim.system.decision.records import SCHEMA, SERIES_FIELDS, unit_record
from oran_causal.sim.system.decision.scenario import PAIR, STRATUM
from oran_causal.sim.system.decision.units import T_UNIT, Policy

SUB = "v4"
BLOCK = Block(188000, 189999)
LEGACY_BLOCK = Block(180000, 183999)
FORBIDDEN = (
    Block(150000, 150399),
    Block(155000, 155399),
    Block(160000, 179999),
    Block(180000, 187999),
    Block(190000, 190399),
)
STAGES = {
    "dev": Block.sized(188000, 20),
    "eval": Block.sized(188100, 600),
    "placebo": Block.sized(188700, 40),
    "gt": Block.sized(188800, 40),
}
FROZEN_STAGES = ("eval", "placebo", "gt")
PROTOCOL = protocol_doc(
    "study1",
    "E6P_DISCOVERY_PROTOCOL_V4.md",
    "6857466c91afa296a5cbaabd8fe0be9cb036dd35358736e855adfc540faf84ec",
    needs_line=False,
)
TAGS = {
    6612: "pi0 draws",
    6613: "label sampling",
    6614: "region probes",
    6615: "G0c bootstrap",
    6616: "CRT draws",
    6617: "GT bootstrap",
}
COLLECTION = Collection(
    T=T_UNIT, open_rule=OPEN_RULE, arb_warmup_s=ARB_WARMUP_S, count_all=COUNT_ALL
)
SLICE60, SLICE120 = 60, 120


check_layout(STAGES, BLOCK, FORBIDDEN)


@dataclasses.dataclass(frozen=True)
class Episode:
    """One protocol episode: record stage, seed and its index `j` in the stage."""

    stage: str
    seed: int
    j: int

    @property
    def key(self) -> tuple[str, str, int]:
        """The record key."""
        return (self.stage, SUB, self.seed)

    @property
    def fold(self) -> int | None:
        """EVAL fold j // 120 (None outside eval)."""
        return self.j // SLICE120 if self.stage == "eval" else None

    def slices(self) -> dict[str, int | None]:
        """EVAL slice indices of the 60- and 120-episode slices (None outside eval)."""
        ev = self.stage == "eval"
        return {
            "slice60": self.j // SLICE60 if ev else None,
            "slice120": self.j // SLICE120 if ev else None,
        }


def episodes(stage: str) -> list[Episode]:
    """The protocol episodes of `stage`, in shard order.

    Raises:
        ValueError: Unknown stage, or a seed outside the v4 block.
    """
    if stage not in STAGES:
        raise ValueError(f"stage in {tuple(STAGES)}")
    b = STAGES[stage]
    return [
        Episode(stage, check_seed(sd, BLOCK, FORBIDDEN), j)
        for j, sd in enumerate(range(b.lo, b.hi + 1))
    ]


def policy(stage: str) -> tuple[Callable[[int], Policy], str]:
    """The logging policy factory of a stage and its record name."""
    if stage == "placebo":
        return (lambda sd: PlaceboPolicy(sd, tables=PI0_V4)), "placebo"
    return (lambda sd: RandomizedUnitPolicy(sd, tables=PI0_V4)), "pi0"


def episode_job(
    ep: Episode, smoke: bool = False, short: float | None = None, max_labels: int | None = None
) -> Record:
    """Runs one collection episode and returns its `e6p-disc-rec/1` record."""
    factory, name = policy(ep.stage)
    spec = EpisodeSpec(
        seed=check_seed(ep.seed, BLOCK, FORBIDDEN),
        smoke=smoke,
        policy=factory,
        collection=COLLECTION,
        gt=ep.stage == "gt",
        short=short,
        max_labels=max_labels,
    )
    c = collect(spec, unit_record)
    counts: dict[str, Any] = label_counts(c)
    counts.update(
        units=len(c.urecs),
        by_family_mode=family_mode_counts(c.urecs, lambda u: u["knob"]),
        **sleep_counts(c.urecs),
    )
    return {
        "arb_warmup_s": ARB_WARMUP_S,
        "count_all": COUNT_ALL,
        **ep.slices(),
        "kind": "episode",
        "schema": SCHEMA,
        "key": list(ep.key),
        "stage": ep.stage,
        "sub": SUB,
        "seed": ep.seed,
        "cfg_seed": spec.cfg_seed,
        "j": ep.j,
        "fold": ep.fold,
        "smoke": smoke,
        "short": short,
        "policy": name,
        "pi0_table": PI0_V4,
        **body(c),
        "counts": counts,
    }


def registry_status(reg: SeedRegistry | None = None) -> dict[str, bool]:
    """Whether the registry holds both discovery blocks and the RNG tags."""
    reg = reg or SeedRegistry()
    return {
        "block_180000_183999": reg.covers(LEGACY_BLOCK),
        "tags_6612_6617": reg.has_tags(TAGS),
        "block_188000_189999": reg.covers(BLOCK),
    }


def guard(stage: str, smoke: bool) -> dict[str, Any]:
    """Checks the run preconditions and returns the freeze and registry status.

    Raises:
        RuntimeError: A frozen stage without the frozen protocol, or an unregistered block.
    """
    fz, reg = PROTOCOL.status(), registry_status()
    if smoke:
        return {"freeze": fz, "registry": reg}
    if stage in FROZEN_STAGES and not fz["frozen"]:
        raise RuntimeError(f"stage {stage} needs the frozen protocol: {fz}")
    if not all(reg.values()):
        raise RuntimeError(f"seed block / RNG tags not registered: {reg}")
    return {"freeze": fz, "registry": reg}


def consts(smoke: dict[str, Any] | None) -> dict[str, Any]:
    """The run constants recorded in the header."""
    return {
        "pair": PAIR,
        "stratum": STRATUM,
        "T": COLLECTION.T,
        "open_rule": OPEN_RULE,
        "step_s": STEP_S,
        "pi0_table": PI0_V4,
        "arb_warmup_s": ARB_WARMUP_S,
        "count_all": COUNT_ALL,
        "v4": {
            "block": [BLOCK.lo, BLOCK.hi],
            "forbidden": [[b.lo, b.hi] for b in FORBIDDEN],
            "stages": {s: [b.lo, b.hi] for s, b in STAGES.items()},
            "sub": SUB,
        },
        "gt": gt_consts(),
        "tags": TAGS,
        "series_fields": SERIES_FIELDS,
        "smoke": smoke,
    }


def run(
    stage: str,
    out: Path,
    part: Part | None = None,
    smoke: bool = False,
    short: float | None = None,
    max_labels: int | None = None,
) -> int:
    """Runs a Study 1 stage (shard `part`) into `out` and returns the number of episodes run.

    Raises:
        ValueError: `short` or `max_labels` without `smoke`.
    """
    if (short or max_labels) and not smoke:
        raise ValueError("short / max_labels are smoke-only")
    part = part or Part()
    st = guard(stage, smoke)
    head = header(
        "study1",
        stage,
        part,
        smoke,
        schema=SCHEMA,
        protocol=PROTOCOL.name,
        **st,
        consts=consts({"short": short, "max_labels": max_labels} if smoke else None),
    )
    jobs = [
        Job(ep.key, ep.seed, lambda ep=ep: episode_job(ep, smoke, short, max_labels))
        for ep in episodes(stage)
    ]
    return run_jobs(out, head, jobs, part, smoke, ("stage", "seed", "counts", "cpu_s", "secs"))

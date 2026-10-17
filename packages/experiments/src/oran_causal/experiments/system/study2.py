"""Study 2 episode stages: confounded incumbent logs and the map referees they drive.

Plant as Study 1. Collection runs the directional unit arbiter (a reject unit defers only its
opening direction) from t = 0 with the incumbent logging policy (per-unit probability rows).
Stages and seeds (block 186000-187999, protocol `E6P_CONFOUNDED_PROTOCOL.md` section 6):

| stage | seeds | jobs |
|---|---|---|
| dev_conf | 186080-186099 | collection (record stage "dev"); repro arms on 186080-186081 |
| disc | 186100-186699 | collection (record stage "eval", fold j // 60) |
| placebo | 186700-186899 | `PlaceboIncumbent` collection |
| gt | 186900-186939 | collection + knockout labels (t0 >= 90) |
| eval | 187000-187159 | every referee arm, one job per distinct map signature |

Collection records are schema `e6p-disc-rec/1` (sub "conf"); arm jobs `e6p-optaka2-rec/1`.
"""

import dataclasses
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from oran_causal.analysis.system import study2 as astudy2
from oran_causal.experiments.paths import ARTIFACTS
from oran_causal.experiments.system.common import (
    LEGACY_SHA256,
    LOAD_FACTOR_L40,
    MAPGATE,
    Block,
    FrozenDoc,
    Job,
    Part,
    Record,
    SeedRegistry,
    cfg_seed,
    check_layout,
    check_seed,
    header,
    outcome,
    protocol_doc,
    run_env,
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
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.collect import (
    Collection,
    IncumbentPolicy,
    PlaceboIncumbent,
    incumbent_consts,
)
from oran_causal.sim.system.decision.records import SCHEMA, SERIES_FIELDS, incumbent_unit_record
from oran_causal.sim.system.decision.scenario import PAIR, STRATUM, make_cfg
from oran_causal.sim.system.decision.units import T_UNIT, Policy

SCHEMA_EVAL = astudy2.SCHEMA_EVAL
SUB = astudy2.SUB
BLOCK = Block(186000, 187999)
LAYOUT = {stage: Block.sized(*base_n) for stage, base_n in astudy2.LAYOUT.items()}
KB_BLOCK = Block(186000, 186079)
RESERVES = (Block(186940, 186999), Block(187160, 187999))
FORBIDDEN = (
    Block(150000, 150399),
    Block(155000, 155399),
    Block(160000, 179999),
    Block(180000, 183999),
    Block(184000, 185999),
    Block(188000, 189999),
    Block(190000, 190399),
    KB_BLOCK,
    *RESERVES,
)
FROZEN_STAGES = ("disc", "placebo", "gt", "eval")
RECORD_STAGE = {"dev_conf": "dev", "disc": "eval", "placebo": "placebo", "gt": "gt"}
REPRO_SEEDS = (186080, 186081)
REPRO_ARMS = ("incumbent", "MG:allaccept", "noarb")
FOLD = 60
REFS = astudy2.REFS
ALL_ARMS = astudy2.ALL_ARMS
TAGS = {
    6622: "incumbent logging draws",
    6623: "random map default_rng([6623, 3])",
    6624: "study bootstraps: DISC [6624, 10, ...], EVAL [6624, 20, n_seeds]",
}
PROTOCOL = protocol_doc(
    "study2",
    "E6P_CONFOUNDED_PROTOCOL.md",
    "f719f43085cf1f95cf7c487e0e2c015e4377f9d83a9d4daa5484f42a996c7a1b",
)
MAPS = FrozenDoc(
    "docs/benchmark/artifacts/E6P_CONF_MAPS.json",
    ARTIFACTS / "E6P_CONF_MAPS.json",
    "6c82357b913d5ee11d501ff1209ec5e2068732e85181dcd27a5216109ff8940a",
    needs_line=False,
)
COLLECTION = Collection(
    T=T_UNIT,
    open_rule=OPEN_RULE,
    arb_warmup_s=ARB_WARMUP_S,
    count_all=COUNT_ALL,
    arbiter_cls=rf.DirectionalUnitArbiter,
)
TESTED_T0, TESTED_H = 90, 90


check_layout(LAYOUT, BLOCK, FORBIDDEN)
if len(ALL_ARMS) != 26 or len(rf.ASSOC_ARMS) != 13:
    raise ValueError("the 26 named arms of section 5.2")


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


def policy(stage: str) -> tuple[Callable[[int], Policy], str]:
    """The logging policy factory of a collection stage and its record name.

    Raises:
        ValueError: `stage` has no collection.
    """
    if stage == "placebo":
        return PlaceboIncumbent, "placebo_incumbent"
    if stage in ("dev_conf", "disc", "gt"):
        return IncumbentPolicy, "incumbent"
    raise ValueError(f"no collection policy for stage {stage}")


def _inc_class(u: Mapping[str, Any]) -> str:
    return f"{u['inc']['cls']}:{u['inc']['key']}"


def _tested(u: Mapping[str, Any], episode_s: float) -> bool:
    return u["t0"] >= TESTED_T0 and u["t0"] + TESTED_H <= episode_s


def _counts(urecs: list[Record], warmup_s: float, episode_s: float) -> dict[str, Any]:
    tested = [u for u in urecs if _tested(u, episode_s)]
    slt = [u for u in tested if u["knob"] == "sleep"]
    return {
        "units": len(urecs),
        "by_family_mode": family_mode_counts(urecs, lambda u: u["knob"]),
        "by_class_mode": family_mode_counts(urecs, _inc_class),
        **sleep_counts(urecs),
        "tested_units": len(tested),
        "tested_sleep_units": len(slt),
        "tested_sleep_rejects": sum(u["mode"] == "reject" for u in slt),
        "units_warmup": sum(u["t0"] < warmup_s for u in urecs),
        "missing_probs": sum(u.get("probs") is None for u in urecs),
    }


def episode_job(
    stage: str,
    seed: int,
    j: int,
    smoke: bool = False,
    short: float | None = None,
    max_labels: int | None = None,
) -> Record:
    """Runs one collection episode and returns its `e6p-disc-rec/1` record."""
    factory, name = policy(stage)
    spec = EpisodeSpec(
        seed=check_stage_seed(seed, stage),
        smoke=smoke,
        policy=factory,
        collection=COLLECTION,
        gt=stage == "gt",
        short=short,
        max_labels=max_labels,
    )
    c = collect(spec, incumbent_unit_record)
    counts: dict[str, Any] = label_counts(c)
    counts.update(_counts(c.urecs, c.cfg.warmup_s, c.cfg.warmup_s + c.cfg.scored_s))
    rs = RECORD_STAGE[stage]
    return {
        "kind": "episode",
        "schema": SCHEMA,
        "key": [rs, SUB, seed],
        "stage": rs,
        "sub": SUB,
        "conf_stage": stage,
        "seed": seed,
        "cfg_seed": spec.cfg_seed,
        "j": int(j),
        "fold": int(j) // FOLD if stage == "disc" else None,
        "smoke": smoke,
        "short": short,
        "policy": name,
        "pi0_table": None,
        "incumbent": incumbent_consts(),
        "directional": True,
        "arbiter": "DirectionalUnitArbiter",
        "passed": int(c.res["arb"].passed),
        "arb_warmup_s": ARB_WARMUP_S,
        "tap_count_all": COUNT_ALL,
        **body(c),
        "counts": counts,
    }


def load_maps(path: Path = MAPS.path) -> dict[str, Any]:
    """Maps and alias table of the frozen maps artifact (re-verified by `load_conf_maps`)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return rf.load_conf_maps(doc)


def arm_aliases(alias: Mapping[str, Any] | None, arm: str) -> tuple[str | None, list[str]]:
    """The decision signature of a simulated map arm and the arms it stands in for."""
    if alias is None or arm not in alias.get("signature", {}):
        return None, []
    others = [a for a, t in alias["alias_of"].items() if t == arm and a != arm]
    return alias["signature"][arm], others


def arm_job(
    seed: int,
    arm: str,
    arbiter: Callable[[str, int, float], Any],
    alias: Mapping[str, Any] | None,
    smoke: bool = False,
    short: float | None = None,
    **fields: Any,
) -> Record:
    """Runs one referee episode of `arm` (Study 2 or 3) and returns its `e6p-optaka2-rec/1` record.

    Args:
        seed: Protocol seed (checked by the caller).
        arm: Arm name; "freeze" runs the plant without xApps.
        arbiter: `arbiter(arm, plant_seed, warmup_s)`, None = accept-all.
        alias: The artifact's alias table (signature and aliases of map arms).
        smoke: Plumbing run (DEV plant seed).
        short: Scored seconds under smoke.
        **fields: Study fields of the record (`sub`, `conf_stage` or `cs_stage`).
    """
    sd = cfg_seed(seed, smoke)
    cfg = make_cfg("*" if arm == "freeze" else PAIR, STRATUM, sd, LOAD_FACTOR_L40)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = arbiter(arm, sd, cfg.warmup_s)
    t_cpu = time.process_time()
    env = run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    sig, aliases = arm_aliases(alias, arm)
    return {
        "kind": "job",
        "schema": SCHEMA_EVAL,
        "key": [seed, arm],
        "seed": seed,
        "cfg_seed": sd,
        "arm": arm,
        **fields,
        "smoke": smoke,
        "short": short,
        "episode_s": env.total_s,
        "warmup_s": cfg.warmup_s,
        "load_factor": LOAD_FACTOR_L40,
        **outcome(env),
        "policy_counts": rf.policy_counts(arb),
        "signature": sig,
        "aliases": aliases,
        "cpu_s": round(cpu, 2),
    }


def eval_arms(alias: Mapping[str, Any]) -> list[str]:
    """Distinct EVAL arms: anchors, references and the artifact's map jobs."""
    return [*rf.ANCHORS, *REFS, *(a for a in alias["jobs"] if a not in rf.ANCHORS)]


def conf_job(
    seed: int, arm: str, conf_stage: str, maps: Mapping[str, Any] | None, smoke: bool, short: Any
) -> Record:
    """One Study 2 referee job (`conf_stage` "eval" or "dev_repro")."""
    check_stage_seed(seed, "dev_conf" if conf_stage == "dev_repro" else "eval")
    m = maps["maps"] if maps else None

    def arbiter(a: str, sd: int, w: float) -> Any:
        return rf.conf_arbiter(a, sd, w, m)

    alias = maps["alias"] if maps else None
    return arm_job(seed, arm, arbiter, alias, smoke, short, sub=SUB, conf_stage=conf_stage)


def jobs(
    stage: str,
    maps: Mapping[str, Any] | None,
    smoke: bool = False,
    short: float | None = None,
    max_labels: int | None = None,
    arms: frozenset[str] | None = None,
) -> list[Job]:
    """The jobs of a stage in shard order (seed-major).

    Raises:
        ValueError: Unknown stage, or eval without the maps.
    """
    if stage not in LAYOUT:
        raise ValueError(f"stage in {tuple(LAYOUT)}")
    out: list[Job] = []
    if stage == "eval":
        if maps is None:
            raise ValueError("eval jobs need the maps artifact")
        for sd in stage_seeds(stage):
            for a in eval_arms(maps["alias"]):
                if arms is None or a in arms:
                    out.append(
                        Job(
                            (sd, a),
                            sd,
                            lambda sd=sd, a=a: conf_job(sd, a, "eval", maps, smoke, short),
                        )
                    )
        return out
    rs = RECORD_STAGE[stage]
    for j, sd in enumerate(stage_seeds(stage)):
        out.append(
            Job(
                (rs, SUB, sd),
                sd,
                lambda sd=sd, j=j: episode_job(stage, sd, j, smoke, short, max_labels),
            )
        )
        if stage == "dev_conf" and sd in REPRO_SEEDS:
            out += [
                Job(
                    (sd, a), sd, lambda sd=sd, a=a: conf_job(sd, a, "dev_repro", None, smoke, short)
                )
                for a in REPRO_ARMS
                if arms is None or a in arms
            ]
    return out


def registry_status(reg: SeedRegistry | None = None) -> dict[str, bool]:
    """Whether the registry holds the block, the tags and every stage range of the layout."""
    reg = reg or SeedRegistry()
    blk = reg.e6("e6p_confounded_episodes", [])
    note = str(reg.e6("e6p_confounded_note", ""))
    return {
        "block_186000_187999": any(lo <= BLOCK.lo and hi >= BLOCK.hi for lo, hi in blk),
        "tags_6622_6624": reg.has_tags(TAGS),
        "layout": all(f"{b.lo}-{b.hi}" in note for b in LAYOUT.values()),
    }


def guard(stage: str, smoke: bool, maps_doc: FrozenDoc = MAPS) -> dict[str, Any]:
    """Checks the protocol's refusals (smoke runs are exempt) and returns the statuses.

    Raises:
        RuntimeError: A frozen stage without the frozen protocol, eval without the frozen maps,
            an unregistered layout, or a frozen stage off Linux.
    """
    fz, reg = PROTOCOL.status(), registry_status()
    ms = maps_doc.status() if stage == "eval" else None
    st = {"freeze": fz, "registry": reg, "maps": ms}
    if smoke:
        return st
    if stage in FROZEN_STAGES and not fz["frozen"]:
        raise RuntimeError(f"stage {stage} needs the frozen protocol: {fz}")
    if ms is not None and not ms["frozen"]:
        raise RuntimeError(f"stage eval needs the frozen maps artifact: {ms}")
    if not all(reg.values()):
        raise RuntimeError(f"seed block / RNG tags / layout not registered: {reg}")
    if stage in FROZEN_STAGES and not sys.platform.startswith("linux"):
        raise RuntimeError(f"stage {stage}: Linux numerics only; platform {sys.platform}")
    return st


def consts(smoke: dict[str, Any] | None, arms: frozenset[str] | None) -> dict[str, Any]:
    """The run constants recorded in the header."""
    return {
        "pair": PAIR,
        "stratum": STRATUM,
        "T": COLLECTION.T,
        "open_rule": OPEN_RULE,
        "step_s": STEP_S,
        "arb_warmup_s": ARB_WARMUP_S,
        "tap_count_all": COUNT_ALL,
        "directional": True,
        "incumbent": incumbent_consts(),
        "layout": {s: [b.lo, b.hi - b.lo + 1] for s, b in LAYOUT.items()},
        "repro_seeds": REPRO_SEEDS,
        "repro_arms": REPRO_ARMS,
        "theta": rf.THETA,
        "k_conf": rf.K_CONF,
        "gt": gt_consts(),
        "tags": TAGS,
        "series_fields": SERIES_FIELDS,
        "smoke": smoke,
        "arms_filter": sorted(arms) if arms else None,
    }


def run(
    stage: str,
    out: Path,
    part: Part | None = None,
    smoke: bool = False,
    short: float | None = None,
    max_labels: int | None = None,
    arms: frozenset[str] | None = None,
    maps_doc: FrozenDoc = MAPS,
) -> int:
    """Runs a Study 2 stage (shard `part`) into `out` and returns the number of jobs run.

    `maps_doc` is the pinned maps artifact of the eval stage (the frozen one by default).

    Raises:
        ValueError: `short` or `max_labels` without `smoke`.
    """
    if (short or max_labels) and not smoke:
        raise ValueError("short / max_labels are smoke-only")
    part = part or Part()
    st = guard(stage, smoke, maps_doc)
    maps = load_maps(maps_doc.path) if stage == "eval" else None
    head = header(
        astudy2.DRIVER,
        stage,
        part,
        smoke,
        conf_stage=stage,
        mapgate_sha256=LEGACY_SHA256[MAPGATE],
        schema=SCHEMA_EVAL if stage == "eval" else SCHEMA,
        protocol=PROTOCOL.name,
        **st,
        maps_source={"jobs": maps["alias"]["jobs"], "alias_of": maps["alias"]["alias_of"]}
        if maps
        else None,
        consts=consts({"short": short, "max_labels": max_labels} if smoke else None, arms),
    )
    js = jobs(stage, maps, smoke, short, max_labels, arms)
    show = ("conf_stage", "seed", "arm", "counts", "psvr", "cpu_s", "secs")
    return run_jobs(out, head, js, part, smoke, show)

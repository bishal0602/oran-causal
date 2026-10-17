"""One entry point per stage of the system-level studies: `run_stage(study, stage, ...)`.

| study | episode stages | analysis stages |
|---|---|---|
| 1 | dev, eval, placebo, gt | discovery |
| 2 | dev_conf, disc, placebo, gt, eval | discovery, maps |
| 3 | dev, eval | bounds, calib, certsafe |

Episode stages append JSONL records to `out` (shard `part` = "i/k"); analysis stages write
their JSON output into the directory `out` and read the record or input locations given in
`inputs`.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.experiments.launch.fingerprint import numeric_env
from oran_causal.experiments.system import artifacts as art
from oran_causal.experiments.system import (
    study1,
    study1_discovery,
    study2,
    study2_discovery,
    study3,
)
from oran_causal.experiments.system.common import Part, read_records, sha_lf
from oran_causal.sim.system.decision import records

EPISODE_STAGES = {
    1: tuple(study1.STAGES),
    2: tuple(study2.LAYOUT),
    3: tuple(study3.LAYOUT),
}
ANALYSIS_STAGES = {1: ("discovery",), 2: ("discovery", "maps"), 3: ("bounds", "calib", "certsafe")}

type Inputs = Mapping[str, Sequence[str]]


def _episodes(study: int, stage: str, out: Path, part: Part, opts: Mapping[str, Any]) -> int:
    smoke, short = bool(opts.get("smoke")), opts.get("short")
    arms = frozenset(opts["arms"]) if opts.get("arms") else None
    if study == 1:
        return study1.run(stage, out, part, smoke, short, opts.get("max_labels"))
    if study == 2:
        return study2.run(stage, out, part, smoke, short, opts.get("max_labels"), arms)
    return study3.run(stage, out, part, smoke, short, arms)


def _one(inputs: Inputs, key: str) -> Path:
    v = inputs.get(key) or ()
    if len(v) != 1:
        raise ValueError(f"input {key!r}: one path expected, got {list(v)}")
    return Path(v[0])


def _discovery(study: int, out: Path, inputs: Inputs) -> Path:
    cache = out / "cache"
    if study == 1:
        s1 = study1_discovery.Inputs(
            eval=tuple(inputs["eval"]),
            placebo=tuple(inputs["placebo"]),
            gt=tuple(inputs["gt"]),
            dev=tuple(inputs["dev"]),
            cache_dir=cache,
        )
        return study1_discovery.run(s1, out)
    s2 = study2_discovery.Inputs(
        dev=tuple(inputs["dev"]),
        disc=tuple(inputs["disc"]),
        placebo=tuple(inputs["placebo"]),
        gt=tuple(inputs["gt"]),
        cache_dir=cache,
        dev_step1=tuple(inputs.get("dev_step1", ())),
    )
    return study2_discovery.run(s2, out)


def _maps(out: Path, inputs: Inputs) -> Path:
    disc_json = _one(inputs, "disc_json")
    a = art.build_maps(art.read_json(disc_json), sha_lf(disc_json))
    path = out / study2.MAPS.path.name
    art.write_artifact(a, path)
    return path


def _bounds(out: Path, inputs: Inputs) -> Path:
    disc_json = _one(inputs, "disc_json")
    files = study1_discovery.find_records(inputs["disc"])
    caches = study1_discovery.build_caches(files, "eval", out / "cache" / "disc", "disc")
    pool = records.load_pool([str(p) for p in caches], H=90, H_pre=90, stages={"eval"})
    ud = pool.unit_data()
    seeds = sorted({int(s) for s in np.unique(ud.seed)})
    maps_sha = sha_lf(study2.MAPS.path)
    pinned = art.read_json(study2.MAPS.path)["sha256"]["disc_conf.json"]
    full = (
        study3.PROTOCOL.status()["frozen"]
        and seeds == study2.stage_seeds("disc")
        and ud.meta.get("p_mismatch") == 0
        and maps_sha == study2.MAPS.want
        and sha_lf(disc_json) == pinned
    )
    rep = art.bounds_report(
        ud,
        art.maps_from_artifact(),
        art.read_json(disc_json),
        {"maps_artifact": maps_sha, "disc_conf.json": sha_lf(disc_json)},
        full,
        numeric_env(),
    )
    path = out / "certsafe_bounds.json"
    study1_discovery.dump(rep, path)
    return path


def _calib(out: Path, inputs: Inputs) -> Path:
    recs = [r for p in study1_discovery.find_records(inputs["records"]) for r in read_records(p)]
    path = out / "certsafe_calib.json"
    study1_discovery.dump(art.calib_report(recs), path)
    return path


def _certsafe(out: Path, inputs: Inputs) -> Path:
    paths = {k: _one(inputs, k) for k in ("bounds", "calib", "disc_json")}
    sha = {
        "certsafe_bounds.json": sha_lf(paths["bounds"]),
        "certsafe_calib.json": sha_lf(paths["calib"]),
        "disc_conf.json": sha_lf(paths["disc_json"]),
    }
    a = art.build_certsafe(
        art.read_json(paths["bounds"]),
        art.read_json(paths["calib"]),
        art.read_json(paths["disc_json"]),
        sha,
    )
    path = out / study3.CERTSAFE.path.name
    art.write_artifact(a, path)
    return path


ANALYSES = {
    (1, "discovery"): lambda out, inputs: _discovery(1, out, inputs),
    (2, "discovery"): lambda out, inputs: _discovery(2, out, inputs),
    (2, "maps"): _maps,
    (3, "bounds"): _bounds,
    (3, "calib"): _calib,
    (3, "certsafe"): _certsafe,
}


def run_stage(
    study: int,
    stage: str,
    part: str | None = None,
    out: Path = Path("out"),
    inputs: Inputs | None = None,
    **opts: Any,
) -> int | Path:
    """Runs one stage of Study 1, 2 or 3.

    Args:
        study: 1, 2 or 3.
        stage: An episode or analysis stage of the study (module docstring).
        part: Shard "i/k" of an episode stage (None = every job).
        out: Episode stages: the JSONL file; analysis stages: the output directory.
        inputs: Analysis stages: input locations by name (`eval`, `placebo`, `gt`, `dev`, `disc`,
            `dev_step1`, `records`, `disc_json`, `bounds`, `calib`).
        **opts: Episode stages: `smoke`, `short`, `max_labels`, `arms`.

    Returns:
        The number of jobs run (episode stages) or the written file (analysis stages).

    Raises:
        ValueError: Unknown study or stage.
    """
    if stage in EPISODE_STAGES.get(study, ()):
        out.parent.mkdir(parents=True, exist_ok=True)
        return _episodes(study, stage, out, Part.parse(part), opts)
    fn = ANALYSES.get((study, stage))
    if fn is None:
        known = EPISODE_STAGES.get(study, ()) + ANALYSIS_STAGES.get(study, ())
        raise ValueError(f"study {study}: stage in {known}")
    out.mkdir(parents=True, exist_ok=True)
    return fn(out, inputs or {})

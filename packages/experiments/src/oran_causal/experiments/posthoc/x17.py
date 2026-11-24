"""X17 run half: the strict maps artifact and the MG:PMRT referee with the strict map.

Post hoc (posthoc protocol, 2026-11-24, X17). `analyse` writes `x17_centre.json` and
`certsafe_bounds_strict.json` (`oran_causal.analysis.posthoc.x17`) and then
`study2_maps_strict.json`: the frozen Study 2 maps artifact with the MG:PMRT map replaced by its
strict-centre coefficients and the signatures, aliases and decision tables re-derived. Every
other map is copied unchanged.

`run` simulates MG:PMRT with the strict map on EVAL seeds of both blocks, with the plant, arm code
and seeds of the frozen runs (`study2.arm_job`, the MapGateV2 arbiter of `conf_arbiter`). Job
records carry sub `x17` and `x17_stage` "confirm" (step 3, `CONFIRM_SEEDS`) or "full" (step 4,
both blocks). The strict artifact is pinned by its LF sha256 on the command line and must name
the frozen maps artifact it was derived from.

    python -m oran_causal.experiments.posthoc x17 analyse --cache-dir C --out D
    python -m oran_causal.experiments.posthoc x17 confirm --runs F... --x17 D/x17_centre.json
        --out D
    python -m oran_causal.experiments.posthoc x17 rescore --runs F... --out D
    python -m oran_causal.experiments.posthoc x17-run run --maps D/study2_maps_strict.json
        --maps-sha256 S --stage confirm|full --part i/k --out F.jsonl [--smoke --short S]
    python -m oran_causal.experiments.posthoc x17-run launch --name NAME --maps-sha256 S
        --stage confirm|full [--maps M] [--kernel j/m] [--procs 4] [--dry-run]

`launch` stages a Kaggle session (the platform of the stored MG:PMRT records) with the strict
artifact bundled at `MAPS_REL` and pinned by its sha256 in the session command: kernel j of m
runs `procs` shards in parallel, shard j * procs + i of m * procs, each into `res_<shard>.jsonl`
of the session output. The confirm stage runs in one kernel. The session reads no stored records;
`x17 confirm` and `x17 rescore` compare its records with them afterwards.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.posthoc import x17 as ax17
from oran_causal.experiments.campaign.host import PLATFORM_ENV, host_info
from oran_causal.experiments.campaign.provenance import COMMIT_ENV, DIRTY_ENV, LOCK_ENV, code_info
from oran_causal.experiments.launch import kaggle
from oran_causal.experiments.launch.commands import LOCK_REL, THREADS, SetupPlan, setup_cmd
from oran_causal.experiments.launch.fingerprint import numeric_env
from oran_causal.experiments.launch.runner import Dirs
from oran_causal.experiments.posthoc import tools
from oran_causal.experiments.system import artifacts, study2, study3
from oran_causal.experiments.system.common import (
    Job,
    Part,
    Record,
    header,
    run_jobs,
    sha_lf,
)
from oran_causal.methods.system import referees as rf

MAPS_STRICT = "study2_maps_strict.json"
MAPS_REL = f"data/records/posthoc_x17/{MAPS_STRICT}"
STAGES = ("confirm", "full", "control")
CENTRES = ("strict", "frozen")
PROCS = 4
WORK_DIR = Path(tempfile.gettempdir()) / "oran-causal-launch"


def strict_maps_artifact(
    frozen: Mapping[str, Any],
    strict_map: Mapping[tuple[str, str, str], float],
    frozen_sha: str,
    x17_sha: str,
) -> dict[str, Any]:
    """The frozen maps artifact with MG:PMRT's map replaced and its derived fields re-derived.

    Args:
        frozen: The frozen Study 2 maps artifact (parsed).
        strict_map: The strict MG:PMRT map of `analyse`.
        frozen_sha: LF sha256 of the frozen artifact.
        x17_sha: LF sha256 of `x17_centre.json`.

    Raises:
        ValueError: The strict map's edges or signs differ from the frozen map's.
    """
    old = rf.map_from_json(frozen["arms"][ax17.ARM]["map"])
    if sorted(old) != sorted(strict_map) or any((old[h] or 0.0) * strict_map[h] < 0 for h in old):
        raise ValueError("the strict MG:PMRT map must keep the frozen edges and signs")
    doc = copy.deepcopy(dict(frozen))
    maps = {arm: rf.map_from_json(v["map"]) for arm, v in doc["arms"].items()}
    maps[ax17.ARM] = dict(strict_map)
    theta = float(doc["mapgate"]["theta"])
    al = rf.alias_table(maps, theta, bool(doc["noarb_alias"]))
    for arm, v in doc["arms"].items():
        v["map"] = rf.map_to_json(maps[arm])
        v["signature"] = al["signature"][arm]
        v["alias_of"] = al["alias_of"][arm]
        v["decision_table"] = {
            artifacts.dir_key(f, d): x
            for (f, d), x in rf.decision_table_v2(maps[arm], theta).items()
        }
    doc.update(jobs=al["jobs"], alias_of=al["alias_of"], all_accept_signature=al["all_accept"])
    doc["x17"] = {
        "protocol": ax17.PROTOCOL,
        "centre": "strict",
        "window_s": ax17.WINDOW_S,
        "replaced": [ax17.ARM],
        "frozen_maps_sha256": frozen_sha,
        "x17_centre_sha256": x17_sha,
    }
    rf.load_conf_maps(doc)
    return doc


def analyse(
    records: Path, cache_dir: Path, out: Path, artifacts_dir: Path = ax17.ARTIFACTS
) -> dict[str, Any]:
    """Steps 1-2 and the strict maps artifact; returns the analysis output plus `maps_sha256`."""
    res = ax17.run(
        records, tools=tools.referee_tools(), cache_dir=cache_dir, artifacts=artifacts_dir
    )
    ax17.write(res, out)
    frozen_path = artifacts_dir / "study2_maps.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    sm = {(str(f), str(r), str(k)): float(b) for f, r, k, b in res["strict_map"]}
    doc = strict_maps_artifact(frozen, sm, sha_lf(frozen_path), sha_lf(out / "x17_centre.json"))
    sha = artifacts.write_artifact(doc, out / MAPS_STRICT)
    return dict(res, maps_sha256=sha)


def load_strict(path: Path, want: str) -> dict[str, Any]:
    """The strict maps artifact, checked against its pin and the frozen maps artifact.

    Raises:
        ValueError: A sha256 other than `want`, or an artifact not derived from the frozen one.
    """
    got = sha_lf(path)
    if got != want:
        raise ValueError(f"{path}: sha256 {got} is not the pinned {want}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("x17") or {}).get("frozen_maps_sha256") != study2.MAPS.want:
        raise ValueError(f"{path}: not derived from the frozen Study 2 maps artifact")
    return rf.load_conf_maps(doc)


def seeds_of(stage: str) -> list[int]:
    """The protocol seeds of a stage: the confirmation seeds, or both full EVAL blocks.

    Raises:
        ValueError: Unknown stage.
    """
    if stage in ("confirm", "control"):
        return list(ax17.CONFIRM_SEEDS)
    if stage == "full":
        return [*study2.stage_seeds("eval"), *study3.stage_seeds("eval")]
    raise ValueError(f"stage in {STAGES}")


def check_seed(seed: int) -> int:
    """The seed, checked against the EVAL layout of its block."""
    if ax17.block_of(seed) == "study2":
        return study2.check_stage_seed(seed, "eval")
    return study3.check_stage_seed(seed, "eval")


def job(
    seed: int,
    stage: str,
    maps: Mapping[str, Any],
    *,
    smoke: bool = False,
    short: float | None = None,
    centre: str = "strict",
) -> Record:
    """One MG:PMRT episode with the map of `centre`, in the frozen EVAL record layout."""
    check_seed(seed)

    def arbiter(a: str, sd: int, w: float) -> Any:
        return rf.conf_arbiter(a, sd, w, maps["maps"])

    return study2.arm_job(
        seed,
        ax17.ARM,
        arbiter,
        maps["alias"],
        smoke,
        short,
        sub=ax17.SUB,
        x17_stage=stage,
        x17_block=ax17.block_of(seed),
        x17_centre=centre,
    )


def jobs(
    stage: str,
    maps: Mapping[str, Any],
    seeds: Sequence[int] | None = None,
    *,
    smoke: bool = False,
    short: float | None = None,
    frozen: Mapping[str, Any] | None = None,
) -> list[Job]:
    """The resumable jobs of a stage in seed order.

    The control stage (amendment A2) runs every seed with the strict and the `frozen` map, so
    both centres share one session's numerical environment.

    Raises:
        ValueError: The control stage without the frozen maps.
    """
    if stage == "control":
        if frozen is None:
            raise ValueError("the control stage needs the frozen maps")
        by = {"strict": maps, "frozen": frozen}
        return [
            Job(
                (s, ax17.ARM, c),
                check_seed(s),
                lambda s=s, c=c: job(s, stage, by[c], smoke=smoke, short=short, centre=c),
            )
            for s in (list(seeds) if seeds else seeds_of(stage))
            for c in CENTRES
        ]
    return [
        Job(
            (s, ax17.ARM),
            check_seed(s),
            lambda s=s: job(s, stage, maps, smoke=smoke, short=short),
        )
        for s in (list(seeds) if seeds else seeds_of(stage))
    ]


def head(
    part: Part,
    stage: str,
    maps_path: Path,
    maps_sha: str,
    *,
    smoke: bool,
    short: float | None,
    environ: Mapping[str, str] | None = None,
) -> Record:
    """The header line of a run call."""
    env = os.environ if environ is None else environ
    return header(
        ax17.DRIVER,
        stage,
        part,
        smoke,
        schema=study2.SCHEMA_EVAL,
        sub=ax17.SUB,
        **{ax17.STAGE_KEY: stage},
        platform=numeric_env(),
        host=env.get(PLATFORM_ENV),
        host_info=host_info(),
        code_commit=env.get(COMMIT_ENV),
        code_dirty=env.get(DIRTY_ENV),
        maps_strict={"path": str(maps_path), "sha256": maps_sha},
        maps_frozen=study2.MAPS.status(),
        protocol=ax17.PROTOCOL,
        consts={
            "arm": ax17.ARM,
            "theta": rf.THETA,
            "k_conf": rf.K_CONF,
            "window_s": ax17.WINDOW_S,
            "smoke": {"short": short} if smoke else None,
        },
    )


def run(
    part: Part,
    out: Path,
    stage: str,
    maps_path: Path,
    maps_sha: str,
    *,
    smoke: bool = False,
    short: float | None = None,
    seeds: Sequence[int] | None = None,
) -> int:
    """Runs shard `part` of the X17 MG:PMRT jobs of `stage` into `out`; returns the jobs run.

    Raises:
        ValueError: `short` without `smoke`.
        RuntimeError: A non-smoke run off Linux (the frozen runs' numerics).
    """
    if short and not smoke:
        raise ValueError("--short is smoke-only")
    if not smoke and not sys.platform.startswith("linux"):
        raise RuntimeError(f"X17 runs need Linux numerics; platform {sys.platform}")
    maps = load_strict(maps_path, maps_sha)
    frozen = study2.load_maps() if stage == "control" else None
    show = ("seed", "arm", "x17_centre", "psvr", "rlf", "cpu_s")
    return run_jobs(
        out,
        head(part, stage, maps_path, maps_sha, smoke=smoke, short=short),
        jobs(stage, maps, seeds, smoke=smoke, short=short, frozen=frozen),
        part,
        smoke,
        show,
    )


def main_analyse(argv: Sequence[str] | None = None) -> int:
    """Command line: `analyse` (steps 1-2 and the strict artifact), else `confirm` / `rescore`."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] != ["analyse"]:
        return ax17.main(args)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(ax17.RECORDS))
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(args[1:])
    res = analyse(Path(a.records), Path(a.cache_dir), Path(a.out))
    keys = ("ok", "reading", "maps_sha256")
    print(json.dumps({k: res[k] for k in keys}, indent=1))
    return 0 if res["ok"] else 1


def session_command(
    out_dir: str,
    code: Mapping[str, Any],
    stage: str,
    maps_sha: str,
    kernel: Part,
    procs: int = PROCS,
    maps: str = MAPS_REL,
) -> str:
    """The shell command of one cloud session: environment setup, then `procs` shards in parallel.

    Kernel `j/m` runs shards `j * procs` to `j * procs + procs - 1` of `m * procs`, each pinned to
    the strict artifact's sha256.
    """
    env = (
        f"{PLATFORM_ENV}=kaggle {COMMIT_ENV}={code['commit']} "
        f"{DIRTY_ENV}={int(bool(code['dirty']))} {LOCK_ENV}={LOCK_REL} {THREADS}"
    )
    total = kernel.count * procs
    runs = "".join(
        f"({env} python -u -m oran_causal.experiments.posthoc x17-run run --maps {maps} "
        f"--maps-sha256 {maps_sha} --stage {stage} --part {s}/{total} "
        f"--out {out_dir}/res_{s}.jsonl > {out_dir}/log_{s}.txt 2>&1) & "
        for s in range(kernel.index * procs, kernel.index * procs + procs)
    )
    return setup_cmd(SetupPlan(out_dir=out_dir)) + runs + "wait"


def kaggle_job(
    name: str,
    code: Mapping[str, Any],
    stage: str,
    maps_sha: str,
    kernel: Part,
    procs: int = PROCS,
    maps: str = MAPS_REL,
) -> kaggle.KaggleJob:
    """The Kaggle session of one kernel; the strict artifact travels in the bundle.

    Raises:
        ValueError: An unknown stage, or the confirm or control stage over more than one kernel.
    """
    if stage not in STAGES:
        raise ValueError(f"stage in {STAGES}")
    if stage in ("confirm", "control") and kernel.count != 1:
        raise ValueError(f"the {stage} stage runs in one kernel")
    cmd = session_command("$JOB_OUT", code, stage, maps_sha, kernel, procs, maps)
    return kaggle.KaggleJob(name, cmd, (maps,))


def _launch(a: argparse.Namespace) -> int:
    load_strict(Path(a.maps), a.maps_sha256)
    job = kaggle_job(
        a.name,
        code_info(),
        a.stage,
        a.maps_sha256,
        Part.parse(a.kernel),
        a.procs,
        Path(a.maps).as_posix(),
    )
    print(f"[x17] kaggle {job.name}: extra {list(job.extra)}")
    print(f"[x17] cmd: {job.cmd}")
    if a.dry_run:
        return 0
    client = kaggle.KaggleClient(owner=a.owner)
    print(json.dumps(kaggle.launch(job, client, Dirs(WORK_DIR)), indent=1, default=str))
    return 0


def _run_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--maps", required=True)
    r.add_argument("--maps-sha256", required=True)
    r.add_argument("--stage", choices=STAGES, required=True)
    r.add_argument("--part", default="0/1")
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", default=None)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--short", type=float, default=None)
    k = sub.add_parser("launch")
    k.add_argument("--name", required=True)
    k.add_argument("--maps", default=MAPS_REL)
    k.add_argument("--maps-sha256", required=True)
    k.add_argument("--stage", choices=STAGES, required=True)
    k.add_argument("--kernel", default="0/1")
    k.add_argument("--procs", type=int, default=PROCS)
    k.add_argument("--owner", default=None)
    k.add_argument("--dry-run", action="store_true")
    return ap


def main_run(argv: Sequence[str] | None = None) -> int:
    """Command line of steps 3-4: `run` a shard of the strict-map MG:PMRT jobs or `launch` one."""
    a = _run_parser().parse_args(argv)
    if a.cmd == "launch":
        return _launch(a)
    run(
        Part.parse(a.part),
        Path(a.out),
        a.stage,
        Path(a.maps),
        a.maps_sha256,
        smoke=a.smoke,
        short=a.short,
        seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_run())

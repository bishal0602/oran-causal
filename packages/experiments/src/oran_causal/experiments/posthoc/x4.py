"""X4: random accept / defer referee on the Study 3 EVAL episodes (run half).

Post hoc and descriptive. Policy `X4:rand@<p>` defers every xApp request that reaches the referee
independently with probability p, from t = 0, all families and directions alike, with one uniform
per request from `default_rng([seed, 6640])` (shared by the three p: common random numbers).
The plant, the seeds (191100-191259) and the record schema are the certified-safe driver's.
The provenance jobs (`noarb`, `never_sleep` on 191100 and 191101) use the frozen driver's own
arbiters. Scoring the records is `oran_causal.analysis.posthoc.x4`.

    python -m oran_causal.experiments.posthoc.x4 run --part i/k --out F.jsonl
        [--smoke] [--short S] [--seeds a,b] [--arms a,b]

`--smoke` is plumbing only (a DEV plant seed, `--short` scored seconds), never a result.
"""

import argparse
import dataclasses
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc.x4 import (
    ARMS,
    COMPARE,
    DRIVER,
    P_OF,
    PROV_ARMS,
    PROV_SEEDS,
    SUB,
    TAG,
)
from oran_causal.experiments.campaign.host import PLATFORM_ENV
from oran_causal.experiments.campaign.provenance import COMMIT_ENV, DIRTY_ENV
from oran_causal.experiments.launch.fingerprint import numeric_env
from oran_causal.experiments.system import study2, study3
from oran_causal.experiments.system.common import (
    LOAD_FACTOR_L40,
    Job,
    Part,
    Record,
    cfg_seed,
    header,
    outcome,
    run_env,
    run_jobs,
)
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.scenario import PAIR, STRATUM, make_cfg

STAGE = "eval"
Obs = dict[str, Any]
Arbiter = Callable[[Obs], dict[str, Any]]


class RandomDefer:
    """Defers each request independently with probability p.

    One uniform per request in `obs["requests"]` order from `default_rng([seed, TAG])`; reads
    nothing else from the observation.
    """

    def __init__(self, p: float, seed: int) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(p)
        self.p, self.rng = float(p), np.random.default_rng([int(seed), TAG])
        self.n = {"req": 0, "rej": 0}

    def __call__(self, obs: Obs) -> dict[str, Any]:
        """The accept / reject decision of every request."""
        dec = []
        for _ in obs["requests"]:
            self.n["req"] += 1
            if self.rng.random() < self.p:
                dec.append("reject")
                self.n["rej"] += 1
            else:
                dec.append("accept")
        return {"decisions": dec, "writes": [], "rollback": []}


def make_arbiter(arm: str, sd: int, warmup_s: float) -> Any:
    """Arbiter of an X4 arm or of a provenance arm (the frozen driver's own).

    Raises:
        KeyError: Unknown arm.
    """
    if arm in P_OF:
        return rf.Counting(RandomDefer(P_OF[arm], sd), is_unit=False)
    if arm in PROV_ARMS:
        return rf.cs_arbiter(arm, sd, warmup_s, None)
    raise KeyError(arm)


def job(seed: int, arm: str, *, smoke: bool = False, short: float | None = None) -> Record:
    """One episode, with the certified-safe job record layout and sub `x4`."""
    study3.check_stage_seed(seed, STAGE)
    sd = cfg_seed(seed, smoke)
    cfg = make_cfg(PAIR, STRATUM, sd, LOAD_FACTOR_L40)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make_arbiter(arm, sd, cfg.warmup_s)
    t_cpu = time.process_time()
    env = run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    return {
        "kind": "job",
        "schema": study2.SCHEMA_EVAL,
        "key": [seed, arm],
        "seed": seed,
        "cfg_seed": sd,
        "arm": arm,
        "sub": SUB,
        "cs_stage": STAGE,
        "smoke": smoke,
        "short": short,
        "episode_s": env.total_s,
        "warmup_s": cfg.warmup_s,
        "load_factor": LOAD_FACTOR_L40,
        **outcome(env),
        "policy_counts": rf.policy_counts(arb),
        "x4": {"p": P_OF.get(arm), "tag": TAG},
        "cpu_s": round(cpu, 2),
    }


def job_pairs(
    seeds: Sequence[int] | None = None, arms: Sequence[str] | None = None
) -> list[tuple[int, str]]:
    """`(seed, arm)`: the provenance jobs first, then the X4 jobs seed-major."""
    seeds = list(seeds) if seeds else study3.stage_seeds(STAGE)
    pairs = [(s, a) for s in PROV_SEEDS for a in PROV_ARMS if s in seeds]
    pairs += [(s, a) for s in seeds for a in ARMS]
    return [(study3.check_stage_seed(s, STAGE), a) for s, a in pairs if not arms or a in arms]


def jobs(
    seeds: Sequence[int] | None = None,
    arms: Sequence[str] | None = None,
    *,
    smoke: bool = False,
    short: float | None = None,
) -> list[Job]:
    """The resumable jobs of the run in shard order."""
    return [
        Job((s, a), s, lambda s=s, a=a: job(s, a, smoke=smoke, short=short))
        for s, a in job_pairs(seeds, arms)
    ]


def head(
    part: Part,
    *,
    smoke: bool,
    short: float | None,
    environ: Mapping[str, str] | None = None,
) -> Record:
    """The header line of a run call."""
    env = os.environ if environ is None else environ
    return header(
        DRIVER,
        STAGE,
        part,
        smoke,
        schema=study2.SCHEMA_EVAL,
        sub=SUB,
        cs_stage=STAGE,
        platform=numeric_env(),
        host=env.get(PLATFORM_ENV),
        code_commit=env.get(COMMIT_ENV),
        code_dirty=env.get(DIRTY_ENV),
        maps=study2.MAPS.status(),
        certsafe_freeze=study3.PROTOCOL.status(),
        consts={
            "pair": PAIR,
            "stratum": STRATUM,
            "load_factor": LOAD_FACTOR_L40,
            "tag": TAG,
            "p": dict(P_OF),
            "prov": [list(PROV_SEEDS), list(PROV_ARMS)],
            "compare": list(COMPARE),
            "smoke": {"short": short} if smoke else None,
        },
    )


def run(
    part: Part,
    out: Path,
    *,
    smoke: bool = False,
    short: float | None = None,
    seeds: Sequence[int] | None = None,
    arms: Sequence[str] | None = None,
) -> int:
    """Runs shard `part` of the X4 jobs into `out`; returns the number of jobs run.

    Raises:
        ValueError: `short` without `smoke`.
    """
    if short and not smoke:
        raise ValueError("--short is smoke-only")
    show = ("seed", "arm", "psvr", "rlf", "energy_j", "st_req", "st_rej", "cpu_s")
    return run_jobs(
        out,
        head(part, smoke=smoke, short=short),
        jobs(seeds, arms, smoke=smoke, short=short),
        part,
        smoke,
        show,
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--part", default="0/1")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--short", type=float, default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--arms", default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: `run` a shard of the X4 jobs."""
    a = _parser().parse_args(argv)
    run(
        Part.parse(a.part),
        Path(a.out),
        smoke=a.smoke,
        short=a.short,
        seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None,
        arms=a.arms.split(",") if a.arms else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

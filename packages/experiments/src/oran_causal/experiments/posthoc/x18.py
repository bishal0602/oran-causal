"""X18 Part C run half: the certified-safe referee at gate budgets no stored job simulates.

Post hoc and descriptive (protocol entry 2026-11-24, X18). Every job of `x18_plan.json` is
CertSafeMapGateV2 with the plan's map and uncertified set, run through the frozen certified-safe
arm runner (plant, `Counting` wrapper and record layout of `e6p_certsafe`) on the Study 3 EVAL
seeds 191100-191259, so that it pairs by seed with the stored records. Records carry sub `x18`
and arm `X18:<signature id>`. The reproduction jobs run the stored CS:PMRT on 191100-191103 as
`X18:repro:CS:PMRT`; `oran_causal.analysis.posthoc.x18 merge` requires them to equal the stored
records. The plan is made by `oran_causal.analysis.posthoc.x18 plan`.

    python -m oran_causal.experiments.posthoc x18-run --plan x18_plan.json --part i/k --out F.jsonl
        [--seeds a,b] [--smoke] [--short S]
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.posthoc.x18 import DRIVER, SUB
from oran_causal.experiments.campaign.host import PLATFORM_ENV
from oran_causal.experiments.campaign.provenance import COMMIT_ENV, DIRTY_ENV
from oran_causal.experiments.launch.fingerprint import numeric_env
from oran_causal.experiments.system import study2, study3
from oran_causal.experiments.system.common import (
    Job,
    Part,
    Record,
    header,
    run_jobs,
    sha_lf,
)
from oran_causal.methods.system import referees as rf

STAGE = "eval"
Entry = Mapping[str, Any]


def read_plan(path: Path) -> dict[str, Any]:
    """The plan JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def check_plan(plan: Mapping[str, Any]) -> None:
    """Checks the plan's theta and that every entry's map and uncertified set give its signature.

    Raises:
        ValueError: A different theta, or a signature that does not reproduce.
    """
    if plan.get("schema") != "x18-plan/1":
        raise ValueError(f"not an X18 plan: {plan.get('schema')}")
    if float(plan["theta"]) != rf.THETA:
        raise ValueError(f"plan theta {plan['theta']} is not the referee's {rf.THETA}")
    for e in [*plan["jobs"], plan["repro"]]:
        unc = [(str(f), int(d)) for f, d in e["uncertified"]]
        sig = rf.signature_key(rf.certsafe_signature(rf.map_from_json(e["map"]), unc, rf.THETA))
        if sig != e["signature"]:
            raise ValueError(f"{e['arm']}: the map and uncertified set do not give the signature")


def arbiter_of(entry: Entry) -> Callable[[str, int, float], Any]:
    """The arm runner's arbiter factory of a plan entry (CS arm wrapper of the frozen driver)."""
    m = rf.map_from_json(entry["map"])
    unc = [(str(f), int(d)) for f, d in entry["uncertified"]]

    def make(arm: str, sd: int, warmup_s: float) -> Any:
        return rf.Counting(rf.certsafe_arbiter(m, unc, rf.THETA, rf.K_CONF), is_unit=True)

    return make


def job(seed: int, entry: Entry, *, smoke: bool = False, short: float | None = None) -> Record:
    """One episode of a plan entry in the certified-safe record layout, sub `x18`."""
    study3.check_stage_seed(seed, STAGE)
    rec = study2.arm_job(
        seed, entry["arm"], arbiter_of(entry), None, smoke, short, sub=SUB, cs_stage=STAGE
    )
    rec["x18"] = {"sig_id": entry.get("sig_id"), "signature": entry["signature"]}
    return rec


def jobs(
    plan: Mapping[str, Any],
    seeds: Sequence[int] | None = None,
    *,
    smoke: bool = False,
    short: float | None = None,
) -> list[Job]:
    """The reproduction jobs, then every plan entry on every seed, seed-major."""
    lo, hi = plan["seeds"]
    todo = list(seeds) if seeds else list(range(lo, hi + 1))
    rep = plan["repro"]
    pairs = [(s, rep) for s in rep["seeds"] if s in todo]
    pairs += [(s, e) for s in todo for e in plan["jobs"]]
    return [
        Job(
            (s, e["arm"]),
            study3.check_stage_seed(s, STAGE),
            lambda s=s, e=e: job(s, e, smoke=smoke, short=short),
        )
        for s, e in pairs
    ]


def head(
    part: Part,
    plan_path: Path,
    plan: Mapping[str, Any],
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
        plan_sha256=sha_lf(plan_path),
        certsafe_freeze=study3.PROTOCOL.status(),
        consts={
            "arms": [e["arm"] for e in plan["jobs"]],
            "repro": plan["repro"]["arm"],
            "smoke": {"short": short} if smoke else None,
        },
    )


def run(
    plan_path: Path,
    out: Path,
    part: Part,
    *,
    smoke: bool = False,
    short: float | None = None,
    seeds: Sequence[int] | None = None,
) -> int:
    """Runs shard `part` of the plan's jobs into `out`; returns the number of jobs run.

    Raises:
        ValueError: `short` without `smoke`, or a plan that does not check.
        RuntimeError: A full run off Linux (the stored records are Linux numerics).
    """
    if short and not smoke:
        raise ValueError("--short is smoke-only")
    if not smoke and not sys.platform.startswith("linux"):
        raise RuntimeError(f"X18 Part C: Linux numerics only; platform {sys.platform}")
    plan = read_plan(plan_path)
    check_plan(plan)
    show = ("seed", "arm", "psvr", "rlf", "st_req", "st_rej", "cpu_s")
    return run_jobs(
        out,
        head(part, plan_path, plan, smoke=smoke, short=short),
        jobs(plan, seeds, smoke=smoke, short=short),
        part,
        smoke,
        show,
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--plan", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--short", type=float, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: run a shard of the X18 Part C jobs."""
    a = _parser().parse_args(argv)
    run(
        Path(a.plan),
        Path(a.out),
        Part.parse(a.part),
        smoke=a.smoke,
        short=a.short,
        seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

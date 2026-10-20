"""Tier t1: a CPU smoke run of every pipeline stage, under 10 minutes.

1. Study 4: one R2 n 500 dataset per environment through `corr`, `pcorr_eq` and `pmrt_eq`, and
   one through `pmrt_nl_eq` (DEV tune seed 3000000, which the EVAL records also hold); the
   declarations are compared with the stored records of the same keys.
2. Two short system episodes (Study 1 EVAL and placebo, `--smoke --short 120`).
3. PMRT at B 199 on the stored Study 1 placebo episodes: no combination declares an edge.
4. One referee job (Study 3 EVAL, `CS:PMRT`, `--smoke --short 120`).
5. The host's numeric fingerprint.
"""

import gzip
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from oran_causal.experiments.campaign.run import Providers, RunOptions, run_units
from oran_causal.experiments.campaign.spec import Unit, expand
from oran_causal.experiments.launch.fingerprint import fingerprint
from oran_causal.experiments.paths import ARTIFACTS, SPECS
from oran_causal.experiments.reproduce.checks import Check, Status
from oran_causal.experiments.reproduce.context import Context, Item, guarded
from oran_causal.experiments.system import stages
from oran_causal.experiments.system.common import read_records
from oran_causal.experiments.system.study1_discovery import SPLIT, build_caches, find_records
from oran_causal.methods.system import pmrt
from oran_causal.methods.system.artifact import load_artifact, pmrt_config

BUDGET_S = 600.0
SEED = 3000000
LINEAR_ARMS = ("corr", "pcorr_eq", "pmrt_eq")
GBM_ARM = "pmrt_nl_eq"
WORLDS = ("E1", "E2", "E3", "E4", "E5")
MIN_AGREEMENT = 0.9
SHORT_S = 120
PMRT_B = 199
PLACEBO = "study1_raw/e6p-disc-v4plc-1-a"
EVAL_MERGED = "study4_eval/eval/merged.jsonl.gz"
REFEREE_ARM = "CS:PMRT"


def smoke_spec(full: Mapping[str, Any]) -> dict[str, Any]:
    """The t1 Study 4 spec: the four arms of the EVAL spec on one tune seed, R2, n 500."""
    block = {
        "role": "tune",
        "regimes": ["R2"],
        "ns": [500],
        "kappas": [0.25],
        "seeds": [SEED, SEED],
    }
    return {
        "name": "t1_smoke",
        "budget_cpu_s": None,
        "arms": {a: full["arms"][a] for a in (*LINEAR_ARMS, GBM_ARM)},
        "blocks": [
            {**block, "worlds": list(WORLDS), "arms": list(LINEAR_ARMS)},
            {**block, "worlds": [WORLDS[0]], "arms": [GBM_ARM]},
        ],
    }


def stored_records(path: Path, keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The records of `keys` in a merged `.jsonl.gz` file (lines prefiltered by key text)."""
    want = {f'"key": "{k}"': k for k in keys}
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            head = line[: line.find(",")]
            if head[1:] in want:
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


def declared(rec: Mapping[str, Any]) -> dict[tuple[str, str], bool]:
    """(source, target) -> declared of a record's edges."""
    return {(e["source"], e["target"]): bool(e["declared"]) for e in rec.get("edges") or ()}


def agreement(mine: Mapping[str, Any], stored: Mapping[str, Any]) -> float:
    """Share of edges with the same declaration in two records of one unit."""
    a, b = declared(mine), declared(stored)
    keys = set(a) | set(b)
    return sum(a.get(k) == b.get(k) for k in keys) / len(keys) if keys else 1.0


def study4(ctx: Context) -> list[Check]:
    """The Study 4 arms on one dataset per environment, compared with the stored records."""
    spec = smoke_spec(json.loads((SPECS / "eval" / "full.json").read_text(encoding="utf-8")))
    units: list[Unit] = expand(spec)
    out = ctx.item_dir("t1") / "study4.jsonl"
    out.unlink(missing_ok=True)
    run_units(spec, units, out, RunOptions(isolate=False, providers=Providers(log=lambda _: None)))
    mine = {r["key"]: r for r in read_records(out)}
    bad = sorted(k for k, r in mine.items() if r.get("status") != "ok")
    ok = not bad and len(mine) == len(units)
    detail = f"{len(mine)}/{len(units)} ok" if not bad else f"not ok: {bad[:3]}"
    checks = [Check("t1 study4", "units", Status.PASS if ok else Status.DIFF, detail)]
    eval_path = ctx.records / EVAL_MERGED
    if not eval_path.exists():
        return [
            *checks,
            Check("t1 study4", "declarations", Status.SKIP, "EVAL records not present"),
        ]
    stored = stored_records(eval_path, mine)
    return [*checks, *(_arm_agreement(a, mine, stored) for a in (*LINEAR_ARMS, GBM_ARM))]


def _arm_agreement(arm: str, mine: Mapping[str, Any], stored: Mapping[str, Any]) -> Check:
    keys = sorted(k for k in mine if mine[k]["arm"] == arm and k in stored)
    if not keys:
        return Check("t1 study4", arm, Status.DIFF, "no stored record of the same key")
    rates = [agreement(mine[k], stored[k]) for k in keys]
    worst = min(rates)
    status = Status.PASS if worst >= MIN_AGREEMENT else Status.DIFF
    return Check(
        "t1 study4", arm, status, f"{len(keys)} datasets, declarations agree {worst:.1%} at worst"
    )


def episodes(ctx: Context) -> list[Check]:
    """Two short Study 1 episodes (EVAL and placebo)."""
    checks = []
    for stage in ("eval", "placebo"):
        out = ctx.item_dir("t1") / f"study1_{stage}.jsonl"
        out.unlink(missing_ok=True)
        n = stages.run_stage(1, stage, out=out, smoke=True, short=SHORT_S)
        kinds = [r.get("kind") for r in read_records(out)]
        ok = n == 1 and kinds.count("episode") == 1
        status = Status.PASS if ok else Status.DIFF
        checks.append(Check("t1 system", f"study1 {stage} episode", status, f"{n} job, {kinds}"))
    return checks


def pmrt_placebo(ctx: Context) -> list[Check]:
    """PMRT at B 199 on the stored placebo episodes: every combination declares nothing."""
    src = ctx.records / PLACEBO
    if not src.exists():
        return [Check("t1 pmrt", f"B {PMRT_B}", Status.SKIP, "Study 1 placebo records not present")]
    cache = build_caches(
        find_records([str(src)]), "placebo", ctx.item_dir("t1") / "cache", "placebo", 1
    )
    pool = pmrt.load_pmrt_pool([str(p) for p in cache], stages=["placebo"])
    params, priors = load_artifact(ARTIFACTS / "study1_pmrt.json")
    run = pmrt.run_integrated(
        pmrt.pmrt_data(pool), params, pmrt_config(params, PMRT_B), SPLIT["placebo"]
    )
    decl = pmrt.declare_all(run, priors, int(pool.pool.n_eps))
    counts = {pmrt.cname(*k): sum(bool(v["declared"]) for v in d.values()) for k, d in decl.items()}
    status = Status.PASS if counts and not any(counts.values()) else Status.DIFF
    return [
        Check(
            "t1 pmrt",
            f"B {PMRT_B} placebo",
            status,
            f"{len(counts)} combinations, {sum(counts.values())} declarations",
        )
    ]


def referee(ctx: Context) -> list[Check]:
    """One short Study 3 referee job."""
    out = ctx.item_dir("t1") / "study3_eval.jsonl"
    out.unlink(missing_ok=True)
    n = stages.run_stage(3, "eval", out=out, smoke=True, short=SHORT_S, arms=[REFEREE_ARM])
    jobs = [r for r in read_records(out) if r.get("kind") == "job"]
    ok = n == 1 and len(jobs) == 1 and jobs[0].get("arm") == REFEREE_ARM
    return [
        Check(
            "t1 referee",
            f"study3 {REFEREE_ARM}",
            Status.PASS if ok else Status.DIFF,
            f"{len(jobs)} job record",
        )
    ]


def host(ctx: Context) -> list[Check]:
    """The numeric fingerprint of this host, written to `<out>/t1/fingerprint.json`."""
    fp = fingerprint()
    (ctx.item_dir("t1") / "fingerprint.json").write_text(
        json.dumps(fp, indent=1, default=str), encoding="utf-8"
    )
    return [Check("t1 host", "fingerprint", Status.PASS, f"{len(fp)} fields")]


STEPS: dict[str, Item] = {
    "study4": study4,
    "episodes": episodes,
    "pmrt": pmrt_placebo,
    "referee": referee,
    "host": host,
}


def run(
    ctx: Context, steps: Mapping[str, Item] = STEPS, clock: Callable[[], float] = time.time
) -> list[Check]:
    """Every t1 step, then the wall-time check against `BUDGET_S`."""
    t0 = clock()
    checks = [c for name, step in steps.items() for c in guarded(f"t1 {name}", step, ctx)]
    wall = clock() - t0
    status = Status.PASS if wall <= BUDGET_S else Status.DIFF
    return [*checks, Check("t1", "wall time", status, f"{wall:.0f} s (limit {BUDGET_S:.0f} s)")]

"""X1: the paired calibration block and the dedicated paper timing block of the runtime table.

Both blocks run one single-threaded process per vCPU (the session's cgroup CPU quota, never
oversubscribed) and stamp every unit with the platform, CPU model, vCPU quota, load before and
after, code commit and package versions, in the campaign record format, so
`oran_causal.analysis.posthoc.x1_runtime` reads them directly.

* `calib`: speed factors. Six anchor arms spanning cheap, medium and expensive, n 500, 1000 and
  4000 (mscr only up to 1000), E2 R2 kappa .25, the DEV seeds 3_000_000-002. Run as its own job on
  the reference host and on each other host type.
* `paper`: the runtime table of the paper. Every Study 4 arm (the EVAL spec arms minus pdcor and
  cmi_knn, plus pmrt_nl_eq and cdl, minus mscr_eq_min) over the n grid (mscr up to 1000), E2 R2
  (granger: E3 R2), seeds 3_000_000-002, on one uncontended reference session.

    python -m oran_causal.experiments.posthoc.x1_timing make-spec
    python -m oran_causal.experiments.posthoc.x1_timing units --block calib
    python -m oran_causal.experiments.posthoc.x1_timing run --block calib --out-dir D

Launching a block on Kaggle, Colab or the VPS is `oran_causal.experiments.launch`.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from oran_causal.analysis.records import job_key
from oran_causal.core.data import Method
from oran_causal.experiments.campaign.host import PLATFORM_ENV, cpu_quota, host_info, load_info
from oran_causal.experiments.campaign.provenance import (
    COMMIT_ENV,
    DIRTY_ENV,
    LOCK_ENV,
    code_info,
    pkg_versions,
)
from oran_causal.experiments.harness.records import Job, Record, append_record, clean, run_one
from oran_causal.experiments.harness.runner import load_method
from oran_causal.experiments.launch.commands import LOCK_REL, THREADS, SetupPlan, setup_cmd
from oran_causal.experiments.paths import SPECS
from oran_causal.sim.controlled.datasets.generate import generate_dataset

VERSION = "xm-exp-c-timing/1"
SPEC_PATH = SPECS / "posthoc" / "timing.json"
EVAL_SPEC_PATH = SPECS / "eval" / "full.json"
EVAL_SPEC_LABEL = "specs/eval/full.json"
CALIB_ANCHORS = ("pcorr_eq", "rcot2_eq", "shap_dag", "pmrt_nl_eq", "cdl", "mscr_eq_min")
DROPPED = ("pdcor", "cmi_knn")
EXTRA_ARMS = {
    "pmrt_nl_eq": {
        "ref": "pmrt_core",
        "config": {"covariates": "eq", "statistic": "gbm"},
        "source": "specs/dev/pmrt_nl.json (R-42 / R-52 winner)",
    },
    "cdl": {
        "ref": "cdl",
        "config": {"arm": "native"},
        "source": "specs/dev/cdl.json (R-50)",
    },
}
CALIB_ONLY_ARMS = {
    "mscr_eq_min": {"ref": "mscr", "config": {"arm": "eq_min"}},
}
MAX_N = {"mscr_eq": 1000, "mscr_native": 1000, "mscr_eq_min": 1000}
PAPER_EXCLUDE = ("mscr_eq_min",)
N_GRID = [500, 1000, 4000, 8000, 24000]
SEEDS = [3_000_000, 3_000_001, 3_000_002]
KAPPA = 0.25
UNKNOWN_COST_S_PER_1000_ROWS = 30.0
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
WORKER_MODULE = "oran_causal.experiments.posthoc.x1_timing"

Spec = dict[str, Any]


def session_command(
    block: str, out_dir: str, platform: str, code: Mapping[str, Any], wall_s: int
) -> str:
    """The shell command of a cloud session that times one block.

    The environment setup of `launch.commands.setup_cmd` (frozen `uv sync`, then the pin check),
    then one `run` of the block under the platform's wall limit.
    """
    env = (
        f"{PLATFORM_ENV}={platform} {COMMIT_ENV}={code['commit']} "
        f"{DIRTY_ENV}={int(bool(code['dirty']))} {LOCK_ENV}={LOCK_REL} {THREADS}"
    )
    return (
        setup_cmd(SetupPlan(out_dir=out_dir))
        + f"{env} timeout {wall_s} python -u -m {WORKER_MODULE} run --block {block} "
        f"--out-dir {out_dir}"
    )


def make_spec(
    eval_spec: Mapping[str, Any],
    dev_by_world_regime: Iterable[Mapping[str, Any]] = (),
    eval_label: str = EVAL_SPEC_LABEL,
) -> Spec:
    """The timing spec: the EVAL arms without the dropped ones, plus the extra arms.

    `pmrt_nl_eq` and `cdl` are added; `mscr_eq_min` is the calibration anchor that the EVAL spec
    no longer carries, kept unless the EVAL spec has it.

    Args:
        eval_spec: The frozen EVAL spec.
        dev_by_world_regime: The DEV runtime table's `by_world_regime` rows; their medians are the
            estimated unit costs used to balance the processes.
        eval_label: The EVAL spec path recorded in `source`.
    """
    arms = {
        a: {
            "ref": d["ref"],
            "config": d.get("config", {}),
            **({"worlds": d["worlds"]} if "worlds" in d else {}),
        }
        for a, d in eval_spec["arms"].items()
        if not a.startswith(DROPPED)
    }
    arms.update({a: dict(d) for a, d in EXTRA_ARMS.items()})
    for a, d in CALIB_ONLY_ARMS.items():
        arms.setdefault(a, dict(d))
    for a, m in MAX_N.items():
        if a in arms:
            arms[a]["max_n"] = m
    est = {
        f"{r['arm']}|{r['world']}|{r['regime']}|n{r['n']}": r["median"] for r in dev_by_world_regime
    }
    return {
        "name": "exp_c_timing",
        "version": VERSION,
        "kappa": KAPPA,
        "seeds": SEEDS,
        "source": f"arms from {eval_label} minus {list(DROPPED)}, plus {sorted(EXTRA_ARMS)}",
        "arms": dict(sorted(arms.items())),
        "blocks": {
            "calib": {"arms": list(CALIB_ANCHORS), "ns": [500, 1000, 4000], "cell": ["E2", "R2"]},
            "paper": {
                "arms": "all",
                "exclude": list(PAPER_EXCLUDE),
                "ns": N_GRID,
                "cell": ["E2", "R2"],
                "cell_E3_only": ["E3", "R2"],
            },
        },
        "est_cost_s": est,
    }


def load_spec(path: Path = SPEC_PATH) -> Spec:
    """Reads the timing spec."""
    return json.loads(path.read_text(encoding="utf-8"))


@dataclasses.dataclass(frozen=True)
class TimingUnit:
    """One (arm, dataset) timing unit.

    Attributes:
        key: The job key.
        arm: Arm name.
        world: World id.
        regime: Regime id.
        n: Rows.
        seed: Dataset seed.
        kappa: Observation-noise level.
        est_cost_s: Estimated cost in reference seconds, used to balance the processes.
    """

    key: str
    arm: str
    world: str
    regime: str
    n: int
    seed: int
    kappa: float
    est_cost_s: float

    def to_json(self) -> dict[str, Any]:
        """The unit as a JSON object, keys in the order the units files use."""
        return dataclasses.asdict(self)


def est_cost(spec: Mapping[str, Any], arm: str, w: str, r: str, n: int) -> float:
    """DEV median cost of an (arm, world, regime, n) cell; unknown cells cost 30 s per 1000 rows."""
    e = spec.get("est_cost_s", {}).get(f"{arm}|{w}|{r}|n{n}")
    return float(e) if e is not None else UNKNOWN_COST_S_PER_1000_ROWS * n / 1000


def units(spec: Mapping[str, Any], block: str) -> list[TimingUnit]:
    """The units of a block, arm by arm, n by n, seed by seed."""
    b = spec["blocks"][block]
    arms = (
        [a for a in spec["arms"] if a not in b.get("exclude", [])]
        if b["arms"] == "all"
        else b["arms"]
    )
    out = []
    for a in arms:
        d = spec["arms"][a]
        w, r = b["cell"]
        if d.get("worlds") and w not in d["worlds"]:
            w, r = b.get("cell_E3_only", [d["worlds"][0], r])
        for n in b["ns"]:
            if d.get("max_n") and n > d["max_n"]:
                continue
            for s in spec["seeds"]:
                out.append(
                    TimingUnit(
                        job_key(a, w, r, None, n, s, spec["kappa"]),
                        a,
                        w,
                        r,
                        n,
                        s,
                        spec["kappa"],
                        est_cost(spec, a, w, r, n),
                    )
                )
    return out


def assign(us: Sequence[TimingUnit], procs: int) -> list[list[TimingUnit]]:
    """Longest-processing-time first: the costliest unit to the least-loaded process.

    Balances wall time while keeping one process per vCPU.
    """
    bins: list[list[TimingUnit]] = [[] for _ in range(procs)]
    load = [0.0] * procs
    for u in sorted(us, key=lambda u: (-u.est_cost_s, u.key)):
        i = load.index(min(load))
        bins[i].append(u)
        load[i] += u.est_cost_s
    return bins


@dataclasses.dataclass(frozen=True)
class Providers:
    """Environment queries of a timing worker; tests replace them for hermetic runs."""

    code: Callable[[], dict[str, Any]] = code_info
    host: Callable[[], dict[str, Any]] = lambda: {**host_info(), "cpu_quota": cpu_quota()}
    load: Callable[[], dict[str, Any]] = load_info
    pkgs: Callable[[], dict[str, Any]] = pkg_versions
    method: Callable[[str], Method] = load_method
    generate: Callable[..., tuple[Any, Any]] = generate_dataset
    run_one: Callable[[Method, Job], Record] = run_one
    clock: Callable[[], float] = time.process_time
    log: Callable[[str], None] = print


def _failed(u: TimingUnit, cfg: Mapping[str, Any]) -> Record:
    return {
        "key": u.key,
        "error": traceback.format_exc()[-2000:],
        "config": dict(cfg),
        "job": {
            "world": u.world,
            "regime": u.regime,
            "lam": None,
            "n": u.n,
            "seed": u.seed,
            "kappa": u.kappa,
        },
    }


def done_keys(path: Path) -> set[str]:
    """Keys of the ok records already in a worker's output."""
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add(r["key"])
    return done


def worker(
    spec: Mapping[str, Any],
    us: Sequence[TimingUnit],
    out: Path,
    block: str,
    providers: Providers | None = None,
) -> int:
    """Times the units one after the other and appends one record per unit.

    A unit whose ok record is already in `out` is skipped, so a cut worker resumes.

    Returns:
        The number of units run.
    """
    p = providers or Providers()
    done = done_keys(out)
    host, pkgs, code = p.host(), p.pkgs(), p.code()
    methods: dict[str, Method] = {}
    n_run = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for u in us:
            if u.key in done:
                continue
            a = spec["arms"][u.arm]
            cfg = dict(a.get("config", {}))
            load0 = p.load()
            t0 = p.clock()
            ds, truth = p.generate(u.world, u.regime, u.n, u.seed, lam=1.0, kappa=u.kappa)
            gen = p.clock() - t0
            try:
                if u.arm not in methods:
                    methods[u.arm] = p.method(a["ref"])
                rec = p.run_one(methods[u.arm], Job(u.key, ds, truth, cfg, gen))
            except Exception:
                rec = _failed(u, cfg)
            rec.update(
                status="ok" if rec.get("error") is None else "error",
                arm=u.arm,
                role="timing",
                block=block,
                spec_name=f"exp_c_{block}",
                campaign=VERSION,
                host=host,
                pkgs=pkgs,
                code=code,
                load={"start": load0, "end": p.load()},
            )
            append_record(fh, clean(rec))
            n_run += 1
            p.log(
                f"[exp-c] {u.key} cpu {rec.get('cpu_s') or float('nan'):.1f}s wall "
                f"{rec.get('wall_s') or float('nan'):.1f}s {rec['status']}"
            )
    return n_run


Spawn = Callable[[Sequence[str], Mapping[str, str], Path], subprocess.Popen]


def spawn_process(argv: Sequence[str], env: Mapping[str, str], log: Path) -> subprocess.Popen:
    """Starts a worker process with its output appended to `log`."""
    return subprocess.Popen(argv, env=dict(env), stdout=log.open("a"), stderr=subprocess.STDOUT)


def worker_env(base: Mapping[str, str]) -> dict[str, str]:
    """The environment of a worker: one thread per numerical library."""
    return {**base, **dict.fromkeys(THREAD_ENV, "1")}


def run(
    spec: Mapping[str, Any],
    block: str,
    out_dir: Path,
    procs: int | None,
    max_units: int | None,
    *,
    environ: Mapping[str, str],
    spawn: Spawn = spawn_process,
    quota: Callable[[], int] = cpu_quota,
    host: Callable[[], dict[str, Any]] = host_info,
    python: str = sys.executable,
) -> int:
    """Runs a block: one worker process per vCPU, balanced by estimated cost.

    Returns:
        The largest worker exit code.
    """
    us = units(spec, block)
    us = us[:max_units] if max_units else us
    p = procs or quota()
    bins = assign(us, p)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = worker_env(environ)
    t0 = time.time()
    ps = []
    for i, b in enumerate(bins):
        units_file = out_dir / f"units_{i}.json"
        units_file.write_text(json.dumps([u.to_json() for u in b], indent=0), encoding="utf-8")
        argv = [
            python,
            "-u",
            "-m",
            WORKER_MODULE,
            "worker",
            "--block",
            block,
            "--units",
            str(units_file),
            "--out",
            str(out_dir / f"res_{i}.jsonl"),
        ]
        ps.append(spawn(argv, env, out_dir / f"log_{i}.txt"))
    rcs = [q.wait() for q in ps]
    info = {
        "version": VERSION,
        "block": block,
        "procs": p,
        "cpu_quota": quota(),
        "units": len(us),
        "wall_s": time.time() - t0,
        "rcs": rcs,
        "host": host(),
        "est_cost_s_per_proc": [sum(u.est_cost_s for u in b) for b in bins],
    }
    (out_dir / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info), flush=True)
    return max(rcs, default=0)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-spec")
    p.add_argument("--eval-spec", default=str(EVAL_SPEC_PATH))
    p.add_argument("--dev-runtime", default=None)
    p.add_argument("--out", default=str(SPEC_PATH))
    p = sub.add_parser("units")
    p.add_argument("--block", required=True)
    p.add_argument("--procs", type=int, default=4)
    p = sub.add_parser("run")
    p.add_argument("--block", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p = sub.add_parser("worker")
    p.add_argument("--block", required=True)
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    return ap


def _print_units(spec: Mapping[str, Any], block: str, procs: int) -> None:
    us = units(spec, block)
    bins = assign(us, procs)
    tot = sum(u.est_cost_s for u in us)
    wall = max(sum(u.est_cost_s for u in b) for b in bins) / 3600
    print(
        f"[exp-c] {block}: {len(us)} units, est. {tot / 3600:.2f} CPU-h (DEV medians; unknown 30 "
        f"s/1000 rows), wall at {procs} procs ~{wall:.2f} h"
    )
    for arm in dict.fromkeys(u.arm for u in us):
        xs = [u for u in us if u.arm == arm]
        print(
            f"  {arm}: {len(xs)} units, n {sorted({u.n for u in xs})}, {xs[0].world} "
            f"{xs[0].regime}, est {sum(u.est_cost_s for u in xs):.0f} s"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: `make-spec`, `units`, `run` and `worker`."""
    a = _parser().parse_args(argv)
    if a.cmd == "make-spec":
        ev = json.loads(Path(a.eval_spec).read_text(encoding="utf-8"))
        dev = []
        if a.dev_runtime and Path(a.dev_runtime).exists():
            dev = json.loads(Path(a.dev_runtime).read_text(encoding="utf-8"))["by_world_regime"]
        sp = make_spec(ev, dev)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(sp, indent=1), encoding="utf-8", newline="\n")
        print(
            f"[exp-c] {a.out}: {len(sp['arms'])} arms; calib {len(units(sp, 'calib'))} units, "
            f"paper {len(units(sp, 'paper'))} units"
        )
    elif a.cmd == "units":
        _print_units(load_spec(), a.block, a.procs)
    elif a.cmd == "run":
        return run(load_spec(), a.block, Path(a.out_dir), a.procs, a.max_units, environ=os.environ)
    else:
        raw = json.loads(Path(a.units).read_text(encoding="utf-8"))
        worker(load_spec(), [TimingUnit(**u) for u in raw], Path(a.out), a.block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Running campaign units: budgets, isolation, infeasibility rules and record stamps.

DEV rules: a unit over its budget is infeasible with its measured cost; larger n of that
(arm, cell) in the part, and with a shared registry every unit of that arm at n >= the
infeasible n, are recorded infeasible without running. EVAL rules: feasibility comes only from
the spec (`infeasible_n`), the budget is a safety cap (`EVAL_CAP_FACTOR` x budget), nothing
propagates, the budget must be the spec's and units must run isolated. A child killed below its
limits is an error in both modes.
"""

import dataclasses
import json
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from oran_causal.core.data import Dataset, Method, Truth
from oran_causal.experiments.campaign.host import campaign_procs, cpu_quota, host_info, load_info
from oran_causal.experiments.campaign.integrity import (
    EVAL_CAP_FACTOR,
    accept_rule,
    check_eval_preconditions,
    check_one_per_cpu,
    integrity_stamp,
    mode_tag,
    record_mode_tag,
    run_mode,
)
from oran_causal.experiments.campaign.provenance import (
    PythonPin,
    code_info,
    pin_check,
    pkg_versions,
    read_lock,
    workspace_python,
)
from oran_causal.experiments.campaign.shard import CostTable, complete_dataset_keys, partition
from oran_causal.experiments.campaign.spec import (
    DatasetKey,
    ProtocolTexts,
    Unit,
    expand,
    is_eval,
    shipped_protocol,
    spec_sha256,
)
from oran_causal.experiments.harness.isolation import (
    IsolatedOutcome,
    can_isolate,
    classify_kill,
    run_isolated,
)
from oran_causal.experiments.harness.records import (
    Job,
    append_record,
    clean,
    done_keys,
    read_records,
    run_one,
)
from oran_causal.experiments.harness.runner import Log, Resolver, load_method, registry_resolve
from oran_causal.sim.controlled.datasets.generate import generate_dataset

CAMPAIGN_VERSION = "oran-causal-campaign/1"

type Record = dict[str, Any]
type Spec = dict[str, Any]
type Family = tuple[str, str, str, float | None, float]


def _locked_pins(python: PythonPin | None) -> dict[str, Any]:
    return pin_check(read_lock(None), python)


@dataclasses.dataclass(frozen=True)
class Providers:
    """Environment queries and sinks of a run; tests replace them for hermetic runs."""

    code: Callable[[], dict[str, Any]] = code_info
    pins: Callable[[PythonPin | None], dict[str, Any]] = _locked_pins
    python_pin: Callable[[], PythonPin | None] = workspace_python
    host: Callable[[], dict[str, Any]] = host_info
    load: Callable[[], dict[str, Any]] = load_info
    pkgs: Callable[[], dict[str, Any]] = pkg_versions
    procs: Callable[[], int | None] = campaign_procs
    quota: Callable[[], int] = cpu_quota
    texts: Callable[[], ProtocolTexts] = shipped_protocol
    resolve: Resolver = registry_resolve
    log: Log = print


@dataclasses.dataclass(frozen=True)
class RunOptions:
    """Run settings.

    Attributes:
        budget: CPU-s budget per unit (None: none; EVAL must equal the spec's).
        isolate: Fork + RLIMIT_CPU per unit; None picks it where available.
        spec_file_sha256: LF sha256 of the spec file, stamped on records.
        skip_keys: Keys not to run (datasets complete on another platform).
        eval_rules: Apply EVAL rules; None follows the spec.
        registry: Shared infeasibility registry file (DEV only).
        stamp: Run-level integrity fields of the caller (cost table sha, part, parts).
        providers: Environment queries.
    """

    budget: float | None = None
    isolate: bool | None = None
    spec_file_sha256: str | None = None
    skip_keys: frozenset[str] = frozenset()
    eval_rules: bool | None = None
    registry: Path | None = None
    stamp: dict[str, Any] | None = None
    providers: Providers = dataclasses.field(default_factory=Providers)


@dataclasses.dataclass(frozen=True)
class Shard:
    """One part of a spec's units.

    Attributes:
        index: Part index.
        count: Number of parts.
        cost_table: Cost estimates that balance the parts.
        skip_complete_from: Shard files of an earlier session whose complete datasets are skipped.
    """

    index: int
    count: int
    cost_table: CostTable | None = None
    skip_complete_from: Sequence[Path] = ()


@dataclasses.dataclass
class _Context:
    spec: Spec
    ev: bool
    isolate: bool
    budget: float | None
    registry: Path | None
    fixed: dict[str, Any]
    providers: Providers


@dataclasses.dataclass
class _State:
    dataset_key: DatasetKey | None = None
    ds: Dataset | None = None
    truth: Truth | None = None
    gen: float = 0.0
    methods: dict[str, Method | str] = dataclasses.field(default_factory=dict)
    over: dict[Family, tuple[int, float, str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _Limits:
    cpu_budget: float | None
    wall_budget: float | None
    cpu: float | None
    wall: float | None


@dataclasses.dataclass(frozen=True)
class _UnitRun:
    unit: Unit
    config: dict[str, Any]
    limits: _Limits


@dataclasses.dataclass(frozen=True)
class Overrun:
    """A measured budget breach.

    Attributes:
        cost: Measured cost in `kind` units.
        kind: `"CPU-s"` or `"wall-s"`.
        budget: The arm's budget.
        limit: The enforced limit (EVAL: the safety cap).
        rss: Peak memory in MB.
        cpu: CPU seconds when `kind` is wall-s.
    """

    cost: float
    kind: str
    budget: float | None
    limit: float
    rss: float | None = None
    cpu: float | None = None


def arm_budgets(spec: Spec, arm: str, budget: float | None) -> tuple[float | None, float | None]:
    """(CPU-s budget, wall-s budget) of an arm; the arm's `budget_cpu_s` overrides the run's."""
    a = spec["arms"][arm]
    cpu = a.get("budget_cpu_s", budget)
    wall = float(a["budget_wall_s"]) if a.get("budget_wall_s") else None
    return (float(cpu) if cpu else None), wall


def read_infeasible_registry(path: Path | None) -> dict[str, tuple[int, float, str]]:
    """Shared DEV registry: arm -> (smallest n over budget, measured cost, cost kind)."""
    out: dict[str, tuple[int, float, str]] = {}
    for r in read_records(path) if path else ():
        if r["arm"] not in out or int(r["n"]) < out[r["arm"]][0]:
            out[r["arm"]] = (int(r["n"]), float(r["cost"]), r.get("kind", "CPU-s"))
    return out


def infeasible_record(
    u: Unit, cfg: dict[str, Any], cpu_s: float | None, reason: str, rss_mb: float | None = None
) -> Record:
    """Record of a unit that is infeasible (not run, or over its budget)."""
    job = {"world": u.world, "regime": u.regime, "lam": u.lam, "n": u.n, "seed": u.seed}
    return {
        "key": u.key,
        "method": u.arm,
        "status": "infeasible",
        "error": None,
        "reason": reason,
        "job": {**job, "kappa": u.kappa},
        "cpu_s": cpu_s,
        "peak_rss_mb": rss_mb,
        "config": cfg,
    }


def failed_record(
    u: Unit, cfg: dict[str, Any], cpu_s: float | None, error: str, rss_mb: float | None = None
) -> Record:
    """Record of a unit that failed (re-run on relaunch)."""
    rec = infeasible_record(u, cfg, cpu_s, "", rss_mb)
    return {**rec, "status": "error", "error": error, "reason": None}


def over_record(u: Unit, cfg: dict[str, Any], over: Overrun, ev: bool) -> Record:
    """Record of a unit over its budget, infeasible with the measured cost."""
    cpu_s = over.cost if over.kind == "CPU-s" else over.cpu
    if ev:
        reason = (
            f"EVAL safety cap {over.limit:.0f} {over.kind} ({EVAL_CAP_FACTOR:g} x budget "
            f"{over.budget}) exceeded (measured {over.cost:.0f}): infeasible, never re-run "
            "(PROTOCOL_A s.7)"
        )
    else:
        reason = f"exceeded the budget {over.budget} {over.kind} (measured {over.cost:.0f})"
    rec = infeasible_record(u, cfg, cpu_s, reason, over.rss)
    rec.update(cost=over.cost, cost_kind=over.kind)
    if over.kind == "wall-s":
        rec["wall_s"] = over.cost
    return rec


def out_modes(path: Path) -> set[tuple[str, str | None]]:
    """Run modes of the records already in `path`."""
    return {record_mode_tag(r) for r in read_records(path)}


def infeasible_keys(path: Path) -> set[str]:
    """Keys of the infeasible records in `path`."""
    return {r["key"] for r in read_records(path) if r.get("status") == "infeasible"}


def _eval_budget(spec: Spec, budget: float | None, isolate: bool) -> float | None:
    budget = spec.get("budget_cpu_s") if budget is None else budget
    if budget != spec.get("budget_cpu_s"):
        raise ValueError(
            f"EVAL refused: --budget {budget} differs from the spec budget_cpu_s "
            f"{spec.get('budget_cpu_s')} (R-59 F6)"
        )
    if not isolate:
        raise ValueError(
            "EVAL refused: units must run isolated (fork + RLIMIT_CPU); "
            "not available or disabled (R-59 F7)"
        )
    return budget


def _refuse_mixed_modes(out: Path, mode: dict[str, Any]) -> None:
    other = out_modes(out) - {mode_tag(mode)}
    if other:
        raise ValueError(
            f"{out} already holds records of another run mode {sorted(other, key=str)}: "
            "EVAL and DEV outputs must be separate files (R-35)"
        )


def _prepare(spec: Spec, out: Path, opts: RunOptions) -> _Context:
    p = opts.providers
    ev = is_eval(spec) if opts.eval_rules is None else opts.eval_rules
    isolate = can_isolate() if opts.isolate is None else opts.isolate
    python = p.python_pin() if is_eval(spec) else None
    code, pins = p.code(), p.pins(python)
    budget = opts.budget
    if is_eval(spec):
        check_eval_preconditions(spec, code, pins, p.texts(), python)
        check_one_per_cpu(p.procs(), p.quota())
        budget = _eval_budget(spec, budget, isolate)
    mode = run_mode(spec)
    _refuse_mixed_modes(out, mode)
    integrity = integrity_stamp(
        spec,
        code,
        pins,
        isolate=isolate,
        budget=budget,
        cap_factor=EVAL_CAP_FACTOR if ev else 1.0,
        spec_file_sha256=opts.spec_file_sha256,
        extra=opts.stamp,
    )
    fixed = {
        "code": code,
        "run_mode": mode,
        "integrity": integrity,
        "pkgs": p.pkgs(),
        "host": p.host(),
    }
    return _Context(spec, ev, isolate, budget, None if ev else opts.registry, fixed, p)


def _limits(ctx: _Context, arm: str) -> _Limits:
    cpu_b, wall_b = arm_budgets(ctx.spec, arm, ctx.budget)
    factor = EVAL_CAP_FACTOR if ctx.ev else 1.0
    cpu = factor * cpu_b if cpu_b else None
    wall = factor * wall_b if wall_b else None
    return _Limits(cpu_b, wall_b, cpu, wall)


def _family(u: Unit) -> Family:
    return (u.arm, u.world, u.regime, u.lam, u.kappa)


def _t3_reason(t3: Any, lim: _Limits) -> str:
    kind = "wall-s" if lim.wall_budget else "CPU-s"
    return f"infeasible at this n (DEV measured cost {float(t3):.0f} {kind}, T3)"


def _skip_record(run: _UnitRun, ctx: _Context, state: _State) -> Record | None:
    u, cfg = run.unit, run.config
    t3 = (ctx.spec["arms"][u.arm].get("infeasible_n") or {}).get(str(u.n))
    if t3 is not None:
        return infeasible_record(u, cfg, None, _t3_reason(t3, run.limits))
    fam = _family(u)
    if not ctx.ev and fam in state.over and u.n >= state.over[fam][0]:
        n0, c0, kind = state.over[fam]
        reason = f"not run: n {n0} exceeded the budget (measured {c0:.0f} {kind})"
        return infeasible_record(u, cfg, None, reason)
    reg = read_infeasible_registry(ctx.registry).get(u.arm) if ctx.registry else None
    if reg is not None and u.n >= reg[0]:
        reason = (
            f"not run: arm infeasible at n {reg[0]} in this session "
            f"(measured {reg[1]:.0f} {reg[2]}; T3 per (arm, n))"
        )
        return infeasible_record(u, cfg, None, reason)
    return None


def _load_dataset(state: _State, u: Unit) -> None:
    if state.dataset_key == u.dataset:
        return
    t0 = time.process_time()
    lam = 1.0 if u.lam is None else u.lam
    state.ds, state.truth = generate_dataset(u.world, u.regime, u.n, u.seed, lam=lam, kappa=u.kappa)
    state.gen, state.dataset_key = time.process_time() - t0, u.dataset


def _load_method(state: _State, u: Unit, ctx: _Context) -> Method | str:
    if u.arm not in state.methods:
        try:
            ref = ctx.spec["arms"][u.arm]["ref"]
            state.methods[u.arm] = load_method(ref, ctx.providers.resolve)
        except Exception:
            state.methods[u.arm] = traceback.format_exc()
    return state.methods[u.arm]


def _job(run: _UnitRun, state: _State) -> Job:
    if state.ds is None or state.truth is None:
        raise RuntimeError("dataset not loaded")
    return Job(run.unit.key, state.ds, state.truth, run.config, state.gen)


def _killed_record(run: _UnitRun, o: IsolatedOutcome, ev: bool) -> Record:
    lim, u, cfg = run.limits, run.unit, run.config
    died = o.signal is not None or o.wall_hit
    why = classify_kill(o.signal, o.cpu_s, lim.cpu, o.wall_hit) if died else "no_record"
    if why == "wall_limit" and lim.wall is not None:
        over = Overrun(o.wall_s, "wall-s", lim.wall_budget, lim.wall, o.maxrss_mb, o.cpu_s)
        return over_record(u, cfg, over, ev)
    if why == "cpu_limit" and lim.cpu is not None:
        over = Overrun(o.cpu_s, "CPU-s", lim.cpu_budget, lim.cpu, o.maxrss_mb)
        return over_record(u, cfg, over, ev)
    how = f"killed by signal {o.signal}" if o.signal else "exited without a record"
    msg = f"child {how} at {o.cpu_s:.0f} CPU-s, maxrss {o.maxrss_mb:.0f} MB (out of memory?)"
    return failed_record(u, cfg, o.cpu_s, msg, o.maxrss_mb)


def _isolated(run: _UnitRun, ctx: _Context, state: _State, method: Method) -> Record:
    job = _job(run, state)
    o = run_isolated(lambda: run_one(method, job), run.limits.cpu, run.limits.wall)
    if o.record is None:
        return _killed_record(run, o, ctx.ev)
    child = {"child_cpu_s": o.cpu_s, "child_maxrss_mb": o.maxrss_mb, "child_wall_s": o.wall_s}
    return {**o.record, **child}


def _in_process(run: _UnitRun, ctx: _Context, state: _State, method: Method) -> Record:
    rec = run_one(method, _job(run, state))
    lim, u, cfg = run.limits, run.unit, run.config
    rss = rec.get("peak_rss_mb")
    if lim.cpu and rec.get("cpu_s") is not None and rec["cpu_s"] > lim.cpu:
        over = Overrun(rec["cpu_s"], "CPU-s", lim.cpu_budget, lim.cpu, rss)
        return over_record(u, cfg, over, ctx.ev)
    if lim.wall and rec.get("wall_s") is not None and rec["wall_s"] > lim.wall:
        over = Overrun(rec["wall_s"], "wall-s", lim.wall_budget, lim.wall, rss, rec.get("cpu_s"))
        return over_record(u, cfg, over, ctx.ev)
    return rec


def _note_over(u: Unit, ctx: _Context, state: _State, rec: Record) -> None:
    if rec.get("status") != "infeasible" or rec.get("cost") is None:
        return
    state.over[_family(u)] = (u.n, rec["cost"], rec["cost_kind"])
    if ctx.registry:
        entry = {"arm": u.arm, "n": u.n, "cost": rec["cost"], "kind": rec["cost_kind"]}
        with ctx.registry.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**entry, "key": u.key}) + "\n")


def _execute(run: _UnitRun, ctx: _Context, state: _State) -> Record:
    _load_dataset(state, run.unit)
    method = _load_method(state, run.unit, ctx)
    if isinstance(method, str):
        rec = failed_record(run.unit, run.config, None, f"method load failed: {method[-1500:]}")
    elif ctx.isolate:
        rec = _isolated(run, ctx, state, method)
    else:
        rec = _in_process(run, ctx, state, method)
    _note_over(run.unit, ctx, state, rec)
    return rec


def _run_unit(u: Unit, ctx: _Context, state: _State) -> Record:
    run = _UnitRun(u, dict(ctx.spec["arms"][u.arm].get("config", {})), _limits(ctx, u.arm))
    load0 = ctx.providers.load()
    rec = _skip_record(run, ctx, state) or _execute(run, ctx, state)
    rec.setdefault("status", "ok" if rec.get("error") is None else "error")
    rec.update(arm=u.arm, role=u.role, **ctx.fixed)
    rec.update(
        limits={"cpu_s": run.limits.cpu, "wall_s": run.limits.wall},
        load={"start": load0, "end": ctx.providers.load()},
        campaign=CAMPAIGN_VERSION,
        spec_name=ctx.spec.get("name"),
    )
    return clean(rec)


def run_units(spec: Spec, units: Sequence[Unit], out: Path, opts: RunOptions | None = None) -> int:
    """Runs `units` (sorted dataset-major), appending one record each to `out`.

    Done and infeasible keys already in `out`, and `opts.skip_keys`, are skipped.

    Returns:
        The number of records written.
    """
    opts = opts or RunOptions()
    ctx = _prepare(spec, out, opts)
    done = done_keys(out) | infeasible_keys(out) | set(opts.skip_keys)
    state = _State()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_run = 0
    with out.open("a", encoding="utf-8") as fh:
        for u in (u for u in units if u.key not in done):
            rec = _run_unit(u, ctx, state)
            append_record(fh, rec)
            n_run += 1
            cpu = rec.get("cpu_s") or float("nan")
            ctx.providers.log(f"[xm-c] {u.key} {rec['status']} cpu {cpu:.2f}s")
    return n_run


def run_part(spec: Spec, shard: Shard, out: Path, opts: RunOptions | None = None) -> int:
    """Runs one part of the spec's units.

    Returns:
        The number of records written.
    """
    opts = opts or RunOptions()
    units = partition(expand(spec, opts.providers.texts()), shard.count, shard.cost_table)
    mine = units[shard.index]
    skip: set[str] = set()
    if shard.skip_complete_from:
        skip = complete_dataset_keys(mine, shard.skip_complete_from, accept_rule(spec))
    stamp = {
        "cost_table_sha256": spec_sha256(shard.cost_table) if shard.cost_table else None,
        "part": shard.index,
        "parts": shard.count,
    }
    budget = opts.budget if opts.budget is not None else spec.get("budget_cpu_s")
    run_opts = dataclasses.replace(opts, budget=budget, skip_keys=frozenset(skip), stamp=stamp)
    return run_units(spec, mine, out, run_opts)

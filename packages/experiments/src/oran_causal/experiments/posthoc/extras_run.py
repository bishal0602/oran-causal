"""Running the extras specs (X2, X3, X5, X6) through the campaign runner.

An extras spec is an ordinary campaign spec plus an `extras` block, run in DEV mode on its own
seed block. This module supplies what the runner lacks for them: the X2 dither design at
generation, the X3 and X6 told designs at the method call, and the X6 gbm bias log.

X2 (dither dose-response): `extras.design` = {delta, n_blocks} replaces the generator's dither
constants while a dataset is generated; delta 0.10 with 20 blocks is the frozen design.
X3 (design misspecification): the data stay as generated, an arm's `told` changes the design told
to the method (the arm's method, ref and config are the frozen arm's):

* `{"width": c}` every dither design's law U(-w, w) told as U(-c w, c w);
* `{"shift": f}` every dither design's setpoint switch times told s = floor(f L + 0.5) rows late
  (L = n / n_blocks rows per block);
* `{"lam": c}` the propensity of every logged design recomputed with lambda x c.

X6 adds `centre` and `redraw` ("told" | "true") to a width variant, for the pmrt_nl gbm arms.
"""

import contextlib
import dataclasses
import math
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc.paths import EXTRAS_PROTOCOL, lf_sha256
from oran_causal.analysis.posthoc.x2x3 import check_told
from oran_causal.core.data import Dataset, Design, Method, Truth
from oran_causal.core.pmrt import nonlinear
from oran_causal.experiments.campaign import extras
from oran_causal.experiments.campaign.run import (
    Execute,
    Providers,
    RunOptions,
    Shard,
    generated_dataset,
    plain_execute,
    run_part,
)
from oran_causal.experiments.campaign.spec import Unit
from oran_causal.experiments.harness.records import Job
from oran_causal.sim.controlled.datasets import e4_logged, generate
from oran_causal.sim.controlled.datasets.generate import dataset_hash

type Spec = dict[str, Any]
type Record = dict[str, Any]

GL_NODES = 64


@contextlib.contextmanager
def patched(module: ModuleType, **attrs: Any) -> Generator[None]:
    """`module` attributes replaced inside the block and restored after, also on error."""
    old = {name: getattr(module, name) for name in attrs}
    for name, value in attrs.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(module, name, value)


def dither_constants(delta: float, n_blocks: int) -> contextlib.AbstractContextManager[None]:
    """The generator's dither delta and block count replaced inside the block."""
    return patched(generate, DITHER_DELTA=float(delta), DITHER_BLOCKS=int(n_blocks))


def design_dataset(design: Mapping[str, Any]) -> Callable[[Unit], tuple[Dataset, Truth]]:
    """A campaign dataset source generating under the X2 dither `design` (only R2 reads it).

    Raises:
        ValueError: At call time, if an R2 dataset has fewer rows than blocks.
    """
    delta, n_blocks = extras.check_design(dict(design))

    def source(u: Unit) -> tuple[Dataset, Truth]:
        if u.regime == "R2" and u.n < n_blocks:
            raise ValueError(f"n {u.n} < n_blocks {n_blocks}")
        with dither_constants(delta, n_blocks):
            return generated_dataset(u)

    return source


def shift_rows(frac: float, n: int, n_blocks: int) -> int:
    """Rows of a switch-time shift of `frac` of a block (L = n / n_blocks rows), half up."""
    return math.floor(float(frac) * int(n) / int(n_blocks) + 0.5)


def _told_dither(ds: Dataset, d: Design, j: int, key: str, factor: float) -> Design:
    if key == "width":
        dist = dict(d.dist or {})
        dist.update(lo=dist["lo"] * factor, hi=dist["hi"] * factor)
        return dataclasses.replace(d, dist=dist)
    s = shift_rows(factor, ds.n, ds.meta["dither"]["n_blocks"])
    fixed = np.asarray(d.fixed_part, float)
    told = np.concatenate([np.repeat(fixed[:1], s), fixed[: len(fixed) - s]]) if s else fixed.copy()
    random = np.where(
        told == fixed, np.asarray(d.random_part, float), np.asarray(ds.X_action, float)[:, j] - told
    )
    return dataclasses.replace(d, fixed_part=told, random_part=random)


def _told_logged(ds: Dataset, d: Design, factor: float) -> Design:
    lam = float(ds.meta["lam"]) * factor
    table = e4_logged.policy_probs(np.asarray(ds.context, float)[:, 0], lam)
    return dataclasses.replace(d, dist={**(d.dist or {}), "lam": lam}, propensity=table)


def tell(ds: Dataset, told: Mapping[str, Any]) -> Dataset:
    """`ds` with the design told to the method changed by `told`; data and truth unchanged.

    Raises:
        ValueError: The variant changes no design of `ds`.
    """
    key, factor = check_told(told)
    designs = list(ds.designs)
    changed = 0
    for j, d in enumerate(designs):
        if key in ("width", "shift") and d.kind == "dither":
            designs[j] = _told_dither(ds, d, j, key, factor)
        elif key == "lam" and d.kind == "logged":
            designs[j] = _told_logged(ds, d, factor)
        else:
            continue
        changed += 1
    if not changed:
        raise ValueError(f"told {dict(told)} changes no design of {ds.world} {ds.regime}")
    return dataclasses.replace(ds, designs=tuple(designs))


def _finite(x: Any) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(x, float).ravel()]


def gl_moments(
    grid_law: nonlinear.Law, law_v: nonlinear.Law, profile: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance per row and target of a profile under a uniform centred law.

    Args:
        grid_law: The law whose grid the profile `[n, G, k]` is on.
        law_v: The uniform law the moments are taken under (Gauss-Legendre, `GL_NODES` nodes).
        profile: The uncentred profile.

    Raises:
        ValueError: A law is not uniform.
    """
    if law_v.dist != "uniform" or grid_law.dist != "uniform":
        raise ValueError(f"X6 needs uniform dither laws, got {grid_law.dist} / {law_v.dist}")
    xi, w = np.polynomial.legendre.leggauss(GL_NODES)
    centred = np.asarray(law_v.ghi, float)[None, :] * xi[:, None]
    wq = w / 2.0
    n, _, k = profile.shape
    mean, var = np.zeros((n, k)), np.zeros((n, k))
    for kk in range(k):
        h = nonlinear.interp(grid_law, np.ascontiguousarray(profile[:, :, kk]), centred)
        mean[:, kk] = wq @ h
        var[:, kk] = np.maximum(wq @ (h * h) - mean[:, kk] ** 2, 0.0)
    return mean, var


@dataclasses.dataclass
class X6Unit:
    """Per-unit context of an X6 gbm unit.

    Attributes:
        centre: Law the profile is centred under, `"told"` or `"true"`.
        redraw: Law the redraws come from, `"told"` or `"true"`.
        true_of: The true design by id of the told design.
        ai_of: Action index by id of the told design.
        laws: `(action index, told law, true law)` by id of the told law.
        shift: Told-minus-true centring shift by id of the told law.
        bias: The bias log by action index.
    """

    centre: str
    redraw: str
    true_of: dict[int, Design]
    ai_of: dict[int, int]
    laws: dict[int, tuple[int, nonlinear.Law, nonlinear.Law]] = dataclasses.field(
        default_factory=dict
    )
    shift: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    bias: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class X6Session:
    """The X6 unit currently running (None outside one); shared by the hooks and the executor."""

    unit: X6Unit | None = None


def _bias_log(shift: np.ndarray, w: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
    post = np.any(w != 0, axis=1)
    corr = []
    for kk in range(w.shape[1]):
        a, b = shift[post, kk], w[post, kk]
        ok = post.sum() > 2 and a.std() > 0 and b.std() > 0
        corr.append(float(np.corrcoef(a, b)[0, 1]) if ok else float("nan"))
    zero = np.zeros(w.shape[1])
    return {
        "z_bias": _finite((shift * w).sum(0) / sd),
        "mean_c": _finite(shift[post].mean(0) if post.any() else zero),
        "rms_c": _finite(np.sqrt((shift[post] ** 2).mean(0)) if post.any() else zero),
        "corr_cw": _finite(corr),
    }


@contextlib.contextmanager
def x6_hooks(session: X6Session) -> Generator[None]:
    """`nonlinear.design_law`, `gbm_profiles` and `GbmStat` wrapped (restored after).

    Inside an X6 gbm unit: the redraw comes from the TRUE law when the unit's redraw is "true"
    (v and offset checked equal); the centring (hence h and the weights) uses the TRUE law when
    its centre is "true"; the shift c_t = E_true[f_t] - E_told[f_t] and the standardised bias
    sum_t c_t w_t / sd are logged per action and target. The grid is always the told law's.
    Outside a unit the frozen functions run unchanged.
    """
    orig_law, orig_profiles, orig_stat = (
        nonlinear.design_law,
        nonlinear.gbm_profiles,
        nonlinear.GbmStat,
    )

    def design_law(design: Design, col: np.ndarray, order: np.ndarray, n_quad: int):
        law = orig_law(design, col, order, n_quad)
        unit = session.unit
        if unit is None or isinstance(law, str) or id(design) not in unit.ai_of:
            return law
        true_design = unit.true_of[id(design)]
        true = law if true_design is design else orig_law(true_design, col, order, n_quad)
        if isinstance(true, str) or not (
            np.array_equal(true.v, law.v) and np.array_equal(true.off, law.off)
        ):
            raise ValueError("X6: the true and told laws differ in v / off (only the width may)")
        if unit.redraw == "true":
            law = dataclasses.replace(law, draw=true.draw)
        unit.laws[id(law)] = (unit.ai_of[id(design)], law, true)
        return law

    def gbm_profiles(xfull, y, focal, cfg, bad_t):
        out = orig_profiles(xfull, y, focal, cfg, bad_t)
        unit = session.unit
        if unit is None:
            return out
        burn = min(int(out.info["burn"]), len(y))
        for ai, f in focal.items():
            entry = unit.laws.get(id(f.law))
            if entry is None:
                continue
            _, told_law, true_law = entry
            raw = out.h[ai] + out.mean[ai][:, None, :]
            m_told, _ = gl_moments(told_law, told_law, raw)
            m_true, v_true = gl_moments(told_law, true_law, raw)
            shift = m_true - m_told
            shift[:burn] = 0.0
            unit.shift[id(told_law)] = shift
            if unit.centre == "true":
                m_true[:burn], v_true[:burn] = 0.0, 0.0
                centred = raw - m_true[:, None, :]
                centred[:burn] = 0.0
                out.h[ai], out.mean[ai], out.var[ai] = centred, m_true, v_true
        return out

    class GbmStat(orig_stat):
        def __init__(self, law, h, var, w):
            super().__init__(law, h, var, w)
            unit = session.unit
            if unit is None or id(law) not in unit.shift:
                return
            unit.bias[unit.laws[id(law)][0]] = _bias_log(unit.shift[id(law)], w, self.sd)

    try:
        with patched(nonlinear, design_law=design_law, gbm_profiles=gbm_profiles, GbmStat=GbmStat):
            yield
    finally:
        session.unit = None


def _stamp_told(rec: Record, ds: Dataset, told: Mapping[str, Any]) -> Record:
    rec["told"] = dict(told)
    rec["dataset_sha256_generated"] = dataset_hash(ds)
    return rec


def told_execute(base: Execute, told_of: Mapping[str, Mapping[str, Any]]) -> Execute:
    """An executor running an arm with its told design.

    The record also keeps the hash of the data as generated; untold arms run unchanged.
    """

    def execute(method: Method, job: Job, unit: Unit) -> Record:
        told = told_of.get(unit.arm)
        if not told:
            return base(method, job, unit)
        rec = base(method, dataclasses.replace(job, ds=tell(job.ds, told)), unit)
        return _stamp_told(rec, job.ds, told)

    return execute


def x6_execute(
    base: Execute,
    session: X6Session,
    told_of: Mapping[str, Mapping[str, Any]],
    gbm_arms: set[str],
) -> Execute:
    """The X6 executor.

    The told design as X3 (width); for the gbm arms also the ablation context (centre and redraw,
    default "told" = the frozen told run) and the bias log in the record.
    """

    def execute(method: Method, job: Job, unit: Unit) -> Record:
        told = told_of.get(unit.arm)
        ds = job.ds
        told_ds = tell(ds, told) if told else ds
        if unit.arm in gbm_arms:
            session.unit = X6Unit(
                centre=(told or {}).get("centre", "told"),
                redraw=(told or {}).get("redraw", "told"),
                true_of={id(dt): d for dt, d in zip(told_ds.designs, ds.designs, strict=True)},
                ai_of={id(dt): i for i, dt in enumerate(told_ds.designs)},
            )
        try:
            rec = base(method, dataclasses.replace(job, ds=told_ds), unit)
            unit_ctx = session.unit
        finally:
            session.unit = None
        if told:
            _stamp_told(rec, ds, told)
        if unit_ctx is not None:
            rec["x6"] = {
                "centre": unit_ctx.centre,
                "redraw": unit_ctx.redraw,
                "targets": list(ds.kpi_names),
                "bias": {ds.action_names[ai]: v for ai, v in sorted(unit_ctx.bias.items())},
            }
        return rec

    return execute


def protocol_sha256(path: Path = EXTRAS_PROTOCOL) -> str | None:
    """LF sha256 of the extras protocol where present."""
    return lf_sha256(path) if path.exists() else None


def providers_for(spec: Spec, base: Providers, session: X6Session) -> Providers:
    """`base` with the dataset source and executor an extras spec needs."""
    experiment = spec["extras"]["experiment"]
    told_of = {a: d["told"] for a, d in spec["arms"].items() if d.get("told") is not None}
    gbm_arms = {
        a for a, d in spec["arms"].items() if (d.get("config") or {}).get("statistic") == "gbm"
    }
    execute = plain_execute
    if experiment == "X6":
        execute = x6_execute(execute, session, told_of, gbm_arms)
    elif told_of:
        execute = told_execute(execute, told_of)
    design = spec["extras"].get("design")
    dataset = design_dataset(design) if design else base.dataset
    return dataclasses.replace(base, dataset=dataset, execute=execute)


def run_extras_part(spec: Spec, shard: Shard, out: Path, opts: RunOptions) -> int:
    """Runs one part of an extras spec through the campaign runner.

    Records carry `integrity.extras` (experiment, design, protocol sha); told arms also `told` and
    `dataset_sha256_generated`, X6 gbm units `x6`.

    Returns:
        The number of records written.
    """
    extras.validate(spec)
    session = X6Session()
    providers = providers_for(spec, opts.providers, session)
    stamp = {**(opts.stamp or {}), "extras": extras.stamp(spec, protocol_sha256())}
    run_opts = dataclasses.replace(opts, providers=providers, stamp=stamp)
    with contextlib.ExitStack() as stack:
        if spec["extras"]["experiment"] == "X6":
            stack.enter_context(x6_hooks(session))
        return run_part(spec, shard, out, run_opts)

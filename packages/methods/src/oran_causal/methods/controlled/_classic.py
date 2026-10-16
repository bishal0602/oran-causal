"""Shared plumbing of the classic arms (pc, notears, shap_dag, two_tower, corr, granger, ...).

Every arm reports a sign; unsigned methods use the sign of the partial correlation of
(source, target) given the arm's conditioning set. Families: action-to-KPI candidates
(placebo included), lagged-KPI-to-KPI candidates and, when `Dataset.meta` lists them, the
diagnostic candidates; each family is its own pooled BY family at q = .05 for p-value
methods. Score-only methods declare score > tau, with tau from the tune placebo scores.
Arms: `"eq"` conditions on the shared design covariates of the source, `"native"` on the
R-3 set; `P_placebo_conf` never conditions another source.
"""

import contextlib
import dataclasses
import re
import sys
import time
from collections.abc import Generator, Mapping
from typing import Any, ClassVar

import numpy as np

from oran_causal.core.covariates import CovariateSpec, design_covariates, eq_min_covariates
from oran_causal.core.data import Dataset, EdgeResult, Result
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.core.scoring import placebo_tau as harness_tau
from oran_causal.core.scoring import tau_rule_name

PLACEBO = "P_placebo"
PLACEBO_CONF = "P_placebo_conf"
RNG_TAG = 7803
Q_BY = 0.05
FAMILIES = ("action", "kpi", "diagnostic")
ARMS = ("eq", "native")
NO_INTERFACE = "no conditioning interface"
EXACT_TOL = 1e-10
NOT_TESTABLE = "exact fit: deterministic world under Z"
COLLINEAR = "source collinear with Z"
META_FAMILY = (
    ("diagnostic_candidates", "diagnostic"),
    ("secondary_candidates", "kpi"),
    ("primary_candidates", "action"),
)
_LAG_PATTERNS = (
    r"^(?P<k>.+?)_lag$",
    r"^(?P<k>.+?)_prev$",
    r"^(?P<k>.+?)_t$",
    r"^(?P<k>.+?)\[t\]$",
    r"^(?P<k>.+?)\(t\)$",
    r"^lag_(?P<k>.+)$",
)


@dataclasses.dataclass(frozen=True)
class Columns:
    """Design matrices of one dataset; all-NaN lagged-KPI columns are dropped.

    Attributes:
        actions: Actions at t, shape `[n, p]`.
        lags: Kept lagged KPIs, shape `[n, k']`.
        lag_names: KPI names of the kept lag columns.
        context: Observed context, shape `[n, c]` (c = 0 when absent).
        Y: KPIs at t + 1, shape `[n, k]`.
    """

    actions: np.ndarray
    lags: np.ndarray
    lag_names: tuple[str, ...]
    context: np.ndarray
    Y: np.ndarray


def columns(data: Dataset) -> Columns:
    """Resolves the design matrices of `data`."""
    lag = np.asarray(data.X_kpi_lag, dtype=float)
    keep = (
        [j for j in range(lag.shape[1]) if np.isfinite(lag[:, j]).all()]
        if lag.ndim == 2 and lag.shape[1]
        else []
    )
    lags = lag[:, keep] if keep else np.zeros((data.n, 0))
    if data.context is None:
        ctx = np.zeros((data.n, 0))
    else:
        ctx = np.asarray(data.context, dtype=float).reshape(data.n, -1)
    names = tuple(data.kpi_names[j] for j in keep)
    return Columns(
        np.asarray(data.X_action, dtype=float), lags, names, ctx, np.asarray(data.Y, dtype=float)
    )


@dataclasses.dataclass(frozen=True)
class CondSet:
    """Conditioning set of one candidate source, in the dataset's row order.

    Attributes:
        names: Column names.
        Z: Columns, shape `[n, c]`.
        mask: Usable rows (False where a lagged action is not a true previous step).
    """

    names: tuple[str, ...]
    Z: np.ndarray
    mask: np.ndarray


def is_diagnostic_column(name: str) -> bool:
    """True for a covariate column that carries `P_placebo_conf`."""
    base = name.removesuffix(":missing")
    return base in (f"concurrent:{PLACEBO_CONF}", f"sp:{PLACEBO_CONF}") or base.startswith(
        f"{PLACEBO_CONF}@t-"
    )


def drop_diagnostic(cs: CondSet, data: Dataset) -> CondSet:
    """Removes every `P_placebo_conf` column from a conditioning set."""
    if PLACEBO_CONF not in data.action_names:
        return cs
    keep = [i for i, nm in enumerate(cs.names) if not is_diagnostic_column(nm)]
    return CondSet(tuple(cs.names[i] for i in keep), cs.Z[:, keep], cs.mask)


def _action_cond_set(data: Dataset, src: int, arm: str) -> CondSet:
    if arm == "eq_min":
        names, m, mask = eq_min_covariates(data, src)
    elif arm == "eq":
        names, m, mask = design_covariates(data, CovariateSpec(focal=src))
    else:
        spec = CovariateSpec(setpoints="none", lagged_actions=False, focal=src, concurrent="all")
        names, m, mask = design_covariates(data, spec)
    cs = CondSet(tuple(names), np.asarray(m, float), np.asarray(mask, bool))
    return cs if data.action_names[src] == PLACEBO_CONF else drop_diagnostic(cs, data)


def cond_set(data: Dataset, family: str, src: int, arm: str) -> CondSet:
    """The whole conditioning set of source `src` in `arm`.

    An action source uses the shared covariate builder (`"eq"`: equal-information set,
    `"native"`: R-3 set, `"eq_min"`: R-3 set plus its own setpoint). A lagged-KPI source
    uses the arm's base set without its own lag column plus every action at t.

    Args:
        data: Row corpus.
        family: `"action"` or `"kpi"`.
        src: Action index, or KPI index of a lagged-KPI source.
        arm: `"eq"`, `"native"` or `"eq_min"`.

    Returns:
        The conditioning set.
    """
    if family == "action":
        return _action_cond_set(data, src, arm)
    eq = arm == "eq"
    spec = CovariateSpec(setpoints="all" if eq else "none", lagged_actions=eq)
    names, m, mask = design_covariates(data, spec)
    k = data.kpi_names[src]
    keep = [i for i, nm in enumerate(names) if nm not in (f"lag_kpi:{k}", f"lag_kpi:{k}:missing")]
    a = np.asarray(data.X_action, float)
    cs = CondSet(
        tuple([names[i] for i in keep] + [f"concurrent:{x}" for x in data.action_names]),
        np.column_stack([np.asarray(m, float)[:, keep], a]),
        np.asarray(mask, bool),
    )
    return drop_diagnostic(cs, data)


def source_column(data: Dataset, cols: Columns, family: str, j: int) -> tuple[np.ndarray, int]:
    """(source column, covariate-builder source index) of a resolved candidate source."""
    if family == "action":
        return cols.actions[:, j], j
    return cols.lags[:, j], data.kpi_names.index(cols.lag_names[j])


def source_kind(data: Dataset, source: str) -> tuple[str, int]:
    """`("action", column)` or `("kpi", KPI index)` of a candidate source name.

    Raises:
        KeyError: The name is neither an action nor a (decorated) lagged KPI.
    """
    if source in data.action_names:
        return "action", data.action_names.index(source)
    if source in data.kpi_names:
        return "kpi", data.kpi_names.index(source)
    for pat in _LAG_PATTERNS:
        m = re.match(pat, source)
        if m and m.group("k") in data.kpi_names:
            return "kpi", data.kpi_names.index(m.group("k"))
    raise KeyError(
        f"candidate source {source!r} is neither an action nor a (lagged) KPI of {data.world}"
    )


def target_index(data: Dataset, target: str) -> int:
    """Y column of a candidate target.

    Raises:
        KeyError: The target is not a KPI.
    """
    if target in data.kpi_names:
        return data.kpi_names.index(target)
    raise KeyError(f"candidate target {target!r} is not a KPI of {data.world}")


def resolve(data: Dataset, cols: Columns, source: str) -> tuple[str, int | None]:
    """(family, column of `cols.actions` or `cols.lags`); None when the lag column is all-NaN."""
    kind, i = source_kind(data, source)
    if kind == "action":
        return "action", i
    name = data.kpi_names[i]
    return "kpi", (cols.lag_names.index(name) if name in cols.lag_names else None)


def zscore(a: np.ndarray) -> np.ndarray:
    """Column z-scores; a constant column is only centred."""
    a = np.asarray(a, dtype=float)
    sd = a.std(0)
    return (a - a.mean(0)) / np.where(sd > 0, sd, 1.0)


def _resid(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(len(v)), z]) if z.shape[1] else np.ones((len(v), 1))
    beta, *_ = np.linalg.lstsq(a, v, rcond=None)
    return v - a @ beta


def pcorr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation of x and y given Z (OLS with intercept); NaN if degenerate."""
    rx, ry = _resid(np.asarray(x, float), z), _resid(np.asarray(y, float), z)
    sx, sy = rx.std(), ry.std()
    if sx <= 1e-12 * (1 + np.abs(x).max()) or sy <= 1e-12 * (1 + np.abs(y).max()):
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def sign_pcorr_given_z(
    data: Dataset, cols: Columns, cand: tuple[str, int, int], arm: str = "native"
) -> int:
    """Sign of the partial correlation of (source, target) given the arm's conditioning set.

    Args:
        data: Row corpus.
        cols: Its design matrices.
        cand: (family, resolved source column, target column).
        arm: Conditioning arm.
    """
    family, j, t = cand
    x, src = source_column(data, cols, family, j)
    cs = cond_set(data, family, src, arm)
    m = cs.mask
    r = pcorr(x[m], cols.Y[m, t], cs.Z[m])
    return int(np.sign(r)) if np.isfinite(r) else 0


def exact_fit(y: np.ndarray, z: np.ndarray) -> bool:
    """True if intercept + Z explains y exactly (residual share <= `EXACT_TOL`, or y constant)."""
    y = np.asarray(y, float)
    tss = float(((y - y.mean()) ** 2).sum())
    if tss <= 0:
        return True
    a = np.column_stack([np.ones(len(y)), z]) if z.shape[1] else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    r = y - a @ beta
    return float(r @ r) <= EXACT_TOL * tss


def not_testable_notes(edges: list[str], collinear: list[str] | None = None) -> dict[str, Any]:
    """Notes for exact-fit candidates and sources collinear with their conditioning set."""
    collinear = list(collinear or [])
    allx = list(edges) + collinear
    if not allx:
        return {}
    why = "; ".join(([NOT_TESTABLE] if edges else []) + ([COLLINEAR] if collinear else []))
    out: dict[str, Any] = {
        "not_testable_reason": why,
        "n_not_testable": len(allx),
        "not_testable_edges": allx,
    }
    if collinear:
        out["not_testable_collinear"] = collinear
    return out


def by_declare(pvals: Any, q: float = Q_BY) -> np.ndarray:
    """BY step-up over the finite p-values; a NaN p-value is not declared and not counted in m."""
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    ok = np.isfinite(p)
    if ok.any():
        out[np.nonzero(ok)[0]] = weighted_step_up(p[ok], None, q, reshape=True)
    return out


def rng(data: Dataset, method_idx: int) -> np.random.Generator:
    """Method stream `default_rng([7803, seed, method_idx])`."""
    return np.random.default_rng([RNG_TAG, int(data.seed), int(method_idx)])


def int_seed(data: Dataset, method_idx: int) -> int:
    """Integer seed drawn from the method stream."""
    return int(rng(data, method_idx).integers(0, 2**31 - 1))


@dataclasses.dataclass
class Scored:
    """Raw output of a method for one candidate, before declaration.

    Attributes:
        source: Source name.
        target: Target name.
        family: `"action"`, `"kpi"` or `"diagnostic"`.
        score: Edge score, NaN if not scorable.
        sign: Sign of the edge.
        p: P-value, if the method has one.
        native: The method's own declaration, if it has one.
    """

    source: str
    target: str
    family: str
    score: float
    sign: int
    p: float | None = None
    native: bool | None = None


def assign_families(data: Dataset, scored: list[Scored]) -> list[Scored]:
    """BY families from the `Dataset.meta` candidate lists where present, else the source kind."""
    lists = [(set(map(tuple, data.meta[k])), f) for k, f in META_FAMILY if k in (data.meta or {})]
    if not lists:
        return scored
    out = []
    for s in scored:
        fam = next((f for cands, f in lists if (s.source, s.target) in cands), s.family)
        out.append(dataclasses.replace(s, family=fam))
    return out


def _by_families(scored: list[Scored], q: float) -> list[bool]:
    dec = [False] * len(scored)
    for fam in FAMILIES:
        idx = [i for i, s in enumerate(scored) if s.family == fam]
        if not idx:
            continue
        d = by_declare([np.nan if scored[i].p is None else scored[i].p for i in idx], q)
        for i, v in zip(idx, d, strict=True):
            dec[i] = bool(v)
    return dec


def declare(
    scored: list[Scored], config: Mapping[str, Any], uses_p: bool
) -> tuple[list[bool], dict[str, Any]]:
    """Primary declarations: BY per family for p-value methods, else score > `config["tau"]`.

    Returns:
        (declared, notes); without a tau a score-only method declares nothing.
    """
    notes: dict[str, Any] = {}
    tau = config.get("tau")
    tau_dec = [bool(tau is not None and np.isfinite(s.score) and s.score > tau) for s in scored]
    if not uses_p:
        notes["declare_rule"] = (
            "score > tau (tau = R-29 conformal .05 cutoff of the tune placebo scores)"
            if tau is not None
            else "untuned: no tau in config, nothing declared"
        )
        return tau_dec, notes
    q = config.get("q", Q_BY)
    notes["declare_rule"] = f"pooled BY per family (action | kpi), q = {q}"
    notes["tau_declared_secondary"] = (
        [f"{s.source}->{s.target}" for s, v in zip(scored, tau_dec, strict=True) if v]
        if tau is not None
        else None
    )
    return _by_families(scored, q), notes


def make_result(
    method: Any, scored: list[Scored], config: dict[str, Any], cpu_s: float, notes: dict[str, Any]
) -> Result:
    """The `Result` of a classic arm from its scored candidates."""
    dec, dnotes = declare(scored, config, method.uses_p)
    edges = tuple(
        EdgeResult(s.source, s.target, float(s.score), s.p, int(s.sign), bool(d))
        for s, d in zip(scored, dec, strict=True)
    )
    n = dict(notes)
    n.update(dnotes)
    n["family"] = {f"{s.source}->{s.target}": s.family for s in scored}
    if any(s.native is not None for s in scored):
        n["native_declared"] = [f"{s.source}->{s.target}" for s in scored if s.native]
    n["tau"] = config.get("tau")
    return Result(method.name, method.version, edges, float(cpu_s), dict(config), n)


def placebo_tau(results: list[Result]) -> dict[str, Any]:
    """Conformal .05 cutoff of the placebo scores pooled over tune results, with its rule name."""
    n = sum(1 for r in results for e in r.edges if e.source == PLACEBO and np.isfinite(e.score))
    return {"tau": float(harness_tau(results)), "tau_rule": tau_rule_name(), "n_placebo_scores": n}


@contextlib.contextmanager
def thread_limit(n: int) -> Generator[None]:
    """Pins BLAS / OpenMP and an already imported torch to `n` threads."""
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    torch = sys.modules.get("torch")
    old = torch.get_num_threads() if torch is not None else None
    try:
        if torch is not None:
            torch.set_num_threads(n)
        with threadpool_limits(n):
            yield
    finally:
        if torch is not None and old is not None:
            torch.set_num_threads(old)


class ClassicBase:
    """Method skeleton: `_score(data, config) -> (scored, notes)`; tune and run are shared."""

    name = "base"
    version = "0"
    uses_p = False
    method_idx = 0
    arms: tuple[str, ...] = ("native",)
    defaults: ClassVar[dict[str, Any]] = {}

    def default_config(self) -> dict[str, Any]:
        """The method's default configuration."""
        return dict(self.defaults)

    def tune(
        self, dev: list[Dataset], truth_free: bool = True, config: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Placebo tau on `dev` for `config` over the defaults; tuning is truth-free only.

        Raises:
            ValueError: `truth_free` is False.
        """
        if not truth_free:
            raise ValueError("xm-classic tuning is truth-free only (CONTRACT sec 5)")
        base = self.default_config()
        base.update(config or {})
        res = [self.run(d, base) for d in dev]
        cfg = dict(base)
        cfg.update(placebo_tau(res))
        cfg["n_dev"] = len(dev)
        cfg["dev_seeds"] = [int(d.seed) for d in dev]
        return cfg

    def run(self, data: Dataset, config: Mapping[str, Any] | None = None) -> Result:
        """Scores and declares every candidate of `data`.

        Raises:
            ValueError: The requested arm is unknown.
        """
        cfg = self.default_config()
        cfg.update(config or {})
        req = cfg.get("arm", self.arms[0])
        if req not in ARMS and req not in self.arms:
            raise ValueError(
                f"arm must be one of {sorted(set(ARMS) | set(self.arms))}, got {req!r}"
            )
        cfg["arm"] = req if req in self.arms else "native"
        nthr = int(cfg.get("threads", 1))
        t0 = time.process_time()
        with thread_limit(nthr):
            scored, notes = self._score(data, cfg)
        scored = assign_families(data, scored)
        notes["threads"] = nthr
        notes["arm"] = cfg["arm"]
        if self.arms == ("native",):
            notes["arm_note"] = NO_INTERFACE
        if req != cfg["arm"]:
            notes["arm_requested"] = req
        return make_result(self, scored, cfg, time.process_time() - t0, notes)

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        raise NotImplementedError

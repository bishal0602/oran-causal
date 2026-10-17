"""Shared core of the conditional-independence arms: mscr, pcorr, rcot2.

Each candidate (source s, target y) is tested as s independent of y given Z. Arm `"native"`:
Z is every other observed source column (actions, non-NaN lagged KPIs, context). Arms `"eq"`
and `"eq_min"`: Z is the source's set from `_classic.cond_set`, mapped onto the dataset
columns; rows outside the covariate row mask are dropped and columns constant on the kept
rows are left out. `P_placebo_conf` is scored in its own BY family and never conditions
another source. A candidate whose target is an exact linear fit of (source, Z) is not
tested. Primary declaration is BY at q = .05 per family; the secondary rule is score > tau,
tau the conformal .05 cutoff of the tune placebo scores. Monte Carlo nulls use at most
`B_MAX` draws with Besag-Clifford stopping at `BC_H` exceedances.
"""

import dataclasses
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.data import Dataset, EdgeResult, Result
from oran_causal.core.scoring import TAU_ALPHA, tau_rule_name
from oran_causal.core.scoring import placebo_tau as harness_tau
from oran_causal.methods.controlled._classic import by_declare, cond_set

COVARIATES_HELPER = "oran_causal.core.covariates"
RNG_TAG = 7802
B_MAX = 9999
PLACEBO = "P_placebo"
PLACEBO_CONF = "P_placebo_conf"
Q = 0.05
ARMS = ("eq", "native", "eq_min")
ARM_DEFAULT = "eq"
EXACT_FIT_RR = 1e-8
NOT_TESTABLE = "exact fit: deterministic world under Z"
PER_EDGE_NOTES = ("mc_draws", "signed_pdcor")


def method_seed(data_seed: int, method_key: int) -> int:
    """Deterministic 32-bit seed of one (dataset, arm) pair from the citests RNG tag."""
    ss = np.random.SeedSequence(entropy=[RNG_TAG, int(data_seed), int(method_key)])
    return int(ss.generate_state(1)[0])


def z_cols_of(d: int, i: int, exclude_z: tuple[int, ...]) -> list[int]:
    """Every column of `d` except `i` and, unless `i` is diagnostic, the diagnostic ones."""
    return [c for c in range(d) if c != i and (c not in exclude_z or i in exclude_z)]


@dataclasses.dataclass(frozen=True)
class Prepared:
    """Column layout of one dataset for the CI tests.

    Attributes:
        S: Observed source columns (actions, lagged KPIs, context, then eq covariates), `[n, d]`.
        S_names: Their names; context columns are prefixed `ctx:`.
        Y: Targets, `[n, k]`.
        pairs: Per candidate, (source column in S, target column in Y).
        n_ctx: Number of context columns.
        family: Per candidate, `"action_kpi"`, `"kpi_kpi"` or `"placebo_conf"`.
        exclude_z: Diagnostic columns of S that never condition another source.
        arm: Arm name.
        row_mask: Arms eq and eq_min: kept rows (S and Y are already restricted).
        eq_dropped: Arms eq and eq_min: covariates left out as constant on the kept rows.
        z_map: Arms eq and eq_min: source column to its conditioning columns.
    """

    S: np.ndarray
    S_names: tuple[str, ...]
    Y: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    n_ctx: int
    family: tuple[str, ...] = ()
    exclude_z: tuple[int, ...] = ()
    arm: str = "native"
    row_mask: np.ndarray | None = None
    eq_dropped: tuple[str, ...] = ()
    z_map: dict[int, tuple[int, ...]] | None = None

    def z_cols(self, i: int) -> list[int]:
        """Conditioning columns of source column `i`."""
        if self.z_map is not None:
            return list(self.z_map[i])
        return z_cols_of(self.S.shape[1], i, self.exclude_z)

    def column_groups(self) -> list[tuple[list[int], list[int]]]:
        """(columns, candidate indices) groups; a candidate's Z is its group minus its source."""
        if self.z_map is not None:
            by: dict[tuple[int, ...], list[int]] = {}
            for e, (i, _) in enumerate(self.pairs):
                by.setdefault(tuple(sorted(set(self.z_map[i]) | {i})), []).append(e)
            return [(list(k), es) for k, es in by.items()]
        base = [c for c in range(self.S.shape[1]) if c not in self.exclude_z]
        groups = [(base, [e for e, (i, _) in enumerate(self.pairs) if i not in self.exclude_z])]
        for x in self.exclude_z:
            groups.append(
                (list(range(self.S.shape[1])), [e for e, (i, _) in enumerate(self.pairs) if i == x])
            )
        return [g for g in groups if g[1]]

    def z_of(self, a: np.ndarray, i: int) -> np.ndarray:
        """`a[:, z_cols(i)]` for a matrix laid out like S."""
        return a[:, self.z_cols(i)]


@dataclasses.dataclass
class _Layout:
    S: np.ndarray
    names: list[str]
    Y: np.ndarray
    pairs: list[tuple[int, int]]
    family: tuple[str, ...]
    n_ctx: int
    act_idx: dict[str, int]
    lag_idx: dict[str, int]
    excl: list[int]


def _source_matrix(data: Dataset) -> tuple[np.ndarray, list[str], dict[str, int], int]:
    cols, names = [data.X_action], list(data.action_names)
    lag_idx: dict[str, int] = {}
    if data.X_kpi_lag is not None and data.X_kpi_lag.size:
        keep = [
            j for j in range(data.X_kpi_lag.shape[1]) if not np.isnan(data.X_kpi_lag[:, j]).all()
        ]
        for j in keep:
            lag_idx[data.kpi_names[j]] = len(names) + len(lag_idx)
        cols.append(data.X_kpi_lag[:, keep])
        names += [data.kpi_names[j] for j in keep]
    n_ctx = 0
    if data.context is not None:
        ctx = np.asarray(data.context, dtype=np.float64).reshape(data.n, -1)
        cnames = data.meta.get("context_names", tuple(f"c{i}" for i in range(ctx.shape[1])))
        cols.append(ctx)
        names += [f"ctx:{c}" for c in cnames]
        n_ctx = ctx.shape[1]
    s = np.concatenate([np.asarray(c, dtype=np.float64).reshape(data.n, -1) for c in cols], axis=1)
    return s, names, lag_idx, n_ctx


def _layout(data: Dataset) -> _Layout:
    s, names, lag_idx, n_ctx = _source_matrix(data)
    y = np.asarray(data.Y, dtype=np.float64)
    if not (np.isfinite(s).all() and np.isfinite(y).all()):
        raise ValueError("citests: non-finite values in sources or targets")
    act_idx = {a: i for i, a in enumerate(data.action_names)}
    tgt_idx = {k: j for j, k in enumerate(data.kpi_names)}
    pairs = []
    for src, t in data.candidates:
        if src in act_idx:
            i = act_idx[src]
        elif src in lag_idx:
            i = lag_idx[src]
        else:
            raise ValueError(
                f"citests: candidate source {src!r} is neither an action nor a non-NaN lagged KPI"
            )
        pairs.append((i, tgt_idx[t]))
    fam = tuple(
        "placebo_conf" if src == PLACEBO_CONF else "action_kpi" if src in act_idx else "kpi_kpi"
        for src, _ in data.candidates
    )
    excl = [act_idx[a] for a in data.action_names if a == PLACEBO_CONF]
    return _Layout(s, names, y, pairs, fam, n_ctx, act_idx, lag_idx, excl)


def prepare(data: Dataset, arm: str = "native") -> Prepared:
    """Column layout of `data` for `arm`.

    Raises:
        ValueError: Unknown arm, non-finite data, or a candidate source that cannot be resolved.
    """
    if arm not in ARMS:
        raise ValueError(f"citests: arm must be one of {ARMS}, got {arm!r}")
    lay = _layout(data)
    if arm == "native":
        return Prepared(
            lay.S, tuple(lay.names), lay.Y, tuple(lay.pairs), lay.n_ctx, lay.family, tuple(lay.excl)
        )
    return _prepare_eq(data, lay, arm)


def _is_conf_column(name: str) -> bool:
    base = name.removesuffix(":missing")
    for pre in ("sp:", "concurrent:"):
        base = base.removeprefix(pre)
    return base.split("@t-")[0] == PLACEBO_CONF


class _ColumnMap:
    """Maps covariate columns onto S, appending the ones S does not have."""

    def __init__(self, data: Dataset, lay: _Layout):
        self.data, self.lay = data, lay
        self.col_of = {f"concurrent:{a}": i for a, i in lay.act_idx.items()}
        self.col_of.update({f"lag_kpi:{k}": i for k, i in lay.lag_idx.items()})
        self.col_of.update({nm: c for c, nm in enumerate(lay.names) if nm.startswith("ctx:")})
        self.extra: list[np.ndarray] = []
        self.mask: np.ndarray | None = None

    def _check_mask(self, m: np.ndarray) -> None:
        if self.mask is None:
            self.mask = m
        elif not np.array_equal(self.mask, m):
            raise ValueError("citests: design_covariates row_mask depends on the focal action")

    def _column(self, nm: str, v: np.ndarray) -> int:
        d = self.lay.S.shape[1]
        if nm in self.col_of:
            c = self.col_of[nm]
            ref = self.lay.S[:, c] if c < d else self.extra[c - d]
            if not np.array_equal(v, ref):
                raise ValueError(f"citests: helper column {nm!r} differs from the dataset column")
            return c
        self.col_of[nm] = c = d + len(self.extra)
        self.extra.append(v)
        self.lay.names.append(nm)
        return c

    def mapped(self, cn: tuple[str, ...], z: np.ndarray, m: np.ndarray) -> list[int]:
        """S columns of one conditioning set."""
        z = np.asarray(z, dtype=np.float64).reshape(self.data.n, -1)
        m = np.asarray(m, dtype=bool)
        if len(cn) != z.shape[1] or m.shape != (self.data.n,):
            raise ValueError("citests: design_covariates returned inconsistent shapes")
        self._check_mask(m)
        return [self._column(nm, z[:, k]) for k, nm in enumerate(cn)]


def _prepare_eq(data: Dataset, lay: _Layout, arm: str) -> Prepared:
    cmap = _ColumnMap(data, lay)
    inv_act = {i: a for a, i in lay.act_idx.items()}
    inv_lag = {i: k for k, i in lay.lag_idx.items()}
    z_map: dict[int, tuple[list[int], bool]] = {}
    for i in sorted({i for i, _ in lay.pairs}):
        if i in inv_act:
            cs = cond_set(data, "action", data.action_names.index(inv_act[i]), arm)
            diag = inv_act[i] == PLACEBO_CONF
        else:
            cs = cond_set(data, "kpi", data.kpi_names.index(inv_lag[i]), arm)
            diag = False
        z_map[i] = (cmap.mapped(cs.names, cs.Z, cs.mask), diag)
    conf_cols = set(lay.excl) | {c for nm, c in cmap.col_of.items() if _is_conf_column(nm)}
    zl = {
        i: [c for c in dict.fromkeys(z) if c != i and (diag or c not in conf_cols)]
        for i, (z, diag) in z_map.items()
    }
    mask = cmap.mask
    assert mask is not None
    s = np.concatenate([lay.S] + [v[:, None] for v in cmap.extra], axis=1)[mask]
    if not np.isfinite(s).all():
        raise ValueError("citests: non-finite design covariates on kept rows")
    const = {c for c in range(s.shape[1]) if np.ptp(s[:, c]) == 0.0}
    dropped = sorted({lay.names[c] for z in zl.values() for c in z if c in const})
    zm = {i: tuple(c for c in z if c not in const) for i, z in zl.items()}
    return Prepared(
        s,
        tuple(lay.names),
        lay.Y[mask],
        tuple(lay.pairs),
        lay.n_ctx,
        lay.family,
        tuple(sorted(conf_cols)),
        arm,
        mask,
        tuple(dropped),
        zm,
    )


def standardize(a: np.ndarray) -> np.ndarray:
    """Column z-scores (population std); a zero-variance column is only centred."""
    sd = a.std(axis=0)
    return (a - a.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def _by_source(pairs: tuple[tuple[int, int], ...]) -> dict[int, list[int]]:
    by_i: dict[int, list[int]] = {}
    for e, (i, _) in enumerate(pairs):
        by_i.setdefault(i, []).append(e)
    return by_i


def pcorr_given_z(
    s: np.ndarray,
    y: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    exclude_z: tuple[int, ...] = (),
    z_cols: Callable[[int], list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial correlation of `s[:, i]` and `y[:, j]` given Z plus an intercept.

    Z is `z_cols(i)` when given, else every other column of `s` outside `exclude_z`. A
    vanishing residual gives rho = 0.

    Returns:
        (rho, df) per pair, df = n - 2 - |Z|.
    """
    n, d = s.shape
    rho = np.zeros(len(pairs))
    df = np.zeros(len(pairs))
    for i, es in _by_source(pairs).items():
        zc = z_cols(i) if z_cols is not None else z_cols_of(d, i, exclude_z)
        others = np.concatenate([s[:, zc], np.ones((n, 1))], axis=1)
        js = sorted({pairs[e][1] for e in es})
        rhs = np.concatenate([s[:, [i]], y[:, js]], axis=1)
        sol, *_ = np.linalg.lstsq(others, rhs, rcond=None)
        res = rhs - others @ sol
        rx = res[:, 0]
        for e in es:
            df[e] = n - 2 - len(zc)
            ry = res[:, 1 + js.index(pairs[e][1])]
            den = np.sqrt(rx @ rx) * np.sqrt(ry @ ry)
            vx, vy = rx @ rx / n, ry @ ry / n
            rho[e] = 0.0 if (den <= 0 or vx < 1e-12 or vy < 1e-12) else float(rx @ ry / den)
    return rho, df


def exact_fit(prep: Prepared) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate: residual share rr of the target on [1, source, Z], and rr < `EXACT_FIT_RR`."""
    s, y = standardize(prep.S), standardize(prep.Y)
    n = s.shape[0]
    rr = np.ones(len(prep.pairs))
    for i, es in _by_source(prep.pairs).items():
        x = np.concatenate([np.ones((n, 1)), s[:, [i]], s[:, prep.z_cols(i)]], axis=1)
        js = sorted({prep.pairs[e][1] for e in es})
        sol, *_ = np.linalg.lstsq(x, y[:, js], rcond=None)
        res = y[:, js] - x @ sol
        tot = ((y[:, js] - y[:, js].mean(axis=0)) ** 2).sum(axis=0)
        for e in es:
            k = js.index(prep.pairs[e][1])
            rr[e] = float(res[:, k] @ res[:, k] / tot[k]) if tot[k] > 0 else 0.0
    return rr, rr < EXACT_FIT_RR


def by_families(p: np.ndarray, family: tuple[str, ...], q: float = Q) -> np.ndarray:
    """Pooled BY within each family; a NaN p-value is never declared."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    fam = np.asarray(family)
    for f in set(family):
        idx = np.nonzero(fam == f)[0]
        out[idx] = by_declare(p[idx], q)
    return out


def tau_from_results(results: list[Result]) -> dict[str, Any]:
    """Conformal tau of one cell's tune results with its rule name and placebo count."""
    n = sum(1 for r in results for e in r.edges if e.source == PLACEBO)
    return {
        "tau": float(harness_tau(results, alpha=TAU_ALPHA)),
        "n_placebo_scores": n,
        "tau_rule": tau_rule_name(),
    }


def progress(cfg: Mapping[str, Any], done: int, total: int) -> None:
    """Appends "done total process_cpu_s" to `cfg["progress_file"]` when set."""
    path = cfg.get("progress_file")
    if path:
        with Path(path).open("a") as f:
            f.write(f"{done} {total} {time.process_time():.3f}\n")


def cell_key(d: Dataset) -> tuple[str, str, int, Any]:
    """(world, regime, n, lambda) of a dataset."""
    return (d.world, d.regime, d.n, d.meta.get("lam"))


@dataclasses.dataclass
class _Outcome:
    score: np.ndarray
    p: np.ndarray | None
    sign: np.ndarray
    declared: np.ndarray


def _run_test(
    method: CITestBase, sub: Prepared, data: Dataset, cfg: dict[str, Any], m: int
) -> dict[str, Any]:
    if not m:
        return {"score": np.zeros(0), "p": np.zeros(0)}
    try:
        return method._test(sub, data, cfg)
    except ValueError as e:
        return {
            "score": np.full(m, np.nan),
            "p": np.full(m, np.nan),
            "notes": {"stop": str(e)},
            "not_testable": dict.fromkeys(range(m), f"STOP: {e}"),
        }


def _not_testable_notes(data: Dataset, nt: dict[int, str]) -> dict[str, Any]:
    if not nt:
        return {}
    return {
        "not_testable": {"->".join(data.candidates[e]): why for e, why in sorted(nt.items())},
        "not_testable_edges": [list(data.candidates[e]) for e in sorted(nt)],
    }


def _signs(method: CITestBase, sub: Prepared, out: dict[str, Any]) -> tuple[np.ndarray, str]:
    if method.native_sign and out.get("sign") is not None:
        return np.sign(np.asarray(out["sign"], dtype=float)).astype(int), "native"
    rho, _ = pcorr_given_z(standardize(sub.S), standardize(sub.Y), sub.pairs, z_cols=sub.z_cols)
    return np.sign(rho).astype(int), "pcorr_given_Z"


def _per_edge(v: Any, keep: np.ndarray, m_all: int) -> list[Any]:
    full: list[Any] = [None] * m_all
    for e, x in zip(keep, v, strict=True):
        full[e] = x
    return full


class CITestBase:
    """Method skeleton of the CI arms.

    Subclasses implement `_test(prep, data, cfg) -> dict` with arrays aligned with
    `prep.pairs`: `score`, `p` (or None) and optionally `sign`, `declared_native`,
    `not_testable` ({local index: reason}) and `notes`. A ValueError from `_test` is a
    recorded STOP: every candidate it was given is not testable with that reason.
    """

    name = "citest"
    version = "0"
    method_key = 0
    native_sign = False

    def default_config(self) -> dict[str, Any]:
        """The arm's default configuration."""
        return {}

    def native_config(self) -> dict[str, Any]:
        """The default configuration in arm `"native"`."""
        return {**self.default_config(), "arm": "native"}

    def tune(
        self,
        dev: list[Dataset],
        truth_free: bool = True,
        base_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tau of one tune cell for `base_config` (default: the primary config).

        Raises:
            ValueError: `truth_free` is False or `dev` spans more than one cell.
        """
        if not truth_free:
            raise ValueError("citests: tuning is truth-free only (CONTRACT sec 5)")
        if len({cell_key(d) for d in dev}) != 1:
            raise ValueError(
                "citests: tune() takes the DEV datasets of exactly one (world, regime, n) cell"
            )
        cfg = {"arm": ARM_DEFAULT, **self.default_config(), **(base_config or {})}
        res = [self.run(d, cfg) for d in dev]
        out = dict(cfg)
        out.update(tau_from_results(res))
        out["tau_cell"] = list(cell_key(dev[0]))
        out["tau_dev_seeds"] = [d.seed for d in dev]
        return out

    def _test(self, prep: Prepared, data: Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _notes(self, prep: Prepared, data: Dataset) -> dict[str, Any]:
        notes: dict[str, Any] = {
            "arm": prep.arm,
            "conditioning": list(prep.S_names),
            "q": Q,
            "family": "pooled BY per family: action_kpi (primary), kpi_kpi (secondary), "
            "placebo_conf (R-10 diagnostic)",
            "family_of_edge": prep.family,
        }
        if prep.arm in ("eq", "eq_min"):
            notes.update(
                covariates_helper=COVARIATES_HELPER,
                eq_dropped=list(prep.eq_dropped),
                n_rows_used=int(prep.S.shape[0]),
                n_rows_dropped=int(data.n - prep.S.shape[0]),
            )
        return notes

    def _outcome(
        self, prep: Prepared, sub: Prepared, out: dict[str, Any], keep: np.ndarray, dead: np.ndarray
    ) -> tuple[_Outcome, str]:
        m_all = len(prep.pairs)
        score = np.full(m_all, np.nan)
        score[keep] = np.asarray(out["score"], dtype=float)
        score[dead] = np.nan
        p = None
        if out.get("p") is not None:
            p = np.full(m_all, np.nan)
            p[keep] = np.asarray(out["p"], dtype=float)
            p[dead] = np.nan
        sign = np.zeros(m_all, dtype=int)
        sign[keep], rule = _signs(self, sub, out)
        sign[dead] = 0
        declared = by_families(p, prep.family) if p is not None else np.zeros(m_all, dtype=bool)
        declared[dead] = False
        return _Outcome(score, p, sign, declared), rule

    def run(self, data: Dataset, config: Mapping[str, Any] | None = None) -> Result:
        """Tests and declares every candidate of `data`."""
        cfg = {"arm": ARM_DEFAULT, **self.default_config(), **(config or {})}
        t0 = time.process_time()
        prep = prepare(data, cfg["arm"])
        notes = self._notes(prep, data)
        m_all = len(prep.pairs)
        rr, degenerate = exact_fit(prep)
        keep = np.nonzero(~degenerate)[0]
        nt: dict[int, str] = {int(e): NOT_TESTABLE for e in np.nonzero(degenerate)[0]}
        if degenerate.any():
            notes["exact_fit_rr_max"] = float(rr[degenerate].max())
        notes["exact_fit_rr_min_testable"] = float(rr[keep].min()) if len(keep) else None
        sub = dataclasses.replace(
            prep,
            pairs=tuple(prep.pairs[e] for e in keep),
            family=tuple(prep.family[e] for e in keep),
        )
        out = _run_test(self, sub, data, cfg, len(keep))
        for k, why in (out.get("not_testable") or {}).items():
            nt[int(keep[int(k)])] = str(why)
        dead = np.zeros(m_all, dtype=bool)
        dead[list(nt)] = True
        notes.update(_not_testable_notes(data, nt))
        oc, notes["sign_rule"] = self._outcome(prep, sub, out, keep, dead)
        declared = oc.declared
        tau = cfg.get("tau")
        if tau is not None:
            notes["declared_tau"] = tuple(bool(np.isfinite(s) and s > tau) for s in oc.score)
            if oc.p is None:
                declared = np.array(notes["declared_tau"], dtype=bool)
        if out.get("declared_native") is not None:
            dn = np.zeros(m_all, dtype=bool)
            dn[keep] = np.asarray(out["declared_native"], dtype=bool)
            notes["declared_native"] = tuple(bool(v) for v in dn)
        notes["family_of_edge"] = prep.family
        for k, v in out.get("notes", {}).items():
            notes[k] = _per_edge(v, keep, m_all) if k in PER_EDGE_NOTES and len(keep) < m_all else v
        edges = tuple(
            EdgeResult(
                s,
                t,
                float(oc.score[e]),
                None if oc.p is None or not np.isfinite(oc.p[e]) else float(oc.p[e]),
                int(oc.sign[e]),
                bool(declared[e]),
            )
            for e, (s, t) in enumerate(data.candidates)
        )
        return Result(
            self.name, self.version, edges, float(time.process_time() - t0), dict(cfg), notes
        )

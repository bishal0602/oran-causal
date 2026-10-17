"""MSCR-CRT v2: design-centred, outcome-residualised linear CRT on the system unit table.

The App C tables cite it as "v2+by". For hypothesis (f, rel, kpi) over family f's units:

- dose: `LEVEL_V2` (accept, half, accept+rb 1; reject 0), the treatment "applied vs rejected";
- regressor: v_u = sgn_u (LEVEL_V2[mode_u] - E_pi0u[LEVEL_V2]) over the unit's logged table;
- outcome: y(rel, kpi) residualised by OLS on episode x sgn fixed effects plus the same target's
  pre-window value (one pooled slope; fixed effects only when the pre column is degenerate);
- statistic: S = sum_u v_u r_u, T = |S|; the null redraws the family's modes from the logged
  tables (every other family held fixed), p = (1 + #{T_b >= T_obs - tol}) / (B + 1).

RNG: `default_rng([seed, 6616, 2, family_idx, lag, split])`, consumed in chunks of one stream.
Support (>= 30 units, >= 5 accepted, >= 5 rejected, >= 3 episodes), statuses and BY at q over
the tested hypotheses of one call follow MSCR-CRT v1.
"""

import dataclasses
import time
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import binom

from oran_causal.core.data import UnitTable
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.sim.system.decision.vocab import FAMILIES, KPIS, LEVEL_ARR, MODES

__all__ = [
    "CLAIMS_V2",
    "CRT_TAG",
    "CRT_UNITS_V2_VERSION",
    "KEY0",
    "KPIS",
    "LEVEL_V2",
    "LEVEL_V2_ARR",
    "SPLIT_PLACEBO",
    "SPLIT_POOLED",
    "V2_STREAM",
    "CrtKey",
    "PiAssignment",
    "UnitCRTConfigV2",
    "by_status",
    "carryover_audit_v2",
    "crt_unit_test_v2",
    "degenerate_columns",
    "design_mean",
    "design_regressor",
    "edges_from_crt_v2",
    "family_tests_v2",
    "placebo_rejection_units_v2",
    "power_floor",
    "residualise",
    "run_crt_units_v2",
    "strata_of",
]

CRT_UNITS_V2_VERSION = "mscr-crt-units-v2"
CRT_TAG = 6616
V2_STREAM = 2
SPLIT_POOLED, SPLIT_PLACEBO = 0, 9
LEVEL_V2 = {"accept": 1.0, "half": 1.0, "reject": 0.0, "accept+rb": 1.0}
LEVEL_V2_ARR = np.array([LEVEL_V2[m] for m in MODES])
CLAIMS_V2 = {
    "declared": "sharp null of no ASSIGNED effect of the family's decisions on this KPI window "
    "rejected (design-centred residualised CRT, mscr-crt-units-v2); BY across the tested "
    "hypotheses of this call",
    "not_detected": "not detected at this support: this is NOT evidence that the edge is absent",
    "undetermined": "undetermined ({reason}): the edge is unknown, neither present nor absent",
}

Target = tuple[str, tuple[str, ...]]


@dataclasses.dataclass(frozen=True)
class UnitCRTConfigV2:
    """Settings of the v2 CRT.

    Attributes:
        B: Null draws per family.
        alpha: Per-hypothesis level of the descriptive rates.
        q: BY level.
        chunk: Maximum draws per pass.
        max_chunk_bytes: Cap on one (chunk x n) float64 block.
        audit_B: Draws of the lag-1 carryover audit.
        min_units: Support: minimum units of the family.
        min_accept: Support: minimum accepted units.
        min_reject: Support: minimum rejected units.
        min_episodes: Support: minimum episodes.
        seed: RNG seed.
    """

    B: int = 9999
    alpha: float = 0.05
    q: float = 0.05
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    audit_B: int = 999
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3
    seed: int = 0


class CrtKey(NamedTuple):
    """RNG stream of one family test: carryover lag (0 or 1) and split id."""

    lag: int = 0
    split: int = SPLIT_POOLED


KEY0 = CrtKey()


class PiAssignment:
    """Logged pi0 assignment model of a unit table (context-free, independent per unit)."""

    def __init__(self, data: UnitTable) -> None:
        self.data = data

    def mode_draws(self, rng: np.random.Generator, rows: npt.ArrayLike, B: int) -> np.ndarray:
        """Returns (B, len(rows)) mode indices redrawn from each row's logged table."""
        rows = np.asarray(rows, int)
        cum = np.cumsum(self.data.probs[rows], axis=1)
        cum[:, -1] = np.inf
        U = rng.random((int(B), len(rows)))
        out = np.zeros(U.shape, np.int64)
        for m in range(cum.shape[1] - 1):
            out += cum[None, :, m] <= U
        return out

    def conditional_draws(
        self,
        rng: np.random.Generator,
        family: str,
        B: int,
        rows: npt.ArrayLike | None = None,
        what: str = "x",
    ) -> np.ndarray:
        """Redraws one family's modes, every other family held fixed.

        Args:
            rng: Generator of the family's stream.
            family: Knob family.
            B: Number of draws.
            rows: Rows of the family (default: all of them, in table order).
            what: "x" (dose times sgn), "level" or "mode".

        Returns:
            Array of shape (B, len(rows)).
        """
        d = self.data
        fi = FAMILIES.index(family)
        rows = d.rows_of(family) if rows is None else np.asarray(rows, int)
        if len(rows) and not np.all(d.family[rows] == fi):
            raise ValueError("rows outside the family")
        M = self.mode_draws(rng, rows, B)
        if what == "mode":
            return M
        L = LEVEL_ARR[M]
        return L if what == "level" else L * d.sgn[rows][None, :]


def power_floor(B: int, m: int, q: float = 0.05) -> dict[str, Any]:
    """BY feasibility of the Monte-Carlo p floor 1 / (B + 1) over m hypotheses."""
    c_m = sum(1.0 / k for k in range(1, max(int(m), 1) + 1))
    thr1 = q / (max(int(m), 1) * c_m)
    floor = 1.0 / (int(B) + 1.0)
    return {
        "B": int(B),
        "m": int(m),
        "p_floor": floor,
        "by_rank1_threshold": thr1,
        "min_hypotheses_at_floor_to_declare": int(np.ceil(floor / thr1 - 1e-12)),
        "rank1_ok": floor <= thr1,
        "min_B_rank1": int(np.ceil(1.0 / thr1 - 1.0)),
    }


def degenerate_columns(cols: dict[str, Any], min_minor_frac: float = 0.01) -> dict[str, str]:
    """Returns {column: reason} for all-NaN, constant or near-constant columns."""
    bad = {}
    for c, col in cols.items():
        v = np.asarray(col, float)
        f = v[np.isfinite(v)]
        if len(f) == 0:
            bad[c] = "all_nan"
            continue
        if float(np.ptp(f)) <= 1e-9 * (1.0 + abs(float(np.mean(f)))):
            bad[c] = "constant"
            continue
        _, counts = np.unique(f, return_counts=True)
        if counts.max() > (1.0 - min_minor_frac) * len(f):
            bad[c] = f"near_constant(modal share {counts.max() / len(f):.3f})"
    return bad


def design_mean(data: UnitTable, rows: npt.ArrayLike) -> np.ndarray:
    """Returns E_pi0u[LEVEL_V2] per row over the logged table."""
    return data.probs[np.asarray(rows, int)] @ LEVEL_V2_ARR


def design_regressor(data: UnitTable, rows: npt.ArrayLike) -> np.ndarray:
    """Returns v_u = sgn_u (LEVEL_V2[mode_u] - E_pi0u[LEVEL_V2]) of the logged modes."""
    rows = np.asarray(rows, int)
    return data.sgn[rows] * (LEVEL_V2_ARR[data.mode[rows]] - design_mean(data, rows))


def strata_of(data: UnitTable, rows: npt.ArrayLike) -> np.ndarray:
    """Returns the episode x sgn stratum id per row."""
    rows = np.asarray(rows, int)
    return data.episode[rows].astype(np.int64) * 2 + (data.sgn[rows] > 0)


def _demean(a: np.ndarray, inv: np.ndarray, k: int) -> np.ndarray:
    s = np.bincount(inv, a, k)
    n = np.bincount(inv, minlength=k).astype(float)
    return a - (s / np.maximum(n, 1.0))[inv]


def residualise(
    y: npt.ArrayLike, pre: npt.ArrayLike | None, strata: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """OLS residual of y on stratum fixed effects plus pre (Frisch-Waugh).

    Args:
        y: Outcome per row.
        pre: Pre-window value per row; None, non-finite or constant within strata gives FE only.
        strata: Stratum id per row.

    Returns:
        The residual and {"resid", "pre_slope", "n_strata"}.
    """
    y = np.asarray(y, float)
    _, inv = np.unique(strata, return_inverse=True)
    k = int(inv.max()) + 1 if len(inv) else 0
    yd = _demean(y, inv, k)
    info = {"resid": "fe", "pre_slope": float("nan"), "n_strata": k}
    if pre is not None:
        pre = np.asarray(pre, float)
        if np.all(np.isfinite(pre)):
            pd = _demean(pre, inv, k)
            ss = float(pd @ pd)
            scale = float(np.sum((pre - pre.mean()) ** 2)) if len(pre) else 0.0
            if ss > 1e-10 * max(scale, 1e-300) and ss > 0:
                b = float(pd @ yd) / ss
                return yd - b * pd, dict(info, resid="fe+pre", pre_slope=b)
    return yd, info


def _chunk(cfg: UnitCRTConfigV2, n: int) -> int:
    return int(max(1, min(int(cfg.chunk), int(cfg.max_chunk_bytes) // (8 * max(int(n), 1)))))


def _stream(cfg: UnitCRTConfigV2, fi: int, key: CrtKey) -> np.random.Generator:
    return np.random.default_rng(
        [int(cfg.seed), CRT_TAG, V2_STREAM, int(fi), int(key.lag), int(key.split)]
    )


def _family_rows(data: UnitTable, family: str, lag: int) -> tuple[np.ndarray, np.ndarray]:
    rows = data.rows_of(family)
    if lag == 1:
        rows = rows[data.prev[rows] >= 0]
    elif lag != 0:
        raise ValueError("lag must be 0 or 1")
    return rows, rows if lag == 0 else data.prev[rows]


def _base(
    data: UnitTable,
    family: str,
    sel: tuple[np.ndarray, np.ndarray],
    cfg: UnitCRTConfigV2,
    key: CrtKey,
) -> dict[str, Any]:
    rows, src = sel
    lv = LEVEL_ARR[data.mode[src]]
    return {
        "version": CRT_UNITS_V2_VERSION,
        "family": family,
        "lag": key.lag,
        "split": key.split,
        "n": len(rows),
        "n_accept": int((lv == 1.0).sum()),
        "n_reject": int((lv == 0.0).sum()),
        "n_half": int((lv == 0.5).sum()),
        "n_episodes": len(np.unique(data.episode[src])),
        "B": cfg.B,
        "p": np.nan,
        "t_obs": np.nan,
        "s_obs": np.nan,
        "beta": np.nan,
        "z_approx": np.nan,
        "sign": 0,
        "resid": None,
        "pre_slope": np.nan,
        "status": "undetermined",
        "reason": "",
    }


def _supported(base: dict[str, Any], cfg: UnitCRTConfigV2) -> bool:
    return not (
        base["n"] < cfg.min_units
        or base["n_accept"] < cfg.min_accept
        or base["n_reject"] < cfg.min_reject
        or base["n_episodes"] < cfg.min_episodes
    )


def _undetermined(base: dict[str, Any], name: str, key: Sequence[str], reason: str) -> dict:
    return dict(
        base,
        target=name,
        key=list(key),
        reason=reason,
        claim=CLAIMS_V2["undetermined"].format(reason=reason),
    )


def _residual_targets(
    data: UnitTable, rows: np.ndarray, targets: Sequence[Target], base: dict[str, Any]
) -> tuple[list[dict | None], list[np.ndarray], list[int], list[dict]]:
    strata = strata_of(data, rows)
    out: list[dict | None] = [None] * len(targets)
    cols, idx, infos = [], [], []
    for t, (nm, key) in enumerate(targets):
        rk = (key[-2], key[-1])
        if key[0] == "pre":
            y, pre = data.pre[rk][rows], None
        else:
            y, pre = data.y[rk][rows], data.pre[rk][rows]
        if not np.all(np.isfinite(y)) or degenerate_columns({"y": y}):
            out[t] = _undetermined(base, nm, key, "degenerate_outcome")
            continue
        r, info = residualise(y, pre, strata)
        cols.append(r)
        idx.append(t)
        infos.append(info)
    return out, cols, idx, infos


class _Stream(NamedTuple):
    cfg: UnitCRTConfigV2
    family: str
    key: CrtKey


def _null_moments(
    data: UnitTable, src: np.ndarray, R: np.ndarray, thr: np.ndarray, stream: _Stream
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cfg, family, key = stream
    uniq, inv = np.unique(src, return_inverse=True)
    sg = data.sgn[src]
    mu = design_mean(data, src)
    pa = PiAssignment(data)
    rng = _stream(cfg, FAMILIES.index(family), key)
    ch = _chunk(cfg, len(uniq))
    m = R.shape[1]
    count = np.zeros(m, np.int64)
    s1 = np.zeros(m)
    s2 = np.zeros(m)
    done = 0
    while done < cfg.B:
        b = min(ch, cfg.B - done)
        M = pa.conditional_draws(rng, family, b, rows=uniq, what="mode")
        V = LEVEL_V2_ARR[M]
        del M
        if len(uniq) != len(src) or np.any(uniq != src):
            V = V[:, inv]
        V -= mu[None, :]
        V *= sg[None, :]
        S = V @ R
        del V
        count += np.count_nonzero(np.abs(S) >= thr[None, :], axis=0)
        s1 += S.sum(0)
        s2 += (S * S).sum(0)
        done += b
    return count, s1, s2, ch


def family_tests_v2(
    data: UnitTable,
    family: str,
    targets: Sequence[Target],
    cfg: UnitCRTConfigV2 | None = None,
    key: CrtKey = KEY0,
) -> list[dict]:
    """CRT v2 of one family against each target, one set of B draws for all targets.

    Args:
        data: Unit table.
        family: Knob family.
        targets: (name, key) pairs; key is a (relation, kpi) key of `data.y` or
            ("pre", relation, kpi) for the pre-window value.
        cfg: Test settings.
        key: Lag and split of the RNG stream.

    Returns:
        One dict per target, status "tested" or "undetermined".
    """
    cfg = cfg or UnitCRTConfigV2()
    rows, src = _family_rows(data, family, key.lag)
    base = _base(data, family, (rows, src), cfg, key)
    if not _supported(base, cfg):
        return [_undetermined(base, nm, k, "support") for nm, k in targets]
    out, cols, idx, infos = _residual_targets(data, rows, targets, base)
    if not cols:
        return cast(list[dict], out)
    R = np.column_stack(cols)
    v = design_regressor(data, src)
    s_obs = v @ R
    t_obs = np.abs(s_obs)
    thr = t_obs - 1e-9 * (1.0 + np.abs(R).sum(0))
    count, s1, s2, ch = _null_moments(data, src, R, thr, _Stream(cfg, family, key))
    vv = float(v @ v)
    sd = np.sqrt(np.maximum(s2 / cfg.B - (s1 / cfg.B) ** 2, 0.0))
    for j, t in enumerate(idx):
        nm, tk = targets[t]
        beta = float(s_obs[j] / vv) if vv > 0 else 0.0
        out[t] = dict(
            base,
            target=nm,
            key=list(tk),
            status="tested",
            p=float((1.0 + count[j]) / (cfg.B + 1.0)),
            t_obs=float(t_obs[j]),
            s_obs=float(s_obs[j]),
            beta=beta,
            sign=int(np.sign(beta)),
            z_approx=float(s_obs[j] / sd[j]) if sd[j] > 0 else float("nan"),
            null_mean=float(s1[j] / cfg.B),
            null_sd=float(sd[j]),
            chunk=ch,
            **infos[j],
        )
    return cast(list[dict], out)


def crt_unit_test_v2(
    data: UnitTable,
    hyp: tuple[str, str, str],
    cfg: UnitCRTConfigV2 | None = None,
    key: CrtKey = KEY0,
) -> dict:
    """One hypothesis (family, relation, kpi); the same p as the batched run (same draws)."""
    family, relation, kpi = hyp
    return family_tests_v2(data, family, [(f"{relation}_{kpi}", (relation, kpi))], cfg, key)[0]


def by_status(results: list[dict], q: float = 0.05) -> int:
    """BY at q over the tested results; sets their status and claim, returns the declared count."""
    tested = [r for r in results if r["status"] in ("tested", "declared", "not_detected")]
    if not tested:
        return 0
    dec_ = weighted_step_up(np.array([r["p"] for r in tested], float), None, q, reshape=True)
    for r, d in zip(tested, dec_, strict=True):
        r["status"] = "declared" if d else "not_detected"
        r["claim"] = CLAIMS_V2[r["status"]]
    return int(dec_.sum())


def _all_targets(data: UnitTable, pre: bool = False) -> list[Target]:
    tg: list[Target] = [(f"{r}_{k}", (r, k)) for r in data.relations for k in data.kpis]
    if pre:
        tg += [(f"pre_{r}_{k}", ("pre", r, k)) for r in data.relations for k in data.kpis]
    return tg


def run_crt_units_v2(
    data: UnitTable,
    cfg: UnitCRTConfigV2 | None = None,
    split: int = SPLIT_POOLED,
    audit: bool = False,
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Runs the 60 hypotheses with BY at q over the tested ones (and the optional lag-1 audit)."""
    cfg = cfg or UnitCRTConfigV2()
    t0 = time.time()
    res = []
    for f in families:
        for r in family_tests_v2(data, f, _all_targets(data), cfg, CrtKey(0, split)):
            r["relation"], r["kpi"] = r["key"]
            res.append(r)
    n_dec = by_status(res, cfg.q)
    n_tested = sum(r["status"] != "undetermined" for r in res)
    out = {
        "version": CRT_UNITS_V2_VERSION,
        "config": dataclasses.asdict(cfg),
        "split": split,
        "power_floor": power_floor(cfg.B, n_tested, cfg.q),
        "selection": f"BY-across-hypotheses ({CRT_UNITS_V2_VERSION}), q={cfg.q:g}, "
        f"over the tested of {len(res)}",
        "n_declared": n_dec,
        "n_tested": n_tested,
        "n_units": int(data.n),
        "results": res,
        "wall_s": round(time.time() - t0, 2),
    }
    if audit:
        out["audit"] = carryover_audit_v2(data, cfg, split, families)
    return out


def carryover_audit_v2(
    data: UnitTable,
    cfg: UnitCRTConfigV2 | None = None,
    split: int = SPLIT_POOLED,
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Descriptive lag-1 audit: the previous unit's regressor against this unit's targets."""
    cfg = cfg or UnitCRTConfigV2()
    acfg = dataclasses.replace(cfg, B=int(cfg.audit_B))
    res = []
    for f in families:
        res.extend(family_tests_v2(data, f, _all_targets(data, pre=True), acfg, CrtKey(1, split)))
    n_dec = by_status(res, cfg.q)
    tested = [r for r in res if r["status"] != "undetermined"]
    return {
        "B": acfg.B,
        "n_declared": n_dec,
        "n_tested": len(tested),
        "rate_p_le_alpha": (sum(r["p"] <= cfg.alpha for r in tested) / len(tested))
        if tested
        else None,
        "power_floor": power_floor(acfg.B, len(tested), cfg.q),
        "smallest": sorted((float(r["p"]), r["family"], r["target"]) for r in tested)[:8],
        "declared": [(r["family"], r["target"]) for r in res if r["status"] == "declared"],
        "results": res,
        "wording": "descriptive carryover audit (BY across the audit's own tests); never enters "
        "the verdict",
    }


def edges_from_crt_v2(run: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Returns {(family, relation, kpi): declared, sign, status, p, beta, z_approx, score}."""
    out = {}
    for r in run["results"]:
        p = r["p"]
        out[(r["family"], r["relation"], r["kpi"])] = {
            "declared": r["status"] == "declared",
            "sign": int(r["sign"]),
            "status": r["status"],
            "p": p,
            "beta": r["beta"],
            "z_approx": r["z_approx"],
            "score": float(-np.log10(p)) if np.isfinite(p) else float("nan"),
        }
    return out


def placebo_rejection_units_v2(
    data: UnitTable,
    cfg: UnitCRTConfigV2 | None = None,
    binom_level: float = 0.01,
    max_by: int = 1,
) -> dict[str, Any]:
    """K0 on placebo records: invalid iff the rejection count is binomially extreme.

    The run is invalid iff P(Binom(m, alpha) >= n_reject) < `binom_level` or there are more
    than `max_by` BY declarations.
    """
    cfg = cfg or UnitCRTConfigV2()
    if np.any(data.applied != MODES.index("accept")):
        raise ValueError("placebo records must have applied_mode == accept for every unit")
    run = run_crt_units_v2(data, cfg, split=SPLIT_PLACEBO, audit=False)
    tested = [r for r in run["results"] if r["status"] != "undetermined"]
    m = len(tested)
    rej = [r for r in tested if r["p"] <= cfg.alpha]
    p_count = float(binom.sf(len(rej) - 1, m, cfg.alpha)) if m else float("nan")
    k_max = int(binom.isf(binom_level, m, cfg.alpha)) if m else 0
    return {
        "version": CRT_UNITS_V2_VERSION,
        "n_hypotheses": len(run["results"]),
        "n_tested": m,
        "alpha": cfg.alpha,
        "rate": len(rej) / m if m else float("nan"),
        "n_reject": len(rej),
        "n_by_declared": run["n_declared"],
        "declared": [
            (r["family"], r["relation"], r["kpi"], r["sign"], r["p"])
            for r in run["results"]
            if r["status"] == "declared"
        ],
        "per_hypothesis": [_placebo_row(r, cfg.alpha) for r in run["results"]],
        "p_count": p_count,
        "k_max": k_max,
        "rule": f"P(Binom(m, {cfg.alpha}) >= n_reject) >= {binom_level} and <= {max_by} "
        "BY declaration(s)",
        "pass": bool(m > 0 and p_count >= binom_level and run["n_declared"] <= max_by),
        "binomial_fail_prob_if_uniform_indep": float(binom.sf(k_max, m, cfg.alpha)) if m else None,
        "run": run,
    }


def _placebo_row(r: dict[str, Any], alpha: float) -> dict[str, Any]:
    return {
        "family": r["family"],
        "relation": r["relation"],
        "kpi": r["kpi"],
        "status": r["status"],
        "p": r["p"],
        "z_approx": r["z_approx"],
        "reject": bool(r["status"] != "undetermined" and r["p"] <= alpha),
    }

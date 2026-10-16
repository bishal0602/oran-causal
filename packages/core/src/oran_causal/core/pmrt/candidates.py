"""Candidate handling shared by the PMRT row statistics: testability, BY families, results."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.core.covariates import EQ, R3, CovariateSpec, concurrent_indices
from oran_causal.core.data import Dataset, EdgeResult, Result
from oran_causal.core.declare import DeclareConfig, HypStats, declare
from oran_causal.core.design import Assignment, assignment
from oran_causal.core.scoring import placebo_tau, tau_rule_name

RNG_TAG = 7801
NOT_APPLICABLE = "not_applicable"
COVARIATE_SETS: Mapping[str, CovariateSpec] = {"eq": EQ, "r3": R3}

Edge = tuple[str, str]


def covariate_spec(name: str, hist_lags: int) -> CovariateSpec:
    """Covariate set `"eq"` or `"r3"` with `hist_lags` lags."""
    if name not in COVARIATE_SETS:
        raise ValueError(f"covariates must be one of {tuple(COVARIATE_SETS)}, got {name!r}")
    return dataclasses.replace(COVARIATE_SETS[name], lags=hist_lags)


def concurrent_columns(data: Dataset, ai: int) -> list[int]:
    """Other designed action columns adjusted for at t when testing action `ai`."""
    return concurrent_indices(data, ai, "designed")


def bad_targets(y: np.ndarray) -> np.ndarray:
    """Targets that are non-finite somewhere or constant."""
    return ~np.all(np.isfinite(y), 0) | (np.nanstd(y, 0) <= 0)


def primary_family(data: Dataset) -> tuple[list[Edge], list[Edge]]:
    """(primary, diagnostic) action-to-KPI candidates; each forms its own BY family."""
    acts, kset = set(data.action_names), set(data.kpi_names)
    family = [(h[0], h[1]) for h in data.candidates if h[0] in acts and h[1] in kset]
    if data.meta.get("primary_candidates") is None:
        return family, []
    prim = {tuple(h) for h in data.meta["primary_candidates"]}
    return [h for h in family if h in prim], [h for h in family if h not in prim]


def declare_families(
    stats: Mapping[Edge, HypStats], data: Dataset, procedure: str, q: float
) -> dict:
    """Declarations over the primary family and, separately, the diagnostic one."""
    family, diag = primary_family(data)
    config = DeclareConfig(q=q)
    dec = declare(stats, procedure, family, config) if stats and family else {}
    if diag and any(h in stats for h in diag):
        dec.update(declare(stats, procedure, diag, config))
    return dec


@dataclasses.dataclass
class CandidateRow:
    """One candidate edge after testing: its score and statistics, or why it was not tested."""

    src: str
    tgt: str
    score: float | None = None
    stats: HypStats | None = None
    why: str | None = None


def build_result(
    data: Dataset, rows: Sequence[CandidateRow], dec: Mapping[Edge, dict], tau: float | None
) -> tuple[tuple[EdgeResult, ...], dict[str, str], list[str]]:
    """Edge results, not-applicable reasons and tau declarations of tested candidates."""
    edges, reasons, tau_decl = [], {}, []
    for r in rows:
        if r.stats is None or r.score is None:
            edges.append(EdgeResult(r.src, r.tgt, float("nan"), None, 0, False))
            reasons[f"{r.src}->{r.tgt}"] = r.why
            continue
        d = dec.get((r.src, r.tgt), {"declared": False})
        edges.append(
            EdgeResult(r.src, r.tgt, r.score, r.stats.p2, int(r.stats.sign), bool(d["declared"]))
        )
        if tau is not None and r.score > tau:
            tau_decl.append(f"{r.src}->{r.tgt}")
    return tuple(edges), reasons, tau_decl


def tune_tau(method: Any, dev: list[Dataset]) -> tuple[dict[str, Any], list[Result]]:
    """Frozen config of a PMRT method plus the placebo-conformal score threshold over `dev`.

    Returns:
        (config with `tau` and `tau_rule`, the DEV results).
    """
    cfg = dataclasses.asdict(method.cfg)
    results = [method.run(d, cfg) for d in dev]
    cfg["tau"] = float(placebo_tau(results))
    cfg["tau_rule"] = tau_rule_name()
    return cfg, results


@dataclasses.dataclass(frozen=True)
class Overrides:
    """Fixed RNG streams per action and fixed weights, for equivalence checks only."""

    rngs: Mapping[str, np.random.Generator] = dataclasses.field(default_factory=dict)
    weights: np.ndarray | None = None


def testable(data: Dataset) -> tuple[dict[int, Assignment], dict[str, str]]:
    """Redrawable actions by index and the reason every other action is not testable."""
    ok, na = {}, {}
    for ai, a in enumerate(data.action_names):
        asg = assignment(data.designs[ai], np.asarray(data.X_action, float)[:, ai])
        if isinstance(asg, str):
            na[a] = asg
        elif not np.any(asg.var > 0):
            na[a] = "zero design variance"
        else:
            ok[ai] = asg
    return ok, na


def candidate_rows(
    data: Dataset,
    per_action: Mapping[str, dict],
    na: Mapping[str, str],
    bad_t: np.ndarray,
    two_sided: bool,
) -> tuple[list[CandidateRow], dict[Edge, HypStats]]:
    """Candidate rows and per-edge statistics from per-action CRT results."""
    kpis = list(data.kpi_names)
    stats: dict[Edge, HypStats] = {}
    rows: list[CandidateRow] = []
    for src, tgt in data.candidates:
        if src not in per_action:
            rows.append(
                CandidateRow(
                    src, tgt, why=na.get(src, "source is not an action with a known design")
                )
            )
            continue
        k = kpis.index(tgt)
        if bad_t[k]:
            rows.append(CandidateRow(src, tgt, why="degenerate target"))
            continue
        r = per_action[src]
        st = _edge_stats(r, k, two_sided)
        stats[(src, tgt)] = st
        score = float(abs(r["z"][k])) if two_sided else float(r["t"][k])
        rows.append(CandidateRow(src, tgt, score, st))
    return rows, stats


def _edge_stats(r: Mapping[str, np.ndarray], k: int, two_sided: bool) -> HypStats:
    if two_sided:
        return HypStats(
            p2=float(r["p2"][k]),
            sign=int(np.sign(r["z"][k])),
            p_plus=float(r["p_plus"][k]),
            p_minus=float(r["p_minus"][k]),
        )
    p = float(r["p"][k])
    return HypStats(p2=p, sign=int(np.sign(r["S"][k])), p_plus=p, p_minus=p)


def edge_notes(data: Dataset, per_action: Mapping[str, dict], key: str) -> dict[str, float]:
    """`{"src->tgt": value}` of one per-action result field over the tested sources."""
    kpis = list(data.kpi_names)
    return {
        f"{s}->{t}": float(per_action[s][key][kpis.index(t)])
        for s, t in data.candidates
        if s in per_action
    }

"""Scoring of declarations against truth, and the placebo score threshold.

Row corpora: `score` compares a `Result` with a `Truth`. Decision-unit maps: `confusion`
and `score_method` compare declared hypotheses with a knockout reference whose statuses
are TRUE, NULL or INDET; INDET is excluded and counted.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.core.data import EdgeResult, Result, Truth

PLACEBO = "P_placebo"
PLACEBO_CONF = "P_placebo_conf"
TAU_ALPHA = 0.05

Edge = tuple[str, str]


def _div(a: float, b: float) -> float:
    return a / b if b else math.nan


def prf(declared: set[Edge], edges: set[Edge], nulls: set[Edge]) -> dict[str, Any]:
    """Precision, recall, F1, false discovery proportion and null false-positive rate."""
    tp, v = len(declared & edges), len(declared & nulls)
    r = len(declared)
    prec, rec = _div(tp, r), _div(tp, len(edges))
    if math.isnan(prec) or math.isnan(rec):
        f1 = math.nan
    else:
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n_declared": r,
        "tp": tp,
        "fp": v,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fdp": v / max(r, 1),
        "null_fpr": _div(v, len(nulls)),
    }


def _parse_edge(method: str, it: Any) -> Edge:
    if isinstance(it, str):
        src, sep, tgt = it.partition("->")
        if not sep or not src or not tgt:
            raise ValueError(f"{method}: bad not-testable edge {it!r} (want 'src->tgt')")
        return src, tgt
    src, tgt = it
    return str(src), str(tgt)


def not_testable_edges(result: Result) -> set[Edge]:
    """Candidates listed under `notes["not_testable_edges"]` or `notes["not_testable"]`."""
    out: set[Edge] = set()
    for key in ("not_testable_edges", "not_testable"):
        v = (result.notes or {}).get(key)
        if v is None:
            continue
        if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
            kind = type(v).__name__
            raise ValueError(
                f"{result.method}: notes[{key!r}] must be a list or dict of edges, got {kind}"
            )
        for it in v.keys() if isinstance(v, dict) else v:
            out.add(_parse_edge(result.method, it))
    return out


def _index_edges(result: Result, cands: set[Edge]) -> dict[Edge, EdgeResult]:
    by_edge: dict[Edge, EdgeResult] = {}
    for e in result.edges:
        key = (e.source, e.target)
        if key not in cands:
            raise ValueError(f"{result.method}: edge {key} is not a candidate")
        if key in by_edge:
            raise ValueError(f"{result.method}: duplicate edge {key}")
        by_edge[key] = e
    return by_edge


def _block(
    sub: set[Edge],
    declared: set[Edge],
    truth: Truth,
    by_edge: Mapping[Edge, EdgeResult],
    nt: set[Edge],
) -> dict[str, Any]:
    edges, nulls = set(truth.edges) & sub, set(truth.null_edges) & sub
    out = prf(declared & sub, edges, nulls)
    signed = [k for k in declared & edges if k in truth.signs and by_edge[k].sign != 0]
    correct = sum(int(np.sign(by_edge[k].sign) == truth.signs[k]) for k in signed)
    out.update(
        sign_acc=_div(correct, len(signed)),
        sign_n=len(signed),
        n_candidates=len(sub),
        n_true=len(edges),
        n_null=len(nulls),
        n_missing=len(sub - set(by_edge)),
        n_not_testable_true=len(nt & edges),
        n_not_testable_null=len(nt & nulls),
    )
    return out


def score(
    result: Result,
    truth: Truth,
    candidates: Iterable[Edge] | None = None,
    kpi_sources: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Scores a `Result` against a `Truth`.

    The top level is the primary set (action-to-KPI candidates, placebos reported apart);
    `secondary` is the lagged-KPI-to-KPI set and `all` both. Not-testable edges are never
    declared.

    Args:
        result: Method output.
        truth: Ground truth.
        candidates: Candidate set; defaults to true and null edges.
        kpi_sources: Lagged-KPI source names; defaults to names starting with `K`.

    Returns:
        Precision, recall, F1, sign accuracy and placebo rates.
    """
    cands = set(candidates) if candidates is not None else set(truth.edges) | set(truth.null_edges)
    by_edge = _index_edges(result, cands)
    ks = set(kpi_sources) if kpi_sources is not None else None

    def is_kpi(s: str) -> bool:
        return s in ks if ks is not None else s.startswith("K")

    nt = not_testable_edges(result)
    if not nt <= cands:
        raise ValueError(
            f"{result.method}: not-testable edges outside the candidate set: {sorted(nt - cands)}"
        )
    flagged = {k for k, e in by_edge.items() if e.declared}
    declared = flagged - nt
    plac = {c for c in cands if c[0] == PLACEBO}
    conf = {c for c in cands if c[0] == PLACEBO_CONF}
    real = cands - plac - conf
    out = _block({c for c in real if not is_kpi(c[0])}, declared, truth, by_edge, nt)
    pd_, cd = declared & plac, declared & conf
    out.update(placebo_declared=len(pd_), placebo_fpr=_div(len(pd_), len(plac)))
    out.update(placebo_conf_declared=len(cd), placebo_conf_fpr=_div(len(cd), len(conf)))
    out["secondary"] = _block({c for c in real if is_kpi(c[0])}, declared, truth, by_edge, nt)
    out["all"] = _block(real, declared, truth, by_edge, nt)
    out.update(
        placebo_not_testable=len(nt & plac),
        n_not_testable_overridden=len(flagged & nt),
        not_testable_edges=sorted(f"{s}->{t}" for s, t in nt),
    )
    out["declared_edges"] = sorted(f"{s}->{t}" for s, t in declared)
    return out


def placebo_tau(
    results: Iterable[Result], max_declarations: int | None = None, alpha: float = TAU_ALPHA
) -> float:
    """Score threshold from the finite `P_placebo` edge scores of tune results.

    Default: the ceil((M + 1)(1 - alpha))-th smallest of the M scores (the largest if the
    index exceeds M), so a new exchangeable placebo score exceeds tau with probability at
    most alpha; M = 0 gives +inf. `max_declarations=k` instead returns the (k + 1)-th
    largest score (-inf if there are at most k scores).
    """
    s = sorted(
        e.score
        for r in results
        for e in r.edges
        if e.source == PLACEBO and e.score is not None and not math.isnan(e.score)
    )
    if max_declarations is not None:
        return float(s[-1 - max_declarations]) if len(s) > max_declarations else -math.inf
    m = len(s)
    if m == 0:
        return math.inf
    k = math.ceil((m + 1) * (1.0 - alpha) - 1e-9)
    return float(s[min(k, m) - 1])


def tau_rule_name(alpha: float = TAU_ALPHA) -> str:
    """Label of the default `placebo_tau` rule."""
    return f"placebo_conformal_{alpha:g}"


def apply_threshold(result: Result, tau: float) -> Result:
    """Copy of `result` with `declared = score > tau` (NaN scores never declared)."""
    edges = tuple(
        EdgeResult(
            e.source,
            e.target,
            e.score,
            e.p,
            e.sign,
            bool(e.score is not None and not math.isnan(e.score) and e.score > tau),
        )
        for e in result.edges
    )
    return Result(
        result.method,
        result.version,
        edges,
        result.cpu_s,
        {**result.config, "tau": tau},
        result.notes,
    )


def _safe(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _tally(
    decl: Mapping[Any, Any], ref: Mapping[Any, Any], hyps: Sequence[tuple]
) -> dict[str, int]:
    c = dict.fromkeys(("tp", "fp", "fn", "tn", "sign_ok", "n_indet", "n_unmapped"), 0)
    for h in hyps:
        g = ref.get(h, {"status": "INDET", "sign": 0})
        d = decl.get(h)
        if g["status"] == "INDET":
            c["n_indet"] += 1
        elif d is None or d.get("declared") is None:
            c["n_unmapped"] += 1
        elif d["declared"] and g["status"] == "TRUE":
            c["tp"] += 1
            c["sign_ok"] += int(int(d.get("sign", 0)) == g["sign"])
        elif d["declared"]:
            c["fp"] += 1
        elif g["status"] == "TRUE":
            c["fn"] += 1
        else:
            c["tn"] += 1
    return c


def confusion(
    decl: Mapping[Any, Any],
    ref: Mapping[Any, Any],
    hyps: Sequence[tuple],
    relation: str | None = None,
) -> dict[str, Any]:
    """Confusion counts and rates of declared hypotheses against a TRUE/NULL/INDET reference.

    Args:
        decl: Per hypothesis `{"declared", "sign"}`; a missing entry or `declared` None is unmapped.
        ref: Per hypothesis `{"status", "sign"}`; missing means INDET.
        hyps: Hypotheses `(family, relation, kpi, ...)` to score.
        relation: Restrict to hypotheses whose second element equals it.

    Returns:
        tp, fp, fn, tn, precision, recall, f1, sign_acc, n_true, n_null, n_indet, n_unmapped.
    """
    c = _tally(decl, ref, [h for h in hyps if relation is None or h[1] == relation])
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    prec, rec = _safe(tp, tp + fp), _safe(tp, tp + fn)
    f1 = (0.0 if tp + fp + fn > 0 else float("nan")) if tp == 0 else 2 * prec * rec / (prec + rec)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": c["tn"],
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "sign_acc": _safe(c["sign_ok"], tp),
        "n_true": tp + fn,
        "n_null": fp + c["tn"],
        "n_indet": c["n_indet"],
        "n_unmapped": c["n_unmapped"],
    }


def score_method(
    decl: Mapping[Any, Any], ref: Mapping[Any, Any], hyps: Sequence[tuple]
) -> dict[str, Any]:
    """Map scores overall and per relation (`nbr` is indirect), plus far declarations."""
    far = [h for h in hyps if h[1] == "far"]
    far_dec = sum(1 for h in far if (decl.get(h) or {}).get("declared"))
    far_fp = sum(
        1
        for h in far
        if (decl.get(h) or {}).get("declared") and ref.get(h, {}).get("status") == "NULL"
    )
    return {
        "overall": confusion(decl, ref, hyps),
        "indirect": confusion(decl, ref, hyps, "nbr"),
        "own": confusion(decl, ref, hyps, "own"),
        "far": confusion(decl, ref, hyps, "far"),
        "far_declared": far_dec,
        "far_fp": far_fp,
        "n_declared": sum(1 for d in decl.values() if d and d.get("declared")),
    }

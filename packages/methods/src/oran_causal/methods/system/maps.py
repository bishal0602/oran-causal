"""Declared referee maps of the confounded study and their offline MapGate replay.

A map is {(family, rel, kpi): beta}. `map_from_declared` turns a baseline's declared hypotheses
into a map (sign x |naive OLS slope|), `map_from_mscr` an MSCR-CRT v2 run. `edge_quality` scores a
map against the knockout GT cells (TRUE right sign / TRUE wrong sign / NULL / INDET / outside).
`replay` decides every logged unit of the given episodes with a fresh `MapGateV2` per map and
episode, in opening order, and counts the decisions that differ between map pairs (flips).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.core.data import UnitTable
from oran_causal.methods.system.mscr_crt import edges_from_crt_v2
from oran_causal.methods.system.referees import K_CONF, THETA, Map, MapGateV2, map_from_gt
from oran_causal.sim.system.decision.vocab import FAMILIES

__all__ = [
    "GT_EXT_CELLS",
    "MAX_FP",
    "clean_map",
    "edge_quality",
    "gt_cells",
    "gt_map",
    "inject",
    "map_from_declared",
    "map_from_mscr",
    "naive_slope",
    "placebo_tau",
    "replay",
    "tuple_scores",
]

MAX_FP = 1

GT_EXT_CELLS = (
    ("carrier", "own", "pv", "TRUE", -6.783080260303688),
    ("carrier", "own", "v", "TRUE", -116.27548806941432),
    ("carrier", "own", "e", "TRUE", 6581.903035313515),
    ("carrier", "own", "rlf", "INDET", 0.005061460592913955),
    ("carrier", "own", "load", "TRUE", -55.611713665943604),
    ("carrier", "nbr", "pv", "TRUE", 0.8250180766449747),
    ("carrier", "nbr", "v", "INDET", 14.370932754880695),
    ("carrier", "nbr", "e", "INDET", 301.05112627527734),
    ("carrier", "nbr", "rlf", "INDET", 0.03470715835140997),
    ("carrier", "nbr", "load", "NULL", 36.4887924801157),
    ("carrier", "far", "pv", "INDET", 0.2566883586406363),
    ("carrier", "far", "v", "NULL", 6.36948662328272),
    ("carrier", "far", "e", "NULL", 117.91223646646007),
    ("carrier", "far", "rlf", "INDET", 0.014461315979754157),
    ("carrier", "far", "load", "NULL", 19.122921185827913),
    ("sleep", "own", "pv", "TRUE", -0.6933333333333335),
    ("sleep", "own", "v", "INDET", -12.68),
    ("sleep", "own", "e", "NULL", -190.07756876627607),
    ("sleep", "own", "rlf", "TRUE", -0.11555555555555555),
    ("sleep", "own", "load", "TRUE", -995.9377777777779),
    ("sleep", "nbr", "pv", "TRUE", 9.066666666666668),
    ("sleep", "nbr", "v", "TRUE", 399.1333333333334),
    ("sleep", "nbr", "e", "TRUE", 4695.624190710959),
    ("sleep", "nbr", "rlf", "INDET", 0.05333333333333334),
    ("sleep", "nbr", "load", "TRUE", 921.2622222222221),
    ("sleep", "far", "pv", "TRUE", 2.417777777777778),
    ("sleep", "far", "v", "TRUE", 33.711111111111116),
    ("sleep", "far", "e", "TRUE", 1367.366021516699),
    ("sleep", "far", "rlf", "TRUE", 0.07555555555555554),
    ("sleep", "far", "load", "TRUE", 74.67555555555555),
    ("ptx", "own", "pv", "TRUE", 2.0338461538461536),
    ("ptx", "own", "v", "TRUE", 41.83794871794872),
    ("ptx", "own", "e", "TRUE", 4811.618351655618),
    ("ptx", "own", "rlf", "INDET", -0.16102564102564101),
    ("ptx", "own", "load", "TRUE", 319.1569230769231),
    ("ptx", "nbr", "pv", "INDET", -0.34974358974358966),
    ("ptx", "nbr", "v", "TRUE", -31.317948717948713),
    ("ptx", "nbr", "e", "TRUE", -503.11579152657976),
    ("ptx", "nbr", "rlf", "INDET", -0.2348717948717949),
    ("ptx", "nbr", "load", "TRUE", -212.9876923076923),
    ("ptx", "far", "pv", "INDET", -0.5394871794871796),
    ("ptx", "far", "v", "TRUE", -20.45641025641026),
    ("ptx", "far", "e", "TRUE", -334.2005622661469),
    ("ptx", "far", "rlf", "INDET", -0.18974358974358976),
    ("ptx", "far", "load", "TRUE", -106.16923076923077),
    ("prot_min", "own", "pv", "TRUE", -5.126126126126126),
    ("prot_min", "own", "v", "NULL", 0.15894465894465903),
    ("prot_min", "own", "e", "NULL", -2.2252273244725624),
    ("prot_min", "own", "rlf", "NULL", 0.0),
    ("prot_min", "own", "load", "NULL", -0.10489060489060488),
    ("prot_min", "nbr", "pv", "NULL", -0.006435006435006429),
    ("prot_min", "nbr", "v", "NULL", -0.258043758043758),
    ("prot_min", "nbr", "e", "NULL", -10.443604805554294),
    ("prot_min", "nbr", "rlf", "NULL", -0.002574002574002574),
    ("prot_min", "nbr", "load", "NULL", -0.05984555984555983),
    ("prot_min", "far", "pv", "NULL", 0.0341055341055341),
    ("prot_min", "far", "v", "NULL", -0.05019305019305022),
    ("prot_min", "far", "e", "NULL", -9.389110710423813),
    ("prot_min", "far", "rlf", "NULL", -0.0006435006435006435),
    ("prot_min", "far", "load", "NULL", 0.1647361647361647),
)

_QUALITY = ("true_right", "true_wrong", "null", "indet", "unknown")


def gt_cells(rows: Iterable[Sequence[Any]] = GT_EXT_CELLS) -> list[dict[str, Any]]:
    """GT cells as dicts (family, relation, kpi, status, mean) from (f, r, k, status, mean) rows."""
    return [
        {"family": f, "relation": r, "kpi": k, "status": s, "mean": m} for f, r, k, s, m in rows
    ]


def gt_map(cells: Iterable[Mapping[str, Any]]) -> dict:
    """Map of the TRUE GT cells (their means)."""
    return map_from_gt(cells, true_only=True)


def _quality_of(c: Mapping[str, Any] | None, b: float | None) -> str:
    if c is None:
        return "unknown"
    if c["status"] == "TRUE":
        s = int(np.sign(b)) if b is not None else 0
        return "true_right" if s == int(np.sign(c["mean"])) else "true_wrong"
    return "null" if c["status"] == "NULL" else "indet"


def edge_quality(M: Map, cells: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Map edges vs the GT cells: TRUE right sign / TRUE wrong sign / NULL / INDET / unknown."""
    st = {(c["family"], c["relation"], c["kpi"]): c for c in cells}
    q: dict[str, list] = {k: [] for k in _QUALITY}
    for h, b in sorted(M.items()):
        q[_quality_of(st.get(h), b)].append(h)
    gt_true = {h for h, c in st.items() if c["status"] == "TRUE"}
    n_true = len(q["true_right"])
    return {
        "n_edges": len(M),
        **{f"n_{k}": len(v) for k, v in q.items()},
        "edges": q,
        "precision_true_right": n_true / len(M) if M else float("nan"),
        "recall_true_right": n_true / len(gt_true) if gt_true else float("nan"),
    }


def naive_slope(data: UnitTable, family: str, rel: str, kpi: str) -> float:
    """OLS slope of y(rel, kpi) on x = level * sgn over the family's units; 0 below 3 units."""
    rows = data.rows_of(family)
    if len(rows) < 3:
        return 0.0
    x, y = data.x[rows], data.y[(rel, kpi)][rows]
    xc = x - x.mean()
    den = float(xc @ xc)
    return float(xc @ (y - y.mean()) / den) if den > 0 else 0.0


def _hyp(h: str | Sequence[str]) -> tuple[str, ...]:
    return tuple(h.split("|")) if isinstance(h, str) else tuple(h)


def map_from_declared(decl: Mapping[Any, Mapping[str, Any]], data: UnitTable) -> tuple[dict, int]:
    """Map sign * |naive slope| over the declared hypotheses; also the number dropped for sign 0."""
    M, drop = {}, 0
    for key, v in decl.items():
        h = _hyp(key)
        if not v.get("declared"):
            continue
        s = int(v.get("sign", 0))
        if s == 0:
            drop += 1
            continue
        f, rel, kpi = h
        M[h] = float(s * abs(naive_slope(data, f, rel, kpi)))
    return M, drop


def map_from_mscr(run: dict[str, Any]) -> dict:
    """MSCR-CRT v2 declared edges with a nonzero sign -> {h: design-centred slope beta}."""
    return {
        h: float(v["beta"])
        for h, v in edges_from_crt_v2(run).items()
        if v["declared"] and v["sign"] != 0
    }


def placebo_tau(
    scores: Mapping[Any, Mapping[str, Any]], relations: Iterable[str], max_fp: int = MAX_FP
) -> dict[str, Any]:
    """Tau = the (max_fp + 1)-th largest finite placebo score over `relations` (declare > tau)."""
    rels = set(relations)
    s = sorted(
        (
            v["score"]
            for h, v in scores.items()
            if _hyp(h)[1] in rels and np.isfinite(v.get("score", np.nan))
        ),
        reverse=True,
    )
    if len(s) <= max_fp:
        return {
            "tau": float("-inf"),
            "max_fp": max_fp,
            "n_scorable": len(s),
            "note": "<= max_fp scorable placebo hypotheses: declares every scorable one",
        }
    return {"tau": float(s[max_fp]), "max_fp": max_fp, "n_scorable": len(s)}


def tuple_scores(sc: Mapping[Any, Any]) -> dict:
    """Score dict with "f|r|k" string keys turned into (f, r, k) tuples."""
    return {_hyp(k) if isinstance(k, str) else k: v for k, v in sc.items()}


def _pairs(maps: Mapping[str, Map], ref: str, pairs: Mapping | None) -> dict:
    if pairs is None:
        if ref not in maps:
            raise ValueError(f"reference map {ref!r} missing")
        pairs = {m: (m, ref) for m in maps if m != ref}
    for a, b in pairs.values():
        if a not in maps or b not in maps:
            raise ValueError(f"pair ({a}, {b}) names a missing map")
    return dict(pairs)


def _episode_decisions(
    rec: Mapping[str, Any], maps: Mapping[str, Map], theta: float, k_conf: int
) -> tuple[list[dict], dict[str, np.ndarray]]:
    gates = {m: MapGateV2(maps[m], theta, k_conf) for m in maps}
    units = [u for u in rec["units"] if u["knob"] in FAMILIES]
    dec = {m: np.zeros(len(units), bool) for m in maps}
    for i, u in enumerate(units):
        view = {"c": int(u["c"]), "ctx": u["ctx"]}
        for m in maps:
            dec[m][i] = gates[m](dict(view))[0] == "reject"
    return units, dec


def _dir_key(u: Mapping[str, Any]) -> str:
    step = float(u["ctx"]["step"])
    return f"{u['knob']}{'+' if step > 0 else '-' if step < 0 else '0'}"


def _flip_counts(
    dec: Mapping[str, np.ndarray], a: str, b: str, pre: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    diff = dec[a] != dec[b]
    return diff, [int(diff.sum()), int((diff & pre).sum()), int((diff & ~pre).sum())]


def replay(
    episodes: Sequence[Mapping[str, Any]],
    maps: Mapping[str, Map],
    ref: str = "GT",
    pairs: Mapping[str, tuple[str, str]] | None = None,
    *,
    theta: float = THETA,
    k_conf: int = K_CONF,
    warm: float = 120.0,
) -> dict[str, Any]:
    """Offline MapGateV2 replay of the logged units of `episodes` under every map.

    A flip is a unit whose decision under map a differs from reference map b. `pairs` maps a
    comparison name to (a, b) (default: every map vs `ref`). "by" keys are
    "<family><+|-|0>:refdefer->accept" (b defers, a accepts) or ":refaccept->defer".

    Returns:
        {"n_units": [per episode], "decisions": {map: [defers per episode]}, "defer_rate": {map},
        "flips": {pair: {"per_episode", "pre_warm", "post_warm", "by": {key: [per episode]}}}}.

    Raises:
        ValueError: `ref` or a pair names a map not in `maps`.
    """
    pairs = _pairs(maps, ref, pairs)
    names = list(maps)
    n_units: list[int] = []
    decisions: dict[str, list[int]] = {m: [] for m in names}
    counts: dict[str, list[list[int]]] = {p: [] for p in pairs}
    by: dict[str, dict[str, np.ndarray]] = {p: {} for p in pairs}
    for e, rec in enumerate(episodes):
        units, dec = _episode_decisions(rec, maps, theta, k_conf)
        n_units.append(len(units))
        for m in names:
            decisions[m].append(int(dec[m].sum()))
        pre = np.array([float(u["t0"]) < warm for u in units], bool)
        keys = [_dir_key(u) for u in units]
        for p, (a, b) in pairs.items():
            diff, c = _flip_counts(dec, a, b, pre)
            counts[p].append(c)
            for i in np.nonzero(diff)[0]:
                k = f"{keys[i]}:{'refdefer->accept' if dec[b][i] else 'refaccept->defer'}"
                by[p].setdefault(k, np.zeros(len(episodes), int))[e] += 1
    total = max(sum(n_units), 1)
    flips = {
        p: {
            "per_episode": [c[0] for c in counts[p]],
            "by": {k: v.tolist() for k, v in sorted(by[p].items())},
            "pre_warm": [c[1] for c in counts[p]],
            "post_warm": [c[2] for c in counts[p]],
        }
        for p in pairs
    }
    return {
        "n_units": n_units,
        "decisions": decisions,
        "defer_rate": {m: float(sum(decisions[m]) / total) for m in names},
        "flips": flips,
    }


def clean_map(M: Map, cells: Iterable[Mapping[str, Any]]) -> dict:
    """M restricted to its GT-TRUE right-sign edges (the method's own magnitudes)."""
    keep = set(edge_quality(M, cells)["edges"]["true_right"])
    return {h: b for h, b in M.items() if h in keep}


def inject(M_ref: Map, M_add: Map) -> dict:
    """M_ref with M_add's edges added (a key in both takes M_add's value)."""
    return {**M_ref, **M_add}

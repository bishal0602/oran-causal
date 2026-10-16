"""Knockout ground truth of the system-level discovery study.

A sampled unit u = (c, x, t0) of a GT episode is labelled by paired rollouts from the copied state
at t0: for each reseed k in `GT_KS` the unit's mode is held at accept and at reject for its T,
every other unit is accepted, and the rollout runs `GT_H` seconds. The per-cell contrast
D[k, kpi, cell] = y(accept) - y(reject) is the unit effect. Orientation "act" keeps D as is;
"dir" multiplies it by sign(prop - cur) of the opening request, the effect of a knob-value
increase.

The unit value of hypothesis (f, r, k) is the mean over reseeds of D summed over the relation's
cells. A hypothesis is TRUE (with the sign of its mean) when the episode-cluster bootstrap
interval excludes 0 and |mean| >= delta_k, NULL when the interval lies in [-delta_k, delta_k],
and INDET otherwise or when unsupported. delta_k = max(`REL_FRAC` x max |mean| of KPI k,
`KPI_FLOOR`[k]).
"""

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from oran_causal.sim.system.decision.vocab import (
    FAMILIES,
    KPIS,
    ORIENTS,
    PRIMARY_ORIENT,
    RELATIONS,
    Hypothesis,
)

GT_TAG = 6617
GT_KS = (1, 2, 3)
GT_H = 90
GT_MODES = ("accept", "reject")
REL_FRAC = 0.05
KPI_FLOOR = {"pv": 0.5}
N_BOOT = 4000
CI = (0.025, 0.975)
GT_MIN_UNITS, GT_MIN_EPS = 10, 3
GT_RATE = {"carrier": 1.0, "sleep": 1.0, "ptx": 1.0, "prot_min": 0.3}

Episode = Mapping[str, Any]
Row = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class GtRule:
    """Classification rule of the knockout ground truth.

    Attributes:
        n_boot: Episode-cluster bootstrap draws.
        rel_frac: Share of the largest |mean| of a KPI that sets its delta.
        floors: Minimum delta per KPI.
        min_units: Units below which a hypothesis is unsupported.
        min_eps: Episodes below which a hypothesis is unsupported.
    """

    n_boot: int = N_BOOT
    rel_frac: float = REL_FRAC
    floors: Mapping[str, float] = dataclasses.field(default_factory=lambda: dict(KPI_FLOOR))
    min_units: int = GT_MIN_UNITS
    min_eps: int = GT_MIN_EPS


GT_RULE = GtRule()


class BootStat(NamedTuple):
    """Pooled unit mean with its episode-cluster bootstrap interval."""

    mean: float
    ci: tuple[float, float]
    n: int
    n_eps: int


def relation_cells(c: int, exp: Iterable[int], n_cells: int) -> dict[str, list[int]]:
    """Returns the sorted cells of each relation of a unit on cell `c` with exposure set `exp`."""
    ex = {int(v) for v in exp}
    return {
        "own": [int(c)],
        "nbr": sorted(ex - {int(c)}),
        "far": sorted(set(range(int(n_cells))) - ex),
    }


def unit_delta(label: Mapping[str, Any]) -> np.ndarray:
    """Returns the accept-minus-reject contrast, shape (reseeds, cell KPIs, cells), of a label.

    Raises:
        ValueError: The label has no per-cell contrasts.
    """
    if "cell" not in label:
        raise ValueError("label has no per-cell contrasts (Labeller(cells=True))")
    return -np.asarray(label["cell"]["reject"], float)


def _unit_row(seed: int, n_cells: int, label: Mapping[str, Any], kpis: Sequence[str]) -> Row:
    delta = np.asarray(label["delta"], float)
    ki = [list(label["kpis"]).index(k) for k in kpis]
    dm = delta[:, ki, :].mean(0)
    sgn = float(np.sign(label["step"])) or 1.0
    rel = relation_cells(label["c"], label["exp"], n_cells)
    cells = {"act": dm, "dir": dm * sgn}
    val = {
        o: {
            r: {k: float(cells[o][j, idx].sum()) for j, k in enumerate(kpis)}
            for r, idx in rel.items()
        }
        for o in ORIENTS
    }
    return {
        "ep": int(seed),
        "family": label["knob"],
        "c": int(label["c"]),
        "t0": label.get("t0"),
        "step": float(label["step"]),
        "val": val,
        "cells": cells,
    }


def unit_rows(episodes: Iterable[Episode], kpis: Sequence[str] = KPIS) -> list[Row]:
    """Returns one row per GT label with its unit values per orientation, relation and KPI.

    Args:
        episodes: Decoded GT episodes `{"seed", "n_cells", "gt_labels": [label, ...]}`; each
            label carries c, knob, exp, step, kpis and delta (reseeds, kpis, cells).
        kpis: KPIs to keep.

    Returns:
        Rows `{"ep", "family", "c", "t0", "step", "val": {orient: {rel: {kpi: y}}},
        "cells": {orient: (kpis, cells) mean-over-reseed contrast}}`.
    """
    return [
        _unit_row(ep["seed"], int(ep["n_cells"]), label, kpis)
        for ep in episodes
        for label in ep["gt_labels"]
    ]


def cluster_boot(
    y: Sequence[float], ep: Sequence[int], rng: np.random.Generator, n_boot: int = N_BOOT
) -> BootStat:
    """Returns the pooled mean and its episode-cluster percentile bootstrap interval."""
    y_a, ep_a = np.asarray(y, float), np.asarray(ep)
    if len(y_a) == 0:
        return BootStat(float("nan"), (float("nan"), float("nan")), 0, 0)
    eps, inv = np.unique(ep_a, return_inverse=True)
    s = np.bincount(inv, y_a, len(eps))
    n = np.bincount(inv, minlength=len(eps)).astype(float)
    idx = rng.integers(len(eps), size=(n_boot, len(eps)))
    bm = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.quantile(bm, CI)
    return BootStat(float(y_a.mean()), (float(lo), float(hi)), len(y_a), len(eps))


def classify(stat: BootStat, delta: float, rule: GtRule = GT_RULE) -> dict[str, Any]:
    """Returns the TRUE / NULL / INDET status, sign and reason of one hypothesis."""
    mean, (lo, hi), n, n_eps = stat
    if n < rule.min_units or n_eps < rule.min_eps or not np.isfinite(mean):
        return {"status": "INDET", "sign": 0, "why": f"unsupported (n={n}, episodes={n_eps})"}
    if (lo > 0 or hi < 0) and abs(mean) >= delta:
        return {
            "status": "TRUE",
            "sign": int(np.sign(mean)),
            "why": "CI excludes 0, |mean| >= delta",
        }
    if -delta <= lo and hi <= delta:
        return {"status": "NULL", "sign": 0, "why": "CI within +-delta"}
    return {"status": "INDET", "sign": 0, "why": "neither rule"}


def _boot_stats(rows: Sequence[Row], orient: str, n_boot: int) -> dict[Hypothesis, BootStat]:
    oi = ORIENTS.index(orient)
    stats = {}
    for fi, f in enumerate(FAMILIES):
        fr = [r for r in rows if r["family"] == f]
        ep = [r["ep"] for r in fr]
        for ri, rel in enumerate(RELATIONS):
            for ki, k in enumerate(KPIS):
                y = [r["val"][orient][rel][k] for r in fr]
                rng = np.random.default_rng([GT_TAG, fi, ri, ki, oi])
                stats[(f, rel, k)] = cluster_boot(y, ep, rng, n_boot)
    return stats


def _deltas(stats: Mapping[Hypothesis, BootStat], rule: GtRule) -> dict[str, float]:
    delta = {}
    for k in KPIS:
        m = [abs(v.mean) for h, v in stats.items() if h[2] == k and np.isfinite(v.mean)]
        delta[k] = max(rule.rel_frac * (max(m) if m else 0.0), float(rule.floors.get(k, 0.0)))
    return delta


def _rule_text(rule: GtRule) -> str:
    return (
        f"TRUE: CI{list(CI)} excludes 0 and |mean| >= delta; NULL: CI within +-delta; delta = "
        f"max({rule.rel_frac} * max|mean| of the KPI, floor {dict(rule.floors)}); INDET otherwise "
        f"/ unsupported (< {rule.min_units} units or < {rule.min_eps} episodes); episode-cluster "
        f"bootstrap B={rule.n_boot}, tag {GT_TAG}"
    )


def gt_table(
    rows: Sequence[Row], orient: str = PRIMARY_ORIENT, rule: GtRule = GT_RULE
) -> dict[str, Any]:
    """Classifies every hypothesis from the unit rows.

    Returns:
        `{"orient", "delta": {kpi: d}, "cells": [{family, relation, kpi, mean, ci, n, n_eps,
        delta, status, sign, why}], "counts": {status: n}, "rule"}` over families x relations x
        KPIs.
    """
    stats = _boot_stats(rows, orient, rule.n_boot)
    delta = _deltas(stats, rule)
    cells, counts = [], {"TRUE": 0, "NULL": 0, "INDET": 0}
    for (f, rel, k), st in stats.items():
        c = classify(st, delta[k], rule)
        counts[c["status"]] += 1
        cells.append(
            {
                "family": f,
                "relation": rel,
                "kpi": k,
                "mean": st.mean,
                "ci": list(st.ci),
                "n": st.n,
                "n_eps": st.n_eps,
                "delta": delta[k],
                **c,
            }
        )
    return {
        "orient": orient,
        "delta": delta,
        "cells": cells,
        "counts": counts,
        "rule": _rule_text(rule),
    }


def _candidate(gt_static: Mapping[str, Any], pico: int) -> int | None:
    cand = gt_static.get("cand", {})
    v = cand.get(pico, cand.get(str(pico)))
    return int(v) if v is not None else None


def _shares(
    seed: int, pico: int, vs: list[np.ndarray], cand: int | None, top: int
) -> dict[str, Any]:
    d = np.sum(vs, 0)
    pos = np.where(np.arange(len(d)) == pico, 0.0, np.maximum(d, 0.0))
    tot = float(pos.sum())
    shares = pos / tot if tot > 0 else np.zeros_like(pos)
    order = [int(q) for q in np.argsort(-shares, kind="stable")[:top] if shares[q] > 0]
    top1 = order[0] if order else None
    return {
        "seed": int(seed),
        "pico": int(pico),
        "n_units": len(vs),
        "cand": cand,
        "top1": top1,
        "top1_is_cand": top1 is not None and top1 == cand,
        "share_cand": float(shares[cand]) if cand is not None else None,
        "shares": {str(q): float(shares[q]) for q in order},
        "pico_load_delta": float(d[pico]),
        "gain_total": tot,
    }


def receiving_shares(
    rows: Sequence[Row], gt_static_by_seed: Mapping[int, Mapping[str, Any]], top: int = 5
) -> list[dict[str, Any]]:
    """Returns, per (seed, pico), each cell's share of the "dir" load gain of the pico's sleeps.

    Only sleep units with a positive step count. The top-1 receiver is compared with the pico's
    energy-saving candidate macro of `gt_static["cand"]`.
    """
    li = list(KPIS).index("load")
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for r in rows:
        if r["family"] == "sleep" and r["step"] > 0:
            groups.setdefault((r["ep"], r["c"]), []).append(r["cells"]["dir"][li])
    return [
        _shares(seed, pico, vs, _candidate(gt_static_by_seed.get(seed) or {}, pico), top)
        for (seed, pico), vs in sorted(groups.items())
    ]


def summarize(
    episodes: Sequence[Episode], orients: Sequence[str] = ORIENTS, rule: GtRule = GT_RULE
) -> dict[str, Any]:
    """Returns label counts, the GT table per orientation and the receiving-cell shares."""
    rows = unit_rows(episodes)
    per: dict[str, int] = {}
    for r in rows:
        per[r["family"]] = per.get(r["family"], 0) + 1
    return {
        "n_labels": len(rows),
        "labels_per_family": per,
        "tables": {o: gt_table(rows, o, rule) for o in orients},
        "receiving": receiving_shares(
            rows, {int(e["seed"]): e.get("gt_static") or {} for e in episodes}
        ),
    }

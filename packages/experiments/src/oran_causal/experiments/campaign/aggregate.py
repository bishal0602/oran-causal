"""DEV aggregate: validity, power and cost per (arm, world, regime, lam, n, kappa) cell.

Measurement seeds only: truth-null and placebo declaration and raw-p rates with seed-cluster
bootstrap CIs, per-seed recall / FDP / sign accuracy, cost. Score-only ("tau") arms declare
score > tau, with tau from the cell's tune-seed records. Truth enters only here.
"""

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from oran_causal.core.data import EdgeResult, Result, Truth
from oran_causal.core.scoring import PLACEBO, PLACEBO_CONF, apply_threshold, placebo_tau, score
from oran_causal.experiments.campaign.cost import design_effect
from oran_causal.experiments.campaign.integrity import record_status
from oran_causal.experiments.campaign.run import CAMPAIGN_VERSION
from oran_causal.experiments.campaign.spec import Unit
from oran_causal.sim.controlled.datasets.truth import truth_for

ALPHA = 0.05
BOOT_REPS = 2000
BOOT_SEED = 20261002
MIN_FLAG_SEEDS = 10
RATE_KINDS = ("null", "plac", "conf")

type Record = dict[str, Any]
type Spec = dict[str, Any]


def record_unit(r: Record) -> Unit:
    """The unit of a stored record."""
    j = r["job"]
    lam = None if j.get("lam") is None else float(j["lam"])
    arm = r.get("arm") or r["key"].split("|")[0]
    kappa = float(j.get("kappa") or 0.0)
    return Unit(
        j["world"],
        j["regime"],
        lam,
        int(j["n"]),
        kappa,
        int(j["seed"]),
        arm,
        r.get("role") or "measure",
    )


def to_result(r: Record) -> Result:
    """The `Result` stored in a record."""
    edges = tuple(
        EdgeResult(
            e["source"],
            e["target"],
            math.nan if e.get("score") is None else float(e["score"]),
            e.get("p"),
            int(e.get("sign") or 0),
            bool(e.get("declared")),
        )
        for e in r.get("edges", [])
    )
    return Result(
        r.get("method") or "?",
        r.get("version") or "?",
        edges,
        float(r.get("method_cpu_s") or 0.0),
        dict(r.get("config") or {}),
        dict(r.get("notes") or {}),
    )


def cluster_ci(
    hits: Any, cnt: Any, reps: int = BOOT_REPS, seed: int = BOOT_SEED
) -> list[float | None]:
    """Seed-cluster bootstrap 95 % percentile CI of sum(hits) / sum(cnt)."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    keep = c > 0
    h, c = h[keep], c[keep]
    if not len(c):
        return [None, None]
    idx = np.random.default_rng(seed).integers(0, len(c), (reps, len(c)))
    bs = h[idx].sum(1) / c[idx].sum(1)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def wilson_deff(hits: Any, cnt: Any, z: float = 1.959964) -> list[float | None]:
    """Wilson 95 % interval on the design-effect-adjusted sample size; defined at 0 hits."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    total = c.sum()
    if total <= 0:
        return [None, None]
    p = h.sum() / total
    deff = design_effect(h, c) if 0 < p < 1 and len(c) > 1 else 1.0
    ne = total / deff
    q = z * z / ne
    mid = (p + q / 2) / (1 + q)
    half = z * math.sqrt(p * (1 - p) / ne + q / (4 * ne)) / (1 + q)
    return [max(0.0, mid - half), min(1.0, mid + half)]


def rate(hits: Any, cnt: Any) -> dict[str, Any]:
    """A pooled rate with its bootstrap and Wilson intervals and the `above` validity flag."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    total = float(c.sum())
    ci = cluster_ci(h, c)
    return {
        "rate": float(h.sum() / total) if total else None,
        "hits": int(h.sum()),
        "n": int(total),
        "n_seeds": int(np.sum(c > 0)),
        "ci": ci,
        "ci_wilson_deff": wilson_deff(h, c),
        "deff": design_effect(h[c > 0], c[c > 0]),
        "above": bool(ci[0] is not None and ci[0] > ALPHA),
    }


def is_primary(src: str) -> bool:
    """Whether a source is a real action (not a lagged KPI or a placebo column)."""
    return not src.startswith("K") and src not in (PLACEBO, PLACEBO_CONF)


def _kind(source: str, target: str, nulls: set[tuple[str, str]]) -> str | None:
    if source == PLACEBO:
        return "plac"
    if source == PLACEBO_CONF:
        return "conf"
    return "null" if (source, target) in nulls else None


def _counts(
    r: Record, sc: dict[str, Any], truth: Truth
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    nt = set(sc.get("not_testable_edges", []))
    declared = set(sc["declared_edges"])
    nulls = {e for e in truth.null_edges if is_primary(e[0])}
    cnt = {k: [0, 0] for k in RATE_KINDS}
    raw = {k: [0, 0] for k in RATE_KINDS}
    for e in r.get("edges", []):
        name = f"{e['source']}->{e['target']}"
        kind = _kind(e["source"], e["target"], nulls)
        if kind is None or name in nt:
            continue
        cnt[kind][1] += 1
        cnt[kind][0] += int(name in declared)
        if e.get("p") is not None:
            raw[kind][1] += 1
            raw[kind][0] += int(float(e["p"]) <= ALPHA)
    return cnt, raw


def seed_row(r: Record, decl: Result, truth: Truth) -> dict[str, Any]:
    """Per-seed validity and power numbers of one measurement record under the declaration `decl`.

    Rate denominators exclude not-testable candidates; raw-p rates also need a p-value.
    """
    sc = score(decl, truth)
    cnt, raw = _counts(r, sc, truth)
    return {
        "seed": int(r["job"]["seed"]),
        "recall": sc["recall"],
        "fdp": sc["fdp"],
        "precision": sc["precision"],
        "sign_acc": sc["sign_acc"],
        "sign_n": sc["sign_n"],
        "n_declared": sc["n_declared"],
        "tp": sc["tp"],
        "n_true": sc["n_true"],
        "null_fp": cnt["null"][0],
        "n_null": cnt["null"][1],
        "raw_null": raw["null"],
        "plac_decl": cnt["plac"][0],
        "n_plac": cnt["plac"][1],
        "raw_plac": raw["plac"],
        "conf_decl": cnt["conf"][0],
        "n_conf": cnt["conf"][1],
        "raw_conf": raw["conf"],
        "nt_true": sc["n_not_testable_true"],
        "nt_null": sc["n_not_testable_null"],
        "declared": sc["declared_edges"],
        "cpu_s": r.get("cpu_s"),
        "peak_rss_mb": r.get("peak_rss_mb"),
    }


def _mean(v: Iterable[Any]) -> float | None:
    vals = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(vals)) if vals else None


def _rates(rows: list[dict[str, Any]], has_p: bool) -> dict[str, Any]:
    out = {
        "null_decl": rate([x["null_fp"] for x in rows], [x["n_null"] for x in rows]),
        "plac_decl": rate([x["plac_decl"] for x in rows], [x["n_plac"] for x in rows]),
    }
    has_conf = any(x["n_conf"] for x in rows)
    if has_conf:
        out["conf_decl"] = rate([x["conf_decl"] for x in rows], [x["n_conf"] for x in rows])
    if has_p:
        out["null_raw05"] = rate([x["raw_null"][0] for x in rows], [x["raw_null"][1] for x in rows])
        out["plac_raw05"] = rate([x["raw_plac"][0] for x in rows], [x["raw_plac"][1] for x in rows])
        if has_conf:
            out["conf_raw05"] = rate(
                [x["raw_conf"][0] for x in rows], [x["raw_conf"][1] for x in rows]
            )
    return out


def _valid_flag(rates: dict[str, Any], n_rows: int, has_p: bool) -> str:
    if n_rows < MIN_FLAG_SEEDS:
        return "few_seeds"
    keys = ("null_raw05", "plac_raw05") if has_p else ("null_decl", "plac_decl")
    return "ABOVE" if any(rates[k]["above"] for k in (*keys, "null_decl", "plac_decl")) else "ok"


def summarise_cell(rows: list[dict[str, Any]], has_p: bool) -> dict[str, Any]:
    """Rates, validity flag and power summary of one cell's per-seed rows."""
    out: dict[str, Any] = {"n_seeds": len(rows), **_rates(rows, has_p)}
    out["valid_flag"] = _valid_flag(out, len(rows), has_p)
    rec = [x["recall"] for x in rows]
    defined = [x for x in rec if x is not None]
    out.update(
        recall=_mean(rec),
        recall_sd=float(np.nanstd(defined, ddof=1)) if len(defined) > 1 else None,
        fdp=_mean([x["fdp"] for x in rows]),
        sign_acc=_mean([x["sign_acc"] for x in rows]),
        n_declared=_mean([x["n_declared"] for x in rows]),
        nt_true=int(sum(x["nt_true"] for x in rows)),
        nt_null=int(sum(x["nt_null"] for x in rows)),
        per_seed={
            str(x["seed"]): {"recall": x["recall"], "fdp": x["fdp"], "sign_acc": x["sign_acc"]}
            for x in rows
        },
    )
    return out


def _new_group() -> dict[str, list[Record]]:
    return {"tune": [], "measure": [], "infeasible": [], "error": []}


class _Aggregator:
    def __init__(self, spec: Spec) -> None:
        self.arms = spec["arms"]
        self.truths: dict[tuple[str, str], Truth] = {}
        self.cells: dict[str, Any] = {}
        self.cost: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cpu": [], "wall": [], "rss": [], "gen": [], "infeasible": [], "errors": 0}
        )

    def _truth(self, u: Unit) -> Truth:
        return self.truths.setdefault((u.world, u.regime), truth_for(u.world, u.regime))

    def _entry(self, u: Unit, g: dict[str, list[Record]]) -> dict[str, Any]:
        mode = self.arms[u.arm]["declare"] if u.arm in self.arms else "by"
        truth = self._truth(u)
        tau = placebo_tau([to_result(r) for r in g["tune"]]) if g["tune"] else None
        rows = [seed_row(r, to_result(r), truth) for r in g["measure"]] if mode == "by" else []
        rows_tau = (
            [seed_row(r, apply_threshold(to_result(r), tau), truth) for r in g["measure"]]
            if tau is not None
            else []
        )
        entry: dict[str, Any] = {
            "arm": u.arm,
            "world": u.world,
            "regime": u.regime,
            "lam": u.lam,
            "n": u.n,
            "kappa": u.kappa,
            "declare": mode,
            "tau": tau,
            "n_tune": len(g["tune"]),
            "n_measure": len(g["measure"]),
            "n_infeasible": len(g["infeasible"]),
            "n_error": len(g["error"]),
        }
        primary = rows if mode == "by" else rows_tau
        if primary:
            entry["primary"] = summarise_cell(primary, mode == "by")
        elif mode == "tau" and g["measure"] and tau is None:
            entry["primary"] = None
            entry["note"] = "untuned: no tune-seed records for this cell (tau undefined)"
        if mode == "by" and rows_tau:
            entry["secondary_tau"] = summarise_cell(rows_tau, False)
        return entry

    def _add_cost(self, u: Unit, g: dict[str, list[Record]]) -> None:
        c = self.cost[f"{u.arm}|{u.world}|{u.regime}|n{u.n}"]
        for r in g["tune"] + g["measure"]:
            c["cpu"].append(r.get("cpu_s"))
            c["wall"].append(r.get("child_wall_s") or r.get("wall_s"))
            c["rss"].append(r.get("peak_rss_mb"))
            c["gen"].append(r.get("gen_cpu_s"))
        for r in g["infeasible"]:
            c["infeasible"].append(
                {
                    "seed": r["job"]["seed"],
                    "cpu_s": r.get("cpu_s"),
                    "cost": r.get("cost"),
                    "cost_kind": r.get("cost_kind"),
                    "measured": r.get("cost") is not None,
                    "reason": r.get("reason"),
                }
            )
        c["errors"] += len(g["error"])

    def finish(self, cell: str, u: Unit, g: dict[str, list[Record]]) -> None:
        self.cells[cell] = self._entry(u, g)
        self._add_cost(u, g)

    def cost_out(self) -> dict[str, Any]:
        out = {}
        for k, v in sorted(self.cost.items()):
            cpu = [x for x in v["cpu"] if x is not None]
            wall = [x for x in v["wall"] if x is not None]
            out[k] = {
                "n_runs": len(cpu),
                "cpu_s_mean": _mean(cpu),
                "cpu_s_max": max(cpu) if cpu else None,
                "wall_s_mean": _mean(wall),
                "wall_s_max": max(wall) if wall else None,
                "peak_rss_mb_max": max([x for x in v["rss"] if x is not None], default=None),
                "gen_cpu_s_mean": _mean(v["gen"]),
                "infeasible": v["infeasible"],
                "errors": v["errors"],
            }
        return out


def _bucket(r: Record, u: Unit) -> str:
    st = record_status(r)
    return u.role if st == "ok" else st


def _stream(agg: _Aggregator, records: Iterable[Record]) -> None:
    done: set[str] = set()
    cur: str | None = None
    cu: Unit | None = None
    cg = _new_group()
    for r in records:
        u = record_unit(r)
        if u.cell != cur:
            if cur is not None and cu is not None:
                agg.finish(cur, cu, cg)
                done.add(cur)
            if u.cell in done:
                raise ValueError(
                    f"aggregate(stream_sorted=True): input not sorted by cell ({u.cell} again)"
                )
            cur, cu, cg = u.cell, u, _new_group()
        cg[_bucket(r, u)].append(r)
    if cur is not None and cu is not None:
        agg.finish(cur, cu, cg)


def _in_memory(agg: _Aggregator, records: Iterable[Record]) -> None:
    by_cell: dict[str, dict[str, list[Record]]] = defaultdict(_new_group)
    units: dict[str, Unit] = {}
    for r in records:
        u = record_unit(r)
        units[u.cell] = u
        by_cell[u.cell][_bucket(r, u)].append(r)
    for c in sorted(by_cell):
        agg.finish(c, units[c], by_cell[c])


def aggregate(records: Iterable[Record], spec: Spec, stream_sorted: bool = False) -> dict[str, Any]:
    """Validity, power and cost per cell of merged records (one per key).

    Args:
        records: Merged records.
        spec: The campaign spec (arm declaration modes).
        stream_sorted: The records come sorted by key, so each cell is finished when the next
            one starts (memory: one cell).

    Returns:
        `{spec_name, campaign, alpha, budget_cpu_s, cells, cost}`.
    """
    agg = _Aggregator(spec)
    (_stream if stream_sorted else _in_memory)(agg, records)
    return {
        "spec_name": spec.get("name"),
        "campaign": CAMPAIGN_VERSION,
        "alpha": ALPHA,
        "budget_cpu_s": spec.get("budget_cpu_s"),
        "cells": agg.cells,
        "cost": agg.cost_out(),
    }

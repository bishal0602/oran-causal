"""Descriptive tables of the bridging study (protocol EXP_B section 5).

Per (arm, slice kind, rule), pooled over the family datasets of the kind's slices: recall on the
GT-TRUE hypotheses, rejection rates on GT-NULL (an equivalence label), on set B (GT-NULL cells
whose GT 95 % CI contains 0, the sharp-null proxy) and on the `P_placebo` column, sign accuracy
and the AUROC of the score, TRUE vs NULL. On the placebo logs every real hypothesis is an exact
sharp null (`plac_log`). Rules: `by` (the adapter's BY declaration, p-value arms), `raw` (p <=
.05, p-value arms), `tau` (score above the conformal placebo threshold: leave-one-slice-out over
the kind's other slices of the same family; pooled and placebo logs use the dataset's own
placebo scores, `in-sample`) and `fixed` (score >= the arm's `fixed_threshold`). A not-testable
edge is never declared. CIs are seed-cluster bootstraps over slices.
"""

import dataclasses
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from oran_causal.analysis.bridging.plan import (
    PLACEBO,
    Hyp,
    Spec,
    hyp_of,
    kinds_of,
    load_gt,
    seed_index,
    spec_families,
)
from oran_causal.analysis.controlled.validity import cluster_ci
from oran_causal.analysis.records import Record, arm_of, iter_records, status_of
from oran_causal.analysis.system.report_data import FROZEN_NEWLINE
from oran_causal.analysis.system.stats import dumps
from oran_causal.sim.system.decision.vocab import FAMILIES

SCHEMA = "xm-expb/1"
ALPHA = 0.05
TAU_LEVEL = 0.05
TABLES_JSON = "exp_b_tables.json"
TABLES_MD = "EXP_B_TABLES.md"
RATE_KEYS = ("rec", "null", "sharp", "plac", "plac_log")
POOLED_KINDS = ("pooled", "placebo")

type Gt = Mapping[Hyp, Mapping[str, Any]]
type Recs = dict[tuple[str, str], dict[str, Record]]
type Taus = dict[tuple[str, str, str], tuple[float | None, str]]


def conformal_tau(scores: Iterable[float | None], level: float = TAU_LEVEL) -> float | None:
    """The ceil((M + 1)(1 - level))-th smallest of M finite placebo scores (largest if beyond M)."""
    v = np.sort(np.asarray([x for x in scores if x is not None and np.isfinite(x)], float))
    if not len(v):
        return None
    k = math.ceil((len(v) + 1) * (1 - level))
    return float(v[min(k, len(v)) - 1])


def _finite(xs: Iterable[float | None]) -> list[float]:
    return [x for x in xs if x is not None and np.isfinite(x)]


def auroc(pos: Iterable[float | None], neg: Iterable[float | None]) -> float | None:
    """Mann-Whitney AUROC of the finite scores, ties counted half; None if a side is empty."""
    p, n = _finite(pos), _finite(neg)
    if not p or not n:
        return None
    r = rankdata(np.r_[p, n])
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def sharp_proxy(gt: Gt) -> set[Hyp]:
    """Set B: GT-NULL cells whose GT 95 % CI contains 0."""
    return {h for h, c in gt.items() if c["status"] == "NULL" and c["ci"][0] <= 0.0 <= c["ci"][1]}


def gt_counts(gt: Gt) -> dict[str, int]:
    """Number of GT cells per status, in first-seen order."""
    c: dict[str, int] = defaultdict(int)
    for v in gt.values():
        c[v["status"]] += 1
    return dict(c)


def _score(e: Mapping[str, Any]) -> float | None:
    x = e.get("score")
    return float(x) if x is not None and np.isfinite(x) else None


def ok_records(spec: Spec, merged: Iterable[Record]) -> tuple[Recs, dict[str, dict[str, int]]]:
    """Ok records per (arm, slice name) and family, and the status counts per arm (file order)."""
    idx = seed_index(spec)
    recs: Recs = defaultdict(dict)
    status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in merged:
        s, f = idx[int(r["job"]["seed"])]
        arm, st = arm_of(r), status_of(r)
        status[arm][st] += 1
        if st == "ok":
            recs[(arm, s.name)][f] = r
    return recs, {a: dict(v) for a, v in status.items()}


def _placebo_scores(recs: Recs, arm: str, names: Sequence[str], family: str) -> list[float | None]:
    edges = (e for n in names for e in recs.get((arm, n), {}).get(family, {}).get("edges", []))
    return [_score(e) for e in edges if e["source"] == PLACEBO]


def taus(spec: Spec, recs: Recs, kinds: Mapping[str, list[str]]) -> Taus:
    """(tau, how) per (arm, slice, family): `loso` (sized kinds), else `in-sample`."""
    out: Taus = {}
    for arm in spec["arms"]:
        for kind, names in kinds.items():
            for name in names:
                for f in spec_families(spec):
                    others = [n for n in names if n != name] if kind not in POOLED_KINDS else [name]
                    tau = conformal_tau(_placebo_scores(recs, arm, others, f))
                    out[(arm, name, f)] = (tau, "loso" if others != [name] else "in-sample")
    return out


def rules_of(arm: Mapping[str, Any]) -> list[str]:
    """Declaration rules of an arm: by / raw / tau for BY arms, tau otherwise, plus fixed."""
    rules = ["by", "raw", "tau"] if arm["declare"] == "by" else ["tau"]
    return rules + (["fixed"] if arm.get("fixed_threshold") is not None else [])


def _declared(e: Mapping[str, Any], rule: str, arm: Mapping[str, Any], tau: float | None) -> bool:
    sc, p = _score(e), e.get("p")
    if rule == "by":
        return bool(e.get("declared"))
    if rule == "raw":
        return p is not None and float(p) <= ALPHA
    if rule == "fixed":
        return sc is not None and sc >= float(arm["fixed_threshold"])
    return sc is not None and tau is not None and sc > tau


@dataclasses.dataclass
class _Cell:
    per: dict[str, list[list[int]]] = dataclasses.field(
        default_factory=lambda: {k: [[], []] for k in RATE_KEYS}
    )
    sign_ok: int = 0
    sign_n: int = 0
    pos: list[float | None] = dataclasses.field(default_factory=list)
    neg: list[float | None] = dataclasses.field(default_factory=list)
    n_ds: int = 0


@dataclasses.dataclass(frozen=True)
class _Ctx:
    gt: Gt
    sharp: set[Hyp]
    kind: str
    rule: str
    arm: Mapping[str, Any]


def _tally(
    h: dict[str, list[int]], cell: _Cell, ctx: _Ctx, e: Mapping[str, Any], dec: bool
) -> None:
    hy = hyp_of(e["source"], e["target"])
    if hy is None or ctx.kind == "placebo":
        key = "plac_log" if hy else "plac"
        h[key][0] += dec
        h[key][1] += 1
        return
    c = ctx.gt[hy]
    st = c["status"]
    if st == "TRUE":
        h["rec"][0] += dec
        h["rec"][1] += 1
        cell.pos.append(_score(e))
        if dec and e.get("sign"):
            cell.sign_n += 1
            cell.sign_ok += int(np.sign(e["sign"]) == c["sign"])
    elif st == "NULL":
        h["null"][0] += dec
        h["null"][1] += 1
        cell.neg.append(_score(e))
        if hy in ctx.sharp:
            h["sharp"][0] += dec
            h["sharp"][1] += 1


def _dataset(cell: _Cell, ctx: _Ctx, r: Record, tau: float | None, h: dict[str, list[int]]) -> None:
    nt = set(map(str, (r.get("notes") or {}).get("not_testable_edges", []) or []))
    for e in r.get("edges", []):
        testable = f"{e['source']}->{e['target']}" not in nt
        _tally(h, cell, ctx, e, testable and _declared(e, ctx.rule, ctx.arm, tau))


def _summary(cell: _Cell, names: Sequence[str], ctx: _Ctx, tau_rules: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_slices": len(names), "n_datasets": cell.n_ds}
    for k, (hits, cnt) in cell.per.items():
        if sum(cnt):
            ci = cluster_ci(np.array(hits), np.array(cnt)) if len(names) > 1 else None
            out[k] = {
                "rate": sum(hits) / sum(cnt),
                "hits": int(sum(hits)),
                "n": int(sum(cnt)),
                "ci": ci,
            }
    out["sign_acc"] = cell.sign_ok / cell.sign_n if cell.sign_n else None
    out["sign_n"] = cell.sign_n
    if ctx.rule in ("tau", "by") and ctx.kind != "placebo":
        out["auroc_true_vs_null"] = auroc(cell.pos, cell.neg)
    if ctx.rule == "tau":
        out["tau_rule"] = tau_rules
    return out


def table_cell(
    spec: Spec, recs: Recs, tau_of: Taus, ctx: _Ctx, arm: str, names: Sequence[str]
) -> dict[str, Any]:
    """One (arm, kind, rule) row: rates with slice-cluster CIs, sign accuracy, AUROC, tau rule."""
    cell = _Cell()
    for name in names:
        h = {k: [0, 0] for k in RATE_KEYS}
        for f in spec_families(spec):
            r = recs.get((arm, name), {}).get(f)
            if r is None:
                continue
            cell.n_ds += 1
            _dataset(cell, ctx, r, tau_of[(arm, name, f)][0], h)
        for k in cell.per:
            cell.per[k][0].append(h[k][0])
            cell.per[k][1].append(h[k][1])
    how = sorted({tau_of[(arm, n, f)][1] for n in names for f in spec_families(spec)})
    return _summary(cell, names, ctx, how)


def v4_reference(
    analysis: Mapping[str, Any], gt: Gt, kinds: Mapping[str, list[str]]
) -> dict[str, Any]:
    """The frozen v4 primary (loadsp_c + wby1s) per kind from its stored hypothesis tables."""
    out = {}
    for kind, names in kinds.items():
        tp = nt = fp = nn = 0
        for name in names:
            if kind == "placebo":
                hyp = analysis["placebo_hyp"]
            else:
                hyp = analysis["slices_result"].get(name, {}).get("hyp", {})
            for h, o in hyp.items():
                dec = bool((o.get("primary") or {}).get("declared"))
                st = "NULL" if kind == "placebo" else gt[tuple(h.split("|"))]["status"]
                if st == "TRUE":
                    tp, nt = tp + dec, nt + 1
                elif st == "NULL":
                    fp, nn = fp + dec, nn + 1
        out[kind] = {
            "recall": tp / nt if nt else None,
            "null_rate": fp / nn if nn else None,
            "n_true": nt,
            "n_null": nn,
        }
    return out


def build(
    spec: Spec,
    merged: Iterable[Record],
    gt: Gt,
    v4_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The tables document (`exp_b_tables.json`) of merged records.

    Args:
        spec: The bridging spec (arms with `declare` / `fixed_threshold`, slices, families).
        merged: Merged records, in file order.
        gt: `plan.load_gt` of the v4 GT.
        v4_analysis: The stored v4 analysis, for the frozen-primary reference row.

    Returns:
        `{"schema", "gt", "status", "cells", "v4_frozen_reference"}`.
    """
    sharp = sharp_proxy(gt)
    recs, status = ok_records(spec, merged)
    kinds = kinds_of(spec)
    tau_of = taus(spec, recs, kinds)
    rows = {}
    for arm, d in spec["arms"].items():
        for kind, names in kinds.items():
            for rule in rules_of(d):
                ctx = _Ctx(gt, sharp, kind, rule, d)
                rows[f"{arm}|{kind}|{rule}"] = table_cell(spec, recs, tau_of, ctx, arm, names)
    ref = v4_reference(v4_analysis, gt, kinds) if v4_analysis else None
    return {
        "schema": SCHEMA,
        "gt": {"counts": gt_counts(gt), "sharp_proxy": sorted("|".join(h) for h in sharp)},
        "status": status,
        "cells": rows,
        "v4_frozen_reference": ref,
    }


def _rate_text(c: Mapping[str, Any], k: str) -> str:
    v = c.get(k)
    if not v:
        return ""
    ci = v.get("ci")
    band = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci and ci[0] is not None else ""
    return f"{v['rate']:.3f} ({v['hits']}/{v['n']}){band}"


def _round(x: float | None) -> str:
    return "" if x is None else str(round(x, 3))


def _md_head(t: Mapping[str, Any], nf: int) -> list[str]:
    return [
        "# Experiment B tables (descriptive; frozen E6 v4 data)",
        "",
        f"GT: {t['gt']['counts']}; sharp-null proxy (set B): {len(t['gt']['sharp_proxy'])} hyp.",
        "",
        "Each rate: value (hits / n) [cluster-bootstrap CI over slices; few slices give narrow or "
        "zero-width CIs].",
        f"`ds` = datasets with an ok record out of slices x {nf} families; `*` = partial coverage "
        "(mscr runs only at",
        "n <= 1000, i.e. sleep; cdl units over the 7200 s budget are infeasible), so that row is "
        "not comparable with",
        "full-coverage rows.",
        "",
        "| arm | kind | rule | ds | recall TRUE | GT-NULL rate | set-B rate | P_placebo rate "
        "| placebo-log rate | sign acc | AUROC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]


def markdown(t: Mapping[str, Any]) -> str:
    """`EXP_B_TABLES.md` of a tables document."""
    nf = len(FAMILIES)
    lines = _md_head(t, nf)
    for key, c in t["cells"].items():
        arm, kind, rule = key.split("|")
        want = c["n_slices"] * nf
        ds = f"{c['n_datasets']}/{want}" + ("*" if c["n_datasets"] < want else "")
        rates = " | ".join(_rate_text(c, k) for k in RATE_KEYS)
        lines.append(
            f"| {arm} | {kind} | {rule} | {ds} | {rates} | "
            f"{_round(c.get('sign_acc'))} | {_round(c.get('auroc_true_vs_null'))} |"
        )
    if t.get("v4_frozen_reference"):
        ref = json.dumps(t["v4_frozen_reference"])
        lines += ["", f"Frozen v4 primary (loadsp_c + wby1s, stored tables): {ref}"]
    return "\n".join(lines) + "\n"


def write(t: Mapping[str, Any], out_dir: str | Path, newline: str = FROZEN_NEWLINE) -> None:
    """Writes `exp_b_tables.json` (`newline` line ends; stored: CRLF) and the LF markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / TABLES_JSON).write_text(dumps(t), encoding="utf-8", newline=newline)
    (out / TABLES_MD).write_text(markdown(t), encoding="utf-8", newline="\n")


def tables(
    spec: Spec,
    merged: str | Path,
    gt: str | Path,
    out_dir: str | Path | None = None,
    v4_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the tables of a merged record file and writes them to `out_dir` if given."""
    t = build(spec, iter_records(merged), load_gt(gt), v4_analysis)
    if out_dir is not None:
        write(t, out_dir)
    return t

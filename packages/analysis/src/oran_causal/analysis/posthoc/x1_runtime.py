"""X1: per-method runtime table of Study 4 (cost per unit in Kaggle reference CPU-seconds).

Reads campaign record files (`res_*.jsonl` shards or `merged.jsonl.gz`) and builds, per (arm, n),
the CPU-seconds per unit (record field `cpu_s`, the process CPU time of `method.run` on one
thread, data generation excluded) as mean, median and maximum on the reference host, the peak RSS
and the feasibility against the 7200 CPU-s budget by the T3 rule: the first n whose maximum unit
cost exceeds the budget makes n and every larger n infeasible.

A unit run on another host type is converted as cost x f with f = sum of the reference cost over
the sum of the host cost on the same calibration units (`calibrate`). Without a factor file a
provisional within-cell estimate is used (`observational_factors`). A host type is
`platform|cpu model` (plus `|gpu` on GPU sessions). GPU arms (spec `budget_wall_s`) are read in
wall-seconds, apart from the CPU arms.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.records import iter_records as _iter_file

VERSION = "xm-exp-c/1"
BUDGET_CPU_S = 7200.0
REF_HOST = "kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz"
N_GRID = (500, 1000, 4000, 8000, 24000)
CALIB_SEEDS = (3_000_000, 3_000_001, 3_000_002)
PRIMARY_KAPPA = 0.25
MULTI_THREAD_RATIO = 1.15
CONTENTION_RATIO = 1.3

Slim = dict[str, Any]


def host_type(r: Mapping[str, Any]) -> str:
    """Host type of a record: `platform|cpu model[|gpu]`."""
    h = r.get("host") or {}
    parts = [str(h.get("platform") or "unknown"), str(h.get("cpu_model") or "unknown")]
    gpu = h.get("gpu")
    if gpu:
        parts.append(str(gpu[0] if isinstance(gpu, list) else gpu).split(",")[0].strip())
    return "|".join(parts)


def _arm(r: Mapping[str, Any]) -> str:
    return r.get("arm") or str(r.get("key", "?")).split("|", 1)[0]


def procs_per_vcpu(r: Mapping[str, Any]) -> float | None:
    """Concurrent campaign or timing processes per vCPU at the unit's start, else None."""
    ld = ((r.get("load") or {}).get("start")) or {}
    k = ld.get("campaign_procs") if ld.get("campaign_procs") is not None else ld.get("timing_procs")
    q = ld.get("cpu_quota")
    return float(k) / float(q) if k is not None and q else None


def slim(r: Mapping[str, Any]) -> Slim:
    """The fields the runtime table needs; records also carry edges and notes."""
    j = r.get("job") or {}
    st = r.get("status") or ("ok" if r.get("error") is None else "error")
    wall = r.get("child_wall_s") if r.get("child_wall_s") is not None else r.get("wall_s")
    cost = None
    if st == "infeasible":
        cost = r.get("cost") if r.get("cost") is not None else r.get("cpu_s")
    return {
        "key": r.get("key"),
        "arm": _arm(r),
        "method": r.get("method"),
        "world": j.get("world"),
        "regime": j.get("regime"),
        "lam": j.get("lam"),
        "kappa": j.get("kappa"),
        "n": j.get("n"),
        "seed": j.get("seed"),
        "role": r.get("role"),
        "status": st,
        "cpu_s": r.get("cpu_s"),
        "wall_s": wall,
        "peak_rss_mb": r.get("peak_rss_mb"),
        "rss_scope": r.get("rss_scope"),
        "gen_cpu_s": r.get("gen_cpu_s"),
        "gpu_s": r.get("gpu_s"),
        "host": host_type(r),
        "cost": cost,
        "cost_kind": r.get("cost_kind"),
        "spec_name": r.get("spec_name"),
        "commit": (r.get("code") or {}).get("commit"),
        "dataset_sha256": r.get("dataset_sha256"),
        "procs_per_vcpu": procs_per_vcpu(r),
        "node": (r.get("host") or {}).get("node"),
    }


def file_sha256(path: str | Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(
    paths: Sequence[str | Path], labels: Mapping[str, str] | None = None
) -> tuple[list[Slim], dict[str, Any]]:
    """Loads record files, one record per key.

    The first ok record of a key wins and an ok record replaces an earlier non-ok one. Every key
    that appears more than once is kept in `info["copies"]` (cross-host pairs).

    Args:
        paths: Record files.
        labels: Optional absolute path to recorded path map, to reproduce a frozen table whose
            inputs were read from another location.

    Returns:
        The unique slim records and the load info (files with line counts and sha256, line,
        duplicate and conflict counts, copies).
    """
    labels = labels or {}
    by_key: dict[str, Slim] = {}
    copies: dict[str, list[Slim]] = defaultdict(list)
    info: dict[str, Any] = {"files": [], "lines": 0, "duplicates": 0, "conflicts": 0}
    for p in paths:
        n0 = info["lines"]
        for r in _iter_file(p):
            info["lines"] += 1
            s = slim(r)
            k = s["key"]
            copies[k].append(s)
            if k not in by_key:
                by_key[k] = s
                continue
            info["duplicates"] += 1
            old = by_key[k]
            if old["status"] != "ok" and s["status"] == "ok":
                by_key[k] = s
            elif old["status"] == s["status"] == "ok" and old["host"] != s["host"]:
                info["conflicts"] += 1
        path = str(Path(p).absolute())
        info["files"].append(
            {
                "path": labels.get(path, path),
                "sha256": file_sha256(p),
                "lines": info["lines"] - n0,
            }
        )
    info["copies"] = {k: v for k, v in copies.items() if len(v) > 1}
    return list(by_key.values()), info


def load_specs(paths: Sequence[str | Path] | None) -> dict[str, dict]:
    """Arm specs of the given campaign spec files."""
    arms: dict[str, dict] = {}
    for p in paths or []:
        arms.update(json.loads(Path(p).read_text(encoding="utf-8")).get("arms", {}))
    return arms


def cost_field(arm_spec: Mapping[str, Any] | None) -> str:
    """`wall_s` for GPU arms (spec `budget_wall_s`), else `cpu_s`."""
    return "wall_s" if (arm_spec or {}).get("budget_wall_s") else "cpu_s"


def unit_cost(s: Mapping[str, Any], fld: str) -> float | None:
    """Cost of one unit in the arm's cost field; a measured over-budget unit carries its `cost`."""
    if s["status"] == "infeasible":
        kind = s["cost_kind"]
        if s["cost"] is not None and (kind is None or ("wall" in str(kind)) == (fld == "wall_s")):
            return float(s["cost"])
        return None
    v = s.get(fld)
    return None if v is None else float(v)


def _cell(s: Mapping[str, Any]) -> tuple:
    return (s["arm"], s["world"], s["regime"], s["lam"], s["kappa"], s["n"])


def observational_factors(
    recs: Sequence[Slim], ref: str, fields: Mapping[str, str]
) -> dict[tuple, dict]:
    """Provisional factor per (arm, host) from cells measured on both the reference and the host."""
    g: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in recs:
        if s["status"] == "ok":
            c = unit_cost(s, fields.get(s["arm"], "cpu_s"))
            if c is not None:
                g[_cell(s)][s["host"]].append(c)
    num: dict[tuple, float] = defaultdict(float)
    den: dict[tuple, float] = defaultdict(float)
    ratios: dict[tuple, list[float]] = defaultdict(list)
    units: dict[tuple, int] = defaultdict(int)
    for cell, hosts in g.items():
        if ref not in hosts:
            continue
        mref = float(np.mean(hosts[ref]))
        for h, xs in hosts.items():
            if h == ref:
                continue
            mh, k = float(np.mean(xs)), len(xs)
            num[(cell[0], h)] += k * mref
            den[(cell[0], h)] += k * mh
            units[(cell[0], h)] += k
            if mh > 0:
                ratios[(cell[0], h)].append(mref / mh)
    out = {}
    for key, total in num.items():
        if den[key] > 0:
            r = ratios[key]
            out[key] = {
                "f": total / den[key],
                "f_hi": max([total / den[key], *r]),
                "source": "observational",
                "n_cells": len(r),
                "n_host_units": units[key],
                "cell_ratio_q25_q50_q75": (
                    [float(x) for x in np.percentile(r, [25, 50, 75])] if r else None
                ),
            }
    return out


def load_factor_file(path: str | Path | None) -> dict[tuple, dict]:
    """Reads a `calibrate` output: `{(arm, host): factor}`; arm `*` serves every arm."""
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    src = d.get("source", "calib")
    return {
        (a, h): {**v, "source": v.get("source", src)}
        for a, hs in d.get("factors", {}).items()
        for h, v in hs.items()
    }


def resolve_factor(
    arm: str, host: str, ref: str, explicit: Mapping[tuple, dict], obs: Mapping[tuple, dict]
) -> dict | None:
    """Factor for a unit: 1 on the reference, else the explicit one, else observational."""
    if host == ref:
        return {"f": 1.0, "f_hi": 1.0, "source": "reference"}
    return explicit.get((arm, host)) or explicit.get(("*", host)) or obs.get((arm, host))


def _stats(xs: Sequence[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(xs)),
        "median": float(np.median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _mean_or_none(xs: Sequence[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


@dataclass
class _Row:
    """Accumulator of the units of one (arm, n)."""

    cost: list[float] = field(default_factory=list)
    raw: list[float] = field(default_factory=list)
    rss: list[float] = field(default_factory=list)
    wall: list[float] = field(default_factory=list)
    gen: list[float] = field(default_factory=list)
    gpu: list[float] = field(default_factory=list)
    hosts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    unconverted: int = 0
    infeasible: list[dict] = field(default_factory=list)
    errors: int = 0
    no_cost: int = 0
    multi_thread: int = 0
    scopes: set = field(default_factory=set)
    fsrc: set = field(default_factory=set)
    contended: int = 0
    wc: list[float] = field(default_factory=list)
    hi: list[float] = field(default_factory=list)
    ppv: list[float] = field(default_factory=list)
    by_role: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    max_key: dict | None = None
    methods: set = field(default_factory=set)
    roles: dict[Any, int] = field(default_factory=lambda: defaultdict(int))

    def add_measures(self, s: Slim) -> None:
        """Adds the RSS, wall, generation, GPU and thread-contention measures of an ok unit."""
        if s["peak_rss_mb"] is not None:
            self.rss.append(float(s["peak_rss_mb"]))
            self.scopes.add(s["rss_scope"])
        for src, dst in (("wall_s", self.wall), ("gen_cpu_s", self.gen), ("gpu_s", self.gpu)):
            if s[src] is not None:
                dst.append(float(s[src]))
        if s["cpu_s"] is not None and s["wall_s"]:
            self.multi_thread += s["cpu_s"] > MULTI_THREAD_RATIO * s["wall_s"] + 0.05
            self.contended += s["wall_s"] > CONTENTION_RATIO * s["cpu_s"] + 0.05
            if s["cpu_s"] > 0:
                self.wc.append(s["wall_s"] / s["cpu_s"])


def _accumulate(
    recs: Sequence[Slim],
    fields: Mapping[str, str],
    ref: str,
    explicit: Mapping[tuple, dict],
    obs: Mapping[tuple, dict],
) -> tuple[dict[tuple, _Row], dict[tuple, dict], dict[tuple, list[float]]]:
    used: dict[tuple, dict] = {}
    rows: dict[tuple, _Row] = defaultdict(_Row)
    wr: dict[tuple, list[float]] = defaultdict(list)
    for s in recs:
        a, n = s["arm"], s["n"]
        row = rows[(a, n)]
        row.methods.add(s["method"] or "?")
        if s["status"] == "error":
            row.errors += 1
            continue
        c = unit_cost(s, fields[a])
        if c is None:
            row.no_cost += 1
            continue
        f = resolve_factor(a, s["host"], ref, explicit, obs)
        if f is not None:
            used[(a, s["host"])] = f
            cr = c * f["f"]
            chi = c * float(f.get("f_hi") or f["f"])
            if s["host"] != ref:
                row.fsrc.add(f["source"])
        else:
            row.unconverted += 1
            cr = chi = c
        row.hosts[s["host"]] += 1
        row.roles[s["role"]] += 1
        if s["status"] == "infeasible":
            row.infeasible.append(
                {"key": s["key"], "cost_raw": c, "cost_ref": cr, "host": s["host"]}
            )
        else:
            row.add_measures(s)
        row.raw.append(c)
        row.hi.append(chi)
        row.by_role[str(s["role"])].append(cr)
        if s["procs_per_vcpu"] is not None:
            row.ppv.append(s["procs_per_vcpu"])
        if not row.cost or cr > max(row.cost):
            row.max_key = {"key": s["key"], "host": s["host"], "cost_raw": c}
        row.cost.append(cr)
        wr[(a, s["world"], s["regime"], n)].append(cr)
    return rows, used, wr


def _table_row(
    a: str,
    n: Any,
    v: _Row,
    ref: str,
    fld: str,
    budget: float,
    arm_spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    nref = v.hosts.get(ref, 0)
    tot = sum(v.hosts.values())
    return {
        "arm": a,
        "method": "/".join(sorted(v.methods)),
        "n": n,
        "cost_field": fld,
        "budget": float((arm_spec or {}).get("budget_wall_s") or budget),
        "n_units": tot,
        "n_ok": tot - len(v.infeasible),
        "n_infeasible": len(v.infeasible),
        "n_errors": v.errors,
        "n_no_cost": v.no_cost,
        "roles": dict(v.roles),
        "share_non_ref": (1 - nref / tot) if tot else None,
        "hosts": dict(v.hosts),
        "n_unconverted": v.unconverted,
        "factor_sources": sorted(v.fsrc),
        "cost_ref": _stats(v.cost),
        "cost_raw": _stats(v.raw),
        "max_unit": v.max_key,
        "cost_ref_hi_max": max(v.hi) if v.hi else None,
        "cost_ref_by_role": {k: _stats(x) for k, x in sorted(v.by_role.items())},
        "procs_per_vcpu": (
            {"median": float(np.median(v.ppv)), "max": float(max(v.ppv))} if v.ppv else None
        ),
        "peak_rss_mb": {
            "median": float(np.median(v.rss)) if v.rss else None,
            "max": max(v.rss) if v.rss else None,
            "scope": sorted(map(str, v.scopes)),
        },
        "wall_s_mean": _mean_or_none(v.wall),
        "gen_cpu_s_mean": _mean_or_none(v.gen),
        "gpu_s_mean": _mean_or_none(v.gpu),
        "n_multi_thread": v.multi_thread,
        "n_contended": v.contended,
        "wall_cpu_ratio_median": float(np.median(v.wc)) if v.wc else None,
        "infeasible_units": v.infeasible[:10],
    }


def build(
    recs: Sequence[Slim],
    info: Mapping[str, Any],
    *,
    ref: str = REF_HOST,
    budget: float = BUDGET_CPU_S,
    arm_specs: Mapping[str, Any] | None = None,
    factors: Mapping[tuple, dict] | None = None,
    observational: bool = True,
    n_grid: tuple[int, ...] = N_GRID,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """The runtime table with its T3 feasibility.

    Args:
        recs: Slim records from `load`.
        info: Load info from `load`.
        ref: Reference host type.
        budget: CPU-seconds budget per unit.
        arm_specs: Arm specs, for GPU arms and the spec `max_n`.
        factors: Explicit speed factors from `load_factor_file`.
        observational: Use provisional within-cell factors where no explicit one exists.
        n_grid: The pre-registered n grid.
        generated_utc: Timestamp to record; the current time when None.

    Returns:
        The table, feasibility, factors and inputs.
    """
    arm_specs = arm_specs or {}
    arms = sorted({s["arm"] for s in recs})
    fields = {a: cost_field(arm_specs.get(a)) for a in arms}
    obs = observational_factors(recs, ref, fields) if observational else {}
    rows, used, wr = _accumulate(recs, fields, ref, factors or {}, obs)
    ordered = sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0))
    table = [_table_row(a, n, v, ref, fields[a], budget, arm_specs.get(a)) for (a, n), v in ordered]
    feas = t3(table, n_grid, {a: (d or {}).get("max_n") for a, d in arm_specs.items()})
    for row in table:
        row["t3"] = feas[row["arm"]]["by_n"].get(str(row["n"]))
    by_wr = [
        {"arm": k[0], "world": k[1], "regime": k[2], "n": k[3], "n_units": len(v), **_stats(v)}
        for k, v in sorted(
            wr.items(), key=lambda kv: (kv[0][0], kv[0][3] or 0, str(kv[0][1]), str(kv[0][2]))
        )
    ]
    stamp = generated_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "version": VERSION,
        "generated_utc": stamp,
        "ref_host": ref,
        "budget_cpu_s": budget,
        "n_grid": list(n_grid),
        "inputs": {k: v for k, v in info.items() if k != "copies"},
        "spec_names": sorted({str(s["spec_name"]) for s in recs}),
        "commits": sorted({str(s["commit"]) for s in recs}),
        "hosts": host_counts(recs),
        "factors": [{"arm": a, "host": h, **f} for (a, h), f in sorted(used.items()) if h != ref],
        "observational_factors_all": [
            {"arm": a, "host": h, **f} for (a, h), f in sorted(obs.items())
        ],
        "table": table,
        "feasibility": feas,
        "by_world_regime": by_wr,
        "spec_t3_fields": spec_fields(feas),
    }


def host_counts(recs: Sequence[Slim]) -> list[dict[str, Any]]:
    """Units per host type, most first."""
    c: dict[str, int] = defaultdict(int)
    for s in recs:
        c[s["host"]] += 1
    return [{"host": h, "n_units": k} for h, k in sorted(c.items(), key=lambda kv: -kv[1])]


def _first_infeasible(rs: Mapping[int, dict]) -> tuple[int | None, float | None]:
    for n in sorted(rs):
        r = rs[n]
        mx = r["cost_ref"]["max"]
        over = [u["cost_ref"] for u in r["infeasible_units"]]
        hi = r.get("cost_ref_hi_max")
        if (hi is not None and hi > r["budget"]) or r["n_infeasible"]:
            return n, max([x for x in [mx, *over] if x is not None], default=None)
    return None, None


def _t3_status(n: int, r: dict | None, first: int | None, a_max_n: int | None) -> str:
    if first is not None and n == first:
        return "infeasible"
    if first is not None and n > first:
        return "infeasible (smaller n infeasible)"
    if r is None and a_max_n and n > a_max_n:
        return "not in grid"
    if r is None or r["cost_ref"]["max"] is None:
        return "no DEV cost"
    return "feasible"


def t3(
    table: Sequence[dict], n_grid: Sequence[int], max_n: Mapping[str, int | None] | None = None
) -> dict[str, Any]:
    """T3 feasibility per arm.

    The first n (ascending) whose maximum unit cost in reference seconds exceeds the budget, or
    that has a measured infeasible unit, makes n and every larger n infeasible. A unit within
    the speed factor's error of the budget counts as over (the test is on cost x f_hi). Grid n
    without a DEV cost are "no DEV cost"; above the arm's spec `max_n` they are "not in grid".
    """
    max_n = max_n or {}
    out = {}
    for a in sorted({r["arm"] for r in table}):
        rs = {r["n"]: r for r in table if r["arm"] == a}
        first, first_cost = _first_infeasible(rs)
        by_n = {}
        for n in sorted(set(n_grid) | set(rs)):
            r = rs.get(n)
            st = _t3_status(n, r, first, max_n.get(a))
            e: dict[str, Any] = {"status": st}
            if r is not None and r["cost_ref"]["max"] is not None:
                e["max_cost"] = r["cost_ref"]["max"]
                e["budget_share"] = r["cost_ref"]["max"] / r["budget"]
                e["max_cost_hi"] = r.get("cost_ref_hi_max")
                if (
                    st == "infeasible"
                    and r["cost_ref"]["max"] <= r["budget"]
                    and not r["n_infeasible"]
                ):
                    e["status"] = "infeasible (within the speed factor's error, R-55)"
                e["provisional"] = r["n_unconverted"] > 0
            by_n[str(n)] = e
        out[a] = {"first_infeasible_n": first, "first_infeasible_cost": first_cost, "by_n": by_n}
    return out


def spec_fields(feas: Mapping[str, Any]) -> dict[str, Any]:
    """What the freeze spec takes for a T3-infeasible arm: `max_n` and `t3_cost_cpu_s`."""
    out = {}
    for a, d in feas.items():
        if d["first_infeasible_n"] is None:
            continue
        ok = [int(n) for n, e in d["by_n"].items() if e["status"] == "feasible"]
        out[a] = {
            "max_n": max(ok) if ok else None,
            "t3_cost_cpu_s": {str(d["first_infeasible_n"]): d["first_infeasible_cost"]},
        }
    return out


def _f(x: float | None, nd: int = 1) -> str:
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.{nd}f}"
    return f"{x:.2f}" if x >= 0.1 else f"{x:.3f}"


def _md_head(out: Mapping[str, Any], title: str) -> list[str]:
    lines = [
        f"# Experiment C: runtime per method ({title})",
        "",
        f"Generated {out['generated_utc']} by `scratchpad/xmethod/exp_c_runtime.py` "
        f"({out['version']}). Unit = one "
        f"(method, dataset): record `cpu_s` = process CPU time of `method.run`, single thread "
        f"(data "
        f"generation excluded, shown apart). Costs in **Kaggle reference CPU-s** "
        f"(`{out['ref_host']}`); other hosts x f "
        f"(R-51). Peak RSS = `peak_rss_mb` (scope per job on Linux; includes interpreter, "
        f"imports and "
        f"dataset), "
        f"not converted. Budget R-13: {out['budget_cpu_s']:,.0f} CPU-s per (method, dataset); "
        f"T3: max "
        f"over "
        f"worlds, regimes, lambdas at n > budget makes n and every larger n infeasible; R-55: "
        f"tested "
        f"on cost x f_hi "
        f"(within the speed factor's error of the budget = over). Rows pool tune + measure seeds, "
        f"kappas and "
        f"lambdas (per role and per world / regime in the json).",
        "",
        "## Inputs",
        "",
    ]
    for f in out["inputs"]["files"]:
        lines.append(f"- `{f['path']}` ({f['lines']} lines, sha256 `{f['sha256'][:16]}`)")
    inp = out["inputs"]
    lines += [
        f"- {inp['lines']} lines, {inp['duplicates']} duplicate keys "
        f"({inp['conflicts']} on two hosts); specs {', '.join(out['spec_names'])}; "
        f"commits {', '.join(c[:7] for c in out['commits'])}",
        "",
        "Hosts (units):",
        "",
    ]
    lines += [f"- `{h['host']}`: {h['n_units']}" for h in out["hosts"]]
    return lines


def _md_factors(out: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Speed factors applied (cost_ref = cost x f)", ""]
    if out["factors"]:
        lines += [
            "| arm | host | f | f_hi | source | cells | host units | cell ratio q25 / q50 / q75 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in out["factors"]:
            q = f.get("cell_ratio_q25_q50_q75")
            ratio = " / ".join(f"{x:.2f}" for x in q) if q else "-"
            lines.append(
                f"| {f['arm']} | `{f['host']}` | {f['f']:.3f} | {_f(f.get('f_hi'))} | "
                f"{f['source']} | "
                f"{f.get('n_cells', '-')} | "
                f"{f.get('n_host_units', f.get('n_units', '-'))} | "
                f"{ratio} |"
            )
        if any(f["source"] == "observational" for f in out["factors"]):
            lines += [
                "",
                "**observational** = PROVISIONAL within-cell estimate (same cell, different "
                "seeds on each host), "
                "not the T3 factor (paired calibration units, `calib`). Replace with `--factors` "
                "before the freeze.",
            ]
    else:
        lines.append("None needed (every unit on the reference host) or none available.")
    unconv = sum(r["n_unconverted"] for r in out["table"])
    if unconv:
        lines += [
            "",
            f"**{unconv} units unconverted** (no factor for their (arm, host)): their rows are "
            "provisional.",
        ]
    return lines


def _t3_summary(fe: Mapping[str, Any]) -> str:
    ok = [n for n, e in fe["by_n"].items() if e["status"] == "feasible"]
    none = [n for n, e in fe["by_n"].items() if e["status"] == "no DEV cost"]
    s = f"feasible n <= {max(map(int, ok))}" if ok else "no feasible n"
    if fe["first_infeasible_n"] is not None:
        s += (
            f"; INFEASIBLE from n {fe['first_infeasible_n']} "
            f"(DEV cost {_f(fe['first_infeasible_cost'])})"
        )
    if none:
        s += f"; no DEV cost at n {', '.join(none)}"
    nig = [n for n, e in fe["by_n"].items() if e["status"] == "not in grid"]
    if nig:
        s += f"; not in grid at n {', '.join(nig)} (spec max_n)"
    return s


def _md_summary(out: Mapping[str, Any]) -> list[str]:
    tab = out["table"]
    ns = sorted({r["n"] for r in tab})
    lines = [
        "",
        "## Summary: median / max CPU-s per unit (Kaggle ref.) by n; T3 status",
        "",
        "| arm | " + " | ".join(f"n {n}" for n in ns) + " | T3 |",
        "|---|" + "---|" * (len(ns) + 1),
    ]
    for a in sorted({r["arm"] for r in tab}):
        cells = []
        for n in ns:
            r = next((x for x in tab if x["arm"] == a and x["n"] == n), None)
            if r is None or r["cost_ref"]["max"] is None:
                cells.append("-")
                continue
            mark = "" if r["t3"]["status"] == "feasible" else " **INF**"
            prov = ("*" if r["n_unconverted"] else "") + (
                "+" if "observational" in r["factor_sources"] else ""
            )
            cells.append(f"{_f(r['cost_ref']['median'])} / {_f(r['cost_ref']['max'])}{prov}{mark}")
        lines.append(f"| {a} | " + " | ".join(cells) + f" | {_t3_summary(out['feasibility'][a])} |")
    lines += [
        "",
        "`-` = no record at that n (above the spec max_n: not in grid; else T3: a cost pilot "
        "decides "
        "if the n is "
        "in the arm's grid); * = some unit "
        "unconverted; + = some unit converted with a provisional (observational) factor. "
        "Pooled over "
        "worlds, "
        "regimes, lambdas, kappas (the kappa sweep is at n 1000) and roles (tune + measure).",
    ]
    return lines


def _md_full_row(r: Mapping[str, Any]) -> str:
    c = r["cost_ref"]
    ro = r["roles"]
    if set(ro) & {"tune", "measure"}:
        roles = " / ".join(str(ro.get(k, 0)) for k in ("tune", "measure"))
    else:
        roles = ", ".join(f"{k} {v}" for k, v in ro.items())
    ppv = r.get("procs_per_vcpu")
    non_ref = "-" if r["share_non_ref"] is None else f"{100 * r['share_non_ref']:.0f} %"
    share = "-" if c["max"] is None else f"{100 * c['max'] / r['budget']:.2f} %"
    return (
        f"| {r['arm']} | {r['method']} | {r['n']} | {r['n_units']} ({roles}) | "
        f"{non_ref} | "
        f"{_f(c['mean'])} | {_f(c['median'])} | {_f(c['min'])} | {_f(c['max'])} | "
        f"{_f(r.get('cost_ref_hi_max'))} | "
        f"{share} | "
        f"{_f(r['peak_rss_mb']['median'])} | {_f(r['peak_rss_mb']['max'])} | "
        f"{_f(r['wall_s_mean'])} | "
        f"{_f(r['wall_cpu_ratio_median'])} | {_f(ppv['median']) if ppv else '-'} | "
        f"{_f(r['gen_cpu_s_mean'])} | {r['n_multi_thread']} | {r['t3']['status']} |"
    )


def _md_full(out: Mapping[str, Any]) -> list[str]:
    tab = out["table"]
    lines = [
        "",
        "## Full table",
        "",
        "| arm | method | n | units (tune / measure) | non-ref | CPU-s mean | median | min | max | "
        "max x f_hi | "
        "max / budget | RSS med MB | RSS max MB | wall-s mean | wall / CPU | procs / vCPU | "
        "gen CPU-s "
        "| multi-thr "
        "| T3 |",
        "|---|" * 1 + "---|" * 18,
    ]
    lines += [_md_full_row(r) for r in tab]
    gpu = [r for r in tab if r["cost_field"] == "wall_s"]
    if gpu:
        lines += [
            "",
            "GPU arms (R-41; wall-s vs a Kaggle T4, never summed with CPU-s): "
            f"{', '.join(sorted({r['arm'] for r in gpu}))}.",
        ]
    errs = [r for r in tab if r["n_errors"] or r["n_no_cost"] or r["n_infeasible"]]
    if errs:
        lines += [
            "",
            "Units without a cost: "
            + "; ".join(
                f"{r['arm']} n {r['n']}: {r['n_errors']} error, {r['n_no_cost']} no cost, "
                f"{r['n_infeasible']} infeasible"
                for r in errs
            ),
        ]
    mt = sum(r["n_multi_thread"] for r in tab)
    ct = [r for r in tab if r["n_contended"]]
    rows_ct = (
        "; rows: " + ", ".join(f"{r['arm']} n {r['n']} ({r['n_contended']})" for r in ct[:20])
        if ct
        else ""
    )
    lines += [
        "",
        f"Single-thread check: {mt} units with cpu_s > {MULTI_THREAD_RATIO} x wall_s + .05 s "
        "(more than one busy thread).",
        "",
        f"Contention check: {sum(r['n_contended'] for r in tab)} units with wall_s > "
        f"{CONTENTION_RATIO} x "
        "cpu_s + .05 s (CPU-starved: more busy processes than cores; CPU-s on shared hyperthreads "
        "runs slower, "
        "so the speed factor must be measured under the same load)" + rows_ct + ".",
    ]
    return lines


def render_md(out: Mapping[str, Any], title: str) -> str:
    """The runtime table as markdown."""
    lines = [*_md_head(out, title), *_md_factors(out), *_md_summary(out), *_md_full(out)]
    if out["spec_t3_fields"]:
        lines += [
            "",
            "## T3 spec fields (for the freeze spec)",
            "",
            "```json",
            json.dumps(out["spec_t3_fields"], indent=1),
            "```",
        ]
    return "\n".join(lines) + "\n"


def _calib_cell_costs(
    at: Sequence[Slim], ref: str, refm: bool, fields: Mapping[str, str], a: str
) -> dict[tuple, list[float]]:
    cm: dict[tuple, list[float]] = defaultdict(list)
    for s in at:
        if s["host"] == ref or not refm:
            c = unit_cost(s, fields.get(a, "cpu_s"))
            if c is not None:
                cm[(s["world"], s["regime"], s["lam"], s["kappa"])].append(c)
    return cm


def _key(a: str, w: str, rg: str, lam: float | None, kap: float, n: int, sd: int) -> str:
    lam_part = [f"lam{lam:g}"] if lam is not None else []
    return "|".join([a, w, rg, *lam_part, f"k{kap:g}", f"n{n}", f"s{sd}"])


def calibration_plan(
    recs: Sequence[Slim], ref: str, fields: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """T3 calibration units per arm.

    Seeds 3_000_000-002 in the arm's three costliest (world, regime) at its largest measured n
    (cost = mean on the reference host if measured there, else the raw mean over all hosts); in
    each (world, regime) the kappa .25 cell, if any, with the costliest lambda. Hosts are every
    non-reference host type the arm ran on.
    """
    fields = fields or {}
    keys = {
        (s["arm"], s["world"], s["regime"], s["lam"], s["kappa"], s["n"], s["seed"]): s["key"]
        for s in recs
    }
    out = {}
    for a in sorted({s["arm"] for s in recs}):
        rs = [
            s
            for s in recs
            if s["arm"] == a
            and s["status"] == "ok"
            and unit_cost(s, fields.get(a, "cpu_s")) is not None
        ]
        if not rs:
            continue
        nmax = max(s["n"] for s in rs)
        at = [s for s in rs if s["n"] == nmax]
        refm = any(s["host"] == ref for s in at)
        best: dict[tuple, tuple] = {}
        for (w, rg, lam, kap), xs in _calib_cell_costs(at, ref, refm, fields, a).items():
            m = float(np.mean(xs))
            prim = kap is not None and abs(kap - PRIMARY_KAPPA) < 1e-9
            cand = (prim, m, lam, kap)
            if (w, rg) not in best or cand[:2] > best[(w, rg)][:2]:
                best[(w, rg)] = cand
        top = sorted(best.items(), key=lambda kv: -kv[1][1])[:3]
        units = []
        for (w, rg), (_, m, lam, kap) in top:
            for sd in CALIB_SEEDS:
                k = keys.get((a, w, rg, lam, kap, nmax, sd))
                units.append(
                    {
                        "key": k or _key(a, w, rg, lam, kap, nmax, sd),
                        "world": w,
                        "regime": rg,
                        "lam": lam,
                        "kappa": kap,
                        "n": nmax,
                        "seed": sd,
                        "in_records": k is not None,
                        "cell_mean_cost": m,
                    }
                )
        hosts = sorted({s["host"] for s in recs if s["arm"] == a} - {ref})
        out[a] = {
            "n": nmax,
            "cost_from": "reference host" if refm else "all hosts (no reference record at n)",
            "units": units,
            "hosts_needing_factor": hosts,
        }
    return {"version": VERSION, "ref_host": ref, "seeds": list(CALIB_SEEDS), "arms": out}


def _session_factors(us: Sequence[dict]) -> dict[str, float]:
    nodes = sorted({nd for u in us for nd in u["host_costs"]})
    if len(nodes) < 2:
        return {}
    out = {}
    for nd in nodes:
        un = [u for u in us if nd in u["host_costs"]]
        hc = sum(u["host_costs"][nd] for u in un)
        if hc > 0:
            out[nd] = sum(u["ref_cost"] for u in un) / hc
    return out


def _pair_units(
    recs_all: Sequence[Slim], ref: str, fields: Mapping[str, str], want: set[str] | None
) -> tuple[dict[tuple, list[dict]], set]:
    by_key: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for s in recs_all:
        if s["status"] != "ok" or (want is not None and s["key"] not in want):
            continue
        c = unit_cost(s, fields.get(s["arm"], "cpu_s"))
        if c is not None:
            by_key[s["key"]][s["host"]].setdefault(
                s.get("node"), (c, s["dataset_sha256"], s["spec_name"])
            )
            meta[s["key"]] = (s["arm"], s["n"])
    pairs: dict[tuple, list[dict]] = defaultdict(list)
    specs: set = set()
    for k, hs in by_key.items():
        if ref not in hs:
            continue
        rc, rsha, rsp = next(iter(hs[ref].values()))
        for h, sess in hs.items():
            if h == ref:
                continue
            cs = {str(nd): v[0] for nd, v in sess.items()}
            shas = {v[1] for v in sess.values()}
            pairs[(meta[k][0], h)].append(
                {
                    "key": k,
                    "n": meta[k][1],
                    "ref_cost": rc,
                    "host_cost": sum(cs.values()) / len(cs),
                    "host_costs": cs,
                    "same_dataset": shas == {rsha} if rsha and None not in shas else None,
                }
            )
            specs |= {v[2] for v in sess.values()} | {rsp}
    return pairs, specs


def _arm_host_factor(us: Sequence[dict], src: str) -> dict[str, Any] | None:
    sh = sum(u["host_cost"] for u in us)
    if sh <= 0:
        return None
    f = sum(u["ref_cost"] for u in us) / sh
    per_n = {}
    for n in sorted({u["n"] for u in us}):
        un = [u for u in us if u["n"] == n]
        hn = sum(u["host_cost"] for u in un)
        if hn > 0:
            per_n[str(n)] = sum(u["ref_cost"] for u in un) / hn
    per_s = _session_factors(us)
    spread = [f, *per_n.values(), *per_s.values()]
    return {
        "f": f,
        "f_lo": min(spread),
        "f_hi": max(spread),
        "f_by_n": per_n,
        "f_by_session": per_s,
        "source": src,
        "n_units": len(us),
        "n_same_dataset": sum(u["same_dataset"] is True for u in us),
        "n_different_dataset": sum(u["same_dataset"] is False for u in us),
        "units": list(us),
    }


def calibrate(
    recs_all: Sequence[Slim],
    ref: str,
    plan: Mapping[str, Any] | None = None,
    fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Speed factors from units with an ok record on the reference and on a host.

    Per (arm, host): f = sum of the reference cost over the sum of the host cost; `f_lo` and
    `f_hi` are the min and max of f and the per-n ratios (its error), widened by the per-session
    factors when a host type has several sessions. Per host, arm `*` (every arm without its own
    factor) pools all arms, with `f_lo`/`f_hi` the min and max over the arms' f. With `plan` only
    its units count. The source is `r55_calibration_block` when every paired record is from the
    calibration block, `t3_calibration` with a plan, else `paired_units`.

    Args:
        recs_all: Slim records, every copy of a key kept.
        ref: Reference host type.
        plan: Output of `calibration_plan`, or None.
        fields: Cost field per arm.

    Returns:
        `{"version", "ref_host", "source", "factors": {arm: {host: factor}}, "missing"}`.
    """
    fields = fields or {}
    want = None if plan is None else {u["key"] for d in plan["arms"].values() for u in d["units"]}
    pairs, specs = _pair_units(recs_all, ref, fields, want)
    if plan is not None:
        src = "t3_calibration"
    elif specs == {"exp_c_calib"}:
        src = "r55_calibration_block"
    else:
        src = "paired_units"
    fac: dict[str, dict] = defaultdict(dict)
    for (a, h), us in sorted(pairs.items()):
        entry = _arm_host_factor(us, src)
        if entry is not None:
            fac[a][h] = entry
    for h in sorted({h for (_, h) in pairs}):
        arms = {a: d[h] for a, d in fac.items() if h in d and a != "*"}
        if not arms:
            continue
        us = [u for (_, hh), x in pairs.items() if hh == h for u in x]
        sh = sum(u["host_cost"] for u in us)
        per_s = _session_factors(us)
        fs = [d["f"] for d in arms.values()] + list(per_s.values())
        fac["*"][h] = {
            "f": sum(u["ref_cost"] for u in us) / sh,
            "f_lo": min(fs),
            "f_hi": max(fs),
            "f_by_session": per_s,
            "source": f"{src} (pooled anchors)",
            "n_units": len(us),
            "arms": sorted(arms),
        }
    missing = []
    if plan is not None:
        for a, d in plan["arms"].items():
            for h in d["hosts_needing_factor"]:
                if h not in fac.get(a, {}):
                    missing.append({"arm": a, "host": h})
                elif fac[a][h]["n_units"] < len(d["units"]):
                    fac[a][h]["incomplete"] = (
                        f"{fac[a][h]['n_units']} of {len(d['units'])} plan units paired"
                    )
    return {
        "version": VERSION,
        "ref_host": ref,
        "source": src,
        "factors": dict(fac),
        "missing": missing,
    }


def write_table(out: Mapping[str, Any], out_dir: str | Path, title: str) -> None:
    """Writes `exp_c_runtime.json` and `EXP_C_RUNTIME.md` (LF) into `out_dir`."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "exp_c_runtime.json").write_text(
        json.dumps(out, indent=1, allow_nan=False), encoding="utf-8", newline="\n"
    )
    (d / "EXP_C_RUNTIME.md").write_text(render_md(out, title), encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("table", "plan", "calib"):
        p = sub.add_parser(name)
        p.add_argument("--records", nargs="+", required=True)
        p.add_argument("--spec", nargs="*", default=[])
        p.add_argument("--ref-host", default=REF_HOST)
        if name == "table":
            p.add_argument("--factors", default=None)
            p.add_argument("--no-observational", action="store_true")
            p.add_argument("--budget", type=float, default=BUDGET_CPU_S)
            p.add_argument("--out-dir", required=True)
            p.add_argument("--title", default="DEV")
            p.add_argument("--generated-utc", default=None)
        elif name == "plan":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--plan", default=None)
            p.add_argument("--out", required=True)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: `table`, `plan` and `calib` over record files."""
    a = _parser().parse_args(argv)
    recs, info = load(a.records)
    arm_specs = load_specs(a.spec)
    fields = {k: cost_field(v) for k, v in arm_specs.items()}
    if a.cmd == "table":
        out = build(
            recs,
            info,
            ref=a.ref_host,
            budget=a.budget,
            arm_specs=arm_specs,
            factors=load_factor_file(a.factors),
            observational=not a.no_observational,
            generated_utc=a.generated_utc,
        )
        write_table(out, a.out_dir, a.title)
        inf = [x for x, d in out["feasibility"].items() if d["first_infeasible_n"] is not None]
        print(
            f"[exp-c] {len(recs)} units, {len(out['table'])} arm x n rows, factors "
            f"{len(out['factors'])}, T3-infeasible arms: {inf or 'none'} -> {a.out_dir}"
        )
    elif a.cmd == "plan":
        out = calibration_plan(recs, a.ref_host, fields)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
        print(
            f"[exp-c] plan: {sum(len(d['units']) for d in out['arms'].values())} units over "
            f"{len(out['arms'])} arms -> {a.out}"
        )
    else:
        copies = info["copies"]
        allc = [c for cs in copies.values() for c in cs]
        allc += [s for s in recs if s["key"] not in copies]
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else None
        out = calibrate(allc, a.ref_host, plan, fields)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
        got = ", ".join(
            f"{arm}@{h.split('|')[0]} {v['f']:.3f} ({v['n_units']})"
            for arm, hs in out["factors"].items()
            for h, v in hs.items()
        )
        print(
            f"[exp-c] calib ({out['source']}): "
            + (got or "no unit with an ok record on both the reference and another host")
            + (f"; missing {len(out['missing'])}" if out["missing"] else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

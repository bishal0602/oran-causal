"""X5 and X6: diagnostics of the E4 R3 placebo excess (X5) and of the told-width bias (X6).

Both are exploratory post hoc analyses. Every rate, interval and label comes from the Study 4
analysis (`analysis.controlled`: screening, cells, cluster bootstrap, validity class); the extra
statistics are the ones declared in the X5 and X6 entries of the extras protocol.

* X5 step 1 reads the Study 4 EVAL records: calibration of the raw placebo p-values of the two
  PMRT arms on E4 R3, and the chance of an INVALID label for a rate that sits exactly at .05.
* X5 steps 2 and 3 read the fresh datasets of `x5_fail` and `x5_adj`: a one-sided exact binomial
  test against .05 per primary cell with Holm adjustment, and an exact McNemar test of the two arms.
* X6 reads the told-width grid on E1 R2: the pooled rates and the bias of the told centre.
"""

import argparse
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from oran_causal.analysis.controlled.cells import build_cells, screen
from oran_causal.analysis.controlled.fmax import boot_mult, fmax_simulate, load_dependence
from oran_causal.analysis.controlled.protocol import ALPHA, VALID_UB
from oran_causal.analysis.controlled.validity import cluster_ci
from oran_causal.analysis.posthoc.paths import CITED_PROTOCOL, PROTOCOL_BASENAME, SPECS, write_text
from oran_causal.analysis.posthoc.x2x3 import load_spec, rate_text, rate_view, repro_check
from oran_causal.analysis.records import load_records
from oran_causal.core.scoring import PLACEBO, PLACEBO_CONF
from oran_causal.sim.controlled.datasets.generate import dataset_hash, generate_dataset
from oran_causal.sim.controlled.datasets.truth import truth_for

E4R3_LAMS = (0.0, 0.5, 1.0, 1.5)
E4R3_NS = (500, 1000, 4000, 8000, 24000)
QQ_LEVELS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.50)
TAIL_LEVELS = (0.01, 0.025, 0.05, 0.10)
KIND_EDGE = {"plac": "P_placebo->K0", "conf": "P_placebo_conf->K0"}
LABEL_NSIM, LABEL_SEED = 10_000, 20261005
TOP = ("x5_step1", "x5", "x6")
X5_ARMS = ("pmrt_eq", "pmrt_r3")
X5_PRIMARY = ((0.0, 8000, "conf_raw"), (0.5, 8000, "conf_raw"), (1.0, 24000, "plac_raw"))
X6_WORLD = "E1"
EVAL_SPEC = SPECS / "eval" / "full.json"
DEPENDENCE = Path("study4_dev") / "fmax_sim" / "dependence.json"

Rec = dict[str, Any]


def _f(x: float | None, d: int = 3) -> str:
    return "NA" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def cp_ci(k: int, n: int) -> list[float]:
    """Two-sided 95 % Clopper-Pearson interval."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def holm(ps: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def edge_kinds(world: str, regime: str) -> dict[str, str]:
    """Kind (`null`, `plac`, `conf`) of every non-KPI null edge of a world and regime, by name."""
    truth = truth_for(world, regime)
    out = {}
    for source, target in sorted(e for e in truth.null_edges if not e[0].startswith("K")):
        kind = "plac" if source == PLACEBO else "conf" if source == PLACEBO_CONF else "null"
        out[f"{source}->{target}"] = kind
    return out


def _arm_of(r: Mapping[str, Any]) -> str:
    return r.get("arm") or r.get("key", "").split("|")[0]


def read_filtered(path: str | Path, keep: Callable[[Rec], bool]) -> list[Rec]:
    """Streams a record file and keeps the records `keep` accepts, in the Study 4 load order."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    recs = []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue
            if r is not None and keep(r):
                recs.append(r)
    recs.sort(key=lambda r: (r.get("key", ""), json.dumps(r, sort_keys=True)))
    return recs


def screened(
    merged: str | Path, spec: Mapping[str, Any], keep: Callable[[Rec], bool] | None = None
) -> tuple[list[Rec], dict, dict, dict]:
    """Used records, screening info, cells and seed rows under the Study 4 rules.

    Args:
        merged: Merged record file.
        spec: Campaign spec.
        keep: Optional record filter applied before screening; only the kept arms and cells are
            then expected.
    """
    recs = load_records(merged) if keep is None else read_filtered(merged, keep)
    use, scr = screen(recs, dict(spec))
    cells, rows = build_cells(use, dict(spec))
    return use, scr, cells, rows


def e4r3_pvals(use: Sequence[Rec], arms: Sequence[str]) -> dict[tuple, dict[int, dict[str, float]]]:
    """`{(arm, lam, n): {seed: {kind: p}}}` of the measurement ok records (raw edge p)."""
    out: dict = defaultdict(dict)
    for r in use:
        j = r["job"]
        if (r.get("status") or "ok") != "ok" or (r.get("role") or "measure") != "measure":
            continue
        arm = _arm_of(r)
        if arm not in arms or j["world"] != "E4" or j["regime"] != "R3":
            continue
        ps = {}
        for e in r.get("edges", []):
            name = f"{e['source']}->{e['target']}"
            for kind, en in KIND_EDGE.items():
                if name == en and e.get("p") is not None:
                    ps[kind] = float(e["p"])
        out[(arm, float(j["lam"]), int(j["n"]))][int(j["seed"])] = ps
    return dict(out)


def _e4r3_cell(cells: Mapping[str, Any], arm: str, lam: float, n: int) -> dict | None:
    for e in cells.values():
        if (
            e["arm"] == arm
            and e["world"] == "E4"
            and e["regime"] == "R3"
            and e["lam"] == lam
            and e["n"] == n
            and e["kappa"] == 0.25
        ):
            return e
    return None


def label_prob(S: int, nsim: int = LABEL_NSIM, seed: int = LABEL_SEED) -> dict[str, Any]:
    """Chance of an INVALID label for one rate at exactly .05 over S independent datasets.

    One candidate per dataset, under the frozen seed-cluster bootstrap (the draws of
    `cluster_ci`); also the chance of a VALID label.
    """
    m = boot_mult(S)
    rng = np.random.default_rng(seed)
    h = (rng.random((nsim, S)) < ALPHA).astype(float)
    bs = (h @ m.T) / S
    lo, hi = np.quantile(bs, 0.025, axis=1), np.quantile(bs, 0.975, axis=1)
    inv = lo > ALPHA
    val = (hi <= VALID_UB) & ~inv
    return {
        "S": S,
        "nsim": nsim,
        "seed": seed,
        "p_invalid": float(inv.mean()),
        "p_valid": float(val.mean()),
        "min_hits_invalid": int(h.sum(1)[inv].min()) if inv.any() else None,
    }


def _step1_kind(
    d: Mapping[int, Mapping[str, float]], prim: Mapping[str, Any], kind: str
) -> tuple[dict[str, Any], set[int]]:
    ps = np.array(sorted(x[kind] for x in d.values() if kind in x))
    fr = prim.get(f"{kind}_raw") or {}
    ks = stats.kstest(ps, "uniform") if len(ps) else None
    rejected = {s for s, x in d.items() if x.get(kind, 1.0) <= ALPHA}
    row = {
        "n": len(ps),
        "frozen_rate": fr.get("rate"),
        "frozen_ci": fr.get("ci"),
        "frozen_label": fr.get("validity"),
        "ks_d": None if ks is None else float(ks.statistic),
        "ks_p": None if ks is None else float(ks.pvalue),
        "qq": {str(q): float(np.quantile(ps, q)) for q in QQ_LEVELS} if len(ps) else None,
        "tail": (
            {
                str(a): {
                    "rate": float(np.mean(ps <= a)),
                    "obs_exp": float(np.mean(ps <= a) / a),
                }
                for a in TAIL_LEVELS
            }
            if len(ps)
            else None
        ),
    }
    return row, rejected


def _step1_cell(
    pv: Mapping[tuple, Mapping[int, Mapping[str, float]]],
    cells: Mapping[str, Any],
    arm: str,
    lam: float,
    n: int,
) -> dict[str, Any]:
    d = pv.get((arm, lam, n), {})
    prim = (_e4r3_cell(cells, arm, lam, n) or {}).get("primary") or {}
    row: dict[str, Any] = {"arm": arm, "lam": lam, "n": n, "seeds": len(d)}
    rej = {}
    for kind in KIND_EDGE:
        row[kind], rej[kind] = _step1_kind(d, prim, kind)
    both = len(rej["plac"] & rej["conf"])
    N = len(d)
    exp = len(rej["plac"]) * len(rej["conf"]) / N if N else None
    row["overlap_plac_conf"] = {
        "both": both,
        "plac": len(rej["plac"]),
        "conf": len(rej["conf"]),
        "expected": exp,
        "p_upper": (
            float(stats.hypergeom.sf(both - 1, N, len(rej["plac"]), len(rej["conf"])))
            if N
            else None
        ),
    }
    row["failing"] = any((lam, n, f"{k}_raw") in X5_PRIMARY for k in KIND_EDGE) and arm == "pmrt_eq"
    return row


def _step1_lambda_overlap(
    pv: Mapping[tuple, Mapping[int, Mapping[str, float]]], arm: str, n: int, kind: str
) -> dict[str, Any] | None:
    sets = {
        lam: {s for s, x in pv.get((arm, lam, n), {}).items() if x.get(kind, 1.0) <= ALPHA}
        for lam in E4R3_LAMS
    }
    seeds = set().union(*(set(pv.get((arm, lam, n), {})) for lam in E4R3_LAMS))
    N = len(seeds)
    if not N:
        return None
    cnt = np.array([sum(s in sets[lam] for lam in E4R3_LAMS) for s in sorted(seeds)])
    ps_l = [len(sets[lam]) / N for lam in E4R3_LAMS]
    dist = np.array([1.0])
    for p in ps_l:
        dist = np.convolve(dist, [1 - p, p])
    pairs = []
    for i, a in enumerate(E4R3_LAMS):
        for b in E4R3_LAMS[i + 1 :]:
            o = len(sets[a] & sets[b])
            pairs.append({"lams": [a, b], "both": o, "expected": len(sets[a]) * len(sets[b]) / N})
    return {
        "arm": arm,
        "n": n,
        "kind": kind,
        "seeds": N,
        "per_lambda_hits": [len(sets[lam]) for lam in E4R3_LAMS],
        "seeds_ge2": int((cnt >= 2).sum()),
        "expected_ge2": float(N * dist[2:].sum()),
        "seeds_ge3": int((cnt >= 3).sum()),
        "expected_ge3": float(N * dist[3:].sum()),
        "lambda_corr_note": "EVAL lambdas share each seed's streams (positive dependence expected)",
        "pairs": pairs,
    }


def _step1_fmax(out: Mapping[str, Any], S: int, arms: Sequence[str], dep: Any) -> dict[str, Any]:
    fm = fmax_simulate(
        [("E4", "R3", lam, n) for lam in E4R3_LAMS for n in E4R3_NS],
        S,
        ["plac_raw", "conf_raw"],
        dep,
    )
    tot = sum(fm["count_dist"].values())

    def invalid_cells(arm: str) -> int:
        return sum(
            1
            for c in out["cells"]
            if c["arm"] == arm and any(c[kk]["frozen_label"] == "INVALID" for kk in KIND_EDGE)
        )

    return {
        "f_max": fm["f_max"],
        "nsim": fm["nsim"],
        "n_cells": fm["n_cells"],
        "p_cell_invalid": fm["p_cell_invalid"],
        "count_dist": fm["count_dist"],
        "p_ge": {
            arm: sum(v for k, v in fm["count_dist"].items() if int(k) >= invalid_cells(arm)) / tot
            for arm in arms
        },
        "note": "counts INVALID cells (a cell is INVALID if either rate is), as the frozen C3 rule",
    }


def x5_step1(
    eval_merged: str | Path,
    eval_spec: Mapping[str, Any] | None = None,
    dependence: str | Path | None = None,
) -> dict[str, Any]:
    """X5 step 1: calibration of the E4 R3 placebo p-values in the Study 4 EVAL records.

    Args:
        eval_merged: Study 4 EVAL merged records (read only).
        eval_spec: The frozen EVAL spec; read from the experiments member when None.
        dependence: Dependence table of the `F_max` simulation.
    """
    spec = eval_spec or json.loads(EVAL_SPEC.read_text(encoding="utf-8"))
    arms = X5_ARMS

    def keep(r: Rec) -> bool:
        j = r.get("job") or {}
        return _arm_of(r) in arms and j.get("world") == "E4" and j.get("regime") == "R3"

    use, scr, cells, _ = screened(eval_merged, spec, keep)
    pv = e4r3_pvals(use, arms)
    out: dict[str, Any] = {
        "source": Path(eval_merged).name,
        "records_used": len(use),
        "cells": [],
        "overlap_lambda": [],
        "label_count": {},
    }
    S_all = set()
    for arm in arms:
        for n in E4R3_NS:
            for lam in E4R3_LAMS:
                S_all.add(len(pv.get((arm, lam, n), {})))
                out["cells"].append(_step1_cell(pv, cells, arm, lam, n))
            for kind in KIND_EDGE:
                ov = _step1_lambda_overlap(pv, arm, n, kind)
                if ov is not None:
                    out["overlap_lambda"].append(ov)
    S = max(S_all) if S_all else 0
    lp = label_prob(S)
    out["label_prob"] = lp
    for arm in arms:
        labs = [c[k]["frozen_label"] for c in out["cells"] if c["arm"] == arm for k in KIND_EDGE]
        k_obs = sum(1 for x in labs if x == "INVALID")
        m = len(labs)
        out["label_count"][arm] = {
            "rates": m,
            "invalid_observed": k_obs,
            "expected": m * lp["p_invalid"],
            "p_ge_observed_indep": (
                float(stats.binom.sf(k_obs - 1, m, lp["p_invalid"])) if m else None
            ),
            "note": "rates treated as independent (P_placebo / P_placebo_conf share a dataset; "
            "lambdas and n share seeds): an approximation",
        }
    out["fmax_dependence"] = _step1_fmax(out, S, arms, load_dependence(dependence))
    out["screen"] = {
        "duplicates": len(scr["duplicates"]),
        "unexpected": len(scr["unexpected"]),
        "role_mismatch": len(scr["role_mismatch"]),
    }
    return out


def _provenance(use: Sequence[Rec], per_cell: int = 10) -> dict[str, Any]:
    """Regenerates the first `per_cell` datasets of each cell with the frozen generator.

    Their hashes must match the records.
    """
    seen: dict[tuple, set] = defaultdict(set)
    ok, bad = 0, []
    for r in sorted(use, key=lambda r: r["key"]):
        j = r["job"]
        ck = (j["lam"], j["n"])
        if len(seen[ck]) >= per_cell or j["seed"] in seen[ck] or (r.get("status") or "ok") != "ok":
            continue
        seen[ck].add(j["seed"])
        ds, _ = generate_dataset(
            j["world"], j["regime"], j["n"], j["seed"], lam=j["lam"], kappa=j["kappa"]
        )
        if dataset_hash(ds) == r.get("dataset_sha256"):
            ok += 1
        else:
            bad.append(r["key"])
    return {"checked": ok + len(bad), "equal": ok, "differ": bad[:5], "ok": not bad}


def _x5_kind_row(d: Mapping[int, Mapping[str, float]], prim: Mapping[str, Any], kind: str) -> dict:
    k = sum(1 for x in d.values() if x.get(kind, 1.0) <= ALPHA)
    m = sum(1 for x in d.values() if kind in x)
    fr = prim.get(f"{kind}_raw") or {}
    return {
        "hits": k,
        "n": m,
        "rate": k / m if m else None,
        "cp_ci": cp_ci(k, m) if m else None,
        "boot_ci": fr.get("ci"),
        "r30": fr.get("validity"),
        "p_binom_gt05": (
            float(stats.binomtest(k, m, ALPHA, alternative="greater").pvalue) if m else None
        ),
    }


def _x5_primary(cells: Sequence[Mapping[str, Any]]) -> tuple[list[dict], str | None]:
    prim_rows: list[dict[str, Any]] = []
    for lam, n, key in X5_PRIMARY:
        kind = key.split("_")[0]
        c = next(
            (r for r in cells if r["arm"] == "pmrt_eq" and r["lam"] == lam and r["n"] == n), None
        )
        extra = {} if c is None else {("n_" if k == "n" else k): v for k, v in c[kind].items()}
        prim_rows.append({"lam": lam, "n": n, "rate_key": key, **extra})
    have = [r for r in prim_rows if r.get("p_binom_gt05") is not None]
    adj = holm([float(r["p_binom_gt05"]) for r in have]) if len(have) == len(prim_rows) else None
    for i, r in enumerate(prim_rows):
        if adj is None:
            r["outcome"] = None
            continue
        r["p_holm"] = adj[i]
        if adj[i] < ALPHA:
            r["outcome"] = "EXCESS"
        else:
            r["outcome"] = "CHANCE" if r["cp_ci"][1] <= VALID_UB else "UNRESOLVED"
    oc = [r["outcome"] for r in prim_rows]
    if None in oc:
        answer = None
    elif "EXCESS" in oc:
        answer = "real excess"
    else:
        answer = "chance" if all(o == "CHANCE" for o in oc) else "unresolved"
    return prim_rows, answer


def _x5_step3(prim_rows: Sequence[Mapping[str, Any]], pvs: Mapping[tuple, Any]) -> list[dict]:
    out = []
    for r in prim_rows:
        kind = r["rate_key"].split("_")[0]
        a, b = (
            pvs.get(("pmrt_eq", r["lam"], r["n"]), {}),
            pvs.get(("pmrt_r3", r["lam"], r["n"]), {}),
        )
        seeds = sorted(s for s in set(a) & set(b) if kind in a[s] and kind in b[s])
        eq_only = sum(1 for s in seeds if a[s][kind] <= ALPHA < b[s][kind])
        r3_only = sum(1 for s in seeds if b[s][kind] <= ALPHA < a[s][kind])
        nd = eq_only + r3_only
        out.append(
            {
                "lam": r["lam"],
                "n": r["n"],
                "rate_key": r["rate_key"],
                "paired": len(seeds),
                "eq_only": eq_only,
                "r3_only": r3_only,
                "p_mcnemar_exact": float(stats.binomtest(eq_only, nd, 0.5).pvalue) if nd else 1.0,
                "triggered": r.get("outcome") == "EXCESS",
            }
        )
    return out


def x5(
    fail_merged: str | Path, adj_merged: str | Path | None, spec_dir: Path | None = None
) -> dict[str, Any]:
    """X5 steps 2 and 3 on the fresh datasets of the failing and the adjacent cells.

    Args:
        fail_merged: Merged records of the three cells that were INVALID in EVAL.
        adj_merged: Merged records of the adjacent cells, or None.
        spec_dir: Directory of the extras specs.
    """
    out: dict[str, Any] = {"cells": [], "primary": [], "checks": {}}
    pvs: dict = {}
    for name, merged in (("x5_fail", fail_merged), ("x5_adj", adj_merged)):
        if not merged:
            continue
        spec = load_spec(name, spec_dir)
        use, scr, cells, _ = screened(merged, spec)
        out["checks"][name] = {
            "records": len(use),
            "duplicates": len(scr["duplicates"]),
            "unexpected": len(scr["unexpected"]),
            "role_mismatch": len(scr["role_mismatch"]),
            "provenance": _provenance(use),
        }
        pv = e4r3_pvals(use, X5_ARMS)
        pvs.update(pv)
        for (arm, lam, n), d in sorted(pv.items()):
            e = _e4r3_cell(cells, arm, lam, n) or {}
            prim = e.get("primary") or {}
            row = {
                "spec": name,
                "arm": arm,
                "lam": lam,
                "n": n,
                "seeds": len(d),
                "status": e.get("status"),
            }
            row.update({kind: _x5_kind_row(d, prim, kind) for kind in KIND_EDGE})
            out["cells"].append(row)
    out["primary"], out["answer"] = _x5_primary(out["cells"])
    out["step3"] = _x5_step3(out["primary"], pvs)
    return out


def _bias_items(
    recs: Sequence[Rec], kinds: Mapping[str, str], kind: str
) -> list[tuple[int, float, bool, Any, Any]]:
    items = []
    for r in recs:
        tg = r["x6"]["targets"]
        pmap = {f"{e['source']}->{e['target']}": e.get("p") for e in r.get("edges", [])}
        for a, b in r["x6"]["bias"].items():
            for ti, t in enumerate(tg):
                name = f"{a}->{t}"
                if kinds.get(name) != kind or pmap.get(name) is None or b["z_bias"][ti] is None:
                    continue
                items.append(
                    (
                        r["job"]["seed"],
                        b["z_bias"][ti],
                        float(pmap[name]) <= ALPHA,
                        b["mean_c"][ti],
                        b["corr_cw"][ti],
                    )
                )
    return items


def _bias_kind(items: Sequence[tuple], mult_cache: dict[int, np.ndarray]) -> dict[str, Any]:
    seeds = sorted({s for s, *_ in items})
    si = {s: i for i, s in enumerate(seeds)}
    S = len(seeds)
    z = np.array([x[1] for x in items])
    zs, zc = np.zeros(S), np.zeros(S)
    for s, zz, *_rest in items:
        zs[si[s]] += zz
        zc[si[s]] += 1
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    hb, cb, ht, ctp = np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S)
    for s, zz, rej, *_rest in items:
        if zz <= q1:
            hb[si[s]] += rej
            cb[si[s]] += 1
        elif zz > q2:
            ht[si[s]] += rej
            ctp[si[s]] += 1
    m = mult_cache.setdefault(S, boot_mult(S))
    with np.errstate(invalid="ignore", divide="ignore"):
        diff_bs = (m @ ht) / (m @ ctp) - (m @ hb) / (m @ cb)
    diff_bs = diff_bs[np.isfinite(diff_bs)]
    mc = np.array([x[3] for x in items if x[3] is not None], float)
    cr = np.array([x[4] for x in items if x[4] is not None], float)
    return {
        "candidates": len(items),
        "seeds": S,
        "mean": float(z.mean()),
        "ci": cluster_ci(zs, zc),
        "sd": float(z.std(ddof=1)) if len(z) > 1 else None,
        "tercile_cuts": [float(q1), float(q2)],
        "rej_bottom": float(hb.sum() / cb.sum()) if cb.sum() else None,
        "rej_top": float(ht.sum() / ctp.sum()) if ctp.sum() else None,
        "top_minus_bottom": (
            float(ht.sum() / ctp.sum() - hb.sum() / cb.sum()) if cb.sum() and ctp.sum() else None
        ),
        "top_minus_bottom_ci": (
            [float(np.quantile(diff_bs, 0.025)), float(np.quantile(diff_bs, 0.975))]
            if len(diff_bs)
            else [None, None]
        ),
        "mean_c": float(np.nanmean(mc)) if len(mc) else None,
        "mean_corr_cw": float(np.nanmean(cr)) if np.isfinite(cr).any() else None,
    }


def bias(recs: Sequence[Rec], kinds: Mapping[str, str], mult_cache: dict[int, np.ndarray]) -> dict:
    """Bias of the told centre per kind (`null`, `plac`).

    Gives the mean z-bias over (seed, candidate) with a seed-cluster interval, the rejection rate
    by z-bias tercile (pooled cut points), top minus bottom with a seed-bootstrap interval, the
    mean c and its correlation.
    """
    out = {}
    for kind in ("null", "plac"):
        items = _bias_items(recs, kinds, kind)
        if items:
            out[kind] = _bias_kind(items, mult_cache)
    return out


def _predictions(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    g = {
        (a["centre"], a["redraw"]): a
        for a in arms
        if a["base"] == "pmrt_nl_eq" and a["width"] == 2.0
    }
    TT, Tt, tT, tt = (
        g.get(("told", "told")),
        g.get(("told", "true")),
        g.get(("true", "told")),
        g.get(("true", "true")),
    )

    def lab(a: Mapping[str, Any] | None) -> str | None:
        return ((a or {}).get("null_raw") or {}).get("validity")

    def ci(a: Mapping[str, Any] | None) -> list:
        return ((a or {}).get("null_raw") or {}).get("ci") or [None, None]

    def zb(a: Mapping[str, Any] | None) -> dict:
        return ((a or {}).get("bias") or {}).get("null") or {}

    p1 = None if Tt is None or tt is None else (lab(Tt) != "INVALID" and lab(tt) != "INVALID")
    p2 = None if TT is None else lab(TT) == "INVALID"
    p3 = None if tT is None or ci(tT)[1] is None else ci(tT)[1] <= VALID_UB
    b_TT, b_tT = zb(TT), zb(tT)
    p4a = (
        None if not b_TT else (b_TT["mean"] > 0 and b_TT["ci"][0] is not None and b_TT["ci"][0] > 0)
    )
    p4b = (
        None
        if not b_TT or b_TT["top_minus_bottom_ci"][0] is None
        else b_TT["top_minus_bottom_ci"][0] > 0
    )
    p4c = None if not b_TT or not b_tT else abs(b_tT["mean"]) < abs(b_TT["mean"]) / 4
    p4 = None if None in (p4a, p4b, p4c) else (p4a and p4b and p4c)
    preds = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P4a": p4a, "P4b": p4b, "P4c": p4c}
    if None in (p1, p2, p3, p4):
        outcome = None
    elif p1 and p2 and p3 and p4:
        outcome = "SUPPORTED"
    elif p2 and lab(tT) == "INVALID":
        outcome = "REFUTED"
    else:
        failed = ", ".join(k for k in ("P1", "P2", "P3", "P4") if not preds[k])
        outcome = f"PARTIAL (failed: {failed})"
    return {**preds, "outcome": outcome}


def x6(merged: str | Path, spec_dir: Path | None = None) -> dict[str, Any]:
    """X6: the told-width grid and the centring and redraw ablation on E1 R2."""
    spec = load_spec("x6_gbm", spec_dir)
    use, scr, cells, _ = screened(merged, spec)
    kinds = edge_kinds(X6_WORLD, "R2")
    x3_like = {**spec, "extras": {**spec["extras"], "experiment": "X3"}}
    out: dict[str, Any] = {
        "checks": {
            "records": len(use),
            "duplicates": len(scr["duplicates"]),
            "unexpected": len(scr["unexpected"]),
            "role_mismatch": len(scr["role_mismatch"]),
            "provenance": repro_check(use, x3_like),
        },
        "arms": [],
    }
    by_arm: dict[str, list] = defaultdict(list)
    for r in use:
        if (r.get("status") or "ok") == "ok" and (r.get("role") or "measure") == "measure":
            by_arm[_arm_of(r)].append(r)
    mult_cache: dict[int, np.ndarray] = {}
    for _ck, e in sorted(cells.items()):
        arm = e["arm"]
        d = spec["arms"][arm]
        told = d.get("told") or {}
        prim = e.get("primary") or {}
        row = {
            "arm": arm,
            "base": d.get("base", arm),
            "width": float(told.get("width", 1.0)),
            "centre": told.get("centre", "told" if told else "exact"),
            "redraw": told.get("redraw", "told" if told else "exact"),
            "seeds": e["n_measure"],
            "null_raw": rate_view(prim.get("null_raw")),
            "plac_raw": rate_view(prim.get("plac_raw")),
        }
        recs = sorted(by_arm.get(arm, []), key=lambda r: r["job"]["seed"])
        if recs and "x6" in recs[0]:
            row["bias"] = bias(recs, kinds, mult_cache)
        out["arms"].append(row)
    out["predictions"] = _predictions(out["arms"])
    out["curve"] = [
        {
            "base": a["base"],
            "width": a["width"],
            "null_raw": a["null_raw"],
            "plac_raw": a["plac_raw"],
            "z_bias_null": ((a.get("bias") or {}).get("null") or {}).get("mean"),
        }
        for a in sorted(out["arms"], key=lambda a: (a["base"], a["width"]))
        if a["centre"] in ("told", "exact") and a["redraw"] in ("told", "exact")
    ]
    return out


def _md_step1(s1: Mapping[str, Any]) -> list[str]:
    lp = s1["label_prob"]
    lines = [
        "## X5 step 1 (EVAL records, read only)",
        "",
        f"Label probability at exactly .05 (S = {lp['S']}, {lp['nsim']} sims): "
        f"P(INVALID) = {_f(lp['p_invalid'], 4)}, P(VALID) = {_f(lp['p_valid'], 4)}.",
        "",
    ]
    for arm, c in s1["label_count"].items():
        lines.append(
            f"- {arm}: {c['invalid_observed']} INVALID of {c['rates']} E4 R3 C3 rates; expected "
            f"{_f(c['expected'], 2)}; P(>= observed | independent) = "
            f"{_f(c['p_ge_observed_indep'], 3)}"
        )
    fd = s1.get("fmax_dependence")
    if fd:
        p_ge = ", ".join(f"{a} {_f(p, 4)}" for a, p in fd["p_ge"].items())
        lines += [
            "",
            f"Dependence-aware (frozen `fmax_simulate`, added after the declaration): F_max "
            f"{fd['f_max']}, P(cell INVALID) {_f(fd['p_cell_invalid'], 4)}, count distribution "
            f"{fd['count_dist']} (nsim {fd['nsim']}); P(count >= observed): {p_ge}",
        ]
    lines += [
        "",
        "| arm | lam | n | kind | rate | label | KS D | KS p | q.01 | q.05 | rej .01 o/e | "
        "rej .025 o/e | rej .05 o/e | rej .10 o/e |",
        "|" + "---|" * 14,
    ]
    for c in s1["cells"]:
        for k in KIND_EDGE:
            x = c[k]
            if not x["n"]:
                continue
            tl = x["tail"]
            star = " *" if c["failing"] and (c["lam"], c["n"], k + "_raw") in X5_PRIMARY else ""
            tails = " | ".join(_f(tl[str(a)]["obs_exp"], 2) for a in TAIL_LEVELS)
            lines.append(
                f"| {c['arm']} | {c['lam']:g} | {c['n']} | {k} | {_f(x['frozen_rate'])} | "
                f"{x['frozen_label'] or ''}{star} | "
                f"{_f(x['ks_d'])} | {_f(x['ks_p'])} | {_f(x['qq']['0.01'])} | "
                f"{_f(x['qq']['0.05'])} | {tails} |"
            )
    lines += [
        "",
        "Overlap of rejecting seeds (p <= .05), P_placebo and P_placebo_conf in a cell:",
        "",
        "| arm | lam | n | plac | conf | both | expected | hypergeom p |",
        "|" + "---|" * 8,
    ]
    for c in s1["cells"]:
        o = c["overlap_plac_conf"]
        lines.append(
            f"| {c['arm']} | {c['lam']:g} | {c['n']} | {o['plac']} | {o['conf']} | {o['both']} | "
            f"{_f(o['expected'], 1)} | {_f(o['p_upper'])} |"
        )
    lines += [
        "",
        "Across the 4 lambdas at one n (EVAL lambdas share each seed's streams):",
        "",
        "| arm | n | kind | hits per lambda | seeds >= 2 (exp. indep.) | seeds >= 3 (exp.) |",
        "|" + "---|" * 6,
    ]
    for o in s1["overlap_lambda"]:
        lines.append(
            f"| {o['arm']} | {o['n']} | {o['kind']} | {o['per_lambda_hits']} | {o['seeds_ge2']} "
            f"({_f(o['expected_ge2'], 1)}) | {o['seeds_ge3']} ({_f(o['expected_ge3'], 1)}) |"
        )
    return lines


def _md_x5(x5t: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## X5 step 2 (fresh datasets)",
        "",
        f"Checks: `{json.dumps(x5t['checks'], sort_keys=True)}`",
        "",
        "Primary (pmrt_eq, one-sided exact binomial vs .05, Holm over 3):",
        "",
        "| lam | n | rate | hits/n | CP 95 % | p one-sided | p Holm | outcome |",
        "|" + "---|" * 8,
    ]
    for r in x5t["primary"]:
        cp = r.get("cp_ci") or [None, None]
        lines.append(
            f"| {r['lam']:g} | {r['n']} | {r['rate_key']} | {r.get('hits')}/"
            f"{r.get('n_', r.get('n'))} | "
            f"[{_f(cp[0])}, {_f(cp[1])}] | "
            f"{_f(r.get('p_binom_gt05'), 4)} | {_f(r.get('p_holm'), 4)} | {r.get('outcome')} |"
        )
    lines += [
        "",
        f"Answer to B: **{x5t['answer']}**",
        "",
        "| spec | arm | lam | n | kind | hits/n | rate | CP 95 % | boot 95 % | R-30 | "
        "p one-sided |",
        "|" + "---|" * 11,
    ]
    for c in x5t["cells"]:
        for k in KIND_EDGE:
            x = c[k]
            boot = x["boot_ci"] or [None, None]
            lines.append(
                f"| {c['spec']} | {c['arm']} | {c['lam']:g} | {c['n']} | {k} | "
                f"{x['hits']}/{x['n']} | "
                f"{_f(x['rate'])} | [{_f(x['cp_ci'][0])}, {_f(x['cp_ci'][1])}] | "
                f"[{_f(boot[0])}, {_f(boot[1])}] | "
                f"{x['r30'] or ''} | {_f(x['p_binom_gt05'], 4)} |"
            )
    lines += ["", "Step 3 (exact McNemar pmrt_eq vs pmrt_r3, read only if EXCESS):", ""]
    for s in x5t["step3"]:
        trig = "triggered" if s["triggered"] else "not triggered"
        lines.append(
            f"- lam {s['lam']:g} n {s['n']} {s['rate_key']}: paired {s['paired']}, eq only "
            f"{s['eq_only']}, r3 only {s['r3_only']}, p {_f(s['p_mcnemar_exact'], 4)}, {trig}"
        )
    return lines


def _md_x6(x6t: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## X6 (E1 R2 n 1000, 200 seeds)",
        "",
        f"Checks: `{json.dumps(x6t['checks'], sort_keys=True)}`",
        "",
        "| arm | width | centring | redraw | truth-null raw [boot CI] | label | "
        "P_placebo raw [CI] | "
        "mean z_bias null [CI] | rej bottom / top tercile | top - bottom [CI] | mean c |",
        "|" + "---|" * 11,
    ]
    for a in x6t["arms"]:
        b = (a.get("bias") or {}).get("null") or {}
        if b:
            tail = (
                f"{_f(b.get('mean'))} [{_f(b['ci'][0])}, {_f(b['ci'][1])}] | "
                f"{_f(b.get('rej_bottom'))} / {_f(b.get('rej_top'))} | "
                f"{_f(b.get('top_minus_bottom'))} "
                f"[{_f(b['top_minus_bottom_ci'][0])}, {_f(b['top_minus_bottom_ci'][1])}] | "
                f"{_f(b.get('mean_c'), 4)} |"
            )
        else:
            tail = "| | | |"
        lines.append(
            f"| {a['arm']} | {a['width']:g} | {a['centre']} | {a['redraw']} | "
            f"{rate_text(a['null_raw'])} | "
            f"{(a['null_raw'] or {}).get('validity') or ''} | {rate_text(a['plac_raw'])} | {tail}"
        )
    lines += ["", f"Predictions: `{json.dumps(x6t['predictions'], sort_keys=True)}`"]
    return lines


def tables_md(t: Mapping[str, Any]) -> str:
    """The X5 and X6 tables as markdown."""
    lines = [
        "# R-60 diagnostics X5 / X6: tables",
        "",
        f"EXPLORATORY, post hoc ({PROTOCOL_BASENAME} Amendments 2026-10-05). Rates are raw "
        "p <= .05 over testable candidates; boot CI = frozen seed-cluster bootstrap, "
        "CP = Clopper-Pearson. R-30 labels are descriptive.",
        "",
    ]
    if t.get("x5_step1"):
        lines += _md_step1(t["x5_step1"])
    if t.get("x5"):
        lines += _md_x5(t["x5"])
    if t.get("x6"):
        lines += _md_x6(t["x6"])
    return "\n".join(lines) + "\n"


def combine_tables(out_dir: str | Path, protocol_sha256: str | None = None) -> dict[str, Any]:
    """Merges the stage outputs `diag_*.json` of a directory.

    Writes `diag_tables.json` and `DIAG_TABLES.md` next to them.
    """
    d = Path(out_dir)
    t: dict[str, Any] = {"protocol_sha256": protocol_sha256 or CITED_PROTOCOL.sha256}
    for k in TOP:
        f = d / f"diag_{k}.json"
        if f.exists():
            t[k] = json.loads(f.read_text(encoding="utf-8"))
    write_text(d / "diag_tables.json", json.dumps(t, indent=1, sort_keys=True) + "\n")
    write_text(d / "DIAG_TABLES.md", tables_md(t))
    return t


def write_stage(res: Mapping[str, Any], key: str, out_dir: str | Path) -> None:
    """Writes the stage output `diag_<key>.json`."""
    write_text(
        Path(out_dir) / f"diag_{key}.json",
        json.dumps(res, indent=1, sort_keys=True, default=float) + "\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: `x5-step1`, `x5`, `x6` and `tables`."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("x5-step1")
    p.add_argument("--eval-merged", required=True)
    p.add_argument("--dependence", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("x5")
    p.add_argument("--fail", required=True)
    p.add_argument("--adj", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("x6")
    p.add_argument("--merged", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        print(tables_md(combine_tables(a.out)))
        return 0
    if a.cmd == "x5-step1":
        res, key = x5_step1(a.eval_merged, dependence=a.dependence), "x5_step1"
    elif a.cmd == "x5":
        res, key = x5(a.fail, a.adj), "x5"
    else:
        res, key = x6(a.merged), "x6"
    write_stage(res, key, a.out)
    print(tables_md({key: res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

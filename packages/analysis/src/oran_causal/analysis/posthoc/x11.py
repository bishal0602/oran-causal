"""X11: sensitivity of the knockout ground-truth labels and of method scores to the GT rule.

Post hoc and descriptive: no verdict, no frozen file changes, no new simulation. The frozen rule
(`sim.system.decision.gt`, orientation "dir") labels each (family, relation, KPI) cell from the
pooled unit mean and its 95 % episode-cluster percentile bootstrap interval:
TRUE(sign) if the interval excludes 0 and |mean| >= delta, NULL if it lies inside +-delta, INDET
otherwise or when unsupported. Variants move one factor at a time: delta x0.5 and x2, the pv floor
x0.5 and x2, and the interval level 90 and 99 % (the same bootstrap draws, other percentiles).

The per-unit GT records are the Kaggle outputs of the two GT kernels of each study. Methods are
every method of the stored pooled Study 1 slice and, for Study 2, the nine PMRT declaration
combinations and every DISC map except `MG:GT`, which is built from the GT itself. The score
convention is the frozen one (`core.scoring.confusion`, INDET excluded). INDET bounds: "wrong"
counts a declared INDET hypothesis as an FP and an undeclared one as an FN; "right" counts a
declared one as a TP with the right sign and an undeclared one as a TN.

Reproduction comes first and is a hard stop: the recomputed GT cells and every stored score must
equal the stored ones, and Study 1 pooled PMRT must give precision .92, recall .79 and labels
29 / 21 / 10.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc.paths import PRIMARY, RECORDS, write_text
from oran_causal.core.scoring import confusion
from oran_causal.sim.system.decision import gt as GT
from oran_causal.sim.system.decision.records import dec, iter_records
from oran_causal.sim.system.decision.vocab import (
    FAMILIES,
    HYPOTHESES,
    KPIS,
    ORIENTS,
    PRIMARY_ORIENT,
    RELATIONS,
)

X2_KPIS = ("pv", "v", "rlf", "e")
FLOOR_PV = GT.KPI_FLOOR["pv"]
LEVELS = (0.90, 0.95, 0.99)
VARIANTS: dict[str, tuple[float, float, float]] = {
    "frozen": (GT.REL_FRAC, FLOOR_PV, 0.95),
    "delta_x0.5": (GT.REL_FRAC * 0.5, FLOOR_PV, 0.95),
    "delta_x2": (GT.REL_FRAC * 2.0, FLOOR_PV, 0.95),
    "floor_x0.5": (GT.REL_FRAC, FLOOR_PV * 0.5, 0.95),
    "floor_x2": (GT.REL_FRAC, FLOOR_PV * 2.0, 0.95),
    "ci90": (GT.REL_FRAC, FLOOR_PV, 0.90),
    "ci99": (GT.REL_FRAC, FLOOR_PV, 0.99),
}
STATUSES = ("TRUE", "NULL", "INDET")
TOL = 1e-12
PAPER_S1 = (0.92, 0.79, (29, 21, 10))
DECLARED = "EXTRAS_PROTOCOL.md amendment 2026-10-11 X9-X13"
MAP_KINDS = {"MG:rand": "comparator map", "blanket2": "comparator map", "MG:PMRT": "PMRT map"}

Cell = dict[str, Any]
Decl = dict[tuple, dict[str, Any]]


class ReproductionError(RuntimeError):
    """The recomputed values do not equal the frozen ones; the analysis stops."""


@dataclasses.dataclass(frozen=True)
class Study:
    """The stored inputs of one study.

    Attributes:
        name: `S1` or `S2`.
        analysis: Stored analysis JSON (Study 1 `analysis_v4.json`, Study 2 `disc_conf.json`).
        label: The path recorded for the analysis in the output.
        job_inputs: The analysis job's input listing, which the GT file sizes are checked against.
        kernels: Directories of the GT kernel outputs.
    """

    name: str
    analysis: Path
    label: str
    job_inputs: Path
    kernels: tuple[Path, ...]


def studies(records: Path = RECORDS) -> dict[str, Study]:
    """The two studies, located in the frozen record sets."""
    s1, s2 = records / "study1_raw", records / "study2_raw"
    return {
        "S1": Study(
            "S1",
            s1 / "e6p-disc-v4an-pmrt-1" / "out" / "analysis_v4.json",
            "scratchpad/e6_dev/runs/e6p-disc-v4an-pmrt-1/out/analysis_v4.json",
            s1 / "e6p-disc-v4an-pmrt-1" / "job_inputs.txt",
            (s1 / "e6p-disc-v4gt-1-a", s1 / "e6p-disc-v4gt-1-b"),
        ),
        "S2": Study(
            "S2",
            s2 / "e6p-conf-discan-2" / "out" / "disc_conf.json",
            "scratchpad/e6_dev/runs/e6p-conf-discan-2/out/disc_conf.json",
            s2 / "e6p-conf-discan-2" / "job_inputs.txt",
            (s2 / "e6p-conf-gt-1-a", s2 / "e6p-conf-gt-1-b"),
        ),
    }


def sha(path: Path) -> str:
    """Sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(a: float, b: float) -> float:
    """Ratio, NaN for a zero denominator."""
    return float(a / b) if b else float("nan")


def nanf(x: float | None) -> float | None:
    """Float, None for a missing or non-finite value."""
    return None if x is None or not np.isfinite(x) else float(x)


def gt_files(study: Study) -> tuple[list[Path], list[dict[str, Any]]]:
    """`res_*.jsonl` of the study's GT kernels, sizes checked against the job's input listing."""
    listing = study.job_inputs.read_text(encoding="utf-8").splitlines()
    files, prov = [], []
    for kdir in study.kernels:
        k = kdir.name
        for p in sorted(kdir.glob("res_*.jsonl")):
            want = [
                ln.split()[-1]
                for ln in listing
                if ln.startswith("F ") and ln.split()[1].endswith(f"/{k}/{p.name}")
            ]
            size = p.stat().st_size
            if len(want) != 1 or int(want[0]) != size:
                raise ReproductionError(f"{p}: size {size} != job input listing {want}")
            files.append(p)
            prov.append(
                {"kernel": f"bishalpanta/{k}", "file": p.name, "bytes": size, "sha256": sha(p)}
            )
    return files, prov


def load_rows(files: Sequence[Path]) -> tuple[list[GT.Row], int]:
    """GT unit rows of the stage "gt" records, deduplicated by (seed, sub) and sorted."""
    eps = []
    for r in iter_records([str(f) for f in files], {"gt"}):
        labels = [dict(label, delta=dec(label["delta"])) for label in r["gt_labels"]]
        eps.append(
            {
                "seed": r["seed"],
                "n_cells": r["n_cells"],
                "gt_static": r["gt_static"],
                "gt_labels": labels,
                "sub": r.get("sub"),
            }
        )
    uniq = {(e["seed"], e.get("sub")): e for e in eps}
    eps = [uniq[k] for k in sorted(uniq, key=lambda k: (k[0], str(k[1])))]
    return GT.unit_rows(eps), len(eps)


def cell_stats(rows: Sequence[GT.Row], orient: str = PRIMARY_ORIENT) -> dict[tuple, dict[str, Any]]:
    """Per (family, relation, KPI): mean, n, n_eps and the interval at every level in `LEVELS`.

    The intervals come from one set of bootstrap draws per cell, drawn exactly as
    `gt.cluster_boot` does (same rng stream).
    """
    oi = ORIENTS.index(orient)
    out: dict[tuple, dict[str, Any]] = {}
    for fi, f in enumerate(FAMILIES):
        fr = [r for r in rows if r["family"] == f]
        ep = np.asarray([r["ep"] for r in fr])
        for ri, rel in enumerate(RELATIONS):
            for ki, k in enumerate(KPIS):
                y = np.asarray([r["val"][orient][rel][k] for r in fr], float)
                rng = np.random.default_rng([GT.GT_TAG, fi, ri, ki, oi])
                if len(y) == 0:
                    nan = (float("nan"), float("nan"))
                    out[(f, rel, k)] = {
                        "mean": float("nan"),
                        "n": 0,
                        "n_eps": 0,
                        "ci": dict.fromkeys(LEVELS, nan),
                    }
                    continue
                eps, inv = np.unique(ep, return_inverse=True)
                s = np.bincount(inv, y, len(eps))
                n = np.bincount(inv, minlength=len(eps)).astype(float)
                idx = rng.integers(len(eps), size=(GT.N_BOOT, len(eps)))
                bm = s[idx].sum(1) / n[idx].sum(1)
                ci = {}
                for level in LEVELS:
                    q = (0.025, 0.975) if level == 0.95 else ((1 - level) / 2, 1 - (1 - level) / 2)
                    lo, hi = np.quantile(bm, q)
                    ci[level] = (float(lo), float(hi))
                out[(f, rel, k)] = {
                    "mean": float(y.mean()),
                    "n": len(y),
                    "n_eps": len(eps),
                    "ci": ci,
                }
    return out


def label_table(
    stats: Mapping[tuple, Mapping[str, Any]], rel_frac: float, floor_pv: float, level: float
) -> dict[str, Any]:
    """GT cells, deltas and label counts for one variant of the rule."""
    floors = {**GT.KPI_FLOOR, "pv": floor_pv}
    delta = {}
    for k in KPIS:
        m = [
            abs(v["mean"])
            for (_f, _rel, kk), v in stats.items()
            if kk == k and np.isfinite(v["mean"])
        ]
        delta[k] = max(rel_frac * (max(m) if m else 0.0), float(floors.get(k, 0.0)))
    cells = []
    for (f, rel, k), v in stats.items():
        stat = GT.BootStat(v["mean"], v["ci"][level], v["n"], v["n_eps"])
        c = GT.classify(stat, delta[k])
        cells.append(
            {
                "family": f,
                "relation": rel,
                "kpi": k,
                "mean": v["mean"],
                "ci": list(v["ci"][level]),
                "n": v["n"],
                "n_eps": v["n_eps"],
                "delta": delta[k],
                **c,
            }
        )
    counts = {s: sum(c["status"] == s for c in cells) for s in STATUSES}
    return {"delta": delta, "cells": cells, "counts": counts}


def gt_reference(table: Mapping[str, Any]) -> dict[tuple, dict[str, Any]]:
    """The reference `{(family, relation, KPI): {status, sign}}` of a GT table."""
    return {
        (c["family"], c["relation"], c["kpi"]): {"status": c["status"], "sign": int(c["sign"])}
        for c in table["cells"]
    }


def check_cells(mine: Sequence[Cell], stored: Sequence[Cell], what: str) -> None:
    """Hard stop unless the recomputed GT cells equal the stored ones.

    Raises:
        ReproductionError: On a differing label, count, mean, interval or delta.
    """
    sk = {(c["family"], c["relation"], c["kpi"]): c for c in stored}
    bad = []
    for c in mine:
        s = sk[(c["family"], c["relation"], c["kpi"])]
        same = (
            c["status"] == s["status"]
            and c["sign"] == s["sign"]
            and c["n"] == s["n"]
            and c["n_eps"] == s["n_eps"]
            and np.allclose(
                [c["mean"], *c["ci"], c["delta"]],
                [s["mean"], *s["ci"], s["delta"]],
                rtol=TOL,
                atol=TOL,
            )
        )
        if not same:
            bad.append((c["family"], c["relation"], c["kpi"], c["status"], s["status"]))
    if bad or len(mine) != len(stored):
        raise ReproductionError(f"REPRODUCTION FAILED ({what}): {bad[:5]}")


def decl_from_list(lst: Sequence[Sequence[Any]]) -> Decl:
    """`[["f|rel|k", sign], ...]` as a declaration of every hypothesis."""
    d: Decl = {h: {"declared": False, "sign": 0} for h in HYPOTHESES}
    for h, s in lst:
        d[tuple(h.split("|"))] = {"declared": True, "sign": int(s)}
    return d


def decl_from_map(m: Sequence[Sequence[Any]]) -> Decl:
    """`[[f, rel, k, beta], ...]` as a declaration of every hypothesis."""
    d: Decl = {h: {"declared": False, "sign": 0} for h in HYPOTHESES}
    for f, rel, k, b in m:
        d[(f, rel, k)] = {"declared": True, "sign": int(np.sign(b)), "beta": b}
    return d


def _x2_scores(decl: Decl, ref: Mapping, dec_h: Sequence[tuple]) -> dict[str, Any]:
    t2 = f2 = so = nt = 0
    for h in dec_h:
        if h[2] not in X2_KPIS or ref[h]["status"] == "INDET":
            continue
        if ref[h]["status"] == "TRUE":
            nt += 1
            ok = decl[h]["sign"] == ref[h]["sign"]
            so += ok
            t2 += ok
            f2 += not ok
        else:
            f2 += 1
    return {
        "x2_precision": nanf(safe(t2, t2 + f2)),
        "x2_sign_acc": nanf(safe(so, nt)),
        "x2_n_scored": t2 + f2,
        "x2_n_true": nt,
    }


def scores(decl: Decl, ref: Mapping, cells: Sequence[Cell], *, x2: bool = False) -> dict[str, Any]:
    """Frozen-convention scores of a declaration, with INDET bounds and map-quality counts."""
    ov = confusion(decl, ref, HYPOTHESES)
    ind = confusion(decl, ref, HYPOTHESES, "nbr")
    dec_h = [h for h, v in decl.items() if v["declared"]]
    i_dec = sum(ref[h]["status"] == "INDET" for h in dec_h)
    i_und = sum(ref[h]["status"] == "INDET" for h in HYPOTHESES if not decl[h]["declared"])
    tp, fp, fn = ov["tp"], ov["fp"], ov["fn"]
    sign_ok = round(ov["sign_acc"] * tp) if tp else 0
    r: dict[str, Any] = {
        "n_declared": len(dec_h),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": ov["tn"],
        "indet_declared": i_dec,
        "indet_undeclared": i_und,
        "precision": nanf(ov["precision"]),
        "recall": nanf(ov["recall"]),
        "f1": nanf(ov["f1"]),
        "sign_acc": nanf(ov["sign_acc"]),
        "precision_indet_wrong": nanf(safe(tp, tp + fp + i_dec)),
        "precision_indet_right": nanf(safe(tp + i_dec, tp + fp + i_dec)),
        "recall_indet_wrong": nanf(safe(tp, tp + fn + i_und)),
        "recall_indet_right": nanf(safe(tp + i_dec, tp + fn + i_dec)),
        "sign_acc_indet_right": nanf(safe(sign_ok + i_dec, tp + i_dec)),
        "ind_precision": nanf(ind["precision"]),
        "ind_recall": nanf(ind["recall"]),
        "ind_f1": nanf(ind["f1"]),
        "far_declared": sum(1 for h in dec_h if h[1] == "far"),
    }
    st = {(c["family"], c["relation"], c["kpi"]): c for c in cells}
    q = {"true_right": 0, "true_wrong": 0, "null": 0, "indet": 0}
    for h in dec_h:
        c = st[h]
        if c["status"] == "TRUE":
            q["true_right" if decl[h]["sign"] == int(np.sign(c["mean"])) else "true_wrong"] += 1
        else:
            q["null" if c["status"] == "NULL" else "indet"] += 1
    n_true = sum(c["status"] == "TRUE" for c in cells)
    r.update({f"n_{k}": v for k, v in q.items()})
    r["precision_true_right"] = nanf(safe(q["true_right"], len(dec_h)))
    r["recall_true_right"] = nanf(safe(q["true_right"], n_true))
    if x2:
        r.update(_x2_scores(decl, ref, dec_h))
    return r


def close(a: Any, b: Any) -> bool:
    """Stored-versus-recomputed equality at `TOL`, a missing value equal to a non-finite one."""
    if a is None or b is None or (isinstance(b, float) and not np.isfinite(b)):
        return (a is None) and (b is None or not np.isfinite(b))
    return abs(float(a) - float(b)) < TOL


def _stored_scores(m: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "precision": m["ov_precision"],
        "recall": m["ov_recall"],
        "f1": m["ov_f1"],
        "sign_acc": m["sign_acc"],
        "ind_precision": m["ind_precision"],
        "ind_recall": m["ind_recall"],
        "ind_f1": m["ind_f1"],
        "n_declared": m["n_declared"],
        "far_declared": m["far_declared"],
    }


def methods_of(
    study: str, a: Mapping[str, Any]
) -> tuple[dict[str, tuple[str, Decl]], dict[str, dict[str, Any]]]:
    """`{method: (kind, declaration)}` of a stored analysis and its stored scores to reproduce."""
    meth: dict[str, tuple[str, Decl]] = {}
    stored: dict[str, dict[str, Any]] = {}
    if study == "S1":
        for k, m in a["slices_result"]["pooled"]["metrics"].items():
            if k == PRIMARY:
                kind = "PMRT"
            elif "+" in k and not k.startswith("v2"):
                kind = "PMRT variant"
            else:
                kind = "baseline"
            meth[k] = (kind, decl_from_list(m["declared"]))
            stored[k] = _stored_scores(m)
        return meth, stored
    for k, m in a["pmrt_combos"].items():
        meth[k] = ("PMRT" if k == PRIMARY else "PMRT variant", decl_from_list(m["declared"]))
        stored[k] = _stored_scores(m["metrics"])
    x2 = a["X"]["X2"]
    stored[PRIMARY].update(
        {
            "x2_precision": x2["precision"],
            "x2_sign_acc": x2["sign_accuracy"],
            "x2_n_scored": x2["n_scored"],
            "x2_n_true": x2["n_true"],
        }
    )
    for k, m in a["maps"].items():
        if k == "MG:GT":
            continue
        eq = a["edge_quality"][k]
        meth[k] = (MAP_KINDS.get(k, "baseline map"), decl_from_map(m))
        stored[k] = {
            "n_declared": eq["n_edges"],
            "n_true_right": eq["n_true_right"],
            "n_true_wrong": eq["n_true_wrong"],
            "n_null": eq["n_null"],
            "n_indet": eq["n_indet"],
            "precision_true_right": eq["precision_true_right"],
            "recall_true_right": eq["recall_true_right"],
        }
    return meth, stored


def label_text(c: Mapping[str, Any]) -> str:
    """`TRUE(+1)`, `NULL`, ... of a GT cell."""
    return c["status"] + ("" if not c["sign"] else f"({c['sign']:+d})")


def check_scores(
    study: str,
    meth: Mapping[str, tuple[str, Decl]],
    stored: Mapping[str, Mapping[str, Any]],
    ref: Mapping,
    cells: Sequence[Cell],
) -> None:
    """Hard stop unless the scores at the frozen setting equal the stored ones.

    Raises:
        ReproductionError: On any differing score.
    """
    bad = []
    for k, (_kind, decl) in meth.items():
        sc = scores(decl, ref, cells, x2=(study == "S2" and k == PRIMARY))
        for m, v in stored[k].items():
            if not close(sc[m], v):
                bad.append((k, m, sc[m], v))
    if bad:
        raise ReproductionError(f"REPRODUCTION FAILED ({study} scores): {bad[:8]}")


def _variant_entry(
    t: Mapping[str, Any],
    base: Mapping[tuple, Cell],
    meth: Mapping[str, tuple[str, Decl]],
    study: str,
    params: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = gt_reference(t)
    changed = []
    for c in t["cells"]:
        h = (c["family"], c["relation"], c["kpi"])
        if (c["status"], c["sign"]) != (base[h]["status"], base[h]["sign"]):
            changed.append(
                {
                    "cell": "|".join(h),
                    "frozen": label_text(base[h]),
                    "variant": label_text(c),
                    "mean": c["mean"],
                    "ci": c["ci"],
                    "delta": c["delta"],
                }
            )
    by_rel = {
        rel: {
            st: sum(c["status"] == st for c in t["cells"] if c["relation"] == rel)
            for st in STATUSES
        }
        for rel in RELATIONS
    }
    ms = {
        k: {
            "kind": kind,
            **scores(decl, ref, t["cells"], x2=(study == "S2" and k == PRIMARY)),
        }
        for k, (kind, decl) in meth.items()
    }
    entry = {
        **params,
        "delta": t["delta"],
        "counts": t["counts"],
        "counts_by_relation": by_rel,
        "changed_vs_frozen": changed,
        "methods": ms,
    }
    label_row = {
        **params,
        **t["counts"],
        **{f"{rel}_{st}": by_rel[rel][st] for rel in RELATIONS for st in by_rel[rel]},
        "delta_pv": t["delta"]["pv"],
        "n_changed": len(changed),
        "changed": "; ".join(f"{c['cell']} {c['frozen']}->{c['variant']}" for c in changed),
    }
    return entry, label_row


def analyse_study(study: Study) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """X11 for one study: the reproduction checks, then every variant and method.

    Returns:
        The study's output, its CSV score rows and its CSV label rows.

    Raises:
        ReproductionError: If the frozen GT cells, the stored scores or (Study 1) the paper
            values are not reproduced.
    """
    a = json.loads(study.analysis.read_text(encoding="utf-8"))
    files, prov = gt_files(study)
    rows, n_eps = load_rows(files)
    if n_eps != a["gt"]["n_episodes"] or len(rows) != a["gt"]["n_labels"]:
        raise ReproductionError(
            f"{study.name}: {n_eps} episodes / {len(rows)} labels != stored "
            f"{a['gt']['n_episodes']} / {a['gt']['n_labels']}"
        )
    check_cells(GT.gt_table(rows)["cells"], a["gt"]["cells"], f"{study.name} gt_table vs stored")
    stats = cell_stats(rows)
    tabs = {v: label_table(stats, *VARIANTS[v]) for v in VARIANTS}
    check_cells(
        tabs["frozen"]["cells"], a["gt"]["cells"], f"{study.name} x11 frozen variant vs stored"
    )
    meth, stored = methods_of(study.name, a)
    ref0 = gt_reference(tabs["frozen"])
    check_scores(study.name, meth, stored, ref0, tabs["frozen"]["cells"])
    if study.name == "S1":
        p = scores(meth[PRIMARY][1], ref0, tabs["frozen"]["cells"])
        paper = (
            round(p["precision"], 2),
            round(p["recall"], 2),
            tuple(tabs["frozen"]["counts"].values()),
        )
        if paper != PAPER_S1:
            raise ReproductionError(f"S1 paper values not reproduced: {paper}")
    base = {(c["family"], c["relation"], c["kpi"]): c for c in tabs["frozen"]["cells"]}
    out: dict[str, Any] = {
        "analysis": study.label,
        "analysis_sha256": sha(study.analysis),
        "gt_records": prov,
        "n_episodes": n_eps,
        "n_labels": len(rows),
        "reproduced": True,
        "variants": {},
    }
    score_rows, label_rows = [], []
    for v, t in tabs.items():
        params = {
            "rel_frac": VARIANTS[v][0],
            "floor_pv": VARIANTS[v][1],
            "ci_level": VARIANTS[v][2],
        }
        entry, label_row = _variant_entry(t, base, meth, study.name, params)
        out["variants"][v] = entry
        label_rows.append({"study": study.name, "variant": v, **label_row})
        score_rows += [
            {"study": study.name, "variant": v, "method": k, **m}
            for k, m in entry["methods"].items()
        ]
    return out, score_rows, label_rows


def run(records: Path = RECORDS, study_set: Mapping[str, Study] | None = None) -> dict[str, Any]:
    """X11 over both studies (`study_set`: other inputs than `studies(records)`).

    Returns:
        `{"result", "scores", "labels"}`: the JSON output and the two CSV tables as row lists.
    """
    res: dict[str, Any] = {
        "item": "X11",
        "declared": DECLARED,
        "post_hoc": True,
        "variants": {
            k: {"rel_frac": v[0], "floor_pv": v[1], "ci_level": v[2]} for k, v in VARIANTS.items()
        },
        "indet_bounds": "wrong: declared INDET = FP, undeclared INDET = FN; "
        "right: declared INDET = TP with the right sign, undeclared INDET = TN",
        "studies": {},
    }
    score_rows: list[dict] = []
    label_rows: list[dict] = []
    for name, study in (study_set or studies(records)).items():
        out, srows, lrows = analyse_study(study)
        res["studies"][name] = out
        score_rows += srows
        label_rows += lrows
    return {"result": res, "scores": score_rows, "labels": label_rows}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Writes rows with the union of their keys, in first-seen order, as CSV."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x11_gt_sensitivity.json`, `x11_gt_sensitivity.csv` and `x11_gt_labels.csv`."""
    d = Path(out_dir)
    write_text(d / "x11_gt_sensitivity.json", json.dumps(out["result"], indent=1))
    write_csv(d / "x11_gt_sensitivity.csv", out["scores"])
    write_csv(d / "x11_gt_labels.csv", out["labels"])


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: recompute X11 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.records))
    write(out, a.out)
    for name, s in out["result"]["studies"].items():
        fz = s["variants"]["frozen"]
        print(
            f"{name}: reproduced ({s['n_episodes']} eps, {s['n_labels']} labels, "
            f"labels {fz['counts']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

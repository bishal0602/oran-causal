import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.analysis.bridging import plan
from oran_causal.analysis.bridging import tables as tab
from oran_causal.experiments.bridging import bridge as B
from oran_causal.experiments.bridging import run as R
from oran_causal.experiments.campaign.run import RunOptions, run_units
from oran_causal.experiments.system import study1_discovery as d1
from oran_causal.sim.system.decision.vocab import FAMILIES, KPIS, RELATIONS

LOAD = [j for j, (_, k) in enumerate(plan.TARGETS) if k == "load"]


def fake_table(
    path: Path, stage: str, seeds: range, per_ep: int = 12, effect: float = 2.0, rng_seed: int = 0
) -> None:
    """A synthetic `e6units` table; on `eval`, sleep raises every nbr KPI by `effect` x action."""
    rng = np.random.default_rng(rng_seed)
    rows = [
        (s, fi, t)
        for s in seeds
        for fi in range(len(FAMILIES))
        for t in np.sort(rng.integers(90, 630, per_ep))
    ]
    seed, fam, t0 = (np.array(c) for c in zip(*rows, strict=True))
    n = len(rows)
    sgn = rng.choice([-1.0, 1.0], n)
    mode = rng.choice([0, 2], n)
    level = (mode == 0).astype(float)
    y = rng.normal(0, 1, (n, 15))
    if stage == "eval":
        nbr = [j for j, (r, _) in enumerate(plan.TARGETS) if r == "nbr"]
        sl = fam == FAMILIES.index("sleep")
        y[np.ix_(sl, nbr)] += effect * (sgn * level)[sl, None]
    arrays: dict[str, Any] = {
        "schema": np.array(B.E6UNITS_SCHEMA),
        "seed": seed.astype(np.int64),
        "episode": seed.astype(np.int64),
        "family": fam.astype(np.int8),
        "t0": t0.astype(np.int32),
        "c": rng.integers(0, 24, n).astype(np.int16),
        "sgn": sgn.astype(np.float32),
        "mode": mode.astype(np.int8),
        "level": level.astype(np.float32),
        "p1": np.full(n, 0.5),
        "probs": np.tile([0.5, 0.0, 0.5, 0.0], (n, 1)),
        "pre": rng.normal(10, 2, (n, 15)),
        "y": y,
        "ctx": rng.normal(size=(n, 3)),
        "ctx_keys": np.array(["a", "b", "c"]),
        "hist": rng.normal(size=(n, 8)),
        "tfrac": t0 / 720.0,
        "families": np.array(FAMILIES),
        "targets": np.array([f"{r}|{k}" for r, k in plan.TARGETS]),
        "stage": np.array(stage),
    }
    np.savez_compressed(path, **arrays)


def fake_gt(path: Path) -> None:
    """GT cells: sleep -> nbr TRUE (+), far INDET, the rest NULL."""
    cells = []
    for f, r, k in ((f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS):
        st = "TRUE" if (f == "sleep" and r == "nbr") else ("NULL" if r != "far" else "INDET")
        ci = [-1.0, 1.0] if st == "NULL" else [1.0, 2.0]
        cells.append({"family": f, "relation": r, "kpi": k, "status": st, "sign": 1, "ci": ci})
    path.write_text(json.dumps({"gt": {"cells": cells}}), encoding="utf-8")


def test_constants_match_the_v4_analyzer() -> None:
    assert plan.SEEDS_V4 == d1.SEEDS
    assert {k: d1.SPLIT[k] for k in plan.SPLIT_V4} == plan.SPLIT_V4
    ours = [(s.name, s.kind, s.split, s.seeds) for s in plan.slice_plan() if s.kind != "pooled"]
    lo, n = plan.SEEDS_V4["eval"]
    v4 = d1.slice_plan(range(lo, lo + n))
    theirs = [(s.name, s.kind, s.split, tuple(s.seeds)) for s in v4 if s.kind in plan.SIZED_KINDS]
    assert ours == theirs


def test_table_schema_is_checked(tmp_path: Path) -> None:
    p = tmp_path / "e6units_eval.npz"
    np.savez_compressed(p, schema=np.array("other/1"), stage=np.array("eval"))
    with pytest.raises(ValueError, match="schema"):
        B.Table.load(p)


def test_dataset_mapping(tmp_path: Path) -> None:
    p = tmp_path / "e6units_eval.npz"
    seeds = range(188100, 188110)
    fake_table(p, "eval", seeds)
    ds = B.make_dataset(B.Table.load(p), "sleep", seeds, 20, "s60_0")
    assert (ds.n, ds.X_action.shape, ds.Y.shape) == (120, (120, 2), (120, 15))
    assert set(np.unique(ds.X_action[:, 0])) <= {-1.0, 0.0, 1.0}
    prop = ds.designs[0].propensity
    assert prop is not None
    assert np.allclose(prop.sum(1), 1.0)
    assert ds.context is not None
    sg = ds.context[:, -2]
    assert np.all(prop[sg > 0, 0] == 0)
    assert np.all(prop[sg < 0, 2] == 0)
    assert len(ds.candidates) == 30
    assert ds.seed == plan.dataset_seed(20, "sleep") == 60_000_201
    assert ds.time_index is not None
    assert np.all(np.diff(ds.time_index) == 1)


def test_context_encoding_and_lag_identity(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    part = np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n))
    zero_obs = np.where(rng.random(n) < 0.3, np.nan, 0.0)
    m = np.column_stack([x, np.full(n, np.nan), np.ones(n), 2 * x + 1, part, zero_obs])
    c, names = B.encode_context(m, ["x", "allnan", "const", "dup", "part", "zero_obs"])
    assert names == ("x", "part", "part:missing", "zero_obs:missing")
    assert np.isfinite(c).all()
    assert c.shape == (n, 4)
    p = tmp_path / "e6units_eval.npz"
    seeds = range(188100, 188110)
    fake_table(p, "eval", seeds)
    z = dict(np.load(p))
    z["pre"][:, LOAD[2]] = 27000.0 - z["pre"][:, LOAD[0]] - z["pre"][:, LOAD[1]]
    np.savez_compressed(p, **z)
    ds = B.make_dataset(B.Table.load(p), "ptx", seeds, 20, "s60_0")
    assert ds.meta["lag_dropped_collinear"] == ["K_far_load"]
    assert np.isnan(ds.X_kpi_lag[:, LOAD[2]]).all()
    assert np.isfinite(ds.Y).all()


def test_truth_from_gt(tmp_path: Path) -> None:
    fake_gt(tmp_path / "gt.json")
    gt = plan.load_gt(tmp_path / "gt.json")
    t = B.truth_from_gt(gt, "sleep")
    assert t.edges == {("A_sleep", f"K_nbr_{k}") for k in KPIS}
    assert all(s == 1 for s in t.signs.values())
    assert ("A_sleep", "K_far_pv") not in t.edges | t.null_edges
    assert ("P_placebo", "K_far_pv") in t.null_edges
    log = B.truth_from_gt(gt, "sleep", placebo_log=True)
    assert not log.edges
    assert len(log.null_edges) == 30


def test_end_to_end(tmp_path: Path) -> None:
    tabs = tmp_path / "tables"
    tabs.mkdir()
    fake_table(tabs / "e6units_eval.npz", "eval", range(188100, 188220), per_ep=4)
    fake_table(tabs / "e6units_placebo.npz", "placebo", range(188700, 188740), per_ep=4, rng_seed=1)
    fake_gt(tmp_path / "gt.json")
    arms = {
        "corr": ("corr", {"arm": "native"}),
        "pcorr_eq": ("pcorr", {"arm": "eq"}),
        "pmrt_eq": ("pmrt_core", {"covariates": "eq"}),
    }
    spec: dict[str, Any] = {
        "name": "t",
        "budget_cpu_s": 600,
        "slices": ["60", "placebo"],
        "gt": str(tmp_path / "gt.json"),
        "counts": str(tmp_path / "counts.json"),
        "arms": {a: {"ref": m, "config": c, "declare": "by"} for a, (m, c) in arms.items()},
    }
    c = R.counts_from_tables(spec, tabs)
    Path(spec["counts"]).write_text(json.dumps({"counts": c}), encoding="utf-8")
    assert (c["s60_0"]["sleep"], c["placebo"]["sleep"], c["s60_5"]["sleep"]) == (240, 160, 0)
    units = R.expand(spec, c)
    assert len(units) == 3 * 4 * 3
    assert len(plan.spec_slices(spec)) == 11
    out = tmp_path / "res.jsonl"
    opts = R.with_loader(RunOptions(budget=600, isolate=False), R.Loader.of(spec, tabs))
    run_units(R.campaign_spec(spec), units, out, opts)
    recs = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == len(units)
    assert all(r["status"] == "ok" for r in recs), [r.get("error") for r in recs][:1]
    assert {r["job"]["seed"] for r in recs} == {u.seed for u in units}
    summ = R.merge(spec, [out], tmp_path / "merged.jsonl")
    assert summ["n_ok"] == len(units)
    assert not summ["errors"]
    assert not summ["missing"]
    t = tab.tables(spec, tmp_path / "merged.jsonl", spec["gt"], tmp_path / "tab")
    cell = t["cells"]["pmrt_eq|60|raw"]
    assert cell["rec"]["rate"] > 0.8
    assert cell["null"]["n"] > 0
    assert t["cells"]["pmrt_eq|placebo|raw"]["plac_log"]["n"] == 4 * 15
    assert (tmp_path / "tab" / tab.TABLES_MD).exists()

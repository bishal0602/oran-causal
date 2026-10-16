import json
import math

import numpy as np
import pytest

from oran_causal.core.scoring import placebo_tau
from oran_causal.experiments.campaign import run as R
from oran_causal.experiments.campaign.aggregate import (
    aggregate,
    cluster_ci,
    rate,
    seed_row,
    to_result,
    wilson_deff,
)
from oran_causal.experiments.campaign.cost import (
    design_effect,
    measured_infeasible,
    null_rate_seeds,
    paired_seeds,
    project,
    recall_power,
)
from oran_causal.experiments.campaign.merge import iter_jsonl, merge
from oran_causal.experiments.campaign.run import Shard, run_part, run_units
from oran_causal.experiments.campaign.spec import expand
from oran_causal.sim.controlled.datasets.truth import truth_for

ONE = Shard(0, 1)


def records(kit, opts, tmp_path, spec):
    out = tmp_path / "r.jsonl"
    run_part(spec, ONE, out, opts())
    return kit.read_jsonl(out)


def test_cluster_ci_and_wilson():
    lo, hi = cluster_ci(np.array([1, 0, 2, 1]), np.array([20, 20, 20, 20]))
    assert lo is not None
    assert hi is not None
    assert lo <= 4 / 80 <= hi
    assert cluster_ci(np.zeros(5), np.full(5, 10)) == [0.0, 0.0]
    w = wilson_deff(np.zeros(5), np.full(5, 10))
    assert w[0] == 0.0
    assert w[1] is not None
    assert 0 < w[1] < 0.1
    r = rate(np.full(10, 5), np.full(10, 10))
    assert r["rate"] == 0.5
    assert r["above"]
    assert not rate(np.ones(10), np.full(10, 20))["above"]


@pytest.mark.claim("PRO-TUNE")
def test_aggregate_tau_from_tune_seeds_only(kit, opts, tmp_path):
    spec = kit.small_spec()
    recs = records(kit, opts, tmp_path, spec)
    agg = aggregate(recs, spec)
    c = agg["cells"]["d_tau|E4|R1|lam1|k0.25|n300"]
    tune = [
        to_result(r)
        for r in recs
        if r["arm"] == "d_tau" and r["role"] == "tune" and r["job"]["n"] == 300
    ]
    assert c["tau"] == placebo_tau(tune)
    assert (c["n_tune"], c["n_measure"]) == (2, 2)
    truth = truth_for("E4", "R1")
    m = [r for r in recs if r["arm"] == "d_tau" and r["role"] == "measure" and r["job"]["n"] == 300]
    exp_plac = sum(
        1 for r in m for e in r["edges"] if e["source"] == "P_placebo" and e["score"] > c["tau"]
    )
    assert c["primary"]["plac_decl"]["hits"] == exp_plac
    n_plac = sum(1 for e in truth.null_edges if e[0] == "P_placebo")
    assert c["primary"]["plac_decl"]["n"] == 2 * n_plac
    u = agg["cells"]["d_tau|E4|R1|lam1|k0.25|n400"]
    assert u["tau"] is None
    assert u["primary"] is None
    assert "untuned" in u["note"]
    b = agg["cells"]["d_by|E4|R1|lam1|k0.25|n300"]
    raw = sum(
        1
        for r in recs
        if r["arm"] == "d_by" and r["role"] == "measure" and r["job"]["n"] == 300
        for e in r["edges"]
        if (e["source"], e["target"]) in truth.null_edges
        and e["source"] != "P_placebo"
        and not e["source"].startswith("K")
        and e["p"] <= 0.05
    )
    assert b["primary"]["null_raw05"]["hits"] == raw
    assert set(b["primary"]["per_seed"]) == {"3000100", "3000101"}
    assert b["primary"]["valid_flag"] == "few_seeds"
    assert "secondary_tau" in b
    assert agg["cost"]["d_by|E4|R1|n300"]["n_runs"] == 4


def test_rate_denominators_exclude_not_testable(kit, opts, tmp_path):
    spec = kit.make_spec(
        blocks=[
            {
                "role": "measure",
                "worlds": ["E1"],
                "regimes": ["R1"],
                "ns": [300],
                "kappas": [0.25],
                "seeds": [3000100],
            }
        ]
    )
    r = next(x for x in records(kit, opts, tmp_path, spec) if x["arm"] == "d_by")
    truth = truth_for("E1", "R1")
    null = sorted(e for e in truth.null_edges if not e[0].startswith(("K", "P_")))[0]
    plac = sorted(e for e in truth.null_edges if e[0] == "P_placebo")[0]
    nt = [f"{s}->{t}" for s, t in (null, plac)]
    for e in r["edges"]:
        if f"{e['source']}->{e['target']}" in nt:
            e.update(score=None, p=None, declared=True)
    r["notes"] = {"not_testable_edges": nt}
    row = seed_row(r, to_result(r), truth)
    n_null = sum(1 for e in truth.null_edges if not e[0].startswith(("K", "P_")))
    n_plac = sum(1 for e in truth.null_edges if e[0] == "P_placebo")
    assert (row["n_null"], row["n_plac"]) == (n_null - 1, n_plac - 1)
    assert row["raw_null"][1] == n_null - 1
    assert row["nt_null"] == 1


def test_streaming_aggregate_equals_in_memory(kit, opts, tmp_path):
    spec = kit.small_spec()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_part(spec, Shard(0, 2), a, opts())
    run_part(spec, Shard(1, 2), b, opts())
    merge(spec, [a, b], tmp_path / "m.jsonl.gz")
    recs = list(iter_jsonl(tmp_path / "m.jsonl.gz"))
    streamed = aggregate(iter_jsonl(tmp_path / "m.jsonl.gz"), spec, stream_sorted=True)
    in_memory = aggregate(recs, spec)
    assert json.dumps(streamed, sort_keys=True, default=str) == json.dumps(
        in_memory, sort_keys=True, default=str
    )
    with pytest.raises(ValueError, match="not sorted"):
        aggregate(recs[1:] + recs[:1], spec, stream_sorted=True)


def test_project_counts_and_caps(kit, opts, tmp_path):
    spec = kit.small_spec()
    agg = aggregate(records(kit, opts, tmp_path, spec), spec)
    pr = project(agg, spec)
    assert pr["n_units"] == len(expand(spec))
    assert pr["cpu_h_total"] >= 0
    assert not pr["arms_without_cost"]
    for v in agg["cost"].values():
        v["cpu_s_mean"] = 10.0
    capped = project(agg, {**spec, "budget_cpu_s": 1.0})
    assert sum(capped["units_capped_at_budget"].values()) == pr["n_units"]
    assert math.isclose(capped["cpu_h_total"] - capped["cpu_h_generation"], pr["n_units"] / 3600)


def test_project_splits_gpu_wall_from_cpu(kit, opts, tmp_path, monkeypatch):
    real = R.run_one
    monkeypatch.setattr(
        R, "run_one", lambda *a, **k: {**real(*a, **k), "cpu_s": 2.0, "wall_s": 3.0}
    )
    spec = kit.small_spec()
    spec["arms"]["d_tau"].update(budget_cpu_s=None, budget_wall_s=7200.0)
    pr = project(aggregate(records(kit, opts, tmp_path, spec), spec), spec)
    n_tau = sum(u.arm == "d_tau" for u in expand(spec))
    n_by = sum(u.arm == "d_by" for u in expand(spec))
    assert math.isclose(pr["gpu_wall_h_by_arm"]["d_tau"], 3.0 * n_tau / 3600)
    assert math.isclose(pr["cpu_h_by_arm"]["d_by"], 2.0 * n_by / 3600)
    assert "d_tau" not in pr["cpu_h_by_arm"]


def test_measured_infeasible_and_projection_of_unrun_units(kit, opts, tmp_path, monkeypatch):
    real = R.run_one
    monkeypatch.setattr(
        R, "run_one", lambda *a, **k: {**real(*a, **k), "cpu_s": 5.0, "wall_s": 50.0}
    )
    spec = kit.small_spec()
    spec["arms"]["d_by"]["budget_cpu_s"] = 1.0
    spec["arms"]["d_tau"].update(budget_cpu_s=None, budget_wall_s=10.0)
    units = [u for u in expand(spec) if u.role == "measure"]
    reg = tmp_path / "registry.jsonl"
    first = [u for u in units if u.seed == 3000100 and u.n == 300]
    rest = [u for u in units if u not in first]
    run_units(spec, first, tmp_path / "p1.jsonl", opts(budget=7200, registry=reg))
    run_units(spec, rest, tmp_path / "p2.jsonl", opts(budget=7200, registry=reg))
    recs = kit.read_jsonl(tmp_path / "p1.jsonl") + kit.read_jsonl(tmp_path / "p2.jsonl")
    agg = aggregate(recs, spec)
    assert measured_infeasible(agg) == {"d_by": (300, 5.0, "CPU-s"), "d_tau": (300, 50.0, "wall-s")}
    pr = project(agg, spec)
    assert sum(pr["units_infeasible_not_run"].values()) == len(expand(spec))
    assert pr["cpu_h_total"] == pr["cpu_h_generation"]
    assert pr["gpu_wall_h_total"] == 0


def test_paired_seeds_and_null_precision():
    assert paired_seeds(0.0) == 2
    s = paired_seeds(0.3)
    assert 32 <= s <= 36
    assert paired_seeds(0.6) > s
    assert null_rate_seeds(1, 1.0) == 457
    assert null_rate_seeds(16, 1.0) == 29
    assert null_rate_seeds(0) is None
    assert design_effect([0, 0, 0], [5, 5, 5]) == 1.0
    assert design_effect([5, 0, 5, 0], [5, 5, 5, 5]) > 1.0


def test_recall_power_uses_valid_paired_seeds():
    def cell(arm, recalls, flag="ok"):
        per_seed = {str(i): {"recall": r} for i, r in enumerate(recalls)}
        return {"arm": arm, "primary": {"valid_flag": flag, "per_seed": per_seed}}

    agg = {
        "cells": {
            "pmrt_eq|E1|R1|k0.25|n500": cell("pmrt_eq", [1, 0.5, 1, 0.75]),
            "corr|E1|R1|k0.25|n500": cell("corr", [0.5, 0.5, 0.75, 0.25]),
            "bad|E1|R1|k0.25|n500": cell("bad", [0, 0, 0, 0], "ABOVE"),
        }
    }
    out = recall_power(agg)
    c = out["cells"]["E1|R1|k0.25|n500"]
    assert c["invalid_arms"] == ["bad"]
    assert len(c["pairs"]) == 1
    pr = c["pairs"][0]
    assert math.isclose(abs(pr["mean_d"]), 0.3125)
    assert pr["seeds_needed"] == paired_seeds(pr["sd_d"])
    assert out["max_over_cells_focal"] == pr["seeds_needed"]

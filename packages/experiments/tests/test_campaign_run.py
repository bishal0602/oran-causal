import dataclasses
import json
import sys
import time

import numpy as np
import pytest

from oran_causal.experiments.campaign import run as R
from oran_causal.experiments.campaign.integrity import check_eval_preconditions
from oran_causal.experiments.campaign.provenance import PythonPin
from oran_causal.experiments.campaign.run import (
    Shard,
    arm_budgets,
    read_infeasible_registry,
    run_part,
    run_units,
)
from oran_causal.experiments.campaign.spec import ROLES, expand, spec_sha256
from oran_causal.experiments.harness.isolation import IsolatedOutcome, classify_kill

LINUX = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="fork isolation is Linux-only"
)
ONE = Shard(0, 1)


def costing(monkeypatch, **fields):
    real = R.run_one
    monkeypatch.setattr(R, "run_one", lambda *a, **k: {**real(*a, **k), **fields})


def test_run_resumable_and_provenance(kit, opts, tmp_path):
    spec = kit.small_spec()
    out = tmp_path / "r.jsonl"
    n_units = len(expand(spec))
    assert run_part(spec, ONE, out, opts()) == n_units
    assert run_part(spec, ONE, out, opts()) == 0
    recs = kit.read_jsonl(out)
    assert len(recs) == n_units
    r = recs[0]
    assert r["status"] == "ok"
    assert r["arm"] in spec["arms"]
    assert r["role"] in ROLES
    assert r["code"]["commit"] == kit.COMMIT["commit"]
    assert r["host"]["platform"] == "test"
    assert r["campaign"] == "oran-causal-campaign/1"
    assert r["versions"]["runner"] == "oran-causal-runner/1"
    assert r["key"].startswith(r["arm"] + "|E4|R1|lam1|k0.25|n")
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2])
    assert run_part(spec, ONE, out, opts()) == 1


def test_budget_marks_infeasible_and_cascades(kit, opts, tmp_path, monkeypatch):
    costing(monkeypatch, cpu_s=5.0)
    spec = kit.small_spec()
    out = tmp_path / "b.jsonl"
    run_part(spec, ONE, out, opts(budget=1.0))
    recs = {r["key"]: r for r in kit.read_jsonl(out)}
    assert all(r["status"] == "infeasible" for r in recs.values())
    reasons = [r["reason"] for r in recs.values()]
    assert any(x.startswith("exceeded the budget") for x in reasons)
    assert any(x.startswith("not run: n 300") for x in reasons)
    assert run_part(spec, ONE, out, opts(budget=1.0)) == 0


@LINUX
def test_isolated_run_matches_in_process(kit, opts, tmp_path):
    spec = kit.small_spec()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_part(spec, ONE, a, opts())
    run_part(spec, ONE, b, opts(isolate=True))
    ra = {r["key"]: r for r in kit.read_jsonl(a)}
    rb = {r["key"]: r for r in kit.read_jsonl(b)}
    assert ra.keys() == rb.keys()
    for k in ra:
        assert ra[k]["edges"] == rb[k]["edges"]
        assert rb[k]["child_cpu_s"] >= 0


@pytest.mark.claim("PRO-FROZEN")
def test_records_stamp_commit_protocol_spec_sha(kit, opts, tmp_path):
    spec = kit.small_spec()
    out = tmp_path / "r.jsonl"
    run_part(spec, ONE, out, opts(spec_file_sha256="f" * 64))
    r = kit.read_jsonl(out)[0]
    it = r["integrity"]
    assert it["code_commit"] == kit.COMMIT["commit"]
    assert it["code_dirty"] is False
    assert it["spec_sha256"] == spec_sha256(spec)
    assert it["spec_file_sha256"] == "f" * 64
    assert it["python"]
    assert "protocol_sha256" in it
    assert r["run_mode"]["mode"] == "dev"


def test_classify_kill():
    assert classify_kill(24, 3.0, 100.0) == "cpu_limit"
    assert classify_kill(9, 110.0, 100.0) == "cpu_limit"
    assert classify_kill(9, 12.0, 100.0) == "killed"
    assert classify_kill(11, 12.0, 100.0) == "killed"
    assert classify_kill(None, 1.0, 100.0, wall_hit=True) == "wall_limit"


def test_eval_preconditions(kit):
    texts, sha = kit.frozen_texts()
    spec = kit.eval_spec(sha)
    good = {"commit": "abc", "dirty": False}
    with pytest.raises(ValueError, match="not clean"):
        check_eval_preconditions(
            spec, {"commit": "abc", "dirty": None}, {"mismatches": {}}, texts, None
        )
    with pytest.raises(ValueError, match=r"no uv\.lock"):
        check_eval_preconditions(spec, good, {"mismatches": None}, texts, None)
    with pytest.raises(ValueError, match="differ from the lock"):
        check_eval_preconditions(
            spec, good, {"mismatches": {"numpy": ["2.4.2", "2.0.2"]}}, texts, None
        )
    with pytest.raises(ValueError, match=r"the workspace pins 0\.0 \(>=0\)"):
        check_eval_preconditions(spec, good, {"mismatches": {}}, texts, PythonPin("0.0", ">=0"))
    check_eval_preconditions(spec, good, {"mismatches": {}}, texts, None)


def eval_opts(kit, opts, **changes):
    texts, _ = kit.frozen_texts()
    pins = {
        "lock": "l",
        "lock_sha256": "1" * 64,
        "n_pins": 0,
        "mismatches": {},
        "torch_build": None,
    }
    providers = {"texts": lambda: texts, "pins": lambda python: dict(pins), "quota": lambda: 4}
    providers.update(changes.pop("providers", {}))
    return opts(providers=providers, **changes)


def test_eval_run_refused_before_any_data(kit, opts, tmp_path, monkeypatch):
    _, sha = kit.frozen_texts()
    spec = kit.eval_spec(sha)
    monkeypatch.setattr(R, "generate_dataset", lambda *a, **k: pytest.fail("EVAL data generated"))
    bad_pins = {"pins": lambda python: {"mismatches": {"numpy": ["2.4.2", "2.0.2"]}}}
    out = tmp_path / "eval.jsonl"
    with pytest.raises(ValueError, match="EVAL refused"):
        run_part(spec, ONE, out, eval_opts(kit, opts, isolate=True, providers=bad_pins))
    with pytest.raises(ValueError, match="9 campaign processes on 4 vCPU"):
        run_part(spec, ONE, out, eval_opts(kit, opts, isolate=True, providers={"procs": lambda: 9}))
    assert not out.exists()


def test_eval_rules_cap_is_infeasible_no_propagation_and_t3(kit, opts, tmp_path, monkeypatch):
    costing(monkeypatch, cpu_s=5.0)
    spec = kit.small_spec()
    units = expand(spec)
    out = tmp_path / "e.jsonl"
    run_units(spec, units, out, opts(budget=2.0, eval_rules=True))
    recs = kit.read_jsonl(out)
    assert len(recs) == len(units)
    assert all(r["status"] == "infeasible" and r["cpu_s"] == 5.0 for r in recs)
    assert all("safety cap 4" in r["reason"] and "never re-run" in r["reason"] for r in recs)
    assert run_units(spec, units, out, opts(budget=2.0, eval_rules=True)) == 0
    out2 = tmp_path / "e2.jsonl"
    run_units(spec, units, out2, opts(budget=3.0, eval_rules=True))
    assert all(r["status"] == "ok" for r in kit.read_jsonl(out2))
    spec_t3 = kit.small_spec()
    spec_t3["arms"]["d_by"]["infeasible_n"] = {"400": 9100.0}
    out3 = tmp_path / "t3.jsonl"
    run_part(spec_t3, ONE, out3, opts())
    t3 = [r for r in kit.read_jsonl(out3) if r["arm"] == "d_by" and r["job"]["n"] == 400]
    assert t3
    assert all(r["status"] == "infeasible" and "DEV measured cost 9100" in r["reason"] for r in t3)


def test_dev_and_eval_outputs_never_mix(kit, opts, tmp_path):
    spec = kit.small_spec()
    out = tmp_path / "r.jsonl"
    out.write_text(
        json.dumps({"key": "x", "run_mode": {"mode": "eval", "protocol_sha256": "a" * 64}}) + "\n"
    )
    with pytest.raises(ValueError, match="another run mode"):
        run_part(spec, ONE, out, opts())


def budget_spec(kit):
    s = kit.small_spec()
    s["arms"]["d_by"]["budget_cpu_s"] = 1.0
    s["arms"]["d_tau"].update(budget_cpu_s=None, budget_wall_s=10.0)
    return s


def test_arm_budgets(kit):
    s = budget_spec(kit)
    assert arm_budgets(s, "d_by", 7200.0) == (1.0, None)
    assert arm_budgets(s, "d_tau", 7200.0) == (None, 10.0)
    assert arm_budgets(s, "d_e3", 7200.0) == (7200.0, None)


def test_registry_shared_across_processes_and_wall_budget(kit, opts, tmp_path, monkeypatch):
    costing(monkeypatch, cpu_s=5.0, wall_s=50.0)
    spec = budget_spec(kit)
    units = [u for u in expand(spec) if u.role == "measure"]
    reg = tmp_path / "registry.jsonl"
    first = [u for u in units if u.seed == 3000100 and u.n == 300]
    rest = [u for u in units if u not in first]
    run_units(spec, first, tmp_path / "p1.jsonl", opts(budget=7200, registry=reg))
    p1 = kit.read_jsonl(tmp_path / "p1.jsonl")
    assert all(r["status"] == "infeasible" for r in p1)
    assert {r["cost_kind"] for r in p1} == {"CPU-s", "wall-s"}
    assert {r["arm"]: r["cost"] for r in p1} == {"d_by": 5.0, "d_tau": 50.0}
    assert read_infeasible_registry(reg) == {
        "d_by": (300, 5.0, "CPU-s"),
        "d_tau": (300, 50.0, "wall-s"),
    }
    run_units(spec, rest, tmp_path / "p2.jsonl", opts(budget=7200, registry=reg))
    p2 = kit.read_jsonl(tmp_path / "p2.jsonl")
    assert len(p2) == len(rest)
    assert all(r["status"] == "infeasible" for r in p2)
    assert all(
        r["reason"].startswith("not run: arm infeasible at n 300 in this session") for r in p2
    )


@LINUX
def test_isolated_wall_limit_kills_child(kit, opts, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "run_one", lambda *a, **k: time.sleep(30) or {})
    spec = kit.small_spec()
    spec["arms"]["d_tau"].update(budget_cpu_s=None, budget_wall_s=1.0)
    units = [u for u in expand(spec) if u.arm == "d_tau"][:1]
    t0 = time.perf_counter()
    run_units(spec, units, tmp_path / "w.jsonl", opts(budget=7200, isolate=True))
    r = kit.read_jsonl(tmp_path / "w.jsonl")[0]
    assert time.perf_counter() - t0 < 15
    assert r["status"] == "infeasible"
    assert r["cost_kind"] == "wall-s"


def test_method_load_failure_is_an_error_record_not_a_crash(kit, opts, tmp_path):
    spec = kit.small_spec()
    spec["arms"]["d_by"]["ref"] = "no_such_arm"
    out = tmp_path / "r.jsonl"
    run_part(spec, ONE, out, opts())
    recs = kit.read_jsonl(out)
    bad = [r for r in recs if r["arm"] == "d_by"]
    assert len(recs) == len(expand(spec))
    assert bad
    assert all(r["status"] == "error" and "method load failed" in r["error"] for r in bad)
    assert all(r["status"] == "ok" for r in recs if r["arm"] == "d_tau")


def test_records_log_load(kit, opts, tmp_path):
    spec = kit.small_spec()
    load = {"loadavg": [1.0, 1.0, 1.0], "campaign_procs": 1, "cpu_quota": 1}
    run_units(
        spec, expand(spec)[:2], tmp_path / "d.jsonl", opts(providers={"load": lambda: dict(load)})
    )
    for r in kit.read_jsonl(tmp_path / "d.jsonl"):
        assert set(r["load"]) == {"start", "end"}
        assert r["load"]["start"]["campaign_procs"] == 1


def fake_fork(monkeypatch):
    kill: dict[str, tuple[int, float]] = {}

    def child(task, limit_s, wall_s=None):
        rec = task()
        if rec["key"] in kill:
            sig, cpu = kill[rec["key"]]
            return IsolatedOutcome(None, cpu, 50.0, sig, cpu, False)
        return IsolatedOutcome(
            rec, rec.get("cpu_s") or 0.0, 50.0, None, rec.get("wall_s") or 0.0, False
        )

    monkeypatch.setattr(R, "run_isolated", child)
    return kill


def budget_eval_spec(kit, budget=100.0):
    _, sha = kit.frozen_texts()
    spec = kit.eval_spec(sha)
    spec["budget_cpu_s"] = budget
    return spec


def test_eval_cap_breach_is_infeasible_never_rerun_oom_is_error(kit, opts, tmp_path, monkeypatch):
    kill = fake_fork(monkeypatch)
    spec = budget_eval_spec(kit)
    o = eval_opts(kit, opts, isolate=True)
    units = expand(spec, o.providers.texts())[:3]
    kill[units[0].key] = (24, 201.0)
    kill[units[1].key] = (9, 12.0)
    out = tmp_path / "e.jsonl"
    assert run_units(spec, units, out, o) == 3
    r = {x["key"]: x for x in kit.read_jsonl(out)}
    cap, oom, ok = r[units[0].key], r[units[1].key], r[units[2].key]
    assert cap["status"] == "infeasible"
    assert cap["cpu_s"] == 201.0
    assert "never re-run" in cap["reason"]
    assert oom["status"] == "error"
    assert "signal 9" in oom["error"]
    assert ok["status"] == "ok"
    assert run_units(spec, units, out, o) == 1
    assert [x["status"] for x in kit.read_jsonl(out)].count("infeasible") == 1
    assert cap["limits"] == {"cpu_s": 200.0, "wall_s": None}
    integ = {k: ok["integrity"][k] for k in ("isolation", "budget_cpu_s", "cap_factor")}
    assert integ == {"isolation": "fork+rlimit", "budget_cpu_s": 100.0, "cap_factor": 2.0}


def test_eval_refuses_cli_budget_and_no_isolation(kit, opts, tmp_path, monkeypatch):
    fake_fork(monkeypatch)
    spec = budget_eval_spec(kit)
    o = eval_opts(kit, opts, isolate=True)
    units = expand(spec, o.providers.texts())[:2]
    out = tmp_path / "e.jsonl"
    with pytest.raises(ValueError, match="F6"):
        run_units(spec, units, out, dataclasses.replace(o, budget=2784.0))
    with pytest.raises(ValueError, match="F6"):
        run_part(spec, ONE, out, dataclasses.replace(o, budget=50.0))
    with pytest.raises(ValueError, match="F7"):
        run_units(spec, units, out, dataclasses.replace(o, budget=100.0, isolate=False))
    assert not out.exists()
    dev = kit.small_spec()
    d = tmp_path / "d.jsonl"
    run_units(dev, expand(dev)[:2], d, opts(budget=7.0))
    rec = kit.read_jsonl(d)[0]
    assert rec["status"] == "ok"
    assert rec["integrity"]["isolation"] == "in-process"
    assert rec["integrity"]["budget_cpu_s"] == 7.0
    assert rec["limits"]["cpu_s"] == 7.0


def test_eval_skip_from_applies_the_merge_accept_rule(kit, opts, tmp_path, monkeypatch):
    fake_fork(monkeypatch)
    spec = budget_eval_spec(kit)
    o = eval_opts(kit, opts, isolate=True)
    n_all = len(expand(spec, o.providers.texts()))
    dev = tmp_path / "dev.jsonl"
    run_part(kit.small_spec(), ONE, dev, opts())
    ev1 = tmp_path / "ev1.jsonl"
    assert run_part(spec, Shard(0, 1, skip_complete_from=[dev]), ev1, o) == n_all
    other = dict(spec, name="another EVAL spec")
    n_other = len(expand(other, o.providers.texts()))
    assert (
        run_part(other, Shard(0, 1, skip_complete_from=[ev1]), tmp_path / "o.jsonl", o) == n_other
    )
    slim = tmp_path / "done.jsonl"
    rows = [
        {
            "key": r["key"],
            "status": r["status"],
            "error": r["error"],
            "run_mode": r["run_mode"],
            "integrity": {"spec_sha256": r["integrity"]["spec_sha256"]},
        }
        for r in kit.read_jsonl(ev1)
    ]
    slim.write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert run_part(spec, Shard(0, 1, skip_complete_from=[slim]), tmp_path / "ev2.jsonl", o) == 0


def test_partition_stamps_part_and_cost_table(kit, opts, tmp_path, monkeypatch):
    fake_fork(monkeypatch)
    spec = budget_eval_spec(kit)
    o = eval_opts(kit, opts, isolate=True)
    ct = {
        f"{u.arm}|{u.world}|{u.regime}|n{u.n}": float(u.n)
        for u in expand(spec, o.providers.texts())
    }
    a = tmp_path / "a.jsonl"
    run_part(spec, Shard(0, 2, ct), a, o)
    ra = kit.read_jsonl(a)[0]["integrity"]
    assert (ra["part"], ra["parts"], ra["cost_table_sha256"]) == (0, 2, spec_sha256(ct))
    assert np.isfinite(ra["budget_cpu_s"])

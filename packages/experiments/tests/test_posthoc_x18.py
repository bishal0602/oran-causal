import json
import math
from pathlib import Path

import pytest

from oran_causal.analysis.posthoc import x18
from oran_causal.experiments.posthoc import tools
from oran_causal.experiments.posthoc import x18 as run18
from oran_causal.methods.system import referees as rf

Z = 1.2815515655446004
SLEEP_MAP = [["sleep", "own", "pv", 1.0], ["ptx", "own", "pv", -1.0]]
G = {"v": 1000.0, "rlf": 30.0}
N = [["sleep", 1, 10.0], ["sleep", -1, 10.0], ["ptx", 1, 10.0], ["ptx", -1, 10.0]]


def _slope(ub: float) -> dict:
    return {"kind": "slope", "beta": 0.0, "se": ub / Z, "n": 100, "z": 0.0}


def bounds(rlf_own_ub: float = 0.09) -> dict:
    rows = []
    for f in ("ptx", "sleep"):
        for rel in ("own", "nbr", "far"):
            for k in ("v", "rlf"):
                ub = rlf_own_ub if (f, rel, k) == ("sleep", "own", "rlf") else 0.01
                rows.append([f, rel, k, _slope(ub)])
    return {"pmrt": rows, "ols": rows}


def artifact(b: dict) -> dict:
    theta = rf.THETA
    t = tools.referee_tools()
    tau = x18.tau_at({"calib": {"G": G, "N": N}}, t, x18.FROZEN_RHO)
    arms, tables = {}, {}
    for arm, meth in (("CS:PMRT", "pmrt"), ("CS:rand", None)):
        m = rf.map_from_json(SLEEP_MAP)
        cert = rf.certify_map(m, rf.bounds_from_json(b[meth]) if meth else None, tau, theta)
        unc = [tuple(x) for x in cert["uncertified"]]
        sig = rf.signature_key(rf.certsafe_signature(m, unc, theta))
        src = "MG:PMRT" if arm == "CS:PMRT" else "MG:rand"
        arms[arm] = {
            "source": src,
            "map": SLEEP_MAP,
            "bound_method": meth,
            "uncertified": [list(x) for x in unc],
            "signature": sig,
        }
        tables[arm] = {"signature": sig}
    al = rf.cs_alias_table(tables, theta, True)
    return {
        "constants": {"rho": x18.FROZEN_RHO},
        "calib": {"G": G, "N": N},
        "tau": rf.tau_to_json(tau),
        "bounds": {"gt": None},
        "arms": arms,
        "jobs": al["jobs"],
        "alias_of": al["alias_of"],
        "noarb_alias": True,
        "all_accept_signature": al["all_accept"],
        "mapgate": {"theta": theta},
    }


@pytest.fixture
def cert():
    b = bounds()
    return x18.run_cert(x18.CertInputs(artifact(b), b, b), tools.referee_tools()), artifact(b)


def test_frozen_rho_reproduces_and_reuses_every_stored_signature(cert):
    out, _art = cert
    assert out["repro"]["ok"]
    at = out["by_rho"][x18.key(x18.FROZEN_RHO)]["arms"]
    assert at["CS:PMRT"]["uncertified"] == [["sleep", 1]]
    assert at["CS:PMRT"]["reuse"] == "CS:PMRT"
    assert at["CS:rand"]["reuse"] == "noarb"
    assert not out["strict_at_frozen_rho"]["changed"]


def test_tau_scales_with_the_budget(cert):
    out, _ = cert
    tau = {
        x18.key(r): {tuple(row[:3]): row[3] for row in v["tau"]}
        for r, v in [(r, out["by_rho"][x18.key(r)]) for r in (1.05, 1.15)]
    }
    k = ("sleep", 1, "rlf")
    assert tau["1.050"][k] == pytest.approx(0.05)
    assert tau["1.150"][k] == pytest.approx(0.15)


def test_a_larger_budget_certifies_the_class_and_needs_a_new_job(cert):
    out, _art = cert
    lo = out["by_rho"]["1.075"]["arms"]["CS:PMRT"]
    hi = out["by_rho"]["1.100"]["arms"]["CS:PMRT"]
    assert lo["uncertified"] == [["sleep", 1]]
    assert lo["reuse"] == "CS:PMRT"
    assert hi["uncertified"] == []
    assert hi["reuse"] is None
    assert hi["changed_from_frozen"]
    sid = hi["sig_id"]
    assert out["signatures"][sid]["arm"] == x18.SIG_ARM + sid
    used = {tuple(u) for u in out["signatures"][sid]["used_by"]}
    assert {("CS:PMRT", "1.100"), ("CS:PMRT", "1.125"), ("CS:PMRT", "1.150")} <= used
    assert out["by_rho"]["1.150"]["arms"]["CS:rand"]["reuse"] == "noarb"


def test_strict_change_at_the_frozen_rho_stops_unless_allowed():
    frozen, strict = bounds(), bounds(rlf_own_ub=0.01)
    inp = x18.CertInputs(artifact(frozen), frozen, strict)
    with pytest.raises(x18.StrictChangeError):
        x18.run_cert(inp, tools.referee_tools())
    out = x18.run_cert(inp, tools.referee_tools(), allow_strict_change=True)
    assert out["strict_at_frozen_rho"]["changed"] == ["CS:PMRT"]


def test_wrong_frozen_bounds_stop_part_b():
    b = bounds()
    with pytest.raises(x18.ReproductionError):
        x18.run_cert(x18.CertInputs(artifact(b), bounds(0.01), b), tools.referee_tools())


def test_plan_checks_and_lists_the_jobs(cert, tmp_path: Path):
    out, art = cert
    plan = x18.make_plan(out, art)
    assert len(plan["jobs"]) == 1
    assert plan["cpu_h"] == pytest.approx(2.8 + 2.8 * 4 / 160, abs=0.01)
    run18.check_plan(plan)
    js = run18.jobs(plan)
    assert len(js) == 4 + 160
    assert [j.key for j in js[:4]] == [(s, x18.REPRO_ARM) for s in x18.REPRO_SEEDS]
    assert js[4].key == (191100, plan["jobs"][0]["arm"])
    assert len(run18.jobs(plan, seeds=[191100, 191200])) == 1 + 2
    arb = run18.arbiter_of(plan["jobs"][0])(plan["jobs"][0]["arm"], 1, 0.0)
    assert isinstance(arb, rf.Counting)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    assert run18.read_plan(p)["schema"] == "x18-plan/1"


def test_plan_refuses_a_doctored_signature(cert):
    out, art = cert
    plan = x18.make_plan(out, art)
    plan["jobs"][0]["uncertified"] = [["sleep", 1]]
    with pytest.raises(ValueError, match="signature"):
        run18.check_plan(plan)


def test_plan_over_the_cap_needs_approval(cert):
    out, art = cert
    one = next(iter(out["signatures"].values()))
    many = dict(out, signatures={f"{i:012d}": dict(one) for i in range(x18.CAP + 1)})
    with pytest.raises(x18.OverCapError):
        x18.make_plan(many, art)
    plan = x18.make_plan(many, art, allow_over_cap=True)
    assert len(plan["jobs"]) == x18.CAP + 1
    assert plan["over_cap_allowed"]


def test_full_run_refuses_off_linux(cert, tmp_path: Path, monkeypatch):
    out, art = cert
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(x18.make_plan(out, art)), encoding="utf-8")
    monkeypatch.setattr(run18.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux"):
        run18.run(p, tmp_path / "o.jsonl", run18.Part())
    with pytest.raises(ValueError, match="smoke"):
        run18.run(p, tmp_path / "o.jsonl", run18.Part(), short=10.0)
    assert math.isfinite(x18.rho_of(1.3))

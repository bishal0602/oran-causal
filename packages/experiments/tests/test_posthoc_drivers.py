"""Post hoc drivers: the X1 timing blocks, X4 and X7 referee episodes, and the X8 variants."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.analysis.posthoc import x4 as a_x4
from oran_causal.analysis.posthoc.x7 import FAMS, arm_name, job_list
from oran_causal.experiments.posthoc import x1_timing as t1
from oran_causal.experiments.posthoc import x4, x7, x8
from oran_causal.experiments.system import study3
from oran_causal.experiments.system.common import Part, read_records

DATA = Path(__file__).parent / "data"
RECORDS = Path(__file__).resolve().parents[3] / "data" / "records"
SKIP = {"cpu_s", "secs", "rss_mb"}


def digest(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()


def _dev_rows(spec: dict) -> list[dict]:
    rows = []
    for k, v in spec["est_cost_s"].items():
        arm, world, regime, n = k.split("|")
        rows.append({"arm": arm, "world": world, "regime": regime, "n": int(n[1:]), "median": v})
    return rows


@pytest.fixture(scope="module")
def timing_spec() -> dict:
    return t1.load_spec()


def test_make_spec_reproduces_the_frozen_timing_spec(timing_spec):
    eval_spec = json.loads(t1.EVAL_SPEC_PATH.read_text(encoding="utf-8"))
    assert t1.make_spec(eval_spec, _dev_rows(timing_spec)) == timing_spec


def test_make_spec_drops_the_dropped_arms_and_adds_the_extra_ones(timing_spec):
    assert not any(a.startswith(t1.DROPPED) for a in timing_spec["arms"])
    assert {"pmrt_nl_eq", "cdl", "mscr_eq_min"} <= set(timing_spec["arms"])
    assert timing_spec["arms"]["mscr_eq"]["max_n"] == 1000


def test_blocks_have_the_declared_unit_counts(timing_spec):
    assert len(t1.units(timing_spec, "calib")) == 51
    paper = t1.units(timing_spec, "paper")
    assert len(paper) == 342
    assert len({u.key for u in paper}) == 342
    assert all(u.arm != "mscr_eq_min" for u in paper)
    assert {u.n for u in paper if u.arm.startswith("mscr_")} <= {500, 1000}
    assert {(u.world, u.regime) for u in paper if u.arm.startswith("granger")} == {("E3", "R2")}


def test_keys_follow_the_campaign_convention(timing_spec):
    u = t1.units(timing_spec, "calib")[0]
    assert u.key == f"{u.arm}|E2|R2|k0.25|n{u.n}|s{u.seed}"


def test_assign_balances_by_estimated_cost(timing_spec):
    us = t1.units(timing_spec, "paper")
    bins = t1.assign(us, 4)
    assert sorted(u.key for b in bins for u in b) == sorted(u.key for u in us)
    load = [sum(u.est_cost_s for u in b) for b in bins]
    assert max(load) - min(load) <= max(u.est_cost_s for u in us)
    assert load == pytest.approx(
        [3980.8894034232208, 3980.8794143418168, 3980.9075309176333, 3980.881897089047]
    )


def test_session_command_sets_up_then_runs_the_block():
    cmd = t1.session_command(
        "calib", "$JOB_OUT", "kaggle", {"commit": "abc", "dirty": False}, 39600
    )
    setup, _, run = cmd.partition("nproc >> $JOB_OUT/install.log; ")
    assert "uv sync --frozen --no-dev" in setup
    assert "pin_check.json" in setup
    assert run == (
        "ORAN_CAUSAL_PLATFORM=kaggle ORAN_CAUSAL_CODE_COMMIT=abc ORAN_CAUSAL_CODE_DIRTY=0 "
        "ORAN_CAUSAL_LOCK_FILE=uv.lock OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "timeout 39600 python -u -m oran_causal.experiments.posthoc.x1_timing run --block calib "
        "--out-dir $JOB_OUT"
    )


class _Method:
    name = "fake"
    version = "0"

    def tune(self, dev, truth_free=True):
        return {}

    def run(self, data, config):
        raise NotImplementedError


def _providers(log: list) -> t1.Providers:
    def generate(world, regime, n, seed, lam, kappa):
        log.append(("gen", world, regime, n, seed))
        return object(), object()

    def run_one(method, job):
        log.append(("run", job.key))
        return {"key": job.key, "error": None, "cpu_s": 1.5, "wall_s": 1.6, "job": {"n": 500}}

    return t1.Providers(
        code=lambda: {"commit": "c", "dirty": False},
        host=lambda: {"platform": "test"},
        load=lambda: {"loadavg": [0.0, 0.0, 0.0]},
        pkgs=lambda: {"numpy": "x"},
        method=lambda ref: _Method(),
        generate=generate,
        run_one=run_one,
        clock=lambda: 0.0,
        log=lambda _m: None,
    )


def test_worker_stamps_every_unit_and_resumes(tmp_path, timing_spec):
    us = t1.units(timing_spec, "calib")[:4]
    out = tmp_path / "res_0.jsonl"
    log: list = []
    assert t1.worker(timing_spec, us, out, "calib", _providers(log)) == 4
    recs = read_records(out)
    assert [r["key"] for r in recs] == [u.key for u in us]
    r = recs[0]
    assert (r["status"], r["role"], r["block"], r["spec_name"]) == (
        "ok",
        "timing",
        "calib",
        "exp_c_calib",
    )
    assert r["campaign"] == t1.VERSION
    assert r["host"] == {"platform": "test"}
    assert set(r["load"]) == {"start", "end"}
    assert r["code"] == {"commit": "c", "dirty": False}
    log.clear()
    assert t1.worker(timing_spec, us, out, "calib", _providers(log)) == 0
    assert not log


class _Proc:
    def __init__(self, rc: int) -> None:
        self.rc = rc

    def wait(self) -> int:
        return self.rc


def test_run_starts_one_worker_per_vcpu_with_one_thread(tmp_path, timing_spec):
    spawned: list = []

    def spawn(argv, env, log):
        spawned.append((list(argv), dict(env), log))
        return _Proc(0)

    rc = t1.run(
        timing_spec,
        "calib",
        tmp_path,
        None,
        None,
        environ={"PATH": "p"},
        spawn=spawn,
        quota=lambda: 3,
        host=lambda: {"platform": "test"},
        python="py",
    )
    assert rc == 0
    assert len(spawned) == 3
    argv, env, log = spawned[0]
    assert argv == [
        "py", "-u", "-m", "oran_causal.experiments.posthoc.x1_timing", "worker", "--block", "calib",
        "--units", str(tmp_path / "units_0.json"), "--out", str(tmp_path / "res_0.jsonl"),
    ]  # fmt: skip
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert log == tmp_path / "log_0.txt"
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["procs"] == 3
    assert info["units"] == 51
    assert info["rcs"] == [0, 0, 0]
    units = [u for i in range(3) for u in json.loads((tmp_path / f"units_{i}.json").read_text())]
    assert len(units) == 51


def _obs(n: int) -> dict:
    return {"t": 0.0, "requests": [{"knob": ("ptx", 0), "prop": 1, "cur": 0, "xapp": "a"}] * n}


def _decide(p: float, seed: int, steps: int = 200, per: int = 5):
    a = x4.RandomDefer(p, seed)
    return [d for _ in range(steps) for d in a(_obs(per))["decisions"]], a


def test_x4_arms_and_rates():
    assert x4.ARMS == ("X4:rand@0.352", "X4:rand@0.25", "X4:rand@0.5")
    assert abs(a_x4.P_PMRT - 37890 / 107525) < 5e-4


def test_x4_decisions_are_deterministic_and_nested():
    d1, a = _decide(0.352, 191100)
    assert d1 == _decide(0.352, 191100)[0]
    assert a.n["req"] == 1000
    assert a.n["rej"] == d1.count("reject")
    assert abs(a.n["rej"] / 1000 - 0.352) < 0.05
    assert _decide(0.352, 191101)[0] != d1
    lo, mid, hi = (_decide(p, 191100)[0] for p in (0.25, 0.352, 0.5))
    for a_, b_ in ((lo, mid), (mid, hi)):
        assert all(y == "reject" for x, y in zip(a_, b_, strict=True) if x == "reject")


def test_x4_edges_and_stream():
    assert set(_decide(0.0, 1)[0]) == {"accept"}
    assert set(_decide(1.0, 1)[0]) == {"reject"}
    assert np.random.default_rng([1, x4.TAG]).random() == x4.RandomDefer(0.5, 1).rng.random()
    with pytest.raises(ValueError, match=r"1\.5"):
        x4.RandomDefer(1.5, 1)


def test_job_lists_equal_the_old_drivers():
    gold = json.loads((DATA / "posthoc_jobs_golden.json").read_text())
    pairs = [list(j) for j in x4.job_pairs()]
    assert len(pairs) == gold["x4_jobs"]["n"] == 4 + 160 * 3
    assert pairs[:6] == gold["x4_jobs"]["head"]
    assert digest(pairs) == gold["x4_jobs"]["sha256"]
    seven = [list(j) for j in job_list(study3.stage_seeds("eval"))]
    assert len(seven) == gold["x7_jobs"]["n"] == 160 * 13
    assert seven[:6] == gold["x7_jobs"]["head"]
    assert digest(seven) == gold["x7_jobs"]["sha256"]
    assert {s for s, _ in pairs} == set(range(191100, 191260))


def test_job_filters_restrict_the_runs():
    assert {a for _, a in x4.job_pairs([191100], ["noarb", "X4:rand@0.25"])} == {
        "noarb",
        "X4:rand@0.25",
    }
    assert [j[1] for j in job_list([191100], ["shadow"])] == ["shadow"]
    with pytest.raises(ValueError, match="outside the block"):
        x4.job_pairs([1])


REQ = [
    {"knob": ("prot_min", 0), "prop": 0.3, "cur": 0.2, "xapp": "SliceGuarantee"},
    {"knob": ("ptx", 1), "prop": -3.0, "cur": 0.0, "xapp": "PowerES"},
    {"knob": ("sleep", 5), "prop": 1.0, "cur": 0.0, "xapp": "ES"},
]


def _seven_obs(t: float = 200.0) -> dict:
    return {"t": t, "requests": [dict(r) for r in REQ]}


def _all(dec: str):
    return lambda obs: {"decisions": [dec] * len(obs["requests"]), "writes": []}


def test_x7_restrict_overrides_only_the_output():
    pm = x7.Restrict(_all("reject"), frozenset({"prot_min"}))
    assert pm(_seven_obs())["decisions"] == ["reject", "accept", "accept"]
    assert pm.n == {"overridden": 2, "kept": 1}
    other = x7.Restrict(_all("reject"), frozenset(FAMS) - {"prot_min"})
    assert other(_seven_obs())["decisions"] == ["accept", "reject", "reject"]


def test_x7_shadow_tallies_pairs_without_changing_decisions():
    sh = x7.Shadowed(_all("accept"), {"s": _all("reject")})
    assert sh(_seven_obs())["decisions"] == ["accept"] * 3
    sh(_seven_obs(10.0))
    assert sh.agree["s"]["prot_min|+1|post"] == [0, 1, 0, 0]
    assert sh.agree["s"]["ptx|-1|pre"] == [0, 1, 0, 0]


def test_x7_shadow_queries_a_copy_of_the_observation():
    def mutating(obs):
        n = len(obs["requests"])
        obs["requests"].clear()
        return {"decisions": ["reject"] * n}

    sh = x7.Shadowed(_all("accept"), {"m": mutating})
    obs = _seven_obs()
    assert sh(obs)["decisions"] == ["accept"] * 3
    assert len(obs["requests"]) == 3


def test_x7_arm_names():
    assert arm_name(None, "shadow") == "X7:sub|shadow"
    assert arm_name("CS:granger@dev", "pm") == "X7:CS:granger@dev|pm"


def _fake_job(monkeypatch, module, name: str = "job") -> list:
    calls: list = []

    def fake(seed, *args, **kw):
        calls.append(seed)
        arm = args[0] if module is x4 else arm_name(args[1], args[0])
        return {
            "kind": "job",
            "key": [seed, arm],
            "seed": seed,
            "arm": arm,
            "smoke": kw.get("smoke", False),
            "psvr": 1.0,
            "rlf": 0,
            "st_req": 0,
            "st_rej": 0,
            "cpu_s": 0.0,
        }

    monkeypatch.setattr(module, name, fake)
    return calls


def test_x4_shards_partition_the_jobs_and_resume(tmp_path, monkeypatch):
    calls = _fake_job(monkeypatch, x4)
    out = tmp_path / "res_1.jsonl"
    n = x4.run(Part(1, 7), out, seeds=[191100, 191101], arms=["noarb", "X4:rand@0.25"])
    jobs = x4.job_pairs([191100, 191101], ["noarb", "X4:rand@0.25"])
    assert n == len([j for u, j in enumerate(jobs) if u % 7 == 1])
    assert len(calls) == n
    recs = read_records(out)
    assert recs[0]["kind"] == "header"
    assert recs[0]["driver"] == "x4_random_defer"
    assert recs[0]["sub"] == "x4"
    assert recs[0]["cs_stage"] == "eval"
    assert recs[0]["consts"]["tag"] == 6640
    assert recs[0]["consts"]["p"]["X4:rand@0.352"] == 0.352
    assert recs[-1]["kind"] == "close"
    assert recs[-1]["n_jobs"] == n
    assert x4.run(Part(1, 7), out, seeds=[191100, 191101], arms=["noarb", "X4:rand@0.25"]) == 0


def test_x4_refuses_short_without_smoke(tmp_path):
    with pytest.raises(ValueError, match="smoke-only"):
        x4.run(Part(), tmp_path / "r.jsonl", short=5.0)


def test_x7_shards_run_their_jobs_and_header(tmp_path, monkeypatch):
    calls = _fake_job(monkeypatch, x7)
    monkeypatch.setattr(x7.study3, "load_artifact", lambda path: {"tables": {}})
    out = tmp_path / "res_2.jsonl"
    n = x7.run(Part(2, 5), out, seeds=[191100], kinds=("shadow", "own"))
    assert n == len([u for u in range(1 + 4) if u % 5 == 2]) == 1
    assert len(calls) == 1
    head = read_records(out)[0]
    assert head["driver"] == "x7_subset_check"
    assert head["consts"]["kinds"] == ["shadow", "own"]
    assert head["consts"]["ref"] == "sub:ES+PowerES"


def test_x8_helpers():
    assert x8.rng_list("1-3,7") == [1, 2, 3, 7]
    assert x8.variants([0, 1], [1, 2]) == [(0, 1), (0, 2), (1, 1), (1, 2)]
    todo = x8.variants(range(10), range(1, 60))
    assert sum(len(x8.shard(todo, i, 6)) for i in range(6)) == 590
    assert x8.shard(todo, 5, 6)[:2] == [todo[5], todo[11]]
    g = x8.groups_of(list(range(188100, 188700)))
    assert len(g) == 10
    assert g[0][0] == 0
    assert g[9][-1] == 599
    assert all(len(v) == 60 for v in g.values())


def test_x8_resume_keys(tmp_path):
    p = tmp_path / "res.jsonl"
    p.write_text(
        '{"group": 1, "shift": 2}\n{"group": 3\n{"group": 0, "shift": 5}\n', encoding="utf-8"
    )
    assert x8.done_keys(p) == {(1, 2), (0, 5)}
    assert x8.done_keys(tmp_path / "none.jsonl") == set()


def test_x8_declaration_block_keeps_determined_hypotheses():
    declared = {
        ("plain_c", "by"): {
            ("carrier", "own", "pv"): {"p": 0.01, "declared": True, "sign": -1},
            ("sleep", "nbr", "v"): {"p": float("nan"), "declared": False, "status": "tested"},
            ("ptx", "far", "e"): {"status": "undetermined"},
        }
    }
    assert x8.decl_block(declared) == {
        "plain_c+by": {"carrier|own|pv": [0.01, True, -1], "sleep|nbr|v": [None, False, 0]}
    }


def near(got, want, rtol=0.2, path=""):
    """Top-level fields equal; shared numbers within `rtol` (episodes diverge across hardware)."""
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        if not path:
            assert sorted(got) == sorted(want), path
        for k in want.keys() & got.keys():
            near(got[k], want[k], rtol, f"{path}/{k}")
    elif isinstance(want, list):
        assert isinstance(got, list), path
        if all(isinstance(w, int | float | str) for w in want):
            return
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            near(g, w, rtol, f"{path}[{i}]")
    elif isinstance(want, int | float) and not isinstance(want, bool):
        assert got == pytest.approx(want, rel=rtol, abs=3), (path, got, want)
    else:
        assert got == want, path


GOLDEN = DATA / f"posthoc_records_golden_{sys.platform}.json"


@pytest.mark.slow
@pytest.mark.skipif(not GOLDEN.exists(), reason="no golden for this platform")
def test_smoke_episodes_match_the_old_drivers_on_this_platform():
    gold = json.loads(GOLDEN.read_text())

    def norm(r):
        return json.loads(json.dumps({k: v for k, v in r.items() if k not in SKIP}, default=str))

    for arm, g in gold["x4"].items():
        near(norm(x4.job(gold["seed"], arm, smoke=True, short=gold["short_s"])), g)
    art = study3.load_artifact()
    for kind, ref in (
        ("shadow", None),
        ("own", "CS:granger@dev"),
        ("pm", "CS:granger@dev"),
        ("other", "CS:granger@dev"),
    ):
        got = norm(x7.job(gold["seed"], kind, ref, art, smoke=True, short=gold["short_s"]))
        near(got, gold["x7"][arm_name(ref, kind)])

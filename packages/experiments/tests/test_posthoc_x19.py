"""X19 cost driver and summary on synthetic stored jobs."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from oran_causal.analysis.posthoc import x19 as report
from oran_causal.core.pmrt.units import PmrtConfig, fit_pmrt
from oran_causal.experiments.posthoc import x19
from oran_causal.experiments.system.common import read_records
from oran_causal.methods.system import pmrt as pm
from oran_causal.sim.system.decision.synthetic import SyntheticSpec, synthetic_records

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::UserWarning")

FAST = PmrtConfig(B=199, cv_reps=1, gb=False)
EVAL0 = 188100


def _recs(seed: int, n_ep: int, seed0: int, stage: str, sub: str = "", placebo: bool = False):
    spec = SyntheticSpec(
        n_ep=n_ep, units_per_ep=60, effect=40.0, family="sleep", seed0=seed0, placebo=placebo
    )
    out = synthetic_records(np.random.default_rng(seed), spec)
    for r in out:
        r["stage"], r["sub"], r["key"] = stage, sub, [stage, sub, r["seed"]]
    return out


def _cache(recs: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.cache_records(recs, path)


def _placeholder(job: Path, name: str, doc: dict) -> None:
    (job / "out").mkdir(parents=True, exist_ok=True)
    (job / "out" / name).write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture(scope="module")
def jobs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    d = tmp_path_factory.mktemp("x19")
    s1, s2, tr = d / "s1", d / "s2", d / "train"
    ev = _recs(1, 6, EVAL0, "eval")
    _cache(ev, s1 / "out" / "cache" / "eval" / "eval_000.npz")
    _cache(_recs(2, 3, 188700, "placebo", placebo=True), s1 / "out/cache/placebo/p_000.npz")
    _cache(_recs(3, 6, 186100, "eval"), s2 / "out" / "cache" / "disc" / "disc_000.npz")
    _cache(_recs(4, 3, 186700, "placebo", placebo=True), s2 / "out/cache/placebo/p_000.npz")
    _cache(_recs(5, 3, 5000, "dev", ""), tr / "out" / "cache" / "dev.npz")
    _cache(_recs(6, 6, 6000, "eval", "v2"), tr / "out" / "cache" / "v2.npz")
    _cache(_recs(7, 3, 7000, "eval", "v3"), tr / "out" / "cache" / "v3.npz")
    _placeholder(s1, "analysis_v4.json", {"slices_result": {}})
    _placeholder(s2, "disc_conf.json", {"disc_hyp": {}})
    raw = d / "raw" / "res_0.jsonl"
    raw.parent.mkdir()
    raw.write_text("".join(json.dumps(r) + "\n" for r in ev[:2]), encoding="utf-8")
    pool = pm.load_pmrt_pool(x19.cache_files(tr, None))
    train = [e for e in range(pool.pool.n_eps) if str(pool.pool.eps["sub"][e]) in ("", "v2")]
    v2 = [e for e in range(pool.pool.n_eps) if str(pool.pool.eps["sub"][e]) == "v2"]
    params = fit_pmrt(pm.pmrt_data(pool, train), pm.LEVEL_V2_ARR, FAST)
    run = pm.run_integrated(pm.pmrt_data(pool, v2), params, FAST, 0)
    priors = {s: pm.prior_from_run(run, s) for s in pm.STATS}
    return {
        "inputs": x19.Inputs(s1, s2, tr, (raw.parent,)),
        "frozen": x19.Frozen(params, priors, FAST, {"label": "synthetic"}),
    }


def test_units_cover_the_declared_cells():
    us = x19.units()
    cells = {u.cell for u in us}
    assert cells == {
        "disc|s1|pooled",
        "disc|s1|s60_0",
        "disc|s1|s120_0",
        "disc|s1|s300_0",
        "disc|s2|pooled",
        "fit|train|all",
        "referee|s2|pooled",
        "logging|all|all",
    }
    assert len(us) == 7 * x19.REPS + 1
    assert len({u.key for u in us}) == len(us)
    assert {u.cell for u in x19.units(kinds=("disc",))} <= {c for c in cells if "disc" in c}


def test_assign_spreads_the_costliest_units_first():
    bins = x19.assign(x19.units(), 4)
    loads = [sum(u.est_cost_s for u in b) for b in bins]
    assert sum(len(b) for b in bins) == len(x19.units())
    assert max(loads) - min(loads) <= max(x19.EST_COST_S.values())


def test_compare_separates_exact_close_and_different():
    a = {"h": {"z": 1.0, "s": 1}, "l": [1, 2]}
    assert x19.compare(a, copy.deepcopy(a))["exact"]
    b = {"h": {"z": 1.0 + 1e-12, "s": 1}, "l": [1, 2]}
    c = x19.compare(a, b)
    assert (c["exact"], c["close"], c["n_diff"]) == (False, True, 1)
    c = x19.compare(a, {"h": {"z": 1.0, "s": -1}, "l": [1, 2]})
    assert (c["exact"], c["close"]) == (False, False)
    assert not x19.compare(a, {"h": {"z": 1.0}, "l": [1, 2]})["close"]


@pytest.mark.parametrize(("study", "name"), [("s1", "pooled"), ("s1", "s60_0"), ("s2", "pooled")])
def test_disc_times_the_phases_and_checks_the_stored_table(jobs, tmp_path, study, name):
    inputs, frozen = jobs["inputs"], jobs["frozen"]
    first = x19.measure_disc(inputs, study, name, frozen)
    assert not first["repro"]["exact"]
    assert set(first["phase_cpu_s"]) == {"load", "features", "test", "declare"}
    assert first["cpu_s"] == pytest.approx(
        sum(first["phase_cpu_s"][p] for p in ("features", "test", "declare"))
    )
    assert first["n_tested"] > 0
    assert first["B"] == FAST.B
    job = tmp_path / study
    src = inputs.s1 if study == "s1" else inputs.s2
    shutil.copytree(src / "out" / "cache", job / "out" / "cache")
    hyp = first["hyp"]
    if study == "s1":
        _placeholder(job, "analysis_v4.json", {"slices_result": {name: {"hyp": hyp}}})
        ins = x19.Inputs(job, inputs.s2, inputs.train)
    else:
        _placeholder(job, "disc_conf.json", {"disc_hyp": hyp})
        ins = x19.Inputs(inputs.s1, job, inputs.train)
    again = x19.measure_disc(ins, study, name, frozen)
    assert again["repro"]["exact"]
    assert again["repro_declared"]
    bad = copy.deepcopy(hyp)
    h = next(k for k, r in bad.items() if r["status"] == "tested")
    bad[h]["plain_c"]["z"] += 1.0
    doc = {"slices_result": {name: {"hyp": bad}}} if study == "s1" else {"disc_hyp": bad}
    _placeholder(job, "analysis_v4.json" if study == "s1" else "disc_conf.json", doc)
    worse = x19.measure_disc(ins, study, name, frozen)
    assert not worse["repro"]["exact"]
    assert worse["repro"]["n_diff"] >= 1


def test_fit_uses_the_training_subs_and_compares_with_the_frozen_parameters(jobs):
    rec = x19.measure_fit(jobs["inputs"], jobs["frozen"])
    assert rec["n_train_episodes"] == 9
    assert rec["n_prior_episodes"] == 6
    fc = rec["fit_check"]
    assert fc["exact"] == fc["hypotheses"] > 0
    assert not fc["missing"]
    assert rec["prior_check"]["exact"]
    assert {"fit", "prior_test"} <= set(rec["phase_cpu_s"])


def test_referee_latency_covers_every_context_with_the_stored_artifacts(jobs):
    maps_doc = x19._artifact("study2_maps.json")
    cs_doc = x19._artifact("study3_certsafe.json")
    rec = x19.measure_referee(jobs["inputs"], maps_doc, cs_doc)
    n = rec["n_contexts"]
    assert n == 6 * 60
    for arm in ("MG:PMRT", "CS:PMRT"):
        assert rec[arm]["n"] == n
        assert 0 < rec[arm]["median_us"] <= rec[arm]["p99_us"] <= rec[arm]["max_us"]
    assert rec["certify"]["repro"]
    assert rec["certify"]["calls"] == x19.CERTIFY_CALLS


def test_unit_contexts_carry_the_family_and_step(jobs):
    pool = x19.records.load_pool(
        x19.cache_files(jobs["inputs"].s2, "disc"), H=90, H_pre=90, stages={"eval"}
    )
    ctxs = x19.unit_contexts(pool)
    assert {c["ctx"]["knob"] for c in ctxs} <= set(x19.FAMILIES)
    assert all(c["ctx"]["step"] in (-1.0, 1.0) for c in ctxs)
    assert all("own_prb_util" in c["ctx"] for c in ctxs)


def test_logging_volume_rates_sizes_and_series(jobs):
    rec = x19.measure_logging(jobs["inputs"])
    v = rec["volume"]
    ra = v["s1_eval"]["rates"]
    assert ra["episodes"] == 6
    assert ra["units"] == 360
    assert ra["hours"] == pytest.approx(6 * 720 / 3600)
    assert ra["units_per_ep_hour"] == pytest.approx(360 / ra["hours"])
    assert sum(ra["by_class"].values()) == 360
    assert v["s1_placebo"]["rates"]["by_policy"] == {"placebo": 3 * 60}
    size = v["s1_eval"]["unit_record"]
    assert size["fixed_bytes"] == 14 + 4 * size["n_ctx"]
    assert size["compressed_bytes_per_unit"] > 0
    eps = v["series"]["episodes"]
    assert len(eps) == 2
    assert eps[0]["shape"][1:] == [len(x19.records.SERIES_FIELDS), 8]
    pr = v["s1_eval"]["projection"]
    assert pr["units_per_cell_day"] == pytest.approx(ra["units_per_cell_hour"] * 24)


def test_logging_without_raw_records_skips_the_series(jobs):
    ins = x19.Inputs(jobs["inputs"].s1, jobs["inputs"].s2, jobs["inputs"].train)
    assert "skipped" in x19.measure_logging(ins)["volume"]["series"]


def _providers(seen: list) -> x19.Providers:
    return x19.Providers(
        code=lambda: {"commit": "c", "dirty": False},
        host=lambda: {"platform": "test"},
        load=lambda: {"loadavg": [0.0, 0.0, 0.0]},
        pkgs=lambda: {"numpy": "x"},
        log=seen.append,
    )


def test_worker_stamps_every_unit_and_resumes(tmp_path, monkeypatch, jobs):
    calls: list = []

    def fake(u, inputs, p):
        calls.append(u.key)
        if u.kind == "fit":
            raise RuntimeError("boom")
        return {"cpu_s": 1.0, "wall_s": 1.5, "peak_rss_mb": 10.0}

    monkeypatch.setattr(x19, "measure", fake)
    us = x19.units(reps=1)
    out = tmp_path / "res_0.jsonl"
    assert x19.worker(us, jobs["inputs"], out, _providers([])) == len(us)
    recs = read_records(out)
    assert [r["key"] for r in recs] == [u.key for u in us]
    by = {r["unit"]["kind"]: r for r in recs}
    assert by["fit"]["status"] == "error"
    assert "boom" in by["fit"]["error"]
    assert by["disc"]["status"] == "ok"
    assert by["disc"]["version"] == x19.VERSION
    assert set(by["disc"]["load"]) == {"start", "end"}
    calls.clear()
    assert x19.worker(us, jobs["inputs"], out, _providers([])) == 1
    assert calls == ["fit|train|all|r0"]


class _Proc:
    def wait(self) -> int:
        return 0


def test_run_starts_one_single_threaded_worker_per_vcpu(tmp_path, jobs):
    spawned: list = []

    def spawn(argv, env, log):
        spawned.append((list(argv), dict(env)))
        return _Proc()

    us = x19.units()
    rc = x19.run(
        us,
        jobs["inputs"],
        tmp_path,
        None,
        environ={"PATH": "p"},
        spawn=spawn,
        quota=lambda: 3,
        host=lambda: {"platform": "test"},
        python="py",
    )
    assert rc == 0
    assert len(spawned) == 3
    argv, env = spawned[0]
    assert argv[:5] == ["py", "-u", "-m", x19.WORKER_MODULE, "worker"]
    assert argv[argv.index("--s1") + 1] == str(jobs["inputs"].s1)
    assert argv.count("--raw") == 1
    assert all(env[k] == "1" for k in x19.THREAD_ENV)
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["units"] == len(us)
    assert info["rcs"] == [0, 0, 0]
    got = [u for i in range(3) for u in json.loads((tmp_path / f"units_{i}.json").read_text())]
    assert sorted(x19.Unit(**u).key for u in got) == sorted(u.key for u in us)


def test_session_command_and_kaggle_job_read_the_mounted_sources():
    code = {"commit": "abc", "dirty": False}
    cmd = x19.session_command("$JOB_OUT", "kaggle", code, "$JOB_SRC")
    setup, _, run = cmd.partition("nproc >> $JOB_OUT/install.log; ")
    assert "uv sync --frozen --no-dev" in setup
    assert run.startswith(
        "ORAN_CAUSAL_PLATFORM=kaggle ORAN_CAUSAL_CODE_COMMIT=abc ORAN_CAUSAL_CODE_DIRTY=0 "
        "ORAN_CAUSAL_LOCK_FILE=uv.lock OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"timeout {x19.WALL_S} python -u -m oran_causal.experiments.posthoc.x19 run "
        "--out-dir $JOB_OUT --s1 $JOB_SRC/e6p-disc-v4an-pmrt-1 --s2 $JOB_SRC/e6p-conf-discan-2 "
        "--train $JOB_SRC/disc-bench-1"
    )
    assert run.count("--raw $JOB_SRC/") == len(x19.RAW_SOURCES)
    job = x19.kaggle_job("x19-cost-1", "me", code)
    assert job.sources == [
        "me/e6p-disc-v4an-pmrt-1",
        "me/e6p-conf-discan-2",
        "me/disc-bench-1",
        "me/e6p-disc-v4ev-1-a",
        "me/e6p-conf-disc-1-a",
    ]
    assert not job.gpu
    assert job.cmd == cmd


def test_launch_dry_run_prints_without_launching(capsys):
    assert x19.main(["launch", "--name", "x19-t", "--owner", "me", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "me/disc-bench-1" in out
    assert "x19 run" in out


def _rec(kind: str, study: str, sl: str, rep: int, **kw) -> dict:
    u = {"kind": kind, "study": study, "slice": sl, "rep": rep}
    base = {
        "key": f"{kind}|{study}|{sl}|r{rep}",
        "unit": u,
        "status": "ok",
        "host": {"platform": "kaggle", "cpu_model": "Intel(R) Xeon(R) CPU @ 2.20GHz"},
        "cpu_s": 10.0 + rep,
        "wall_s": 11.0 + rep,
        "phase_cpu_s": {"load": 2.0, "test": 8.0 + rep},
        "peak_rss_mb": 500.0,
        "rss_scope": "job",
    }
    return {**base, **kw}


def _summary_records(exact: bool) -> list[dict]:
    rep = {"exact": exact, "close": True, "n_diff": 0 if exact else 1}
    recs = [
        _rec("disc", "s1", "pooled", r, repro=rep, repro_declared=True, n_units=9) for r in range(3)
    ]
    recs += [
        _rec(
            "referee",
            "s2",
            "pooled",
            r,
            n_contexts=5,
            certify={"median_us": 30.0, "repro": True},
            **{k: {"n": 5, "median_us": 2.0, "p99_us": 9.0} for k in ("MG:PMRT", "CS:PMRT")},
        )
        for r in range(3)
    ]
    recs.append(_rec("fit", "train", "all", 0, fit_check={"exact": 1}, prior_check={"exact": 1}))
    vol = {
        "s1_eval": {
            "rates": {
                "episodes": 2,
                "hours": 0.4,
                "units_per_ep_hour": 300.0,
                "units_per_cell_hour": 12.5,
            },
            "unit_record": {"fixed_bytes": 30, "compressed_bytes_per_unit": 20.0},
            "projection": {"units_per_cell_day": 300.0},
        },
        "series": {"skipped": "no raw records given"},
    }
    recs.append(_rec("logging", "all", "all", 0, volume=vol))
    recs.append({**_rec("disc", "s2", "pooled", 0), "status": "error", "error": "x"})
    return recs


def test_summary_statistics_and_reproduction_gate():
    s = report.summarise(_summary_records(True))
    d = s["disc"]["s1|pooled"]
    assert d["cpu_s"] == {"n": 3, "median": 11.0, "min": 10.0, "max": 12.0}
    assert d["phase_cpu_s"]["test"]["median"] == 9.0
    assert d["episodes_planned"] == 600
    assert d["n_units"] == 9
    assert s["on_reference_host"]
    assert s["errors"] == ["disc|s2|pooled|r0"]
    assert not s["reproduced"]
    ok = [r for r in _summary_records(True) if r["unit"]["study"] != "s2" or r["status"] == "ok"]
    ok = [r for r in ok if r.get("status") == "ok"]
    assert report.summarise(ok)["reproduced"]
    assert report.summarise(ok)["referee"]["MG:PMRT"]["median_us"]["median"] == 2.0
    text = report.markdown(report.summarise(ok))
    assert "| s1 | pooled |" in text
    assert "MapGateV2" in text
    assert "projection" in text


def test_report_writes_both_files_and_fails_without_exact_reproduction(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    good = [r for r in _summary_records(True) if r["status"] == "ok"]
    (runs / "res_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in good))
    out = tmp_path / "out"
    assert report.main(["--runs", str(runs), "--out-dir", str(out)]) == 0
    assert json.loads((out / "x19_cost.json").read_text())["reproduced"]
    assert (out / "x19_cost.md").read_text().startswith("# X19")
    bad = [r for r in _summary_records(False) if r["status"] == "ok"]
    (runs / "res_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad))
    assert report.main(["--runs", str(runs), "--out-dir", str(out)]) == 1

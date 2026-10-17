import json

import numpy as np
import pytest

from oran_causal.experiments.harness import runner
from oran_causal.experiments.harness.records import Job, clean, done_keys, run_one
from oran_causal.methods.controlled.dummy import DummyMethod
from oran_causal.sim.controlled.datasets.generate import dataset_hash, generate_dataset

SEED = 3_000_000
BASE = {
    "methods": ["dummy"],
    "worlds": ["E1", "E5"],
    "regimes": ["R1"],
    "ns": [100],
    "seeds": [SEED, SEED + 1],
}


def quiet(_msg: str) -> None:
    pass


def test_runner_end_to_end_resume_and_merge(tmp_path):
    spec = {
        "methods": ["dummy"],
        "worlds": ["E1", "E4"],
        "regimes": ["R1", "R3"],
        "ns": [200],
        "seeds": [SEED, SEED + 1],
        "lams": [0.0, 1.5],
        "kappas": [0.1],
    }
    n_jobs = 2 * 1 + 2 * 2 * 2
    assert len(runner.all_keys(spec)) == n_jobs
    outs = [tmp_path / f"res_{i}.jsonl" for i in range(2)]
    assert sum(runner.run_part(spec, (i, 2), outs[i], log=quiet) for i in range(2)) == n_jobs
    assert runner.run_part(spec, (0, 2), outs[0], log=quiet) == 0
    summ = runner.merge(spec, outs, tmp_path / "merged.jsonl")
    assert summ["n_records"] == n_jobs
    assert not summ["missing"]
    assert not summ["errors"]
    r = json.loads((tmp_path / "merged.jsonl").read_text().splitlines()[0])
    for k in ("dataset_sha256", "cpu_s", "gen_cpu_s", "peak_rss_mb", "scores", "edges"):
        assert k in r
    j = r["job"]
    lam = j["lam"] if j["lam"] is not None else 1.0
    ds, _ = generate_dataset(j["world"], j["regime"], 200, j["seed"], lam=lam, kappa=j["kappa"])
    assert dataset_hash(ds) == r["dataset_sha256"]


def test_runner_rejects_non_dev_seeds():
    spec = {"methods": ["dummy"], "worlds": ["E1"], "regimes": ["R1"], "ns": [100], "kappas": [0]}
    with pytest.raises(ValueError, match="DEV block"):
        runner.dataset_groups({**spec, "seeds": [3_100_000, 3_100_001]})


def test_runner_records_errors_without_stopping():
    class Boom:
        name, version = "boom", "0"

        def tune(self, dev, truth_free=True):
            return {}

        def run(self, data, config):
            raise RuntimeError("boom")

    ds, tr = generate_dataset("E1", "R1", 50, SEED)
    rec = run_one(Boom(), Job("boom|x", ds, tr, {}, 0.0))
    assert "RuntimeError" in rec["error"]


def test_runner_requires_kappas():
    with pytest.raises(ValueError, match="kappas"):
        runner.dataset_groups(BASE)
    for bad in ([], [-1], [None], "0.3", [float("nan")]):
        with pytest.raises(ValueError, match="kappas"):
            runner.dataset_groups({**BASE, "kappas": bad})


def test_runner_kappa_in_keys_configs_and_records(tmp_path):
    configs = {"default": {"q": 0.05}, "E1|R1|n100": {"q": 0.1}, "E1|R1|k0.3|n100": {"q": 0.2}}
    spec = {**BASE, "kappas": [0, 0.3], "configs": {"dummy": configs}}
    keys = runner.all_keys(spec)
    assert len(keys) == 2 * 2 * 2
    assert "dummy|E1|R1|k0|n100|s3000000" in keys
    assert "dummy|E5|R1|k0.3|n100|s3000001" in keys
    assert runner.method_config(spec, "dummy", ("E1", "R1", None, 100, SEED, 0.0)) == {"q": 0.1}
    assert runner.method_config(spec, "dummy", ("E1", "R1", None, 100, SEED, 0.3)) == {"q": 0.2}
    assert runner.cell_key("E1", "R1", None, 100) == "E1|R1|n100"
    out = tmp_path / "r.jsonl"
    assert runner.run_part(spec, (0, 1), out, log=quiet) == 8
    summ = runner.merge(spec, [out], tmp_path / "m.jsonl")
    assert not summ["missing"]
    assert not summ["errors"]
    assert set(summ["per_cell"]) == {
        f"dummy|{w}|R1|k{k}|n100" for w in ("E1", "E5") for k in ("0", "0.3")
    }
    cell = summ["per_cell"]["dummy|E1|R1|k0|n100"]
    assert cell["n_not_testable_true_total"] == 0
    assert cell["n_not_testable_null_total"] == 0
    recs = {json.loads(x)["key"]: json.loads(x) for x in out.read_text().splitlines()}
    r = recs["dummy|E5|R1|k0|n100|s3000000"]
    assert r["job"]["kappa"] == 0.0
    assert r["config"] == {"q": 0.05}
    ds, _ = generate_dataset("E5", "R1", 100, SEED, kappa=0.0)
    assert r["dataset_sha256"] == dataset_hash(ds)
    assert recs["dummy|E1|R1|k0.3|n100|s3000001"]["job"]["kappa"] == 0.3


def test_legacy_ref_and_registry_name_resolve_to_the_dummy_arm():
    assert isinstance(runner.load_method("cdd_oran.xmethod.runner:DummyMethod"), DummyMethod)
    assert isinstance(runner.load_method("dummy"), DummyMethod)
    instance = DummyMethod()
    assert runner.load_method("x", resolve=lambda ref: instance) is instance


def test_clean_makes_json_safe():
    raw = {"a": np.float64(1.5), "b": [np.int64(2), float("nan")], 3: np.array([1.0, float("inf")])}
    assert clean(raw) == {"a": 1.5, "b": [2, None], "3": [1.0, None]}


def test_done_keys_skip_errors_and_broken_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    rows = [{"key": "a", "error": None}, {"key": "b", "error": "x"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"key": "c"', encoding="utf-8")
    assert done_keys(path) == {"a"}
    assert done_keys(tmp_path / "missing.jsonl") == set()

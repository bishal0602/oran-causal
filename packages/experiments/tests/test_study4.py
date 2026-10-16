import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from oran_causal.experiments.campaign.integrity import FROZEN_EVAL_PYTHON
from oran_causal.experiments.campaign.merge import iter_jsonl, merge
from oran_causal.experiments.campaign.shard import assignment
from oran_causal.experiments.campaign.spec import expand, sha_lf, spec_sha256
from oran_causal.experiments.controlled.study4 import (
    integrity_problems,
    load_cost_table,
    load_spec,
    plan,
    run_study,
)
from oran_causal.experiments.harness.isolation import can_isolate
from oran_causal.experiments.paths import EVAL_COST_TABLE, PROTOCOLS, SPECS, WORKSPACE_ROOT

EVAL_SPEC = SPECS / "eval" / "full.json"
RECORDS = WORKSPACE_ROOT / "data" / "records"
EVAL_MERGED = RECORDS / "study4_eval" / "eval" / "merged.jsonl.gz"
ASSIGNMENT_SHA = "ad2fa4b3761943bbddfe92fa7736ca32c5108b8fcf3efca47852ff2ca55b447b"
COST_TABLE_SHA = "cf05e8809ef1741a3840881ea9b466b78e1b280289c4be8b8a5f82290356307a"
COST_TABLE_CANONICAL_SHA = "9a68459241f5b5711ebd6768baf999e362d9e252b4bcee1bf0384b2373caf14a"
DEV_SETS = {
    "cdl": ("dev/cdl.json", "dev/cdl/merged.jsonl"),
    "ci_c": ("dev/ci_c.json", "dev/ci_c/merged.jsonl"),
    "full": ("dev/full.json", "dev/full/merged.jsonl"),
    "pilot": ("dev/pilot.json", "dev/pilot/merged.jsonl"),
    "pilot_ci_cpu": ("dev/pilot_ci_cpu.json", "dev/pilot_ci_cpu/merged.jsonl"),
    "pilot_ci_gpu": ("dev/pilot_ci_gpu.json", "dev/pilot_ci_gpu/merged.jsonl"),
    "pmrt_nl": ("dev/pmrt_nl.json", "dev/pmrt_nl/merged.jsonl"),
    "pmrt_nl_cost": ("dev/pmrt_nl_cost.json", "dev/pmrt_nl_cost/merged_e2.jsonl"),
    "pmrt_nl_cost_e4": ("dev/pmrt_nl_cost_e4.json", "dev/pmrt_nl_cost/merged_e4r3.jsonl"),
}
needs_records = pytest.mark.skipif(
    not EVAL_MERGED.exists(), reason="study4 record sets not fetched"
)


def digest(pairs) -> str:
    body = "".join(f"{k}\t{p}\n" for k, p in sorted(pairs))
    return hashlib.sha256(body.encode()).hexdigest()


@pytest.mark.claim("PRO-FROZEN")
def test_eval_cost_table_matches_its_sidecar_and_the_amendment():
    sidecar = EVAL_COST_TABLE.with_name(EVAL_COST_TABLE.name + ".sha256").read_text()
    digest_, name = sidecar.split()
    assert name.lstrip("*") == EVAL_COST_TABLE.name
    assert digest_ == sha_lf(EVAL_COST_TABLE.read_bytes()) == COST_TABLE_SHA
    assert COST_TABLE_SHA in (PROTOCOLS / "study4" / "PROTOCOL_A_AMENDMENTS.md").read_text()
    assert spec_sha256(load_cost_table(EVAL_COST_TABLE)) == COST_TABLE_CANONICAL_SHA


@pytest.mark.claim("PRO-FROZEN")
def test_eval_plan_reproduces_the_frozen_key_set_and_shards(kit):
    spec, _ = load_spec(EVAL_SPEC)
    cost = load_cost_table(EVAL_COST_TABLE)
    units = expand(spec)
    assert len(units) == 150960
    assert digest(assignment(units, 96, cost).items()) == ASSIGNMENT_SHA
    p = plan(spec, 96, cost)
    assert (p["mode"], p["units"], len(p["part_est_cpu_s"])) == ("eval", 150960, 96)
    assert sum(p["by_role"].values()) == 150960


def smoke_spec(tmp_path: Path) -> Path:
    spec = {
        "name": "t1_smoke",
        "budget_cpu_s": None,
        "arms": {
            "dummy": {"ref": "cdd_oran.xmethod.runner:DummyMethod", "declare": "by"},
            "corr": {
                "ref": "cdd_oran.xmethod.methods.corr:Corr",
                "config": {"arm": "native"},
                "declare": "by",
            },
            "pmrt_eq": {
                "ref": "cdd_oran.xmethod.methods.pmrt_core:PmrtCore",
                "config": {"covariates": "eq"},
                "declare": "by",
            },
            "pcorr_eq": {
                "ref": "cdd_oran.xmethod.methods.pcorr:PCorrMethod",
                "config": {"arm": "eq"},
                "declare": "by",
            },
        },
        "blocks": [
            {
                "role": "measure",
                "worlds": ["E1", "E2", "E3", "E4", "E5"],
                "regimes": ["R2"],
                "ns": [500],
                "kappas": [0.25],
                "seeds": [3000000],
            }
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


@pytest.mark.smoke
@pytest.mark.claim("DEF-ENV-ARMS")
def test_t1_one_job_per_environment_end_to_end(kit, opts, tmp_path):
    spec_path = smoke_spec(tmp_path)
    o = opts(isolate=can_isolate())
    summary = run_study(spec_path, tmp_path / "run", parts=2, opts=o)
    assert integrity_problems(summary) == []
    assert summary["n_ok"] == summary["n_expected"] == 20
    fields = json.loads((kit.DATA / "eval_record_fields.json").read_text())
    child = {"child_cpu_s", "child_maxrss_mb", "child_wall_s"}
    for r in iter_jsonl(tmp_path / "run" / "merged.jsonl.gz"):
        have, want = set(r), set(fields["record"])
        assert have - want == set()
        assert want - have <= (set() if can_isolate() else child)
        for block in ("job", "versions", "code", "run_mode", "integrity", "host", "limits"):
            assert set(r[block]) <= set(fields[block]), block
        assert r["versions"]["runner"].startswith("oran-causal")
        assert r["campaign"].startswith("oran-causal")


@pytest.mark.slow
@needs_records
@pytest.mark.claim("PRO-FROZEN")
def test_eval_records_keys_and_parts_equal_the_plan(kit):
    spec, _ = load_spec(EVAL_SPEC)
    cost = load_cost_table(EVAL_COST_TABLE)
    stamped = [(r["key"], r["integrity"]["part"]) for r in iter_jsonl(EVAL_MERGED)]
    assert len(stamped) == 150960
    assert digest(stamped) == digest(assignment(expand(spec), 96, cost).items())


@pytest.mark.slow
@needs_records
@pytest.mark.parametrize("name", sorted(DEV_SETS))
def test_dev_records_keys_equal_the_plan_and_summary(name):
    spec_rel, merged_rel = DEV_SETS[name]
    spec, _ = load_spec(SPECS / spec_rel)
    want = {u.key for u in expand(spec)}
    merged = RECORDS / "study4_dev" / f"{merged_rel}.gz"
    have = {r["key"] for r in iter_jsonl(merged)}
    frozen = json.loads((RECORDS / "study4_dev" / f"{merged_rel}.summary.json").read_text())
    assert have <= want
    assert sorted(want - have) == frozen["missing"]
    assert len(want) == frozen["n_expected"]


@pytest.mark.slow
@needs_records
def test_merging_the_frozen_eval_records_is_byte_identical(tmp_path):
    spec, _ = load_spec(EVAL_SPEC)
    plain = tmp_path / "in.jsonl"
    with gzip.open(EVAL_MERGED, "rb") as src, plain.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    out = tmp_path / "out.jsonl"
    summary = merge(spec, [plain], out)
    assert out.read_bytes() == plain.read_bytes()
    frozen = json.loads((EVAL_MERGED.parent / "merged.jsonl.summary.json").read_text())
    assert summary == frozen
    assert frozen["python"] == [FROZEN_EVAL_PYTHON]

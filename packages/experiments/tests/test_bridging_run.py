import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from oran_causal.analysis.bridging.plan import dataset_seed, spec_families, spec_slices
from oran_causal.experiments.bridging import bridge as B
from oran_causal.experiments.bridging import run as R
from oran_causal.experiments.campaign.shard import partition
from oran_causal.experiments.campaign.spec import sha_lf, spec_sha256
from oran_causal.experiments.paths import SPECS, WORKSPACE_ROOT
from oran_causal.sim.controlled.datasets.generate import dataset_hash

GOLDEN = Path(__file__).parent / "data" / "bridging_partition_golden.json"
PLACEBO_SRC = WORKSPACE_ROOT / "data" / "records" / "study1_raw" / "e6p-disc-v4plc-1-a"
SPEC_RECORDS = {
    "exp_b_pilot": ("f7bca0d4933512abe8d94f8cabc91c66", "603160b372f4847a82ee01b84b08802a"),
    "exp_b_rest": ("14300e90b240e16a11a6ef8acd36ba69", "773242cbdac2315249ef13e9f4608a50"),
}
needs_records = pytest.mark.skipif(
    not (R.RESULTS / "merged.jsonl.gz").exists(), reason="bridging records not present"
)


def _spec(name: str) -> dict:
    return R.load_spec(SPECS / "exp_b" / f"{name}.json")


def _sha(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


@pytest.mark.parametrize("name", ["exp_b", "exp_b_pilot", "exp_b_rest"])
def test_expansion_and_partition_match_cdd_oran(name: str) -> None:
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))["specs"][name]
    spec = _spec(name)
    assert spec_sha256(spec) == g["spec_sha256"]
    units = R.expand(spec, R.spec_counts(spec))
    assert len(units) == g["n_units"]
    assert _sha([u.key for u in units]) == g["units_sha256"]
    cost = None
    if g["cost_table"]:
        if not (R.RESULTS / "p1_projection.json").exists():
            pytest.skip("p1_projection.json not present")
        proj = json.loads((R.RESULTS / "p1_projection.json").read_text(encoding="utf-8"))
        cost = proj["cost_table"]
    parts = partition(units, g["parts"], cost)
    assert [len(p) for p in parts] == g["part_sizes"]
    assert [_sha([u.key for u in p]) for p in parts] == g["part_sha256"]
    assert [parts[0][0].key, parts[0][-1].key] == g["first_part"]
    assert [parts[-1][0].key, parts[-1][-1].key] == g["last_part"]


@pytest.mark.parametrize("name", sorted(SPEC_RECORDS))
def test_spec_hashes_match_the_records(name: str) -> None:
    spec_sha, file_sha = SPEC_RECORDS[name]
    assert spec_sha256(R.campaign_spec(_spec(name))).startswith(spec_sha)
    assert sha_lf((SPECS / "exp_b" / f"{name}.json").read_bytes()).startswith(file_sha)


def test_sub_spec_slices() -> None:
    pilot = _spec("exp_b_pilot")
    assert pilot["base_spec"] == "scratchpad/xmethod/specs/exp_b/exp_b.json"
    assert [s.name for s in spec_slices(pilot)] == ["s60_0", "placebo"]
    rest = {s.name for s in spec_slices(_spec("exp_b_rest"))}
    assert len(rest) == 17
    assert not rest & {"s60_0", "placebo"}


@pytest.mark.parametrize(
    ("arms", "families", "match"),
    [
        ({"a|b": {"ref": "x", "declare": "by"}}, None, "needs 'ref'"),
        ({"a": {"ref": "x", "declare": "max"}}, None, "needs 'ref'"),
        ({"a": {"ref": "x", "declare": "tau"}}, ["sleep", "x"], "unknown families"),
    ],
)
def test_load_spec_rejects(tmp_path: Path, arms: dict, families: list | None, match: str) -> None:
    spec = {"arms": arms} | ({"families": families} if families else {})
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        R.load_spec(p)


def test_counts_from_v4_reports_missing_slices() -> None:
    spec = {"slices": ["60"], "arms": {}}
    with pytest.raises(ValueError, match="no unit counts"):
        R.counts_from_v4(spec, {"slices_result": {}})


def test_project_interpolates_log_log(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = {
        "slices": ["placebo"],
        "budget_cpu_s": 50,
        "counts": "unused",
        "arms": {"a": {"ref": "x", "declare": "by", "size_factor": 2}},
    }
    counts = {"placebo": {"carrier": 100, "sleep": 400, "ptx": 1600, "prot_min": 0}}
    dev = {"a": {"100": {"E1": [1.0]}, "400": {"E1": [4.0], "E2": [2.0]}}}
    monkeypatch.setattr(R, "spec_counts", lambda _spec: counts)
    res = R.project(spec, dev)
    t = res["cost_table"]
    assert t["a|E6|carrier|n100"] == pytest.approx(2.0)
    assert t["a|E6|sleep|n400"] == pytest.approx(8.0)
    assert t["a|E6|ptx|n1600"] == pytest.approx(32.0)
    assert res["n_units"] == 3
    assert res["projected_over_budget"] == []


@needs_records
def test_remerge_reproduces_the_stored_merge(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    with gzip.open(R.RESULTS / "merged.jsonl.gz", "rb") as fi, src.open("wb") as fo:
        shutil.copyfileobj(fi, fo)
    out = tmp_path / "merged.jsonl.gz"
    R.merge(_spec("exp_b"), [src], out, newline="\r\n")
    with gzip.open(out, "rb") as fh:
        assert fh.read() == src.read_bytes()
    stored = R.RESULTS / "merged.jsonl.summary.json"
    assert (tmp_path / "merged.jsonl.summary.json").read_bytes() == stored.read_bytes()


@needs_records
@pytest.mark.skipif(not PLACEBO_SRC.exists(), reason="study1_raw placebo logs not present")
def test_placebo_table_rebuilds_the_record_datasets(tmp_path: Path) -> None:
    info = B.build_table({"placebo": [PLACEBO_SRC]}, tmp_path, workers=1)["placebo"]
    ref = json.loads((R.RESULTS / "p0" / B.INFO_JSON).read_text(encoding="utf-8"))["placebo"]
    assert info == ref
    sha: dict[int, set[str]] = {}
    with gzip.open(R.RESULTS / "merged.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("dataset_sha256"):
                sha.setdefault(r["job"]["seed"], set()).add(r["dataset_sha256"])
    spec = _spec("exp_b_pilot")
    tab = B.Table.load(tmp_path / "e6units_placebo.npz")
    counts = R.spec_counts(spec)
    for s in (s for s in spec_slices(spec) if s.stage == "placebo"):
        for f in spec_families(spec):
            ds = B.make_dataset(tab, f, s.seeds, s.split, s.name)
            assert ds.n == counts[s.name][f]
            assert sha[dataset_seed(s.split, f)] == {dataset_hash(ds)}

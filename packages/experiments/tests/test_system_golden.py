import json
from pathlib import Path

import pytest

from oran_causal.experiments.system import study1, study2, study3
from oran_causal.experiments.system.common import to_json

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / "system_jobs_golden.json"
ENERGY_RTOL = 1e-3
UNITS_RTOL = 0.1
FAST = (
    "s1/dev/188000/0/40",
    "s2c/dev_conf/186080/0/40",
    "s2a/eval/187000/MG:PMRT/40",
    "s3/dev/191000/CAL:allaccept/40",
)
SLOW = (
    "s1/placebo/188700/0/40",
    "s1/gt/188800/0/100/2",
    "s2c/placebo/186700/0/40",
    "s2c/gt/186900/0/100/2",
    "s2a/dev_repro/186080/incumbent/40",
    "s2a/eval/187000/freeze/40",
    "s3/eval/191100/CS:PMRT/40",
)


def run_spec(spec: str) -> dict:
    kind, stage, seed, x, *rest = spec.split("/")
    short = float(rest[0])
    max_labels = int(rest[1]) if len(rest) > 1 else None
    if kind == "s1":
        return study1.episode_job(study1.Episode(stage, int(seed), int(x)), True, short, max_labels)
    if kind == "s2c":
        return study2.episode_job(stage, int(seed), int(x), True, short, max_labels)
    if kind == "s2a":
        maps = study2.load_maps() if stage == "eval" else None
        return study2.conf_job(int(seed), x, stage, maps, True, short)
    art = study3.load_artifact() if stage == "eval" else None
    return study3.cs_job(int(seed), x, stage, art, True, short)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["jobs"]


def check(spec: str, golden: dict) -> None:
    rec = json.loads(json.dumps(run_spec(spec), default=to_json))
    want = golden[spec]
    got = (rec.get("lab_outcome") or rec).get("energy_j")
    assert len(rec.get("gt_labels", [])) == want["n_labels"]
    assert len(rec.get("units", [])) == pytest.approx(want["n_units"], rel=UNITS_RTOL)
    assert got == pytest.approx(want["energy_j"], rel=ENERGY_RTOL)


@pytest.mark.parametrize("spec", FAST)
def test_smoke_job_matches_old_code(spec: str, golden: dict) -> None:
    check(spec, golden)


@pytest.mark.slow
@pytest.mark.parametrize("spec", SLOW)
def test_smoke_job_matches_old_code_slow(spec: str, golden: dict) -> None:
    check(spec, golden)

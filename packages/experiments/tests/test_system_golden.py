import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

from oran_causal.experiments.system import study1, study2, study3
from oran_causal.experiments.system.common import to_json

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / f"system_jobs_golden_{'win32' if sys.platform == 'win32' else 'linux'}.json"
SKIP = {"cpu_s", "secs", "rss_mb", "collect_cpu_s", "label_cpu_s", "panel_cpu_s", "_spec"}
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


def norm(o: Any, in_labels: bool = False) -> Any:
    if isinstance(o, dict):
        if set(o) == {"b64", "shape", "dtype"}:
            raw = zlib.decompress(base64.b64decode(o["b64"]))
            return {"f32": raw.hex(), "shape": o["shape"]}
        return {
            k: norm(v, in_labels or k == "gt_labels")
            for k, v in o.items()
            if k not in SKIP and not (in_labels and k in ("secs", "cpu_s"))
        }
    if isinstance(o, list):
        return [norm(v, in_labels) for v in o]
    return o


def digest(rec: dict) -> str:
    return hashlib.sha256(json.dumps(norm(rec), sort_keys=True).encode()).hexdigest()


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
    if not GOLDEN.exists():
        pytest.skip(f"no golden for platform {sys.platform}")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["jobs"]


def check(spec: str, golden: dict) -> None:
    rec = json.loads(json.dumps(run_spec(spec), default=to_json))
    want = golden[spec]
    got = (rec.get("lab_outcome") or rec).get("energy_j")
    assert (len(rec.get("units", [])), len(rec.get("gt_labels", []))) == (
        want["n_units"],
        want["n_labels"],
    )
    assert got == want["energy_j"]
    assert digest(rec) == want["sha256"]


@pytest.mark.parametrize("spec", FAST)
def test_smoke_job_equals_old_code(spec: str, golden: dict) -> None:
    check(spec, golden)


@pytest.mark.slow
@pytest.mark.parametrize("spec", SLOW)
def test_smoke_job_equals_old_code_slow(spec: str, golden: dict) -> None:
    check(spec, golden)

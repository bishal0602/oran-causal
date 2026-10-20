import json
import shutil
from pathlib import Path

import pytest

from oran_causal.core.pmrt.units import PmrtConfig
from oran_causal.methods.system import artifact as ar
from oran_causal.methods.system import pmrt as pm

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "packages" / "experiments" / "artifacts"
ARTIFACT = ARTIFACTS / ar.ARTIFACT_NAME
MSCRPLUS = ARTIFACTS / ar.MSCRPLUS_NAME
PROTOCOL = ROOT / "packages" / "experiments" / "protocols" / "study1" / "DISCOVERY_PMRT_ADDENDUM.md"


def test_frozen_artifact_verifies_against_its_pinned_hash() -> None:
    rep = ar.verify(ARTIFACT, MSCRPLUS)
    assert rep["problems"] == []
    assert rep["sha256"] == ar.ARTIFACT_SHA256 == ar.sidecar_sha(ARTIFACT)
    assert rep["hyp"] == 60
    table = json.loads(
        (ROOT / "packages" / "experiments" / "manifests" / "recorded.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = table["inputs"][f"artifacts/{ar.ARTIFACT_NAME}"]["sha256"]
    assert recorded in PROTOCOL.read_text(encoding="utf-8")


def test_artifact_loads_the_frozen_statistic_and_layer() -> None:
    params, priors = ar.load_artifact(ARTIFACT)
    assert set(priors) == set(pm.STATS)
    assert all(p.n_prior == pm.N_PRIOR for p in priors.values())
    assert len(params["hyp"]) == 60
    for hp in params["hyp"].values():
        assert set(hp["core"]["K"]) == {"plain", "loadsp"}
        assert set(hp["h"]) == {"plain", "loadsp"}
        assert hp["gb"] is None
    cfg = ar.pmrt_config(params, 199)
    assert cfg == PmrtConfig(B=199)


def test_verify_reports_a_changed_artifact(tmp_path: Path) -> None:
    a = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    a["layer"]["thr"] = 2.0
    a["config"]["n0"] = 10.0
    bad = tmp_path / ar.ARTIFACT_NAME
    bad.write_text(json.dumps(a, separators=(",", ":")), encoding="utf-8", newline="\n")
    shutil.copy(f"{ARTIFACT}.sha256", f"{bad}.sha256")
    problems = ar.verify(bad, MSCRPLUS)["problems"]
    for p in ("sha256", "config", "layer.thr", "superseded.config", "superseded.layer.thr"):
        assert p in problems


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        ar.load_artifact(f)

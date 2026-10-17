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
LEGACY = ARTIFACTS / ar.LEGACY_NAME
PROTOCOL = (
    ROOT
    / "packages"
    / "experiments"
    / "protocols"
    / "study1"
    / "E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md"
)


def test_frozen_artifact_verifies_against_its_pinned_hash() -> None:
    rep = ar.verify(ARTIFACT, LEGACY)
    assert rep["problems"] == []
    assert rep["sha256"] == ar.ARTIFACT_SHA256 == ar.sidecar_sha(ARTIFACT)
    assert rep["hyp"] == 60
    assert ar.ARTIFACT_SHA256 in PROTOCOL.read_text(encoding="utf-8")


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


def test_legacy_artifact_holds_the_same_learned_content() -> None:
    new, old = (json.loads(p.read_text(encoding="utf-8")) for p in (ARTIFACT, LEGACY))
    assert ar.learned_diff(new, old) == []
    assert new["supersedes"]["sha256"] == ar.sha_file(LEGACY) == ar.sidecar_sha(LEGACY)
    p_new, _ = ar.load_artifact(ARTIFACT)
    p_old, _ = ar.load_artifact(LEGACY)
    assert p_new["rho"] == p_old["rho"]
    assert p_new["ctx_keys"] == p_old["ctx_keys"]


def test_verify_reports_a_changed_artifact(tmp_path: Path) -> None:
    a = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    a["layer"]["thr"] = 2.0
    a["config"]["n0"] = 10.0
    bad = tmp_path / ar.ARTIFACT_NAME
    bad.write_text(json.dumps(a, separators=(",", ":")), encoding="utf-8", newline="\n")
    shutil.copy(f"{ARTIFACT}.sha256", f"{bad}.sha256")
    problems = ar.verify(bad, LEGACY)["problems"]
    for p in ("sha256", "config", "layer.thr", "legacy.config", "legacy.layer.thr"):
        assert p in problems


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        ar.load_artifact(f)

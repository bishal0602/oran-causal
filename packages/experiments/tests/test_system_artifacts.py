from pathlib import Path

import pytest

from oran_causal.experiments.paths import ARTIFACTS, WORKSPACE_ROOT
from oran_causal.experiments.system import artifacts as art
from oran_causal.experiments.system.common import read_records, sha_lf

RECORDS = WORKSPACE_ROOT / "data" / "records"
DISC = RECORDS / "study2_raw" / "e6p-conf-discan-2" / "out" / "disc_conf.json"
BOUNDS = RECORDS / "study2_3_decision" / "runs" / "e6p-cs-bounds-1" / "out" / "certsafe_bounds.json"
CALIB = RECORDS / "study2_3_decision" / "runs" / "e6p-cs-dev-1" / "certsafe_calib.json"
CS_DEV = RECORDS / "study2_3_decision" / "runs" / "e6p-cs-dev-1" / "all.jsonl"
frozen_inputs = pytest.mark.skipif(
    not all(p.exists() for p in (DISC, BOUNDS, CALIB, CS_DEV)),
    reason="frozen Study 2 / 3 records not fetched",
)


def test_verify_frozen_artifacts() -> None:
    vm, vc = art.verify_maps(), art.verify_certsafe()
    assert vm["ok"], vm["checks"]
    assert vc["ok"], vc["checks"]
    assert vm["matches_pin"]
    assert vc["matches_pin"]


def test_verify_detects_a_changed_signature(tmp_path: Path) -> None:
    a = art.read_json(ARTIFACTS / "E6P_CONF_MAPS.json")
    a["arms"]["MG:GT"]["signature"] = "x"
    p = tmp_path / "m.json"
    art.write_artifact(a, p)
    v = art.verify_maps(p)
    assert not v["ok"]
    assert not v["checks"]["signatures"]
    assert not v["matches_pin"]
    assert p.with_name("m.json.sha256").read_text() == f"{sha_lf(p)}  m.json\n"


def test_build_maps_refuses_invalid_discovery() -> None:
    disc = {"schema": art.SCHEMA_DISC, "label": {"label": "INVALID"}}
    with pytest.raises(ValueError, match="INVALID"):
        art.build_maps(disc, "0")
    with pytest.raises(ValueError, match="not a disc_conf"):
        art.build_maps({"schema": "x"}, "0")


def test_calibrate() -> None:
    def job(arm: str, v: float, n: int) -> dict:
        pc = {"policy": {"dir_sleep_+1": n}}
        out = dict.fromkeys(
            ("prot_viol", "prot_ue_s", "energy_j", "ue_s", "nonprot_embb_viol"), 1.0
        )
        return out | {"viol_ue_s": v, "rlf": 2.0, "ll_viol": 0.0, "policy_counts": pc, "arm": arm}

    R = {
        191000: {"CAL:allaccept": job("CAL:allaccept", 10.0, 2), "noarb": job("noarb", 10.0, 0)},
        191001: {"CAL:allaccept": job("CAL:allaccept", 20.0, 4), "noarb": job("noarb", 20.0, 0)},
    }
    cb = art.calibrate(R)
    assert cb["G"] == {"v": 15.0, "rlf": 2.0}
    assert ["sleep", 1, 3.0] in cb["N"]
    assert cb["repro"] == {"per_seed": {191000: 0.0, 191001: 0.0}, "ok": True}
    assert cb["noarb_alias"]


@frozen_inputs
@pytest.mark.claim("PRO-FROZEN")
def test_rebuild_frozen_artifacts_byte_identical(tmp_path: Path) -> None:
    disc = art.read_json(DISC)
    m = art.build_maps(disc, sha_lf(DISC))
    art.write_artifact(m, tmp_path / "E6P_CONF_MAPS.json")
    sha = {
        "certsafe_bounds.json": sha_lf(BOUNDS),
        "certsafe_calib.json": sha_lf(CALIB),
        "disc_conf.json": sha_lf(DISC),
    }
    c = art.build_certsafe(art.read_json(BOUNDS), art.read_json(CALIB), disc, sha)
    art.write_artifact(c, tmp_path / "E6P_CERTSAFE.json")
    for name in ("E6P_CONF_MAPS.json", "E6P_CERTSAFE.json"):
        for f in (name, name + ".sha256"):
            assert (tmp_path / f).read_bytes() == (ARTIFACTS / f).read_bytes(), f
    assert art.verify_maps(disc_json=DISC)["checks"]["disc_maps"]


@frozen_inputs
def test_calib_report_reproduces_frozen_calibration() -> None:
    new = art.calib_report(read_records(CS_DEV))
    old = art.read_json(CALIB)
    for k in ("mode", "seeds", "n", "G", "N", "noarb_alias", "tau_preview"):
        assert new[k] == old[k], k
    assert {str(s): v for s, v in new["repro"]["per_seed"].items()} == old["repro"]["per_seed"]
    assert [list(k[:5]) for k in new["platform"]["keys"]] == [
        k[:5] for k in old["platform"]["keys"]
    ]

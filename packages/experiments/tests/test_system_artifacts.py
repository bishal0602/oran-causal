from pathlib import Path

import pytest

from oran_causal.experiments.paths import ARTIFACTS, WORKSPACE_ROOT
from oran_causal.experiments.system import artifacts as art
from oran_causal.experiments.system.common import sha_lf

RECORDS = WORKSPACE_ROOT / "data" / "records"


def test_verify_detects_a_changed_signature(tmp_path: Path) -> None:
    a = art.read_json(ARTIFACTS / "study2_maps.json")
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

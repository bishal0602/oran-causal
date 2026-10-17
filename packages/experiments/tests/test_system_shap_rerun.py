import math

from oran_causal.experiments.paths import WORKSPACE_ROOT
from oran_causal.experiments.system import shap_rerun as sr
from oran_causal.methods.system import baselines as bl


def test_control_diff_paths() -> None:
    frozen = {"a": {"b": [1.0, math.nan, "x"]}, "cpu_s": 1.0, "keep": 2, "gone": 0}
    assert sr.control_diff(frozen, frozen) == []
    got = {"a": {"b": [1.0, math.nan, "y"]}, "cpu_s": 9.0, "keep": 3, "new": 0}
    assert sorted(sr.control_diff(frozen, got)) == ["/a/b/2", "/gone", "/keep", "/new"]
    assert sr.control_diff({"l": [1, 2]}, {"l": [1]}) == ["/l (len 2 vs 1)"]
    assert sr.control_diff({"x": 0.1}, {"x": 0.1 * (1 + 1e-9)}) == []
    assert sr.control_diff({"x": 0.1}, {"x": 0.1 * (1 + 1e-4)}) == ["/x"]
    assert sr.control_diff({"x": True}, {"x": 1.0}) == []


def test_renamed_moves_map_arms_only() -> None:
    res = {k: {"MG:gbdt_perm@dev": 1, "MG:pmrt@dev": 2} for k in sr.SPLICE_KEYS}
    res["report"] = {"MG:gbdt_perm@dev": 3}
    out = sr._renamed(res, "gbdt_perm", "shap_gbdt")
    assert all(set(out[k]) == {"MG:shap_gbdt@dev", "MG:pmrt@dev"} for k in sr.SPLICE_KEYS)
    assert out["report"] == res["report"]


def test_methods_are_named_by_importance() -> None:
    assert {m: bl.IMPORTANCE[m] for m in sr.METHODS} == {
        "shap_gbdt": "tree_shap",
        "gbdt_perm": "permutation",
    }


def test_driver_verify_uses_the_stored_keys() -> None:
    path = WORKSPACE_ROOT / "packages" / "experiments" / "artifacts" / "M.json"
    v = {"path": str(path), "ok": True, "sha256": "s", "pinned": "p", "matches_pin": False}
    assert sr.driver_verify(v, "MAPS") == {
        "path": "packages/experiments/artifacts/M.json",
        "ok": True,
        "sha256": "s",
        "driver_MAPS_SHA256": "p",
        "driver_matches": False,
    }


def test_portable_writes_workspace_paths_relative() -> None:
    assert sr.portable(WORKSPACE_ROOT / "data" / "x.json") == "data/x.json"
    assert sr.portable("--study") == "--study"

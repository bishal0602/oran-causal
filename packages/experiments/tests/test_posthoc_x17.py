"""X17: frozen slope paths stay pinned, the strict maps artifact and the MG:PMRT run driver."""

import functools
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.posthoc import x17 as ax17
from oran_causal.experiments.posthoc import tools, x17
from oran_causal.experiments.system import artifacts, study2, study2_discovery
from oran_causal.experiments.system.common import Part, read_records, sha_lf
from oran_causal.methods.system import referees as rf

FROZEN_MAPS = study2.MAPS.path
NO_UNITS: Any = None


def _capture(monkeypatch) -> list:
    seen: list = []

    def fake(ud, f, rel, kpi, *, window_s=rf.WINDOW_S):
        seen.append(window_s)
        return {"kind": "slope", "beta": 1.0, "se": 1.0, "n": 2, "z": 1.0}

    monkeypatch.setattr(rf, "design_slope_bound", fake)
    monkeypatch.setattr(rf, "naive_slope_se", lambda ud, f, rel, kpi: {"beta": 1.0})
    return seen


def test_frozen_artifact_paths_pin_the_frozen_centre(monkeypatch):
    seen = _capture(monkeypatch)
    artifacts.bounds_tables(NO_UNITS, {"MG:PMRT": {("sleep", "own", "pv"): 1.0}})
    study2_discovery.pmrt_map({("sleep", "own", "pv"): {"declared": True, "sign": 1}}, NO_UNITS)
    assert seen
    assert set(seen) == {rf.FROZEN_WINDOW_S}


def test_strict_bounds_only_when_asked(monkeypatch):
    seen = _capture(monkeypatch)
    artifacts.bounds_tables(NO_UNITS, {"MG:PMRT": {("sleep", "own", "pv"): 1.0}}, rf.WINDOW_S)
    assert rf.WINDOW_S in seen
    assert seen[-1] == rf.FROZEN_WINDOW_S


def test_referee_tools_slope_is_the_frozen_one():
    t = tools.referee_tools()
    assert isinstance(t.design_slope, functools.partial)
    assert t.design_slope.keywords == {"window_s": rf.FROZEN_WINDOW_S}
    assert t.decision_signature is rf.decision_signature


def _frozen() -> dict:
    return json.loads(FROZEN_MAPS.read_text(encoding="utf-8"))


def _pmrt(frozen: dict, scale: float = 1.0) -> dict:
    m = rf.map_from_json(frozen["arms"]["MG:PMRT"]["map"])
    return {h: scale * float(b) for h, b in m.items() if b is not None}


def test_strict_artifact_with_the_frozen_map_keeps_every_derived_field():
    frozen = _frozen()
    doc = x17.strict_maps_artifact(frozen, _pmrt(frozen), "f" * 64, "e" * 64)
    for k in ("jobs", "alias_of", "all_accept_signature"):
        assert doc[k] == frozen[k]
    for arm, v in frozen["arms"].items():
        assert doc["arms"][arm]["signature"] == v["signature"]
        assert doc["arms"][arm]["decision_table"] == v["decision_table"]
    assert doc["x17"]["replaced"] == ["MG:PMRT"]


def test_strict_artifact_scales_only_pmrt_and_refuses_a_sign_flip():
    frozen = _frozen()
    m = _pmrt(frozen, 1.05)
    doc = x17.strict_maps_artifact(frozen, m, "f" * 64, "e" * 64)
    assert rf.map_from_json(doc["arms"]["MG:PMRT"]["map"]) == m
    for arm in frozen["arms"]:
        if arm != "MG:PMRT":
            assert doc["arms"][arm]["map"] == frozen["arms"][arm]["map"]
    h = next(iter(m))
    with pytest.raises(ValueError, match="signs"):
        x17.strict_maps_artifact(frozen, {**m, h: -m[h]}, "f" * 64, "e" * 64)


def test_load_strict_checks_the_pin_and_the_source(tmp_path):
    frozen = _frozen()
    path = tmp_path / x17.MAPS_STRICT
    bad = x17.strict_maps_artifact(frozen, _pmrt(frozen), "0" * 64, "e")
    sha = artifacts.write_artifact(bad, path)
    with pytest.raises(ValueError, match="pinned"):
        x17.load_strict(path, "1" * 64)
    with pytest.raises(ValueError, match="derived"):
        x17.load_strict(path, sha)
    good = x17.strict_maps_artifact(frozen, _pmrt(frozen), study2.MAPS.want, "e")
    sha = artifacts.write_artifact(good, path)
    assert sha == sha_lf(path)
    assert "MG:PMRT" in x17.load_strict(path, sha)["maps"]


def test_stage_seeds_and_blocks():
    assert x17.seeds_of("confirm") == list(ax17.CONFIRM_SEEDS)
    full = x17.seeds_of("full")
    assert len(full) == 320
    assert (full[0], full[-1]) == (187000, 191259)
    assert {ax17.block_of(s) for s in full[:160]} == {"study2"}
    assert {ax17.block_of(s) for s in full[160:]} == {"study3"}
    with pytest.raises(ValueError, match="stage"):
        x17.seeds_of("eval")


def test_run_shards_and_header(tmp_path, monkeypatch):
    calls: list = []

    def fake(seed, stage, maps, **kw):
        calls.append(seed)
        return {
            "kind": "job",
            "key": [seed, "MG:PMRT"],
            "seed": seed,
            "arm": "MG:PMRT",
            "smoke": kw.get("smoke", False),
            "psvr": 1.0,
            "rlf": 0,
            "cpu_s": 0.0,
        }

    monkeypatch.setattr(x17, "job", fake)
    monkeypatch.setattr(x17, "load_strict", lambda path, want: {"maps": {}, "alias": {}})
    monkeypatch.setattr(sys, "platform", "linux")
    out = tmp_path / "res_0.jsonl"
    n = x17.run(Part(0, 2), out, "confirm", tmp_path / "m.json", "a" * 64)
    assert n == 4
    assert calls == list(ax17.CONFIRM_SEEDS)[::2]
    head = read_records(out)[0]
    assert head["driver"] == ax17.DRIVER
    assert head[ax17.STAGE_KEY] == "confirm"
    assert head["maps_strict"]["sha256"] == "a" * 64
    assert x17.run(Part(0, 2), out, "confirm", tmp_path / "m.json", "a" * 64) == 0


def test_run_refuses_short_without_smoke_and_non_linux(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="smoke-only"):
        x17.run(Part(), tmp_path / "r.jsonl", "confirm", Path("m"), "a", short=5.0)
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux"):
        x17.run(Part(), tmp_path / "r.jsonl", "confirm", Path("m"), "a")

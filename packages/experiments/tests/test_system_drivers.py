from pathlib import Path

import pytest

from oran_causal.experiments.system import stages, study1, study2, study3
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.collect import RandomizedUnitPolicy


def test_study1_episodes() -> None:
    sizes = {s: len(study1.episodes(s)) for s in study1.STAGES}
    assert sizes == {"dev": 20, "eval": 600, "placebo": 40, "gt": 40}
    ev = study1.episodes("eval")
    assert (ev[0].seed, ev[-1].seed) == (188100, 188699)
    assert ev[0].key == ("eval", "v4", 188100)
    assert ev[359].fold == 2
    assert ev[359].slices() == {"slice60": 5, "slice120": 2}
    assert study1.episodes("gt")[0].fold is None
    assert study1.episodes("dev")[3].slices() == {"slice60": None, "slice120": None}
    with pytest.raises(ValueError, match="stage in"):
        study1.episodes("prof")


def test_study1_policies() -> None:
    factory, name = study1.policy("placebo")
    assert name == "placebo"
    assert type(factory(1)).__name__ == "PlaceboPolicy"
    factory, name = study1.policy("eval")
    assert name == "pi0"
    pol = factory(1)
    assert isinstance(pol, RandomizedUnitPolicy)
    assert pol.tables == study1.PI0_V4


def test_study2_jobs_and_seeds() -> None:
    maps = study2.load_maps()
    ev = study2.jobs("eval", maps)
    arms = study2.eval_arms(maps["alias"])
    assert len(ev) == 160 * len(arms)
    assert arms[: len(rf.ANCHORS)] == list(rf.ANCHORS)
    assert len(set(arms)) == len(arms)
    dev = study2.jobs("dev_conf", None)
    assert len(dev) == 20 + 2 * len(study2.REPRO_ARMS)
    assert [j.key for j in dev[:4]] == [
        ("dev", "conf", 186080),
        (186080, "incumbent"),
        (186080, "MG:allaccept"),
        (186080, "noarb"),
    ]
    assert study2.jobs("disc", None)[5].key == ("eval", "conf", 186105)
    assert len(study2.jobs("eval", maps, arms=frozenset({"noarb"}))) == 160
    with pytest.raises(ValueError, match="maps"):
        study2.jobs("eval", None)
    with pytest.raises(ValueError, match="outside"):
        study2.check_stage_seed(186100, "dev_conf")
    with pytest.raises(ValueError, match="no collection policy"):
        study2.policy("eval")


def test_study3_jobs() -> None:
    art = study3.load_artifact()
    dev = study3.jobs("dev", None)
    assert len(dev) == 40 + 2
    assert [j.key for j in dev[:3]] == [
        (191000, rf.CAL_ARM),
        (191000, "noarb"),
        (191001, rf.CAL_ARM),
    ]
    arms = study3.eval_arms(art["alias"])
    assert len(study3.jobs("eval", art)) == 160 * len(arms)
    assert {"MG:PMRT", "never_sleep", "incumbent", study3.PRIMARY_ARM} <= set(arms)
    with pytest.raises(ValueError, match="artifact"):
        study3.jobs("eval", None)


@pytest.mark.claim("PRO-FROZEN")
@pytest.mark.claim("PRO-SEEDS")
def test_frozen_documents_and_registry() -> None:
    for doc in (study1.PROTOCOL, study2.PROTOCOL, study3.PROTOCOL, study2.MAPS, study3.CERTSAFE):
        assert doc.status()["frozen"], doc.name
    for status in (study1.registry_status(), study2.registry_status(), study3.registry_status()):
        assert all(status.values()), status


@pytest.mark.claim("PRO-FROZEN")
def test_guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert study1.guard("eval", smoke=False)["freeze"]["frozen"]
    monkeypatch.setattr("sys.platform", "win32")
    with pytest.raises(RuntimeError, match="Linux"):
        study2.guard("disc", smoke=False)
    with pytest.raises(RuntimeError, match="Linux"):
        study3.guard("dev", smoke=False)
    assert study2.guard("dev_conf", smoke=False)["maps"] is None
    assert study3.guard("eval", smoke=True)["artifact"]["frozen"]
    changed = study2.MAPS.__class__(study2.MAPS.name, tmp_path / "none.json", study2.MAPS.want)
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(RuntimeError, match="maps artifact"):
        study2.guard("eval", smoke=False, maps_doc=changed)


def test_smoke_only_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smoke-only"):
        study1.run("dev", tmp_path / "o.jsonl", short=10.0)
    with pytest.raises(ValueError, match="smoke-only"):
        study3.run("dev", tmp_path / "o.jsonl", short=10.0)


def test_run_stage_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage in"):
        stages.run_stage(1, "maps", out=tmp_path)
    assert stages.ANALYSIS_STAGES[3] == ("bounds", "calib", "certsafe")

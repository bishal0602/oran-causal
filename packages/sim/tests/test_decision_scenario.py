import dataclasses

import pytest

from oran_causal.sim.system.decision import scenario as SC


def test_study_config_is_the_frozen_p3_surge_l40_plant() -> None:
    cfg = SC.make_cfg(SC.PAIR, SC.STRATUM, 188700, SC.LOAD_FACTOR["L40"])
    assert (cfg.seed, cfg.mix, cfg.scenario, cfg.load_factor) == (
        188700,
        "ES",
        "surge",
        1.7554687500000001,
    )
    assert (cfg.n_ue, cfg.n_pico, cfg.mobility, cfg.kpm, cfg.update) == (
        300,
        3,
        "ped",
        "nominal",
        False,
    )
    assert (cfg.warmup_s, cfg.scored_s) == (120.0, 600.0)
    assert cfg.e6p.xapps == ("PowerES", "SliceGuarantee")
    assert cfg.e6p.ptx_on
    assert cfg.e6p.prot_on
    SC.check_frozen(cfg)


def test_check_frozen_rejects_an_altered_field() -> None:
    cfg = SC.make_cfg("P3", 3, 1, SC.LOAD_FACTOR["L40"])
    with pytest.raises(ValueError, match="n_ue"):
        SC.check_frozen(dataclasses.replace(cfg, n_ue=200))
    with pytest.raises(ValueError, match="warmup_s"):
        SC.check_frozen(dataclasses.replace(cfg, warmup_s=20.0))


def test_smoke_shortens_only_the_episode() -> None:
    cfg = SC.make_cfg("P3", 3, 1, SC.LOAD_FACTOR["L40"], smoke=True)
    assert (cfg.warmup_s, cfg.scored_s) == (SC.SMOKE_S, SC.SMOKE_S)
    SC.check_frozen(cfg, smoke=True)
    with pytest.raises(ValueError, match="frozen"):
        SC.check_frozen(cfg)


def test_pairs_strata_and_deployed_xapps() -> None:
    assert SC.deployed("P3") == ("ES", "PowerES", "SliceGuarantee")
    assert SC.deployed("P2") == ("PowerES", "Coverage")
    assert SC.STRATA[SC.STRATUM] == ("surge", "L40")
    cfg = SC.make_cfg("*", 0, 1, SC.LOAD_FACTOR["L10"])
    assert (cfg.mix, cfg.scenario, cfg.e6p.xapps) == ("none", "base", ())

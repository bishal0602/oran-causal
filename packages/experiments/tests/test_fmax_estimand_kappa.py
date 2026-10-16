import json

import numpy as np
import pytest

from oran_causal.analysis.controlled.fmax import fmax_simulate, load_dependence
from oran_causal.experiments.campaign import estimand, fmax, kappa
from oran_causal.experiments.harness.records import clean
from oran_causal.experiments.paths import ARTIFACTS


def frozen_table() -> dict:
    return json.loads((ARTIFACTS / "fmax_table.json").read_text(encoding="utf-8"))


def as_json(x: dict) -> dict:
    return json.loads(json.dumps(clean(x)))


def frozen_dependence(kit) -> dict:
    dep = load_dependence(kit.DATA / "fmax_dependence.json")
    assert dep is not None
    return dep


@pytest.mark.claim("PRO-FROZEN")
def test_fmax_small_set_reproduces_the_frozen_table(kit):
    cells = fmax.table_sets()["R1_E3"]
    got = as_json(fmax_simulate(cells, 40, ("null_raw", "plac_raw"), frozen_dependence(kit)))
    assert got == frozen_table()["sets"]["R1_E3|S40"]


def test_fmax_table_sets_are_the_preregistered_cells():
    sets = fmax.table_sets()
    assert list(sets) == ["R1R2_all", "R1_all", "R2_all", "R1R2_E3", "R1_E3", "E4R3"]
    assert len(sets["R1R2_all"]) == 50
    assert {c[2] for c in sets["E4R3"]} == set(fmax.LAMS)
    assert all(c[2] == 1.0 for c in sets["R1_all"] if c[0] == "E4")


def test_pair_type_corr_groups_candidate_pairs():
    cands = [("A", "K1"), ("A", "K2"), ("B", "K1"), ("C", "K3")]
    zm = np.random.default_rng(0).standard_normal((50, 4))
    got = fmax.pair_type_corr(zm, cands)
    assert (got["n_pairs_same_source"], got["n_pairs_same_target"], got["n_pairs_other"]) == (
        1,
        1,
        4,
    )
    c = np.corrcoef(zm.T)
    assert got["same_source"] == pytest.approx(c[0, 1])
    assert got["same_target"] == pytest.approx(c[0, 2])


@pytest.mark.slow
@pytest.mark.claim("PRO-FROZEN")
def test_fmax_full_table_and_s_e4_reproduce_the_frozen_files(kit):
    dep = frozen_dependence(kit)
    assert as_json(fmax.table(dep)) == frozen_table()
    assert as_json(fmax.se4(dep))["S_E4"] == 300


@pytest.mark.claim("DEF-ENV")
def test_estimand_contrast_is_zero_on_null_edges(monkeypatch):
    monkeypatch.setattr(estimand, "S_STATES", 40)
    monkeypatch.setattr(estimand, "DEV_SEEDS", estimand.DEV_SEEDS[:2])
    row = estimand.check_cell("E1", "R1", 1.0, 60)
    assert row["pass_primary"]
    assert row["pass_secondary"]
    assert row["true_edges_zero_effect"] == []
    assert row["units_B"] == 2 * 60 * estimand.M_VALUES
    assert row["placebo_by_construction"] == ["P_placebo"]


def test_wrapped_advance_restores_the_classes():
    before = {c: c.__dict__["advance"] for c in estimand.PATCHED}
    rec = estimand.Recorder("E1")
    with estimand.wrapped_advance(rec, "E1", np.random.default_rng(0), skip=0):
        assert all(c.__dict__["advance"] is not before[c] for c in estimand.PATCHED)
    assert {c: c.__dict__["advance"] for c in estimand.PATCHED} == before


@pytest.mark.slow
@pytest.mark.claim("DEF-ENV")
def test_estimand_check_reproduces_the_frozen_run(kit):
    got = as_json(estimand.run(1000, log=lambda _msg: None))
    want = json.loads((kit.DATA / "estimand_check.json").read_text(encoding="utf-8"))
    assert got["all_pass_primary"] is want["all_pass_primary"] is True
    assert len(got["cells"]) == len(want["cells"]) == 18
    for g, w in zip(got["cells"], want["cells"], strict=True):
        for key in (
            "pass_primary",
            "pass_secondary",
            "violations_primary",
            "true_edges_zero_effect",
        ):
            assert g[key] == w[key]
        for fam in ("primary_null", "primary_true", "secondary_null", "secondary_true"):
            for edge, stats in w[fam].items():
                for name, value in stats.items():
                    assert g[fam][edge][name] == pytest.approx(value, rel=1e-9, abs=1e-12)


def synthetic_design(rng: np.random.Generator) -> kappa.DesignData:
    rows = kappa.N_DEV * 30
    x = rng.normal(size=(rows, 6))
    y = np.column_stack([3 * x[:, i] + rng.normal(scale=0.5, size=rows) for i in range(5)])
    ep = np.repeat(np.arange(kappa.N_DEV), 30)
    cell = np.tile(np.arange(3), rows // 3)
    return kappa.DesignData(x, y, ep, cell)


def test_kappa_point_and_bootstrap_are_deterministic():
    d = synthetic_design(np.random.default_rng(0))
    names = [f"base|f{i}" for i in range(5)] + ["lead|f5"]
    cols = kappa.columns(names, "base")
    assert cols == [0, 1, 2, 3, 4]
    train, test = kappa.halves()
    a = kappa.point(d, train, test, cols)
    assert a == kappa.point(d, train, test, cols)
    assert set(a) == {f"{s}/{k}" for s, _ in kappa.SPLITS for k in ("pooled", "within")}
    boots = kappa.bootstrap(d, cols, range(3), log=lambda _msg: None)
    assert boots == kappa.bootstrap(d, cols, range(3), log=lambda _msg: None)
    ci = kappa.bootstrap_ci(boots)
    lo, hi = ci["pooled"]["u"]["pv"]
    assert lo <= hi


def test_kappa_helpers():
    assert np.isinf(kappa.kappa_of(1.0))
    assert kappa.kappa_of(0.5) == pytest.approx(1.0)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert kappa.r2(y, y) == 1.0
    assert kappa.r2(y, y, np.array([0, 0, 1, 1])) == 1.0
    a = np.arange(6.0).reshape(3, 2)
    assert np.isnan(kappa.lag(a, 1)[0]).all()
    assert (kappa.lag(a, 1)[1:] == a[:-1]).all()
    assert (kappa.lag(a, -1)[:-1] == a[1:]).all()

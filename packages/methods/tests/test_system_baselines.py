import dataclasses
import sys

import numpy as np
import pytest

from oran_causal.core.data import UnitTable
from oran_causal.methods.system import baselines as bd
from oran_causal.sim.system.decision.records import build_unit_data
from oran_causal.sim.system.decision.synthetic import SyntheticSpec, synthetic_records
from oran_causal.sim.system.decision.vocab import HYPOTHESES

pytestmark = pytest.mark.filterwarnings(
    "ignore::RuntimeWarning", "ignore::UserWarning", "ignore::FutureWarning"
)

PLANT = ("sleep", "nbr", "pv")  # synthetic_records: sleep adds effect * x to nbr load and pv
OWN = ("ptx", "own", "e")  # synthetic_records: ptx adds 50 * x to own energy (the nuisance)


@pytest.fixture(scope="module")
def data() -> UnitTable:
    spec = SyntheticSpec(n_ep=6, units_per_ep=60, effect=120.0)
    return build_unit_data(synthetic_records(np.random.default_rng(3), spec))


def _recovers(scores: dict, h: tuple[str, str, str] = PLANT, sign: int = 1) -> None:
    tau = bd.tune_tau(scores, "far_fpr")
    assert np.isfinite(tau["tau"])
    d = bd.declare(scores, tau["tau"])
    assert d[h]["declared"], (h, scores[h], tau)
    assert d[h]["sign"] == sign
    assert sum(v["declared"] for k, v in d.items() if k[1] == "far") <= 1


@pytest.mark.parametrize("name", ["corr", "granger", "int"])
def test_fast_baselines_recover_the_planted_edge(data: UnitTable, name: str) -> None:
    _recovers(bd.SCORERS[name](data))


@pytest.mark.slow
def test_shap_gbdt_recovers_the_planted_edge(data: UnitTable) -> None:
    sc = bd.score_shap(data)
    assert "fit_shap_importances" in sc["_note"]
    assert sc["_importance"] == bd.IMPORTANCE["shap_gbdt"] == "tree_shap"
    _recovers(sc)


def test_shap_gbdt_never_falls_back(data: UnitTable, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "oran_causal.methods.controlled.shap_dag", None)
    with pytest.raises(ImportError):
        bd.score_shap(data)
    with pytest.raises(ImportError):
        bd.run_baselines(data, {"pooled": data}, ("shap_gbdt",))


@pytest.mark.slow
def test_gbdt_perm_is_selected_by_name_and_recorded(data: UnitTable) -> None:
    sc = bd.SCORERS["gbdt_perm"](data, seed=0)
    assert sc["_importance"] == bd.IMPORTANCE["gbdt_perm"] == "permutation"
    assert sc[PLANT]["score"] > 0
    sub = data.subset(data.family == 2)
    rb = bd.run_baselines(sub, {"pooled": sub}, ("gbdt_perm",))
    assert rb["gbdt_perm"]["importance"] == "permutation"


def test_two_tower_recovers_the_planted_edge(data: UnitTable) -> None:
    sc = bd.score_two_tower(data, sizes={"epochs": 300})
    assert "'epochs': 300" in sc["_note"]
    _recovers(sc)


def test_qacm_recovers_an_own_cell_edge_and_never_declares_nbr(data: UnitTable) -> None:
    sc = bd.score_qacm(data, relations=("own", "far"), ann=False)
    tau = bd.tune_tau(sc, "far_fpr")
    d = bd.declare(sc, tau["tau"], bd.DECL_RELATIONS["qacm"])
    assert d[OWN]["declared"]
    assert d[OWN]["sign"] == 1
    assert not any(v["declared"] for k, v in d.items() if k[1] != "own")


def test_qacm_ann_path_runs(data: UnitTable) -> None:
    ann = bd.score_qacm(data.subset(data.family == 2), relations=("own",), ann=True)
    assert np.isfinite(ann[OWN]["score"])
    assert ann[OWN]["model"] in ("ANN", "PR")


def test_granger_by_and_tau_rules() -> None:
    nan = float("nan")
    sc: dict = {h: {"score": nan, "sign": 0, "p": nan} for h in HYPOTHESES}
    far = [h for h in HYPOTHESES if h[1] == "far"]
    for i, h in enumerate(far):
        sc[h] = {"score": float(i), "sign": 1, "p": 10.0**-i}
    t = bd.tune_tau(sc, "far_fpr")
    assert t["tau"] == len(far) - 2  # second-largest far score
    d = bd.declare(sc, t["tau"])
    assert sum(v["declared"] for v in d.values()) == 1
    few: dict = {h: {"score": nan, "sign": 0} for h in HYPOTHESES}
    few[far[0]] = {"score": 1.0, "sign": 1}
    assert bd.tune_tau(few, "far_fpr")["tau"] == float("inf")
    sc[PLANT] = {"score": 100.0, "sign": 1, "p": 1e-30}
    tp = bd.tune_tau(sc, "physics_f1")
    assert bd.declare(sc, tp["tau"])[PLANT]["declared"]
    gb = bd.granger_by(sc)
    assert gb[PLANT]["declared"]
    assert not gb[far[0]]["declared"]


def test_run_baselines_freezes_dev_tau(data: UnitTable) -> None:
    dev = data.subset(data.episode < 3)
    ev = data.subset(data.episode >= 3)
    sup = dataclasses.replace(bd.Support(), min_episodes=2)
    out = bd.run_baselines(dev, {"pooled": ev}, methods=("corr", "granger"), sup=sup)
    assert out["corr"]["tau"]["rule"] == "far_fpr"
    assert "granger_by" in out
    s = out["corr"]["splits"]["pooled"]
    tau = out["corr"]["tau"]["tau"]
    for v in s["declared"].values():
        assert v["declared"] == (np.isfinite(v["score"]) and v["score"] > tau)


def test_method_baseline_declares_with_the_dev_tau(data: UnitTable) -> None:
    dev = data.subset(data.episode < 3)
    ev = data.subset(data.episode >= 3)
    m = bd.method_baseline("corr", dev)
    tau = bd.tune_tau(bd.score_corr(dev), "far_fpr")["tau"]
    ref = bd.declare(bd.score_corr(ev), tau)
    got = m(ev, 0)
    assert {h for h, v in got.items() if v["declared"]} == {
        h for h, v in ref.items() if v["declared"]
    }

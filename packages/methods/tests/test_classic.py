import dataclasses
import importlib.machinery
import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from oran_causal.core.covariates import CovariateSpec, design_covariates
from oran_causal.core.data import Result
from oran_causal.core.scoring import score
from oran_causal.methods.controlled import _classic
from oran_causal.methods.controlled._classic import (
    PLACEBO,
    Scored,
    by_declare,
    columns,
    cond_set,
    declare,
    is_diagnostic_column,
    pcorr,
    placebo_tau,
    source_kind,
)
from oran_causal.methods.registry import ARMS
from oran_causal.sim.controlled.datasets.generate import generate_dataset

CLASSIC = ("pc", "notears", "shap_dag", "two_tower", "corr", "granger", "pcorr_hac", "cdl")
FAST = {"two_tower": {"epochs": 30}, "cdl": {"max_gradient_steps": 60}}
VENDORED = Path(_classic.__file__).parent / "_vendor" / "notears" / "linear.py"
R3 = CovariateSpec(setpoints="none", lagged_actions=False, concurrent="all")
EXACT_FIT = [
    "granger",
    pytest.param("pc", marks=pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")),
]


def _run(name, data, **cfg):
    return ARMS[name]().run(data, dict(FAST.get(name, {}), **cfg))


@pytest.fixture(scope="module")
def syn(make_classic):
    return make_classic(800, 3_000_000, b=1.0)


@pytest.mark.parametrize("name", CLASSIC)
def test_result_contract(name, syn):
    data, _ = syn
    r = _run(name, data)
    assert isinstance(r, Result)
    assert r.method == name
    assert r.version
    assert [(e.source, e.target) for e in r.edges] == list(data.candidates)
    assert all(e.sign in (-1, 0, 1) for e in r.edges)
    assert all(isinstance(e.declared, bool) for e in r.edges)
    assert set(r.notes["family"].values()) <= {"action", "kpi"}
    assert r.notes["family"][f"{PLACEBO}->Y0"] == "action"
    assert r.notes["family"]["Y0->Y0"] == "kpi"
    assert r.cpu_s >= 0
    assert r.notes["threads"] == 1
    if ARMS[name]().uses_p:
        assert all(e.p is None or 0.0 <= e.p <= 1.0 for e in r.edges)
    else:
        assert not any(e.declared for e in r.edges)


@pytest.mark.parametrize("name", ["pc", "notears", "corr", "granger"])
def test_planted_strong_edges_rank_top(name, make_classic):
    data, truth = make_classic(3000, 3_000_001, b=1.0)
    r = _run(name, data)
    sc = {(e.source, e.target): e.score for e in r.edges}
    for (s, t), sign in truth.signs.items():
        nulls = [
            sc[(a, t)]
            for a in data.action_names
            if (a, t) not in truth.edges and np.isfinite(sc[(a, t)])
        ]
        assert sc[(s, t)] > max(nulls), (s, t)
        e = next(e for e in r.edges if (e.source, e.target) == (s, t))
        assert e.sign == sign, (name, s, t)


@pytest.mark.claim("PRO-TUNE")
def test_placebo_tau_rule(make_classic):
    m = ARMS["notears"]()
    dev = [make_classic(500, 3_000_000 + i, b=1.0)[0] for i in range(10)]
    cfg = m.tune(dev)
    pl = sorted(e.score for d in dev for e in m.run(d, {}).edges if e.source == PLACEBO)
    k = math.ceil((len(pl) + 1) * 0.95 - 1e-9)
    assert len(pl) == 30
    assert k == 30
    assert cfg["tau"] == pytest.approx(pl[min(k, len(pl)) - 1])
    assert cfg["tau_rule"] == "placebo_conformal_0.05"
    assert cfg["n_placebo_scores"] == 30
    n_pl = sum(e.declared for d in dev for e in m.run(d, cfg).edges if e.source == PLACEBO)
    assert n_pl == 0
    assert placebo_tau([])["tau"] == float("inf")


@pytest.mark.claim("PRO-TUNE")
def test_tune_respects_arm(make_classic):
    dev = [make_classic(300, 3_000_020 + i, b=1.0, regime="R2")[0] for i in range(3)]
    pc = ARMS["pc"]()
    eq, nat, dflt = (
        pc.tune(dev, config={"arm": "eq"}),
        pc.tune(dev, config={"arm": "native"}),
        pc.tune(dev),
    )
    assert (eq["arm"], nat["arm"], dflt["arm"]) == ("eq", "native", "eq")
    assert dflt["tau"] == eq["tau"]
    assert eq["tau"] != nat["tau"]
    assert nat["tau"] == placebo_tau([pc.run(d, {"arm": "native"}) for d in dev])["tau"]
    nt = ARMS["notears"]()
    assert (
        nt.tune(dev, config={"arm": "eq"})["tau"] == nt.tune(dev, config={"arm": "native"})["tau"]
    )


@pytest.mark.claim("PRO-TUNE")
def test_tune_refuses_truth():
    with pytest.raises(ValueError, match="truth-free"):
        ARMS["corr"]().tune([], truth_free=False)


@pytest.mark.claim("THM-BY-FIN")
def test_by_is_per_family():
    s = [
        Scored("A0", "Y0", "action", 5.0, 1, 1e-4),
        Scored("A1", "Y0", "action", 0.1, 1, 0.8),
        Scored("Y0", "Y0", "kpi", 3.0, 1, 0.03),
        Scored("Y1", "Y0", "kpi", 0.1, 1, 0.9),
    ]
    dec, notes = declare(s, {"q": 0.05}, uses_p=True)
    assert dec == [True, False, False, False]
    assert list(by_declare([1e-4, 0.8, 0.03, 0.9])) == [True, False, False, False]
    s[2] = dataclasses.replace(s[2], p=0.01)
    assert declare(s, {"q": 0.05}, uses_p=True)[0][2]
    assert "per family" in notes["declare_rule"]


@pytest.mark.claim("THM-BY-FIN")
def test_by_declare_counts_finite_pvalues_only():
    assert by_declare([0.008, np.nan, np.nan, np.nan, 0.6]).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert not by_declare([0.008, 0.9, 0.9, 0.9, 0.6]).any()
    assert not by_declare([np.nan, np.nan]).any()


def test_source_name_resolution(syn):
    data, _ = syn
    assert source_kind(data, "A1") == ("action", 1)
    for nm in ("Y2", "Y2_lag", "Y2[t]", "lag_Y2"):
        assert source_kind(data, nm) == ("kpi", 2)
    with pytest.raises(KeyError):
        source_kind(data, "nope")


@pytest.mark.parametrize("name", ["pc", "notears", "corr", "granger"])
def test_world_without_lags(name, make_classic):
    data, _ = make_classic(400, 3_000_002)
    data = dataclasses.replace(data, X_kpi_lag=np.full_like(data.X_kpi_lag, np.nan))
    r = _run(name, data)
    kpi = [e for e in r.edges if r.notes["family"][f"{e.source}->{e.target}"] == "kpi"]
    assert kpi
    assert all(np.isnan(e.score) and not e.declared for e in kpi)
    assert any(np.isfinite(e.score) for e in r.edges)


@pytest.mark.parametrize("name", ["pc", "notears"])
def test_context_column_runs(name, make_classic):
    data, _ = make_classic(400, 3_000_003)
    ctx = np.random.default_rng(0).normal(size=(data.n, 2))
    r = _run(name, dataclasses.replace(data, context=ctx))
    assert len(r.edges) == len(data.candidates)


def test_pcorr_sign_rule(syn):
    data, _ = syn
    for name in ("pc", "shap_dag", "two_tower"):
        r = _run(name, data)
        assert r.notes["sign_rule"] == "pcorr_given_Z"
        sg = {(e.source, e.target): e.sign for e in r.edges}
        assert sg[("A0", "Y0")] == 1
        assert sg[("A1", "Y1")] == -1
    rng = np.random.default_rng(1)
    z = rng.normal(size=5000)
    x = z + rng.normal(size=5000)
    y = 2 * z - 0.5 * x + rng.normal(size=5000)
    assert np.corrcoef(x, y)[0, 1] > 0
    assert pcorr(x, y, z[:, None]) < 0


def _vendored() -> ModuleType:
    spec = importlib.util.spec_from_loader(
        "notears_linear_vendored",
        importlib.machinery.SourceFileLoader("notears_linear_vendored", str(VENDORED)),
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendored_notears_is_byte_identical():
    import hashlib

    digests = {
        VENDORED: "39644acd5b5d5ebc5fb5be5783365137eaa3b309df73bc0a75bad1942a0a426f",
        VENDORED.parent
        / "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    }
    for path, digest in digests.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path.name


def test_notears_bk_equals_vendored_without_knowledge():
    from oran_causal.methods.controlled.notears import notears_linear_bk

    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 5))
    x[:, 2] += 1.5 * x[:, 0]
    x[:, 4] -= x[:, 2]
    w1 = _vendored().notears_linear(x, lambda1=0.1, loss_type="l2")
    w2 = notears_linear_bk(x, lambda1=0.1, loss_type="l2")
    assert np.array_equal(w1, w2)
    fb = np.zeros((5, 5), dtype=bool)
    fb[0, 2] = True
    w3 = notears_linear_bk(x, lambda1=0.1, loss_type="l2", w_threshold=0.0, forbid=fb)
    assert w3[0, 2] == 0.0


def test_pc_equals_causallearn_pc(make_classic):
    from causallearn.graph.GraphClass import CausalGraph
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    from oran_causal.methods.controlled.pc import PcSettings, run_pc

    data, _ = make_classic(1500, 3_000_004, b=1.0)
    cols = columns(data)
    d = np.column_stack([cols.actions, cols.lags, cols.Y])
    names = (
        [f"A:{a}" for a in data.action_names]
        + [f"L:{k}" for k in cols.lag_names]
        + [f"Y:{k}" for k in data.kpi_names]
    )
    kinds = [nm[0] for nm in names]
    tiers = [1 if k == "Y" else 0 for k in kinds]
    forbidden = {
        (i, j)
        for i, ki in enumerate(kinds)
        for j, kj in enumerate(kinds)
        if i != j and ((kj == "A" and ki in "LYA") or (ki == "A" and kj in "LC"))
    }
    cg1, rec = run_pc(d, names, tiers, forbidden, PcSettings())
    nodes = CausalGraph(d.shape[1], names).G.nodes
    bk = BackgroundKnowledge()
    for i, t in enumerate(tiers):
        bk.add_node_to_tier(nodes[i], t)
    for i, j in forbidden:
        bk.add_forbidden_by_node(nodes[i], nodes[j])
    cg2 = pc(
        d, 0.05, "fisherz", stable=True, uc_rule=0, uc_priority=2,
        background_knowledge=bk, show_progress=False, node_names=names,
    )  # fmt: skip
    assert np.array_equal(cg1.G.graph, cg2.G.graph)
    assert rec.n_tests > 0
    assert rec.fallbacks == 0


@pytest.mark.parametrize("arm", ["eq", "native"])
def test_pc_native_matches_score_at_alpha(syn, arm):
    data, _ = syn
    r = _run("pc", data, arm=arm)
    nat = set(r.notes["native_declared"])
    for e in r.edges:
        if np.isfinite(e.score):
            at_alpha = e.score >= -np.log10(0.05) - 1e-12
            assert (f"{e.source}->{e.target}" in nat) == at_alpha, (e.source, e.target)


def test_granger_equals_statsmodels(make_classic):
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests

    data, _ = make_classic(600, 3_000_005, b=1.0, burn=50)
    r = _run("granger", data, arm="native")
    assert r.notes["label"].startswith("pairwise Granger")
    p = {(e.source, e.target): e.p for e in r.edges}
    k = np.vstack([data.X_kpi_lag[:1], data.Y])
    for src, tgt in (("Y1", "Y2"), ("Y0", "Y1"), ("Y2", "Y0")):
        s, t = data.kpi_names.index(src), data.kpi_names.index(tgt)
        res = grangercausalitytests(np.column_stack([k[:, t], k[:, s]]), maxlag=[1])
        assert p[(src, tgt)] == pytest.approx(res[1][0]["ssr_ftest"][1], rel=1e-8)
    y, own, x = data.Y[:, 0], data.X_kpi_lag[:, 0], data.X_action[:, 0]
    full = sm.OLS(y, sm.add_constant(np.column_stack([own, x]))).fit()
    assert p[("A0", "Y0")] == pytest.approx(float(full.f_test("x2 = 0").pvalue), rel=1e-8)


def test_corr_equals_numpy(syn):
    data, _ = syn
    r = _run("corr", data)
    e = next(e for e in r.edges if (e.source, e.target) == ("A1", "Y1"))
    assert e.score == pytest.approx(abs(np.corrcoef(data.X_action[:, 1], data.Y[:, 1])[0, 1]))


def test_shap_dag_hgbdt_option_equals_e2_port(make_classic):
    from oran_causal.methods.controlled.shap_dag import fit_shap_importances

    data, _ = make_classic(300, 3_000_006, b=1.0)
    r = _run("shap_dag", data, seed=0, regressor="hgbdt")
    imp, _ = fit_shap_importances(data.X_action, data.Y[:, 1], seed=0)
    sc = {(e.source, e.target): e.score for e in r.edges}
    got = np.array([sc[(a, "Y1")] for a in data.action_names])
    assert np.allclose(got, imp / np.std(data.Y[:, 1]))
    assert all(np.isnan(sc[(k, "Y1")]) for k in data.kpi_names)
    assert r.notes["regressor"] == "hgbdt"


def test_shap_dag_default_is_xgboost_package_defaults(make_classic):
    import shap
    import xgboost

    data, _ = make_classic(300, 3_000_006, b=1.0)
    r = _run("shap_dag", data, seed=0)
    assert r.notes["regressor"] == "xgboost"
    assert r.notes["packages"]["xgboost"] == xgboost.__version__
    model = xgboost.XGBRegressor(random_state=0, n_jobs=1).fit(data.X_action, data.Y[:, 1])
    assert model.get_params()["early_stopping_rounds"] is None
    imp = np.abs(shap.TreeExplainer(model).shap_values(data.X_action)).mean(0)
    sc = {(e.source, e.target): e.score for e in r.edges}
    got = np.array([sc[(a, "Y1")] for a in data.action_names])
    assert np.allclose(got, imp / np.std(data.Y[:, 1]), rtol=1e-6)
    assert max(sc[(a, "Y1")] for a in data.action_names) == sc[("A1", "Y1")]


def test_shap_dag_no_early_stopping_at_any_n():
    from oran_causal.methods.controlled.shap_dag import make_regressor

    assert make_regressor("hgbdt", 0).get_params()["early_stopping"] is False
    assert make_regressor("xgboost", 0).get_params()["early_stopping_rounds"] is None
    with pytest.raises(ValueError, match="unknown regressor"):
        make_regressor("rf", 0)


def test_two_tower_score_is_row_share_of_gate(make_classic):
    import torch

    from oran_causal.methods.controlled.two_tower import (
        LABEL,
        TowerSizes,
        TrainConfig,
        fit_two_tower,
    )

    data, _ = make_classic(300, 3_000_007, b=1.0)
    r = _run("two_tower", data, seed=0, epochs=20)
    torch.set_num_threads(1)
    s, _ = fit_two_tower(data.X_action, data.Y, TowerSizes(4, 3), TrainConfig(20, 0.01, 1e-3, 0))
    sc = {(e.source, e.target): e.score for e in r.edges}
    got = np.array([[sc[(a, k)] for a in data.action_names] for k in data.kpi_names])
    gate = np.array([[r.notes["gate"][k][a] for a in data.action_names] for k in data.kpi_names])
    assert np.allclose(gate, s, atol=1e-6)
    assert np.allclose(got, s / s.sum(1, keepdims=True), atol=1e-6)
    assert np.allclose(got.sum(1), 1.0)
    nat = {
        f"{a}->{k}"
        for i, k in enumerate(data.kpi_names)
        for j, a in enumerate(data.action_names)
        if s[i, j] >= 0.10 * s[i].max()
    }
    assert set(r.notes["native_declared"]) == nat
    assert r.notes["label"] == LABEL
    assert "not the published" in LABEL


def test_pc_notes_fallback_rate_and_granger_label(syn):
    data, _ = syn
    r = _run("pc", data)
    rate = r.notes["fisherz_pinv_fallbacks"] / r.notes["n_ci_tests"]
    assert r.notes["fisherz_pinv_fallback_rate"] == pytest.approx(rate)
    assert "pMax" in r.notes["score"]
    assert _run("granger", data, arm="native").notes["label"].startswith("pairwise Granger")
    assert _run("granger", data).notes["label"].startswith("conditional Granger")


@pytest.mark.claim("DEF-ENV-ARMS")
def test_cond_set_from_shared_helper(make_classic):
    data, _ = make_classic(300, 3_000_009, regime="R2")
    data = dataclasses.replace(data, context=np.random.default_rng(0).normal(size=(data.n, 1)))
    eq = cond_set(data, "action", 0, "eq")
    names, m, mask = design_covariates(data, CovariateSpec(focal=0))
    assert eq.names == names
    assert np.array_equal(eq.Z, m)
    assert np.array_equal(eq.mask, mask)
    assert int(eq.mask.sum()) == data.n - 2
    assert not eq.mask[:2].any()
    assert {f"sp:{a}" for a in data.action_names} <= set(eq.names)
    assert {f"{a}@t-{k}" for a in data.action_names for k in (1, 2)} <= set(eq.names)
    assert "concurrent:A0" not in eq.names
    assert "concurrent:A1" in eq.names
    assert "ctx:c0" in eq.names
    nat = cond_set(data, "action", 0, "native")
    assert nat.names == design_covariates(data, dataclasses.replace(R3, focal=0))[0]
    assert nat.mask.all()
    assert not any(nm.startswith(("sp:", "A1@")) for nm in nat.names)
    kp = cond_set(data, "kpi", 1, "eq")
    assert "lag_kpi:Y1" not in kp.names
    assert "lag_kpi:Y0" in kp.names
    assert {f"concurrent:{a}" for a in data.action_names} <= set(kp.names)
    r1 = cond_set(make_classic(300, 3_000_009, regime="R1")[0], "action", 0, "eq")
    assert not any(nm.startswith("sp:") for nm in r1.names)


@pytest.mark.claim("DEF-ENV-ARMS")
def test_granger_eq_equals_statsmodels_ols(make_classic):
    import statsmodels.api as sm

    data, _ = make_classic(500, 3_000_010, b=1.0, regime="R2")
    r = _run("granger", data)
    assert r.notes["arm"] == "eq"
    assert r.notes["n_rows_used"] == data.n - 2
    p = {(e.source, e.target): e.p for e in r.edges}
    cases = (
        ("action", 0, "A0", "Y0", data.X_action[:, 0]),
        ("action", 2, "A2", "Y1", data.X_action[:, 2]),
        ("kpi", 1, "Y1", "Y2", data.X_kpi_lag[:, 1]),
    )
    for fam, src, name, tgt, x in cases:
        cs = cond_set(data, fam, src, "eq")
        m = cs.mask
        design = sm.add_constant(np.column_stack([cs.Z[m], x[m]]))
        fit = sm.OLS(data.Y[m, data.kpi_names.index(tgt)], design).fit()
        restriction = np.zeros((1, design.shape[1]))
        restriction[0, -1] = 1.0
        assert p[(name, tgt)] == pytest.approx(float(fit.f_test(restriction).pvalue), rel=1e-6), (
            name,
            tgt,
        )


def test_pc_eq_arm_exogenous_design_nodes(make_classic):
    data, _ = make_classic(800, 3_000_011, b=1.0, regime="R2")
    r = _run("pc", data)
    assert r.notes["arm"] == "eq"
    assert r.notes["n_rows_used"] == data.n - 2
    assert "sp:A0" in r.notes["eq_covariates"]
    assert "A0@t-1" in r.notes["eq_covariates"]
    n_design = (
        len(r.notes["eq_covariates"])
        - len(r.notes["eq_dropped"])
        - len(r.notes["dropped_constant"])
    )
    assert r.notes["n_vars"] == len(data.action_names) + 2 * len(data.kpi_names) + n_design
    assert [(e.source, e.target) for e in r.edges] == list(data.candidates)
    rn = _run("pc", data, arm="native")
    assert rn.notes["arm"] == "native"
    assert "eq_covariates" not in rn.notes
    assert rn.notes["n_vars"] == len(data.action_names) + 2 * len(data.kpi_names)


@pytest.mark.parametrize("name", ["corr", "notears", "shap_dag", "two_tower"])
def test_native_only_methods_note_no_interface(name, syn):
    data, _ = syn
    r = _run(name, data, arm="eq")
    assert r.notes["arm"] == "native"
    assert r.notes["arm_note"] == "no conditioning interface"
    assert r.notes["arm_requested"] == "eq"
    with pytest.raises(ValueError, match="arm must be"):
        _run(name, data, arm="both")


def test_meta_families_and_diagnostic_by_family(make_classic):
    data, _ = make_classic(600, 3_000_008, b=1.0)
    cands = data.candidates
    meta = {
        "primary_candidates": tuple(c for c in cands if c[0] in data.action_names and c[0] != "A2"),
        "secondary_candidates": tuple(c for c in cands if c[0] in data.kpi_names),
        "diagnostic_candidates": tuple(c for c in cands if c[0] == "A2"),
    }
    r = _run("corr", dataclasses.replace(data, meta=meta))
    fam = r.notes["family"]
    assert fam["A2->Y0"] == "diagnostic"
    assert fam["A0->Y0"] == "action"
    assert fam["Y0->Y0"] == "kpi"
    sc = [
        Scored(e.source, e.target, fam[f"{e.source}->{e.target}"], e.score, e.sign, e.p)
        for e in r.edges
    ]
    for f in ("action", "kpi", "diagnostic"):
        idx = [i for i, x in enumerate(sc) if x.family == f]
        assert [r.edges[i].declared for i in idx] == list(by_declare([sc[i].p for i in idx]))


@pytest.mark.parametrize("name", EXACT_FIT)
def test_exact_fit_candidates_are_not_testable(name, make_classic):
    data, _ = make_classic(300, 3_000_012, b=1.0, noise=0.0, regime="R2")
    r = _run(name, data)
    by = {(e.source, e.target): e for e in r.edges}
    for k in data.kpi_names:
        e = by[(PLACEBO, k)]
        assert np.isnan(e.score)
        assert e.p is None
        assert not e.declared
    assert r.notes["not_testable_reason"] == "exact fit: deterministic world under Z"
    assert "not_testable" not in r.notes
    assert r.notes["n_not_testable"] == len(r.notes["not_testable_edges"]) > 0
    assert f"{PLACEBO}->Y0" in r.notes["not_testable_edges"]
    assert "A0->Y0" not in r.notes["not_testable_edges"]
    assert np.isfinite(by[("A0", "Y0")].score)
    if name == "granger":
        assert by[("A0", "Y0")].p < 1e-10
    clean = _run(name, make_classic(300, 3_000_012, b=1.0, regime="R2")[0])
    assert "not_testable_edges" not in clean.notes
    assert "not_testable_reason" not in clean.notes


@pytest.mark.parametrize("name", EXACT_FIT)
def test_score_accepts_not_testable_notes(name, make_classic):
    data, truth = make_classic(300, 3_000_012, b=1.0, noise=0.0, regime="R2")
    r = _run(name, data)
    s = score(r, truth, candidates=data.candidates, kpi_sources=data.kpi_names)
    listed = sorted(r.notes["not_testable_edges"])
    assert listed
    assert s["not_testable_edges"] == listed
    assert s["n_not_testable_overridden"] == 0
    assert s["placebo_not_testable"] == len(data.kpi_names)
    assert not set(listed) & set(s["declared_edges"])


@pytest.mark.parametrize("arm", ["eq", "native"])
def test_granger_collinear_source_is_not_testable(arm, make_classic):
    data, _ = make_classic(300, 3_000_030, b=1.0, regime="R2")
    lag = data.X_kpi_lag.copy()
    lag[:, 2] = lag[:, 1]
    r = _run("granger", dataclasses.replace(data, X_kpi_lag=lag), arm=arm)
    nan = sorted(f"{e.source}->{e.target}" for e in r.edges if np.isnan(e.score))
    assert nan
    assert sorted(r.notes["not_testable_edges"]) == nan == sorted(r.notes["not_testable_collinear"])
    assert r.notes["not_testable_reason"] == "source collinear with Z"
    assert r.notes["n_not_testable"] == len(nan)
    assert all(e.p is None and not e.declared for e in r.edges if np.isnan(e.score))


def _conf_scrambled(ds):
    j = ds.action_names.index("P_placebo_conf")
    x = ds.X_action.copy()
    x[:, j] = np.random.default_rng(11).permutation(x[:, j])[::-1]
    des = list(ds.designs)
    if des[j].propensity is not None:
        des[j] = dataclasses.replace(des[j], propensity=des[j].propensity[::-1].copy())
    return dataclasses.replace(ds, X_action=x, designs=tuple(des))


@pytest.mark.parametrize("regime", ["R3", "R4"])
def test_cond_set_never_holds_placebo_conf(regime):
    ds, _ = generate_dataset("E4", regime, 300, 3_000_000, lam=1.5, kappa=0.25)
    conf = ds.action_names.index("P_placebo_conf")
    for arm in ("eq", "native", "eq_min"):
        for ai in range(len(ds.action_names)):
            names = cond_set(ds, "action", ai, arm).names
            if ai == conf:
                assert "concurrent:P0" in names or arm == "eq"
                spec = CovariateSpec(focal=ai) if arm == "eq" else dataclasses.replace(R3, focal=ai)
                assert names == design_covariates(ds, spec)[0]
            else:
                assert not any(
                    is_diagnostic_column(nm) or "P_placebo_conf" in nm for nm in names
                ), (arm, ai)
        assert not any("P_placebo_conf" in nm for nm in cond_set(ds, "kpi", 0, arm).names)
    assert is_diagnostic_column("P_placebo_conf@t-2:missing")
    assert not is_diagnostic_column("concurrent:P_placebo")


def _rows(r):
    return repr(
        [
            (e.source, e.target, e.score, e.p, e.sign, e.declared)
            for e in r.edges
            if e.source != "P_placebo_conf"
        ]
    )


@pytest.mark.parametrize(
    ("name", "arm"),
    [
        ("pc", "eq"),
        ("pc", "native"),
        ("granger", "eq"),
        ("pcorr_hac", "native"),
        ("pcorr_hac", "eq_min"),
        ("notears", "native"),
        ("shap_dag", "native"),
        ("two_tower", "native"),
    ],
)
@pytest.mark.parametrize("regime", ["R3", "R4"])
def test_placebo_conf_does_not_change_primary(name, arm, regime):
    ds, _ = generate_dataset("E4", regime, 400, 3_000_001, lam=1.5, kappa=0.25)
    a, b = _run(name, ds, arm=arm), _run(name, _conf_scrambled(ds), arm=arm)
    assert _rows(a) == _rows(b)
    diag = [e for e in a.edges if e.source == "P_placebo_conf"]
    assert diag
    assert all(np.isfinite(e.score) for e in diag)
    if name == "pc":
        assert a.notes["diagnostic_graph"]["n_vars"] > a.notes["n_vars"]
        assert not any("P_placebo_conf" in x for x in a.notes.get("eq_covariates", []))
    if name in ("notears", "shap_dag", "two_tower"):
        assert "R-37" in a.notes["diagnostic_fit"]["rule"]
    if name == "two_tower":
        assert all("P_placebo_conf" not in g for g in a.notes["gate"].values())

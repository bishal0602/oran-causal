import dataclasses

import numpy as np
import pytest
from scipy import stats

from oran_causal.core.covariates import CovariateSpec, design_covariates
from oran_causal.core.data import EdgeResult, Result
from oran_causal.core.pvalues import SequentialP, sequential_p_from_draws
from oran_causal.core.scoring import placebo_tau, tau_rule_name
from oran_causal.methods.controlled import mscr
from oran_causal.methods.controlled._citests import (
    PLACEBO,
    CITestBase,
    by_families,
    method_seed,
    pcorr_given_z,
    prepare,
    standardize,
    tau_from_results,
)
from oran_causal.methods.controlled._classic import cond_set
from oran_causal.methods.controlled._rcot import (
    frozen_config_v2,
    rcot_statistic,
    residual_collapse,
    rff_residuals,
    z_blocks,
)
from oran_causal.methods.controlled.mscr import MSCRMethod
from oran_causal.methods.controlled.mscr_bank import Execution, MSCRConfig, discover_mscr
from oran_causal.methods.controlled.pcorr import PCorrMethod, partial_correlation_scores
from oran_causal.methods.controlled.rcot2 import RCoT2Method, block_perm_sequential
from oran_causal.sim.controlled.datasets.generate import generate_dataset

DEV0 = 3_000_000
FAST = {
    "mscr": {"n_perm": 299, "bc_h": 20, "n_jobs": 1},
    "pcorr": {},
    "rcot2": {"block_perm_reps": 99},
}
ARMS = (MSCRMethod, PCorrMethod, RCoT2Method)
R3 = CovariateSpec(setpoints="none", lagged_actions=False, concurrent="all")


def _tau(scores):
    edges = tuple(
        EdgeResult(PLACEBO, f"k{i}", float(v), None, 0, False) for i, v in enumerate(scores)
    )
    return placebo_tau([Result("tau", "0", edges, 0.0, {})])


def _canon(nm):
    return nm.removeprefix("concurrent:").removeprefix("lag_kpi:")


def _ols_p(prep, i, j):
    x = np.column_stack([np.ones(prep.S.shape[0]), prep.S[:, i], prep.S[:, prep.z_cols(i)]])
    beta, *_ = np.linalg.lstsq(x, prep.Y[:, j], rcond=None)
    res = prep.Y[:, j] - x @ beta
    dof = x.shape[0] - x.shape[1]
    se = np.sqrt(res @ res / dof * np.linalg.inv(x.T @ x)[1, 1])
    return 2 * stats.t.sf(abs(beta[1] / se), dof), np.sign(beta[1])


def test_sequential_p_rules():
    draws = np.random.default_rng(0).normal(size=500)
    for obs in (-3.0, 0.0, 1.5, 9.0):
        seq = SequentialP(obs, 500, 20)
        for v in draws:
            if seq.add(v):
                break
        assert (seq.p, seq.k) == sequential_p_from_draws(obs, draws, 20)
    assert sequential_p_from_draws(-1.0, np.zeros(100), 20) == (20 / 20, 20)
    assert sequential_p_from_draws(5.0, np.zeros(100), 20) == (1 / 101, 100)
    assert sequential_p_from_draws(0.5, np.r_[np.ones(30), np.zeros(70)], None) == (31 / 101, 100)


@pytest.mark.claim("PRO-TUNE")
def test_tau_rule_conformal():
    assert _tau([0.1, 0.5, 0.3, np.nan]) == 0.5
    assert _tau([]) == np.inf
    assert _tau([np.nan]) == np.inf
    s = np.arange(1.0, 40.0)
    assert _tau(s) == 38.0
    assert int(np.sum(s > _tau(s))) == 1
    s = np.arange(1.0, 101.0)
    assert _tau(s) == 96.0
    assert int(np.sum(s > _tau(s))) == 4


@pytest.mark.claim("THM-BY-FIN")
def test_by_families_are_separate():
    p = np.array([1e-4, 0.5, 0.5, 1e-4, 0.5])
    fam = ("action_kpi", "action_kpi", "action_kpi", "kpi_kpi", "kpi_kpi")
    assert by_families(p, fam).tolist() == [True, False, False, True, False]
    assert by_families(np.array([np.nan, 1e-6]), ("action_kpi", "action_kpi")).tolist() == [
        False,
        True,
    ]


def test_prepare_layout_nan_lag_and_context(make_ci):
    ds, _ = make_ci("planted", 300, DEV0)
    prep = prepare(ds)
    assert prep.S.shape == (ds.n, len(ds.action_names) + len(ds.kpi_names))
    assert prep.family.count("action_kpi") == len(ds.action_names) * len(ds.kpi_names)
    lag = ds.X_kpi_lag.copy()
    lag[:, 1] = np.nan
    ctx = np.random.default_rng(1).normal(size=(ds.n, 1))
    cands = tuple(c for c in ds.candidates if c[0] != "K1")
    ds2 = dataclasses.replace(
        ds, X_kpi_lag=lag, context=ctx, candidates=cands, meta={"context_names": ("Z",)}
    )
    p2 = prepare(ds2)
    assert "K1" not in p2.S_names
    assert p2.S_names[-1] == "ctx:Z"
    assert p2.n_ctx == 1
    with pytest.raises(ValueError, match="neither an action"):
        prepare(dataclasses.replace(ds2, candidates=(*ds2.candidates, ("K1", "K0"))))


def test_pcorr_given_z_matches_e1_partial_correlation(make_ci):
    ds, _ = make_ci("planted", 300, DEV0)
    prep = prepare(ds)
    xs, ys = standardize(prep.S), standardize(prep.Y)
    rho, df = pcorr_given_z(xs, ys, prep.pairs)
    ref = partial_correlation_scores(xs, ys)
    np.testing.assert_allclose(rho, [ref[j, i] for i, j in prep.pairs], atol=1e-10)
    assert df[0] == ds.n - 2 - (xs.shape[1] - 1)


def test_mscr_native_equals_discover_mscr(make_ci):
    ds, _ = make_ci("planted", 400, DEV0)
    m = MSCRMethod()
    r = m.run(ds, {**m.native_config(), "n_perm": 199, "n_jobs": 1})
    prep = prepare(ds)
    tested = sorted({i for i, _ in prep.pairs})
    x = prep.S[:, tested + [i for i in range(prep.S.shape[1]) if i not in tested]]
    seed = method_seed(ds.seed, m.method_key)
    run = Execution(1, "cpu")
    ref = discover_mscr(x, prep.Y, len(tested), seed, MSCRConfig(n_perm=199), run=run)
    pos = {i: c for c, i in enumerate(tested)}
    for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
        assert e.p == ref.pvals[j, pos[i]]
        assert e.score == ref.s_star[j, pos[i]]


def test_rcot2_fixed_b_block_perm_equals_explicit_loop():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(150, 3))
    x = z[:, 0] + rng.normal(size=150)
    cfg = dataclasses.replace(frozen_config_v2(), block_perm_reps=49)
    zs = (z - z.mean(0)) / z.std(0)
    blocks = z_blocks(zs, block=cfg.block_size)

    def stat(cand, target):
        fx, fy, rx, ry = rff_residuals(cand, target, z, cfg, np.random.default_rng(123))
        return rcot_statistic(rx, ry)[0], residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon)

    for y in (0.5 * x + z[:, 1] + rng.normal(size=150), rng.normal(size=150)):
        s0, _ = stat(x, y)
        perm_rng = np.random.default_rng(123 * 7 + 99991)
        cnt = 0
        for _ in range(cfg.block_perm_reps):
            perm = np.arange(150)
            for blk in blocks:
                perm[blk] = blk[perm_rng.permutation(blk.size)]
            cnt += stat(x[perm], y)[0] >= s0
        got = block_perm_sequential(x, y, z, cfg, 123, h=None)
        assert got == (s0, (1 + cnt) / (cfg.block_perm_reps + 1), False, 49)


@pytest.mark.parametrize("cls", ARMS)
def test_result_contract(cls, make_ci):
    ds, _ = make_ci("planted", 400, DEV0)
    if cls is RCoT2Method:
        ds = dataclasses.replace(
            ds, candidates=tuple(c for c in ds.candidates if c[0] in ("P0", PLACEBO, "K3"))
        )
    m = cls()
    r = m.run(ds, FAST[m.name])
    assert isinstance(r, Result)
    assert r.method == m.name
    assert r.cpu_s >= 0
    assert [(e.source, e.target) for e in r.edges] == list(ds.candidates)
    assert all(e.sign in (-1, 0, 1) for e in r.edges)
    assert all(e.p is None or 0 <= e.p <= 1 for e in r.edges)
    assert r.notes["sign_rule"] == ("native" if m.name == "pcorr" else "pcorr_given_Z")
    assert len(r.notes["declared_native"]) == len(r.edges)
    assert len(r.notes["family_of_edge"]) == len(r.edges)
    r2 = m.run(ds, FAST[m.name])
    assert [(e.score, e.p) for e in r.edges] == [(e.score, e.p) for e in r2.edges]


def test_pcorr_finds_linear_planted_edges_with_signs(make_ci):
    ds, _ = make_ci("planted", 2000, DEV0)
    r = PCorrMethod().run(ds, {})
    dec = {(e.source, e.target): e.sign for e in r.edges if e.declared}
    assert dec.get(("P0", "K0")) == 1
    assert dec.get(("P1", "K1")) == -1
    assert dec.get(("K3", "K3")) == 1
    assert not any(s == PLACEBO for s, _ in dec)


def test_mscr_finds_gated_edge(make_ci):
    ds, _ = make_ci("planted", 2000, DEV0)
    cands = tuple(c for c in ds.candidates if c[1] == "K2" and c[0].startswith("P"))
    r = MSCRMethod().run(
        dataclasses.replace(ds, candidates=cands), {"n_perm": 999, "bc_h": 20, "n_jobs": 1}
    )
    dec = {e.source for e in r.edges if e.declared}
    assert {"P2", "P3"} <= dec
    assert PLACEBO not in dec


@pytest.mark.claim("PRO-TUNE")
def test_tune_returns_tau_from_placebo_only(make_ci):
    devs = [make_ci("null", 300, DEV0 + s)[0] for s in range(3)]
    m = PCorrMethod()
    cfg = m.tune(devs)
    sc = sorted(
        (e.score for d in devs for e in m.run(d, {}).edges if e.source == PLACEBO), reverse=True
    )
    assert cfg["tau"] == sc[0]
    assert cfg["n_placebo_scores"] == 3 * len(devs[0].kpi_names)
    r = m.run(devs[0], cfg)
    assert len(r.notes["declared_tau"]) == len(r.edges)
    with pytest.raises(ValueError, match="one \\(world, regime, n\\) cell"):
        m.tune([devs[0], make_ci("null", 400, DEV0)[0]])
    with pytest.raises(ValueError, match="truth-free"):
        m.tune(devs, truth_free=False)


def test_placebo_conf_is_own_family_and_not_a_conditioner(make_ci):
    ds, _ = make_ci("planted", 400, DEV0)
    conf = np.random.default_rng(9).uniform(-1, 1, size=(ds.n, 1))
    ds2 = dataclasses.replace(
        ds,
        action_names=(*ds.action_names, "P_placebo_conf"),
        X_action=np.c_[ds.X_action, conf],
        designs=(*ds.designs, ds.designs[0]),
        candidates=ds.candidates + tuple(("P_placebo_conf", k) for k in ds.kpi_names),
    )
    prep = prepare(ds2)
    assert prep.family[-1] == "placebo_conf"
    assert prep.exclude_z == (len(ds.action_names),)
    for m, cfg in ((PCorrMethod(), {}), (MSCRMethod(), {"n_perm": 199, "n_jobs": 1})):
        r1, r2 = m.run(ds, cfg), m.run(ds2, cfg)
        rows1 = [(e.score, e.p, e.sign, e.declared) for e in r1.edges]
        assert rows1 == [(e.score, e.p, e.sign, e.declared) for e in r2.edges[: len(r1.edges)]]


def test_rcot2_lpd4_null_and_frozen_hbe_match_momentchi2():
    from momentchi2 import sw

    from oran_causal.methods.controlled._rcot import _hbe_pvalue
    from oran_causal.methods.controlled.rcot2 import _momentchi2, rcot_pvalue_lpd4

    _, lpb4 = _momentchi2()
    rng = np.random.default_rng(11)
    prodc = rng.normal(size=(400, 25)) @ rng.normal(size=(25, 25)) * 0.1
    prodc -= prodc.mean(0)
    w = np.linalg.eigvalsh(prodc.T @ prodc / 400)
    w = w[w > 1e-12]
    for stat in (0.5, w.sum(), 3 * w.sum()):
        assert abs(_hbe_pvalue(stat, prodc) - (1 - sw(w, stat))) < 1e-12
        assert 0 <= 1 - lpb4(w, stat) <= 1
    z = rng.normal(size=(300, 3))
    x = z[:, 0] + rng.normal(size=300)
    cfg = dataclasses.replace(frozen_config_v2(), dz=100)
    _, p, g, how = rcot_pvalue_lpd4(x, 0.8 * x + rng.normal(size=300), z, cfg, 5)
    assert not g
    assert how in ("lpb4", "hbe_fallback")
    assert p < 1e-3


@pytest.mark.claim("DEF-ENV-ARMS")
def test_arm_defaults(make_ci):
    for cls in ARMS:
        assert cls().native_config()["arm"] == "native"
    ds, _ = make_ci("planted", 200, DEV0)
    assert PCorrMethod().run(ds, {}).notes["arm"] == "eq"
    with pytest.raises(ValueError, match="arm must be"):
        prepare(ds, "bogus")


def _check_native_z(ds, nat, i, nm_i):
    cn, _, _ = design_covariates(ds, dataclasses.replace(R3, focal=nm_i))
    want = {_canon(c) for c in cn} - ({"P_placebo_conf"} if nm_i != "P_placebo_conf" else set())
    assert {nat.S_names[c] for c in nat.z_cols(i)} == want


def _check_eq_z(ds, eq, i, nm_i):
    cn, _, _ = design_covariates(ds, CovariateSpec(focal=nm_i))
    want = {_canon(c) for c in cn}
    if nm_i != "P_placebo_conf":
        want = {c for c in want if "P_placebo_conf" not in c}
        conf = {c for c, nm in enumerate(eq.S_names) if "P_placebo_conf" in nm}
        assert not conf & set(eq.z_cols(i))
    assert {eq.S_names[c] for c in eq.z_cols(i)} == want


@pytest.mark.claim("DEF-ENV-ARMS")
@pytest.mark.parametrize(
    ("world", "regime"), [("E2", "R2"), ("E4", "R3"), ("E4", "R4"), ("E5", "R1")]
)
def test_eq_arm_layout_on_controlled_worlds(world, regime):
    ds, _ = generate_dataset(world, regime, 300, DEV0 + 2)
    nat, eq = prepare(ds, "native"), prepare(ds, "eq")
    _, _, mask = design_covariates(ds)
    assert eq.S.shape[0] == mask.sum() == eq.Y.shape[0]
    assert eq.S_names[: nat.S.shape[1]] == nat.S_names
    np.testing.assert_array_equal(eq.S[:, : nat.S.shape[1]], nat.S[mask])
    assert eq.pairs == nat.pairs
    for i in sorted({i for i, _ in nat.pairs}):
        nm_i = nat.S_names[i]
        if nm_i in ds.action_names:
            _check_native_z(ds, nat, i, nm_i)
            _check_eq_z(ds, eq, i, nm_i)


def test_eq_arm_pcorr_matches_ols():
    ds, _ = generate_dataset("E2", "R2", 400, DEV0 + 3)
    ds = dataclasses.replace(
        ds, candidates=tuple(c for c in ds.candidates if c[0] in ("P0", "P_placebo", "K1"))
    )
    r = PCorrMethod().run(ds, {"arm": "eq"})
    prep = prepare(ds, "eq")
    for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
        p, sign = _ols_p(prep, i, j)
        assert e.p is not None
        assert abs(e.p - p) <= 1e-8 * max(p, 1e-300) + 1e-14
        assert e.sign == sign
    assert r.notes["n_rows_dropped"] == 2
    assert r.notes["covariates_helper"] == "oran_causal.core.covariates"


def test_deterministic_world_not_testable():
    ds, _ = generate_dataset("E1", "R1", 300, DEV0 + 4)
    for arm in ("eq", "native"):
        r = PCorrMethod().run(ds, {"arm": arm})
        assert set(r.notes["not_testable"].values()) == {"exact fit: deterministic world under Z"}
        assert len(r.notes["not_testable_edges"]) == len(ds.candidates)
        assert all(
            e.p is None and np.isnan(e.score) and not e.declared and e.sign == 0 for e in r.edges
        )
    r2 = PCorrMethod().run(generate_dataset("E2", "R1", 300, DEV0 + 4)[0], {})
    assert "not_testable" not in r2.notes
    assert r2.notes["exact_fit_rr_min_testable"] > 1e-3


def test_partial_degenerate_alignment(make_ci):
    ds, _ = make_ci("planted", 150, DEV0)
    y = ds.Y.copy()
    y[:, 0] = 0.7 * ds.X_action[:, 0] - 0.2 * ds.X_kpi_lag[:, 1]
    cands = (("P0", "K0"), ("P0", "K1"), ("P_placebo", "K0"), ("K3", "K3"))
    ds = dataclasses.replace(ds, Y=y, candidates=cands)
    m = RCoT2Method()
    r = m.run(ds, {**m.block_perm_config(), "block_perm_reps": 99, "arm": "native"})
    dead = [e.target == "K0" for e in r.edges]
    assert [e.p is None for e in r.edges] == dead
    assert [list(x) for x in r.notes["not_testable_edges"]] == [["P0", "K0"], ["P_placebo", "K0"]]
    assert len(r.notes["mc_draws"]) == len(r.edges)
    assert [v is None for v in r.notes["mc_draws"]] == dead
    assert len(r.notes["declared_native"]) == len(r.edges)
    assert r.edges[3].p is not None
    assert r.edges[3].p < 0.05


@pytest.mark.parametrize(
    ("world", "regime"), [("E2", "R2"), ("E4", "R3"), ("E4", "R4"), ("E5", "R1")]
)
@pytest.mark.parametrize("arm", ["eq", "eq_min"])
def test_eq_arm_equals_classic_cond_set(world, regime, arm):
    ds, _ = generate_dataset(world, regime, 300, DEV0 + 9, kappa=0.25)
    prep = prepare(ds, arm)
    for i in sorted({i for i, _ in prep.pairs}):
        nm = prep.S_names[i]
        if nm in ds.action_names:
            cs = cond_set(ds, "action", ds.action_names.index(nm), arm)
        else:
            cs = cond_set(ds, "kpi", ds.kpi_names.index(nm), arm)
        got = {prep.S_names[c] for c in prep.z_cols(i)}
        assert got == {_canon(c) for c in cs.names} - set(prep.eq_dropped), (world, regime, arm, nm)


@pytest.mark.parametrize(
    ("world", "regime"), [("E2", "R2"), ("E2", "R1"), ("E4", "R3"), ("E5", "R2")]
)
def test_eq_min_layout(world, regime):
    ds, _ = generate_dataset(world, regime, 300, DEV0 + 11, kappa=0.25)
    nat, em = prepare(ds, "native"), prepare(ds, "eq_min")
    assert em.S.shape[0] == ds.n
    assert em.pairs == nat.pairs
    assert em.arm == "eq_min"
    for i in sorted({i for i, _ in nat.pairs}):
        nm = nat.S_names[i]
        want = {nat.S_names[c] for c in nat.z_cols(i)}
        if nm in ds.action_names and ds.designs[ds.action_names.index(nm)].kind == "dither":
            want |= {f"sp:{nm}"}
        assert {em.S_names[c] for c in em.z_cols(i)} == want, (world, regime, nm)
        if nm != "P_placebo_conf":
            assert not any("P_placebo_conf" in em.S_names[c] for c in em.z_cols(i))


@pytest.mark.parametrize("cls", ARMS)
def test_eq_min_runs(cls):
    ds, _ = generate_dataset("E2", "R2", 200, DEV0 + 12, kappa=0.25)
    ds = dataclasses.replace(
        ds, candidates=tuple(c for c in ds.candidates if c[0] in ("P0", PLACEBO, "K1"))
    )
    r = cls().run(ds, {**FAST[cls.name], "arm": "eq_min"})
    assert r.notes["arm"] == "eq_min"
    assert [(e.source, e.target) for e in r.edges] == list(ds.candidates)
    assert all(e.p is None or 0 <= e.p <= 1 for e in r.edges)
    if cls is PCorrMethod:
        prep = prepare(ds, "eq_min")
        for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
            assert e.p is not None
            assert abs(e.p - _ols_p(prep, i, j)[0]) < 1e-9


@pytest.mark.claim("PRO-TUNE")
def test_tau_is_the_shared_placebo_tau(make_ci):
    devs = [make_ci("null", 200, DEV0 + s)[0] for s in range(2)]
    res = [PCorrMethod().run(d, {"arm": "native"}) for d in devs]
    sc = [e.score for r in res for e in r.edges if e.source == PLACEBO]
    out = tau_from_results(res)
    assert out["tau"] == placebo_tau(res, alpha=0.05) == _tau(sc)
    assert out["n_placebo_scores"] == len(sc)
    assert out["tau_rule"] == tau_rule_name(0.05)
    assert _tau(np.arange(1.0, 20.0)) == 19.0


class _Stub(CITestBase):
    name, version, method_key = "stub", "0", 9

    def __init__(self, kind: str):
        self.kind = kind

    def _test(self, prep, data, cfg):
        m = len(prep.pairs)
        if self.kind == "stop":
            raise ValueError("rank-deficient source design")
        p = np.full(m, 1e-6)
        p[0] = np.nan
        return {"score": np.ones(m), "p": p, "not_testable": {0: "guard: test"}}


def test_guard_and_stop_listed_as_not_testable(make_ci):
    ds, _ = make_ci("planted", 200, DEV0)
    r = _Stub("guard").run(ds, {"arm": "native"})
    first = "{}->{}".format(*ds.candidates[0])
    assert r.notes["not_testable"] == {first: "guard: test"}
    assert r.notes["not_testable_edges"] == [list(ds.candidates[0])]
    e0 = r.edges[0]
    assert e0.p is None
    assert np.isnan(e0.score)
    assert e0.sign == 0
    assert not e0.declared
    fam = r.notes["family_of_edge"]
    assert all(e.declared for k, e in enumerate(r.edges[1:], 1) if fam[k] == "action_kpi")
    r = _Stub("stop").run(ds, {"arm": "native"})
    assert len(r.notes["not_testable"]) == len(ds.candidates)
    assert r.notes["stop"].startswith("rank")
    assert all(v.startswith("STOP: rank") for v in r.notes["not_testable"].values())
    assert all(e.p is None and not e.declared for e in r.edges)


def test_mscr_bank_stream_per_column_group(monkeypatch):
    groups: list[int] = []
    real = mscr._streams

    def spy(seed, targets, n_perm, group_index):
        groups.append(group_index)
        return real(seed, targets, n_perm, group_index)

    monkeypatch.setattr(mscr, "_streams", spy)
    ds, _ = generate_dataset("E4", "R3", 300, DEV0 + 13, kappa=0.25)
    MSCRMethod().run(ds, {**FAST["mscr"], "arm": "eq"})
    assert 0 in groups
    assert 1 in groups
    s0 = real(5, [0], 299, 0)[0].random(3)
    assert np.array_equal(s0, np.random.default_rng([5, 0, 299]).random(3))
    s1 = real(5, [0], 299, 1)[0].random(3)
    assert np.array_equal(s1, np.random.default_rng([5, 0, 299, 1]).random(3))

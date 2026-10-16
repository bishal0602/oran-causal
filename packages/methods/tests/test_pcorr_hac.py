import dataclasses
import itertools

import numpy as np
import pytest
from scipy import stats

from oran_causal.core.covariates import info_order
from oran_causal.core.scoring import score
from oran_causal.methods.controlled import _fixedb as fb
from oran_causal.methods.controlled import pcorr_hac as hac
from oran_causal.methods.controlled._classic import columns, cond_set, pcorr, resolve, source_column
from oran_causal.sim.controlled.datasets.generate import generate_dataset

SEED = 3_000_000


def _statsmodels(y, x, z, lags):
    import statsmodels.api as sm  # noqa: PLC0415

    design = np.column_stack([hac.hac_design(z), x])
    kw = {"maxlags": lags, "use_correction": True}
    f = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds=kw, use_t=True)
    return f.params[-1], f.bse[-1], f.pvalues[-1], f.df_resid


def _shuffled(ds, seed):
    perm = np.random.default_rng(seed).permutation(ds.n)
    des = tuple(
        dataclasses.replace(d, random_part=d.random_part[perm], fixed_part=d.fixed_part[perm])
        for d in ds.designs
    )
    return dataclasses.replace(
        ds, X_action=ds.X_action[perm], X_kpi_lag=ds.X_kpi_lag[perm], Y=ds.Y[perm],
        designs=des, time_index=ds.time_index[perm],
    )  # fmt: skip


@pytest.mark.parametrize("rho", [0.0, 0.6])
@pytest.mark.parametrize("lags", [None, 0, 4])
def test_hac_equals_statsmodels_synthetic(rho, lags):
    r = np.random.default_rng(5)
    n = 700
    z = r.normal(size=(n, 4))
    z = np.column_stack([z, z[:, 0] + z[:, 1], np.full(n, 3.0)])
    e = np.zeros(n)
    w = r.normal(size=n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + w[t]
    x = z[:, 2] + e
    y = 0.3 * z[:, 0] + 0.1 * x + r.normal(size=n)
    out = hac.hac_test(y, x, z, lags)
    assert out["status"] == hac.OK
    assert out["k"] == 6
    g, se, p, df = _statsmodels(y, x, z, out["maxlags"])
    assert out["g"] == pytest.approx(g, rel=1e-10)
    assert out["se"] == pytest.approx(se, rel=1e-10)
    assert out["p"] == pytest.approx(p, rel=1e-8, abs=1e-14)
    assert out["df"] == df


@pytest.mark.parametrize(
    ("world", "regime", "arm"),
    [("E2", "R2", "native"), ("E2", "R2", "eq_min"), ("E4", "R3", "native"), ("E5", "R1", "eq_min"),
     ("E3", "R2", "eq_min")],
)  # fmt: skip
def test_arm_equals_statsmodels_on_world(world, regime, arm):
    ds, _ = generate_dataset(world, regime, 600, SEED, lam=1.0, kappa=0.25)
    res = hac.PcorrHac().run(ds, {"arm": arm})
    cols, order = columns(ds), info_order(ds)
    by = {(e.source, e.target): e for e in res.edges}
    for s, t in ds.candidates[::3]:
        fam, j = resolve(ds, cols, s)
        assert j is not None
        x, src = source_column(ds, cols, fam, j)
        cs = cond_set(ds, fam, src, arm)
        lags = res.notes["maxlags"][f"{s}->{t}"]
        g, _, p, _ = _statsmodels(cols.Y[order, ds.kpi_names.index(t)], x[order], cs.Z[order], lags)
        assert by[(s, t)].p == pytest.approx(p, rel=1e-7, abs=1e-14), (s, t)
        assert by[(s, t)].sign == int(np.sign(g))
        assert by[(s, t)].sign == int(np.sign(pcorr(x, cols.Y[:, ds.kpi_names.index(t)], cs.Z)))


def test_andrews_bandwidth():
    v = np.zeros(4000)
    r = np.random.default_rng(0)
    for t in range(1, len(v)):
        v[t] = 0.5 * v[t - 1] + r.normal()
    lags, rho, s_t = hac.andrews_lags(v)
    a = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    assert rho == pytest.approx(0.5, abs=0.05)
    assert s_t == pytest.approx(1.1447 * (a * 4000) ** (1 / 3))
    assert lags == int(np.floor(s_t))
    assert hac.andrews_lags(np.zeros(50))[0] == 0
    assert hac.andrews_lags(np.arange(30.0))[0] <= 28


def test_time_order_not_row_order():
    ds, _ = generate_dataset("E2", "R2", 500, SEED, kappa=0.25)
    a = hac.PcorrHac().run(ds, {"arm": "eq_min"})
    b = hac.PcorrHac().run(_shuffled(ds, 3), {"arm": "eq_min"})
    assert [e.p for e in a.edges] == pytest.approx([e.p for e in b.edges], rel=1e-9)


def test_contract_families_and_determinism():
    ds, tr = generate_dataset("E4", "R4", 400, SEED, lam=1.5, kappa=0.25)
    m = hac.PcorrHac()
    r = m.run(ds)
    assert r.config["arm"] == "native"
    assert r.notes["arm"] == "native"
    assert [(e.source, e.target) for e in r.edges] == list(ds.candidates)
    assert r.notes["family"]["P_placebo_conf->K0"] == "diagnostic"
    assert r.notes["family"]["P0->K0"] == "action"
    assert r.notes["family"]["K0->K0"] == "kpi"
    assert "per family" in r.notes["declare_rule"]
    assert all(e.p is None or 0 <= e.p <= 1 for e in r.edges)
    r2 = m.run(ds)
    assert [(e.p, e.sign, e.declared) for e in r.edges] == [
        (e.p, e.sign, e.declared) for e in r2.edges
    ]
    score(r, tr)
    with pytest.raises(ValueError, match="pcorr_hac arm"):
        m.run(ds, {"arm": "eq"})


def test_not_testable_exact_fit_kappa0():
    ds, tr = generate_dataset("E1", "R1", 300, SEED, kappa=0.0)
    r = hac.PcorrHac().run(ds)
    nt = r.notes["not_testable_edges"]
    assert nt
    assert r.notes["not_testable_reason"].startswith("exact fit")
    assert "not_testable" not in r.notes
    by = {f"{e.source}->{e.target}": e for e in r.edges}
    assert all(np.isnan(by[k].score) and by[k].p is None and not by[k].declared for k in nt)
    assert "P0->K0" in nt
    sc = score(r, tr)
    assert sc["n_not_testable_overridden"] == 0
    assert sc["n_not_testable_true"] >= 1


def test_collinear_source_not_testable(make_classic):
    data, _ = make_classic(300, SEED)
    x = data.X_action.copy()
    x[:, 2] = 0.5
    r = hac.PcorrHac().run(dataclasses.replace(data, X_action=x))
    assert set(r.notes["not_testable_collinear"]) == {f"A2->{k}" for k in data.kpi_names}


def test_planted_edges_found_and_global_null(make_classic):
    data, truth = make_classic(3000, SEED + 1, b=1.0)
    r = hac.PcorrHac().run(data)
    dec = {(e.source, e.target) for e in r.edges if e.declared}
    assert set(truth.edges) <= dec
    for (s, t), sign in truth.signs.items():
        assert next(e for e in r.edges if (e.source, e.target) == (s, t)).sign == sign
    null, _ = make_classic(1000, SEED + 2, b=0.0)
    assert not any(e.declared for e in hac.PcorrHac().run(null).edges)


def test_eq_min_arm_uses_focal_setpoint():
    ds, _ = generate_dataset("E2", "R2", 400, SEED, kappa=0.25)
    nat = hac.PcorrHac().run(ds, {"arm": "native"})
    eqm = hac.PcorrHac().run(ds, {"arm": "eq_min"})
    assert eqm.notes["z_size"]["P0->K0"] == nat.notes["z_size"]["P0->K0"] + 1
    assert eqm.notes["z_size"]["K0->K1"] == nat.notes["z_size"]["K0->K1"]
    kk = [i for i, (s, _) in enumerate(ds.candidates) if s in ds.kpi_names]
    assert [nat.edges[i].p for i in kk] == [eqm.edges[i].p for i in kk]
    assert eqm.config["arm"] == "eq_min"
    assert "eq_min" in eqm.notes["label"]


@pytest.mark.parametrize("b", [0.02, 0.1, 0.5, 1.0])
def test_fixedb_reproduces_kv2005_table1(b):
    for pc, *_ in fb.KV_TABLE1_BARTLETT:
        assert fb.fixedb_p(fb.kv_cv(b, pc), b) == pytest.approx(2 * (1 - pc), rel=0.10), (b, pc)


def test_fixedb_shape():
    assert fb.fixedb_p(1.96, 0.0) == pytest.approx(2 * stats.norm.sf(1.96))
    ps = [fb.fixedb_p(t, 0.1) for t in (0.5, 1.0, 2.0, 3.0, 5.0)]
    assert all(a > c for a, c in itertools.pairwise(ps))
    assert ps[-1] > 0
    assert fb.fixedb_p(-2.0, 0.1) == fb.fixedb_p(2.0, 0.1)
    assert fb.fixedb_p(2.0, 0.3) > fb.fixedb_p(2.0, 0.1) > fb.fixedb_p(2.0, 0.0)
    eps = 1e-9
    assert fb.fixedb_p(2.0, fb.B0 - eps) == pytest.approx(fb.fixedb_p(2.0, fb.B0 + eps), abs=1e-6)
    mix = 0.6 * fb.fixedb_p(2.0, 0.123) + 0.4 * fb.fixedb_p(2.0, 0.124)
    assert fb.fixedb_p(2.0, 0.1234) == pytest.approx(mix, rel=1e-9)
    assert fb.fixedb_p(float("inf"), 0.2) == 0.0


def test_arm_fixed_b_variant():
    ds, _ = generate_dataset("E2", "R2", 500, SEED, kappa=0.25)
    m = hac.PcorrHac()
    rt, rb = m.run(ds, {"arm": "eq_min"}), m.run(ds, {"arm": "eq_min", "inference": "fixed_b"})
    assert rt.config["inference"] == "t"
    assert rb.config["inference"] == "fixed_b"
    assert m.version == "1.1"
    assert "fixed-b" in rb.notes["hac"]["reference"]
    assert rt.notes["t_df_b"] == rb.notes["t_df_b"]
    for et, eb in zip(rt.edges, rb.edges, strict=True):
        if et.p is None:
            assert eb.p is None
            continue
        t, df, b = rt.notes["t_df_b"][f"{et.source}->{et.target}"]
        lags = rt.notes["maxlags"][f"{et.source}->{et.target}"]
        assert b == pytest.approx((lags + 1) / ds.n)
        assert et.sign == eb.sign
        assert et.p == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-9)
        assert eb.p == pytest.approx(fb.fixedb_p(t, b), rel=1e-12)
    with pytest.raises(ValueError, match="inference must be"):
        m.run(ds, {"inference": "normal"})

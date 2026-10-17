import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from oran_causal.core import covariates as cov
from oran_causal.core.data import Dataset
from oran_causal.core.pmrt.candidates import concurrent_columns
from oran_causal.core.pmrt.rows import PmrtCore

CFG = {"B": 199, "seq_h": None}
MakeData = Callable[..., Dataset]


def perturb(ds: Dataset, seed: int = 7, time_index: bool = True) -> Dataset:
    rng = np.random.default_rng(seed)
    n = ds.n
    t = np.arange(n) + np.cumsum(rng.random(n) < 0.03) * 3
    perm = rng.permutation(n)
    xk = ds.X_kpi_lag.copy()
    xk[rng.random(xk.shape) < 0.02] = np.nan

    def shuffled(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else np.asarray(a)[perm]

    des = tuple(
        dataclasses.replace(
            d,
            random_part=shuffled(d.random_part),
            fixed_part=shuffled(d.fixed_part),
            propensity=shuffled(d.propensity),
        )
        for d in ds.designs
    )
    return dataclasses.replace(
        ds,
        X_action=ds.X_action[perm],
        X_kpi_lag=xk[perm],
        Y=ds.Y[perm],
        designs=des,
        time_index=t[perm] if time_index else None,
        context=shuffled(ds.context),
    )


def test_names_blocks_and_encoding(make_rows: MakeData) -> None:
    ds = make_rows(300, regime="R2")
    names, m, mask = cov.design_covariates(ds)
    acts = ds.action_names
    lagged = [f"{a}@t-{lag}" for lag in (1, 2) for a in acts]
    assert names == (*(f"lag_kpi:{k}" for k in ds.kpi_names), *(f"sp:{a}" for a in acts), *lagged)
    assert ds.designs[1].fixed_part is not None
    assert np.array_equal(m[:, names.index("sp:P1")], ds.designs[1].fixed_part)
    j = names.index("P0@t-1")
    assert np.array_equal(m[1:, j], ds.X_action[:-1, 0])
    assert m[0, j] == 0
    assert np.array_equal(mask, np.arange(300) >= 2)
    p = perturb(ds)
    npn, mp, maskp = cov.design_covariates(p)
    assert "gap@t-1" in npn
    assert "gap@t-2" in npn
    assert any(x.endswith(":missing") for x in npn)
    order = cov.info_order(p)
    ts = np.asarray(p.time_index)[order]
    ok1 = np.r_[False, np.diff(ts) == 1]
    ok2 = np.r_[False, False, (ts[2:] - ts[:-2]) == 2]
    assert np.array_equal(maskp[order], ok1 & ok2)
    assert np.array_equal(mp[order][:, npn.index("gap@t-1")], (~ok1).astype(float))
    assert cov.design_covariates(p, cov.CovariateSpec(lagged_actions=False))[2].all()


def test_r3_set_and_context_names(make_logged: MakeData, make_rows: MakeData) -> None:
    ds = make_logged(200)
    names, m, mask = cov.design_covariates(ds, cov.R3)
    assert names == ("lag_kpi:K0", "lag_kpi:K0:missing", "ctx:Z")
    assert mask.all()
    assert ds.context is not None
    assert np.array_equal(m[:, 2], ds.context[:, 0])
    nameless = dataclasses.replace(ds, meta={})
    assert cov.design_covariates(nameless, cov.R3)[0][2] == "ctx:c0"
    d2 = make_rows(200, regime="R2")
    no_sp = cov.design_covariates(d2, cov.CovariateSpec(setpoints="none"))[0]
    assert not any(x.startswith("sp:") for x in no_sp)
    assert any(x.endswith("@t-2") for x in no_sp)
    only_sp = cov.design_covariates(d2, cov.CovariateSpec(kpi_lag=False, lagged_actions=False))[0]
    assert only_sp == tuple(f"sp:{a}" for a in d2.action_names)


def test_concurrent_rules(make_logged: MakeData, make_rows: MakeData) -> None:
    ds = make_logged(200)
    assert cov.concurrent_indices(ds, 2) == []
    assert cov.concurrent_indices(ds, 0) == [1]
    assert cov.concurrent_indices(ds, 2, "all") == [0, 1]
    names, m, _ = cov.design_covariates(ds, cov.CovariateSpec(focal="P0"))
    assert names[-1] == "concurrent:P_placebo"
    assert np.array_equal(m[:, -1], ds.X_action[:, 1])
    assert cov.design_covariates(ds, cov.CovariateSpec(focal=0))[0] == names
    d = make_rows(200)
    for ai in range(3):
        assert (
            cov.concurrent_indices(d, ai)
            == concurrent_columns(d, ai)
            == [j for j in range(3) if j != ai]
        )
    with pytest.raises(ValueError, match="concurrent"):
        cov.concurrent_indices(d, 0, "some")


def test_eq_min_set(make_rows: MakeData) -> None:
    ds = make_rows(200, regime="R2")
    names, _, _ = cov.eq_min_covariates(ds, "P0")
    assert names == ("lag_kpi:K0", "lag_kpi:K1", "sp:P0", "concurrent:P1", "concurrent:P_placebo")


def test_pmrt_r3_option(make_rows: MakeData) -> None:
    ds = make_rows(400, regime="R2")
    eq = PmrtCore().run(ds, CFG)
    r3 = PmrtCore().run(ds, {**CFG, "covariates": "r3"})
    assert eq.notes["covariates"]["set"] == "eq"
    assert r3.notes["covariates"]["set"] == "r3"
    base = r3.notes["covariates"]["base"]
    assert base == [f"lag_kpi:{k}" for k in ds.kpi_names]
    assert any(x.startswith("sp:") for x in eq.notes["covariates"]["base"])
    adj_eq, adj_r3 = eq.notes["adjust"]["P0"], r3.notes["adjust"]["P0"]
    assert adj_r3["d"] == len(base) + len(adj_r3["concurrent"]) < adj_eq["d"]
    assert eq.notes["z"] != r3.notes["z"]
    with pytest.raises(ValueError, match="covariates"):
        PmrtCore().run(ds, {**CFG, "covariates": "full"})

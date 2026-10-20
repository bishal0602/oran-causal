import dataclasses

import numpy as np
import pytest

from oran_causal.sim.controlled.datasets.generate import generate_dataset

torch = pytest.importorskip("torch")

from oran_causal.methods.controlled._cdl_model import CDL, CdlSettings  # noqa: E402
from oran_causal.methods.controlled.cdl import CDLMethod, gradient_steps  # noqa: E402

FAST = {"max_gradient_steps": 200, "feature_fc_dims": [8, 8], "generative_fc_dims": [8, 8]}


def _model(informed: bool) -> CDL:
    torch.manual_seed(0)
    settings = CdlSettings(
        4, 1, 2, (8, 8), (8, 8), lr=1e-3, eval_tau=0.9, eval_steps=1, informed_drop=informed
    )
    return CDL(settings, torch.device("cpu"))


def test_port_target_dim_and_uninformed_mask():
    s, y = torch.rand(16, 4), torch.rand(16, 2)
    a0 = torch.zeros(16, 1)
    m = _model(False)
    m.fit_step(s, y, a0)
    m.cmi_step(s, y, a0)
    assert m.mask_CMI.shape == (2, 5)
    losses = []
    for informed in (False, False, True):
        m = _model(informed)
        torch.manual_seed(1)
        losses.append(float(m.fit_step(s, y, a0)))
    assert losses[0] == losses[1]
    assert losses[0] != losses[2]


@pytest.fixture(scope="module")
def syn(make_classic):
    return make_classic(400, 3_000_000, b=1.0)


@pytest.fixture(scope="module")
def res(syn):
    return CDLMethod().run(syn[0], dict(FAST, tau=0.01))


def test_contract(syn, res):
    d, _ = syn
    assert [(e.source, e.target) for e in res.edges] == list(d.candidates)
    assert all(np.isfinite(e.score) and e.p is None and e.sign in (-1, 0, 1) for e in res.edges)
    assert res.notes["sign_rule"] == "pcorr_given_Z"
    assert res.notes["arm"] == "native"
    assert res.notes["gradient_steps"] == 200
    assert res.notes["fit"]["device"] == "cpu"
    assert [e.declared for e in res.edges] == [e.score > 0.01 for e in res.edges]
    cmi = res.notes["fit"]["cmi"]
    assert set(cmi) == set(d.kpi_names)
    assert list(cmi["Y0"])[-1] == "action_node"
    e = next(x for x in res.edges if (x.source, x.target) == ("A1", "Y2"))
    assert e.score == cmi["Y2"]["A1"]
    assert set(res.notes["native_declared"]) == {
        f"{x.source}->{x.target}" for x in res.edges if x.score >= 0.16
    }


def test_gradient_step_rule():
    cfg = CDLMethod().default_config()
    assert [gradient_steps(n, cfg) for n in (500, 1000, 4000, 15754, 24000)] == [
        508,
        1016,
        4063,
        16000,
        16000,
    ]


def test_deterministic(syn):
    cfg = dict(FAST, max_gradient_steps=30, tau=0.01)
    a, b = (CDLMethod().run(syn[0], cfg) for _ in range(2))
    assert [e.score for e in a.edges] == [e.score for e in b.edges]


def test_untuned_declares_nothing(syn):
    r = CDLMethod().run(syn[0], dict(FAST, max_gradient_steps=20))
    assert not any(e.declared for e in r.edges)
    assert "untuned" in r.notes["declare_rule"]


@pytest.mark.claim("PRO-TUNE")
def test_tune_sets_placebo_tau(make_classic):
    dev = [make_classic(300, 3_000_000 + i, b=1.0)[0] for i in range(2)]
    cfg = CDLMethod().tune(dev, config=dict(FAST, max_gradient_steps=20))
    assert np.isfinite(cfg["tau"])
    assert cfg["n_placebo_scores"] == 6
    assert cfg["dev_seeds"] == [3_000_000, 3_000_001]


def _conf_scrambled(ds):
    j = ds.action_names.index("P_placebo_conf")
    x = ds.X_action.copy()
    x[:, j] = np.random.default_rng(11).permutation(x[:, j])[::-1]
    des = list(ds.designs)
    if des[j].propensity is not None:
        des[j] = dataclasses.replace(des[j], propensity=des[j].propensity[::-1].copy())
    return dataclasses.replace(ds, X_action=x, designs=tuple(des))


def _rows(r):
    return repr(
        [(e.source, e.target, e.score, e.sign) for e in r.edges if e.source != "P_placebo_conf"]
    )


@pytest.mark.parametrize("regime", ["R3", "R4"])
def test_placebo_conf_does_not_change_primary(regime):
    ds, _ = generate_dataset("E4", regime, 300, 3_000_001, lam=1.5, kappa=0.25)
    m = CDLMethod()
    cfg = dict(FAST, max_gradient_steps=60)
    a, b = m.run(ds, cfg), m.run(_conf_scrambled(ds), cfg)
    assert _rows(a) == _rows(b)
    assert all("P_placebo_conf" not in c for c in a.notes["fit"]["cmi"]["K0"])
    assert "R-37" in a.notes["diagnostic_fit"]["rule"]
    assert "P_placebo_conf" in a.notes["diagnostic_fit"]["cmi"]["K0"]
    assert all(np.isfinite(e.score) for e in a.edges if e.source == "P_placebo_conf")

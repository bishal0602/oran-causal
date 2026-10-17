import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from oran_causal.core.pmrt import units as pu
from oran_causal.core.pmrt.units import PmrtConfig, PmrtData, fit_pmrt
from oran_causal.core.ridge import ResidualConfig
from oran_causal.methods.system import pmrt as pm
from oran_causal.sim.system.decision import records
from oran_causal.sim.system.decision.synthetic import SyntheticSpec, synthetic_records
from oran_causal.sim.system.decision.vocab import FAMILIES

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::UserWarning")

DATA = Path(__file__).parent / "data"
GOLDEN = "system_pmrt_golden.json.gz"
RTOL = 1e-6
FAST = PmrtConfig(B=499, cv_reps=1, gb=False)
GOLD_CFG = PmrtConfig(B=199, cv_reps=1, gb=False)


def _recs(seed: int, n_ep: int, seed0: int, units: int = 70, effect: float = 40.0) -> list:
    spec = SyntheticSpec(n_ep=n_ep, units_per_ep=units, effect=effect, family="sleep", seed0=seed0)
    return synthetic_records(np.random.default_rng(seed), spec)


def _pd(tmp: Path, recs: list, name: str) -> PmrtData:
    f = tmp / f"{name}.npz"
    pm.cache_records(recs, f)
    return pm.pmrt_data(pm.load_pmrt_pool([str(f)]))


@pytest.fixture(scope="module")
def train_test(tmp_path_factory: pytest.TempPathFactory) -> dict:
    d = tmp_path_factory.mktemp("pmrt")
    te_recs = _recs(12, 12, 6000)
    tr = _pd(d, _recs(11, 16, 5000), "tr")
    params = fit_pmrt(tr, pm.LEVEL_V2_ARR, FAST)
    return {
        "te": _pd(d, te_recs, "te"),
        "nu": _pd(d, _recs(13, 12, 7000, effect=0.0), "nu"),
        "params": params,
        "recs_te": te_recs,
        "dir": d,
    }


def _fl(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(x, float).ravel()]


def _fit_summary(params: dict) -> dict:
    out = {}
    for h, hp in params["hyp"].items():
        c = hp["core"]
        out[h] = {
            "best": hp["best"],
            "z_plain": float(c["z_plain"]),
            "lam": float(c["lam"]),
            "lw_shrink": float(c["lw_shrink"]),
            "z_in": {k: float(v) for k, v in hp["z_in"].items()},
            "cv": {k: float(v) for k, v in hp["cv"].items()},
            "mu_full": _fl(c["mu_full"]),
            "B": _fl(c["B"]),
            "prior": {nm: [float(p[1]), float(p[-1])] for nm, p in c["prior"].items()},
            "sdir": {k: float(v) for k, v in c["sdir"].items()},
        }
    return out


def _feat_summary(pd: PmrtData) -> dict:
    return {
        "n": int(pd.n),
        "hist": _fl(pd.hist),
        "post": _fl(np.nansum(pd.post, axis=(1, 2))),
        "pre": _fl(np.nansum(pd.pre, axis=(1, 2))),
        "post_nan": int(np.isnan(pd.post).sum()),
        "ctx_keys": list(pd.ctx_keys),
        "tfrac": _fl(pd.tfrac),
    }


def _run_summary(run: dict) -> list:
    out = []
    for r in run["results"]:
        o = {k: r[k] for k in ("family", "relation", "kpi", "status")}
        if r["status"] == "tested":
            o["arms"] = {
                a: [float(v["p"]), float(v["z"]), int(v["sign"])] for a, v in r["arms"].items()
            }
        out.append(o)
    return out


def _integ_summary(run: dict) -> dict:
    out = {}
    for h, o in run.items():
        if o.get("status") != "tested":
            out[h] = o.get("reason")
            continue
        out[h] = {
            s: [float(o[s][k]) for k in ("p2", "p_plus", "p_minus", "z")] + [int(o[s]["sign"])]
            for s in pm.STATS
        }
    return out


def _golden_summary(tmp: Path) -> dict:
    tr = pm.pmrt_data_from_records(_recs(11, 10, 5000, units=60), tmp)
    te = pm.pmrt_data_from_records(_recs(12, 8, 6000, units=60), tmp)
    params = fit_pmrt(tr, pm.LEVEL_V2_ARR, GOLD_CFG)
    run = pm.run_pmrt(te, params, GOLD_CFG)
    edges = pm.edges_pmrt(run, "best")
    ir_tr = pm.run_integrated(tr, params, GOLD_CFG, 0)
    ir_te = pm.run_integrated(te, params, GOLD_CFG, 0)
    priors = {s: pm.prior_from_run(ir_tr, s, n_prior=10) for s in pm.STATS}
    dec = pm.declare_all(ir_te, priors, 8)
    return {
        "fit": _fit_summary(params),
        "rho": params["rho"],
        "feat_te": _feat_summary(te),
        "run": _run_summary(run),
        "edges_best": {
            "|".join(h): [bool(v["declared"]), int(v["sign"])] for h, v in edges.items()
        },
        "integrated": _integ_summary(ir_te),
        "declare": {
            pm.cname(s, lay): sorted(
                "|".join(h) + f":{v['sign']}" for h, v in dv.items() if v["declared"]
            )
            for (s, lay), dv in dec.items()
        },
        "used_p": {
            pm.cname(s, lay): [float(p) for p in pm.used_p(dv)] for (s, lay), dv in dec.items()
        },
    }


def _close(a: object, b: object, path: str = "") -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=RTOL, abs=1e-12, nan_ok=True), path
    else:
        assert type(a) is type(b), path
        assert a == b, path


@pytest.mark.claim("DEF-PMRT-WEIGHT")
def test_pipeline_reaches_the_same_conclusions_as_the_old_code(tmp_path: Path) -> None:
    with gzip.open(DATA / GOLDEN, "rt", encoding="utf-8") as fh:
        old = json.load(fh)
    new = json.loads(json.dumps(_golden_summary(tmp_path)))
    assert new.keys() == old.keys()
    for k in old:
        _close(new[k], old[k], k)


def test_bins_rebuild_the_unit_target_and_table(train_test: dict, tmp_path: Path) -> None:
    pd = train_test["te"]
    f = pm.cache_records(train_test["recs_te"], tmp_path / "x.npz")
    ud0 = records.load_pool([f], H=90).unit_data()
    for k in ("episode", "c", "t0", "mode", "sgn", "family"):
        assert np.array_equal(getattr(pd.ud, k), getattr(ud0, k))
    for rel in pd.ud.relations:
        for j, kpi in enumerate(pd.ud.kpis):
            y = pd.ud.y[(rel, kpi)]
            post = pd.post[:, j, :, :3].sum(-1) - pd.pre[:, j, :, :3].sum(-1)
            yy = (post * pd.ud.rel_mask[rel]).sum(1)
            assert np.allclose(yy, y, rtol=1e-4, atol=1e-2 * (1 + np.abs(y).mean()))
    late = pd.ud.t0 + 150 > 720
    assert np.all(np.isnan(pd.post[late][..., 3]))
    assert np.all(np.isfinite(pd.post[~late][..., 3]))


def test_hist_is_past_only(train_test: dict) -> None:
    pd = train_test["te"]
    ud = pd.ud
    i = int(np.nonzero(np.abs(pd.hist).sum(1) > 0)[0][0])
    same_ep = np.nonzero(ud.episode == ud.episode[i])[0]
    v = pu.v_design(ud.mode, ud.probs, ud.sgn, pm.LEVEL_V2_ARR)
    own = [(ud.c[j] == ud.c[i]) and 0 < ud.t0[i] - ud.t0[j] <= pm.HIST_WIN for j in same_ep]
    for f in range(len(FAMILIES)):
        want = sum(v[j] for j, ok in zip(same_ep, own, strict=True) if ok and ud.family[j] == f)
        assert np.isclose(pd.hist[i, f], want)


@pytest.mark.claim("DEF-PMRT-WEIGHT")
def test_family_weights_ignore_later_units(train_test: dict) -> None:
    pd, params = train_test["te"], train_test["params"]
    ud = pd.ud
    order = np.lexsort((ud.c, ud.t0, ud.episode))
    sub = pd.subset(np.sort(order[: len(order) * 2 // 3]))
    for f in ("sleep", "prot_min"):
        rows, w, _, _ = pu.family_columns(pd, f, params, FAST)
        rows_s, w_s, _, _ = pu.family_columns(sub, f, params, FAST)
        m = len(rows_s)
        assert m > 10
        key = list(zip(ud.episode[rows[:m]], ud.t0[rows[:m]], ud.c[rows[:m]], strict=True))
        key_s = list(zip(sub.ud.episode[rows_s], sub.ud.t0[rows_s], sub.ud.c[rows_s], strict=True))
        assert key == key_s
        assert np.allclose(w[:m], w_s, rtol=1e-9, atol=1e-9)


@pytest.mark.claim("ASM-ORIENT")
def test_orientation_is_fixed_by_the_training_fit(train_test: dict) -> None:
    params = train_test["params"]
    metas = [pm.lean_columns(train_test[k], "sleep", params, FAST)[3] for k in ("te", "nu")]
    for m_te, m_nu in zip(*metas, strict=True):
        if m_te is None or m_nu is None:
            continue
        assert m_te["o"] == m_nu["o"]
        assert m_te["z_plain_train"] == m_nu["z_plain_train"]
        assert set(m_te["o"].values()) <= {-1.0, 0.0, 1.0}
    core = params["hyp"]["sleep|nbr|pv"]["core"]
    want = float(core["sdir"]["loadsp"]) if abs(core["z_plain"]) >= 2.0 else 0.0
    assert pu.arm_direction("loadsp", core) == want


def test_crt_calibrated_on_planted_null_and_chunk_invariant(train_test: dict) -> None:
    pd, params = train_test["nu"], train_test["params"]
    run = pm.run_pmrt(pd, params, FAST)
    arms = ("best", "plain", "pred")
    ps = [r["arms"][a]["p"] for r in run["results"] if r["status"] == "tested" for a in arms]
    assert np.mean(np.array(ps) <= 0.05) <= 0.15
    assert sum(e["declared"] for e in pm.edges_pmrt(run, "best").values()) <= 1
    run2 = pm.run_pmrt(pd, params, dataclasses.replace(FAST, max_chunk_bytes=50_000))
    for r1, r2 in zip(run["results"], run2["results"], strict=True):
        if r1["status"] == "tested":
            assert {a: v["p"] for a, v in r1["arms"].items()} == {
                a: v["p"] for a, v in r2["arms"].items()
            }


def test_max_combination_p_dominates_its_argmax_arm(train_test: dict) -> None:
    run = pm.run_pmrt(train_test["te"], train_test["params"], FAST)
    for r in run["results"]:
        if r["status"] != "tested":
            continue
        for c, arms in pu.COMBOS.items():
            via = r["arms"][c]["via"]
            assert via in arms
            assert r["arms"][c]["p"] >= r["arms"][via]["p"] - 1e-12
            assert r["arms"][c]["p"] >= min(r["arms"][a]["p"] for a in arms) - 1e-12


def test_planted_effect_detected_on_fresh_episodes(train_test: dict) -> None:
    run = pm.run_pmrt(train_test["te"], train_test["params"], dataclasses.replace(FAST, B=4999))
    e = pm.edges_pmrt(run, "best")
    for h in (("sleep", "nbr", "load"), ("sleep", "nbr", "pv")):
        assert e[h]["declared"], (h, e[h])
        assert e[h]["sign"] == 1, (h, e[h])
    assert not [h for h, v in e.items() if v["declared"] and h[1] == "far"]


def test_integrated_statistic_matches_the_full_arm_tests(train_test: dict) -> None:
    pd, params = train_test["te"], train_test["params"]
    integ = pm.integrated_family(pd, "sleep", params, FAST, pm.SPLIT_POOLED)
    full = {(r["relation"], r["kpi"]): r for r in pm.family_tests_pmrt(pd, "sleep", params, FAST)}
    for t, o in integ.items():
        if o["status"] != "tested":
            continue
        for a in pm.MAX_OF:
            assert abs(o[a]["z_raw"]) == pytest.approx(abs(full[t]["arms"][a]["z"]), rel=1e-12)
            if np.isfinite(o[a]["p_plus"]):
                assert min(o[a]["p_plus"], o[a]["p_minus"]) <= o[a]["p2"] + 1e-12
        assert o["max"]["p2"] >= min(o[a]["p2"] for a in pm.MAX_OF) - 1e-12
        assert o["max"]["via"] in pm.MAX_OF


def test_redraws_are_chunk_invariant(train_test: dict) -> None:
    ud = train_test["te"].ud
    rows = ud.rows_of("sleep")
    cfg = dataclasses.replace(FAST, B=37)
    a = np.vstack(list(pm.redraws(ud, rows, cfg, pm.stream(cfg, 1, 0))))
    small = dataclasses.replace(cfg, max_chunk_bytes=8 * len(rows) * 5)
    b = np.vstack(list(pm.redraws(ud, rows, small, pm.stream(small, 1, 0))))
    assert a.shape == (37, len(rows))
    assert np.array_equal(a, b)


def test_design_regressor_is_mean_zero_under_the_logged_table() -> None:
    recs = synthetic_records(np.random.default_rng(3), SyntheticSpec(n_ep=3, units_per_ep=40))
    d = records.build_unit_data(recs)
    rows = d.rows_of("prot_min")
    lv = pm.LEVEL_V2_ARR
    m = d.probs[rows] @ lv
    ev = d.sgn[rows] * sum(d.probs[rows][:, j] * (lv[j] - m) for j in range(len(lv)))
    assert np.allclose(ev, 0.0)


def test_eproc_config_residual_part_is_the_ridge_default() -> None:
    assert pm.EProcConfig().residual() == ResidualConfig()
    assert pm.EProcConfig(burn=3).residual().burn == 3


def test_params_round_trip_without_boosted_models(tmp_path: Path) -> None:
    pd = pm.pmrt_data_from_records(_recs(21, 6, 8000, units=50), tmp_path)
    params = fit_pmrt(pd, pm.LEVEL_V2_ARR, PmrtConfig(cv_reps=1), families=("sleep",))
    path = tmp_path / "p.pkl"
    pm.save_params(params, path)
    summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(summary["hyp"]) == set(params["hyp"])
    full, lean = pm.load_params(path), pm.load_params_nogb(path)
    assert all(hp["gb"] is not None for hp in full["hyp"].values())
    assert all(hp["gb"] is None for hp in lean["hyp"].values())
    for h, hp in lean["hyp"].items():
        assert np.array_equal(hp["core"]["B"], params["hyp"][h]["core"]["B"])


def test_subsets_are_disjoint_and_reproducible(train_test: dict) -> None:
    pool = pm.load_pmrt_pool([str(train_test["dir"] / "te.npz")]).pool
    ids = pool.episodes()
    a = pm.subsets(pool, ids, 3, 5)
    assert [s["name"] for s in a] == ["rand0", "rand1", "rand2", "rand3"]
    assert [s["split"] for s in a] == [20, 21, 22, 23]
    eps = np.concatenate([s["episodes"] for s in a])
    assert len(set(eps.tolist())) == len(eps)
    b = pm.subsets(pool, ids, 3, 5)
    assert all(np.array_equal(x["episodes"], y["episodes"]) for x, y in zip(a, b, strict=True))
    seeds = pool.eps["seed"]
    spec = f"seed={seeds[0]},seed={seeds[2]}"
    assert np.array_equal(pm.select(pool, spec), np.isin(seeds, seeds[[0, 2]]).nonzero()[0])

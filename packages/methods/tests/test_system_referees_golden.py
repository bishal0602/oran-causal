import dataclasses
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.core.gate import tau_table
from oran_causal.methods.system import baselines as bd
from oran_causal.methods.system import maps as mp
from oran_causal.methods.system import mscr_crt as v2
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.records import build_unit_data
from oran_causal.sim.system.decision.synthetic import SyntheticSpec, synthetic_records
from oran_causal.sim.system.decision.vocab import FAMILIES

pytestmark = pytest.mark.filterwarnings(
    "ignore::RuntimeWarning", "ignore::UserWarning", "ignore::FutureWarning"
)

DATA = Path(__file__).parent / "data"
RTOL = 1e-6
ARTIFACTS = Path(__file__).parents[2] / "experiments" / "artifacts"
GOLDEN = {
    "win32": "system_referees_golden_win32.json.gz",
    "linux": "system_referees_golden_linux.json.gz",
}


@pytest.fixture(scope="module")
def gold() -> dict:
    name = GOLDEN.get(sys.platform)
    if name is None or not (DATA / name).exists():
        pytest.skip(f"no golden for {sys.platform}")
    with gzip.open(DATA / name, "rt", encoding="utf-8") as fh:
        return _strip_wall(json.load(fh))


def js(o: Any) -> Any:
    if isinstance(o, dict):
        return {
            ("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): js(v) for k, v in o.items()
        }
    if isinstance(o, list | tuple):
        return [js(v) for v in o]
    if isinstance(o, np.ndarray):
        return js(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, set):
        return sorted(js(v) for v in o)
    return o


_TIMING = ("wall_s", "cpu_s")


def _strip_wall(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_wall(v) for k, v in d.items() if k not in _TIMING}
    if isinstance(d, list):
        return [_strip_wall(v) for v in d]
    return d


def _same(a: Any, b: Any, path: str = "") -> None:
    if isinstance(b, dict):
        assert isinstance(a, dict), path
        assert sorted(a) == sorted(b), (path, sorted(set(a) ^ set(b)))
        for k in b:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(b, list):
        assert isinstance(a, list), path
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(b, float) and math.isnan(b):
        assert isinstance(a, float), (path, a)
        assert math.isnan(a), (path, a)
    else:
        assert a == b, (path, a, b)


def _close(a: Any, b: Any, path: str = "", rtol: float = RTOL) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert sorted(a) == sorted(b), (path, sorted(set(a) ^ set(b)))
        for k in b:
            _close(a[k], b[k], f"{path}/{k}", rtol)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{path}[{i}]", rtol)
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=rtol, abs=1e-12, nan_ok=True), (path, a, b)
    else:
        assert type(a) is type(b), (path, a, b)
        assert a == b, (path, a, b)


def _norm(o: Any) -> Any:
    return json.loads(json.dumps(js(o)))


def _table(rng_seed: int, **kw: Any) -> Any:
    return build_unit_data(synthetic_records(np.random.default_rng(rng_seed), SyntheticSpec(**kw)))


@pytest.fixture(scope="module")
def planted() -> Any:
    return _table(11, n_ep=8, units_per_ep=60, effect=200.0)


@pytest.fixture(scope="module")
def crt_run(planted: Any) -> dict:
    return v2.run_crt_units_v2(planted, v2.UnitCRTConfigV2(B=199))


@pytest.fixture(scope="module")
def base_data() -> Any:
    return _table(3, n_ep=6, units_per_ep=60, effect=120.0)


@pytest.fixture(scope="module")
def base_scores(base_data: Any) -> dict:
    return {name: bd.SCORERS[name](base_data) for name in ("corr", "granger", "int")}


def test_crt_matches_old(gold: dict, planted: Any, crt_run: dict) -> None:
    g = gold["crt"]
    _close(_norm(_strip_wall(crt_run)), g["run"], "run")
    _close(_norm(v2.edges_from_crt_v2(crt_run)), g["edges"], "edges")
    plc = _table(21, n_ep=8, units_per_ep=60, effect=60.0, placebo=True)
    k0 = v2.placebo_rejection_units_v2(plc, v2.UnitCRTConfigV2(B=199))
    _close(_norm(_strip_wall(k0)), g["k0"], "k0")
    cfg = v2.UnitCRTConfigV2(B=99, audit_B=99)
    aud = v2.carryover_audit_v2(planted, cfg, families=("sleep", "ptx"))
    _close(_norm(_strip_wall(aud)), g["audit"], "audit")
    sp = v2.run_crt_units_v2(
        planted, v2.UnitCRTConfigV2(B=99, seed=3), split=v2.SPLIT_PLACEBO, families=("ptx",)
    )
    _close(_norm(_strip_wall(sp)), g["split_run"], "split")
    _close(_norm([v2.power_floor(999, 60), v2.power_floor(9999, 60)]), g["power_floor"])


def test_fast_baselines_match_old(gold: dict, base_data: Any, base_scores: dict) -> None:
    g = gold["baselines"]
    for name, sc in base_scores.items():
        t = bd.tune_tau(sc, "far_fpr")
        got = {
            "scores": sc,
            "tau": t,
            "tau_physics": bd.tune_tau(sc, "physics_f1"),
            "declared": bd.declare(sc, t["tau"]),
        }
        _close(_norm(got), g[name], name)
    _close(_norm(bd.granger_by(base_scores["granger"])), g["granger_by"], "granger_by")
    q = bd.score_qacm(base_data, relations=("own", "far"), ann=False)
    tq = bd.tune_tau(q, "far_fpr")
    got = {"scores": q, "tau": tq, "declared": bd.declare(q, tq["tau"], bd.DECL_RELATIONS["qacm"])}
    _close(_norm(got), g["qacm"], "qacm")
    dev = base_data.subset(base_data.episode < 3)
    ev = base_data.subset(base_data.episode >= 3)
    rb = bd.run_baselines(
        dev,
        {"pooled": ev},
        methods=("corr", "granger"),
        sup=dataclasses.replace(bd.Support(), min_episodes=2),
    )
    _close(_norm(_strip_wall(rb)), g["run_baselines"], "run_baselines")


@pytest.mark.slow
def test_learned_baselines_match_old(gold: dict, base_data: Any) -> None:
    g = gold["baselines"]
    sh = _norm(bd.score_shap(base_data))
    assert "fit_shap_importances" in sh.pop("_note")
    assert sh.pop("_importance") == "tree_shap"
    _close(sh, {k: v for k, v in g["shap"]["scores"].items() if k != "_note"}, "shap")
    if "two_tower" not in g:
        pytest.skip("golden built without torch")
    sub = base_data.subset(base_data.family == 2)
    _close(_norm(bd.score_qacm(sub, relations=("own",), ann=True)), g["qacm_ann"], "qacm_ann")
    tt = _norm(bd.score_two_tower(base_data, sizes={"epochs": 300}))
    old = dict(g["two_tower"])
    assert tt.pop("_note").rsplit(", ", 1)[1] == old.pop("_note").rsplit(", ", 1)[1]
    _close(tt, old, "two_tower", rtol=1e-3)


def _pressured_episodes() -> list[dict]:
    recs = synthetic_records(np.random.default_rng(5), SyntheticSpec(n_ep=4, units_per_ep=60))
    rng = np.random.default_rng(6)
    for r in recs:
        for u in r["units"]:
            u["ctx"]["own_prot_below_frac"] = float(rng.uniform(0, 0.12))
            u["ctx"]["nbr_max_prot_below_frac"] = float(rng.uniform(0, 0.12))
    return recs


def _test_maps() -> dict:
    cells = mp.gt_cells()
    gt = mp.gt_map(cells)
    return {
        "GT": gt,
        "own": rf.own_only(gt),
        "flip": rf.sign_flip(gt, ("carrier",)),
        "blanket": rf.blanket_saving_map(),
        "rand": rf.random_sized_map(gt, cells),
    }


def test_maps_match_old(gold: dict, base_data: Any, base_scores: dict, crt_run: dict) -> None:
    g = gold["maps"]
    cells = mp.gt_cells()
    maps = _test_maps()
    _close(_norm(maps["GT"]), g["gt_map"], "gt_map")
    _close(_norm(maps), g["maps"], "maps")
    eps = _pressured_episodes()
    _close(_norm(mp.replay(eps, maps)), g["replay"], "replay")
    rp2 = mp.replay(
        eps,
        maps,
        pairs={"x": ("own", "flip"), "y": ("rand", "GT")},
        theta=0.0,
        k_conf=2,
        warm=200.0,
    )
    _close(_norm(rp2), g["replay_pairs"], "replay_pairs")
    eq = {k: mp.edge_quality(m, cells) for k, m in maps.items()}
    _close(_norm(eq), g["edge_quality"], "edge_quality")
    _close(_norm(mp.clean_map(maps["rand"], cells)), g["clean_rand"])
    _close(_norm(mp.inject(maps["own"], maps["flip"])), g["inject"])
    corr = base_scores["corr"]
    decl = bd.declare(corr, bd.tune_tau(corr, "far_fpr")["tau"])
    _close(_norm(list(mp.map_from_declared(decl, base_data))), g["map_from_declared"])
    pt = [
        mp.placebo_tau(js(corr), ("own", "nbr", "far")),
        mp.placebo_tau(base_scores["granger"], ("far",), max_fp=0),
        mp.placebo_tau(corr, ("own",), max_fp=100),
    ]
    _close(_norm(pt), g["placebo_tau"], "placebo_tau")
    hyps = (("sleep", "nbr", "pv"), ("ptx", "own", "e"), ("carrier", "far", "load"))
    _close(_norm([mp.naive_slope(base_data, *h) for h in hyps]), g["naive_slope"], "naive_slope")
    _close(_norm(mp.map_from_mscr(crt_run)), g["map_from_mscr"], "map_from_mscr")


def _ctx(knob: str, step: float, own: float = math.nan, nbr: float = math.nan) -> dict:
    return {"knob": knob, "step": float(step), rf.PRESSURE["own"]: own, rf.PRESSURE["nbr"]: nbr}


def _units(n: int = 4000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = int(rng.integers(3))
        f = str(rng.choice(FAMILIES))
        step = float(rng.choice([-1, 1, 0]))
        own, nbr = float(rng.uniform(0, 0.2)), float(rng.uniform(0, 0.2))
        out.append({"c": c, "ctx": _ctx(f, step, own, nbr)})
    return out


def _run_gate(g: Any, us: list[dict]) -> str:
    return "".join("r" if g(dict(u))[0] == "reject" else "a" for u in us)


def _frozen_maps() -> dict:
    doc = json.loads((ARTIFACTS / "E6P_CONF_MAPS.json").read_text(encoding="utf-8"))
    return {arm: rf.map_from_json(v["map"]) for arm, v in doc["arms"].items()}


def test_referee_gates_match_old(gold: dict) -> None:
    g = gold["referees"]
    us = _units()
    frozen = _frozen_maps()
    assert sorted(frozen) == sorted(g["v2"])
    for arm, M in frozen.items():
        a, c = rf.MapGateV2(M), rf.CertSafeMapGateV2(M, [("ptx", 1)])
        got = {
            "dec": _run_gate(a, us),
            "n": a.n,
            "table": rf.decision_table_v2(M),
            "sig": rf.decision_signature(M),
        }
        _same(_norm(got), g["v2"][arm], f"v2/{arm}")
        got_cs = {
            "dec": _run_gate(c, us),
            "n": c.n,
            "sig": rf.certsafe_signature(M, [("ptx", 1)]),
        }
        _same(_norm(got_cs), g["cs"][arm], f"cs/{arm}")
    gt = mp.gt_map(mp.gt_cells())
    for name, M in {
        "GT": gt,
        "own": rf.own_only(gt),
        "flip": rf.sign_flip(gt, ("carrier",)),
    }.items():
        a = rf.MapGate(M)
        got = {
            "dec": _run_gate(a, us),
            "n": a.n,
            "table": rf.decision_table(M),
            "table_v2": rf.decision_table_v2(M),
        }
        _same(_norm(got), g["v1"][name], f"v1/{name}")
    _same(_norm(rf.decision_signature(rf.blanket_saving_map())), g["blanket_sig"])
    _same(_norm(rf.all_accept_signature()), g["all_accept_sig"])


def test_certification_matches_old(gold: dict, planted: Any) -> None:
    g = gold["referees"]
    classes = [(f, d) for f in FAMILIES for d in (1, -1)]
    tau = tau_table(
        {"rlf": 43.15, "v": 20318.0}, {fd: 10.0 + i for i, fd in enumerate(classes)}, rf.GATE_SPACE
    )
    _close(_norm(rf.tau_to_json(tau)), g["tau"], "tau")
    tight = {
        (f, r, k): {"kind": "slope", "beta": 0.0, "se": 0.0}
        for f in FAMILIES
        for r in rf.RELS_V2
        for k in rf.GUARD_KPIS
    }
    wide = {
        **tight,
        ("ptx", "nbr", "rlf"): {"kind": "slope", "beta": -0.10, "se": 0.05},
        ("sleep", "far", "v"): {"kind": "slope", "beta": 30.0, "se": 400.0},
    }
    for arm, M in _frozen_maps().items():
        got = {
            "tight": rf.certify_map(M, tight, tau),
            "wide": rf.certify_map(M, wide, tau),
            "none": rf.certify_map(M, None, tau),
        }
        _close(_norm(got), g["certify"][arm], f"certify/{arm}")
    hyps = [(f, r, k) for f in FAMILIES for r in ("own", "nbr", "far") for k in ("pv", "rlf")]
    ds = {"|".join(h): rf.design_slope_bound(planted, *h) for h in hyps}
    _close(_norm(ds), g["design_slope"], "design_slope")
    ns = {"|".join(h): rf.naive_slope_se(planted, *h) for h in hyps}
    _close(_norm(ns), g["naive_slope"], "naive_slope")
    _close(_norm(rf.gt_ci_bounds(g["gt_cells_ci"])), g["gt_ci_bounds"], "gt_ci_bounds")

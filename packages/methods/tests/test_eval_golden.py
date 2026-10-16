import functools
import gzip
import json
import math
from pathlib import Path

import pytest

from oran_causal.core.data import Dataset
from oran_causal.methods.registry import ARMS
from oran_causal.sim.controlled.datasets.generate import dataset_hash, generate_dataset

GOLDEN = Path(__file__).parent / "data" / "eval_golden.json.gz"
LEARNED = {"cdl": 0.05, "two_tower": 0.1, "shap_dag": 0.05, "notears": 1e-4}
NATIVE_AGREEMENT = 0.9
SLOW = {"cdl", "mscr_eq", "mscr_native", "pmrt_nl_eq"}
REL, ABS = 1e-9, 1e-11


def _records() -> list[dict]:
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as f:
        return json.load(f)["records"]


def _params():
    for r in _records():
        j = r["job"]
        marks = [pytest.mark.slow] if r["arm"] in SLOW else []
        yield pytest.param(r, id=f"{r['arm']}-{j['world']}-{j['regime']}", marks=marks)


@functools.cache
def _dataset(cell: tuple[str, str, int, int, float, float]) -> tuple[Dataset, str]:
    world, regime, n, seed, lam, kappa = cell
    ds, _ = generate_dataset(world, regime, n, seed, lam=lam, kappa=kappa)
    return ds, dataset_hash(ds)


def _num(v) -> float:
    return math.nan if v is None else float(v)


def _close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(tol, REL * abs(a)) if tol else abs(a - b) <= ABS + REL * abs(a)


@pytest.mark.parametrize("rec", list(_params()))
def test_arm_reproduces_frozen_eval_record(rec):
    j = rec["job"]
    lam = j["lam"] if j["lam"] is not None else 1.0
    ds, digest = _dataset((j["world"], j["regime"], j["n"], j["seed"], lam, j["kappa"]))
    assert digest == rec["dataset_sha256"]
    res = ARMS[rec["method"]]().run(ds, dict(rec["config"]))
    new = {(e.source, e.target): e for e in res.edges}
    tol = LEARNED.get(rec["arm"], 0.0)
    for src, tgt, score, p, sign, declared in rec["edges"]:
        e = new[(src, tgt)]
        assert e.declared == declared, (src, tgt)
        assert e.sign == sign, (src, tgt)
        assert (e.p is None) == (p is None), (src, tgt)
        assert _close(_num(p), _num(e.p), tol), (src, tgt, p, e.p)
        assert _close(_num(score), float(e.score), tol), (src, tgt, score, e.score)
    native = res.notes.get("native_declared", res.notes.get("declared_native"))
    if rec["native"] is None:
        return
    if rec["arm"] not in LEARNED:
        assert list(native) == list(rec["native"])
        return
    differ = set(native) ^ set(rec["native"])
    assert 1 - len(differ) / len(rec["edges"]) >= NATIVE_AGREEMENT, sorted(differ)

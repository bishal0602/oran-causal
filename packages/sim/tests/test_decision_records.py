import json
from pathlib import Path

import numpy as np
import pytest

from oran_causal.core.data import UnitTable
from oran_causal.sim.system.decision import records as R
from oran_causal.sim.system.decision.synthetic import SyntheticSpec, synthetic_records
from oran_causal.sim.system.decision.vocab import LEVEL

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _write(recs: list[dict], path: Path, header: bool = True) -> None:
    with path.open("w") as fh:
        if header:
            fh.write(json.dumps({"kind": "header", "schema": R.SCHEMA}) + "\n")
        for r in recs:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> dict:
    d = tmp_path_factory.mktemp("cache")
    spec = SyntheticSpec(n_ep=12, units_per_ep=50, effect=60.0)
    recs = synthetic_records(np.random.default_rng(3), spec)
    for r in recs[::3]:
        r["units"][0]["ctx"]["own_prb_util"] = None
        r["units"][1]["ctx"].pop("nbr_mean_prb_util")
    recs = recs[::-1]
    f1, f2 = d / "res_0.jsonl", d / "res_1.jsonl"
    _write(recs[:7], f1)
    _write([*recs[7:], recs[0]], f2, header=False)
    c1, c2 = d / "a.npz", d / "b.npz"
    R.build_cache([str(f1)], c1, stages={"dev"}, src="synthA")
    R.build_cache([str(f2)], c2, stages={"dev"}, src="synthB")
    return {"recs": recs, "cache": [str(c1), str(c2)]}


def same_table(a: UnitTable, b: UnitTable) -> None:
    assert a.n == b.n
    for k in ("episode", "seed", "fold", "family", "c", "t0", "mode", "applied", "p", "probs"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    for k in ("step", "sgn", "prev"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert list(a.xapp) == list(b.xapp)
    for name in ("y", "pre", "ycell", "rel_mask"):
        da, db = getattr(a, name), getattr(b, name)
        assert list(da) == list(db)
        assert all(np.array_equal(da[k], db[k]) for k in da), name
    assert list(a.z) == list(b.z)
    assert all(np.array_equal(a.z[k], b.z[k], equal_nan=True) for k in a.z)
    assert a.meta["dropped"]["window"] == b.meta["dropped"]["window"]


def test_episode_subset_matches_build_unit_data(synth: dict) -> None:
    recs = sorted(synth["recs"], key=lambda r: (r["seed"], r.get("sub", "")))
    pool = R.load_pool(synth["cache"])
    ids = [e for e in range(pool.n_eps) if pool.eps["seed"][e] % 2 == 0]
    same_table(R.build_unit_data([r for r in recs if r["seed"] % 2 == 0]), pool.unit_data(ids))


def test_cache_dedupes_and_skips_other_stages(synth: dict, tmp_path: Path) -> None:
    recs = synth["recs"][:3]
    f = tmp_path / "res_x.jsonl"
    other = dict(recs[0], stage="gt", key=["gt", "", 1])
    _write([*recs, recs[1], other], f)
    s = R.build_cache([str(f)], tmp_path / "x.npz", stages={"dev"})
    assert s["episodes"] == 3
    assert s["stages"] == ["dev"]
    with pytest.raises(ValueError, match="no episode records"):
        R.build_cache([str(f)], tmp_path / "y.npz", stages={"eval"})
    both = [str(tmp_path / "x.npz"), synth["cache"][0]]
    seeds = {r["seed"] for r in synth["recs"][:7]} | {r["seed"] for r in recs}
    assert R.load_pool(both).n_eps == len(seeds)
    assert R.load_pool(both, stages={"eval"}).n_eps == 0
    assert R.load_pool(both, stages={"dev"}).n_eps > 0
    with pytest.raises(ValueError, match="not cached"):
        R.load_pool(both, H=45)


@pytest.mark.claim("DEF-SYS-HYP")
def test_targets_are_window_differences_over_the_relation() -> None:
    recs = synthetic_records(np.random.default_rng(2), SyntheticSpec(n_ep=2, units_per_ep=10))
    d = R.build_unit_data(recs)
    data = R.dec(recs[0]["lab_series"]["data"]).astype(float)
    u = recs[0]["units"][0]
    t0 = int(u["t0"])
    ue = recs[0]["lab_series"]["fields"].index("ue")
    nbr = [q for q in u["exp"] if q != u["c"]]
    want = data[t0 : t0 + 90, ue][:, nbr].sum() - data[t0 - 90 : t0, ue][:, nbr].sum()
    assert d.y[("nbr", "load")][0] == pytest.approx(want)
    assert d.pre[("own", "load")][0] == pytest.approx(data[t0 - 90 : t0, ue, u["c"]].sum())
    assert d.level[0] == LEVEL[u["mode"]]
    assert d.sgn[0] == np.sign(u["step"])
    assert d.x[0] == LEVEL[u["mode"]] * np.sign(u["step"])
    assert set(d.z) >= {"ctx_cur", "ctx_step", "pre_far_pv", "pre_nbr_load"}
    for k in d.kpis:
        total = sum(d.y[(r, k)] for r in ("own", "nbr", "far"))
        assert np.allclose(total, d.ycell[k].sum(1))


def test_privileged_fields_are_never_read() -> None:
    recs = synthetic_records(np.random.default_rng(4), SyntheticSpec(n_ep=3, units_per_ep=20))
    clean = R.build_unit_data(recs)
    for r in recs:
        for k in R.PRIVILEGED_KEYS:
            r[k] = object()
    poisoned = R.build_unit_data(recs)
    same_table(clean, poisoned)


def test_codec_and_unit_records_roundtrip() -> None:
    a = np.arange(24, dtype=float).reshape(2, 3, 4) / 7
    assert np.array_equal(R.dec(R.enc(a)), a.astype("<f4"))
    u = {
        "c": 3,
        "x": "ES",
        "x_idx": 0,
        "knob": "sleep",
        "t0": 120.0,
        "mode": "reject",
        "p": 0.5,
        "exp": [2, 3],
        "n_req": 2,
        "ctx": {"cur": 0.0, "prop": 1.0, "step": 1.0},
        "probs": {"accept": 0.85, "reject": 0.15},
        "inc": {"cls": "saving"},
    }
    r = R.unit_record(7, u, 1)
    assert (r["i"], r["applied_mode"], r["step"], r["lab_kpi"], r["rb_issued"]) == (
        7,
        "reject",
        1.0,
        None,
        False,
    )
    inc = R.incumbent_unit_record(7, u, 1)
    assert inc["probs"] == u["probs"]
    assert inc["inc"] == u["inc"]

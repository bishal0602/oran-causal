import gzip
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.core.data import Dataset, Design
from oran_causal.core.pmrt.engine import Redraw, RedrawConfig, crt, crt_stat
from oran_causal.core.pmrt.rows import PmrtCore

DATA = Path(__file__).parent / "data"


NUMPY_24 = "pmrt_rows_golden.json.gz"
NUMPY_25 = {
    "win32": "pmrt_rows_golden_np25.json.gz",
    "linux": "pmrt_rows_golden_np25_linux.json.gz",
}
SCORE_RTOL = 1e-12


@cache
def golden(name: str = NUMPY_24) -> dict[str, Any]:
    with gzip.open(DATA / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def arr(x: Any) -> np.ndarray | None:
    return None if x is None else np.asarray(x, float)


def to_dataset(spec: dict[str, Any]) -> Dataset:
    lag = np.array([[np.nan if v is None else v for v in row] for row in spec["lag"]], float)
    designs = tuple(
        Design(
            d["kind"],
            d["dist"],
            random_part=arr(d["random_part"]),
            fixed_part=arr(d["fixed_part"]),
            propensity=arr(d["propensity"]),
        )
        for d in spec["designs"]
    )
    actions, kpis = tuple(spec["actions"]), tuple(spec["kpis"])
    x = np.asarray(spec["X"], float)
    return Dataset(
        world="E1",
        regime="R1",
        n=len(x),
        seed=spec["seed"],
        action_names=actions,
        kpi_names=kpis,
        X_action=x,
        X_kpi_lag=lag,
        Y=np.asarray(spec["Y"], float),
        designs=designs,
        candidates=tuple((a, k) for a in actions for k in kpis),
        time_index=arr(spec["time_index"]),
        context=arr(spec["context"]),
        meta=spec["meta"],
    )


CASES = [(d["name"], c) for d in golden()["datasets"] for c in golden()["configs"]]


def run_case(
    name: str, dataset: str, config: str
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    spec = next(d for d in golden(name)["datasets"] if d["name"] == dataset)
    res = PmrtCore().run(to_dataset(spec), golden(name)["configs"][config])
    got = [
        {
            "src": e.source,
            "tgt": e.target,
            "score": None if np.isnan(e.score) else e.score,
            "p": e.p,
            "sign": e.sign,
            "declared": e.declared,
        }
        for e in res.edges
    ]
    return res, got, spec["results"][config]


@pytest.mark.claim("DEF-PMRT-PVAL", "DEF-ENV-ARMS")
@pytest.mark.parametrize(("dataset", "config"), CASES)
def test_pmrt_rows_bit_equal_to_pmrt_core(dataset: str, config: str) -> None:
    if sys.platform not in NUMPY_25:
        pytest.skip(f"no reference run for platform {sys.platform}")
    res, got, want = run_case(NUMPY_25[sys.platform], dataset, config)
    assert res.version == want["version"]
    assert got == want["edges"]
    assert res.notes["not_applicable"] == want["not_applicable"]
    for key, value in want["notes"].items():
        assert res.notes[key] == value, key


@pytest.mark.claim("DEF-PMRT-PVAL")
@pytest.mark.parametrize(("dataset", "config"), CASES)
def test_pmrt_rows_same_conclusion_as_numpy_24_run(dataset: str, config: str) -> None:
    _, got, want = run_case(NUMPY_24, dataset, config)
    for g, w in zip(got, want["edges"], strict=True):
        assert (g["src"], g["tgt"], g["p"], g["sign"], g["declared"]) == (
            w["src"],
            w["tgt"],
            w["p"],
            w["sign"],
            w["declared"],
        )
        if w["score"] is None:
            assert g["score"] is None
        else:
            assert g["score"] == pytest.approx(w["score"], rel=SCORE_RTOL)


def uniform_redraw(n: int, seed: int) -> Redraw:
    v = np.random.default_rng(seed).uniform(-0.5, 0.5, n)

    def draw(rng: np.random.Generator, b: int) -> np.ndarray:
        return rng.random((b, n)) - 0.5

    return Redraw(v, np.full(n, 1 / 12), draw)


@pytest.mark.claim("DEF-PMRT-PVAL", "THM-EXACT")
def test_crt_is_chunk_invariant() -> None:
    redraw = uniform_redraw(80, 1)
    w = np.random.default_rng(2).standard_normal((80, 3))
    runs = [
        crt(redraw, w, np.random.default_rng(5), RedrawConfig(b_max=999, chunk=c, seq_h=h))
        for c in (7, 1000)
        for h in (None, 20)
    ]
    for a, b in ((runs[0], runs[2]), (runs[1], runs[3])):
        assert np.array_equal(a["p2"], b["p2"])
        assert np.array_equal(a["draws"], b["draws"])


@pytest.mark.claim("THM-EXACT")
def test_crt_null_pvalues_are_super_uniform() -> None:
    n, reps = 60, 400
    rng = np.random.default_rng(11)
    w = rng.standard_normal((n, 1))
    p = []
    for r in range(reps):
        redraw = uniform_redraw(n, 100 + r)
        p.append(
            crt(redraw, w, np.random.default_rng(r), RedrawConfig(b_max=199, seq_h=None))["p2"][0]
        )
    p = np.array(p)
    for t in (0.05, 0.1, 0.25, 0.5):
        assert np.mean(p <= t) <= t + 3 * np.sqrt(t * (1 - t) / reps)


def test_besag_clifford_stops_at_h() -> None:
    redraw = uniform_redraw(50, 3)
    w = np.zeros((50, 1))
    out = crt(redraw, w, np.random.default_rng(1), RedrawConfig(b_max=999, seq_h=20))
    assert out["p2"][0] == 1.0
    assert out["draws"][0] == 20


def test_crt_stat_matches_crt_two_sided() -> None:
    redraw = uniform_redraw(40, 4)
    w = np.random.default_rng(5).standard_normal((40, 2))
    sd = np.sqrt(redraw.var @ (w * w))

    def stat(v: np.ndarray) -> np.ndarray:
        return np.abs(v @ w) / sd

    config = RedrawConfig(b_max=499, seq_h=None)
    a = crt(redraw, w, np.random.default_rng(9), config)
    b = crt_stat(stat, redraw.v, redraw.draw, np.random.default_rng(9), config)
    assert np.array_equal(a["p2"], b["p"])

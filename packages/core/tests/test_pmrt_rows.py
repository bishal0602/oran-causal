import numpy as np
import pytest

from oran_causal.core.pmrt.engine import Redraw, RedrawConfig, crt, crt_stat


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

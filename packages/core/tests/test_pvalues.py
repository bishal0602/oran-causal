import numpy as np
import pytest

from oran_causal.core.pvalues import ExceedanceCounter, SequentialP, sequential_p_from_draws


@pytest.mark.claim("DEF-PMRT-PVAL")
@pytest.mark.parametrize("h", [None, 5, 20])
def test_sequential_p_matches_draw_array(h: int | None) -> None:
    rng = np.random.default_rng(1)
    for obs in (0.1, 1.0, 2.5):
        null = rng.normal(size=200)
        seq = SequentialP(obs, 200, h)
        for x in null:
            if seq.add(float(x)):
                break
        p, used = sequential_p_from_draws(obs, null, h)
        assert seq.p == pytest.approx(p)
        assert seq.k == used


@pytest.mark.claim("DEF-PMRT-PVAL")
def test_counter_matches_sequential_p() -> None:
    rng = np.random.default_rng(2)
    exceed = rng.random((300, 4)) < np.array([0.01, 0.05, 0.3, 0.9])
    counter = ExceedanceCounter(4, 20)
    for chunk in np.array_split(exceed, 7):
        counter.update(chunk)
    p, used = counter.pvalues(300)
    for j in range(4):
        ref_p, ref_k = sequential_p_from_draws(0.5, exceed[:, j].astype(float), 20)
        assert p[j] == pytest.approx(ref_p)
        assert used[j] == ref_k
    fixed = ExceedanceCounter(4, None)
    fixed.update(exceed)
    assert np.array_equal(fixed.pvalues(300)[0], (1 + exceed.sum(0)) / 301)

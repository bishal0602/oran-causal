"""Monte Carlo p-values: fixed B and Besag-Clifford (1991) sequential stopping."""

import numpy as np

BC_H = 20


class ExceedanceCounter:
    """Exceedance counts of several statistics over redraws consumed in stream order.

    With `seq_h` set, a statistic stops at the draw k where its count reaches h and gets
    p = h / k; a statistic that never stops gets (1 + count) / (B + 1). With `seq_h` None,
    every statistic gets the fixed-B p-value. Both rules are chunk-invariant.
    """

    def __init__(self, m: int, seq_h: int | None):
        """Starts empty counters for `m` statistics."""
        self.seq_h = seq_h
        self.cnt = np.zeros(m, np.int64)
        self.k_stop = np.zeros(m, np.int64)
        self.done = 0

    def all_stopped(self) -> bool:
        """True when sequential stopping is on and every statistic has stopped."""
        return self.seq_h is not None and bool(np.all(self.k_stop > 0))

    def update(self, exceed: np.ndarray) -> None:
        """Adds one chunk of exceedance indicators, shape `[b, m]`, in draw order."""
        if self.seq_h is None:
            self.cnt += np.count_nonzero(exceed, axis=0)
        else:
            cs = self.cnt[None, :] + np.cumsum(exceed, axis=0, dtype=np.int64)
            hit = (self.k_stop == 0) & (cs[-1] >= self.seq_h)
            if hit.any():
                self.k_stop[hit] = self.done + 1 + np.argmax(cs[:, hit] >= self.seq_h, axis=0)
            self.cnt = cs[-1]
        self.done += len(exceed)

    def pvalues(self, b_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (p, draws used) per statistic for a maximum of `b_max` redraws."""
        b1 = b_max + 1.0
        if self.seq_h is None:
            return (1 + self.cnt) / b1, np.full(len(self.cnt), self.done)
        stopped = self.k_stop > 0
        p = np.where(stopped, self.seq_h / np.maximum(self.k_stop, 1), (1 + self.cnt) / b1)
        return p, np.where(stopped, self.k_stop, self.done)


class SequentialP:
    """Besag-Clifford sequential p-value of one statistic fed one null draw at a time.

    `h=None` gives the fixed-B Monte Carlo p-value.
    """

    def __init__(self, obs: float, b_max: int, h: int | None):
        """Starts with no draws for observed value `obs`."""
        self.obs, self.b_max, self.h = obs, int(b_max), h
        self.k = 0
        self.count = 0

    def add(self, null_value: float) -> bool:
        """Adds one null draw; returns `done`."""
        self.k += 1
        if null_value >= self.obs:
            self.count += 1
        return self.done

    @property
    def done(self) -> bool:
        """True after h exceedances or `b_max` draws."""
        return (self.h is not None and self.count >= self.h) or self.k >= self.b_max

    @property
    def p(self) -> float:
        """Current p-value."""
        if self.h is not None and self.count >= self.h:
            return self.h / self.k
        return (1.0 + self.count) / (self.b_max + 1.0)


def sequential_p_from_draws(obs: float, null: np.ndarray, h: int | None) -> tuple[float, int]:
    """Besag-Clifford p of an ordered array of null draws, identical to feeding `SequentialP`.

    Returns:
        (p, number of draws used).
    """
    b = len(null)
    exceed = np.cumsum(np.asarray(null) >= obs)
    if h is not None and exceed[-1] >= h:
        k = int(np.searchsorted(exceed, h)) + 1
        return h / k, k
    return (1.0 + float(exceed[-1])) / (b + 1.0), b

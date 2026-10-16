from collections.abc import Callable

import numpy as np
import pytest

from oran_causal.core.data import Dataset, Design, Truth

CLASSIC_ACTIONS = ("A0", "A1", "A2", "P_placebo")
CLASSIC_KPIS = ("Y0", "Y1", "Y2")
CLASSIC_PLANTED = {("A0", "Y0"): +1, ("A1", "Y1"): -1, ("Y0", "Y0"): +1, ("Y1", "Y2"): +1}
CI_ACTIONS = ("P0", "P1", "P2", "P3", "P_placebo")
CI_KPIS = ("K0", "K1", "K2", "K3")


def _classic_actions(
    r: np.random.Generator, t_len: int, p: int, regime: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if regime == "R1":
        return r.uniform(0.0, 1.0, (t_len, p)), None, None
    if regime != "R2":
        raise ValueError(regime)
    sp = np.empty((t_len, p))
    sp[0] = 0.5
    for t in range(1, t_len):
        sp[t] = np.clip(sp[t - 1] + r.normal(0, 0.03, p), 0.25, 0.75)
    d = r.uniform(-0.25, 0.25, (t_len, p))
    return sp + d, sp, d


def _classic_kpis(
    r: np.random.Generator,
    a: np.ndarray,
    shape: tuple[int, int],
    *,
    b: float,
    noise: float,
    nonlinear: bool,
) -> np.ndarray:
    t_len, nk = shape
    k = np.zeros((t_len + 1, nk))
    for t in range(t_len):
        a0, a1 = a[t, 0], a[t, 1]
        f0, f1 = (np.sin(2 * np.pi * a0), -4.0 * (a1 - 0.5) ** 2) if nonlinear else (a0, -a1)
        e = r.normal(0, noise, 3)
        k[t + 1, 0] = b * f0 + 0.5 * b * k[t, 0] + e[0]
        k[t + 1, 1] = b * f1 + e[1]
        k[t + 1, 2] = 0.7 * b * k[t, 1] + e[2]
        if nk > 3:
            k[t + 1, 3:] = r.normal(0, 1, nk - 3)
    return k


def classic_synth(
    n: int,
    seed: int,
    *,
    b: float = 0.5,
    noise: float = 1.0,
    regime: str = "R1",
    nonlinear: bool = False,
    burn: int = 50,
) -> tuple[Dataset, Truth]:
    """Synthetic SCM of the classic tests (CDD-ORAN `xmethod/methods/_classic_synth.py`).

    Y0 = b A0 + .5 b K0_t + e, Y1 = -b A1 + e, Y2 = .7 b K1_t + e; A2 and P_placebo are exact
    nulls. R1: iid U(0, 1) actions; R2: bounded slow setpoint plus U(-.25, .25) dither.
    """
    r = np.random.default_rng(
        [9_100, int(seed), int(n), int(b * 1000), int(regime == "R2"), int(nonlinear), 0, 0]
    )
    t_len = n + burn
    p = len(CLASSIC_ACTIONS)
    a, fixed, rand = _classic_actions(r, t_len, p, regime)
    k = _classic_kpis(r, a, (t_len, len(CLASSIC_KPIS)), b=b, noise=noise, nonlinear=nonlinear)
    sl = slice(burn, t_len)
    if regime == "R1":
        designs = tuple(Design("iid", {"name": "uniform", "lo": 0.0, "hi": 1.0}) for _ in range(p))
    else:
        assert fixed is not None
        assert rand is not None
        designs = tuple(
            Design(
                "dither",
                {"name": "uniform", "lo": -0.25, "hi": 0.25},
                random_part=rand[sl, j].copy(),
                fixed_part=fixed[sl, j].copy(),
            )
            for j in range(p)
        )
    cands = tuple((s, t) for s in CLASSIC_ACTIONS + CLASSIC_KPIS for t in CLASSIC_KPIS)
    data = Dataset(
        "SYN",
        regime,
        n,
        int(seed),
        CLASSIC_ACTIONS,
        CLASSIC_KPIS,
        a[sl].copy(),
        k[sl].copy(),
        k[burn + 1 : t_len + 1].copy(),
        designs,
        cands,
        time_index=np.arange(n),
        meta={"synthetic": True, "b": b, "nonlinear": nonlinear},
    )
    planted = CLASSIC_PLANTED if b != 0 else {}
    signs = {e: v for e, v in planted.items() if not (nonlinear and e[0].startswith("A"))}
    truth = Truth(
        "SYN", regime, frozenset(planted), signs, frozenset(c for c in cands if c not in planted)
    )
    return data, truth


def ci_candidates() -> tuple[tuple[str, str], ...]:
    return tuple((a, k) for a in CI_ACTIONS for k in CI_KPIS) + tuple(
        (s, k) for s in CI_KPIS for k in CI_KPIS
    )


def ci_synth(kind: str, n: int, seed: int, world: str = "SYN") -> tuple[Dataset, Truth]:
    """Synthetic data of the CI tests (CDD-ORAN `xmethod/methods/_citests_synth.py`).

    `"planted"`: K0 <- .6 P0, K1 <- -.6 P1, K2 <- .8 P2 1{P3 > .4}, K3 <- .5 lagged K3;
    `"null"`: K_j(t+1) = .5 K_j(t) + noise.
    """
    rng = np.random.default_rng([7802, 99, int(seed), n, 0 if kind == "planted" else 1])
    a = rng.uniform(-1.0, 1.0, size=(n, len(CI_ACTIONS)))
    lag = np.zeros((n, len(CI_KPIS)))
    y = np.zeros((n, len(CI_KPIS)))
    k_prev = rng.normal(size=len(CI_KPIS))
    for t in range(n):
        lag[t] = k_prev
        eps = rng.normal(size=len(CI_KPIS))
        if kind == "planted":
            row = (
                np.array(
                    [
                        0.6 * a[t, 0],
                        -0.6 * a[t, 1],
                        0.8 * a[t, 2] * (a[t, 3] > 0.4),
                        0.5 * k_prev[3],
                    ]
                )
                + eps
            )
        elif kind == "null":
            row = 0.5 * k_prev + eps
        else:
            raise ValueError(kind)
        y[t] = row
        k_prev = row
    if kind == "planted":
        edges = {
            ("P0", "K0"): 1,
            ("P1", "K1"): -1,
            ("P2", "K2"): 1,
            ("P3", "K2"): 1,
            ("K3", "K3"): 1,
        }
    else:
        edges = {(k, k): 1 for k in CI_KPIS}
    cands = ci_candidates()
    designs = tuple(Design("iid", {"name": "uniform", "lo": -1.0, "hi": 1.0}) for _ in CI_ACTIONS)
    ds = Dataset(
        world, "R1", n, int(seed), CI_ACTIONS, CI_KPIS, a, lag, y, designs, cands,
        time_index=np.arange(n), meta={"synthetic": kind},
    )  # fmt: skip
    truth = Truth(
        world, "R1", frozenset(edges), dict(edges), frozenset(c for c in cands if c not in edges)
    )
    return ds, truth


@pytest.fixture(scope="session")
def make_classic() -> Callable[..., tuple[Dataset, Truth]]:
    return classic_synth


@pytest.fixture(scope="session")
def make_ci() -> Callable[..., tuple[Dataset, Truth]]:
    return ci_synth

"""Synthetic data for style specimens ONLY. Never import this from a paper figure."""

import numpy as np

SEED = 20261002
RNG = np.random.default_rng(SEED)


def reseed() -> None:
    """Fresh generator at the start of each specimen build, so a specimen's data do not depend on
    which figures were built before it in the same process (the CLI, or a test module that builds
    every figure).
    """
    global RNG
    RNG = np.random.default_rng(SEED)


METHODS = ["Proposed", "Baseline A", "Baseline B", "Baseline C"]
XAPPS = ["MRO", "MLB", "ES", "CCO", "TS"]
NCPS = ["Tx power", "CIO", "TTT", "Hysteresis", "Tilt"]
KPIS = ["Thrpt.", "Latency", "HO rate", "RLF", "Energy"]


def utility(steps: np.ndarray) -> np.ndarray:
    params = [(0.56, 0.34, 4), (0.50, 0.30, 8), (0.53, 0.21, 1.8), (0.50, 0.12, 5)]
    return np.array(
        [a + b * (1 - np.exp(-steps / t)) + RNG.normal(0, 0.013, steps.size) for a, b, t in params]
    )


def convergence(x: np.ndarray, n_seeds: int = 10) -> np.ndarray:
    """(method, seed, x) F1 curves for proposed and baseline A."""
    out = []
    for top, tau in [(0.93, 9), (0.78, 13)]:
        base = top * (1 - np.exp(-x / tau))
        noise = RNG.normal(0, 1, (n_seeds, 1)) * (0.09 * np.exp(-x / 18) + 0.02)
        out.append(np.clip(base + noise, 0, 1))
    return np.array(out)


def final_f1(n_seeds: int = 10) -> np.ndarray:
    return np.array(
        [
            np.clip(RNG.normal(mu, sd, n_seeds), 0, 0.99)
            for mu, sd in [(0.92, 0.025), (0.79, 0.04), (0.74, 0.05), (0.63, 0.06)]
        ]
    )


BAR_VALUES = np.array(
    [[0.94, 0.81, 0.72, 0.66], [0.91, 0.77, 0.80, 0.58], [0.92, 0.79, 0.76, 0.62]]
)
BAR_ERRORS = np.array(
    [[0.02, 0.04, 0.05, 0.05], [0.03, 0.04, 0.04, 0.06], [0.02, 0.03, 0.04, 0.05]]
)
CMI = np.array(
    [
        [0.31, 0.06, 0.03, 0.04, 0.27],
        [0.05, 0.02, 0.24, 0.14, 0.01],
        [0.02, 0.03, 0.19, 0.06, 0.02],
        [0.03, 0.01, 0.17, 0.09, 0.02],
        [0.22, 0.04, 0.05, 0.13, 0.08],
    ]
)
SIGNED = np.array(
    [
        [0.72, -0.31, 0.05, -0.12, -0.64],
        [0.08, 0.03, 0.58, -0.41, 0.02],
        [-0.04, 0.06, -0.47, 0.29, 0.01],
        [0.02, -0.02, -0.38, 0.22, 0.04],
        [0.41, -0.12, 0.18, -0.33, 0.27],
    ]
)
SHARES = np.array(
    [
        [0.31, 0.22, 0.12, 0.20, 0.15],
        [0.18, 0.34, 0.10, 0.14, 0.24],
        [0.24, 0.16, 0.27, 0.19, 0.14],
        [0.12, 0.21, 0.18, 0.33, 0.16],
    ]
)

"""E4 variant for regime R3: the confounder is observed and the behaviour propensities are logged.

The outcome mechanism, the latent tape slots and the timeline are those of `E4Env`. Only the
behaviour action changes, from the continuous clipped normal to its rounding onto the do-grid
`V = {0, 0.01, ..., 1}`::

    W_q = 0.5 + lam * Z_q + eta_q,   eta_q ~ N(0, 0.5^2)
    A_q = V[k],   k = floor(100 * clip(W_q, 0, 1) + 0.5)

so that `P(A_q = V[k] | Z_q)` is an exact, logged probability with full support on `V`.
"""

import math

import numpy as np

from oran_causal.sim.controlled.e4 import E4Env

GRID = np.arange(101, dtype=float) / 100.0
ETA_SD = 0.5

_erfc = np.vectorize(math.erfc, otypes=[float])


def _phi_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def _interval_prob(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    upper = a > 0
    return np.where(upper, _phi_cdf(-a) - _phi_cdf(-b), _phi_cdf(b) - _phi_cdf(a))


def grid_index(w: np.ndarray | float) -> np.ndarray:
    """Returns the grid index of the rounded, clipped behaviour draw `w`."""
    return np.floor(100.0 * np.clip(np.asarray(w, dtype=float), 0.0, 1.0) + 0.5).astype(np.int64)


def policy_probs(z: np.ndarray, lam: float, eta_sd: float = ETA_SD) -> np.ndarray:
    """Returns the `[n, 101]` table of `P(A = V[k] | Z = z)`; rows sum to 1."""
    m = 0.5 + float(lam) * np.asarray(z, dtype=float).reshape(-1, 1)
    edges = np.concatenate([[-np.inf], (np.arange(100, dtype=float) + 0.5) / 100.0, [np.inf]])
    x = (edges[None, :] - m) / eta_sd
    return _interval_prob(x[:, :-1], x[:, 1:])


def policy_prob_of(k: np.ndarray, z: np.ndarray, lam: float, eta_sd: float = ETA_SD) -> np.ndarray:
    """Returns `P(A = V[k_i] | Z = z_i)` per row: the propensity of the realised action."""
    k = np.asarray(k, dtype=np.int64)
    m = 0.5 + float(lam) * np.asarray(z, dtype=float)
    lo_edge = np.where(k == 0, -np.inf, (k - 0.5) / 100.0)
    hi_edge = np.where(k == 100, np.inf, (k + 0.5) / 100.0)
    return _interval_prob((lo_edge - m) / eta_sd, (hi_edge - m) / eta_sd)


class E4LoggedEnv(E4Env):
    """E4 with the behaviour action rounded onto the do-grid (observation mode only)."""

    def __init__(self, env_seed: int = 0, lam: float = 1.0, episode: int = 0) -> None:
        super().__init__(env_seed=env_seed, lam=lam, mode=self.MODE_OBS, episode=episode)

    def behavior_index(self) -> int:
        """Returns the grid index of the discretised behaviour action at the pending coordinate."""
        eta = self._draw_eta(self.time)
        return int(grid_index(0.5 + self.lam * self.Z + eta))

    def behavior_action(self) -> float:
        """Returns the grid value of the discretised behaviour action."""
        return float(GRID[self.behavior_index()])

"""E1: linear graph with one-step lag.

Latent updates, with `K*_prev` the previous-step latent KPI::

    K0(t) = a0 * P0(t-1)
    K1(t) = a1 * P1(t-1)
    K2(t) = a2 * P2(t-1) + b2 * K0(t-1)
    K3(t) = a3 * P3(t-1) + b3 * K1(t-1)

The coefficients default to the locked structural model and are constructor arguments so
that single-edge shadows can be built.
"""

from typing import ClassVar

import numpy as np

from oran_causal.sim.controlled.base import ControlledEnv, Edges

SIGMA_K01 = float(np.sqrt(1.0 / 12.0))
SIGMA_K23 = float(np.sqrt(1.25 / 12.0))
ADJACENCY_EDGES: Edges = [(0, 0), (1, 1), (2, 2), (3, 3), (2, 4), (3, 5)]


class E1Env(ControlledEnv):
    """Four parameters, four KPIs; KPIs 2 and 3 inherit from KPIs 0 and 1."""

    num_params = 4
    num_kpis = 4
    id_ranges: ClassVar[list[tuple[float, float]]] = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    mu = np.array([0.5, 0.5, 0.75, 0.75], dtype=float)
    sigma = np.array([SIGMA_K01, SIGMA_K01, SIGMA_K23, SIGMA_K23], dtype=float)
    kpi_to_xapp: ClassVar[dict[int, int]] = {0: 0, 1: 1, 2: 2, 3: 3}
    xapp_kpi_indices: ClassVar[list[tuple[int, ...]]] = [(0,), (1,), (2,), (3,)]
    xapp_param_indices: ClassVar[list[tuple[int, ...]]] = [(0,), (1,), (2,), (3,)]
    kpi_thresholds = (0.5, 0.5, 0.75, 0.75)
    directions = (0, 0, 0, 0)

    def __init__(  # noqa: PLR0913
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        *,
        a: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        b2: float = 0.5,
        b3: float = 0.5,
        episode: int = 0,
    ) -> None:
        self.a = np.asarray(a, dtype=float)
        self.b2 = float(b2)
        self.b3 = float(b3)
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    @property
    def adjacency_edges(self) -> Edges:
        """The locked E1 edges."""
        return list(ADJACENCY_EDGES)

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.a[1] * p[1]
        k2 = self.a[2] * p[2] + self.b2 * k[0]
        k3 = self.a[3] * p[3] + self.b3 * k[1]
        return np.array([k0, k1, k2, k3], dtype=float)

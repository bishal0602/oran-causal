"""E3: depth-three cascade with a fan-out and a non-descendant control.

Latent updates, with `K*_prev` the previous-step latent KPI::

    K0(t) = P0(t-1)
    K1(t) = K0(t-1)
    K2(t) = P1(t-1) + K1(t-1)
    K3(t) = P2(t-1) + K1(t-1)
    K4(t) = P3(t-1)

The graph is `P0 -> K0 -> K1 -> {K2, K3}` plus `P1 -> K2`, `P2 -> K3` and `P3 -> K4`.
`K1` is an internal conduit and `K4` a non-descendant control; both are excluded from the
decision panel. The coefficient arguments allow single-edge shadows; `truncate_fanout` zeroes
the contribution of `K1` to both `K2` and `K3`.
"""

import math
from typing import ClassVar

import numpy as np

from oran_causal.sim.controlled.base import ControlledEnv, Edges

SIGMA_K01 = math.sqrt(1.0 / 12.0)
SIGMA_K23 = math.sqrt(2.0 / 12.0)
TRUE_ADJACENCY: Edges = [(0, 0), (1, 4), (2, 1), (2, 5), (3, 2), (3, 5), (4, 3)]
TRUNCATED_OMITTED_EDGES: Edges = [(2, 5), (3, 5)]


class E3Env(ControlledEnv):
    """Four parameters, five KPIs."""

    num_params = 4
    num_kpis = 5
    id_ranges: ClassVar[list[tuple[float, float]]] = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    mu = np.array([0.5, 0.5, 1.0, 1.0, 0.5], dtype=float)
    sigma = np.array([SIGMA_K01, SIGMA_K01, SIGMA_K23, SIGMA_K23, SIGMA_K01], dtype=float)
    xapp_kpi_indices: ClassVar[list[tuple[int, ...]]] = [(0,), (1,), (2,), (3,), (4,)]
    kpi_to_xapp: ClassVar[dict[int, int]] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    xapp_param_indices: ClassVar[list[tuple[int, ...]]] = [(0,), (0,), (1,), (2,), (3,)]
    directions = (0, 0, 1, 1, 0)
    kpi_thresholds = (0.5, 0.5, 1.0, 1.0, 0.5)
    panel_kpi_ids = (0, 2, 3)
    excluded_kpi_ids = (1, 4)

    def __init__(  # noqa: PLR0913
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        *,
        a: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        c10: float = 1.0,
        c25: float = 1.0,
        c35: float = 1.0,
        truncate_fanout: bool = False,
        episode: int = 0,
    ) -> None:
        self.a = np.asarray(a, dtype=float)
        self.c10 = float(c10)
        self.truncate_fanout = bool(truncate_fanout)
        self.c25 = 0.0 if self.truncate_fanout else float(c25)
        self.c35 = 0.0 if self.truncate_fanout else float(c35)
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.c10 * k[0]
        k2 = self.a[1] * p[1] + self.c25 * k[1]
        k3 = self.a[2] * p[2] + self.c35 * k[1]
        k4 = self.a[3] * p[3]
        return np.array([k0, k1, k2, k3, k4], dtype=float)

    @property
    def adjacency_edges(self) -> Edges:
        """Edges implied by the active coefficients; direct parameter edges are always present."""
        edges = [(0, 0)]
        if self.c10 != 0.0:
            edges.append((1, 4))
        edges.append((2, 1))
        if self.c25 != 0.0:
            edges.append((2, 5))
        edges.append((3, 2))
        if self.c35 != 0.0:
            edges.append((3, 5))
        edges.append((4, 3))
        return edges

"""E2: shared-control conflict with a fan-out through one parameter.

Six Gaussian-bump mechanisms with the shared control `P0` fanning out into `K0, K1, K2, K5`.
With `decoy_omit_p0_k5` the committed pre-decision `P0` is frozen inside the `K5` mechanism
only, so the decoy model predicts no effect of `P0` on `K5`; realised scoring always uses the
true structural model.
"""

from math import exp
from typing import ClassVar

import numpy as np

from oran_causal.sim.controlled.base import ControlledEnv, Edges, Snapshot

KPI_MEAN_STD: list[tuple[float, float]] = [
    (21.482606, 27.694787),
    (26.776967, 34.651755),
    (40.931657, 44.767996),
    (14.975340, 32.369335),
    (18.755966, 40.460422),
    (-0.956730, 4.771898),
]
TRUE_ADJACENCY: Edges = [
    (0, 0), (0, 1),
    (1, 0), (1, 1), (1, 2),
    (2, 3), (2, 0),
    (3, 1), (3, 4), (3, 5),
    (4, 1), (4, 4), (4, 5),
    (5, 0), (5, 6), (5, 7),
]  # fmt: skip
DECOY_OMITTED_EDGE = (5, 0)
WIDTH_FLOOR = 1e-1


def safe_exp(x: float) -> float:
    """Returns `x` if `|x| > 0.1`, else 0.1: the width floor of the bump mechanisms."""
    return x if abs(x) > WIDTH_FLOOR else WIDTH_FLOOR


class E2Env(ControlledEnv):
    """Eight parameters, six KPIs; `K5` carries the decision-critical shared-knob effect."""

    num_params = 8
    num_kpis = 6
    id_ranges: ClassVar[list[tuple[float, float]]] = [
        (-100.0, 100.0),
        (-10.0, 50.0),
        (-20.0, 20.0),
        (-60.0, 60.0),
        (-20.0, 20.0),
        (-50.0, 150.0),
        (1.5, 4.0),
        (-100.0, 150.0),
    ]
    mean = np.array([m for m, _ in KPI_MEAN_STD], dtype=float)
    std = np.array([s for _, s in KPI_MEAN_STD], dtype=float)
    kpi_to_xapp: ClassVar[dict[int, int]] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
    xapp_kpi_indices: ClassVar[list[tuple[int, ...]]] = [(0,), (1,), (2,), (3, 4), (5,)]
    xapp_param_indices: ClassVar[list[tuple[int, ...]]] = [
        (0, 1),
        (0, 1, 2),
        (0, 3),
        (4, 5, 1),
        (0, 6, 7),
    ]
    directions = (0, 0, 0, 0, 0, 1)
    kpi_thresholds = (55.0, 95.0, 85.0, 75.0, 80.0, -25.0)

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        decoy_omit_p0_k5: bool = False,
        episode: int = 0,
    ) -> None:
        self.decoy_omit_p0_k5 = bool(decoy_omit_p0_k5)
        self._decoy_p0_ref = 0.0
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def reset(self, episode: int | None = None) -> dict[str, np.ndarray]:
        """Re-initialises and captures the committed `P0` used by the decoy."""
        state = super().reset(episode=episode)
        self._decoy_p0_ref = float(self.prev_params[0])
        return state

    def restore(self, snap: Snapshot) -> None:
        """Restores a snapshot and re-captures the committed `P0` used by the decoy."""
        super().restore(snap)
        self._decoy_p0_ref = float(self.prev_params[0])

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        p0_k5 = self._decoy_p0_ref if self.decoy_omit_p0_k5 else p[0]
        k0 = 80.0 * exp(-((p[0]) ** 2) / (2.0 * safe_exp(p[1]) ** 2))
        k1 = 100.0 * exp(-((p[0] + p[2]) ** 2) / (2.0 * safe_exp(p[1]) ** 2))
        k2 = 120.0 * exp(-((p[0] + 45.0) ** 2) / (2.0 * safe_exp(p[3]) ** 2))
        k3 = 120.0 * exp(-((p[5] + p[1] - 30.0) ** 2) / (2.0 * safe_exp(p[4]) ** 2))
        k4 = 150.0 * exp(-((p[5] + p[1] - 50.0) ** 2) / (2.0 * safe_exp(p[4]) ** 2))
        k5 = -35.0 * exp(-((p[7] + p0_k5 - 25.0) ** 2) / (2.0 * safe_exp(p[6]) ** 2))
        return np.array([k0, k1, k2, k3, k4, k5], dtype=float)

    @property
    def adjacency_edges(self) -> Edges:
        """True edges, minus `P0 -> K5` when the decoy omits it."""
        if self.decoy_omit_p0_k5:
            return [e for e in TRUE_ADJACENCY if e != DECOY_OMITTED_EDGE]
        return list(TRUE_ADJACENCY)

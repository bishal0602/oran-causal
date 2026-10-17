"""E4: confounded decision with a latent confounder.

One acted parameter `A = P0`, one KPI `K_out = K0`::

    K_out(q+1) = alpha * A_q + theta * Z_q + c        alpha = -1, theta = 2.5, c = 0

The latent `Z` is neither a parameter nor a KPI. It drives the behaviour action and the
outcome, so the observed adjacency holds only `P0 -> K0`; the two latent edges are listed in
`LATENT_EDGES`.

Two collection modes:

* `"obs"`: `A = clip(0.5 + lam * Z + eta, 0, 1)` with `eta ~ N(0, eta_scale^2)`, the policy
  that confounds action and outcome.
* `"do"`: `A` is a randomised draw from the do-grid, independent of `Z`.

`Z`, `eta` and the do-action each use their own tape namespace. Pending (`Z`) and committed
(`prev_Z`) latents are carried through `snapshot` and `restore`.
"""

from typing import ClassVar

import numpy as np

from oran_causal.sim.controlled.base import ControlledEnv, Edges, Snapshot

MU_OUT = -0.5
SIGMA_OUT = float(np.sqrt(1.0 / 12.0))
TRUE_ADJACENCY: Edges = [(0, 0)]
LATENT_EDGES = [("Z", "A_behavior"), ("Z", "K_out")]
MODE_OBS = "obs"
MODE_DO = "do"
DO_GRID_N = 101


class E4Env(ControlledEnv):
    """Single-outcome linear-Gaussian SCM with a latent confounder."""

    num_params = 1
    num_kpis = 1
    id_ranges: ClassVar[list[tuple[float, float]]] = [(0.0, 1.0)]
    mu = np.array([MU_OUT], dtype=float)
    sigma = np.array([SIGMA_OUT], dtype=float)
    kpi_thresholds = (0.0,)
    directions = (0,)
    xapp_kpi_indices: ClassVar[list[tuple[int, ...]]] = [(0,)]
    kpi_to_xapp: ClassVar[dict[int, int]] = {0: 0}
    xapp_param_indices: ClassVar[list[tuple[int, ...]]] = [(0,)]
    panel_kpi_ids = (0,)
    MODE_OBS = MODE_OBS
    MODE_DO = MODE_DO
    _NS_LATENT = 3
    _NS_ETA = 4
    _NS_DOACT = 5

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        *,
        lam: float = 1.0,
        alpha: float = -1.0,
        theta: float = 2.5,
        c: float = 0.0,
        eta_scale: float = 0.5,
        mode: str = MODE_DO,
        episode: int = 0,
    ) -> None:
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.c = float(c)
        self.eta_scale = float(eta_scale)
        if mode not in (MODE_OBS, MODE_DO):
            raise ValueError(f"mode must be {MODE_OBS!r} or {MODE_DO!r}, got {mode!r}")
        self.mode = mode
        self.Z: float
        self.prev_Z: float
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def _draw_Z(self, time: int) -> float:
        return float(self._coord_rng(self._NS_LATENT, 0, int(time)).standard_normal())

    def _draw_eta(self, time: int) -> float:
        return float(self._coord_rng(self._NS_ETA, 0, int(time)).standard_normal()) * self.eta_scale

    def _draw_do_action(self, time: int) -> float:
        idx = int(self._coord_rng(self._NS_DOACT, 0, int(time)).integers(0, DO_GRID_N))
        return idx / (DO_GRID_N - 1)

    def reset(self, episode: int | None = None) -> dict[str, np.ndarray]:
        """Re-initialises and draws the latent for time 0."""
        super().reset(episode)
        self.Z = self._draw_Z(self.time)
        self.prev_Z = self.Z
        return self.latent_state()

    def behavior_action(self) -> float:
        """Returns the clipped confounded behaviour action at the pending coordinate."""
        eta = self._draw_eta(self.time)
        return float(np.clip(0.5 + self.lam * self.Z + eta, 0.0, 1.0))

    def generate_action(self) -> float:
        """Returns the action the current mode logs: confounded in `"obs"`, a do-draw in `"do"`."""
        if self.mode == MODE_OBS:
            return self.behavior_action()
        return self._draw_do_action(self.time)

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        k_out = self.alpha * p[0] + self.theta * self.prev_Z + self.c
        return np.array([k_out], dtype=float)

    def advance(self) -> np.ndarray:
        """Computes the outcome from the committed action and latent, then commits both."""
        new_kpis = np.asarray(self._update_kpis(self.prev_params, self.prev_kpis), dtype=float)
        self.prev_params = self.params.copy()
        self.prev_Z = self.Z
        self.prev_kpis = new_kpis
        self.time += 1
        self.Z = self._draw_Z(self.time)
        return self.prev_kpis.copy()

    def score_interventional_mean(self, a: float) -> float:
        """Returns `E_Z[K_out | do(A=a)]`, with `Z` integrated to its mean 0."""
        return self.alpha * float(a) + self.theta * 0.0 + self.c

    def snapshot(self) -> Snapshot:
        """Saves the base state plus the pending and committed latents."""
        return (*super().snapshot(), float(self.Z), float(self.prev_Z))

    def restore(self, snap: Snapshot) -> None:
        """Restores a snapshot, including the latents when present."""
        super().restore(snap)
        if len(snap) >= 7:
            self.Z = float(snap[5])
            self.prev_Z = float(snap[6])

    @property
    def adjacency_edges(self) -> Edges:
        """The observed edge `P0 -> K0`; the latent edges are in `LATENT_EDGES`."""
        return [(0, 0)]

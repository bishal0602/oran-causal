"""Base class of the controlled environments.

Every exogenous value (initial parameters, process noise, observation noise) is a pure
function of the coordinate `(env_seed, episode, time, namespace, variable)`: a fresh
generator is derived from a `SeedSequence` over those coordinates for each draw, so two
arms that take different actions see byte-identical exogenous draws. Actions follow an
apply/advance split with one-step actuation latency.
"""

from typing import ClassVar

import numpy as np

Snapshot = tuple
Edges = list[tuple[int, int]]


class ControlledEnv:
    """Environment with a coordinate-keyed exogenous tape.

    Subclasses define `num_params`, `num_kpis`, `id_ranges` and `_update_kpis`.

    Attributes:
        env_seed: Seed of the exogenous tape.
        obs_noise_scale: Standard deviation of the observation noise; 0 disables it.
        episode: Episode coordinate of the tape.
        time: Logical time coordinate.
        params: Pending parameter vector.
        prev_params: Committed parameter vector.
        prev_kpis: Committed latent KPI vector.
    """

    num_params: int
    num_kpis: int
    id_ranges: ClassVar[list[tuple[float, float]]]

    _NS_INIT = 0
    _NS_PROC = 1
    _NS_OBS = 2

    def __init__(self, env_seed: int = 0, obs_noise_scale: float = 0.0, episode: int = 0) -> None:
        self.env_seed = int(env_seed)
        self.obs_noise_scale = float(obs_noise_scale)
        self.episode = int(episode)
        self.time = 0
        self.params: np.ndarray
        self.prev_params: np.ndarray
        self.prev_kpis: np.ndarray
        self.reset()

    def _coord_rng(self, namespace: int, variable: int, time: int) -> np.random.Generator:
        entropy = tuple(
            int(c) & 0xFFFFFFFFFFFFFFFF
            for c in (self.env_seed, self.episode, int(time), int(namespace), int(variable))
        )
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def _process_noise(self, variable: int, time: int) -> float:
        return float(self._coord_rng(self._NS_PROC, variable, time).standard_normal())

    @property
    def adjacency_edges(self) -> Edges:
        """True `(kpi_index, source_index)` edges; sources at or above `num_params` are KPIs."""
        raise NotImplementedError

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def true_adj_matrix(self) -> np.ndarray:
        """Returns the true adjacency matrix, rows indexed by `num_params + kpi_index`."""
        size = self.num_params + self.num_kpis
        matrix = np.zeros((size, size), dtype=np.float32)
        for kpi_index, source_index in self.adjacency_edges:
            matrix[self.num_params + kpi_index, source_index] = 1.0
        return matrix

    def reset(self, episode: int | None = None) -> dict[str, np.ndarray]:
        """Re-initialises at `time = 0` with initial parameters drawn from the tape.

        Args:
            episode: New episode coordinate; keeps the current one when None.

        Returns:
            The latent state.
        """
        if episode is not None:
            self.episode = int(episode)
        self.time = 0
        self.params = np.array(
            [
                self._coord_rng(self._NS_INIT, i, 0).uniform(low, high)
                for i, (low, high) in enumerate(self.id_ranges)
            ],
            dtype=float,
        )
        self.prev_params = self.params.copy()
        self.prev_kpis = np.zeros(self.num_kpis, dtype=float)
        return self.latent_state()

    def apply_action(self, param_id: int, value: float) -> None:
        """Writes one coordinate of the pending parameter vector without advancing."""
        self.params[int(param_id)] = float(value)

    def advance(self) -> np.ndarray:
        """Computes the KPIs from the committed state, commits the parameters, ticks time.

        Returns:
            The new latent KPI vector.
        """
        new_kpis = np.asarray(self._update_kpis(self.prev_params, self.prev_kpis), dtype=float)
        self.prev_params = self.params.copy()
        self.prev_kpis = new_kpis
        self.time += 1
        return self.prev_kpis.copy()

    def step(self, param_id: int, value: float) -> np.ndarray:
        """Applies one action and advances."""
        self.apply_action(param_id, value)
        return self.advance()

    def neutral_step(self) -> np.ndarray:
        """Advances with the parameters held."""
        return self.advance()

    def latent_kpis(self) -> np.ndarray:
        """Returns the noiseless latent KPI vector."""
        return self.prev_kpis.copy()

    def observed_kpis(self) -> np.ndarray:
        """Returns the latent KPIs plus observation noise keyed on the current time."""
        kpis = self.prev_kpis.copy()
        if self.obs_noise_scale > 0.0:
            eps = np.array(
                [
                    self._coord_rng(self._NS_OBS, v, self.time).normal(0.0, self.obs_noise_scale)
                    for v in range(self.num_kpis)
                ],
                dtype=float,
            )
            kpis = kpis + eps
        return kpis

    def latent_state(self) -> dict[str, np.ndarray]:
        """Returns the committed parameters and latent KPIs."""
        return {"params": self.prev_params.copy(), "kpis": self.prev_kpis.copy()}

    def snapshot(self) -> Snapshot:
        """Saves the pending and committed state and the `(episode, time)` coordinates."""
        return (
            self.params.copy(),
            self.prev_params.copy(),
            self.prev_kpis.copy(),
            self.episode,
            self.time,
        )

    def restore(self, snap: Snapshot) -> None:
        """Restores a `snapshot`; a 3-tuple of arrays leaves `(episode, time)` unchanged."""
        params, prev_params, prev_kpis = snap[0], snap[1], snap[2]
        self.params = params.copy()
        self.prev_params = prev_params.copy()
        self.prev_kpis = prev_kpis.copy()
        if len(snap) >= 5:
            self.episode = int(snap[3])
            self.time = int(snap[4])

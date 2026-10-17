"""E5: composed shared-knob conflict.

A shared knob `P0` whose harmful effect on `K_harm` is subdominant and gated, composed with a
two-hop chain through a conduit KPI, a latent confounder and optional noise. Each layer has
its own constructor switch: `subdom` (gated direct edge), `chain_gamma` (gated chain edge),
`theta` with `mode` and `lam` (confounder), `obs_noise_scale` and `process_noise` (noise).

Mechanisms with one-step lag, `K_mid` being the previous-step conduit::

    K_ben  = A_BEN * exp(-(P0 - MU_BEN)^2 / (2 W_BEN^2))
    K_mid  = C_CHAIN * P0
    K_harm = base(G1, G2) + [gate] * (subdom * (P0 - C0) + chain_gamma * K_mid_prev)
             + theta * Z_prev
    K_dist = A_DIST * exp(-(P3 - MU_DIST)^2 / (2 W_DIST^2))

    base(G1, G2) = A_BASE * exp(-(G2 - MU_BASE)^2 / (2 W_BASE^2)) + B1 * G1
    gate(G1, G2) = 1[G1 <= TAU1 and |G2 - C2| <= W_GATE]

The decision panel is `K_ben` (satisfy above), `K_harm` (satisfy below) and `K_dist`
(satisfy above); `K_mid` is an excluded conduit. The confounder `Z` is latent: it drives
`P0` in `"obs"` mode and `K_harm`.
"""

from math import exp
from typing import ClassVar

import numpy as np

from oran_causal.sim.controlled.base import ControlledEnv, Edges, Snapshot

MODE_OBS = "obs"
MODE_DO = "do"
LAM_DEFAULT = 1.0
ETA_SCALE = 0.5
BEHAVIOR_P0_SCALE = 30.0


def _gate(g1: float, g2: float, tau1: float, c2: float, w_gate: float) -> bool:
    return (g1 <= tau1) and (abs(g2 - c2) <= w_gate)


class E5Env(ControlledEnv):
    """Four parameters `[P0, G1, G2, P3]` and four KPIs `[K_ben, K_harm, K_mid, K_dist]`."""

    num_params = 4
    num_kpis = 4
    P0, G1, G2, P3 = 0, 1, 2, 3
    K_BEN, K_HARM, K_MID, K_DIST = 0, 1, 2, 3
    id_ranges: ClassVar[list[tuple[float, float]]] = [
        (-100.0, 100.0),
        (1.5, 4.0),
        (-100.0, 150.0),
        (-20.0, 20.0),
    ]
    A_BEN, MU_BEN, W_BEN = 12.0, 80.0, 40.0
    C_CHAIN = 1.0
    A_BASE, MU_BASE, W_BASE, B1 = 72.0, 25.0, 25.0, 2.0
    TAU1, C2, W_GATE = 2.75, 25.0, 12.6
    C0 = 0.0
    A_DIST, MU_DIST, W_DIST = 70.0, 0.0, 10.0
    kpi_thresholds = (55.0, 80.0, 0.0, 55.0)
    directions = (0, 1, 0, 0)
    xapp_kpi_indices: ClassVar[list[tuple[int, ...]]] = [(0,), (1,), (2,), (3,)]
    kpi_to_xapp: ClassVar[dict[int, int]] = {0: 0, 1: 1, 2: 2, 3: 3}
    panel_kpi_ids = (0, 1, 3)
    excluded_kpi_ids = (2,)
    MODE_OBS = MODE_OBS
    MODE_DO = MODE_DO
    _NS_LATENT = 3
    _NS_ETA = 4
    _NS_DOACT = 5
    _DO_GRID = np.linspace(-100.0, 100.0, 201)

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        *,
        subdom: float = 0.20,
        chain_gamma: float = 0.20,
        theta: float = -2.5,
        lam: float = LAM_DEFAULT,
        mode: str = MODE_DO,
        process_noise: bool = False,
        episode: int = 0,
    ) -> None:
        self.subdom = float(subdom)
        self.chain_gamma = float(chain_gamma)
        self.theta = float(theta)
        self.lam = float(lam)
        if mode not in (MODE_OBS, MODE_DO):
            raise ValueError(f"mode must be {MODE_OBS!r} or {MODE_DO!r}, got {mode!r}")
        self.mode = mode
        self.process_noise = bool(process_noise)
        self.Z: float
        self.prev_Z: float
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def _draw_Z(self, time: int) -> float:
        return float(self._coord_rng(self._NS_LATENT, 0, int(time)).standard_normal())

    def _draw_eta(self, time: int) -> float:
        return float(self._coord_rng(self._NS_ETA, 0, int(time)).standard_normal()) * ETA_SCALE

    def _draw_do_p0(self, time: int) -> float:
        idx = int(self._coord_rng(self._NS_DOACT, 0, int(time)).integers(0, len(self._DO_GRID)))
        return float(self._DO_GRID[idx])

    def reset(self, episode: int | None = None) -> dict[str, np.ndarray]:
        """Re-initialises and draws the latent for time 0."""
        state = super().reset(episode=episode)
        self.Z = self._draw_Z(self.time)
        self.prev_Z = self.Z
        return state

    def behavior_p0(self) -> float:
        """Returns the confounded behaviour `P0`, clipped to its range."""
        eta = self._draw_eta(self.time)
        raw = 0.0 + self.lam * BEHAVIOR_P0_SCALE * self.Z + eta
        lo, hi = self.id_ranges[self.P0]
        return float(np.clip(raw, lo, hi))

    def generate_p0(self) -> float:
        """Returns the logged `P0`: confounded in `"obs"`, a randomised do-draw in `"do"`."""
        return self.behavior_p0() if self.mode == MODE_OBS else self._draw_do_p0(self.time)

    def _base(self, g1: float, g2: float) -> float:
        return (
            self.A_BASE * exp(-((g2 - self.MU_BASE) ** 2) / (2.0 * self.W_BASE**2)) + self.B1 * g1
        )

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        p0, g1, g2, p3 = p[self.P0], p[self.G1], p[self.G2], p[self.P3]
        k_ben = self.A_BEN * exp(-((p0 - self.MU_BEN) ** 2) / (2.0 * self.W_BEN**2))
        k_mid = self.C_CHAIN * p0
        k_harm = self._base(g1, g2)
        if _gate(g1, g2, self.TAU1, self.C2, self.W_GATE):
            k_harm += self.subdom * (p0 - self.C0)
            k_harm += self.chain_gamma * k[self.K_MID]
        k_harm += self.theta * self.prev_Z
        k_dist = self.A_DIST * exp(-((p3 - self.MU_DIST) ** 2) / (2.0 * self.W_DIST**2))
        out = np.array([k_ben, k_harm, k_mid, k_dist], dtype=float)
        if self.process_noise:
            out = out + np.array(
                [self._process_noise(v, self.time) for v in range(self.num_kpis)], dtype=float
            )
        return out

    def advance(self) -> np.ndarray:
        """Computes the KPIs from the committed state, then commits parameters and latent."""
        new_kpis = np.asarray(self._update_kpis(self.prev_params, self.prev_kpis), dtype=float)
        self.prev_params = self.params.copy()
        self.prev_Z = self.Z
        self.prev_kpis = new_kpis
        self.time += 1
        self.Z = self._draw_Z(self.time)
        return self.prev_kpis.copy()

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
        """Active `(kpi_index, source_index)` edges; layer edges follow their coefficient."""
        edges = [
            (self.K_BEN, self.P0),
            (self.K_MID, self.P0),
            (self.K_HARM, self.G1),
            (self.K_HARM, self.G2),
            (self.K_DIST, self.P3),
        ]
        if self.subdom != 0.0:
            edges.append((self.K_HARM, self.P0))
        if self.chain_gamma != 0.0:
            edges.append((self.K_HARM, self.num_params + self.K_MID))
        return edges

    @property
    def latent_edges(self) -> list[tuple[str, str]]:
        """Latent-confounder edges; empty when the confound layer is off."""
        if self.theta == 0.0:
            return []
        edges = [("Z", "K_harm")]
        if self.mode == MODE_OBS and self.lam != 0.0:
            edges.append(("Z", "P0_behavior"))
        return edges

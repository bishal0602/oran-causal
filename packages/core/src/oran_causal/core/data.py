"""Row-corpus contract shared by dataset generators, methods and scorers.

A `Dataset` is one (world, regime, n, seed) corpus of one-step rows. A `Method` maps a
`Dataset` to a `Result` with one entry per candidate edge. Ground truth never enters a
`Dataset`; it lives in `Truth` and is read only by the scorer.
"""

import dataclasses
from typing import Any, Protocol

import numpy as np

WORLDS = ("E1", "E2", "E3", "E4", "E5")
REGIMES = ("R1", "R2", "R3", "R4")


@dataclasses.dataclass(frozen=True)
class Design:
    """Known assignment design of one action column.

    Attributes:
        kind: `"iid"` (column drawn i.i.d. from `dist`, R1), `"dither"` (setpoint plus an
            i.i.d. dither; `fixed_part` holds the setpoint and `random_part` the realised
            dither, R2), `"logged"` (context-dependent policy with per-row probabilities or
            densities in `propensity`, R3) or `"none"` (no known design: R4 latent
            confounding, or a non-action column).
        dist: Distribution of the random part, for example
            `{"name": "uniform", "lo": 0, "hi": 1}` or
            `{"name": "categorical", "values": [...], "p": [...]}`.
        random_part: Realised randomised component of the column.
        fixed_part: Non-randomised component of the column.
        propensity: Logged per-row probabilities or densities.
        redraw_seed_tag: Tag that keys the redraw RNG stream of this column.
    """

    kind: str
    dist: dict[str, Any] | None = None
    random_part: np.ndarray | None = None
    fixed_part: np.ndarray | None = None
    propensity: np.ndarray | None = None
    redraw_seed_tag: int | None = None


@dataclasses.dataclass(frozen=True)
class Dataset:
    """One corpus of one-step rows.

    Attributes:
        world: Environment id, one of `WORLDS`.
        regime: Data regime id, one of `REGIMES`.
        n: Number of rows.
        seed: Generator seed.
        action_names: Treatment column names.
        kpi_names: KPI column names.
        X_action: Actions at t, shape `[n, p]`.
        X_kpi_lag: KPIs at t, shape `[n, k]`; all-NaN if the world has none.
        Y: KPIs at t+1, shape `[n, k]`.
        designs: One `Design` per action column.
        candidates: Scored (source, target) pairs; sources are actions or lagged KPIs.
        time_index: Row time, shape `[n]`, for temporal methods.
        context: Observed context, shape `[n, c]` (R3), else None.
        meta: Free-form generator metadata.
    """

    world: str
    regime: str
    n: int
    seed: int
    action_names: tuple[str, ...]
    kpi_names: tuple[str, ...]
    X_action: np.ndarray
    X_kpi_lag: np.ndarray
    Y: np.ndarray
    designs: tuple[Design, ...]
    candidates: tuple[tuple[str, str], ...]
    time_index: np.ndarray | None = None
    context: np.ndarray | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Truth:
    """Ground truth of one (world, regime) pair.

    Attributes:
        world: Environment id.
        regime: Data regime id.
        edges: True (source, target) edges among the candidates.
        signs: +1 or -1 for edges with a defined sign.
        null_edges: Candidates with exactly zero effect.
    """

    world: str
    regime: str
    edges: frozenset[tuple[str, str]]
    signs: dict[tuple[str, str], int]
    null_edges: frozenset[tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class EdgeResult:
    """Score of one candidate edge.

    Attributes:
        source: Source variable.
        target: Target variable.
        score: Raw edge score, larger is stronger; NaN if not scorable.
        p: P-value if the method has one.
        sign: +1, -1 or 0 (unsigned).
        declared: Whether the method declares the edge.
    """

    source: str
    target: str
    score: float
    p: float | None
    sign: int
    declared: bool


@dataclasses.dataclass(frozen=True)
class Result:
    """Output of one `Method.run` call."""

    method: str
    version: str
    edges: tuple[EdgeResult, ...]
    cpu_s: float
    config: dict[str, Any]
    notes: dict[str, Any] = dataclasses.field(default_factory=dict)


class Method(Protocol):
    """A discovery method over `Dataset` corpora."""

    name: str
    version: str

    def tune(self, dev: list[Dataset], truth_free: bool = True) -> dict[str, Any]:
        """Returns a frozen config computed from DEV datasets only."""
        ...

    def run(self, data: Dataset, config: dict[str, Any]) -> Result:
        """Scores every candidate edge of `data` under `config`."""
        ...

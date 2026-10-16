"""Row-corpus contract shared by dataset generators, methods and scorers.

A `Dataset` is one (world, regime, n, seed) corpus of one-step rows. A `Method` maps a
`Dataset` to a `Result` with one entry per candidate edge. Ground truth never enters a
`Dataset`; it lives in `Truth` and is read only by the scorer. A `UnitTable` is the unit-level
corpus of the system-level studies: one row per logged decision unit.
"""

import dataclasses
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

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


@dataclasses.dataclass(frozen=True)
class UnitTable:
    """Pooled unit table of logged decision units, one row per unit.

    A unit's target for relation r and KPI k is the KPI summed over r's cells in the H seconds
    from the unit's opening minus the same sum over the H_pre seconds before it.

    Attributes:
        episode: Episode index of each row (position among the pooled episodes).
        seed: Protocol seed of the row's episode.
        fold: Evaluation fold of the row's episode, -1 if none.
        family: Index into `families` of the unit's knob family.
        xapp: xApp that opened the unit (object array of str).
        c: Cell of the unit.
        t0: Opening second.
        mode: Logged mode, an index into the mode list of the logging policy.
        applied: Applied mode index (differs from `mode` under a placebo policy).
        p: Logged propensity of the logged mode.
        probs: Logged mode-probability row of the unit, shape `[n, modes]`.
        step: Proposed minus current value of the opening request.
        sgn: sign(step), with 0 mapped to +1.
        y: Target per (relation, KPI), each of shape `[n]`.
        pre: Pre-window sum per (relation, KPI), each of shape `[n]`.
        ycell: Per-cell target per KPI, each of shape `[n, n_cells]`.
        rel_mask: Cells of each relation per row, each of shape `[n, n_cells]`.
        z: Conditioner columns by name, each of shape `[n]`.
        prev: Row of the previous unit of the same (episode, cell, xApp, family), -1 if none.
        n_cells: Number of cells.
        H: Post-window length in seconds.
        H_pre: Pre-window length in seconds.
        relations: Relation names, in the order of `y`.
        kpis: KPI names, in the order of `y`.
        families: Knob family names indexed by `family`.
        levels: Treatment dose of each mode index.
        meta: Builder metadata (episodes, dropped units, checks).
    """

    episode: np.ndarray
    seed: np.ndarray
    fold: np.ndarray
    family: np.ndarray
    xapp: np.ndarray
    c: np.ndarray
    t0: np.ndarray
    mode: np.ndarray
    applied: np.ndarray
    p: np.ndarray
    probs: np.ndarray
    step: np.ndarray
    sgn: np.ndarray
    y: dict[tuple[str, str], np.ndarray]
    pre: dict[tuple[str, str], np.ndarray]
    ycell: dict[str, np.ndarray]
    rel_mask: dict[str, np.ndarray]
    z: dict[str, np.ndarray]
    prev: np.ndarray
    n_cells: int
    H: int
    H_pre: int
    relations: tuple[str, ...]
    kpis: tuple[str, ...]
    families: tuple[str, ...]
    levels: np.ndarray
    meta: dict[str, Any]

    @property
    def n(self) -> int:
        """Number of units."""
        return len(self.t0)

    @property
    def level(self) -> np.ndarray:
        """Dose of each unit's logged mode."""
        return self.levels[self.mode]

    @property
    def x(self) -> np.ndarray:
        """Oriented treatment: dose times the request direction."""
        return self.level * self.sgn

    def rows_of(self, family: str) -> np.ndarray:
        """Returns the rows of one knob family."""
        return np.nonzero(self.family == self.families.index(family))[0]

    def subset(self, rows: npt.ArrayLike) -> UnitTable:
        """Returns the table restricted to `rows` (indices or a boolean mask), `prev` remapped."""
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.nonzero(rows)[0]
        pos = np.full(self.n, -1)
        pos[rows] = np.arange(len(rows))
        prev = self.prev[rows]
        prev = np.where(prev >= 0, pos[np.maximum(prev, 0)], -1)
        pick = {
            f.name: getattr(self, f.name)[rows]
            for f in dataclasses.fields(self)
            if f.name not in ("prev", "levels") and isinstance(getattr(self, f.name), np.ndarray)
        }
        return dataclasses.replace(
            self,
            **pick,
            prev=prev,
            y={k: v[rows] for k, v in self.y.items()},
            pre={k: v[rows] for k, v in self.pre.items()},
            ycell={k: v[rows] for k, v in self.ycell.items()},
            rel_mask={k: v[rows] for k, v in self.rel_mask.items()},
            z={k: v[rows] for k, v in self.z.items()},
            meta=dict(self.meta),
        )

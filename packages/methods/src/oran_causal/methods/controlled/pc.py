"""The PC algorithm (Spirtes, Glymour and Scheines 2000) via causal-learn.

causal-learn pieces: skeleton discovery, background-knowledge orientation, `uc_sepset`, Meek;
defaults alpha = .05, stable, uc_rule 0, uc_priority 2; CI test Fisher-z or KCI.

Nodes: the actions at t, the arm's base covariate set (no focal) and the KPIs at t+1.
Arm `"native"`: lagged KPIs (kind L) and context (kind C). Arm `"eq"`: also setpoints and
lagged actions as exogenous tier-0 nodes (kind X); rows outside the row mask are dropped and
an X column equal to an earlier node is dropped. Background knowledge: tier 0 (t) precedes
tier 1 (t+1); nothing but context and X enters an action; an action enters no L, C or X
node; nothing from L or Y enters X. The primary graph has no `P_placebo_conf` node; its
own candidates come from a second run with every node. In arm eq a target that is an exact
linear fit of the source's conditioning set is not testable.

Score: -log10 of the largest CI p-value of the pair over the skeleton tests (pcalg `pMax`,
Kalisch et al. 2012, JSS 47(11)); native = adjacency at alpha. A singular Fisher-z
correlation sub-matrix is recomputed with the pseudo-inverse and counted. Sign: partial
correlation given the arm's conditioning set.
"""

import dataclasses
import importlib.metadata
from math import log, sqrt
from typing import Any, ClassVar

import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from scipy.stats import norm

from oran_causal.core.covariates import CovariateSpec, design_covariates
from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._classic import (
    PLACEBO_CONF,
    ClassicBase,
    Columns,
    Scored,
    columns,
    cond_set,
    exact_fit,
    is_diagnostic_column,
    not_testable_notes,
    resolve,
    sign_pcorr_given_z,
    source_column,
    target_index,
)


class _RecordingCIT:
    """Wraps a causal-learn CIT: max p-value per pair, pinv fallback for a singular Fisher-z."""

    def __init__(self, cit: Any, data: np.ndarray):
        self.cit = cit
        self.method = cit.method
        self.maxp: dict[tuple[int, int], float] = {}
        self.n_tests = 0
        self.fallbacks = 0
        self._corr: np.ndarray | None = None
        self._n = data.shape[0]
        self._data = data

    def _fisherz_pinv(self, x: int, y: int, s: tuple[int, ...]) -> float:
        if self._corr is None:
            self._corr = np.corrcoef(self._data.T)
        var = [int(x), int(y)] + [int(v) for v in s]
        inv = np.linalg.pinv(self._corr[np.ix_(var, var)])
        den = sqrt(abs(inv[0, 0] * inv[1, 1]))
        r = 0.0 if den == 0 else -inv[0, 1] / den
        if abs(r) >= 1:
            r = (1.0 - np.finfo(float).eps) * np.sign(r)
        z = 0.5 * log((1 + r) / (1 - r))
        stat = sqrt(max(self._n - len(s) - 3, 1)) * abs(z)
        return float(2 * (1 - norm.cdf(abs(stat))))

    def __call__(self, x: int, y: int, s: Any = None, *args: Any) -> float:
        s = () if s is None else tuple(s)
        try:
            p = float(self.cit(x, y, s))
        except ValueError:
            if self.method != "fisherz":
                raise
            self.fallbacks += 1
            p = self._fisherz_pinv(x, y, s)
        self.n_tests += 1
        key = (min(int(x), int(y)), max(int(x), int(y)))
        self.maxp[key] = max(self.maxp.get(key, -1.0), p)
        return p


@dataclasses.dataclass(frozen=True)
class PcSettings:
    """Settings of one PC search.

    Attributes:
        alpha: CI test level.
        indep_test: causal-learn CI test name.
        stable: Order-independent skeleton.
        uc_rule: Unshielded-collider rule (only 0 is wired).
        uc_priority: Collider conflict priority.
        max_k: Maximal conditioning set size.
    """

    alpha: float = 0.05
    indep_test: str = "fisherz"
    stable: bool = True
    uc_rule: int = 0
    uc_priority: int = 2
    max_k: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PcSettings:
        """Settings from a method configuration."""
        return cls(
            config["alpha"],
            config["indep_test"],
            config["stable"],
            config["uc_rule"],
            config["uc_priority"],
            config.get("max_k"),
        )


def run_pc(
    d: np.ndarray,
    names: list[str],
    tiers: list[int],
    forbidden: set[tuple[int, int]],
    settings: PcSettings,
) -> tuple[Any, _RecordingCIT]:
    """causal-learn PC with tier and forbidden-edge background knowledge.

    Returns:
        (causal graph, CI recorder).

    Raises:
        ValueError: `uc_rule` is not 0.
    """
    rec = _RecordingCIT(CIT(d, settings.indep_test), d)
    nodes = CausalGraph(d.shape[1], names).G.nodes
    bk = BackgroundKnowledge()
    for i, t in enumerate(tiers):
        bk.add_node_to_tier(nodes[i], t)
    for i, j in forbidden:
        bk.add_forbidden_by_node(nodes[i], nodes[j])
    cg = SkeletonDiscovery.skeleton_discovery(
        d,
        settings.alpha,
        rec,
        settings.stable,
        background_knowledge=bk,
        verbose=False,
        show_progress=False,
        node_names=names,
        max_k=settings.max_k,
    )
    orient_by_background_knowledge(cg, bk)
    if settings.uc_rule != 0:
        raise ValueError("only uc_rule = 0 (causal-learn default) is wired")
    cg = UCSepset.uc_sepset(cg, settings.uc_priority, background_knowledge=bk)
    cg = Meek.meek(cg, background_knowledge=bk)
    return cg, rec


def forbidden_edges(kinds: list[str]) -> set[tuple[int, int]]:
    """Forbidden directed edges (i, j) between node kinds A, L, C, X and Y."""
    out = set()
    for i, ki in enumerate(kinds):
        for j, kj in enumerate(kinds):
            if i == j:
                continue
            if (
                (kj == "A" and ki in ("L", "Y", "A"))
                or (ki == "A" and kj in ("L", "C", "X"))
                or (kj == "X" and ki in ("L", "Y"))
            ):
                out.add((i, j))
    return out


@dataclasses.dataclass(frozen=True)
class _Base:
    names: tuple[str, ...]
    kinds: list[str]
    H: np.ndarray
    mask: np.ndarray


@dataclasses.dataclass
class _Nodes:
    names: list[str] = dataclasses.field(default_factory=list)
    mats: list[np.ndarray] = dataclasses.field(default_factory=list)
    kinds: list[str] = dataclasses.field(default_factory=list)
    eq_dropped: list[str] = dataclasses.field(default_factory=list)

    def add(self, name: str, col: np.ndarray, kind: str) -> None:
        self.names.append(name)
        self.mats.append(col)
        self.kinds.append(kind)


def _nodes(data: Dataset, cols: Columns, base: _Base, with_conf: bool) -> _Nodes:
    nd = _Nodes()
    for i, a in enumerate(data.action_names):
        if a != PLACEBO_CONF or with_conf:
            nd.add(f"A:{a}", np.asarray(data.X_action, float)[base.mask, i], "A")
    for i, (nm, k) in enumerate(zip(base.names, base.kinds, strict=True)):
        if not with_conf and is_diagnostic_column(nm):
            continue
        c = base.H[:, i]
        if k == "X" and any(np.array_equal(c, m) for m in nd.mats):
            nd.eq_dropped.append(nm)
            continue
        nd.add(f"{k}:{nm}", c, k)
    for i, kp in enumerate(data.kpi_names):
        nd.add(f"Y:{kp}", cols.Y[base.mask, i], "Y")
    return nd


def _graph(
    data: Dataset, cols: Columns, settings: PcSettings, base: _Base, with_conf: bool
) -> dict[str, Any]:
    nd = _nodes(data, cols, base, with_conf)
    d = np.column_stack(nd.mats)
    keep = [i for i in range(d.shape[1]) if np.std(d[:, i]) > 0]
    idx = {nd.names[i]: k for k, i in enumerate(keep)}
    d, kn, kk = d[:, keep], [nd.names[i] for i in keep], [nd.kinds[i] for i in keep]
    tiers = [1 if k == "Y" else 0 for k in kk]
    cg, rec = run_pc(d, kn, tiers, forbidden_edges(kk), settings)
    return {
        "idx": idx,
        "G": cg.G.graph,
        "rec": rec,
        "n_vars": d.shape[1],
        "eq_dropped": nd.eq_dropped,
        "dropped_constant": [nd.names[i] for i in range(len(nd.names)) if i not in keep],
    }


def _base(data: Dataset, eq: bool) -> _Base:
    hn, h, mask = design_covariates(
        data, CovariateSpec(setpoints="all" if eq else "none", lagged_actions=eq)
    )
    if not eq:
        mask = np.ones(data.n, bool)
    kinds = [
        "L" if nm.startswith("lag_kpi:") else "C" if nm.startswith("ctx:") else "X" for nm in hn
    ]
    return _Base(tuple(hn), kinds, np.asarray(h, float)[mask], mask)


def _node_name(data: Dataset, cols: Columns, fam: str, j: int | None) -> str | None:
    if j is None:
        return None
    return f"A:{data.action_names[j]}" if fam == "action" else f"L:lag_kpi:{cols.lag_names[j]}"


def _notes(
    main: dict[str, Any], diag: dict[str, Any] | None, base: _Base, eq: bool, exact: list[str]
) -> dict:
    rec = main["rec"]
    notes: dict[str, Any] = {
        "sign_rule": "pcorr_given_Z",
        "n_ci_tests": rec.n_tests,
        "fisherz_pinv_fallbacks": rec.fallbacks,
        "fisherz_pinv_fallback_rate": rec.fallbacks / rec.n_tests if rec.n_tests else 0.0,
        "n_vars": main["n_vars"],
        "dropped_constant": main["dropped_constant"],
        "score": "-log10 max CI p over the pair's skeleton tests (pcalg pMax)",
    }
    if diag is not None:
        notes["diagnostic_graph"] = {
            "rule": f"R-37: {PLACEBO_CONF} candidates read from a second PC run with every node; "
            "primary graph without it",
            "n_vars": diag["n_vars"],
            "n_ci_tests": diag["rec"].n_tests,
        }
    if eq:
        notes.update(
            {
                "eq_covariates": [
                    nm
                    for nm, k in zip(base.names, base.kinds, strict=True)
                    if k == "X" and not is_diagnostic_column(nm)
                ],
                "eq_dropped": main["eq_dropped"],
                "n_rows_used": int(base.mask.sum()),
                **not_testable_notes(exact),
            }
        )
    return notes


class PC(ClassicBase):
    """PC arm (`"eq"`, `"native"`)."""

    name = "pc"
    version = f"causal-learn {importlib.metadata.version('causal-learn')}"
    uses_p = False
    method_idx = 1
    arms = ("eq", "native")
    defaults: ClassVar[dict[str, Any]] = {
        "alpha": 0.05,
        "indep_test": "fisherz",
        "stable": True,
        "uc_rule": 0,
        "uc_priority": 2,
        "max_k": None,
        "arm": "eq",
    }

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        eq = config["arm"] == "eq"
        settings = PcSettings.from_config(config)
        base = _base(data, eq)
        main = _graph(data, cols, settings, base, with_conf=False)
        has_conf = any(s == PLACEBO_CONF for s, _ in data.candidates)
        diag = _graph(data, cols, settings, base, with_conf=True) if has_conf else None
        out: list[Scored] = []
        exact: list[str] = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            g = diag if s == PLACEBO_CONF and diag is not None else main
            sname, tname = _node_name(data, cols, fam, j), f"Y:{data.kpi_names[ti]}"
            if j is None or sname not in g["idx"] or tname not in g["idx"]:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            if eq:
                cs = cond_set(data, fam, source_column(data, cols, fam, j)[1], "eq")
                if exact_fit(cols.Y[cs.mask, ti], cs.Z[cs.mask]):
                    exact.append(f"{s}->{t}")
                    out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                    continue
            a, b = g["idx"][sname], g["idx"][tname]
            mp = g["rec"].maxp.get((min(a, b), max(a, b)))
            score = float("nan") if mp is None else float(-np.log10(max(mp, 1e-300)))
            adj = bool(g["G"][a, b] != 0 or g["G"][b, a] != 0)
            sign = sign_pcorr_given_z(data, cols, (fam, j, ti), config["arm"])
            out.append(Scored(s, t, fam, score, sign, None, adj))
        return out, _notes(main, diag, base, eq, exact)

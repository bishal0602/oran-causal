"""Declaration layer: per-hypothesis statistics to declared edges.

The hypothesis space, the gate DAG, the graphical-Holm design and the p-filter groups are
inputs: this module holds no domain vocabulary.
"""

import dataclasses
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.core.multiplicity import (
    W_CAP,
    c_harm,
    cap_weights,
    dagger,
    e_bh,
    expected_rejections,
    graphical_holm,
    p_filter,
    projected_means,
    rw_weights,
    weighted_step_up,
)

FDR_LAYER_VERSION = "fdr-layer-v1"
PROCEDURES = (
    "by",
    "bh_invalid",
    "wby",
    "by1s",
    "wby1s",
    "dagger",
    "dagger1s",
    "gholm",
    "gholm1s",
    "pfilter_w1s",
    "ebh",
    "ebh1s",
    "webh1s",
)


@dataclasses.dataclass(frozen=True)
class _Spec:
    kind: str
    one_sided: bool
    weighted: bool
    structure: str | None


_SPEC = {
    "by": _Spec("p", False, False, None),
    "bh_invalid": _Spec("p", False, False, "bh"),
    "wby": _Spec("p", False, True, None),
    "by1s": _Spec("p", True, False, None),
    "wby1s": _Spec("p", True, True, None),
    "dagger": _Spec("p", False, False, "dagger"),
    "dagger1s": _Spec("p", True, False, "dagger"),
    "gholm": _Spec("p", False, False, "gholm"),
    "gholm1s": _Spec("p", True, False, "gholm"),
    "pfilter_w1s": _Spec("p", True, True, "pfilter"),
    "ebh": _Spec("e", False, False, None),
    "ebh1s": _Spec("e", True, False, None),
    "webh1s": _Spec("e", True, True, None),
}


@dataclasses.dataclass(frozen=True)
class HypStats:
    """Output of any statistic for one hypothesis.

    Attributes:
        p2: Two-sided p-value; NaN marks an untested hypothesis.
        sign: Sign of the estimate.
        p_plus: One-sided p-value for a positive effect; NaN if not available.
        p_minus: One-sided p-value for a negative effect; NaN if not available.
        e2: Two-sided e-value; NaN if not available.
        e_plus: One-sided e-value for a positive effect.
        e_minus: One-sided e-value for a negative effect.
        e_sign: Sign attached to the e-values.
    """

    p2: float = float("nan")
    sign: int = 0
    p_plus: float = float("nan")
    p_minus: float = float("nan")
    e2: float = float("nan")
    e_plus: float = float("nan")
    e_minus: float = float("nan")
    e_sign: int = 0


@dataclasses.dataclass(frozen=True)
class Prior:
    """Signed z-scores of the statistic on data independent of the evaluation data.

    Attributes:
        z: Prior z-score per hypothesis.
        n_prior: Number of episodes behind the prior.
        source: Free-form label of the prior data.
    """

    z: Mapping[Any, float]
    n_prior: int
    source: str = ""

    @classmethod
    def from_stats(
        cls, d: Mapping[str, Mapping[str, Any]], n_prior: int, source: str = ""
    ) -> Prior:
        """Builds a prior from `{"f|rel|kpi": {"z", "status"}}`; undetermined or NaN z gives 0."""
        z: dict[Hashable, float] = {}
        for k, v in d.items():
            tested = v.get("status", "tested") != "undetermined"
            x = v.get("z", float("nan")) if tested else float("nan")
            z[tuple(k.split("|"))] = float(x) if x is not None and np.isfinite(x) else 0.0
        return cls(z=z, n_prior=int(n_prior), source=source)


@dataclasses.dataclass(frozen=True)
class WeightConfig:
    """Prior-to-weight settings.

    Attributes:
        n_target: Number of episodes of the evaluation data.
        q: Target FDR level.
        floor: Share of the unweighted level every hypothesis keeps.
        cap: Maximum final weight.
    """

    n_target: int
    q: float = 0.05
    floor: float = 0.2
    cap: float = W_CAP


def prior_weights[H: Hashable](
    prior: Prior, hyps: Sequence[H], config: WeightConfig
) -> dict[H, float]:
    """Mean-1 declaration weights from an independent prior.

    w = floor + (1 - floor) * w_RW, where w_RW are Roeder-Wasserman weights of the projected
    prior means at the anticipated BY level, capped so that no final weight exceeds `cap`.

    Args:
        prior: Independent prior z-scores.
        hyps: Hypotheses in declaration order.
        config: Weight settings.

    Returns:
        Weight per hypothesis.
    """
    z = np.array([prior.z.get(h, 0.0) for h in hyps])
    xi = projected_means(z, prior.n_prior, config.n_target)
    k0 = expected_rejections(xi, config.q)
    w_rw = rw_weights(xi, config.q * k0 / c_harm(len(hyps)))
    if np.isfinite(config.cap):
        w_rw = cap_weights(w_rw, (config.cap - config.floor) / (1.0 - config.floor))
    w = config.floor + (1.0 - config.floor) * w_rw
    return dict(zip(hyps, w.tolist(), strict=True))


def prior_directions[H: Hashable](
    prior: Prior, hyps: Sequence[H], thr: float = 3.0
) -> dict[H, int]:
    """Prior direction per hypothesis: sign(z) when |z| >= thr, else 0."""
    return {
        h: (int(np.sign(prior.z.get(h, 0.0))) if abs(prior.z.get(h, 0.0)) >= thr else 0)
        for h in hyps
    }


@dataclasses.dataclass(frozen=True)
class Structure:
    """Hypothesis structure used by the structured procedures.

    Attributes:
        parents: Parent indices of every hypothesis (DAGGER).
        holm_weights: Initial weights of the graphical Holm design.
        holm_graph: Transition matrix of the graphical Holm design.
        groups: Group label of every hypothesis (p-filter).
    """

    parents: Sequence[Sequence[int]] | None = None
    holm_weights: np.ndarray | None = None
    holm_graph: np.ndarray | None = None
    groups: Sequence[object] | None = None


@dataclasses.dataclass(frozen=True)
class DeclareConfig:
    """Settings of one declaration call.

    Attributes:
        q: Target error level.
        prior: Independent prior; required by weighted and one-sided procedures.
        n_target: Evaluation episodes, for the weights.
        floor: Weight floor.
        thr: |z| threshold of a prior direction.
        structure: Structure of the structured procedures.
    """

    q: float = 0.05
    prior: Prior | None = None
    n_target: int | None = None
    floor: float = 0.2
    thr: float = 3.0
    structure: Structure = Structure()


def _one_value(s: HypStats, d: int, kind: str) -> tuple[float, int, int]:
    if kind == "p":
        two, plus, minus, sign = s.p2, s.p_plus, s.p_minus, s.sign
    else:
        two, plus, minus, sign = s.e2, s.e_plus, s.e_minus, s.e_sign
    v = {1: plus, -1: minus}.get(d, two)
    if d and not np.isfinite(v):
        v, d = two, 0
    return v, d if d else sign, d


def _vectors[H: Hashable](
    stats: Mapping[H, HypStats],
    dirs: Mapping[H, int],
    spec: _Spec,
    hyps: Sequence[H],
) -> tuple[np.ndarray, list[int], list[bool]]:
    vals, signs, used = [], [], []
    for h in hyps:
        d = dirs.get(h, 0) if spec.one_sided else 0
        v, sg, d_used = _one_value(stats.get(h, HypStats()), d, spec.kind)
        vals.append(v)
        signs.append(int(sg))
        used.append(bool(d_used))
    return np.array(vals, float), signs, used


def _required(value: Any, name: str, procedure: str) -> Any:
    if value is None:
        raise ValueError(f"{procedure} needs {name}")
    return value


def _decide(
    x: np.ndarray, w: np.ndarray, spec: _Spec, procedure: str, config: DeclareConfig
) -> np.ndarray:
    st, q = config.structure, config.q
    if spec.kind == "e":
        return e_bh(x, q, w if spec.weighted else None)
    if spec.structure is None:
        return weighted_step_up(x, w, q, reshape=True)
    if spec.structure == "bh":
        return weighted_step_up(x, None, q, reshape=False)
    if spec.structure == "dagger":
        return dagger(x, _required(st.parents, "parents", procedure), q, reshape=True)
    if spec.structure == "gholm":
        w0 = _required(st.holm_weights, "holm_weights", procedure)
        return graphical_holm(x, w0, _required(st.holm_graph, "holm_graph", procedure), q)
    return p_filter(x, _required(st.groups, "groups", procedure), w, q, reshape=True)


def _weights(spec: _Spec, hyps: Sequence[Hashable], config: DeclareConfig) -> np.ndarray:
    if not spec.weighted:
        return np.ones(len(hyps))
    prior = _required(config.prior, "a Prior (independent data)", "weighting")
    wc = WeightConfig(_required(config.n_target, "n_target", "weighting"), config.q, config.floor)
    wd = prior_weights(prior, hyps, wc)
    return np.array([wd[h] for h in hyps])


def declare[H: Hashable](
    stats: Mapping[H, HypStats],
    procedure: str,
    hyps: Sequence[H],
    config: DeclareConfig | None = None,
) -> dict:
    """Applies one procedure of `PROCEDURES` to per-hypothesis statistics.

    Args:
        stats: Statistics per hypothesis; a missing hypothesis is untested.
        procedure: One of `PROCEDURES`.
        hyps: Hypotheses in declaration order.
        config: Level, prior and structure; None uses the defaults.

    Returns:
        Per hypothesis: `declared`, `sign`, the p- or e-value, weight `w`, `one_sided`,
        `procedure` and `status` (`declared`, `not_detected` or `undetermined`).
    """
    config = config or DeclareConfig()
    spec = _SPEC[procedure]
    if (spec.one_sided or spec.weighted) and config.prior is None:
        raise ValueError(f"{procedure} needs a Prior (independent data)")
    dirs = (
        prior_directions(config.prior, hyps, config.thr) if spec.one_sided and config.prior else {}
    )
    w = _weights(spec, hyps, config)
    x, signs, used = _vectors(stats, dirs, spec, hyps)
    dec = _decide(x, w, spec, procedure, config)
    out = {}
    for i, h in enumerate(hyps):
        ok = bool(np.isfinite(x[i]))
        hit = bool(dec[i] and ok)
        out[h] = {
            "declared": hit,
            "sign": signs[i],
            spec.kind: float(x[i]),
            "w": float(w[i]),
            "one_sided": used[i],
            "procedure": procedure,
            "status": ("declared" if hit else "not_detected") if ok else "undetermined",
        }
    return out

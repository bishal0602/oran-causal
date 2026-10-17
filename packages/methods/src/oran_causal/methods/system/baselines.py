"""Discovery baselines on the system unit table (the same table as MSCR-CRT).

Every scorer returns a score >= 0 and a sign per hypothesis (f, rel, k) of the 60-edge space;
NaN = not scorable (the CRT's support rule or a degenerate target). Treatment x = level * sgn,
target y = the relation's KPI post - pre, pre = the pre-window value.

- corr: |Pearson corr(x, y)| over the family's units.
- granger: OLS post = a + b pre + c x, F(1, n - 3) test of c; score -log10 p. "granger_by" is BY
  at q = .05 over the scorable hypotheses (untuned).
- shap_gbdt (Sharma et al. 2025): HistGBDT + TreeSHAP per hypothesis on [x, sgn, pre]; score
  mean|SHAP(x)| / std(y), sign = sign(corr(x, y)).
- two_tower: the controlled two-tower with 4 family inputs and 15 targets, `TT_SIZES` (l1 0.5
  fixed a priori on synthetic data); score = the gate a[target, family].
- int (PACIFISTA, del Prever et al. 2025): INT distance of y * sgn between accepted and rejected
  units; sign = sign(mean_accept - mean_reject).
- qacm (Wadud et al. 2024 Sec. VI-B): own-cell KPI predictor (ANN or PR, the better held-out
  R^2); score = mean |pred(X) - pred(X | change 0)|. Own cell only.

Thresholds (`tune_tau`): "far_fpr" (primary) takes the (max_far + 1)-th largest DEV far score;
"physics_f1" maximises DEV F1 against the physics prior (sensitivity only).
"""

import dataclasses
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import f as fdist

from oran_causal.core.data import UnitTable
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.methods.system.published import _ANN, _PR, _r2, int_distance
from oran_causal.sim.system.decision.vocab import (
    FAMILIES,
    HYPOTHESES,
    KPIS,
    PHYSICS_PRIOR,
    RELATIONS,
    Hypothesis,
)

__all__ = [
    "DECL_RELATIONS",
    "MAX_FAR_DEV",
    "SCORERS",
    "SUPPORT",
    "TAU_RULES",
    "TT_SIZES",
    "TUNED",
    "Support",
    "Tuned",
    "declare",
    "desc_method",
    "granger_by",
    "granger_pvalues",
    "method_baseline",
    "run_baselines",
    "score_corr",
    "score_int",
    "score_qacm",
    "score_shap",
    "score_two_tower",
    "supported",
    "tune_tau",
]

TT_SIZES = {"d": 16, "r": 8, "hidden": 32, "epochs": 400, "lr": 0.01, "l1": 0.5}
MAX_FAR_DEV = 1
TAU_RULES = ("far_fpr", "physics_f1")
TUNED = ("shap_gbdt", "corr", "granger", "two_tower", "int", "qacm")

Scores = dict[Any, Any]


@dataclasses.dataclass(frozen=True)
class Support:
    """Support rule of a family (the CRT's)."""

    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3


SUPPORT = Support()


def supported(data: UnitTable, family: str, sup: Support = SUPPORT) -> bool:
    """True iff the family has enough units, accepted, rejected and episodes."""
    rows = data.rows_of(family)
    lv = data.level[rows]
    return (
        len(rows) >= sup.min_units
        and int((lv == 1.0).sum()) >= sup.min_accept
        and int((lv == 0.0).sum()) >= sup.min_reject
        and len(np.unique(data.episode[rows])) >= sup.min_episodes
    )


def _empty(relations: Sequence[str] = RELATIONS) -> Scores:
    return {h: {"score": float("nan"), "sign": 0} for h in HYPOTHESES if h[1] in relations}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _sign_of(c: float) -> int:
    return int(np.sign(c)) if np.isfinite(c) else 0


def _each(
    data: UnitTable, sup: Support, relations: Sequence[str] = RELATIONS
) -> Iterator[tuple[str, np.ndarray, Hypothesis]]:
    for f in FAMILIES:
        if not supported(data, f, sup):
            continue
        rows = data.rows_of(f)
        for r in relations:
            for k in KPIS:
                yield f, rows, (f, r, k)


def score_corr(data: UnitTable, sup: Support = SUPPORT) -> Scores:
    """|Pearson corr(x, y)| per hypothesis, sign = sign(corr)."""
    out = _empty()
    for _f, rows, h in _each(data, sup):
        c = _corr(data.x[rows], data.y[h[1:]][rows])
        if np.isfinite(c):
            out[h] = {"score": abs(c), "sign": int(np.sign(c))}
    return out


def granger_pvalues(data: UnitTable, sup: Support = SUPPORT) -> Scores:
    """Lagged F-test of x in post = a + b pre + c x; score -log10 p, sign = sign(c)."""
    out = _empty()
    for _f, rows, h in _each(data, sup):
        pre = data.pre[h[1:]][rows]
        post = data.y[h[1:]][rows] + pre
        x = data.x[rows]
        n = len(rows)
        A = np.column_stack([np.ones(n), pre, x])
        if np.std(post) == 0 or np.linalg.matrix_rank(A) < 3:
            continue
        beta, *_ = np.linalg.lstsq(A, post, rcond=None)
        res = post - A @ beta
        s2 = float(res @ res) / (n - 3)
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = float(np.sqrt(max(cov[2, 2], 0.0)))
        if se <= 0:
            continue
        F = (beta[2] / se) ** 2
        p = float(fdist.sf(F, 1, n - 3))
        out[h] = {"score": float(-np.log10(max(p, 1e-300))), "sign": int(np.sign(beta[2])), "p": p}
    return out


def granger_by(scores: Scores, q: float = 0.05) -> dict[Any, dict[str, Any]]:
    """Untuned Granger: BY at q over the scorable hypotheses."""
    keys = [h for h, v in scores.items() if np.isfinite(v["score"])]
    dec = (
        weighted_step_up(np.array([scores[h]["p"] for h in keys], float), None, q, reshape=True)
        if keys
        else []
    )
    out = {
        h: {"declared": False, "sign": v["sign"], "score": v["score"]} for h, v in scores.items()
    }
    for h, d in zip(keys, dec, strict=True):
        out[h]["declared"] = bool(d)
    return out


def _shap_fn() -> tuple[Callable[..., tuple[np.ndarray, float]], str]:
    try:
        from oran_causal.methods.controlled.shap_dag import fit_shap_importances
    except ImportError as e:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import permutation_importance

        def fit(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
            m = GradientBoostingRegressor(random_state=seed).fit(X, y)
            pi = permutation_importance(m, X, y, n_repeats=5, random_state=seed)
            return np.abs(pi.importances_mean), float(m.score(X, y))

        return fit, f"FALLBACK sklearn GBDT + permutation importance (shap not importable: {e!r})"
    return fit_shap_importances, "shap_dag.fit_shap_importances (HistGBDT + shap.TreeExplainer)"


def score_shap(data: UnitTable, sup: Support = SUPPORT, seed: int = 0) -> Scores:
    """mean|SHAP(x)| / std(y) of a HistGBDT on [x, sgn, pre]; sign = sign(corr(x, y))."""
    fit, note = _shap_fn()
    out = _empty()
    for _f, rows, h in _each(data, sup):
        y = data.y[h[1:]][rows]
        sd = float(np.std(y))
        if sd == 0:
            continue
        X = np.column_stack([data.x[rows], data.sgn[rows], data.pre[h[1:]][rows]])
        imp, r2 = fit(X, y, seed=seed)
        c = _corr(data.x[rows], y)
        out[h] = {"score": float(imp[0]) / sd, "sign": _sign_of(c), "r2": float(r2)}
    out["_note"] = note
    return out


def score_two_tower(
    data: UnitTable,
    sup: Support = SUPPORT,
    seed: int = 0,
    sizes: Mapping[str, Any] | None = None,
) -> Scores:
    """Two-tower gate a[target, family] on family-masked x; sign = sign(corr(x, y))."""
    from oran_causal.methods.controlled.two_tower import (
        TowerSizes,
        TrainConfig,
        fit_two_tower,
    )

    sz: dict[str, Any] = dict(TT_SIZES, **(sizes or {}))
    out = _empty()
    fams = [f for f in FAMILIES if supported(data, f, sup)]
    if not fams or data.n < 3:
        return out
    X = np.column_stack([data.x * (data.family == FAMILIES.index(f)) for f in fams])
    tg = [(r, k) for r in RELATIONS for k in KPIS]
    Y = np.column_stack([data.y[t] for t in tg])
    tower = TowerSizes(len(fams), len(tg), d=sz["d"], r=sz["r"], hidden=sz["hidden"])
    train = TrainConfig(epochs=sz["epochs"], lr=sz["lr"], l1=sz["l1"], seed=seed)
    S, r2 = fit_two_tower(X, Y, tower, train)
    for j, f in enumerate(fams):
        rows = data.rows_of(f)
        for i, (r, k) in enumerate(tg):
            if np.std(data.y[(r, k)][rows]) == 0:
                continue
            c = _corr(data.x[rows], data.y[(r, k)][rows])
            out[(f, r, k)] = {"score": float(S[i, j]), "sign": _sign_of(c)}
    out["_note"] = f"fit_two_tower with TowerSizes {sz}, pseudo-R2 {r2:.3f}"
    return out


def score_int(data: UnitTable, sup: Support = SUPPORT) -> Scores:
    """INT distance of y * sgn between accepted and rejected units."""
    out = _empty()
    for _f, rows, h in _each(data, sup):
        yd = data.y[h[1:]][rows] * data.sgn[rows]
        lv = data.level[rows]
        a, b = yd[lv == 1.0], yd[lv == 0.0]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        out[h] = {"score": int_distance(a, b), "sign": int(np.sign(a.mean() - b.mean()))}
    return out


class _QacmFit(NamedTuple):
    seed: int
    holdout: float
    min_samples: int
    ann: bool


def _qacm_one(data: UnitTable, rows: np.ndarray, h: Hypothesis, cfg: _QacmFit) -> dict | None:
    f, r, k = h
    pre = data.pre[(r, k)][rows]
    post = data.y[(r, k)][rows] + pre
    mu, sd = float(post.mean()), float(post.std())
    if sd == 0 or len(rows) < cfg.min_samples:
        return None
    delta = data.level[rows] * data.step[rows]
    cur = data.z.get("ctx_cur", np.zeros(data.n))[rows]
    util = data.z.get("ctx_own_prb_util", np.zeros(data.n))[rows]
    util = np.where(np.isfinite(util), util, np.nanmedian(util) if np.isfinite(util).any() else 0.0)
    X = np.column_stack([delta, cur, (pre - mu) / sd, util])
    fm, fs = X.mean(0), X.std(0) + 1e-9
    Xs = (X - fm) / fs
    X0 = X.copy()
    X0[:, 0] = 0.0
    X0s = (X0 - fm) / fs
    yz = (post - mu) / sd
    key = (FAMILIES.index(f), RELATIONS.index(r), KPIS.index(k))
    perm = np.random.default_rng([cfg.seed, 6616, 7, 7, *key]).permutation(len(yz))
    n_te = max(int(len(yz) * cfg.holdout), 1)
    te, tr = perm[:n_te], perm[n_te:]
    cands: dict[str, Any] = {"PR": _PR().fit(Xs[tr], yz[tr])}
    if cfg.ann:
        cands["ANN"] = _ANN(Xs.shape[1], cfg.seed).fit(Xs[tr], yz[tr])
    r2 = {m: _r2(yz[te], g.predict(Xs[te])) for m, g in cands.items()}
    best = max(r2, key=lambda m: r2[m])
    g = cands[best]
    chg = delta != 0
    if not chg.any():
        return None
    eff = g.predict(Xs[chg]) - g.predict(X0s[chg])
    return {
        "score": float(np.mean(np.abs(eff))),
        "sign": int(np.sign(np.mean(eff * np.sign(delta[chg])))),
        "model": best,
        "r2": r2,
    }


def score_qacm(
    data: UnitTable,
    sup: Support = SUPPORT,
    relations: Sequence[str] = ("own",),
    seed: int = 0,
    *,
    holdout: float = 0.2,
    min_samples: int = 30,
    ann: bool = True,
) -> Scores:
    """QACM KPI-predictor effect of the applied change, per hypothesis of `relations`."""
    out = _empty()
    cfg = _QacmFit(seed, holdout, min_samples, ann)
    for _f, rows, h in _each(data, sup, relations):
        v = _qacm_one(data, rows, h, cfg)
        if v is not None:
            out[h] = v
    return out


def _physics_reference(prior: Mapping[Hypothesis, int] | None) -> dict[Hypothesis, dict]:
    prior = PHYSICS_PRIOR if prior is None else prior
    return {
        h: (
            {"status": "TRUE", "sign": int(prior[h])}
            if h in prior
            else {"status": "NULL", "sign": 0}
        )
        for h in HYPOTHESES
    }


def _f1(decl: Mapping[Hypothesis, Mapping[str, Any]], ref: Mapping[Hypothesis, Mapping]) -> float:
    tp = fp = fn = 0
    for h in HYPOTHESES:
        g = ref.get(h, {"status": "INDET", "sign": 0})
        d = decl.get(h)
        if g["status"] == "INDET" or d is None or d.get("declared") is None:
            continue
        if d["declared"]:
            tp += g["status"] == "TRUE"
            fp += g["status"] != "TRUE"
        else:
            fn += g["status"] == "TRUE"
    if tp == 0:
        return 0.0 if (tp + fp + fn) > 0 else float("nan")
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _tau_physics(scores: Scores, prior: Mapping[Hypothesis, int] | None) -> dict[str, Any]:
    ref = _physics_reference(prior)
    fin = sorted(
        {v["score"] for h, v in scores.items() if isinstance(h, tuple) and np.isfinite(v["score"])}
    )
    best = (-1.0, float("inf"))
    for tau in [-np.inf, *fin]:
        f1 = _f1(declare(scores, tau), ref)
        f1 = 0.0 if not np.isfinite(f1) else f1
        if f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and tau > best[1]):
            best = (f1, tau)
    return {
        "tau": float(best[1]),
        "rule": "physics_f1",
        "dev_proxy_f1": best[0],
        "prior": "edge_score.PHYSICS_PRIOR",
    }


def tune_tau(
    scores: Scores,
    rule: str = "far_fpr",
    max_far: int = MAX_FAR_DEV,
    prior: Mapping[Hypothesis, int] | None = None,
) -> dict[str, Any]:
    """Threshold from DEV scores; declare iff score > tau.

    Raises:
        ValueError: Unknown rule.
    """
    if rule == "far_fpr":
        far = sorted(
            (
                v["score"]
                for h, v in scores.items()
                if isinstance(h, tuple) and h[1] == "far" and np.isfinite(v["score"])
            ),
            reverse=True,
        )
        if len(far) <= max_far:
            return {
                "tau": float("inf"),
                "rule": rule,
                "max_far": max_far,
                "n_far_scorable": len(far),
                "note": "untunable (too few scorable far hypotheses): declares nothing",
            }
        return {
            "tau": float(far[max_far]),
            "rule": rule,
            "max_far": max_far,
            "n_far_scorable": len(far),
        }
    if rule == "physics_f1":
        return _tau_physics(scores, prior)
    raise ValueError(f"rule in {TAU_RULES}")


def declare(
    scores: Scores, tau: float, relations: Sequence[str] = RELATIONS
) -> dict[Hypothesis, dict[str, Any]]:
    """{h: declared, sign, score}; NaN or outside `relations` is never declared."""
    out = {}
    for h in HYPOTHESES:
        v = scores.get(h, {"score": float("nan"), "sign": 0})
        s = v["score"]
        out[h] = {
            "declared": bool(h[1] in relations and np.isfinite(s) and s > tau),
            "sign": int(v["sign"]),
            "score": s,
        }
    return out


SCORERS: dict[str, Callable[..., Scores]] = {
    "shap_gbdt": score_shap,
    "corr": score_corr,
    "granger": granger_pvalues,
    "two_tower": score_two_tower,
    "int": score_int,
    "qacm": score_qacm,
}
DECL_RELATIONS = {"qacm": ("own",)}


def _keys(scores: Scores) -> dict[str, Any]:
    return {"|".join(h): v for h, v in scores.items() if isinstance(h, tuple)}


def _scorer_kw(m: str, sup: Support, seed: int, ann: bool) -> tuple[dict, dict]:
    kw: dict[str, Any] = {"sup": sup}
    if m in ("shap_gbdt", "two_tower", "qacm"):
        kw["seed"] = seed
    if m == "qacm":
        kw["ann"] = ann
    dev_kw = dict(kw, relations=("own", "far")) if m == "qacm" else kw
    return kw, dev_kw


def _granger_by_splits(
    res: dict[str, Any], dev: UnitTable, dev_scores: Scores, splits: Mapping[str, UnitTable]
) -> dict[str, Any]:
    gb: dict[str, Any] = {
        "tau": None,
        "note": "untuned: BY at q = 0.05 over the scorable hypotheses",
        "splits": {},
    }
    for name, d in splits.items():
        sc = dev_scores if d is dev else res["splits"][name]["scores"]
        sc = {tuple(k.split("|")) if isinstance(k, str) else k: v for k, v in sc.items()}
        gb["splits"][name] = {"declared": granger_by(sc)}
    return gb


def run_baselines(
    dev: UnitTable,
    splits: Mapping[str, UnitTable],
    methods: Sequence[str] = TUNED,
    sup: Support = SUPPORT,
    seed: int = 0,
    *,
    log: Callable[[str], None] | None = None,
    ann: bool = True,
) -> dict[str, Any]:
    """Tunes each baseline's tau on `dev`, freezes it, scores and declares every split.

    Also returns "granger_by" (untuned) when "granger" runs.
    """
    out = {}
    for m in methods:
        t0, w0 = time.process_time(), time.time()
        fn = SCORERS[m]
        kw, dev_kw = _scorer_kw(m, sup, seed, ann)
        dev_scores = fn(dev, **dev_kw)
        tau = tune_tau(dev_scores, "far_fpr")
        tau_phys = tune_tau(dev_scores, "physics_f1")
        rels = DECL_RELATIONS.get(m, RELATIONS)
        res: dict[str, Any] = {
            "tau": tau,
            "tau_physics": tau_phys,
            "note": dev_scores.get("_note"),
            "decl_relations": list(rels),
            "dev_scores": _keys(dev_scores),
            "splits": {},
        }
        for name, d in splits.items():
            sc = dev_scores if d is dev else fn(d, **kw)
            res["splits"][name] = {
                "declared": declare(sc, tau["tau"], rels),
                "declared_physics_tau": declare(sc, tau_phys["tau"], rels),
                "scores": _keys(sc),
            }
        res["cpu_s"] = round(time.process_time() - t0, 1)
        res["wall_s"] = round(time.time() - w0, 1)
        out[m] = res
        if log:
            log(
                f"   baseline {m}: tau {tau['tau']:.4g} (physics {tau_phys['tau']:.4g}) wall "
                f"{res['wall_s']} s (cpu {res['cpu_s']} s)"
            )
        if m == "granger":
            out["granger_by"] = _granger_by_splits(res, dev, dev_scores, splits)
    return out


class Tuned:
    """A baseline with its frozen DEV tau: `method(data, split)` returns the declarations.

    Attributes:
        label: Method name.
        tau: The tuned threshold record, None for untuned methods.
    """

    def __init__(
        self, label: str, tau: dict[str, Any] | None, fn: Callable[[UnitTable], dict]
    ) -> None:
        self.label, self.tau, self._fn = label, tau, fn

    def __call__(self, data: UnitTable, split: int) -> dict[Hypothesis, dict[str, Any]]:
        """Declarations on `data` (the split id does not enter the baselines)."""
        return self._fn(data)


def _with_p(sc: Scores, d: dict) -> dict:
    for h, v in d.items():
        if isinstance(sc.get(h), dict) and "p" in sc[h]:
            v["p"] = sc[h]["p"]
    return d


def method_baseline(name: str, dev: UnitTable) -> Tuned:
    """DEV-tuned (far-FPR) corr or granger, or the untuned granger_by; granger carries "p"."""
    if name == "granger_by":

        def gby(data: UnitTable) -> dict:
            sc = granger_pvalues(data)
            return _with_p(sc, granger_by(sc))

        return Tuned("granger_by", None, gby)
    fn = SCORERS[name]
    tau = tune_tau(fn(dev), "far_fpr")

    def tuned(data: UnitTable) -> dict:
        sc = fn(data)
        return _with_p(sc, declare(sc, tau["tau"]))

    return Tuned(name, tau, tuned)


def desc_method(name: str, dev: UnitTable, seed: int = 0) -> Tuned:
    """A descriptive baseline with the DEV far-FPR tau, as `run_baselines` tunes it."""
    fn = SCORERS[name]
    kw = {"seed": seed} if name in ("shap_gbdt", "two_tower", "qacm") else {}
    dev_kw = dict(kw, relations=("own", "far")) if name == "qacm" else kw
    tau = tune_tau(fn(dev, **dev_kw), "far_fpr")
    rels = DECL_RELATIONS.get(name, RELATIONS)

    def desc(data: UnitTable) -> dict:
        return declare(fn(data, **kw), tau["tau"], rels)

    return Tuned(name, tau, desc)

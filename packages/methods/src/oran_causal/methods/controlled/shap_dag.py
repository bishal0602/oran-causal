"""SHAP-DAG (Sharma et al., arXiv:2510.13031, NFV-SDN 2025), discovery stage.

One gradient-boosted regressor per KPI at t+1 on the actions at t, explained with
`shap.TreeExplainer`; mean |SHAP| per action per KPI. Regressor `"xgboost"` (the paper's
choice) is `XGBRegressor` at package defaults, seeded; `"hgbdt"` is the E2 port's
`HistGradientBoostingRegressor(max_iter=300, learning_rate=.1, early_stopping=False)` as a
sensitivity option. Lagged-KPI candidates are not scorable unless `include_lag`. Score =
mean |SHAP| / sd(target); `native` = the E2 relative rule mean |SHAP| >= tau_rel * max.
Sign: partial correlation given the native set. The primary regressors have no
`P_placebo_conf` feature; its own candidates come from a second fit with every feature.
"""

from typing import Any, ClassVar

import numpy as np
import shap
import sklearn
import xgboost
from sklearn.ensemble import HistGradientBoostingRegressor

from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._classic import (
    PLACEBO_CONF,
    ClassicBase,
    Columns,
    Scored,
    columns,
    int_seed,
    resolve,
    sign_pcorr_given_z,
    target_index,
)

Feature = tuple[str, int]


def make_regressor(regressor: str, seed: int, threads: int = 1) -> Any:
    """The unfitted regressor `"xgboost"` or `"hgbdt"`; never early-stopped.

    Raises:
        ValueError: Unknown regressor.
    """
    if regressor == "xgboost":
        return xgboost.XGBRegressor(random_state=seed, n_jobs=threads)
    if regressor == "hgbdt":
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=seed, early_stopping=False
        )
    raise ValueError(f"unknown regressor {regressor!r}")


def fit_shap(
    x: np.ndarray, y: np.ndarray, seed: int, regressor: str = "xgboost", threads: int = 1
) -> tuple[np.ndarray, float]:
    """Fits one regressor; returns (mean |SHAP| per feature, training R^2)."""
    model = make_regressor(regressor, seed, threads)
    model.fit(x, y)
    r2 = float(model.score(x, y))
    sv = shap.TreeExplainer(model).shap_values(x)
    return np.abs(sv).mean(axis=0), r2


def fit_shap_importances(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """E2 port: `HistGradientBoostingRegressor(max_iter=300, learning_rate=.1)`, else defaults.

    Returns:
        (mean |SHAP| per feature, training R^2).
    """
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=seed)
    model.fit(x, y)
    r2 = float(model.score(x, y))
    sv = shap.TreeExplainer(model).shap_values(x)
    return np.abs(sv).mean(axis=0), r2


def _fit_all(
    data: Dataset, cols: Columns, config: dict[str, Any], seed: int, with_conf: bool
) -> tuple[list[Feature], dict[int, tuple[np.ndarray, float]], dict[str, float]]:
    conf = data.action_names.index(PLACEBO_CONF) if PLACEBO_CONF in data.action_names else None
    feats: list[Feature] = [
        ("action", j) for j in range(cols.actions.shape[1]) if with_conf or j != conf
    ]
    if config["include_lag"]:
        feats += [("kpi", j) for j in range(cols.lags.shape[1])]
    x = np.column_stack(
        [cols.actions[:, j] if f == "action" else cols.lags[:, j] for f, j in feats]
    )
    imps: dict[int, tuple[np.ndarray, float]] = {}
    r2s: dict[str, float] = {}
    for ti in sorted({target_index(data, t) for _, t in data.candidates}):
        y = cols.Y[:, ti]
        if np.std(y) == 0:
            continue
        imp, r2 = fit_shap(x, y, seed, config["regressor"], int(config.get("threads", 1)))
        imps[ti] = (np.asarray(imp, float), float(np.std(y)))
        r2s[data.kpi_names[ti]] = float(r2)
    return feats, imps, r2s


class ShapDag(ClassicBase):
    """SHAP-DAG arm (native only)."""

    name = "shap_dag"
    version = "XGBRegressor (package defaults) + shap.TreeExplainer; hgbdt sensitivity option"
    uses_p = False
    method_idx = 3
    defaults: ClassVar[dict[str, Any]] = {
        "include_lag": False,
        "tau_rel_native": 0.10,
        "regressor": "xgboost",
    }

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        seed = (
            int_seed(data, self.method_idx) if config.get("seed") is None else int(config["seed"])
        )
        main = _fit_all(data, cols, config, seed, False)
        has_conf = any(s == PLACEBO_CONF for s, _ in data.candidates)
        diag = _fit_all(data, cols, config, seed, True) if has_conf else None
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            fe, im, _ = diag if s == PLACEBO_CONF and diag is not None else main
            if j is None or (fam, j) not in fe or ti not in im:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            imp, sd = im[ti]
            v = float(imp[fe.index((fam, j))])
            native = bool(imp.max() > 0 and v >= config["tau_rel_native"] * float(imp.max()))
            out.append(
                Scored(
                    s, t, fam, v / sd, sign_pcorr_given_z(data, cols, (fam, j, ti)), None, native
                )
            )
        notes: dict[str, Any] = {
            "sign_rule": "pcorr_given_Z",
            "train_r2": main[2],
            "seed": seed,
            "regressor": config["regressor"],
            "packages": {
                "shap": shap.__version__,
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            "score": "mean|SHAP| / sd(target)",
            "native_rule": f"E2 relative rule, tau_rel = {config['tau_rel_native']}",
        }
        if diag is not None:
            notes["diagnostic_fit"] = {
                "rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                "feature; primary fit without it",
                "train_r2": diag[2],
            }
        return out, notes

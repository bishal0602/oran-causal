"""Linear NOTEARS (Zheng, Aragam, Ravikumar and Xing, NeurIPS 2018), the authors' code.

Reference code: `_vendor/notears/linear.py` (xunzheng/notears @ 4a9ab19, Apache-2.0,
byte-identical). Authors' defaults: loss l2, lambda1 = 0.1, max_iter = 100, h_tol = 1e-8,
rho_max = 1e16, w_threshold = 0.3.

Adaptations: the variables (actions, non-NaN lagged KPIs, context, KPIs at t+1) are
z-scored; background knowledge (as in pc) fixes the L-BFGS-B bounds of forbidden entries at
(0, 0), the authors' own mechanism for the diagonal; score = |W[source, target]| of the
unthresholded estimate, `native` = score >= w_threshold, sign = sign(W). The primary fit has
no `P_placebo_conf` variable; its own candidates come from a second fit with every variable.
"""

from typing import Any, ClassVar

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._classic import (
    PLACEBO_CONF,
    ClassicBase,
    Columns,
    Scored,
    columns,
    resolve,
    target_index,
    zscore,
)

VENDOR_COMMIT = "xunzheng/notears@4a9ab19fe502e392503da331773c5223d82d3666"


def notears_linear_bk(
    X,
    lambda1,
    loss_type,
    max_iter=100,
    h_tol=1e-8,
    rho_max=1e16,
    w_threshold=0.3,
    forbid=None,
):
    """Vendored `notears_linear` (Apache-2.0) whose only change is the bounds list.

    `forbid` (`[d, d]` bool, True = W[i, j] fixed at 0) extends the authors' diagonal
    (0, 0) bounds; everything else is the authors' code.
    """

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / X.shape[0] * (R**2).sum()
            G_loss = -1.0 / X.shape[0] * X.T @ R
        elif loss_type == "logistic":
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError("unknown loss type")
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    _, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    fb = (
        np.eye(d, dtype=bool)
        if forbid is None
        else (np.asarray(forbid, dtype=bool) | np.eye(d, dtype=bool))
    )
    bnds = [(0, 0) if fb[i, j] else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == "l2":
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new: Any = None
        h_new: Any = None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def forbidden_matrix(kinds: list[str]) -> np.ndarray:
    """Background knowledge over node kinds A, L, C, Y: True = W[i, j] fixed at 0."""
    d = len(kinds)
    forbid = np.zeros((d, d), dtype=bool)
    for i, ki in enumerate(kinds):
        for j, kj in enumerate(kinds):
            if (
                (ki == "Y" and kj != "Y")
                or (kj == "A" and ki != "C")
                or (ki == "A" and kj in ("L", "C"))
            ):
                forbid[i, j] = True
    return forbid


def _variables(
    data: Dataset, cols: Columns, with_conf: bool
) -> tuple[list[str], list[np.ndarray], list[str]]:
    blocks = [
        ("A", data.action_names, cols.actions),
        ("L", cols.lag_names, cols.lags),
        ("C", tuple(f"c{i}" for i in range(cols.context.shape[1])), cols.context),
        ("Y", data.kpi_names, cols.Y),
    ]
    names, mats, kinds = [], [], []
    for kind, nm, m in blocks:
        for i, v in enumerate(nm):
            if kind == "A" and v == PLACEBO_CONF and not with_conf:
                continue
            names.append(f"{kind}:{v}")
            mats.append(m[:, i])
            kinds.append(kind)
    return names, mats, kinds


def _fit(data: Dataset, cols: Columns, config: dict[str, Any], with_conf: bool) -> dict[str, Any]:
    names, mats, kinds = _variables(data, cols, with_conf)
    d = np.column_stack(mats)
    keep = [i for i in range(d.shape[1]) if np.std(d[:, i]) > 0]
    d, kk = d[:, keep], [kinds[i] for i in keep]
    idx = {names[i]: k for k, i in enumerate(keep)}
    if config["standardize"]:
        d = zscore(d)
    nv = d.shape[1]
    forbid = (
        forbidden_matrix(kk) if config["background_knowledge"] else np.zeros((nv, nv), dtype=bool)
    )
    w = notears_linear_bk(
        d,
        config["lambda1"],
        config["loss_type"],
        max_iter=config["max_iter"],
        h_tol=config["h_tol"],
        rho_max=config["rho_max"],
        w_threshold=0.0,
        forbid=forbid,
    )
    return {"W": w, "idx": idx, "d": nv}


class Notears(ClassicBase):
    """NOTEARS arm (native only)."""

    name = "notears"
    version = f"linear, {VENDOR_COMMIT}"
    uses_p = False
    method_idx = 2
    defaults: ClassVar[dict[str, Any]] = {
        "lambda1": 0.1,
        "loss_type": "l2",
        "max_iter": 100,
        "h_tol": 1e-8,
        "rho_max": 1e16,
        "w_threshold": 0.3,
        "standardize": True,
        "background_knowledge": True,
    }

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        main = _fit(data, cols, config, with_conf=False)
        has_conf = any(s == PLACEBO_CONF for s, _ in data.candidates)
        diag = _fit(data, cols, config, with_conf=True) if has_conf else None
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            g = diag if s == PLACEBO_CONF and diag is not None else main
            sname = (
                None
                if j is None
                else (f"A:{data.action_names[j]}" if fam == "action" else f"L:{cols.lag_names[j]}")
            )
            tname = f"Y:{data.kpi_names[ti]}"
            if sname not in g["idx"] or tname not in g["idx"]:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            w = float(g["W"][g["idx"][sname], g["idx"][tname]])
            out.append(
                Scored(s, t, fam, abs(w), int(np.sign(w)), None, abs(w) >= config["w_threshold"])
            )
        w, d = main["W"], main["d"]
        notes: dict[str, Any] = {
            "sign_rule": "sign(W)",
            "h_final": float(np.trace(slin.expm(w * w)) - d),
            "n_vars": d,
            "nnz_native": int((np.abs(w) >= config["w_threshold"]).sum()),
            "score": "|W| unthresholded",
        }
        if diag is not None:
            notes["diagnostic_fit"] = {
                "rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                "variable; primary fit without it",
                "n_vars": diag["d"],
            }
        return out, notes

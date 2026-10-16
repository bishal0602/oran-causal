"""Partial-correlation test with a Newey-West HAC standard error.

For candidate source -> target with conditioning set Z (`_classic.cond_set`), OLS of
Y_k(t+1) = a + g source(t) + Z c + u on the rows in time order tests H0: g = 0. SE(g) is the
Newey and West (1987) Bartlett HAC with the Andrews (1991, Econometrica 59(3) 817-858) AR(1)
plug-in bandwidth S_T = 1.1447 (alpha(1) T)^(1/3), alpha(1) = 4 rho^2 / ((1 - rho)^2
(1 + rho)^2), rho the AR(1) coefficient of the score series v_t = x~_t u_t, maxlags
floor(S_T) capped at n - 2, rho clipped to [-.99, .99]; small-sample factor n / (n - k) and
a t(n - k) reference, as statsmodels `OLS.fit(cov_type="HAC", use_correction=True,
use_t=True)`. Variant `inference="fixed_b"` refers the same statistic to the Kiefer and
Vogelsang (2005) fixed-b limit at b = (L + 1) / n. score = -log10 p, sign = sign(g).
Arms: `"native"` (R-3 set) and `"eq_min"` (R-3 set plus the focal setpoint).
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from scipy import linalg, stats

from oran_causal.core.covariates import info_order
from oran_causal.core.data import Dataset, Result
from oran_causal.methods.controlled._classic import (
    ClassicBase,
    Scored,
    columns,
    cond_set,
    exact_fit,
    not_testable_notes,
    resolve,
    source_column,
    target_index,
)
from oran_causal.methods.controlled._fixedb import fixedb_p

ANDREWS_BARTLETT = 1.1447
RHO_CLIP = 0.99
COLLINEAR_TOL = 1e-10
OK, EXACT, COLLINEAR = "ok", "exact", "collinear"
INFERENCE = ("t", "fixed_b")


def andrews_lags(v: np.ndarray) -> tuple[int, float, float]:
    """(maxlags L, rho, S_T) of the Andrews (1991) AR(1) plug-in bandwidth, Bartlett kernel."""
    v = np.asarray(v, float)
    n = len(v)
    den = float(v[:-1] @ v[:-1])
    rho = float(v[1:] @ v[:-1]) / den if den > 0 else 0.0
    rho = float(np.clip(rho, -RHO_CLIP, RHO_CLIP))
    alpha = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    s_t = ANDREWS_BARTLETT * (alpha * n) ** (1.0 / 3.0)
    return int(min(np.floor(s_t), max(n - 2, 0))), rho, float(s_t)


def independent_columns(z: np.ndarray) -> np.ndarray:
    """Ascending indices of a maximal subset of Z's columns independent together with the intercept.

    Pivoted QR of the centred, unit-norm columns; constant columns are dropped.
    """
    if z.shape[1] == 0:
        return np.zeros(0, np.int64)
    zc = z - z.mean(0)
    nrm = np.linalg.norm(zc, axis=0)
    live = np.nonzero(nrm > COLLINEAR_TOL * np.maximum(np.linalg.norm(z, axis=0), 1e-300))[0]
    if not len(live):
        return np.zeros(0, np.int64)
    _, r, piv = linalg.qr(zc[:, live] / nrm[live], mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    return np.sort(live[piv[: int(np.sum(d > COLLINEAR_TOL * d[0]))]])


def hac_design(z: np.ndarray) -> np.ndarray:
    """[1, Z without exactly collinear or constant columns]."""
    return np.column_stack([np.ones(z.shape[0]), z[:, independent_columns(z)]])


def _bartlett_lrv(v: np.ndarray, lags: int) -> float:
    lrv = float(v @ v)
    for j in range(1, lags + 1):
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * float(v[j:] @ v[:-j])
    return lrv


def hac_test(
    y: np.ndarray, x: np.ndarray, z: np.ndarray, maxlags: int | None = None, inference: str = "t"
) -> dict[str, Any]:
    """OLS t-test of x in y ~ 1 + Z + x with the HAC standard error; rows must be in time order.

    Args:
        y: Target.
        x: Source.
        z: Conditioning columns.
        maxlags: Fixed Newey-West lag; None uses the Andrews bandwidth.
        inference: `"t"` (t(n - k) reference) or `"fixed_b"` (Kiefer-Vogelsang 2005).

    Returns:
        {status, g, se, t, p, df, maxlags, rho, s_t, k, b}; status `"exact"` or
        `"collinear"` marks a candidate that is not testable (NaNs).

    Raises:
        ValueError: Unknown `inference`.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    nan = float("nan")
    if inference not in INFERENCE:
        raise ValueError(f"inference must be one of {INFERENCE}, got {inference!r}")
    out: dict[str, Any] = {
        "status": OK, "g": nan, "se": nan, "t": nan, "p": nan, "df": 0,
        "maxlags": None, "rho": nan, "s_t": nan, "k": 0, "b": nan,
    }  # fmt: skip
    a = hac_design(np.asarray(z, float).reshape(len(y), -1))
    if exact_fit(y, np.column_stack([a[:, 1:], x])):
        out["status"] = EXACT
        return out
    n, k = len(y), a.shape[1] + 1
    xr = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]
    yr = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    sxx = float(xr @ xr)
    if sxx <= COLLINEAR_TOL**2 * max(float(x @ x), 1e-300) or n - k <= 0:
        out["status"] = COLLINEAR
        return out
    g = float(xr @ yr) / sxx
    v = xr * (yr - g * xr)
    lags, rho, s_t = andrews_lags(v) if maxlags is None else (int(maxlags), nan, nan)
    df = n - k
    se = float(np.sqrt(max(_bartlett_lrv(v, lags), 0.0) / sxx**2 * n / df))
    t = g / se if se > 0 else (np.inf if g != 0 else 0.0)
    b = (lags + 1.0) / n
    p = float(2.0 * stats.t.sf(abs(t), df)) if inference == "t" else fixedb_p(float(t), b)
    out.update(
        g=g,
        se=se,
        t=float(t),
        p=p,
        df=int(df),
        maxlags=int(lags),
        rho=rho,
        s_t=s_t,
        k=int(k),
        b=float(b),
    )
    return out


def _hac_notes(config: dict[str, Any], lags: dict[str, int], rhos: list[float]) -> dict[str, Any]:
    lv = list(lags.values())
    hac: dict[str, Any] = {
        "kernel": "bartlett",
        "bandwidth": "andrews1991_ar1"
        if config.get("maxlags") is None
        else f"fixed {config['maxlags']}",
        "use_correction": True,
        "reference": "t(n - k)"
        if config["inference"] == "t"
        else "fixed-b (Kiefer-Vogelsang 2005, Bartlett, b = (L + 1) / n)",
        "maxlags_median": float(np.median(lv)) if lv else None,
        "maxlags_max": int(max(lv)) if lv else None,
    }
    if rhos and np.isfinite(rhos).any():
        hac["rho_median"] = float(np.nanmedian(rhos))
    return hac


class PcorrHac(ClassicBase):
    """HAC partial-correlation arm (`"native"`, `"eq_min"`)."""

    name = "pcorr_hac"
    version = "1.1"
    uses_p = True
    method_idx = 7
    arms = ("native", "eq_min")
    defaults: ClassVar[dict[str, Any]] = {
        "q": 0.05,
        "arm": "native",
        "maxlags": None,
        "inference": "t",
    }

    def run(self, data: Dataset, config: Mapping[str, Any] | None = None) -> Result:
        """Scores and declares every candidate of `data`.

        Raises:
            ValueError: The arm is not `"native"` or `"eq_min"`.
        """
        arm = (config or {}).get("arm", self.defaults["arm"])
        if arm not in self.arms:
            raise ValueError(f"pcorr_hac arm must be one of {self.arms}, got {arm!r}")
        return super().run(data, config)

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        arm = config["arm"]
        order = info_order(data)
        out: list[Scored] = []
        bad: dict[str, list[str]] = {EXACT: [], COLLINEAR: []}
        lags: dict[str, int] = {}
        zsize: dict[str, int] = {}
        tstat: dict[str, tuple[float, int, float]] = {}
        rhos: list[float] = []
        n_used: set[int] = set()
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            if j is None:
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            x, src = source_column(data, cols, fam, j)
            cs = cond_set(data, fam, src, arm)
            rows = order[cs.mask[order]]
            n_used.add(len(rows))
            r = hac_test(
                cols.Y[rows, ti], x[rows], cs.Z[rows], config.get("maxlags"), config["inference"]
            )
            key = f"{s}->{t}"
            zsize[key] = int(cs.Z.shape[1])
            if r["status"] != OK:
                bad[r["status"]].append(key)
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            lags[key] = r["maxlags"]
            tstat[key] = (r["t"], r["df"], r["b"])
            rhos.append(r["rho"])
            p = r["p"]
            out.append(Scored(s, t, fam, float(-np.log10(max(p, 1e-300))), int(np.sign(r["g"])), p))
        zdesc = "R-3 set" if arm == "native" else "R-3 set + sp:<focal> (eq_min, R-33)"
        notes = {
            "sign_rule": "sign(OLS coefficient) = sign of pcorr given Z",
            **not_testable_notes(bad[EXACT], bad[COLLINEAR]),
            "label": "partial-correlation t-test, Newey-West HAC SE "
            "(Bartlett, Andrews 1991 AR(1) bandwidth), "
            f"Z = {zdesc}",
            "hac": _hac_notes(config, lags, rhos),
            "maxlags": lags,
            "z_size": zsize,
            "n_rows_used": sorted(n_used),
            "t_df_b": tstat,
        }
        return out, notes

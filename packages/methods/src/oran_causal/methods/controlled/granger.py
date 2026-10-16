"""One-lag Granger F-test in two arms.

For candidate source -> target KPI k, restricted model Y_k(t+1) = a + c K_k(t), full model
adds g source(t); F = (RSS_r - RSS_f) / (RSS_f / (n - q_f)) ~ F(1, n - q_f), score =
-log10 p, sign = sign(g). Arm `"native"` is the pairwise test (statsmodels
`grangercausalitytests` `ssr_ftest` at maxlag 1); without the target's own lag the
restricted model is intercept only. Arm `"eq"` is conditional Granger / VARX: the
restricted model is the intercept plus the source's whole conditioning set from
`_classic.cond_set`. A restricted model that already fits the target exactly, or a source
that adds no rank, makes the candidate not testable.
"""

from typing import Any, ClassVar

import numpy as np
from scipy.stats import f as fdist

from oran_causal.core.covariates import design_covariates
from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._classic import (
    EXACT_TOL,
    ClassicBase,
    Columns,
    Scored,
    columns,
    cond_set,
    not_testable_notes,
    resolve,
    source_column,
    target_index,
)

EXACT = "exact"
DEGENERATE = "degenerate"
LABEL_EQ = (
    "conditional Granger / VARX (one lag; restricted model = R-3 set + design covariates Z_eq, "
    "R-17)"
)
LABEL_NATIVE = "pairwise Granger (bivariate F-test, one lag; restricted model = target's own lag)"


def _rss(a: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, int]:
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    r = y - a @ beta
    return float(r @ r), beta, int(np.linalg.matrix_rank(a))


def granger_f(
    y: np.ndarray, own: np.ndarray | None, x: np.ndarray, z: np.ndarray | None = None
) -> tuple[float, float, float, str]:
    """One-lag Granger F-test of `x` for `y`.

    The restricted model is intercept + own lag (if any) + the columns of `z`; the full
    model adds `x`.

    Returns:
        (F, p, coefficient of x, status); status `"ok"`, `"exact"` (the restricted model
        fits y exactly) or `"degenerate"` (x adds no rank or no residual df), NaNs unless ok.
    """
    nan = float("nan")
    n = len(y)
    base = [np.ones(n)] + ([own] if own is not None else [])
    if z is not None and z.shape[1]:
        base += [z[:, i] for i in range(z.shape[1])]
    ar, af = np.column_stack(base), np.column_stack([*base, x])
    rss_r, _, rank_r = _rss(ar, y)
    rss_f, beta, rank_f = _rss(af, y)
    tss = float(((y - y.mean()) ** 2).sum())
    if tss <= 0 or rss_r <= EXACT_TOL * tss:
        return nan, nan, nan, EXACT
    if rank_f <= rank_r or n - rank_f <= 0 or rss_f <= 0:
        return nan, nan, nan, DEGENERATE
    df = n - rank_f
    f = max(rss_r - rss_f, 0.0) / (rss_f / df)
    return float(f), float(fdist.sf(f, 1, df)), float(beta[-1]), "ok"


def _test(
    data: Dataset, cols: Columns, cand: tuple[str, int, int], own: np.ndarray | None, eq: bool
) -> tuple[float, float, float, str]:
    fam, j, ti = cand
    if eq:
        x, src = source_column(data, cols, fam, j)
        cs = cond_set(data, fam, src, "eq")
        m = cs.mask
        return granger_f(cols.Y[m, ti], None, x[m], cs.Z[m])
    x = cols.actions[:, j] if fam == "action" else cols.lags[:, j]
    if fam == "kpi" and cols.lag_names[j] == data.kpi_names[ti]:
        own = None
    return granger_f(cols.Y[:, ti], own, x)


class Granger(ClassicBase):
    """Granger arm (`"eq"` conditional, `"native"` pairwise)."""

    name = "granger"
    version = "1.0"
    uses_p = True
    method_idx = 6
    arms = ("eq", "native")
    defaults: ClassVar[dict[str, Any]] = {"q": 0.05, "lags": 1, "arm": "eq"}

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        eq = config["arm"] == "eq"
        out: list[Scored] = []
        no_own: set[str] = set()
        bad: dict[str, list[str]] = {EXACT: [], DEGENERATE: []}
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            tname = data.kpi_names[ti]
            own = cols.lags[:, cols.lag_names.index(tname)] if tname in cols.lag_names else None
            if own is None:
                no_own.add(tname)
            if j is None:
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            _, p, g, st = _test(data, cols, (fam, j, ti), own, eq)
            if st != "ok":
                bad[st].append(f"{s}->{t}")
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            out.append(Scored(s, t, fam, float(-np.log10(max(p, 1e-300))), int(np.sign(g)), p))
        notes = {
            "sign_rule": "sign(granger coefficient)",
            "lags": 1,
            **not_testable_notes(bad[EXACT], bad[DEGENERATE]),
            "label": LABEL_EQ if eq else LABEL_NATIVE,
        }
        if eq:
            base, _, mask = design_covariates(data)
            notes.update(
                {
                    "eq_covariates": [*base, "concurrent:<designed actions>"],
                    "n_rows_used": int(mask.sum()),
                }
            )
        if no_own:
            notes["no_own_lag_targets"] = sorted(no_own)
        return out, notes

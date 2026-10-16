"""Marginal Pearson correlation baseline.

score = |r(source, target)|, p = the two-sided Pearson t-test p-value (valid for iid rows,
approximate under serial dependence), sign = sign(r). Declaration: pooled BY per family at
q = .05.
"""

from typing import Any, ClassVar

import numpy as np
from scipy import stats

from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._classic import (
    ClassicBase,
    Scored,
    columns,
    resolve,
    target_index,
)


class Corr(ClassicBase):
    """Marginal correlation arm."""

    name = "corr"
    version = "1.0"
    uses_p = True
    method_idx = 5
    defaults: ClassVar[dict[str, Any]] = {"q": 0.05}

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            x = None if j is None else (cols.actions[:, j] if fam == "action" else cols.lags[:, j])
            y = cols.Y[:, ti]
            if x is None or np.std(x) == 0 or np.std(y) == 0:
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            r, p = stats.pearsonr(x, y)
            out.append(Scored(s, t, fam, float(abs(r)), int(np.sign(r)), float(p)))
        return out, {"sign_rule": "sign(r)", "test": "scipy.stats.pearsonr (t-test)"}

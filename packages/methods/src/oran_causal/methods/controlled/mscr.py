"""MSCR-v2 arm: max-stratified correlation-ratio test with a shared permutation bank.

S* = max over single conditioners g != i of the stratified correlation ratio; the null is
the within-stratum random-partition bank, S*_null = max_{g != i} bank. Banks per target j
use `default_rng([seed, j, n_perm])`; a further column group g >= 1 (the `P_placebo_conf`
group) uses `[seed, j, n_perm, g]`. S* is a max over single conditioners, so the eq
covariates enter only as more single conditioners. Deviations from the v2 E2 use: the bank
holds `B_MAX` draws with Besag-Clifford stopping applied in stream order (`{"n_perm": 2999,
"bc_h": None}` reproduces `discover_mscr`); every candidate source is tested; the
conditioner pool is the common Z; the bank seed comes from the citests stream; the primary
family is pooled BY, v2's per-target BY is kept as `declared_native`.
"""

from typing import Any

import numpy as np

from oran_causal.core.data import Dataset
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.core.pvalues import BC_H, sequential_p_from_draws
from oran_causal.methods.controlled._citests import (
    B_MAX,
    CITestBase,
    Prepared,
    method_seed,
    progress,
)
from oran_causal.methods.controlled.mscr_bank import (
    MSCR_VERSION,
    Execution,
    MSCRConfig,
    build_banks,
    frozen_config,
    null_s_star,
    resolve_n_jobs,
    s_star,
)


def _streams(
    seed: int, targets: list[int], n_perm: int, group_index: int
) -> list[np.random.Generator]:
    if group_index == 0:
        return [np.random.default_rng([int(seed), j, n_perm]) for j in targets]
    return [np.random.default_rng([int(seed), j, n_perm, group_index]) for j in targets]


class MSCRMethod(CITestBase):
    """MSCR-v2 arm."""

    name = "mscr"
    version = f"xm-{MSCR_VERSION}-bc"
    method_key = 1

    def default_config(self) -> dict[str, Any]:
        """Primary configuration: frozen constants, `B_MAX` draws, Besag-Clifford h = `BC_H`."""
        c = frozen_config()
        return {
            "nc": c.nc,
            "nb": c.nb,
            "min_stratum": c.min_stratum,
            "n_perm": B_MAX,
            "bc_h": BC_H,
            "device": "cpu",
            "n_jobs": None,
            "seed": None,
        }

    def native_config(self) -> dict[str, Any]:
        """The v2 configuration: B = 2999 fixed, arm native."""
        return {
            **self.default_config(),
            "n_perm": frozen_config().n_perm,
            "bc_h": None,
            "arm": "native",
        }

    def _test(self, prep: Prepared, data: Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
        m = len(prep.pairs)
        out = {"score": np.zeros(m), "p": np.zeros(m), "k": np.zeros(m, dtype=int)}
        for g, (cols, es) in enumerate(prep.column_groups()):
            self._group(prep, data, cfg, (cols, es, g), out)
        q = frozen_config().q
        native = np.zeros(m, dtype=bool)
        for j in {j for _, j in prep.pairs}:
            es = [
                e
                for e, (_, jj) in enumerate(prep.pairs)
                if jj == j and prep.family[e] != "placebo_conf"
            ]
            native[es] = weighted_step_up(out["p"][es], None, q, reshape=True)
        return {
            "score": out["score"],
            "p": out["p"],
            "declared_native": native,
            "notes": {
                "mc_draws": out["k"].tolist(),
                "native_rule": "per-target BY q=.05 over the target's candidate sources",
            },
        }

    def _group(
        self,
        prep: Prepared,
        data: Dataset,
        cfg: dict[str, Any],
        group: tuple[list[int], list[int], int],
        out: dict[str, np.ndarray],
    ) -> None:
        cols, es, group_index = group
        tested = sorted({prep.pairs[e][0] for e in es})
        x = prep.S[:, tested + [i for i in cols if i not in tested]]
        if x.shape[1] < 2:
            raise ValueError("mscr: need >= 2 source columns")
        mcfg = MSCRConfig(
            cfg["nc"], cfg["nb"], cfg["min_stratum"], cfg["n_perm"], frozen_config().q
        )
        seed = (
            cfg["seed"] if cfg.get("seed") is not None else method_seed(data.seed, self.method_key)
        )
        targets = sorted({prep.pairs[e][1] for e in es})
        rngs = _streams(seed, targets, mcfg.n_perm, group_index)
        run = Execution(resolve_n_jobs(cfg.get("n_jobs")), cfg.get("device", "cpu"))
        banks = build_banks(x, prep.Y[:, targets], mcfg, rngs, run)
        bank_of = dict(zip(targets, banks, strict=True))
        pos = {i: c for c, i in enumerate(tested)}
        for e in es:
            i, j = prep.pairs[e]
            denom, null, strata = bank_of[j]
            c = pos[i]
            out["score"][e] = s_star(x[:, c], c, denom, strata, mcfg)
            out["p"][e], out["k"][e] = sequential_p_from_draws(
                out["score"][e], null_s_star(null, c), cfg["bc_h"]
            )
            progress(cfg, int(np.count_nonzero(out["k"])), len(prep.pairs))

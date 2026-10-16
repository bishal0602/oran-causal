"""PMRT on one-step row corpora (controlled environments, regimes R1-R4).

Hypothesis (a, k): the random part of action a's assignment at row t has no effect on KPI
k at t + 1. The statistic is S = sum_t v_t w_t with predictable past-only ridge residuals
w_t; only a's random part is redrawn. Candidates whose source has no known design are
reported not applicable. Declarations use plain BY at q over the primary candidates.
"""

import dataclasses
import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from oran_causal.core.covariates import design_covariates, info_order
from oran_causal.core.data import Dataset, Result
from oran_causal.core.design import Assignment
from oran_causal.core.pmrt import nonlinear
from oran_causal.core.pmrt.candidates import (
    NOT_APPLICABLE,
    RNG_TAG,
    Overrides,
    bad_targets,
    build_result,
    candidate_rows,
    concurrent_columns,
    covariate_spec,
    declare_families,
    edge_notes,
    primary_family,
    testable,
    tune_tau,
)
from oran_causal.core.pmrt.engine import Redraw, RedrawConfig, crt
from oran_causal.core.ridge import LAMS, AdjustConfig, predictable_weights
from oran_causal.core.scoring import PLACEBO

PMRT_CORE_VERSION = "pmrt-core-v1"
PMRT_CORE_V2_VERSION = "pmrt-core-v2"
STATISTICS = ("linear", "gbm", "rff", "poly")


@dataclasses.dataclass(frozen=True)
class PmrtCoreConfig:
    """Settings of PMRT on rows.

    Attributes:
        B: Maximum number of redraws.
        seq_h: Besag-Clifford stopping count; None for fixed B.
        q: BY level.
        procedure: Declaration procedure.
        huber_c: Huber clip of the weights; None for none.
        hist_lags: Lagged action rows among the covariates.
        concurrent_actions: Adjust for the other designed actions at t.
        covariates: `"eq"` or `"r3"`.
        min_burn: Minimum leading rows with weight 0.
        burn_per_cov: Burn-in rows per covariate.
        growth: Geometric growth of the refit boundaries.
        lams: Ridge lambda grid.
        chunk: Redraws per chunk.
        max_chunk_bytes: Memory cap of one chunk.
        split: RNG split index.
        tau: Secondary score threshold.
        statistic: `"linear"`, `"gbm"`, `"rff"` or `"poly"`.
        adjust: Adjustment of the nonlinear statistics.
    """

    B: int = 9999
    seq_h: int | None = 20
    q: float = 0.05
    procedure: str = "by"
    huber_c: float | None = None
    hist_lags: int = 2
    concurrent_actions: bool = True
    covariates: str = "eq"
    min_burn: int = 30
    burn_per_cov: float = 1.0
    growth: float = 1.25
    lams: tuple[float, ...] = LAMS
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    split: int = 0
    tau: float | None = None
    statistic: str = "linear"
    adjust: str | None = None

    def adjust_config(self) -> AdjustConfig:
        """Expanding-window ridge settings of this configuration."""
        return AdjustConfig(self.min_burn, self.burn_per_cov, self.growth, self.lams, self.huber_c)

    def redraw_config(self) -> RedrawConfig:
        """Redraw settings of this configuration."""
        return RedrawConfig(self.B, self.chunk, self.max_chunk_bytes, self.seq_h)


def config_from_dict(cfg: Mapping[str, Any] | None) -> PmrtCoreConfig:
    """Config from a dict; unknown keys are ignored and `lams` becomes a tuple."""
    fields = {f.name for f in dataclasses.fields(PmrtCoreConfig)}
    kw: dict[str, Any] = {
        k: (tuple(v) if k == "lams" else v) for k, v in dict(cfg or {}).items() if k in fields
    }
    return PmrtCoreConfig(**kw)


class _LinearRun:
    def __init__(self, data: Dataset, cfg: PmrtCoreConfig, overrides: Overrides):
        self.data, self.cfg, self.over = data, cfg, overrides
        self.order = info_order(data)
        y = np.asarray(data.Y, float)[self.order]
        self.bad_t = bad_targets(y)
        self.yf = np.where(np.isfinite(y), y, 0.0)
        self.cov_names: tuple[str, ...] = ()
        self.xc = np.zeros((data.n, 0))
        if overrides.weights is None:
            spec = covariate_spec(cfg.covariates, cfg.hist_lags)
            self.cov_names, m, _ = design_covariates(data, spec)
            self.xc = m[self.order]
        self.a_ord = np.asarray(data.X_action, float)[self.order]
        self.shared: tuple[np.ndarray, dict] | None = None
        self.winfo: dict[str, dict] = {}

    def weights(self, ai: int, a: str) -> np.ndarray:
        if self.over.weights is not None:
            self.winfo[a] = {"override": True}
            return np.asarray(self.over.weights, float)
        conc = concurrent_columns(self.data, ai) if self.cfg.concurrent_actions else []
        if conc:
            x = np.column_stack([self.xc, self.a_ord[:, conc]])
            w, info = predictable_weights(x, self.yf, self.cfg.adjust_config())
            self.winfo[a] = dict(info, concurrent=[self.data.action_names[j] for j in conc])
            return w
        if self.shared is None:
            self.shared = predictable_weights(self.xc, self.yf, self.cfg.adjust_config())
        self.winfo[a] = dict(self.shared[1], concurrent=[])
        return self.shared[0]

    def test_action(self, ai: int, a: str, asg: Assignment) -> dict:
        w = np.where(self.bad_t[None, :], 0.0, self.weights(ai, a))
        seed = [int(self.data.seed), RNG_TAG, ai, int(self.cfg.split)]
        rng = self.over.rngs.get(a) or np.random.default_rng(seed)
        order = self.order

        def draw(r: np.random.Generator, b: int) -> np.ndarray:
            return asg.draw(r, b)[:, order]

        redraw = Redraw(asg.v[order], asg.var[order], draw)
        return crt(redraw, w, rng, self.cfg.redraw_config())


class PmrtCore:
    """PMRT on row corpora (`oran_causal.core.data.Method`)."""

    name = "pmrt_core"
    version = PMRT_CORE_VERSION

    def __init__(self, cfg: PmrtCoreConfig | None = None):
        """Uses `cfg`, or the defaults."""
        self.cfg = cfg or PmrtCoreConfig()

    def tune(self, dev: list[Dataset], truth_free: bool = True) -> dict[str, Any]:
        """Default config plus the placebo-conformal threshold tau over `dev` (truth unused)."""
        cfg, results = tune_tau(self, dev)
        scores = sorted(
            (
                e.score
                for r in results
                for e in r.edges
                if e.source == PLACEBO and math.isfinite(e.score)
            ),
            reverse=True,
        )
        cfg["tune_info"] = {
            "n_dev": len(dev),
            "n_placebo_scores": len(scores),
            "placebo_top3": scores[:3],
            "truth_free": True,
        }
        return cfg

    def run(
        self,
        data: Dataset,
        config: Mapping[str, Any] | None = None,
        overrides: Overrides | None = None,
    ) -> Result:
        """Scores every candidate of `data`.

        Args:
            data: Row corpus.
            config: Config dict; None uses the instance config.
            overrides: Fixed streams or weights for equivalence checks.

        Returns:
            The `Result`; a nonlinear statistic is reported as version `pmrt-core-v2`.
        """
        t_cpu = time.process_time()
        cfg = config_from_dict(config) if config is not None else self.cfg
        if cfg.statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, got {cfg.statistic!r}")
        if cfg.statistic != "linear":
            return self._run_nonlinear(data, cfg, t_cpu, overrides)
        if cfg.adjust is not None:
            raise ValueError(
                "adjust applies to the nonlinear statistics only (linear uses the ridge residual)"
            )
        covariate_spec(cfg.covariates, cfg.hist_lags)
        run = _LinearRun(data, cfg, overrides or Overrides())
        laws, na = testable(data)
        per_action = {
            data.action_names[ai]: run.test_action(ai, data.action_names[ai], asg)
            for ai, asg in laws.items()
        }
        rows, stats = candidate_rows(data, per_action, na, run.bad_t, two_sided=True)
        dec = declare_families(stats, data, cfg.procedure, cfg.q)
        edges, reasons, tau_decl = build_result(data, rows, dec, cfg.tau)
        family, diag = primary_family(data)
        notes = {
            NOT_APPLICABLE: reasons,
            "z": edge_notes(data, per_action, "z"),
            "p_plus": edge_notes(data, per_action, "p_plus"),
            "p_minus": edge_notes(data, per_action, "p_minus"),
            "adjust": run.winfo,
            "covariates": {"set": cfg.covariates, "base": list(run.cov_names)},
            "n_tested": len(stats),
            "by_family_m": len(family),
            "by_diag_m": len(diag),
            "declared_tau": tau_decl if cfg.tau is not None else None,
            "rng": f"default_rng([seed, {RNG_TAG}, action_index, {cfg.split}])",
        }
        return Result(
            self.name,
            self.version,
            edges,
            time.process_time() - t_cpu,
            dataclasses.asdict(cfg),
            notes,
        )

    def _run_nonlinear(
        self, data: Dataset, cfg: PmrtCoreConfig, t_cpu: float, overrides: Overrides | None
    ) -> Result:
        if cfg.huber_c is not None or overrides is not None:
            raise ValueError("huber_c / overrides apply to the linear statistic only")
        fields = {f.name for f in dataclasses.fields(nonlinear.PmrtNlConfig)}
        shared = fields & {f.name for f in dataclasses.fields(cfg)}
        res = nonlinear.PmrtNl(nonlinear.PmrtNlConfig(**{k: getattr(cfg, k) for k in shared})).run(
            data
        )
        notes = dict(
            res.notes, statistic_module=f"{nonlinear.__name__} {nonlinear.PMRT_NL_VERSION}"
        )
        cpu = time.process_time() - t_cpu
        return Result(
            self.name, PMRT_CORE_V2_VERSION, res.edges, cpu, dataclasses.asdict(cfg), notes
        )

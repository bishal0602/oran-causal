"""F_max calibration of the Study 4 validity rule (PROTOCOL_A section 10, ruling R-30).

`dependence`: on DEV seeds 3_000_100-3_000_199 (n 1000, kappa .25) every primary null
candidate is tested with a fast partial-correlation z given the equal-information covariates;
the between-candidate correlation of z by pair type, the across-lambda correlation in E4 R3 and
a nested-n check feed the simulation (`analysis.controlled.fmax.fmax_simulate`). `table` and
`se4`: the frozen F_max table and the E4 seed-count rule.
"""

import itertools
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from oran_causal.analysis.controlled.fmax import fmax_simulate
from oran_causal.analysis.controlled.protocol import FMAX_NSIM, FMAX_SEED
from oran_causal.core.covariates import CovariateSpec, design_covariates
from oran_causal.core.data import Dataset
from oran_causal.experiments.harness.records import clean
from oran_causal.sim.controlled.datasets.generate import generate_dataset
from oran_causal.sim.controlled.datasets.truth import REGIMES_OF, truth_for

DEP_SEEDS = range(3_000_100, 3_000_200)
N_DEP = 1000
KAPPA = 0.25
LAMS = (0.0, 0.5, 1.0, 1.5)
TABLE_NS = (500, 1000, 4000, 8000, 24000)
WORLDS = ("E1", "E2", "E3", "E4", "E5")
SE4_CANDIDATES = (200, 300, 400, 600)

type Cell = tuple[str, str, float | None, int]
type Candidate = tuple[str, str]
type Generator = Callable[..., tuple[Dataset, Any]]


def _pair_type(a: Candidate, b: Candidate) -> str:
    if a[0] == b[0]:
        return "same_source"
    return "same_target" if a[1] == b[1] else "other"


def null_candidates(world: str, regime: str) -> list[Candidate]:
    """Primary null candidates (action -> KPI, placebos included) of a (world, regime)."""
    t = truth_for(world, regime)
    return sorted(e for e in t.null_edges if not e[0].startswith("K"))


def _residual(d: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v - d @ np.linalg.lstsq(d, v, rcond=None)[0]


def _z_one(data: Dataset, source: str, kpi: str) -> float:
    ai, ki = data.action_names.index(source), data.kpi_names.index(kpi)
    _, cov, _ = design_covariates(data, CovariateSpec(focal=ai))
    d = np.column_stack([np.ones(data.n), cov]) if cov.size else np.ones((data.n, 1))
    rx = _residual(d, data.X_action[:, ai].astype(float))
    ry = _residual(d, data.Y[:, ki].astype(float))
    r = float(np.dot(rx, ry) / np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    df = data.n - 2 - (d.shape[1] - 1)
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    return float(stats.norm.ppf(np.clip(stats.t.cdf(t, df), 1e-15, 1 - 1e-15)))


def z_stats(data: Dataset, cands: Sequence[Candidate]) -> np.ndarray:
    """Partial-correlation z of each candidate given its source's equal-information covariates."""
    return np.array([_z_one(data, s, k) for s, k in cands])


def pair_type_corr(zm: np.ndarray, cands: Sequence[Candidate]) -> dict[str, Any]:
    """Mean correlation of z across seeds over candidate pairs, by pair type."""
    c = np.corrcoef(zm.T) if zm.shape[1] > 1 else np.ones((1, 1))
    acc: dict[str, list[float]] = {"same_source": [], "same_target": [], "other": []}
    for i, j in itertools.combinations(range(len(cands)), 2):
        acc[_pair_type(cands[i], cands[j])].append(c[i, j])
    means = {k: (float(np.mean(v)) if v else 0.0) for k, v in acc.items()}
    return means | {f"n_pairs_{k}": len(v) for k, v in acc.items()}


def _z_matrix(
    cell: tuple[str, str, int, float], cands: Sequence[Candidate], gen: Generator
) -> np.ndarray:
    world, regime, n, lam = cell
    return np.array(
        [z_stats(gen(world, regime, n, s, lam=lam, kappa=KAPPA)[0], cands) for s in DEP_SEEDS]
    )


def _dependence_row(world: str, regime: str, gen: Generator) -> dict[str, Any]:
    cands = null_candidates(world, regime)
    lams = LAMS if (world == "E4" and regime == "R3") else (1.0,)
    zl = {lam: _z_matrix((world, regime, N_DEP, lam), cands, gen) for lam in lams}
    zm = zl[1.0]
    row = {
        "candidates": [f"{a}->{b}" for a, b in cands],
        "z_mean": float(zm.mean()),
        "z_sd": float(zm.std(ddof=1)),
        **pair_type_corr(zm, cands),
    }
    if len(lams) > 1:
        cl = np.zeros((len(lams), len(lams)))
        for j in range(len(cands)):
            cl += np.corrcoef(np.column_stack([zl[lam][:, j] for lam in lams]).T)
        row["lambda_corr"] = (cl / len(cands)).round(4).tolist()
        row["lambdas"] = list(lams)
    return row


def _nested_n_check(gen: Generator) -> dict[str, Any]:
    chk = {}
    for r in ("R1", "R2"):
        cands = null_candidates("E1", r)
        zn = {n: _z_matrix(("E1", r, n, 1.0), cands, gen) for n in (500, 1000, 4000)}
        for a, b in ((500, 1000), (1000, 4000), (500, 4000)):
            cc = float(
                np.mean([np.corrcoef(zn[a][:, j], zn[b][:, j])[0, 1] for j in range(len(cands))])
            )
            chk[f"E1|{r}|{a}-{b}"] = {
                "corr": round(cc, 4),
                "brownian": round(float(np.sqrt(a / b)), 4),
            }
    return chk


def dependence(
    gen: Generator = generate_dataset, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """Dependence inputs of the simulation from DEV seeds 3_000_100-3_000_199."""
    out: dict[str, Any] = {
        "seeds": [DEP_SEEDS.start, DEP_SEEDS.stop - 1],
        "n": N_DEP,
        "kappa": KAPPA,
        "worlds": {},
    }
    for w in WORLDS:
        for r in (r for r in REGIMES_OF[w] if r != "R4"):
            out["worlds"][f"{w}|{r}"] = _dependence_row(w, r, gen)
            log(f"{w} {r}")
    out["nested_n_check"] = _nested_n_check(gen)
    return out


def table_sets() -> dict[str, list[Cell]]:
    """The pre-registered cell sets of the F_max table."""
    e4 = {w: (1.0 if w == "E4" else None) for w in WORLDS}
    return {
        "R1R2_all": [(w, r, e4[w], n) for w in WORLDS for r in ("R1", "R2") for n in TABLE_NS],
        "R1_all": [(w, "R1", e4[w], n) for w in WORLDS for n in TABLE_NS],
        "R2_all": [(w, "R2", e4[w], n) for w in WORLDS for n in TABLE_NS],
        "R1R2_E3": [("E3", r, None, n) for r in ("R1", "R2") for n in TABLE_NS],
        "R1_E3": [("E3", "R1", None, n) for n in TABLE_NS],
        "E4R3": [("E4", "R3", lam, n) for lam in LAMS for n in TABLE_NS],
    }


def table(dep: dict[str, Any] | None, sets: Iterable[str] | None = None) -> dict[str, Any]:
    """F_max and leg probabilities for the pre-registered cell sets (E4 R3 at S 200)."""
    out: dict[str, Any] = {"nsim": FMAX_NSIM, "sim_seed": FMAX_SEED, "sets": {}}
    all_sets = table_sets()
    for name in all_sets if sets is None else sets:
        keys = ("plac_raw", "conf_raw") if name == "E4R3" else ("null_raw", "plac_raw")
        for s in (200,) if name == "E4R3" else range(40, 101, 10):
            out["sets"][f"{name}|S{s}"] = fmax_simulate(all_sets[name], s, keys, dep)
    return out


def se4(dep: dict[str, Any] | None) -> dict[str, Any]:
    """S_E4: the smallest candidate S at which an exactly nominal arm passes C3 w.p. >= .90."""
    cells = [("E4", "R3", lam, n) for lam in LAMS for n in TABLE_NS]
    out: dict[str, Any] = {"candidates": {}}
    for s in SE4_CANDIDATES:
        out["candidates"][str(s)] = fmax_simulate(cells, s, ("plac_raw", "conf_raw"), dep)
    out["S_E4"] = next((int(s) for s, r in out["candidates"].items() if r["p_pass"] >= 0.90), None)
    return out


def write_json(result: dict[str, Any], path: Path) -> None:
    """Writes a result as sorted, indented JSON with a trailing newline (the frozen file format)."""
    path.write_text(
        json.dumps(clean(result), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

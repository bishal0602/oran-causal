"""Paired power comparison of the primary PMRT arm with the other arms, and seed counts.

All numbers come from per-seed recall of the cell summaries (`cells.build_cells`).
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from oran_causal.analysis.controlled.cells import Cells, build_cells, counts, screen
from oran_causal.analysis.controlled.protocol import (
    FOCAL,
    PRIMARY_KAPPA,
    S_CAP,
    S_FLOOR,
    T1_NS,
)
from oran_causal.analysis.controlled.spec import Spec, focal_arm
from oran_causal.analysis.records import Record

PAIRED_ALPHA = 0.05
PAIRED_POWER = 0.8
PAIRED_DELTA = 0.15
PAIRED_S_MAX = 100_000


def paired_seeds(
    sd_d: float,
    delta: float = PAIRED_DELTA,
    alpha: float = PAIRED_ALPHA,
    power: float = PAIRED_POWER,
    s_max: int = PAIRED_S_MAX,
) -> int:
    """Seeds for a two-sided paired t-test to detect a mean recall gap `delta`.

    Args:
        sd_d: Standard deviation of the per-seed paired difference.
        delta: Gap to detect.
        alpha: Test level.
        power: Target power.
        s_max: Upper bound of the search.

    Returns:
        The smallest seed count, at least 2.
    """
    if not np.isfinite(sd_d) or sd_d <= 0:
        return 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    s = max(2, math.ceil(((za + zb) * sd_d / delta) ** 2))
    while s < s_max:
        needed = (
            (stats.t.ppf(1 - alpha / 2, s - 1) + stats.t.ppf(power, s - 1)) * sd_d / delta
        ) ** 2
        if s >= needed:
            return s
        s += 1
    return s_max


def min_detectable_gap(
    sd_d: float, s: int, alpha: float = PAIRED_ALPHA, power: float = PAIRED_POWER
) -> float:
    """Returns the smallest mean gap a paired t-test with `s` seeds detects."""
    return float(
        (stats.t.ppf(1 - alpha / 2, s - 1) + stats.t.ppf(power, s - 1)) * sd_d / math.sqrt(s)
    )


def achieved_power(
    sd_d: float, s: int, delta: float = PAIRED_DELTA, alpha: float = PAIRED_ALPHA
) -> float:
    """Returns the power of the two-sided paired t-test with `s` seeds for a mean gap `delta`."""
    if not np.isfinite(sd_d) or sd_d <= 0:
        return 1.0
    df, nc = s - 1, delta * math.sqrt(s) / sd_d
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def paired_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[float]:
    """Returns `a[seed] - b[seed]` over the shared seeds in order, skipping None and NaN."""
    seeds = sorted(
        (
            s
            for s in set(a) & set(b)
            if a[s] is not None
            and b[s] is not None
            and not math.isnan(a[s])
            and not math.isnan(b[s])
        ),
        key=int,
    )
    return [a[s] - b[s] for s in seeds]


def gate_ok(validity: str | None, gate: str) -> bool:
    """Tells whether a validity passes a gate: `VALID` only, or anything but INVALID."""
    if gate == "VALID":
        return validity == "VALID"
    return validity in ("VALID", "INCONCLUSIVE", "NO_READ")


def _focal_summary(focal_cell: dict[str, Any], declare: str) -> tuple[dict[str, Any] | None, str]:
    if declare == "by":
        return focal_cell["primary"], "BY"
    return focal_cell.get("secondary_tau"), "tau"


def _diff_row(key: str, cell: dict[str, Any], rule: str, diffs: np.ndarray) -> dict[str, Any]:
    sd = float(np.std(diffs, ddof=1))
    half = float(stats.t.ppf(0.975, len(diffs) - 1) * sd / math.sqrt(len(diffs))) if sd > 0 else 0.0
    return {
        "cell": key.split("|", 1)[1],
        "arm": cell["arm"],
        "rule": rule,
        "block": cell["block"],
        "world": cell["world"],
        "regime": cell["regime"],
        "lam": cell["lam"],
        "n": cell["n"],
        "kappa": cell["kappa"],
        "descriptive_only": cell["world"] == "E4",
        "n_seeds": len(diffs),
        "mean_d": float(diffs.mean()),
        "sd_d": sd,
        "ci": [float(diffs.mean()) - half, float(diffs.mean()) + half],
    }


def paired_diffs(
    cells: Cells, gate: str = "not_invalid", focal: str = FOCAL
) -> list[dict[str, Any]]:
    """Recall of the focal PMRT arm minus each other arm, per cell, under the same rule.

    Both arms must pass `gate` (`not_invalid` by default, sensitivity read `VALID`). P arms are
    compared with the focal arm's BY declarations, score-only arms with the focal arm scored
    with tau. Per-cell 95 % paired-t interval, descriptive only. Cells where either arm's power
    is not applicable have no per-seed recall and are left out.

    Args:
        cells: Cells of the primary kappa.
        gate: Validity gate for both arms.
        focal: The primary PMRT arm.

    Returns:
        One row per compared cell.
    """
    out = []
    for key, cell in sorted(cells.items()):
        if (
            cell["arm"] == focal
            or not counts(cell)
            or not gate_ok(cell["primary"]["validity"], gate)
        ):
            continue
        focal_cell = cells.get(f"{focal}|{key.split('|', 1)[1]}")
        if focal_cell is None or not counts(focal_cell):
            continue
        if not gate_ok(focal_cell["primary"]["validity"], gate):
            continue
        summary, rule = _focal_summary(focal_cell, cell["declare"])
        if not summary or not gate_ok(summary["validity"], gate):
            continue
        diffs = np.array(
            paired_differences(summary["per_seed_recall"], cell["primary"]["per_seed_recall"])
        )
        if len(diffs) < 2:
            continue
        out.append(_diff_row(key, cell, rule, diffs))
    return out


def _t1_pairs(cells: Cells, focal: str) -> list[dict[str, Any]]:
    pairs = []
    for key, cell in sorted(cells.items()):
        if (
            cell["arm"] == focal
            or cell["block"] != "primary"
            or cell["kappa"] != PRIMARY_KAPPA
            or cell["n"] not in T1_NS
            or cell["world"] == "E4"
            or not counts(cell)
            or cell["primary"]["validity"] == "INVALID"
        ):
            continue
        focal_cell = cells.get(f"{focal}|{key.split('|', 1)[1]}")
        if (
            focal_cell is None
            or not counts(focal_cell)
            or focal_cell["primary"]["validity"] == "INVALID"
        ):
            continue
        summary = (
            focal_cell["primary"] if cell["declare"] == "by" else focal_cell.get("secondary_tau")
        )
        if not summary or summary["validity"] in ("INVALID", "few_seeds"):
            continue
        d = paired_differences(summary["per_seed_recall"], cell["primary"]["per_seed_recall"])
        if len(d) < 2:
            continue
        sd = float(np.std(d, ddof=1))
        pairs.append(
            {
                "cell": key.split("|", 1)[1],
                "arm": cell["arm"],
                "n_seeds": len(d),
                "sd_d": sd,
                "seeds_needed": paired_seeds(sd),
            }
        )
    return pairs


def t1_seed_count(
    records: list[Record], spec: Spec, s_cap: int = S_CAP, focal: str | None = None
) -> dict[str, Any]:
    """Chooses the Study 4 seed count from DEV records (protocol T1).

    Per kappa-.25 cell at n in `T1_NS` (E4 excluded: one true edge), every pair of the focal
    PMRT arm and a primary-block arm under the same declaration rule, neither INVALID on DEV,
    gives `paired_seeds(sd_d)`; `S_power` is the maximum and
    `S = min(s_cap, max(40, S_power rounded up to a multiple of 10))`. The minimum detectable
    gap and the achieved power at S are reported per pair for when the cap binds.

    Args:
        records: DEV campaign records.
        spec: DEV campaign spec.
        s_cap: Compute ceiling on the seed count.
        focal: The PMRT arm of the pairs; defaults to the spec's focal arm.

    Returns:
        `S`, `S_power`, whether the cap binds, the pair count and the largest pairs.
    """
    focal = focal or focal_arm(spec)
    use, _ = screen(records, spec)
    cells, _ = build_cells(use, spec)
    pairs = _t1_pairs(cells, focal)
    s_power = max((p["seeds_needed"] for p in pairs), default=None)
    s = S_FLOOR if s_power is None else min(s_cap, max(S_FLOOR, 10 * math.ceil(s_power / 10)))
    for p in pairs:
        p["mdg_at_S"] = min_detectable_gap(p["sd_d"], s) if p["sd_d"] > 0 else 0.0
        p["power_at_S"] = achieved_power(p["sd_d"], s)
    worst = sorted(pairs, key=lambda p: (-p["seeds_needed"], p["cell"], p["arm"]))[:15]
    return {
        "S": s,
        "focal": focal,
        "S_power": s_power,
        "cap": s_cap,
        "cap_binds": bool(s_power is not None and s_power > s_cap),
        "n_pairs": len(pairs),
        "largest": worst,
        "max_mdg_at_S": max((p["mdg_at_S"] for p in pairs), default=None),
        "min_power_at_S": min((p["power_at_S"] for p in pairs), default=None),
    }

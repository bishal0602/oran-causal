"""Extras specs: the post hoc X2, X3, X5 and X6 campaigns.

An extras spec has an `extras` block naming its experiment. It runs in DEV mode on its own seed
block (X2 and X3 `XMETHOD_EXTRAS`, X5 and X6 `XMETHOD_DIAG`), never changes the C1-C3 verdicts, and
only records of the same spec (canonical sha) are accepted on merge or relaunch.
"""

import math
from typing import Any

from oran_causal.analysis.posthoc.x2x3 import ABLATION_KEYS, EXTRAS_VERSION, check_told
from oran_causal.experiments.harness.runner import seed_range
from oran_causal.sim.controlled.datasets.generate import DITHER_ENVELOPE

type Spec = dict[str, Any]

EXTRAS_SEEDS = range(3_200_000, 3_200_200)
X5_SEEDS = (3_300_000, 3_301_999)
X6_SEEDS = (3_302_000, 3_302_199)
SEED_BLOCKS: dict[str, tuple[str, range]] = {
    "X2": ("XMETHOD_EXTRAS", EXTRAS_SEEDS),
    "X3": ("XMETHOD_EXTRAS", EXTRAS_SEEDS),
    "X5": ("XMETHOD_DIAG", range(X5_SEEDS[0], X5_SEEDS[1] + 1)),
    "X6": ("XMETHOD_DIAG", range(X6_SEEDS[0], X6_SEEDS[1] + 1)),
}


def is_extras(spec: Spec | None) -> bool:
    """Whether the spec has an `extras` block."""
    return isinstance((spec or {}).get("extras"), dict)


def seed_list(seeds: Any, spec: Spec) -> list[int]:
    """Seeds of an extras block: its experiment's own range only.

    Raises:
        ValueError: A seed is not a number or lies outside the experiment's seed block.
    """
    if isinstance(seeds, str) or any(isinstance(x, str) for x in seeds):
        raise ValueError(f"seeds must be numbers: {seeds!r}")
    experiment = spec["extras"].get("experiment")
    name, allowed = SEED_BLOCKS[experiment]
    out = seed_range(seeds)
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(
            f"{experiment} specs use seeds {allowed.start}-{allowed.stop - 1} only ({name}): "
            f"{bad[:5]}"
        )
    return out


def check_design(design: dict[str, Any]) -> tuple[float, int]:
    """`(delta, n_blocks)` of an X2 dither design.

    Raises:
        ValueError: The dither would leave the setpoint envelope, or the block count is invalid.
    """
    delta, n_blocks = float(design["delta"]), int(design["n_blocks"])
    if not (math.isfinite(delta) and delta > 0.0 and DITHER_ENVELOPE + delta < 0.5):
        raise ValueError(f"dither delta must be in (0, {0.5 - DITHER_ENVELOPE}): {delta}")
    if n_blocks < 1 or n_blocks != design["n_blocks"]:
        raise ValueError(f"n_blocks must be an integer >= 1: {design['n_blocks']!r}")
    return delta, n_blocks


def validate(spec: Spec) -> None:
    """Checks an extras spec.

    X2 has a design and no told arms; X3 and X6 have told arms and no design; X5 has neither;
    every block's seeds lie in the experiment's block.

    Raises:
        ValueError: The spec is not a valid extras spec.
    """
    if not is_extras(spec):
        raise ValueError("not an extras spec (no 'extras' block)")
    extras = spec["extras"]
    experiment = extras.get("experiment")
    if experiment not in SEED_BLOCKS:
        raise ValueError(f"extras.experiment must be one of {tuple(SEED_BLOCKS)}")
    told = {a: d["told"] for a, d in spec["arms"].items() if d.get("told") is not None}
    if experiment == "X2":
        if not extras.get("design") or told:
            raise ValueError("X2 spec: needs extras.design and no told arms")
        check_design(extras["design"])
    else:
        if extras.get("design"):
            raise ValueError(f"{experiment} spec: the data stay as generated (no extras.design)")
        if experiment == "X5" and told:
            raise ValueError("X5 spec: frozen arms only (no told arms)")
        for t in told.values():
            check_told(t)
            if experiment != "X6" and any(k in t for k in ABLATION_KEYS):
                raise ValueError("centre / redraw are X6 only")
    for block in spec.get("blocks", []):
        seed_list(block["seeds"], spec)


def stamp(spec: Spec, protocol_sha256: str | None) -> dict[str, Any]:
    """The `integrity.extras` block of the spec's records."""
    extras = spec["extras"]
    experiment = extras["experiment"]
    return {
        "version": EXTRAS_VERSION,
        "experiment": experiment,
        "design": extras.get("design"),
        "seed_block": SEED_BLOCKS[experiment][0],
        "protocol_sha256": protocol_sha256,
    }

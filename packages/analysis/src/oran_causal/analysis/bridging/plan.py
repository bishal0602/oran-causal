"""Datasets of the bridging study: the Study 1 evaluation slices, their dataset seeds and the GT.

The bridging study runs the Study 4 method set on the frozen Study 1 (v4) collection. One
dataset is the units of one knob family in one slice: the pooled 600 evaluation episodes, the
disjoint 60-, 120- and 300-episode slices of the v4 analyzer, or the 40 placebo-log episodes.
A dataset's seed is 60 000 000 + 10 x its v4 split + the family index. The truth is the stored
v4 GT block, one cell per (family, relation, KPI).
"""

import dataclasses
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from oran_causal.sim.system.decision.vocab import FAMILIES, KPIS, RELATIONS

WORLD = "E6"
TARGETS = tuple((r, k) for r in RELATIONS for k in KPIS)
KPI_NAMES = tuple(f"K_{r}_{k}" for r, k in TARGETS)
PLACEBO = "P_placebo"
SEEDS_V4 = {"dev": (188000, 20), "eval": (188100, 600), "placebo": (188700, 40), "gt": (188800, 40)}
SPLIT_V4 = {"pooled": 0, "60": 20, "120": 40, "300": 60, "placebo": 9}
SIZED_KINDS = ("60", "120", "300")
DEFAULT_KINDS = ("60", "120", "300", "pooled")
DATASET_SEED_BASE = 60_000_000

type Hyp = tuple[str, str, str]
type Spec = Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Slice:
    """One slice of episodes.

    Attributes:
        name: `pooled`, `s<size>_<i>` or `placebo`.
        kind: `pooled`, `60`, `120`, `300` or `placebo`.
        split: v4 split index (the dataset seed's split part).
        stage: Record stage of its episodes, `eval` or `placebo`.
        seeds: Episode seeds.
    """

    name: str
    kind: str
    split: int
    stage: str
    seeds: tuple[int, ...]


def action_name(family: str) -> str:
    """The action column of a knob family."""
    return f"A_{family}"


def dataset_seed(split: int, family: str) -> int:
    """Dataset identity: 60 000 000 + 10 x v4 split + family index, disjoint from Study 4's."""
    return DATASET_SEED_BASE + 10 * int(split) + FAMILIES.index(family)


def _block(lo: int, n: int) -> tuple[int, ...]:
    return tuple(range(lo, lo + n))


def slice_plan(kinds: Iterable[str] = DEFAULT_KINDS) -> list[Slice]:
    """The v4 analyzer's evaluation slices of `kinds` (pooled first), then the placebo logs."""
    kinds = set(kinds)
    lo, n = SEEDS_V4["eval"]
    out = []
    if "pooled" in kinds:
        out.append(Slice("pooled", "pooled", SPLIT_V4["pooled"], "eval", _block(lo, n)))
    for kind in (k for k in SIZED_KINDS if k in kinds):
        size = int(kind)
        out += [
            Slice(f"s{kind}_{i}", kind, SPLIT_V4[kind] + i, "eval", _block(lo + i * size, size))
            for i in range(n // size)
        ]
    if "placebo" in kinds:
        out.append(
            Slice(
                "placebo", "placebo", SPLIT_V4["placebo"], "placebo", _block(*SEEDS_V4["placebo"])
            )
        )
    return out


def spec_families(spec: Spec) -> tuple[str, ...]:
    """The spec's knob families (default: all)."""
    return tuple(spec.get("families", FAMILIES))


def spec_slices(spec: Spec) -> list[Slice]:
    """The spec's slices: its `slices` kinds, narrowed by `slice_names`, minus `exclude_slices`."""
    out = slice_plan(spec["slices"])
    if spec.get("slice_names"):
        names = set(spec["slice_names"])
        out = [s for s in out if s.name in names]
    excluded = set(spec.get("exclude_slices", []))
    return [s for s in out if s.name not in excluded]


def seed_index(spec: Spec) -> dict[int, tuple[Slice, str]]:
    """(slice, family) of every dataset seed of the spec."""
    return {
        dataset_seed(s.split, f): (s, f) for s in spec_slices(spec) for f in spec_families(spec)
    }


def kinds_of(spec: Spec) -> dict[str, list[str]]:
    """Slice names per kind, in slice-plan order."""
    kinds: dict[str, list[str]] = {}
    for s in spec_slices(spec):
        kinds.setdefault(s.kind, []).append(s.name)
    return kinds


def load_gt(path: str | Path) -> dict[Hyp, dict[str, Any]]:
    """{(family, relation, kpi): cell} of a stored v4 GT block (`analysis_v4.json["gt"]`)."""
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    g = g.get("gt", g)
    return {(c["family"], c["relation"], c["kpi"]): c for c in g["cells"]}


def hyp_of(source: str, target: str) -> Hyp | None:
    """The v4 hypothesis of an (action, KPI) edge; None for the placebo column."""
    if source == PLACEBO:
        return None
    rel, kpi = target[2:].split("_", 1)
    return (source[2:], rel, kpi)

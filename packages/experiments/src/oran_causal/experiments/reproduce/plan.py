"""Tiers t2 and t3 as a dry run: shard plan and cost per platform from `manifests/tiers.yaml`.

Costs are reference CPU-hours (the Kaggle CPU host). A platform's host CPU-hours are the
reference hours divided by its speed factor; its wall time spreads them over its sessions and
processes. Study 4 is priced unit by unit from its spec and the frozen cost table and is cut
into one part per VPS process by the campaign partition.
"""

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from oran_causal.experiments.campaign.shard import partition, unit_cost
from oran_causal.experiments.campaign.spec import ProtocolTexts, expand, shipped_protocol
from oran_causal.experiments.paths import MANIFESTS, MEMBER_ROOT

TIERS_MANIFEST = MANIFESTS / "tiers.yaml"
PLAN_TIERS = ("t2", "t3")


@dataclasses.dataclass(frozen=True)
class Platform:
    """Capacity of one platform lane.

    Attributes:
        name: `kaggle`, `colab` or `vps`.
        factor: Speed relative to the reference host.
        sessions: Concurrent sessions.
        procs: Processes per session.
        session_h: Wall limit of one session.
    """

    name: str
    factor: float
    sessions: int
    procs: int
    session_h: float

    @property
    def slots(self) -> int:
        """Concurrent processes over all sessions."""
        return self.sessions * self.procs


@dataclasses.dataclass(frozen=True)
class Row:
    """One priced stage of a tier.

    Attributes:
        item: Study or post hoc item.
        stage: Stage name.
        platform: Where it runs.
        low: Reference CPU-h, lower end.
        high: Reference CPU-h, upper end (equal to `low` unless extrapolated).
        basis: `M` measured, `P` protocol estimate, `E` extrapolation.
    """

    item: str
    stage: str
    platform: str
    low: float
    high: float
    basis: str


@dataclasses.dataclass(frozen=True)
class Shards:
    """The Study 4 partition on its platform.

    Attributes:
        platform: Where the parts run.
        units: Units in the tier.
        datasets: Dataset groups among them.
        part_ref_h: Reference CPU-h of each part.
    """

    platform: str
    units: int
    datasets: int
    part_ref_h: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class TierPlan:
    """What a tier would run, priced.

    Attributes:
        tier: `t2` or `t3`.
        rows: Priced stages.
        frozen: (item, stage) pairs reused from the frozen records in this tier.
        platforms: The lanes used, by name.
        shards: The Study 4 partition, if Study 4 is in the plan.
    """

    tier: str
    rows: tuple[Row, ...]
    frozen: tuple[tuple[str, str], ...]
    platforms: Mapping[str, Platform]
    shards: Shards | None


def load_manifest(path: Path = TIERS_MANIFEST) -> dict[str, Any]:
    """The tier manifest."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def platforms(manifest: Mapping[str, Any]) -> dict[str, Platform]:
    """The manifest's platform lanes."""
    return {
        name: Platform(name, float(manifest["factors"][name]), **cap)
        for name, cap in manifest["platforms"].items()
    }


def _range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, Sequence):
        low, high = value
        return float(low), float(high)
    return float(value), float(value)


def study4_units(campaign: Mapping[str, Any], tier: str, texts: ProtocolTexts) -> list[Any]:
    """The Study 4 units of `tier`: the spec's units less the `n` the tier drops."""
    spec = json.loads((MEMBER_ROOT / campaign["spec"]).read_text(encoding="utf-8"))
    drop = set(campaign.get("drop_n_in", {}).get(tier, []))
    return [u for u in expand(spec, texts) if u.n not in drop]


def _campaign(
    item: Mapping[str, Any], tier: str, lane: Platform, texts: ProtocolTexts
) -> tuple[Row, Shards]:
    campaign = item["campaign"]
    table = json.loads((MEMBER_ROOT / campaign["cost_table"]).read_text(encoding="utf-8"))
    units = study4_units(campaign, tier, texts)
    parts = partition(units, lane.slots, table)
    part_h = tuple(sum(unit_cost(u, table) for u in p) / 3600 for p in parts)
    total = sum(unit_cost(u, table) for u in units) / 3600
    row = Row(item["name"], "campaign", lane.name, total, total, "M")
    return row, Shards(lane.name, len(units), len({u.dataset for u in units}), part_h)


def plan(
    tier: str,
    manifest: Mapping[str, Any] | None = None,
    platform: str | None = None,
    texts: ProtocolTexts | None = None,
) -> TierPlan:
    """Prices `tier`, optionally only the items assigned to `platform`.

    Raises:
        ValueError: `tier` is not t2 or t3, or `platform` is not in the manifest.
    """
    if tier not in PLAN_TIERS:
        raise ValueError(f"tier {tier!r}: the dry run covers {', '.join(PLAN_TIERS)}")
    manifest = manifest if manifest is not None else load_manifest()
    lanes = platforms(manifest)
    if platform is not None and platform not in lanes:
        raise ValueError(f"platform {platform!r}: expected one of {', '.join(lanes)}")
    rows: list[Row] = []
    frozen: list[tuple[str, str]] = []
    shards = None
    for item in manifest["items"]:
        if tier not in item["tiers"] or platform not in (None, item["platform"]):
            continue
        lane = lanes[item["platform"]]
        if "campaign" in item:
            row, shards = _campaign(item, tier, lane, texts or shipped_protocol())
            rows.append(row)
            continue
        for st in item["stages"]:
            if tier in st.get("frozen_in", []):
                frozen.append((item["name"], st["name"]))
                continue
            low, high = _range(st["ref_cpu_h"])
            rows.append(Row(item["name"], st["name"], lane.name, low, high, st["basis"]))
    used = {r.platform for r in rows}
    return TierPlan(
        tier, tuple(rows), tuple(frozen), {k: v for k, v in lanes.items() if k in used}, shards
    )


def totals(p: TierPlan) -> dict[str, tuple[float, float]]:
    """Reference CPU-h (low, high) per platform and under `all`."""
    out: dict[str, tuple[float, float]] = {}
    for r in p.rows:
        low, high = out.get(r.platform, (0.0, 0.0))
        out[r.platform] = (low + r.low, high + r.high)
    out["all"] = (sum(r.low for r in p.rows), sum(r.high for r in p.rows))
    return out


def wall_h(ref_h: float, lane: Platform) -> float:
    """Wall hours of `ref_h` reference CPU-h spread evenly over the lane's processes."""
    return ref_h / lane.factor / lane.slots


def _span(low: float, high: float, digits: int = 1) -> str:
    return f"{low:.{digits}f}" if math.isclose(low, high) else f"{low:.{digits}f}-{high:.{digits}f}"


def render(p: TierPlan) -> list[str]:
    """The dry-run report as markdown lines."""
    lines = [
        f"# Tier {p.tier} dry run (reference CPU-h; no launch)",
        "",
        "| Item | Stage | Platform | Ref CPU-h | Basis |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.item} | {r.stage} | {r.platform} | {_span(r.low, r.high, 2)} | {r.basis} |"
        for r in p.rows
    ]
    if p.frozen:
        lines += ["", "Frozen in this tier: " + ", ".join(f"{i}.{s}" for i, s in p.frozen)]
    lines += [
        "",
        "| Platform | Ref CPU-h | Factor | Host CPU-h | Processes | Wall h |",
        "|---|---|---|---|---|---|",
    ]
    tot = totals(p)
    for name, lane in p.platforms.items():
        low, high = tot[name]
        host = _span(low / lane.factor, high / lane.factor)
        wall = _span(wall_h(low, lane), wall_h(high, lane))
        lines.append(
            f"| {name} | {_span(low, high, 2)} | {lane.factor} | {host} | {lane.slots} | {wall} |"
        )
    lines.append(f"| all | {_span(*tot['all'], 2)} | | | | |")
    if p.shards is not None:
        s = p.shards
        longest = max(s.part_ref_h) / p.platforms[s.platform].factor
        lines += [
            "",
            f"Study 4 shard plan on {s.platform}: {s.units} units in {s.datasets} datasets, "
            f"{len(s.part_ref_h)} parts of {min(s.part_ref_h):.2f} to "
            f"{max(s.part_ref_h):.2f} ref CPU-h; the longest part takes {longest:.1f} h wall.",
        ]
    return lines

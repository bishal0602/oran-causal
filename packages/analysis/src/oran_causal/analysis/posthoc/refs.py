"""The referee functions the system-study re-analyses need from the methods member.

Analysis may not import `methods`, so the re-analyses that decode a stored map, recompute a
certification or score a map's edges take these callables as an argument. The wiring is
`oran_causal.experiments.posthoc.tools.referee_tools`.
"""

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

Map = dict[tuple[str, str, str], float | None]


@dataclasses.dataclass(frozen=True)
class RefereeTools:
    """Callables of `methods.system.referees` and `methods.system.maps`.

    Attributes:
        map_from_json: Decodes `[[family, relation, kpi, beta], ...]` to a map.
        bounds_from_json: Decodes `[[family, relation, kpi, bound], ...]`, or None.
        tau_from_json: Decodes `[[family, d, kpi, tau], ...]`.
        certify_map: `core.gate.certify` of a map with the deferrable classes of MapGateV2:
            `(map, bounds, tau) -> {"classes", "uncertified"}`.
        can_defer: Whether MapGateV2 of a map defers class `(family, d)`.
        edge_quality: Map edges against GT cells.
        certsafe_signature: Decision signature with the uncertified classes accepted.
        signature_key: Stable string of a signature.
        design_slope: PMRT design slope and standard error of a guard edge on a unit table.
        guard_kpis: Guard KPIs of the gate.
        relations: Relations of a guard edge.
        levels: Dose of every mode of the Study 2 and 3 design (`LEVEL_V2_ARR`).
    """

    map_from_json: Callable[[Iterable[Sequence[Any]]], Map]
    bounds_from_json: Callable[[Iterable[Sequence[Any]] | None], dict | None]
    tau_from_json: Callable[[Iterable[Sequence[Any]]], dict]
    certify_map: Callable[..., dict[str, Any]]
    can_defer: Callable[[Map, str, int], bool]
    edge_quality: Callable[[Map, Iterable[Mapping[str, Any]]], dict[str, Any]]
    certsafe_signature: Callable[..., tuple]
    signature_key: Callable[[tuple], str]
    design_slope: Callable[..., dict[str, Any]]
    guard_kpis: tuple[str, ...]
    relations: tuple[str, ...]
    levels: np.ndarray

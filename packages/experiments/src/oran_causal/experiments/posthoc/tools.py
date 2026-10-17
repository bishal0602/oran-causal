"""Wiring of the methods-member referee functions into the post hoc re-analyses.

`oran_causal.analysis` may not import `methods`; the re-analyses that decode a stored map or
recompute a certification take a `RefereeTools` instead, built here.
"""

from oran_causal.analysis.posthoc.refs import RefereeTools
from oran_causal.methods.system import maps, mscr_crt, referees


def referee_tools() -> RefereeTools:
    """The referee functions of `methods.system` as the analyses' `RefereeTools`."""
    return RefereeTools(
        map_from_json=referees.map_from_json,
        bounds_from_json=referees.bounds_from_json,
        tau_from_json=referees.tau_from_json,
        certify_map=referees.certify_map,
        can_defer=referees.can_defer,
        edge_quality=maps.edge_quality,
        certsafe_signature=referees.certsafe_signature,
        signature_key=referees.signature_key,
        design_slope=referees.design_slope_bound,
        guard_kpis=referees.GUARD_KPIS,
        relations=referees.RELS_V2,
        levels=mscr_crt.LEVEL_V2_ARR,
    )

"""Candidate sets, regimes and ground truth of the controlled environments.

Truth comes from the structural definitions: the environment adjacency lists give the direct
edges, every other candidate is an exact null, and a sign is set only where the mechanism is
monotone in that parent over the whole parameter range.
"""

from oran_causal.core.data import Truth
from oran_causal.sim.controlled import e2, e3, e4
from oran_causal.sim.controlled.base import ControlledEnv, Edges
from oran_causal.sim.controlled.e1 import ADJACENCY_EDGES as E1_EDGES
from oran_causal.sim.controlled.e1 import E1Env
from oran_causal.sim.controlled.e2 import E2Env
from oran_causal.sim.controlled.e3 import E3Env
from oran_causal.sim.controlled.e4 import E4Env
from oran_causal.sim.controlled.e5 import E5Env

PLACEBO = "P_placebo"
PLACEBO_CONF = "P_placebo_conf"
E4_LAMBDAS = (0.0, 0.5, 1.0, 1.5)

ENV_CLASSES: dict[str, type[ControlledEnv]] = {
    "E1": E1Env,
    "E2": E2Env,
    "E3": E3Env,
    "E4": E4Env,
    "E5": E5Env,
}

REGIMES_OF = {
    "E1": ("R1", "R2"),
    "E2": ("R1", "R2"),
    "E3": ("R1", "R2"),
    "E4": ("R1", "R2", "R3", "R4"),
    "E5": ("R1", "R2"),
}

SIGNS: dict[str, dict[tuple[str, str], int]] = {
    "E1": {
        ("P0", "K0"): 1,
        ("P1", "K1"): 1,
        ("P2", "K2"): 1,
        ("P3", "K3"): 1,
        ("K0", "K2"): 1,
        ("K1", "K3"): 1,
    },
    "E2": {("P6", "K5"): -1},
    "E3": {
        ("P0", "K0"): 1,
        ("K0", "K1"): 1,
        ("P1", "K2"): 1,
        ("K1", "K2"): 1,
        ("P2", "K3"): 1,
        ("K1", "K3"): 1,
        ("P3", "K4"): 1,
    },
    "E4": {("P0", "K0"): -1},
    "E5": {("P0", "K2"): 1, ("P0", "K1"): 1, ("K2", "K1"): 1},
}

ROLES: dict[str, dict[str, str]] = {
    "E1": {},
    "E2": {"P0": "shared knob"},
    "E3": {"K1": "conduit", "K4": "non-descendant control"},
    "E4": {"P0": "A", "K0": "K_out"},
    "E5": {
        "P0": "P0 shared knob",
        "P1": "G1 gate axis",
        "P2": "G2 gate axis",
        "P3": "P3 distractor",
        "K0": "K_ben",
        "K1": "K_harm",
        "K2": "K_mid conduit",
        "K3": "K_dist",
    },
}


def candidates_for(world: str, regime: str = "R1") -> tuple[tuple[str, str], ...]:
    """Lists the scored `(source, target)` pairs of a world.

    Every action (including the placebo columns of the regime) times every KPI comes first,
    then every lagged KPI times every KPI.

    Args:
        world: Environment id.
        regime: Data regime id; R3 and R4 add the confounded placebo column.

    Returns:
        The candidate pairs.
    """
    env_cls = ENV_CLASSES[world]
    actions = [f"P{i}" for i in range(env_cls.num_params)] + [PLACEBO]
    if regime in ("R3", "R4"):
        actions.append(PLACEBO_CONF)
    kpis = [f"K{j}" for j in range(env_cls.num_kpis)]
    return tuple((a, k) for a in actions for k in kpis) + tuple((s, k) for s in kpis for k in kpis)


def roles_for(world: str) -> dict[str, str]:
    """Returns the variable roles of a world, for reports only."""
    return dict(ROLES[world])


def _true_edges(world: str) -> Edges:
    if world == "E2":
        return e2.TRUE_ADJACENCY
    if world == "E3":
        return e3.TRUE_ADJACENCY
    if world == "E4":
        return e4.TRUE_ADJACENCY
    if world == "E5":
        return E5Env().adjacency_edges
    return E1_EDGES


def truth_for(world: str, regime: str) -> Truth:
    """Returns the ground truth of a `(world, regime)` pair.

    Args:
        world: Environment id.
        regime: Data regime id; the structure is the same in every regime.

    Returns:
        The true edges, their signs and the exact nulls.

    Raises:
        AssertionError: If the structural definitions are inconsistent with the candidates.
    """
    num_params = ENV_CLASSES[world].num_params
    edges = frozenset(
        (f"P{s}" if s < num_params else f"K{s - num_params}", f"K{k}")
        for k, s in _true_edges(world)
    )
    candidates = set(candidates_for(world, regime))
    if not edges <= candidates:
        raise AssertionError(f"{world}: true edges outside the candidate set")
    signs = dict(SIGNS[world])
    if not set(signs) <= edges:
        raise AssertionError(f"{world}: sign table names a non-edge")
    return Truth(
        world=world,
        regime=regime,
        edges=edges,
        signs=signs,
        null_edges=frozenset(candidates - edges),
    )

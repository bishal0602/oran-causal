from collections.abc import Sequence
from typing import Any

from oran_causal.sim.system import config as C
from oran_causal.sim.system.baselines import (
    CellPriorityLock,
    all_subsets,
    region_hindsight,
    region_subset,
)
from oran_causal.sim.system.env import E6Env

ORDER = ("SliceGuarantee", "ES")


def req(x: str, k: tuple, cur: float, prop: float, t: float = 1.0) -> dict[str, Any]:
    return {"xapp": x, "ver": 1, "knob": k, "cur": cur, "prop": prop, "t": t}


def obs(t: float, reqs: list[dict], config: dict, xapps: Sequence[str] = ORDER) -> dict[str, Any]:
    return {
        "t": float(t),
        "requests": reqs,
        "config": dict(config),
        "new_reports": [],
        "static": {"xapps": {x: [] for x in xapps}},
    }


def test_cell_lock_higher_priority_change_blocks_lower_priority_on_that_cell_for_60s() -> None:
    arb = CellPriorityLock(ORDER)
    cfg = {("prot_min", 3): 0.0, ("carrier", 3): 2.0, ("carrier", 4): 2.0}
    reqs = [req("ES", ("carrier", 3), 2.0, 1.0), req("SliceGuarantee", ("prot_min", 3), 0.0, 0.05)]
    assert arb(obs(100, reqs, cfg))["decisions"] == ["reject", "accept"]
    cfg[("prot_min", 3)] = 0.05
    reqs = [req("ES", ("carrier", 3), 2.0, 1.0), req("ES", ("carrier", 4), 2.0, 1.0)]
    assert arb(obs(101, reqs, cfg))["decisions"] == ["reject", "accept"]
    cfg[("carrier", 4)] = 1.0
    assert arb(obs(159, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))["decisions"] == ["reject"]
    assert arb(obs(160, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))["decisions"] == ["accept"]


def test_cell_lock_never_blocks_the_higher_priority_xapp() -> None:
    arb = CellPriorityLock(ORDER)
    cfg = {("prot_min", 3): 0.0, ("carrier", 3): 2.0}
    assert arb(obs(10, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))["decisions"] == ["accept"]
    cfg[("carrier", 3)] = 1.0
    d = arb(obs(11, [req("SliceGuarantee", ("prot_min", 3), 0.0, 0.05)], cfg))
    assert d["decisions"] == ["accept"]


def test_cell_lock_ignores_accepted_requests_that_did_not_take_effect() -> None:
    arb = CellPriorityLock(ORDER)
    cfg = {("prot_min", 3): 0.0, ("carrier", 3): 2.0}
    arb(obs(10, [req("SliceGuarantee", ("prot_min", 3), 0.0, 0.05)], cfg))
    assert arb(obs(11, [req("ES", ("carrier", 3), 2.0, 1.0)], cfg))["decisions"] == ["accept"]


def test_region_hindsight_picks_the_best_admissible_subset_per_region() -> None:
    cost = {
        0: {("A", "B"): 5, ("A",): 3, ("B",): 4, (): 1},
        1: {("A", "B"): 5, ("A",): 6, ("B",): 2, (): 7},
    }
    calls = []

    def evaluate(kb: dict[int, tuple[str, ...]]) -> tuple[float, bool]:
        calls.append(kb)
        return sum(cost[g][tuple(v)] for g, v in kb.items()), kb[0] != ()

    res = region_hindsight(evaluate, [0, 1], ("A", "B"))
    assert res["best"] == {0: ("A",), 1: ("B",)}
    assert res["best_ok"]
    assert res["best_val"] == 3 + 2
    assert calls[0] == {0: ("A", "B"), 1: ("A", "B")}
    assert res["n_eval"] == 1 + 3 + 3
    assert len(all_subsets(("A", "B", "C"))) == 8


def test_region_hindsight_keeps_incumbent_without_strict_improvement() -> None:
    res = region_hindsight(lambda kb: (1.0, True), [0, 1, 2], ("A", "B"))
    assert all(v == ("A", "B") for v in res["best"].values())


def test_region_subset_only_changes_kept_xapps_knobs_in_kept_regions() -> None:
    P = C.E6PConfig(ptx_on=True, prot_on=True, xapps=("SliceGuarantee",))
    cfg = C.E6Config(seed=3, mix="ES", load="high", warmup_s=10, scored_s=80, e6p=P)
    env = E6Env(cfg, log=False, wg3=True)
    site = [int(x) for x in env.plant.lay.cell_site]
    keep = {g: (("SliceGuarantee",) if g < 4 else ()) for g in sorted(set(site))}
    init = env.config()
    env.run(region_subset(keep, site))
    changed = [k for k, v in env.config().items() if v != init[k]]
    assert changed
    assert all(k[0] == "prot_min" and site[k[1]] < 4 for k in changed)
    assert env.stats["rej"] > 0

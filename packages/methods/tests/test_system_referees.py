import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oran_causal.core.gate import tau_table
from oran_causal.methods.system import maps as mp
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.units import UnitArbiter
from oran_causal.sim.system.decision.vocab import FAMILIES

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

DATA = Path(__file__).parent / "data"
ARTIFACTS = Path(__file__).parents[2] / "experiments" / "artifacts"


def ctx(f: str, step: float, own: float = 0.0, nbr: float = 0.0) -> dict[str, Any]:
    return {
        "knob": f,
        "step": float(step),
        "own_prot_below_frac": own,
        "nbr_max_prot_below_frac": nbr,
    }


class OnlyKeys(dict):
    """A unit that raises on any key outside `allowed`."""

    allowed: tuple[str, ...] = ("ctx",)

    def __getitem__(self, k: str) -> Any:
        if k not in self.allowed:
            raise AssertionError(f"gate read unit[{k!r}]")
        return dict.__getitem__(self, k)

    def get(self, k: object, default: Any = None, /) -> Any:
        raise AssertionError(f"gate read unit.get({k!r})")


class OnlyCtxC(OnlyKeys):
    allowed = ("ctx", "c")


def gt_maps() -> dict[str, dict]:
    doc = json.loads((DATA / "system_gt1_cells.json").read_text(encoding="utf-8"))
    return {
        "gt_ext": mp.gt_map(mp.gt_cells()),
        "gt1": mp.gt_map(mp.gt_cells(doc["rows"])),
    }


@pytest.fixture(scope="module")
def conf() -> dict[str, Any]:
    doc = json.loads((ARTIFACTS / "study2_maps.json").read_text(encoding="utf-8"))
    return rf.load_conf_maps(doc)


@pytest.fixture(scope="module")
def certsafe_doc() -> dict[str, Any]:
    return json.loads((ARTIFACTS / "study3_certsafe.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cs(certsafe_doc: dict[str, Any]) -> dict[str, Any]:
    return rf.load_certsafe(certsafe_doc)


def test_v1_reads_only_ctx() -> None:
    g = rf.MapGate({("sleep", "nbr", "pv"): 1.0, ("sleep", "nbr", "e"): 1.0})
    assert g(OnlyKeys(ctx=ctx("sleep", 1))) == ("reject", 1.0)
    assert g(OnlyKeys(ctx=ctx("sleep", -1))) == ("accept", 1.0)
    assert g.n == {"defer_sleep": 1, "accept_sleep": 1}


V1_CASES = [
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): 5.0}, 1, 0.0, 0.0, True),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): 5.0}, -1, 1.0, 1.0, False),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.04, 0.0, False),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.06, 0.0, True),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.0, 0.9, False),
    ({("x", "nbr", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.9, 0.0, False),
    ({("x", "nbr", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.0, 0.9, True),
    ({("x", "own", "pv"): 2.0, ("x", "nbr", "pv"): -3.0}, 1, 1.0, 1.0, False),
    ({("x", "own", "pv"): 1.0}, 1, 0.0, 0.0, True),
    ({("x", "far", "pv"): 9.0, ("x", "far", "e"): 9.0}, 1, 1.0, 1.0, False),
    ({("x", "own", "v"): 3.0}, 1, 0.0, 0.0, True),
    ({("x", "own", "v"): 3.0, ("x", "own", "pv"): None}, 1, 0.0, 0.0, False),
    ({("x", "own", "pv"): 1.0}, 0, 1.0, 1.0, False),
    ({("x", "own", "pv"): 1.0, ("x", "own", "e"): -1.0}, 1, math.nan, math.nan, False),
    ({}, 1, 1.0, 1.0, False),
]


@pytest.mark.parametrize(("M", "step", "own", "nbr", "defer"), V1_CASES)
def test_v1_decision_table_synthetic(
    M: dict, step: int, own: float, nbr: float, defer: bool
) -> None:
    assert rf.MapGate(M, 0.05).decide(ctx("x", step, own, nbr))[0] is defer


def test_v1_theta_edge_and_builders() -> None:
    M = {
        ("x", "own", "pv"): 1.0,
        ("x", "own", "e"): -1.0,
        ("x", "nbr", "pv"): 0.5,
        ("x", "far", "v"): 1.0,
    }
    assert not rf.MapGate(M, 0.05).decide(ctx("x", 1, own=0.05))[0]
    assert rf.MapGate(M, 0.0).decide(ctx("x", 1, own=1e-6))[0]
    assert rf.own_only(M) == {("x", "own", "pv"): 1.0, ("x", "own", "e"): -1.0}
    assert rf.sign_flip(M, ("x",))[("x", "nbr", "pv")] == -0.5
    cells = [
        {"family": "x", "relation": "own", "kpi": "pv", "status": "TRUE", "mean": 2.0},
        {"family": "x", "relation": "own", "kpi": "e", "status": "NULL", "mean": 0.1},
        {"family": "x", "relation": "nbr", "kpi": "pv", "status": "INDET", "mean": 9.0},
    ]
    assert rf.map_from_gt(cells) == {("x", "own", "pv"): 2.0, ("x", "own", "e"): None}
    assert rf.map_from_gt(cells, true_only=True) == {("x", "own", "pv"): 2.0}


@pytest.mark.parametrize("name", ["gt_ext", "gt1"])
def test_v1_gt_map_implications(name: str) -> None:
    M = gt_maps()[name]
    T = rf.decision_table(M, 0.05)
    assert T[("sleep", 1)] == "defer"
    assert T[("sleep", -1)] == "accept"
    assert T[("carrier", 1)] == "accept"
    assert T[("carrier", -1)].startswith("defer iff own")
    assert T[("ptx", -1)] == "accept"
    assert T[("ptx", 1)] == "defer"
    assert T[("prot_min", 1)] == "accept"
    assert T[("prot_min", -1)] == "defer"
    To = rf.decision_table(rf.own_only(M), 0.05)
    assert To[("sleep", 1)] == "accept"
    assert To[("sleep", -1)] == "defer"
    Tf = rf.decision_table(rf.sign_flip(M, ("carrier",)), 0.05)
    assert Tf[("carrier", 1)].startswith("defer iff own")
    assert Tf[("carrier", -1)] == "accept"
    assert {k: v for k, v in Tf.items() if k[0] != "carrier"} == {
        k: v for k, v in T.items() if k[0] != "carrier"
    }


def test_v1_gt1_and_gt_ext_agree() -> None:
    m = gt_maps()
    assert rf.decision_table(m["gt1"]) == rf.decision_table(m["gt_ext"])


STATIC = {
    "cells": 3,
    "neighbours": [[1], [0], [1]],
    "is_macro": np.array([True, True, False]),
    "knobs": [("carrier", 0), ("carrier", 1), ("sleep", 2), ("prot_min", 0), ("ptx", 0)],
    "xapps": {"ES": [], "SliceGuarantee": [], "PowerES": []},
}
CFG = {
    ("carrier", 0): 2.0,
    ("carrier", 1): 2.0,
    ("sleep", 2): 0.0,
    ("prot_min", 0): 0.1,
    ("ptx", 0): 40.0,
}


def _obs(t: float, reqs: list[dict]) -> dict[str, Any]:
    return {
        "t": float(t),
        "requests": reqs,
        "config": dict(CFG),
        "new_reports": [],
        "static": STATIC,
        "locked": {},
        "changes": 0,
        "churn_cap": None,
    }


def test_v1_unit_arbiter_from_t0_and_hold() -> None:
    M = {("sleep", "nbr", "pv"): 9.0, ("sleep", "nbr", "e"): 4.0, ("sleep", "own", "pv"): -0.7}
    arb = rf.mapgate_arbiter(M, record=True)
    assert isinstance(arb, UnitArbiter)
    assert arb.warmup_s == 0.0
    assert arb.T == 60.0
    assert arb.open_rule == "feasible"
    sl = {"xapp": "ES", "ver": 1, "knob": ("sleep", 2), "cur": 0.0, "prop": 1.0, "t": 0.0}
    assert arb(_obs(0, [sl]))["decisions"] == ["reject"]
    assert arb.units[0]["mode"] == "reject"
    assert arb(_obs(30, [sl]))["decisions"] == ["reject"]
    assert arb(_obs(61, [dict(sl, cur=1.0, prop=0.0)]))["decisions"] == ["accept"]


@pytest.mark.claim("DEF-REF-RULES")
def test_v2_reads_only_ctx_and_cell() -> None:
    g = rf.MapGateV2({("sleep", "nbr", "pv"): 1.0, ("sleep", "nbr", "e"): 1.0})
    assert g(OnlyCtxC(ctx=ctx("sleep", 1), c=2)) == ("reject", 1.0)
    assert g(OnlyCtxC(ctx=ctx("sleep", -1), c=2)) == ("accept", 1.0)
    assert g.n == {"defer_sleep": 1, "accept_sleep": 1}


V2_CASES = [
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): 5.0}, 1, 0.0, 0.0, True),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): 5.0}, -1, 1.0, 1.0, False),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.05, 0.0, False),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.06, 0.0, True),
    ({("x", "own", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.0, 0.9, False),
    ({("x", "nbr", "pv"): 2.0, ("x", "own", "e"): -5.0}, 1, 0.0, 0.9, True),
    ({("x", "own", "pv"): 1.0}, 1, 0.0, 0.0, False),
    ({("x", "own", "pv"): 1.0}, 1, 0.5, 0.0, True),
    ({("x", "own", "pv"): 1.0, ("x", "own", "e"): None}, 1, 0.0, 0.0, False),
    (
        {("x", "own", "pv"): 1.0, ("x", "far", "pv"): -2.0, ("x", "own", "e"): 1.0},
        1,
        1.0,
        1.0,
        False,
    ),
    ({("x", "own", "pv"): 1.0, ("x", "far", "e"): 3.0}, 1, 0.0, 0.0, True),
    ({("x", "far", "pv"): 1.0, ("x", "own", "e"): -1.0}, 1, 1.0, 1.0, False),
    ({("x", "far", "pv"): 1.0, ("x", "own", "e"): 1.0}, 1, 0.0, 0.0, True),
    (
        {("x", "own", "pv"): -1.0, ("x", "own", "v"): 9.0, ("x", "own", "e"): 1.0},
        1,
        1.0,
        1.0,
        False,
    ),
    ({("x", "own", "v"): 3.0, ("x", "own", "e"): 1.0}, 1, 0.0, 0.0, True),
    (
        {("x", "own", "pv"): None, ("x", "own", "v"): 3.0, ("x", "own", "e"): 1.0},
        1,
        0.0,
        0.0,
        True,
    ),
    ({("x", "own", "rlf"): 0.1, ("x", "own", "e"): 1.0}, 1, 0.0, 0.0, True),
    ({("x", "own", "rlf"): -0.1, ("x", "own", "e"): 1.0}, 1, 0.0, 0.0, False),
    ({("x", "own", "pv"): 1.0, ("x", "own", "e"): 1.0}, 0, 1.0, 1.0, False),
    ({("x", "own", "pv"): 1.0, ("x", "own", "e"): -1.0}, 1, math.nan, math.nan, False),
    ({}, 1, 1.0, 1.0, False),
]


@pytest.mark.claim("DEF-REF-RULES")
@pytest.mark.parametrize(("M", "step", "own", "nbr", "defer"), V2_CASES)
def test_v2_decision_synthetic(M: dict, step: int, own: float, nbr: float, defer: bool) -> None:
    assert rf.MapGateV2(M, 0.05).decide(ctx("x", step, own, nbr))[0] is defer


def _u(c: int, s: int = 1) -> dict[str, Any]:
    return {"ctx": ctx("x", s), "c": c}


@pytest.mark.claim("DEF-REF-RULES")
def test_v2_guard_conflict_and_duty_bound() -> None:
    Mc = {
        ("x", "own", "pv"): 1.0,
        ("x", "own", "v"): -5.0,
        ("x", "own", "rlf"): 1.0,
        ("x", "own", "e"): 1.0,
    }
    assert rf.MapGateV2(Mc).classify("x", 1)["conflict"] is True
    g = rf.MapGateV2(Mc)
    assert [g(_u(0))[0] for _ in range(5)] == ["reject", "accept", "reject", "accept", "reject"]
    assert g(_u(1))[0] == "reject"
    assert g(_u(1))[0] == "accept"
    assert g.n == {"defer_x": 4, "release_x": 3}
    g2 = rf.MapGateV2(Mc, k_conf=2)
    assert [g2(_u(0))[0] for _ in range(4)] == ["reject", "reject", "accept", "reject"]
    g3 = rf.MapGateV2(Mc)
    assert [g3(_u(0, s))[0] for s in (1, -1, 1)] == ["reject", "accept", "reject"]
    Mu = {**Mc, ("x", "own", "v"): 5.0, ("x", "own", "rlf"): -1.0}
    gu = rf.MapGateV2(Mu)
    assert not gu.classify("x", 1)["conflict"]
    assert [gu(_u(0))[0] for _ in range(4)] == ["reject"] * 4


def test_v2_directional_units() -> None:
    M = {("ptx", "own", "pv"): 1.0, ("ptx", "own", "e"): 1.0}
    arb = rf.mapgate_v2_arbiter(M, record=True)
    assert isinstance(arb, rf.DirectionalUnitArbiter)
    assert arb.warmup_s == 0.0
    assert arb.T == 60.0
    up = {"xapp": "PowerES", "ver": 1, "knob": ("ptx", 0), "cur": 40.0, "prop": 41.0, "t": 0.0}
    assert arb(_obs(0, [up]))["decisions"] == ["reject"]
    assert arb(_obs(20, [up]))["decisions"] == ["reject"]
    down = dict(up, prop=39.0)
    assert arb(_obs(30, [down]))["decisions"] == ["accept"]
    assert arb.passed == 1
    assert len(arb.units) == 1
    a1 = rf.mapgate_arbiter(M)
    a1(_obs(0, [up]))
    assert a1(_obs(30, [down]))["decisions"] == ["reject"]


@pytest.mark.claim("DEF-REF-RULES")
@pytest.mark.parametrize(("name", "ptx_up"), [("gt_ext", "defer [bounded]"), ("gt1", "defer")])
def test_v2_gt_map_implications(name: str, ptx_up: str) -> None:
    M = gt_maps()[name]
    T = rf.decision_table_v2(M, 0.05)
    assert T[("sleep", 1)] == "defer"
    assert T[("sleep", -1)] == "accept"
    assert T[("carrier", 1)] == "accept"
    assert T[("carrier", -1)] == "defer iff own pressure > theta"
    assert T[("ptx", 1)] == ptx_up
    assert T[("ptx", -1)] == "accept"
    assert T[("prot_min", 1)] == "accept"
    assert T[("prot_min", -1)] == "defer iff own pressure > theta"
    To = rf.decision_table_v2(rf.own_only(M), 0.05)
    assert To[("sleep", 1)] == "accept"
    assert To[("ptx", 1)] == "defer"
    Tf = rf.decision_table_v2(rf.sign_flip(M, ("carrier",)), 0.05)
    assert Tf[("carrier", 1)] == "defer iff own pressure > theta"
    assert Tf[("carrier", -1)] == "accept"


def test_v2_random_and_blanket_maps() -> None:
    keys = [(f, r, k) for f in FAMILIES for r in ("own", "nbr", "far") for k in ("pv", "v", "e")]
    cells = [
        {"family": f, "relation": r, "kpi": k, "mean": 1.0 + i} for i, (f, r, k) in enumerate(keys)
    ]
    ref = {(f, "own", "pv"): 1.0 for f in FAMILIES}
    R1, R2 = rf.random_sized_map(ref, cells), rf.random_sized_map(ref, cells)
    assert R1 == R2
    assert len(R1) == len(ref)
    assert rf.random_sized_map(ref, cells, tag=1) != R1
    assert all(abs(v) == 1.0 + keys.index(k) for k, v in R1.items())
    T = rf.decision_table_v2(rf.blanket_saving_map())
    for f, s in rf.SAVING_DIR.items():
        assert T[(f, s)] == "defer iff own/nbr pressure > theta"
        assert T[(f, -s)] == "accept"
    assert T[("prot_min", 1)] == T[("prot_min", -1)] == "accept"


def _units(n: int = 4000, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = int(rng.integers(3))
        f = str(rng.choice(FAMILIES))
        s = float(rng.choice([-1, 1, 0]))
        own, nbr = float(rng.uniform(0, 0.2)), float(rng.uniform(0, 0.2))
        out.append({"c": c, "ctx": ctx(f, s, own, nbr)})
    return out


def _run(g: Any, units: list[dict[str, Any]]) -> list[str]:
    return [g(u)[0] for u in units]


def test_frozen_artifacts_reproduce_their_alias_tables(
    conf: dict[str, Any], cs: dict[str, Any], certsafe_doc: dict[str, Any]
) -> None:
    assert conf["alias"]["jobs"][0] == "MG:PMRT"
    assert set(cs["tables"]) == set(rf.CS_ARMS)
    assert cs["alias"]["jobs"] == certsafe_doc["jobs"]
    assert cs["maps"]["MG:PMRT"] == conf["maps"]["MG:PMRT"]
    bad = json.loads(json.dumps(certsafe_doc))
    bad["arms"]["CS:PMRT"]["uncertified"] = [["ptx", 1]]
    with pytest.raises(ValueError, match="signature of CS:PMRT"):
        rf.load_certsafe(bad)


def test_default_certsafe_gate_equals_mapgate_v2(conf: dict[str, Any]) -> None:
    units = _units()
    for arm, M in conf["maps"].items():
        a, b = rf.MapGateV2(M), rf.CertSafeMapGateV2(M, ())
        assert _run(a, units) == _run(b, units), arm
        assert {k: v for k, v in b.n.items() if not k.startswith(("dir_", "uncert_"))} == a.n
        assert rf.certsafe_signature(M, ()) == rf.decision_signature(M)


def test_gate_withholds_only_uncertified_deferrals(conf: dict[str, Any]) -> None:
    M = conf["maps"]["MG:PMRT"]
    units = _units(seed=1)
    base = _run(rf.MapGateV2(M), units)
    g = rf.CertSafeMapGateV2(M, [("ptx", 1)])
    got = _run(g, units)
    ptx_up = [u["ctx"]["knob"] == "ptx" and u["ctx"]["step"] > 0 for u in units]
    for u, up, x, y in zip(units, ptx_up, base, got, strict=True):
        if up:
            assert y == "accept"
        elif u["ctx"]["knob"] != "ptx":
            assert x == y
    withheld = sum(x == "reject" for up, x in zip(ptx_up, base, strict=True) if up)
    assert g.n["uncert_ptx"] == withheld > 0
    dirs = [f"dir_{f}_{d:+d}" for f in FAMILIES for d in (1, -1, 0)]
    assert sum(g.n.get(k, 0) for k in dirs) == len(units)
    M2 = {k: 2.5 * v for k, v in M.items()}
    assert rf.certsafe_signature(M, [("ptx", 1)]) == rf.certsafe_signature(M2, [("ptx", 1)])
    assert _run(rf.CertSafeMapGateV2(M2, [("ptx", 1)]), units) == got
    assert rf.certsafe_signature(M, [("ptx", 1)]) != rf.certsafe_signature(M, [])
    allu = [(f, d) for f in FAMILIES for d in (1, -1)]
    assert set(_run(rf.CertSafeMapGateV2(M, allu), units)) == {"accept"}
    assert rf.certsafe_signature(M, allu) == rf.all_accept_signature()
    arb = rf.certsafe_arbiter(M, [("ptx", 1)])
    assert isinstance(arb, rf.DirectionalUnitArbiter)
    assert isinstance(arb.policy, rf.CertSafeMapGateV2)
    assert arb.policy.uncertified == {("ptx", 1)}


def test_certify_on_the_frozen_pmrt_map(conf: dict[str, Any]) -> None:
    tau = tau_table(
        {"rlf": 43.15, "v": 20318.0},
        {(f, d): 10.0 for f in FAMILIES for d in (1, -1)},
        rf.GATE_SPACE,
    )
    tight = {
        (f, r, k): {"kind": "slope", "beta": 0.0, "se": 0.0}
        for f in FAMILIES
        for r in rf.RELS_V2
        for k in rf.GUARD_KPIS
    }
    pm = conf["maps"]["MG:PMRT"]
    c0 = rf.certify_map(pm, tight, tau)
    assert c0["uncertified"] == []
    deferrable = {k for k, v in c0["classes"].items() if v["can_defer"]}
    assert deferrable == {"carrier|-1", "sleep|1", "ptx|1", "prot_min|-1"}
    wide = {**tight, ("ptx", "nbr", "rlf"): {"kind": "slope", "beta": -0.10, "se": 0.05}}
    assert rf.certify_map(pm, wide, tau)["uncertified"] == [["ptx", 1]]
    assert rf.certify_map(pm, None, tau)["uncertified"] == [
        ["carrier", -1],
        ["sleep", 1],
        ["ptx", 1],
        ["prot_min", -1],
    ]
    bl = conf["maps"]["blanket2"]
    unc = [tuple(x) for x in rf.certify_map(bl, None, tau)["uncertified"]]
    assert rf.certsafe_signature(bl, unc) == rf.all_accept_signature()


def test_gated_pmrt_policy_is_never_sleep(cs: dict[str, Any]) -> None:
    t = cs["tables"]["CS:PMRT"]
    g = rf.CertSafeMapGateV2(t["map"], t["uncertified"])
    units = _units(seed=2)
    want = [
        "reject" if u["ctx"]["knob"] == "sleep" and u["ctx"]["step"] > 0 else "accept"
        for u in units
    ]
    assert _run(g, units) == want


def test_arm_factories_cover_every_arm(conf: dict[str, Any], cs: dict[str, Any]) -> None:
    for arm in (*rf.ANCHORS, "incumbent", "never_sleep", "B2", "MG:allaccept", *conf["maps"]):
        arb = rf.conf_arbiter(arm, 187000, 120.0, conf["maps"])
        assert (arb is None) == (arm == "noarb")
        assert isinstance(rf.policy_counts(arb), dict)
    for arm in (*rf.ANCHORS, "incumbent", "never_sleep", rf.CAL_ARM, "MG:PMRT", *cs["tables"]):
        assert (rf.cs_arbiter(arm, 191100, 120.0, cs) is None) == (arm == "noarb")
    with pytest.raises(KeyError):
        rf.conf_arbiter("MG:nope", 0, 120.0, conf["maps"])
    with pytest.raises(KeyError):
        rf.cs_arbiter("CS:nope", 0, 120.0, cs)
    with pytest.raises(KeyError):
        rf.anchor_arbiter("sub", 120.0)

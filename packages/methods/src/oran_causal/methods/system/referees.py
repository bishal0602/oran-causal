"""Map-driven WG3 referees of the system decision studies (Studies 2 and 3).

MapGate(M, theta) is a `UnitArbiter` policy: at unit open it reads only the unit context and
returns ("reject", 1.0) = defer or ("accept", 1.0); the mode holds for the unit's T. Map M:
{(family, rel, kpi): beta | None}, beta = the effect of accepting a +1 knob-direction step; None
is declared with no effect, a missing key is undeclared.

v1: harm KPI k_h = pv if any (f, rel, pv) with rel in (own, nbr) is declared, else v; defer iff
d * sum_r beta(f, r, k_h) > 0 and (d * sum_r beta(f, r, e) >= 0 or the max protected-UE
pressure over the harmful relations > theta).

v2 (`MapGateV2`): D_k = d * sum over (own, nbr, far) of beta(f, rel, k); k* = the first of
(pv, v, rlf) with D_k != 0; harm iff D_k* > 0; energy class from sign(D_e); defer iff harm and
(costly or pressure > theta). A guard conflict (the first of (v, rlf) with D != 0 is < 0) bounds a
knob's run of deferred units at k_conf. `DirectionalUnitArbiter` defers only the opening
direction of a reject unit.

Certified-safe v2 (`CertSafeMapGateV2`) never defers an uncertified (family, d) class; the
certification is `oran_causal.core.gate.certify` with `can_defer` from the decision signature.
"""

import functools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.core.data import UnitTable
from oran_causal.core.design import v_design
from oran_causal.core.gate import GateSpace, certify, design_slope_se, tau_table
from oran_causal.core.pmrt.units import unit_order
from oran_causal.methods.system.mscr_crt import LEVEL_V2_ARR
from oran_causal.sim.system.baselines import (
    CellPriorityLock,
    KnobLock,
    freeze,
    priority,
    subset,
)
from oran_causal.sim.system.decision.collect import IncumbentPolicy
from oran_causal.sim.system.decision.units import T_UNIT, UnitArbiter
from oran_causal.sim.system.decision.vocab import FAMILIES
from oran_causal.sim.system.env import Decision, Obs

__all__ = [
    "ANCHORS",
    "ASSOC",
    "ASSOC_ARMS",
    "BOUND_METHOD",
    "B_GATES",
    "CAL_ARM",
    "CS_ARMS",
    "CS_OF",
    "GATE_SPACE",
    "GUARDS_V2",
    "GUARD_FIELD",
    "GUARD_KPIS",
    "K_CONF",
    "LOCK_S",
    "MAP_ARMS",
    "OPEN_RULE",
    "PRESSURE",
    "PRIORITY_V2",
    "RELS",
    "RELS_V2",
    "SAVING_DIR",
    "THETA",
    "WARM",
    "BlanketGate",
    "CertSafeMapGateV2",
    "Counting",
    "DirectionalUnitArbiter",
    "MapGate",
    "MapGateV2",
    "alias_table",
    "all_accept_signature",
    "anchor_arbiter",
    "blanket_saving_map",
    "bounds_from_json",
    "bounds_to_json",
    "can_defer",
    "certify_map",
    "certsafe_arbiter",
    "certsafe_signature",
    "conf_arbiter",
    "cs_alias_table",
    "cs_arbiter",
    "cs_tables",
    "decision_signature",
    "decision_table",
    "decision_table_v2",
    "design_slope_bound",
    "direction",
    "gt_ci_bounds",
    "load_certsafe",
    "load_conf_maps",
    "map_from_gt",
    "map_from_json",
    "map_to_json",
    "mapgate_arbiter",
    "mapgate_v2_arbiter",
    "naive_slope_se",
    "never_sleep",
    "own_only",
    "policy_counts",
    "random_sized_map",
    "sign_flip",
    "signature_key",
    "tau_from_accept_all",
    "tau_from_json",
    "tau_to_json",
]

Map = Mapping[tuple[str, str, str], float | None]

RELS = ("own", "nbr")
PRESSURE = {"own": "own_prot_below_frac", "nbr": "nbr_max_prot_below_frac"}
THETA = 0.05
RELS_V2 = ("own", "nbr", "far")
PRIORITY_V2 = ("pv", "v", "rlf")
GUARDS_V2 = ("v", "rlf")
K_CONF = 1
OPEN_RULE = "feasible"
SAVING_DIR = {"carrier": -1, "sleep": 1, "ptx": -1}
GUARD_KPIS = GUARDS_V2
GUARD_FIELD = {"v": "viol_ue_s", "rlf": "rlf"}
GATE_SPACE = GateSpace(FAMILIES, RELS_V2, GUARD_KPIS)
WARM = 120.0
LOCK_S = 60.0
B_GATES = {"B1": frozenset({"ptx_up"}), "B2": frozenset({"ptx_up", "sleep", "car_off"})}

ANCHORS = ("freeze", "sub:ES+PowerES", "noarb", "sub:ES", "sub:PowerES", "sub:SliceGuarantee")
ASSOC = (
    "corr@dev",
    "corr@plc",
    "granger@dev",
    "granger@plc",
    "shap_gbdt@dev",
    "shap_gbdt@plc",
    "int@dev",
    "int@plc",
    "qacm@dev",
    "qacm@plc",
    "two_tower@dev",
    "two_tower@plc",
    "granger_by",
)
ASSOC_ARMS = tuple(f"MG:{b}" for b in ASSOC)
MAP_ARMS = ("MG:PMRT", "MG:GT", "MG:rand", "blanket2", *ASSOC_ARMS)
CAL_ARM = "CAL:allaccept"
CS_OF = {
    "CS:PMRT": "MG:PMRT",
    "CS:GT": "MG:GT",
    "CS:rand": "MG:rand",
    "CS:blanket2": "blanket2",
    **{f"CS:{b}": f"MG:{b}" for b in ASSOC},
}
CS_ARMS = tuple(CS_OF)
BOUND_METHOD: dict[str, str | None] = {
    "CS:PMRT": "pmrt",
    "CS:GT": "gt",
    "CS:rand": None,
    "CS:blanket2": None,
    **{f"CS:{b}": "ols" for b in ASSOC},
}


def _nz(x: Any) -> float:
    x = float(x)
    return 0.0 if math.isnan(x) else x


def _sgn(x: float) -> int:
    return (x > 0) - (x < 0)


class MapGate:
    """MapGate v1 policy `policy(unit) -> (mode, propensity)`; `n` counts defer_/accept_<family>."""

    def __init__(self, M: Map, theta: float = THETA) -> None:
        self.M, self.theta = dict(M), float(theta)
        self.n: dict[str, int] = {}

    def _beta(self, f: str, rel: str, kpi: str) -> float:
        b = self.M.get((f, rel, kpi))
        return 0.0 if b is None else float(b)

    def harm_kpi(self, f: str) -> str:
        """The pv KPI if the map declares any own or nbr pv edge of f, else v."""
        return "pv" if any((f, r, "pv") in self.M for r in RELS) else "v"

    def decide(self, ctx: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        """(defer, why) from a unit context (knob, step and the two pressure fields)."""
        f = ctx["knob"]
        step = float(ctx["step"])
        d = (step > 0) - (step < 0)
        kh = self.harm_kpi(f)
        contrib = {r: d * self._beta(f, r, kh) for r in RELS}
        harm = sum(contrib.values()) > 0
        costly = d * sum(self._beta(f, r, "e") for r in RELS) >= 0
        harmful = [r for r in RELS if contrib[r] > 0]
        pressure = max((_nz(ctx.get(PRESSURE[r], float("nan"))) for r in harmful), default=0.0)
        defer = bool(d != 0 and harm and (costly or pressure > self.theta))
        why = {
            "d": d,
            "kpi": kh,
            "harm": harm,
            "costly": costly,
            "harmful": harmful,
            "pressure": pressure,
        }
        return defer, why

    def __call__(self, unit: Mapping[str, Any]) -> tuple[str, float]:
        """Mode and propensity of a newly opened unit."""
        ctx = unit["ctx"]
        defer, _ = self.decide(ctx)
        key = ("defer_" if defer else "accept_") + str(ctx["knob"])
        self.n[key] = self.n.get(key, 0) + 1
        return ("reject" if defer else "accept"), 1.0


def mapgate_arbiter(
    M: Map,
    theta: float = THETA,
    *,
    T: float = T_UNIT,
    open_rule: str = OPEN_RULE,
    record: bool = False,
) -> UnitArbiter:
    """UnitArbiter(MapGate(M, theta)) active from t = 0, T = 60 s, open rule "feasible"."""
    return UnitArbiter(MapGate(M, theta), T=T, warmup_s=0.0, record=record, open_rule=open_rule)


def map_from_gt(cells: Iterable[Mapping[str, Any]], true_only: bool = False) -> dict:
    """Map from GT cells: TRUE -> its mean, NULL -> None (unless `true_only`), INDET -> absent."""
    M: dict[tuple[str, str, str], float | None] = {}
    for c in cells:
        key = (c["family"], c["relation"], c["kpi"])
        if c["status"] == "TRUE":
            M[key] = float(c["mean"])
        elif c["status"] == "NULL" and not true_only:
            M[key] = None
    return M


def own_only(M: Map) -> dict:
    """Drops every nbr and far edge."""
    return {k: v for k, v in M.items() if k[1] == "own"}


def sign_flip(M: Map, families: Iterable[str] = ("carrier",)) -> dict:
    """Negates every edge of `families` (None stays None)."""
    fam = set(families)
    return {k: (-v if (k[0] in fam and v is not None) else v) for k, v in M.items()}


def decision_table(M: Map, theta: float = THETA) -> dict[tuple[str, int], str]:
    """v1 rule per (family, d): "defer", "accept" or "defer iff <rels> pressure > theta"."""
    g = MapGate(M, theta)
    out = {}
    for f in FAMILIES:
        for d in (1, -1):
            lo, _ = g.decide(
                {"knob": f, "step": float(d), PRESSURE["own"]: 0.0, PRESSURE["nbr"]: 0.0}
            )
            hi, why = g.decide(
                {"knob": f, "step": float(d), PRESSURE["own"]: 1.0, PRESSURE["nbr"]: 1.0}
            )
            if lo:
                out[(f, d)] = "defer"
            elif hi:
                out[(f, d)] = "defer iff " + "/".join(why["harmful"]) + " pressure > theta"
            else:
                out[(f, d)] = "accept"
    return out


class MapGateV2:
    """MapGate v2 policy; `n` counts defer_/accept_/release_<family> (release = duty bound)."""

    def __init__(self, M: Map, theta: float = THETA, k_conf: int = K_CONF) -> None:
        self.M, self.theta, self.k_conf = dict(M), float(theta), int(k_conf)
        self.n: dict[str, int] = {}
        self.run: dict[tuple[int, str], int] = {}

    def _beta(self, f: str, rel: str, kpi: str) -> float:
        b = self.M.get((f, rel, kpi))
        return 0.0 if b is None else float(b)

    def classify(self, f: str, d: int) -> dict[str, Any]:
        """The map-only part of the rule for family f and direction d."""
        D = {k: d * sum(self._beta(f, r, k) for r in RELS_V2) for k in (*PRIORITY_V2, "e")}
        kstar = next((k for k in PRIORITY_V2 if _sgn(D[k]) != 0), None)
        harm = bool(d != 0 and kstar is not None and D[kstar] > 0)
        energy = {-1: "saving", 0: "neutral", 1: "costly"}[_sgn(D["e"])]
        harmful = [
            r for r in RELS if harm and kstar is not None and d * self._beta(f, r, kstar) > 0
        ]
        g = next((k for k in GUARDS_V2 if _sgn(D[k]) != 0), None)
        conflict = bool(harm and g is not None and D[g] < 0)
        return {
            "d": d,
            "D": D,
            "kpi": kstar,
            "harm": harm,
            "energy": energy,
            "harmful": harmful,
            "guard": g,
            "conflict": conflict,
        }

    def decide(self, ctx: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        """(defer before the duty bound, why) from a unit context."""
        why = self.classify(ctx["knob"], _sgn(float(ctx["step"])))
        pressure = max(
            (_nz(ctx.get(PRESSURE[r], float("nan"))) for r in why["harmful"]), default=0.0
        )
        why["pressure"] = pressure
        return bool(why["harm"] and (why["energy"] == "costly" or pressure > self.theta)), why

    def __call__(self, unit: Mapping[str, Any]) -> tuple[str, float]:
        """Mode and propensity of a newly opened unit, with the duty bound."""
        ctx = unit["ctx"]
        f = str(ctx["knob"])
        key = (int(unit["c"]), f)
        defer, why = self.decide(ctx)
        tag = "defer_" if defer else "accept_"
        if defer and why["conflict"] and self.run.get(key, 0) >= self.k_conf:
            defer, tag = False, "release_"
        self.run[key] = self.run.get(key, 0) + 1 if defer else 0
        self.n[tag + f] = self.n.get(tag + f, 0) + 1
        return ("reject" if defer else "accept"), 1.0


class DirectionalUnitArbiter(UnitArbiter):
    """UnitArbiter whose reject units defer only their opening direction.

    `passed` counts the opposite-direction requests let through.
    """

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.passed = 0

    def __call__(self, obs: Mapping[str, Any]) -> Decision:
        """The UnitArbiter decisions with opposite-direction rejects turned into accepts."""
        out = super().__call__(obs)
        dec = list(out["decisions"])
        for i, (r, d) in enumerate(zip(obs["requests"], dec, strict=True)):
            if d != "reject":
                continue
            u = self.active.get((int(r["knob"][1]), r["xapp"]))
            if u is not None and _sgn(float(r["prop"] - r["cur"])) != _sgn(float(u["ctx"]["step"])):
                dec[i] = "accept"
                self.passed += 1
        out["decisions"] = dec
        return out


def mapgate_v2_arbiter(
    M: Map,
    theta: float = THETA,
    k_conf: int = K_CONF,
    *,
    T: float = T_UNIT,
    open_rule: str = OPEN_RULE,
    record: bool = False,
) -> DirectionalUnitArbiter:
    """DirectionalUnitArbiter(MapGateV2(M, theta, k_conf)) active from t = 0."""
    return DirectionalUnitArbiter(
        MapGateV2(M, theta, k_conf), T=T, warmup_s=0.0, record=record, open_rule=open_rule
    )


def _v2_label(c: Mapping[str, Any]) -> str:
    if not c["harm"]:
        s = "accept"
    elif c["energy"] == "costly":
        s = "defer"
    elif c["harmful"]:
        s = "defer iff " + "/".join(c["harmful"]) + " pressure > theta"
    else:
        s = "accept (far-only harm)"
    if s.startswith("defer") and c["conflict"]:
        s += " [bounded]"
    return s


def decision_table_v2(M: Map, theta: float = THETA) -> dict[tuple[str, int], str]:
    """v2 rule per (family, d); a guard-conflicted deferral gets " [bounded]"."""
    g = MapGateV2(M, theta)
    return {(f, d): _v2_label(g.classify(f, d)) for f in FAMILIES for d in (1, -1)}


def random_sized_map(
    M_ref: Map, cells: Sequence[Mapping[str, Any]], tag: int = 6623, key: int = 2
) -> dict:
    """Random map with |M_ref| edges drawn from `cells`, random signs, |beta| = |cell mean|.

    RNG `default_rng([tag, key])`; key 2 = K-A2, the option (a) study uses key 3.
    """
    rng = np.random.default_rng([int(tag), int(key)])
    keys = [(c["family"], c["relation"], c["kpi"]) for c in cells]
    mag = {(c["family"], c["relation"], c["kpi"]): abs(float(c["mean"])) for c in cells}
    idx = sorted(int(j) for j in rng.choice(len(keys), size=len(M_ref), replace=False))
    sg = rng.choice([-1.0, 1.0], size=len(M_ref))
    return {keys[i]: float(s * mag[keys[i]]) for i, s in zip(idx, sg, strict=True)}


def blanket_saving_map() -> dict:
    """No-map control: every energy-saving request hurts own and nbr pv and saves own e."""
    M = {}
    for f, s in SAVING_DIR.items():
        M[(f, "own", "pv")] = float(s)
        M[(f, "nbr", "pv")] = float(s)
        M[(f, "own", "e")] = float(-s)
    return M


def _signature_entry(c: Mapping[str, Any], theta: float) -> tuple:
    if not c["harm"]:
        return ("accept",)
    if c["energy"] == "costly" or float(theta) < 0.0:
        return ("defer", bool(c["conflict"]))
    if c["harmful"]:
        return ("defer_iff", tuple(c["harmful"]), bool(c["conflict"]))
    return ("accept",)


def decision_signature(M: Map, theta: float = THETA) -> tuple:
    """The map-dependent inputs of `MapGateV2.decide` and its duty bound per (family, d).

    Entries: ("accept",); ("defer", conflict); ("defer_iff", harmful, conflict). Equal
    signatures give identical policies; theta is part of the signature.
    """
    g = MapGateV2(M, theta)
    out = [(f, d, _signature_entry(g.classify(f, d), theta)) for f in FAMILIES for d in (1, -1)]
    return (("theta", float(theta)), *out)


def signature_key(sig: tuple) -> str:
    """Stable JSON string of a decision signature."""
    return json.dumps(sig, separators=(",", ":"))


def all_accept_signature(theta: float = THETA) -> tuple:
    """The signature of a map that never defers."""
    return decision_signature({}, theta)


def map_to_json(M: Map) -> list[list[Any]]:
    """[[family, relation, kpi, beta | None], ...] sorted by key."""
    return [[k[0], k[1], k[2], None if v is None else float(v)] for k, v in sorted(M.items())]


def map_from_json(rows: Iterable[Sequence[Any]]) -> dict:
    """Inverse of `map_to_json`."""
    return {(str(f), str(r), str(k)): (None if b is None else float(b)) for f, r, k, b in rows}


def can_defer(M: Map, f: str, d: int, theta: float = THETA) -> bool:
    """True iff MapGateV2(M) defers (f, d) in some context.

    Raises:
        KeyError: (f, d) is not a request class.
    """
    for f_, d_, s in decision_signature(M, theta)[1:]:
        if f_ == f and d_ == d:
            return s[0] != "accept"
    raise KeyError((f, d))


def certify_map(
    M: Map, bounds: Mapping[Any, Any] | None, tau: Mapping, theta: float = THETA
) -> dict[str, Any]:
    """`core.gate.certify` of map M with MapGateV2's deferrable classes."""
    return certify(M, bounds, tau, functools.partial(can_defer, M, theta=theta), GATE_SPACE)


class CertSafeMapGateV2(MapGateV2):
    """MapGateV2 whose deferrals of the uncertified (family, d) classes are accepts instead.

    Extra counters: uncert_<f> (deferrals withheld), dir_<f>_<+1|-1> (units per direction).
    """

    def __init__(
        self,
        M: Map,
        uncertified: Iterable[Sequence[Any]] = (),
        theta: float = THETA,
        k_conf: int = K_CONF,
    ) -> None:
        super().__init__(M, theta, k_conf)
        self.uncertified = frozenset((str(f), int(d)) for f, d in uncertified)
        self._withheld = False

    def decide(self, ctx: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        """MapGateV2's decision with uncertified deferrals withheld."""
        defer, why = super().decide(ctx)
        why["uncertified"] = (str(ctx["knob"]), int(why["d"])) in self.uncertified
        self._withheld = bool(defer and why["uncertified"])
        return (False if self._withheld else defer), why

    def __call__(self, unit: Mapping[str, Any]) -> tuple[str, float]:
        """Mode and propensity of a newly opened unit."""
        out = super().__call__(unit)
        ctx = unit["ctx"]
        f = str(ctx["knob"])
        dk = f"dir_{f}_{_sgn(float(ctx['step'])):+d}"
        self.n[dk] = self.n.get(dk, 0) + 1
        if self._withheld:
            self.n["uncert_" + f] = self.n.get("uncert_" + f, 0) + 1
        return out


def certsafe_arbiter(
    M: Map,
    uncertified: Iterable[Sequence[Any]] = (),
    theta: float = THETA,
    k_conf: int = K_CONF,
    *,
    T: float = T_UNIT,
    open_rule: str = OPEN_RULE,
    record: bool = False,
) -> DirectionalUnitArbiter:
    """DirectionalUnitArbiter(CertSafeMapGateV2(M, uncertified)) with the MapGateV2 settings."""
    return DirectionalUnitArbiter(
        CertSafeMapGateV2(M, uncertified, theta, k_conf),
        T=T,
        warmup_s=0.0,
        record=record,
        open_rule=open_rule,
    )


def certsafe_signature(
    M: Map, uncertified: Iterable[Sequence[Any]] = (), theta: float = THETA
) -> tuple:
    """`decision_signature` with every uncertified (f, d) entry replaced by ("accept",)."""
    unc = {(str(f), int(d)) for f, d in uncertified}
    sig = decision_signature(M, theta)
    return (sig[0], *((f, d, ("accept",) if (f, d) in unc else s) for f, d, s in sig[1:]))


def design_slope_bound(ud: UnitTable, f: str, rel: str, kpi: str) -> dict[str, Any]:
    """PMRT design-based slope of a guard edge with its HC0 standard error (information order)."""
    rows = unit_order(ud, ud.rows_of(f))
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows], LEVEL_V2_ARR)
    pre = ud.pre[(rel, kpi)][rows]
    return design_slope_se(v, ud.y[(rel, kpi)][rows], pre, ud.episode[rows], ud.sgn[rows])


def naive_slope_se(ud: UnitTable, f: str, rel: str, kpi: str) -> dict[str, Any]:
    """OLS slope of y on x = level * sgn over the family's units with its iid standard error."""
    rows = ud.rows_of(f)
    none = {"kind": "slope", "beta": 0.0, "se": None, "n": len(rows), "z": None}
    if len(rows) < 3:
        return none
    x, y = ud.x[rows], ud.y[(rel, kpi)][rows]
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return none
    beta = float(xc @ (y - y.mean()) / den)
    e = (y - y.mean()) - beta * xc
    se = float(np.sqrt(float(e @ e) / (len(rows) - 2) / den))
    return {
        "kind": "slope",
        "beta": beta,
        "se": se,
        "n": len(rows),
        "z": beta / se if se > 0 else None,
    }


def gt_ci_bounds(cells: Iterable[Mapping[str, Any]]) -> dict:
    """{(f, rel, k): {"kind": "ci", "lo", "hi"}} for the guard KPIs of GT cells (95 % CI)."""
    return {
        (c["family"], c["relation"], c["kpi"]): {
            "kind": "ci",
            "lo": float(c["ci"][0]),
            "hi": float(c["ci"][1]),
        }
        for c in cells
        if c["kpi"] in GUARD_KPIS
    }


def bounds_to_json(b: Mapping | None) -> list | None:
    """Bounds as sorted [f, rel, k, record] rows."""
    return None if b is None else [[f, r, k, v] for (f, r, k), v in sorted(b.items())]


def bounds_from_json(rows: Iterable[Sequence[Any]] | None) -> dict | None:
    """Inverse of `bounds_to_json`."""
    return None if rows is None else {(str(f), str(r), str(k)): dict(v) for f, r, k, v in rows}


def tau_to_json(tau: Mapping) -> list[list[Any]]:
    """Tau as sorted [f, d, k, value] rows."""
    return [[f, int(d), k, float(v)] for (f, d, k), v in sorted(tau.items())]


def tau_from_json(rows: Iterable[Sequence[Any]]) -> dict:
    """Inverse of `tau_to_json`."""
    return {(str(f), int(d), str(k)): float(v) for f, d, k, v in rows}


def direction(r: Mapping[str, Any]) -> str:
    """A request's direction class from its knob, cur and prop."""
    typ, cur, prop = r["knob"][0], r["cur"], r["prop"]
    if typ == "carrier":
        return "car_off" if prop < cur else "car_on"
    if typ == "sleep":
        return "sleep" if prop > cur else "wake"
    if typ == "ptx":
        return "ptx_down" if prop < cur else "ptx_up"
    if typ == "prot_min":
        return "pm_up" if prop > cur else "pm_down"
    return typ


class BlanketGate:
    """Rejects every request whose direction is in `rej` once obs["t"] > warmup_s."""

    def __init__(self, rej: Iterable[str], warmup_s: float) -> None:
        self.rej, self.warmup_s = frozenset(rej), float(warmup_s)
        self.n = {"rej_" + d: 0 for d in sorted(self.rej)}

    def __call__(self, obs: Obs) -> Decision:
        """Accept or reject each request of the second."""
        t = obs["t"]
        dec = []
        for r in obs["requests"]:
            d = direction(r)
            if t > self.warmup_s and d in self.rej:
                dec.append("reject")
                self.n["rej_" + d] += 1
            else:
                dec.append("accept")
        return {"decisions": dec, "writes": [], "rollback": []}


def _fam_dir(r: Mapping[str, Any]) -> tuple[str, int]:
    return r["knob"][0], (r["prop"] > r["cur"]) - (r["prop"] < r["cur"])


class Counting:
    """Wraps an obs-only arbiter and tallies its rejects per family and phase (before WARM or not).

    For a UnitArbiter it also tallies units opened and deferred, and collateral rejects (a request
    rejected by a unit whose opening request had the other direction). Decisions are unchanged.
    """

    def __init__(self, inner: Callable[[Obs], Decision], is_unit: bool) -> None:
        self.inner, self.is_unit = inner, is_unit
        self.n: dict[str, dict[str, int]] = {
            "defer": {},
            "defer_units": {},
            "units": {},
            "collateral": {},
        }

    def _inc(self, grp: str, key: str) -> None:
        self.n[grp][key] = self.n[grp].get(key, 0) + 1

    def _units(self, ph: str) -> None:
        inner: Any = self.inner
        for u in inner.opened:
            self._inc("units", f"{u['knob']}_{ph}")
            if u["mode"] == "reject":
                self._inc("defer_units", f"{u['knob']}_{ph}")

    def __call__(self, obs: Obs) -> Decision:
        """The inner decisions, counted."""
        out = self.inner(obs)
        ph = "pre" if obs["t"] < WARM else "post"
        if self.is_unit:
            self._units(ph)
        inner: Any = self.inner
        for r, d in zip(obs["requests"], out["decisions"], strict=True):
            if d != "reject":
                continue
            f, sd = _fam_dir(r)
            self._inc("defer", f"{f}_{ph}")
            if self.is_unit:
                u = inner.active.get((int(r["knob"][1]), r["xapp"]))
                if u is not None and sd != (u["ctx"]["step"] > 0) - (u["ctx"]["step"] < 0):
                    self._inc("collateral", f)
        return out


def never_sleep() -> Counting:
    """Static reference: reject every pico sleep request from t = 0."""
    return Counting(BlanketGate({"sleep"}, warmup_s=-1.0), is_unit=False)


_PREFIXED: dict[str, Callable[[str], Callable[[Obs], Decision]]] = {
    "sub": lambda rest: subset(rest.split("+")),
    "prio": lambda rest: functools.partial(priority, order=tuple(rest.split(">"))),
    "cell": lambda rest: CellPriorityLock(tuple(rest.split(">")), hold_s=LOCK_S),
}


def anchor_arbiter(arm: str, warmup_s: float) -> Callable[[Obs], Decision] | None:
    """Arbiter of an anchor or blanket arm: freeze, noarb (None), sub:, prio:, cell:, lock, B1, B2.

    Raises:
        KeyError: Unknown arm.
    """
    if arm in B_GATES:
        return BlanketGate(B_GATES[arm], warmup_s)
    if arm == "freeze":
        return freeze
    if arm == "noarb":
        return None
    if arm == "lock":
        return KnobLock(lease_s=LOCK_S)
    head, sep, rest = arm.partition(":")
    if sep and head in _PREFIXED:
        return _PREFIXED[head](rest)
    raise KeyError(arm)


def _incumbent(sd: int) -> Counting:
    inner = DirectionalUnitArbiter(
        IncumbentPolicy(sd), T=T_UNIT, warmup_s=0.0, record=False, open_rule=OPEN_RULE
    )
    return Counting(inner, is_unit=True)


def conf_arbiter(
    arm: str, sd: int, warmup_s: float, maps: Mapping[str, Map] | None
) -> Callable[[Obs], Decision] | None:
    """Arbiter of a Study 2 (confounded) EVAL or repro arm; None = accept-all.

    Raises:
        KeyError: Unknown arm.
    """
    if arm in ANCHORS:
        return anchor_arbiter(arm, warmup_s)
    if arm == "incumbent":
        return _incumbent(sd)
    if arm == "never_sleep":
        return never_sleep()
    if arm == "B2":
        return Counting(BlanketGate(B_GATES["B2"], warmup_s), is_unit=False)
    if arm == "MG:allaccept":
        return Counting(mapgate_v2_arbiter({}, THETA, K_CONF), is_unit=True)
    if maps is not None and arm in maps:
        return Counting(mapgate_v2_arbiter(maps[arm], THETA, K_CONF), is_unit=True)
    raise KeyError(arm)


def cs_arbiter(
    arm: str, sd: int, warmup_s: float, art: Mapping[str, Any] | None
) -> Callable[[Obs], Decision] | None:
    """Arbiter of a Study 3 (certified-safe) arm; `art` from `load_certsafe`.

    Raises:
        KeyError: Unknown arm.
    """
    if arm in ANCHORS:
        return anchor_arbiter(arm, warmup_s)
    if arm == "incumbent":
        return _incumbent(sd)
    if arm == "never_sleep":
        return never_sleep()
    if arm == CAL_ARM:
        return Counting(certsafe_arbiter({}, (), THETA, K_CONF), is_unit=True)
    if arm == "MG:PMRT" and art is not None:
        return Counting(mapgate_v2_arbiter(art["maps"]["MG:PMRT"], THETA, K_CONF), is_unit=True)
    if art is not None and arm in art["tables"]:
        t = art["tables"][arm]
        return Counting(certsafe_arbiter(t["map"], t["uncertified"], THETA, K_CONF), is_unit=True)
    raise KeyError(arm)


def policy_counts(arb: Any) -> dict[str, Any]:
    """The `policy_counts` field of a Study 2 / 3 job record."""
    if isinstance(arb, Counting):
        pc: dict[str, Any] = dict(arb.n)
        pol = getattr(arb.inner, "policy", None)
        pc["policy"] = dict(getattr(pol, "n", None) or getattr(arb.inner, "n", None) or {})
        if hasattr(arb.inner, "passed"):
            pc["passed"] = {"all": int(arb.inner.passed)}
        return pc
    if arb is not None:
        return {"policy": dict(getattr(arb, "n", {}) or {})}
    return {}


def _alias(order: Sequence[str], sig: Mapping[str, str], theta: float, noarb_alias: bool) -> dict:
    aa = signature_key(all_accept_signature(theta))
    groups: dict[str, list[str]] = {}
    alias: dict[str, str] = {}
    for a in order:
        groups.setdefault(sig[a], []).append(a)
    for k, arms in groups.items():
        tgt = "noarb" if (k == aa and noarb_alias) else arms[0]
        for a in arms:
            alias[a] = tgt
    return {
        "signature": dict(sig),
        "alias_of": alias,
        "groups": groups,
        "jobs": [a for a in order if alias[a] == a],
        "all_accept": aa,
        "noarb_alias": bool(noarb_alias),
    }


def alias_table(
    maps: Mapping[str, Map], theta: float = THETA, noarb_alias: bool = False
) -> dict[str, Any]:
    """Groups map arms by decision signature; each group is simulated once (its first arm).

    An all-accept group is aliased to "noarb" iff `noarb_alias`.
    """
    order = [a for a in MAP_ARMS if a in maps] + sorted(a for a in maps if a not in MAP_ARMS)
    sig = {a: signature_key(decision_signature(maps[a], theta)) for a in order}
    return _alias(order, sig, theta, noarb_alias)


def cs_tables(
    maps: Mapping[str, Map],
    bounds: Mapping[str, Mapping | None],
    tau: Mapping,
    theta: float = THETA,
) -> dict[str, dict[str, Any]]:
    """Per CS arm whose source map is in `maps`: its artifact table.

    The table holds source, map, bound method, cert, uncertified, signature and decision table;
    `bounds` is keyed by bound method.
    """
    out = {}
    for arm in CS_ARMS:
        src = CS_OF[arm]
        if src not in maps:
            continue
        M = maps[src]
        meth = BOUND_METHOD[arm]
        cert = certify_map(M, bounds.get(meth) if meth else None, tau, theta)
        unc = [tuple(x) for x in cert["uncertified"]]
        dt = {
            f"{f}{'+' if d > 0 else '-'}": ("accept (uncertified: " + s + ")")
            if (f, d) in unc
            else s
            for (f, d), s in decision_table_v2(M, theta).items()
        }
        out[arm] = {
            "source": src,
            "map": M,
            "bound_method": meth,
            "cert": cert,
            "uncertified": unc,
            "signature": signature_key(certsafe_signature(M, unc, theta)),
            "decision_table": dt,
        }
    return out


def cs_alias_table(
    tables: Mapping[str, Mapping[str, Any]], theta: float = THETA, noarb_alias: bool = False
) -> dict[str, Any]:
    """`alias_table` over CS arms with the certified-safe signature."""
    order = [a for a in CS_ARMS if a in tables]
    return _alias(order, {a: tables[a]["signature"] for a in order}, theta, noarb_alias)


def load_conf_maps(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Maps and alias table of the Study 2 maps artifact (parsed JSON), checked to reproduce.

    Raises:
        ValueError: The alias table does not reproduce.
    """
    maps = {arm: map_from_json(v["map"]) for arm, v in doc["arms"].items()}
    al = alias_table(maps, float(doc["mapgate"]["theta"]), bool(doc["noarb_alias"]))
    if al["jobs"] != doc["jobs"] or al["alias_of"] != doc["alias_of"]:
        raise ValueError(
            f"maps artifact: alias table does not reproduce ({doc['jobs']} vs {al['jobs']})"
        )
    return {"maps": maps, "alias": al}


def load_certsafe(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Tables, alias table and source maps of the Study 3 artifact (parsed JSON), checked.

    Raises:
        ValueError: A signature or the alias table does not reproduce.
    """
    theta = float(doc["mapgate"]["theta"])
    tables = {}
    for arm, v in doc["arms"].items():
        M = map_from_json(v["map"])
        unc = [tuple(x) for x in v["uncertified"]]
        sig = signature_key(certsafe_signature(M, unc, theta))
        if sig != v["signature"]:
            raise ValueError(f"certsafe artifact: signature of {arm} does not reproduce")
        tables[arm] = {"source": v["source"], "map": M, "uncertified": unc, "signature": sig}
    al = cs_alias_table(tables, theta, bool(doc["noarb_alias"]))
    if al["jobs"] != doc["jobs"] or al["alias_of"] != doc["alias_of"]:
        raise ValueError(f"certsafe artifact: alias table does not reproduce ({doc['jobs']})")
    maps = {v["source"]: tables[k]["map"] for k, v in doc["arms"].items()}
    return {"tables": tables, "alias": al, "maps": maps}


def tau_from_accept_all(g: Mapping[str, float], n: Mapping[tuple[str, int], float]) -> dict:
    """Gate tolerances tau(f, d, k) over the study's request classes and guard edges."""
    return tau_table(g, n, GATE_SPACE)

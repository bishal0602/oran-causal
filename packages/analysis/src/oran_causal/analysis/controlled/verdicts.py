"""Claim verdicts C1, C2a, C2b and C3 of Study 4 (protocol section 10).

C1: design-blind tests are invalid on the R2 design. C2a: the design-based test is valid on
R1 and R2. C2b: design-covariate adjustment restores validity for the eq arms whose native
partner fails C1. C3: the design-based test stays valid under the logged confounded policy
(E4 R3). The rules read the cell summaries of `cells.build_cells`; `F_max` comes from the
simulation in `fmax`.
"""

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from oran_causal.analysis.controlled.cells import Cell, Cells, Rows, cell_class, counts
from oran_causal.analysis.controlled.fmax import Dependence, fmax_simulate
from oran_causal.analysis.controlled.protocol import (
    C3_P_KEYS,
    C3_TAU_KEYS,
    EQ_TAU_KEYS,
    MIN_COVERAGE,
    P_KEYS,
    PRIMARY_KAPPA,
)
from oran_causal.analysis.controlled.spec import (
    Spec,
    arm_kind,
    arm_label,
    focal_arm,
    native_partner,
)
from oran_causal.analysis.controlled.validity import combine, rate

Verdict = dict[str, Any]
Predicate = Callable[[Cell], bool]
R1R2 = ("R1", "R2")


@dataclasses.dataclass(frozen=True)
class Evidence:
    """The cell summaries a verdict rule reads.

    Attributes:
        cells: Cell summaries of all arms.
        rows: Per-seed rows by cell.
        spec: Campaign spec.
        dep: Dependence table for `F_max`, or None.
    """

    cells: Cells
    rows: Rows
    spec: Spec
    dep: Dependence | None


def rate_text(x: dict[str, Any] | None) -> str:
    """Formats a rate with its interval, or `n/a`."""
    if not x or x.get("rate") is None:
        return "n/a"
    if x.get("ci"):
        return f"{x['rate']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
    return f"{x['rate']:.3f}"


def pooled_rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Pools a rate over rows with clusters `(world, seed)`, sorted."""
    agg: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for x in rows:
        cluster = agg[(x["world"], x["seed"])]
        cluster[0] += x["cnt"][key][0]
        cluster[1] += x["cnt"][key][1]
    ordered = sorted(agg)
    return rate([agg[k][0] for k in ordered], [agg[k][1] for k in ordered])


def set_of(cells: Cells, arm: str, pred: Predicate) -> list[str]:
    """Lists an arm's pre-registered cells (primary kappa, no T3 cells) satisfying `pred`."""
    return sorted(
        k
        for k, e in cells.items()
        if e["arm"] == arm
        and e["kappa"] == PRIMARY_KAPPA
        and e["status"] != "infeasible_t3"
        and pred(e)
    )


def _fmax(ev: Evidence, keys_used: list[str], rate_keys: Sequence[str]) -> Verdict:
    if not keys_used:
        return {"f_max": 0}
    cells = ev.cells
    seeds = max(cells[k]["primary"]["n_seeds"] for k in keys_used)
    sim_cells = [
        (cells[k]["world"], cells[k]["regime"], cells[k]["lam"], cells[k]["n"]) for k in keys_used
    ]
    return fmax_simulate(sim_cells, seeds, [k.replace("_decl", "_raw") for k in rate_keys], ev.dep)


def _component_verdict(use: list[str], planned: list[str], *, passed: bool, capped: bool) -> str:
    if not use or len(use) < MIN_COVERAGE * len(planned):
        return "NOT EVALUABLE"
    if passed:
        return "PARTIAL" if capped else "SUPPORTED"
    return "NOT SUPPORTED"


def validity_component(ev: Evidence, planned: list[str], rate_keys: Sequence[str]) -> Verdict:
    """C2 / C3 rule for one arm.

    (a) The number of INVALID counted cells is at most the simulated `F_max`; (b) every pooled
    rate is VALID. SUPPORTED if both, capped at PARTIAL when a planned cell is not counted or
    is under-seeded; NOT EVALUABLE if fewer than 90 % of the planned cells are counted.

    Args:
        ev: The cell summaries.
        planned: The planned cell keys of the arm.
        rate_keys: The component's rates.

    Returns:
        Counts, `F_max`, pooled rates, the two legs and the verdict.
    """
    cells = ev.cells
    use = [k for k in planned if counts(cells[k])]
    classes = {k: cell_class(cells[k], rate_keys) for k in use}
    invalid = sorted(k for k, c in classes.items() if c == "INVALID")
    pooled = {rk: pooled_rate([x for k in use for x in ev.rows[k]], rk) for rk in rate_keys}
    pooled_class = combine((p or {}).get("validity") for p in pooled.values())
    fm = _fmax(ev, use, rate_keys)
    a_ok, b_ok = len(invalid) <= fm["f_max"], pooled_class == "VALID"
    not_counted = sorted(set(planned) - set(use))
    under = sorted(k for k in use if cells[k]["status"] == "under_seeded")
    rest = [k for k in use if cells[k]["n"] != 500]
    return {
        "n_planned": len(planned),
        "n_cells": len(use),
        "n_invalid": len(invalid),
        "n_valid": sum(c == "VALID" for c in classes.values()),
        "n_inconclusive": sum(c in ("INCONCLUSIVE", "NO_READ") for c in classes.values()),
        "f_max": fm["f_max"],
        "f_max_sim": {k: fm.get(k) for k in ("S", "p_pass", "p_count_le_fmax", "p_pooled_valid")},
        "invalid_cells": invalid,
        "pooled": pooled,
        "pooled_validity": pooled_class,
        "a_ok": a_ok,
        "b_ok": b_ok,
        "not_counted": not_counted,
        "under_seeded": under,
        "verdict": _component_verdict(
            use, planned, passed=a_ok and b_ok, capped=bool(not_counted or under)
        ),
        "without_n500": {
            "n_cells": len(rest),
            "n_invalid": sum(classes[k] == "INVALID" for k in rest),
        },
    }


def _c1_verdict(
    *,
    assessable: bool,
    r1_invalid: int,
    r1_f_max: int,
    r2_invalid: int,
    r2_planned: int,
    pooled_inv: bool,
) -> str:
    if not assessable:
        return "NOT EVALUABLE"
    if r1_invalid > r1_f_max:
        return "INVALID IN R1"
    if 2 * r2_invalid >= r2_planned and pooled_inv:
        return "FAILURE"
    return "NOT A FAILURE"


def c1_arm(ev: Evidence, arm: str) -> Verdict:
    """C1 for one design-blind arm.

    FAILURE iff (i) INVALID in at least half of its planned R2 cells (fixed denominator), (ii) a
    pooled R2 rate INVALID and (iii) the number of INVALID R1 cells is at most `F_max(R1)`.
    INVALID IN R1 if (iii) fails.
    """
    cells = ev.cells
    p2 = set_of(cells, arm, lambda e: e["regime"] == "R2")
    p1 = set_of(cells, arm, lambda e: e["regime"] == "R1")
    u2 = [k for k in p2 if counts(cells[k])]
    u1 = [k for k in p1 if counts(cells[k])]
    n2 = sum(cell_class(cells[k], P_KEYS) == "INVALID" for k in u2)
    n1 = sum(cell_class(cells[k], P_KEYS) == "INVALID" for k in u1)
    pooled2 = {rk: pooled_rate([x for k in u2 for x in ev.rows[k]], rk) for rk in P_KEYS}
    pooled1 = {rk: pooled_rate([x for k in u1 for x in ev.rows[k]], rk) for rk in P_KEYS}
    pooled_inv = any((p or {}).get("validity") == "INVALID" for p in pooled2.values())
    fm1 = _fmax(ev, u1, P_KEYS)
    complete = (
        len(u2) == len(p2)
        and len(u1) == len(p1)
        and not any(cells[k]["status"] == "under_seeded" for k in u1 + u2)
    )
    assessable = (
        bool(p2) and len(u2) >= MIN_COVERAGE * len(p2) and len(u1) >= MIN_COVERAGE * len(p1)
    )
    verdict = _c1_verdict(
        assessable=assessable,
        r1_invalid=n1,
        r1_f_max=fm1["f_max"],
        r2_invalid=n2,
        r2_planned=len(p2),
        pooled_inv=pooled_inv,
    )
    return {
        "verdict": verdict,
        "complete": complete,
        "assessable": assessable,
        "r2_invalid": n2,
        "r2_planned": len(p2),
        "r2_counted": len(u2),
        "r1_invalid": n1,
        "r1_planned": len(p1),
        "r1_counted": len(u1),
        "r1_f_max": fm1["f_max"],
        "pooled_r2": pooled2,
        "pooled_r1": pooled1,
        "pooled_r2_invalid": pooled_inv,
        "r2_legs": bool(assessable and 2 * n2 >= len(p2) and pooled_inv),
    }


def majority(passes: list[bool], *, complete: bool, n_min: int = 1) -> str:
    """Majority verdict over arms: SUPPORTED at half or more, NOT SUPPORTED at none, else PARTIAL.

    NOT EVALUABLE below `n_min` arms; SUPPORTED is capped at PARTIAL when not `complete`.
    """
    if len(passes) < n_min:
        return "NOT EVALUABLE"
    k = sum(passes)
    if 2 * k >= len(passes):
        verdict = "SUPPORTED"
    elif k == 0:
        verdict = "NOT SUPPORTED"
    else:
        verdict = "PARTIAL"
    return "PARTIAL" if (verdict == "SUPPORTED" and not complete) else verdict


def _c1_wording(hac_ok: list[str]) -> str:
    if not hac_ok:
        return "design-blind tests are invalid on the R2 (setpoint + dither) design tested"
    return (
        "design-blind tests with i.i.d. / exchangeable nulls are invalid on the R2 design tested; "
        "the serial-dependence-robust design-blind test (" + ", ".join(hac_ok) + ") is not"
    )


def _c1(ev: Evidence) -> tuple[Verdict, dict[str, Verdict]]:
    arms = ev.spec["arms"]
    blind = sorted(a for a, d in arms.items() if arm_kind(d) == "native" and d["declare"] == "by")
    design_blind = [a for a in blind if arms[a].get("set_D", True) is True]
    c1_arms = {a: c1_arm(ev, a) for a in design_blind}
    counted = [a for a in design_blind if c1_arms[a]["assessable"]]
    c1: Verdict = {
        "D": design_blind,
        "D_counted": counted,
        "arms": c1_arms,
        "descriptive": {a: c1_arm(ev, a) for a in blind if a not in design_blind},
        "n_failures": sum(c1_arms[a]["verdict"] == "FAILURE" for a in counted),
        "verdict": majority(
            [c1_arms[a]["verdict"] == "FAILURE" for a in counted],
            complete=all(v["complete"] for v in c1_arms.values()),
            n_min=3,
        ),
    }
    dropped = [a for a in design_blind if arms[a].get("c1_sensitivity_drop")]
    if dropped:
        keep = [a for a in counted if a not in dropped]
        c1["sensitivity_without"] = {
            "dropped": dropped,
            "D_counted": keep,
            "n_failures": sum(c1_arms[a]["verdict"] == "FAILURE" for a in keep),
            "verdict": majority(
                [c1_arms[a]["verdict"] == "FAILURE" for a in keep],
                complete=all(c1_arms[a]["complete"] for a in design_blind if a not in dropped),
                n_min=3,
            ),
        }
    hac_ok = [a for a in design_blind if "hac" in a and c1_arms[a]["verdict"] == "NOT A FAILURE"]
    c1["wording"] = _c1_wording(hac_ok)
    return c1, c1_arms


def _r1r2(ev: Evidence, arm: str) -> Verdict:
    planned = set_of(ev.cells, arm, lambda e: e["regime"] in R1R2)
    return validity_component(ev, planned, P_KEYS)


def _is_by_eq(spec: Spec, arm: str) -> bool:
    return arm_kind(spec["arms"][arm]) == "eq" and spec["arms"][arm]["declare"] == "by"


def _c2b_majority(evaluable: dict[str, Verdict], *, all_evaluable: bool) -> str:
    passes = [v["verdict"] in ("SUPPORTED", "PARTIAL") for v in evaluable.values()]
    complete = all_evaluable and all(v["verdict"] != "PARTIAL" for v in evaluable.values())
    return majority(passes, complete=complete)


@dataclasses.dataclass(frozen=True)
class C2b:
    """The C2b verdict with its member, excluded and evaluable arms."""

    verdict: Verdict
    members: dict[str, Verdict]
    excluded: dict[str, Verdict]
    evaluable: dict[str, Verdict]


def _c2b(ev: Evidence, c1: Verdict, c1_arms: dict[str, Verdict]) -> C2b:
    spec = ev.spec
    arms = spec["arms"]
    members, excluded = {}, {}
    for a in sorted(arms):
        if not _is_by_eq(spec, a):
            continue
        partner = native_partner(a, spec)
        if arms[a].get("c2b", True) is not True:
            excluded[a] = {"native": partner, "label": arm_label(a, spec), **_r1r2(ev, a)}
        elif partner in c1_arms and c1_arms[partner]["verdict"] == "FAILURE":
            members[a] = {"native": partner, **_r1r2(ev, a)}
    evaluable = {a: v for a, v in members.items() if v["verdict"] != "NOT EVALUABLE"}
    verdict: Verdict = {
        "arms": members,
        "excluded": excluded,
        "verdict": _c2b_majority(evaluable, all_evaluable=len(evaluable) == len(members)),
    }
    extra = {}
    for a in sorted(arms):
        if not _is_by_eq(spec, a) or arms[a].get("c2b", True) is not True:
            continue
        partner = native_partner(a, spec)
        pv = c1_arms.get(partner) or c1["descriptive"].get(partner)
        if a not in members and pv and pv["verdict"] == "INVALID IN R1" and pv["r2_legs"]:
            extra[a] = {"native": partner, **_r1r2(ev, a)}
    both = {**members, **extra}
    evaluable2 = {a: v for a, v in both.items() if v["verdict"] != "NOT EVALUABLE"}
    verdict["with_r1_failing_partners"] = {
        "added": sorted(extra),
        "arms": extra,
        "verdict": _c2b_majority(evaluable2, all_evaluable=len(evaluable2) == len(both)),
    }
    return C2b(verdict, members, excluded, evaluable)


def _membership(spec: Spec, arm: str, c2b: C2b, partner_verdict: Verdict | None) -> str:
    declare = spec["arms"][arm]["declare"]
    if arm in c2b.members:
        return "member"
    if arm in c2b.excluded:
        return "excluded (R-40)"
    if declare != "by":
        return f"not a member (declare {declare})"
    shown = partner_verdict["verdict"] if partner_verdict else "not assessed"
    return f"not a member (native {native_partner(arm, spec)}: {shown})"


def _named_sentence(arm: str, partner: str | None, own: Verdict) -> str:
    return (
        f"{arm} is INVALID in R2 (pooled truth-null raw rate "
        f"{rate_text(own['pooled_r2'].get('null_raw'))}) as already in R1 "
        f"({rate_text(own['pooled_r1'].get('null_raw'))}): the test is miscalibrated without the "
        f"design "
        f"(its native partner {partner} is INVALID IN R1), so it is outside C2b's membership rule, "
        f"and "
        f"adding design covariates does not make it valid"
    )


def _eq_row(ev: Evidence, arm: str, c1: Verdict, c1_arms: dict[str, Verdict], c2b: C2b) -> Verdict:
    spec = ev.spec
    declare = spec["arms"][arm]["declare"]
    partner = native_partner(arm, spec)
    pv = c1_arms.get(partner) or c1["descriptive"].get(partner)
    keys = P_KEYS if declare == "by" else EQ_TAU_KEYS
    planned = set_of(ev.cells, arm, lambda e: e["regime"] in R1R2)
    x = validity_component(ev, planned, keys)
    own = c1_arm(ev, arm) if declare == "by" else None
    return {
        "native": partner,
        "native_c1": pv["verdict"] if pv else None,
        "membership": _membership(spec, arm, c2b, pv),
        "label": arm_label(arm, spec),
        "verdict": x["verdict"],
        "n_cells": x["n_cells"],
        "n_invalid": x["n_invalid"],
        "f_max": x["f_max"],
        "pooled": x["pooled"],
        "r2_invalid": own["r2_invalid"] if own else None,
        "r2_planned": own["r2_planned"] if own else None,
        "pooled_r1": own["pooled_r1"] if own else None,
        "pooled_r2": own["pooled_r2"] if own else None,
        "_own": own,
    }


def _eq_table(
    ev: Evidence, c1: Verdict, c1_arms: dict[str, Verdict], c2b: C2b
) -> tuple[dict[str, Verdict], list[Verdict]]:
    arms = ev.spec["arms"]
    table: dict[str, Verdict] = {}
    named: list[Verdict] = []
    for a in sorted(arms):
        if arm_kind(arms[a]) != "eq":
            continue
        row = _eq_row(ev, a, c1, c1_arms, c2b)
        own = row.pop("_own")
        table[a] = row
        if (
            own
            and own["r2_legs"]
            and row["native_c1"] == "INVALID IN R1"
            and row["membership"] not in ("member", "excluded (R-40)")
        ):
            sentence = _named_sentence(a, row["native"], own)
            named.append({"arm": a, "native": row["native"], "sentence": sentence})
    return table, named


def _c2_wording(
    c2a: Verdict, c2b: Verdict, evaluable: dict[str, Verdict], named: list[Verdict]
) -> str:
    restored = sorted(a for a, v in evaluable.items() if v["verdict"] in ("SUPPORTED", "PARTIAL"))
    a_ok = c2a["verdict"] in ("SUPPORTED", "PARTIAL")
    if a_ok and c2b["verdict"] == "SUPPORTED":
        wording = (
            "using the design restores validity (design-based inference and design-covariate "
            "adjustment)"
        )
    elif a_ok:
        wording = (
            "design-based inference restores validity; adding design covariates does not suffice"
        )
        if restored:
            wording += f" (it does for {', '.join(restored)})"
    else:
        wording = "not supported: the design-based test is not shown valid in R1 / R2"
    if named:
        wording += (
            "; "
            + ", ".join(x["arm"] for x in named)
            + " INVALID in R2 as already in R1 (not restored by design covariates)"
        )
    return wording


def _c3(ev: Evidence) -> tuple[Verdict, dict[str, Verdict], Callable[[str], Verdict]]:
    arms = ev.spec["arms"]

    def c3_for(arm: str) -> Verdict:
        keys = C3_P_KEYS if arms[arm]["declare"] == "by" else C3_TAU_KEYS
        planned = set_of(ev.cells, arm, lambda e: e["world"] == "E4" and e["regime"] == "R3")
        return validity_component(ev, planned, keys)

    focal = focal_arm(ev.spec)
    c3 = c3_for(focal) if focal in arms else {"verdict": "NOT EVALUABLE"}
    c3_eq = {a: c3_for(a) for a in sorted(arms) if arm_kind(arms[a]) == "eq"}
    also = sorted(a for a, v in c3_eq.items() if v["verdict"] in ("SUPPORTED", "PARTIAL"))
    if c3["verdict"] in ("SUPPORTED", "PARTIAL"):
        wording = "PMRT stays valid under the logged confounded policy (R3) by construction"
        if also:
            wording += f"; so do {', '.join(also)}, so the property is not specific to PMRT"
        c3["wording"] = wording
    else:
        c3["wording"] = "not supported"
    return c3, c3_eq, c3_for


def verdicts(cells: Cells, rows: Rows, spec: Spec, dep: Dependence | None) -> Verdict:
    """Computes the claim verdicts C1, C2a, C2b, C3 and the overall claim.

    Args:
        cells: Cells of all arms.
        rows: Per-seed rows by cell.
        spec: Campaign spec.
        dep: Dependence table for `F_max`, or None.

    Returns:
        The components with their evidence, the eq-arm table, the named-arm sentences, the
        wording of C2 and C3 and `claim` (SUPPORTED only if all four components are).
    """
    ev = Evidence(cells, rows, spec, dep)
    arms = spec["arms"]
    focal = focal_arm(spec)
    c1, c1_arms = _c1(ev)
    c2a = validity_component(ev, set_of(cells, focal, lambda e: e["regime"] in R1R2), P_KEYS)
    c2b = _c2b(ev, c1, c1_arms)
    eq_table, named = _eq_table(ev, c1, c1_arms, c2b)
    c3, c3_eq, c3_for = _c3(ev)
    pmrt_secondary = {
        a: {"label": arm_label(a, spec), "C2a": _r1r2(ev, a), "C3": c3_for(a)}
        for a in sorted(arms)
        if arm_kind(arms[a]) == "pmrt" and a != focal
    }
    components = {
        "C1": c1["verdict"],
        "C2a": c2a["verdict"],
        "C2b": c2b.verdict["verdict"],
        "C3": c3["verdict"],
    }
    return {
        "focal": focal,
        "C1": c1,
        "C2a": c2a,
        "C2b": c2b.verdict,
        "C2_wording": _c2_wording(c2a, c2b.verdict, c2b.evaluable, named),
        "C2_named": named,
        "eq_arms_table": eq_table,
        "C3": c3,
        "C3_eq_arms": c3_eq,
        "pmrt_secondary": pmrt_secondary,
        "components": components,
        "claim": "SUPPORTED"
        if all(v == "SUPPORTED" for v in components.values())
        else "NOT SUPPORTED",
        "arm_labels": {a: arm_label(a, spec) for a in sorted(arms) if arm_label(a, spec)},
    }

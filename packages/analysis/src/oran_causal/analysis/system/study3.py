"""Study 3 criteria: the certified-safe referee (follow-up 2b).

Gate A on every arm with one paired bootstrap, as Study 2: E (the gated PMRT arm is eligible),
D1 and D2 over the 13 gated associational arms (Study 2 definitions), D3 (R*(CS:PMRT) -
R*(never_sleep) > 0 with paired LB90 > 0) and the secondary S1 (upper end of the 90 % CI of the
RLF ratio <= 1.20). Verdict precedence: INVALID (Study 2 K0 or K0n) > NOT ELIGIBLE > PASS (D1, D2
and D3) > PARTIAL (one or two) > FAIL.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.analysis.system import study2
from oran_causal.analysis.system.stats import (
    ANCHORS,
    N_BOOT,
    GateA,
    JobFilter,
    Records,
    ci,
    gate_a,
    platform_key,
)

Check = dict[str, Any]

SCHEMA_EVAL_REPORT = "e6p-certsafe-eval/1"
SCHEMA_EVAL = study2.SCHEMA_EVAL
SUB = "certsafe"
DRIVER = "e6p_certsafe"
LAYOUT = {"dev": (191000, 40), "eval": (191100, 160)}
CAL_ARM = "CAL:allaccept"
REFS = ("incumbent", "never_sleep", "MG:PMRT")
CS_OF = {
    "CS:PMRT": "MG:PMRT",
    "CS:GT": "MG:GT",
    "CS:rand": "MG:rand",
    "CS:blanket2": "blanket2",
    **{f"CS:{b}": f"MG:{b}" for b in study2.ASSOC},
}
CS_ASSOC = tuple(f"CS:{b}" for b in study2.ASSOC)
CS_ARMS = tuple(CS_OF)
ALL_ARMS = (*ANCHORS, *REFS, *CS_ARMS)
PRIMARY_ARM = "CS:PMRT"
CERTSAFE_KEY = "cdd_oran/decision/certsafe.py"
RLF_CI_MAX = 1.20
SECONDARY = ("noarb", "incumbent", "never_sleep", "MG:PMRT", "CS:GT", "CS:rand", "CS:blanket2")
COUNT_GROUPS = ("defer", "defer_units", "units", "policy", "passed")
VERDICT_KEYS = ("verdict", "K0", "K0n", "E", "D1", "D2", "D3", "S1", "n_seeds")


def stage_seeds(stage: str) -> list[int]:
    """Seeds of one stage."""
    b, n = LAYOUT[stage]
    return list(range(b, b + n))


def job_filter(stage: str, allow_smoke: bool = False) -> JobFilter:
    """Job and header filter of the certsafe driver's records of `stage`."""
    return JobFilter(
        schema=SCHEMA_EVAL,
        sub=SUB,
        stage_key="cs_stage",
        stage=stage,
        driver=DRIVER,
        allow_smoke=allow_smoke,
    )


def d3_check(
    point: Mapping[str, Mapping[str, Any]],
    boot: Mapping[str, Mapping[str, Any]],
    primary: str = PRIMARY_ARM,
    ref: str = "never_sleep",
) -> Check:
    """D3: R*(primary) - R*(ref) > 0 with paired LB90 (5th percentile) > 0."""
    d = float(point[primary]["Rstar"] - point[ref]["Rstar"])
    diff = boot[primary]["Rstar"] - boot[ref]["Rstar"]
    fin = diff[np.isfinite(diff)]
    lb = float(np.quantile(fin, 0.05)) if fin.size else float("nan")
    return {
        "pass": bool(np.isfinite(d) and d > 0 and np.isfinite(lb) and lb > 0),
        "delta3": d,
        "lb90": lb,
        "ci90": ci(diff),
        "ref": ref,
    }


def s1_check(
    boot: Mapping[str, Mapping[str, Any]], primary: str = PRIMARY_ARM, cap: float = RLF_CI_MAX
) -> Check:
    """S1: the 95th percentile of the primary's bootstrap RLF ratio is at most `cap`."""
    r = boot[primary]["guard"]["rlf"]
    fin = r[np.isfinite(r)]
    ub = float(np.quantile(fin, 0.95)) if fin.size else float("nan")
    return {"pass": bool(np.isfinite(ub) and ub <= cap), "rlf_ratio_ub90": ub, "cap": cap}


def eval_verdict(
    *, k0: bool, k0n: bool, e: bool, d1: bool, d2: bool, d3: bool, why: Sequence[str] = ()
) -> dict[str, Any]:
    """The Study 3 verdict; complete iff `why` (the incompleteness reasons) is empty."""
    if not (k0 and k0n):
        v = "INVALID"
    elif not e:
        v = "NOT ELIGIBLE"
    elif d1 and d2 and d3:
        v = "PASS"
    elif d1 or d2 or d3:
        v = "PARTIAL"
    else:
        v = "FAIL"
    complete = not why
    return {
        "verdict": v,
        "label": v if complete else f"NOT A VERDICT (would-be: {v}; {'; '.join(why)})",
        "complete": complete,
        "precedence": "INVALID > NOT ELIGIBLE (E) > PASS (D1, D2, D3) > PARTIAL > FAIL",
    }


@dataclasses.dataclass(frozen=True)
class EvalInputs:
    """Frozen inputs of the Study 3 evaluation besides the EVAL records.

    Attributes:
        disc: The Study 2 discovery analysis (`disc_conf.json`; K0 and K0n).
        artifact: The parsed certsafe artifact (`E6P_CERTSAFE.json`).
        artifact_verify: Its verification report (`ok`, `driver_matches`, ...).
        frozen: Whether the protocol is frozen.
        argv: Command line recorded in the report.
    """

    disc: Mapping[str, Any]
    artifact: Mapping[str, Any]
    artifact_verify: Mapping[str, Any]
    frozen: bool
    argv: Sequence[str] = ()


def _incomplete(
    R: Records,
    heads: Sequence[Mapping[str, Any]],
    st: GateA,
    missing: Sequence[str],
    inp: EvalInputs,
) -> list[str]:
    av = inp.artifact_verify
    why = []
    if st.seeds != stage_seeds("eval"):
        why.append(f"eval seeds with every arm: {st.n} / 160")
    if missing:
        why.append(f"missing arms {list(missing)}")
    if any(r.get("smoke") for arms in R.values() for r in arms.values()):
        why.append("smoke records")
    if not av["ok"] or not av["driver_matches"]:
        why.append("certsafe artifact sha / checks (CERTSAFE_SHA256)")
    if not inp.frozen:
        why.append("protocol not frozen (FROZEN_SHA256_CS)")
    plats = {platform_key(h.get("platform")) for h in heads if not h.get("smoke")}
    if len(plats) != 1:
        why.append(f"platform fingerprints differ across eval shards: {sorted(map(str, plats))}")
    if {h.get("certsafe_sha256") for h in heads} - {inp.artifact["sha256"][CERTSAFE_KEY]}:
        why.append("certsafe.py sha256 of the eval runs differs from the artifact")
    return why


def evaluate(
    R: Records, heads: Sequence[Mapping[str, Any]], inp: EvalInputs, n_boot: int = N_BOOT
) -> dict[str, Any]:
    """The Study 3 evaluation report (`eval_certsafe.json` without `timing_s`).

    Args:
        R: EVAL job records `{seed: {arm: record}}` with aliased arms filled (`stats.alias_fill`).
        heads: EVAL run headers.
        inp: Discovery report, certsafe artifact and status inputs.
        n_boot: Bootstrap resamples.

    Returns:
        The report; `verdict_record` selects the verdict file.

    Raises:
        ValueError: No EVAL seed has every anchor and arm.
    """
    arms = [x for x in ALL_ARMS if any(x in R[s] for s in R)]
    missing = [x for x in ALL_ARMS if x not in arms]
    st = gate_a(R, arms, n_boot)
    if not st.n:
        raise ValueError("no EVAL seed with every anchor and arm")
    P, B = st.point, st.boot
    rows = study2.arm_rows(R, st, COUNT_GROUPS, cpu=False)
    A = [x for x in CS_ASSOC if x in P]
    sig = {arm: v["signature"] for arm, v in inp.artifact["arms"].items()}
    pa = PRIMARY_ARM
    have = pa in P and bool(A)
    E = study2.eligibility(rows, pa)
    D1 = study2.d1_check(P, B, pa, A) if have else {"pass": False, "note": "arms missing"}
    D2 = study2.d2_check(B, pa, A, sig) if have else {"pass": False, "note": "arms missing"}
    D3 = (
        d3_check(P, B)
        if pa in P and "never_sleep" in P
        else {"pass": False, "note": "arms missing"}
    )
    S1 = s1_check(B) if pa in P else {"pass": False, "note": "arm missing"}
    k0, k0n = bool(inp.disc["K0"]["pass"]), bool((inp.disc.get("K0n") or {}).get("pass"))
    why = _incomplete(R, heads, st, missing, inp)
    return {
        "schema": SCHEMA_EVAL_REPORT,
        "argv": list(inp.argv),
        "artifact_verify": inp.artifact_verify,
        "n_seeds": st.n,
        "arms_present": arms,
        "missing_arms": missing,
        "gate_a": st.summary(),
        "arms": rows,
        "K0": k0,
        "K0n": k0n,
        "E": E,
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "S1": S1,
        "secondary": study2.secondary(st, pa, SECONDARY),
        "verdict": eval_verdict(
            k0=k0, k0n=k0n, e=E["pass"], d1=D1["pass"], d2=D2["pass"], d3=D3["pass"], why=why
        ),
        "cpu_h_eval": study2.cpu_hours(R),
    }


def verdict_record(rep: Mapping[str, Any]) -> dict[str, Any]:
    """The `verdict_certsafe.json` subset of an evaluation report."""
    return {k: rep[k] for k in VERDICT_KEYS}

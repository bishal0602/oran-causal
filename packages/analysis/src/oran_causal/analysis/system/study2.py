"""Study 2 criteria: PMRT on confounded incumbent logs, then the MapGateV2 referee (option (a)).

Discovery (protocol section 7.1) reuses Study 1's K0, K0n and G and adds K1 (>= 500 tested sleep
units with >= 75 rejects) and X1 (premise declared +), X2 (precision >= .80 and sign accuracy
>= .90 over the declared pv / v / rlf / e edges, INDET excluded), X3 (>= min(3, |C*|) chain hits).
Label precedence: INVALID > NO-CHAIN > UNDERPOWERED > DISC-PASS (X1, X2, X3) / DISC-PARTIAL (X2)
/ DISC-FAIL.

Evaluation (section 7.2) runs Gate A on every arm with one paired bootstrap: E (the PMRT map arm
is eligible), D1 (Delta1 = R*(PMRT) - max over the associational maps >= .10 with the best map
re-selected per resample and LB90 > 0) and D2 (one-sided paired bootstrap p per distinct
associational signature, Holm at .05). Verdict precedence: INVALID > NOT ELIGIBLE > PASS (D1 and
D2) / PARTIAL (one) / FAIL; "NOT A VERDICT" unless the data are complete and every hash matches.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from oran_causal.analysis.system import study1
from oran_causal.analysis.system.stats import (
    ANCHORS,
    COUNT_GROUPS,
    N_BOOT,
    GateA,
    JobFilter,
    Records,
    ci,
    gate_a,
    platform_key,
    sum_counts,
)

Check = dict[str, Any]

SCHEMA_EVAL_REPORT = "e6p-conf-eval/1"
SCHEMA_EVAL = "e6p-optaka2-rec/1"
SCHEMA_DISC = "e6p-conf-disc/1"
SUB = "conf"
DRIVER = "e6p_conf"
LAYOUT = {
    "dev_conf": (186080, 20),
    "disc": (186100, 600),
    "placebo": (186700, 200),
    "gt": (186900, 40),
    "eval": (187000, 160),
}
REFS = ("incumbent", "never_sleep", "B2")
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
ALL_ARMS = (*ANCHORS, *REFS, *MAP_ARMS)
PRIMARY_ARM = "MG:PMRT"
SECONDARY = ("noarb", "incumbent", "blanket2", "never_sleep", "B2", "MG:GT", "MG:rand")
K1_RULE = {"min_sleep_units": 500, "min_sleep_rejects": 75}
X2_RULE: dict[str, Any] = {"kpis": ("pv", "v", "rlf", "e"), "precision": 0.80, "sign_acc": 0.90}
X3_RULE = {"chain_min": 3}
D1_MARGIN = 0.10
ALPHA_D2 = 0.05
NONINF = -0.10
NONINF_STATEMENT = (
    "pre-registered: the study does NOT expect and will NOT claim that the PMRT map beats the "
    "best static rule; reported whatever it shows"
)
EVAL_FILTER = JobFilter(
    schema=SCHEMA_EVAL, sub=SUB, stage_key="conf_stage", stage="eval", driver=DRIVER
)
VERDICT_KEYS = (
    "verdict",
    "K0",
    "K0n",
    "E",
    "D1",
    "D2",
    "never_sleep",
    "discovery_label",
    "maps_verify",
    "n_seeds",
)


def stage_seeds(stage: str) -> list[int]:
    """Seeds of one stage of the section 6 layout."""
    b, n = LAYOUT[stage]
    return list(range(b, b + n))


def _x2(decl: study1.Decl, ref: study1.Ref) -> Check:
    tp = fp = sign_ok = n_true = 0
    rows = []
    for h, v in sorted(decl.items()):
        if not v or not v.get("declared") or h[2] not in X2_RULE["kpis"]:
            continue
        g = ref.get(tuple(h), {"status": "INDET", "sign": 0})
        s = int(v.get("sign", 0))
        rows.append(["|".join(h), s, g["status"], int(g["sign"])])
        if g["status"] == "INDET":
            continue
        if g["status"] == "TRUE":
            n_true += 1
            sign_ok += int(s == int(g["sign"]))
        if g["status"] == "TRUE" and s == int(g["sign"]):
            tp += 1
        else:
            fp += 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    sacc = sign_ok / n_true if n_true else float("nan")
    ok = bool(
        np.isfinite(prec)
        and prec >= X2_RULE["precision"] - 1e-12
        and np.isfinite(sacc)
        and sacc >= X2_RULE["sign_acc"] - 1e-12
    )
    return {
        "pass": ok,
        "precision": prec,
        "sign_accuracy": sacc,
        "n_scored": tp + fp,
        "n_true": n_true,
        "edges": rows,
        "rule": {k: list(v) if isinstance(v, tuple) else v for k, v in X2_RULE.items()},
    }


def x_checks(decl: study1.Decl, ref: study1.Ref) -> dict[str, Check]:
    """X1, X2 and X3 of the pooled DISC declarations of the primary combination."""
    prem = decl.get(study1.PREMISE) or {}
    m = study1.subset_metrics(decl, ref)
    need = min(X3_RULE["chain_min"], int(m["chain_true"]))
    return {
        "X1": {
            "pass": bool(prem.get("declared")) and int(prem.get("sign", 0)) == 1,
            "premise": {k: prem.get(k) for k in ("declared", "sign", "p")},
        },
        "X2": _x2(decl, ref),
        "X3": {
            "pass": bool(m["chain_hits"] >= need),
            "chain_hits": int(m["chain_hits"]),
            "chain_true": int(m["chain_true"]),
            "need": need,
            "chain": m["chain"],
        },
    }


def k1_check(units: Mapping[str, Any], rule: Mapping[str, Any] = K1_RULE) -> Check:
    """K1 on the pooled DISC unit counts."""
    ok = (
        units["sleep_units"] >= rule["min_sleep_units"]
        and units["sleep_rejects"] >= rule["min_sleep_rejects"]
    )
    return {
        "pass": bool(ok),
        "sleep_units": units["sleep_units"],
        "sleep_rejects": units["sleep_rejects"],
        "rule": dict(rule),
        "note": "provisional thresholds (protocol section 7.1 / 13.5)",
    }


def disc_label(
    k0: Check | None, k0n: Check | None, g: Check | None, k1: Check | None, x: Mapping[str, Check]
) -> dict[str, Any]:
    """The discovery label by the section 7.1 precedence."""
    if not (k0 and k0["pass"]) or not (k0n and k0n["pass"]):
        lab = "INVALID"
    elif not (g and g["pass"]):
        lab = "NO-CHAIN"
    elif not (k1 and k1["pass"]):
        lab = "UNDERPOWERED"
    elif x["X1"]["pass"] and x["X2"]["pass"] and x["X3"]["pass"]:
        lab = "DISC-PASS"
    elif x["X2"]["pass"]:
        lab = "DISC-PARTIAL"
    else:
        lab = "DISC-FAIL"
    return {
        "label": lab,
        "stops_before_eval": lab == "INVALID",
        "precedence": "INVALID (K0 or K0n) > NO-CHAIN (G) > UNDERPOWERED (K1) > "
        "DISC-PASS (X1 X2 X3) / DISC-PARTIAL (X2) / DISC-FAIL",
    }


def data_status(
    checks: Mapping[str, Mapping[str, Any]], gt_seeds: Sequence[int], dry: bool
) -> dict[str, Any]:
    """Completeness of the discovery data from the per-stage record checks.

    Args:
        checks: Per stage (dev_conf, disc, placebo) `{seeds, smoke, ok}` of the record checks.
        gt_seeds: Seeds of the GT episodes.
        dry: Dry run (never complete).

    Returns:
        Per stage status, the GT status and `all_ok`.
    """
    st: dict[str, Any] = {}
    for name in ("dev_conf", "disc", "placebo"):
        c = checks.get(name)
        want = stage_seeds(name)
        st[name] = {
            "ok": bool(c and c["seeds"] == want and not c["smoke"] and c["ok"]),
            "n": len(c["seeds"]) if c else 0,
            "expected": [want[0], want[-1]],
            "smoke": bool(c and c["smoke"]),
        }
    want = stage_seeds("gt")
    st["gt"] = {
        "ok": sorted(int(s) for s in gt_seeds) == want,
        "n": len(gt_seeds),
        "expected": [want[0], want[-1]],
    }
    st["all_ok"] = all(v["ok"] for v in st.values() if isinstance(v, dict)) and not dry
    return st


def disc_mode(rep: Mapping[str, Any], dry: bool) -> str:
    """The discovery analysis mode from its data, artifact, protocol, platform and maps status."""
    full = bool(
        rep["data"]["all_ok"]
        and rep["K0n"] is not None
        and rep["artifact"]["sha_ok"]
        and rep["artifact"]["code_ok"]
        and rep["protocol"]["frozen"]
        and rep["platform"]["equal"]
        and rep["platform"]["linux"]
        and not rep["missing_arms"]
    )
    return "full" if full else ("dry-run" if dry else "partial")


def d1_check(
    point: Mapping[str, Mapping[str, Any]],
    boot: Mapping[str, Mapping[str, Any]],
    primary: str,
    assoc: Sequence[str],
    margin: float = D1_MARGIN,
) -> Check:
    """D1: Delta1 >= margin and LB90 > 0, the best associational map re-selected per resample."""
    best = max(assoc, key=lambda b: point[b]["Rstar"])
    d1 = float(point[primary]["Rstar"] - point[best]["Rstar"])
    stack = np.stack([boot[b]["Rstar"] for b in assoc])
    diff = boot[primary]["Rstar"] - np.max(stack, 0)
    fin = diff[np.isfinite(diff)]
    lb = float(np.quantile(fin, 0.05)) if fin.size else float("nan")
    sel = np.argmax(stack, 0)
    return {
        "pass": bool(np.isfinite(d1) and d1 >= margin - 1e-12 and np.isfinite(lb) and lb > 0),
        "delta1": d1,
        "best_point": best,
        "lb90": lb,
        "ci90": ci(diff),
        "n_nonfinite": int(diff.size - fin.size),
        "best_selected_share": {b: float(np.mean(sel == i)) for i, b in enumerate(assoc)},
        "margin": margin,
    }


def holm(pvals: Mapping[str, float], alpha: float = ALPHA_D2) -> dict[str, bool]:
    """Holm step-down: `{name: rejected}`."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(order)
    rej, stop = {}, False
    for i, k in enumerate(order):
        stop = stop or not pvals[k] <= alpha / (m - i) + 1e-15
        rej[k] = not stop
    return rej


def d2_check(
    boot: Mapping[str, Mapping[str, Any]],
    primary: str,
    assoc: Sequence[str],
    signature: Mapping[str, Any],
    alpha: float = ALPHA_D2,
) -> Check:
    """D2: one hypothesis per distinct associational signature, Holm at alpha.

    Passes iff every hypothesis is rejected.

    The one-sided paired bootstrap p is (1 + #{R*_boot(primary) - R*_boot(b) <= 0}) / (N + 1);
    NaN differences count against the primary.
    """
    groups: dict[Any, list[str]] = {}
    for b in assoc:
        groups.setdefault(signature[b], []).append(b)
    pv, rows = {}, {}
    for sig, bs in groups.items():
        b = bs[0]
        diff = boot[primary]["Rstar"] - boot[b]["Rstar"]
        p = float((1 + np.count_nonzero(~(diff > 0))) / (len(diff) + 1))
        pv[b] = p
        rows[b] = {
            "members": bs,
            "p": p,
            "same_signature_as_primary": sig == signature.get(primary),
        }
    rej = holm(pv, alpha)
    for b, row in rows.items():
        row["rejected"] = rej[b]
    return {
        "pass": bool(rows) and all(rej.values()),
        "n_hypotheses": len(rows),
        "hypotheses": rows,
        "alpha": alpha,
        "procedure": "Holm over the distinct signatures of A, one-sided paired bootstrap",
    }


def eval_verdict(
    *, k0: bool, k0n: bool, e: bool, d1: bool, d2: bool, why: Sequence[str] = ()
) -> dict[str, Any]:
    """Section 7.2 verdict; complete iff `why` (the incompleteness reasons) is empty."""
    if not (k0 and k0n):
        v = "INVALID"
    elif not e:
        v = "NOT ELIGIBLE"
    elif d1 and d2:
        v = "PASS"
    elif d1 or d2:
        v = "PARTIAL"
    else:
        v = "FAIL"
    complete = not why
    return {
        "verdict": v,
        "label": v if complete else f"NOT A VERDICT (would-be: {v}; {'; '.join(why)})",
        "complete": complete,
        "precedence": "INVALID (K0 or K0n) > NOT ELIGIBLE (E) > PASS (D1 and D2) / PARTIAL (one) "
        "/ FAIL",
    }


def arm_rows(
    R: Records, st: GateA, groups: Sequence[str] = COUNT_GROUPS, cpu: bool = True
) -> dict[str, dict[str, Any]]:
    """Per arm: point statistics, 90 % CIs, policy counts and (optionally) mean CPU seconds."""
    out = {}
    for x, p in st.point.items():
        b = st.boot[x]
        row = dict(
            p,
            R_ci90=ci(b["R"]),
            Rstar_ci90=ci(b["Rstar"]),
            retention_ci90=ci(b["retention"]),
            guard_ratio_ci90={k: ci(v) for k, v in b["guard"].items()},
            counts={g: sum_counts(R, st.seeds, x, g) for g in groups},
        )
        if cpu:
            row["cpu_s_mean"] = float(np.mean([R[s][x].get("cpu_s", np.nan) for s in st.seeds]))
        row["alias_of"] = R[st.seeds[0]][x].get("alias_of")
        out[x] = row
    return out


def eligibility(rows: Mapping[str, Mapping[str, Any]], primary: str) -> Check:
    """E: the primary arm is eligible at its point estimate."""
    return {
        "pass": bool(rows[primary]["eligible"]) if primary in rows else False,
        "retention_ci90": rows.get(primary, {}).get("retention_ci90"),
        "guard_ratio_ci90": rows.get(primary, {}).get("guard_ratio_ci90"),
    }


def secondary(st: GateA, primary: str, arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    """R(primary) - R(c) with its paired 90 % CI for each present contrast arm `c`."""
    P, B = st.point, st.boot
    return {
        c: {"dR": P[primary]["R"] - P[c]["R"], "ci90": ci(B[primary]["R"] - B[c]["R"])}
        for c in arms
        if c in P and primary in P
    }


def cpu_hours(R: Records) -> float:
    """CPU hours of every simulated (non-aliased) job."""
    return float(
        sum(
            r.get("cpu_s", 0.0)
            for arms in R.values()
            for r in arms.values()
            if not r.get("alias_of")
        )
        / 3600
    )


@dataclasses.dataclass(frozen=True)
class EvalInputs:
    """Frozen inputs of the Study 2 evaluation besides the EVAL records.

    Attributes:
        disc: The discovery analysis (`disc_conf.json`).
        maps: The parsed maps artifact (`E6P_CONF_MAPS.json`).
        maps_verify: The artifact verification report (`ok`, `driver_matches`, ...).
        frozen: Whether the protocol is frozen.
        argv: Command line recorded in the report.
    """

    disc: Mapping[str, Any]
    maps: Mapping[str, Any]
    maps_verify: Mapping[str, Any]
    frozen: bool
    argv: Sequence[str] = ()


def _never_sleep(sec: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    ns = sec.get("never_sleep")
    if not ns:
        return None
    lb = ns["ci90"][0]
    return dict(
        ns,
        noninferior_lb90_gt=NONINF,
        noninferior=bool(lb is not None and lb > NONINF),
        statement=NONINF_STATEMENT,
    )


def _efficiency(st: GateA) -> dict[str, Any] | None:
    P, B = st.point, st.boot
    if PRIMARY_ARM not in P or "MG:GT" not in P:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "ratio": P[PRIMARY_ARM]["R"] / P["MG:GT"]["R"] if P["MG:GT"]["R"] else float("nan"),
            "ci90": ci(B[PRIMARY_ARM]["R"] / B["MG:GT"]["R"]),
        }


def _incomplete(
    R: Records,
    heads: Sequence[Mapping[str, Any]],
    st: GateA,
    missing: Sequence[str],
    inp: EvalInputs,
) -> list[str]:
    disc, mv = inp.disc, inp.maps_verify
    why = []
    if disc.get("mode") != "full":
        why.append(f"disc analysis mode {disc.get('mode')}")
    if not disc.get("data", {}).get("all_ok"):
        why.append("discovery data incomplete")
    if st.seeds != stage_seeds("eval"):
        why.append(f"eval seeds with every arm: {st.n} / {len(stage_seeds('eval'))}")
    if missing:
        why.append(f"missing arms {list(missing)}")
    if any(r.get("smoke") for arms in R.values() for r in arms.values()):
        why.append("smoke records")
    if not mv["ok"] or not mv["driver_matches"]:
        why.append("maps artifact sha / checks (MAPS_SHA256)")
    if not inp.frozen:
        why.append("protocol not frozen (FROZEN_SHA256_CONF)")
    if not (disc["artifact"]["sha_ok"] and disc["artifact"]["code_ok"]):
        why.append("v4 artifact / code sha256")
    plats = {platform_key(h.get("platform")) for h in heads if not h.get("smoke")}
    dplat = {
        tuple(tuple(y) if isinstance(y, list) else y for y in x)
        for x in disc.get("platform", {}).get("keys", [])
    }
    if len(plats | dplat) != 1:
        why.append(f"platform fingerprints differ across stages: {sorted(map(str, plats | dplat))}")
    mg = {h.get("mapgate_sha256") for h in heads}
    if mg and mg != {inp.maps["mapgate"]["sha256"]}:
        why.append("mapgate.py sha256 of the eval runs differs from the maps artifact")
    return why


def evaluate(
    R: Records, heads: Sequence[Mapping[str, Any]], inp: EvalInputs, n_boot: int = N_BOOT
) -> dict[str, Any]:
    """The Study 2 evaluation report (`eval_conf.json` without `timing_s`).

    Args:
        R: EVAL job records `{seed: {arm: record}}` with aliased arms filled (`stats.alias_fill`).
        heads: EVAL run headers.
        inp: Discovery report, maps artifact and status inputs.
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
    rows = arm_rows(R, st)
    P, B = st.point, st.boot
    A = [x for x in ASSOC_ARMS if x in P]
    sig = {arm: v["signature"] for arm, v in inp.maps["arms"].items()}
    have = PRIMARY_ARM in P and bool(A)
    E = eligibility(rows, PRIMARY_ARM)
    D1 = d1_check(P, B, PRIMARY_ARM, A) if have else {"pass": False, "note": "arms missing"}
    D2 = d2_check(B, PRIMARY_ARM, A, sig) if have else {"pass": False, "note": "arms missing"}
    sec = secondary(st, PRIMARY_ARM, SECONDARY)
    k0 = bool(inp.disc["K0"]["pass"])
    k0n = bool((inp.disc.get("K0n") or {}).get("pass"))
    why = _incomplete(R, heads, st, missing, inp)
    return {
        "schema": SCHEMA_EVAL_REPORT,
        "argv": list(inp.argv),
        "maps_verify": inp.maps_verify,
        "n_seeds_any": len(R),
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
        "never_sleep": _never_sleep(sec),
        "secondary": sec,
        "map_efficiency": _efficiency(st),
        "verdict": eval_verdict(k0=k0, k0n=k0n, e=E["pass"], d1=D1["pass"], d2=D2["pass"], why=why),
        "discovery_label": inp.disc["label"]["label"],
        "cpu_h_eval": cpu_hours(R),
    }


def verdict_record(rep: Mapping[str, Any]) -> dict[str, Any]:
    """The `verdict_conf.json` subset of an evaluation report."""
    return {k: rep[k] for k in VERDICT_KEYS}


def recompute_disc(rep: Mapping[str, Any]) -> dict[str, Any]:
    """A stored discovery analysis with its criteria, label, data status and mode recomputed.

    K0n, whose raw p-values are not stored, is kept.
    """
    out = dict(rep)
    ref = study1.gt_ref(rep["gt"]["cells"])
    out["G"] = study1.g_check(rep["gt"]["cells"])
    out["K0"] = study1.k0_check(study1.hyp_decl(rep["placebo_hyp"]))
    out["K1"] = k1_check(rep["units"]["disc"])
    out["X"] = x_checks(study1.hyp_decl(rep["disc_hyp"]), ref)
    out["label"] = disc_label(out["K0"], rep["K0n"], out["G"], out["K1"], out["X"])
    out["data"] = data_status(rep["record_checks"], rep["gt"]["seeds"], dry=False)
    out["mode"] = disc_mode(out, dry=False)
    return out


TESTS_ONLY = ("placebo_decl", "disc_decl", "K0n_raw")


def disc_from_tests(tests: Mapping[str, Any]) -> dict[str, Any]:
    """The discovery analysis (`disc_conf.json`) of the discovery tests (`tests_conf.json`).

    `placebo_combos` is K0 of every combination's placebo declarations, `K0n` the primary
    combination's null-outcome check and `K0n_descriptive` every combination's (with the number
    of episode groups); the schema becomes `SCHEMA_DISC`; then `recompute_disc`.
    """
    blocks = study1.placebo_null_blocks(dict(tests) | {"placebo_baselines": {}})
    rep = {k: v for k, v in tests.items() if k not in TESTS_ONLY}
    rep["placebo_combos"] = blocks["placebo_descriptive"]
    if "K0n_descriptive" in blocks:
        rep["K0n_descriptive"] = blocks["K0n_descriptive"] | {"groups": tests["K0n_raw"]["groups"]}
    rep["K0n"] = blocks["K0n"]
    rep["schema"] = SCHEMA_DISC
    return recompute_disc(rep)

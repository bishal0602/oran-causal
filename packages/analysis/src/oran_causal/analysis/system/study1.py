"""Study 1 criteria and verdict: frozen PMRT on fresh randomised slices (discovery protocol v4).

The criteria read the test outputs of the discovery stage (per-slice declarations and their
`subset_metrics`, the placebo declarations, the null-outcome variants) and the knockout ground
truth. Only the primary combination (loadsp_c + wby1s) decides; C is the chain set and C* its
members that are TRUE with the expected sign in the ground truth.

- K0: placebo, P(Binom(m, .05) >= n_reject) >= .01 over the m finite p-values the layer uses, and
  at most one declaration.
- K0n: null outcomes, pooled rate(p <= .05) <= .075, every family's rate <= .10, and at most 5
  variants with a declaration.
- K1: every 120-slice has >= 60 tested sleep units with >= 15 logged rejects.
- G: the ground truth labels sleep -> nbr pv TRUE with sign +1.
- P1: in >= 3 of the 5 120-slices the premise is declared + and >= min(3, |C*|) members of C*
  are declared with the GT sign.
- P2: PMRT's indirect F1 is >= each baseline's in >= 6 of 10 60-slices and >= 3 of 5 120-slices;
  ties count for PMRT, an undefined PMRT F1 is a loss and an undefined baseline F1 counts as 0.
- P3: pooled premise declared +, >= min(3, |C*|) chain hits, precision >= .80, sign accuracy >= .90.
- S: pooled sign accuracy >= .90 and mean sign accuracy over the defined 120-slices >= .90.

Verdict precedence: INVALID (K0 or K0n) > NO-CHAIN (G) > UNDERPOWERED (K1) > PASS (P1, P2, P3, S)
/ PARTIAL (P3 and S) / KILL.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import binom

from oran_causal.core.scoring import score_method
from oran_causal.sim.system.decision.vocab import HYPOTHESES

Decl = Mapping[Any, Mapping[str, Any]]
Ref = Mapping[Any, Mapping[str, Any]]
Check = dict[str, Any]

PRIMARY = ("loadsp_c", "wby1s")
PRIMARY_NAME = "+".join(PRIMARY)
SUB = "v4"
SEEDS = {"dev": (188000, 20), "eval": (188100, 600), "placebo": (188700, 40), "gt": (188800, 40)}
N60, N120, N300 = 60, 120, 300
SPLIT = {"pooled": 0, "60": 20, "120": 40, "300": 60, "placebo": 9, "null": 0}
H, H_PRE = 90, 90
BASELINES = ("corr", "granger", "granger_by")
DESC_METHODS = ("shap_gbdt", "int", "qacm", "two_tower")
K0_RULE = {"alpha": 0.05, "binom_level": 0.01, "max_decl": 1}
K0N_RULE = {"rate": 0.075, "family_rate": 0.10, "max_var_decl": 5, "alpha": 0.05}
K1_RULE = {"min_sleep_units": 60, "min_sleep_rejects": 15}
P1_RULE = {"min_slices": 3, "chain_min": 3}
P2_RULE = {"min_60": 6, "min_120": 3}
P3_RULE = {"chain_min": 3, "precision": 0.80, "sign_acc": 0.90}
S_RULE = {"sign_acc": 0.90}
CHAIN = (
    ("sleep", "nbr", "load", 1),
    ("sleep", "nbr", "pv", 1),
    ("sleep", "nbr", "v", 1),
    ("ptx", "nbr", "load", -1),
)
PREMISE = ("sleep", "nbr", "pv")
PRECEDENCE = (
    "INVALID (K0 or K0n) > NO-CHAIN (G) > UNDERPOWERED (K1) > PASS (P1 P2 P3 S) / PARTIAL "
    "(P3 S) / KILL"
)
VERDICT_KEYS = ("mode", "verdict", "K0", "K0n", "G", "K1", "P1", "P2", "P3", "S", "protocol")
VERDICT_KEYS_TAIL = ("artifact", "data")


def _finite(v: Any) -> bool:
    return v is not None and math.isfinite(float(v))


def _ge(v: Any, t: float) -> bool:
    return bool(_finite(v) and v >= t - 1e-12)


def gt_ref(cells: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Reference `{(family, relation, kpi): {status, sign}}` from GT table cells."""
    return {
        (c["family"], c["relation"], c["kpi"]): {"status": c["status"], "sign": int(c["sign"])}
        for c in cells
    }


def subset_metrics(decl: Decl, ref: Ref) -> dict[str, Any]:
    """Chain hits, premise hit, indirect and overall scores of one declaration set.

    Args:
        decl: Per hypothesis `{declared, sign, p?, z_approx?}`.
        ref: GT reference from `gt_ref`.

    Returns:
        chain_hits, chain_true, premise_hit, the indirect (nbr) and overall scores, sign
        accuracy, declaration counts and per chain member `{gt_true, hit, p, z}`.
    """
    sc = score_method(decl, ref, HYPOTHESES)
    c_true = hits = 0
    members = {}
    for f, r, k, s in CHAIN:
        g = ref.get((f, r, k), {"status": "INDET", "sign": 0})
        d = decl.get((f, r, k)) or {}
        ok_gt = g["status"] == "TRUE" and int(g["sign"]) == s
        hit = bool(d.get("declared")) and int(d.get("sign", 0)) == s
        c_true += ok_gt
        hits += ok_gt and hit
        members["|".join((f, r, k))] = {
            "gt_true": ok_gt,
            "hit": hit,
            "p": d.get("p"),
            "z": d.get("z_approx"),
        }
    prem = decl.get(PREMISE) or {}
    ind, ov = sc["indirect"], sc["overall"]
    return {
        "chain_hits": hits,
        "chain_true": c_true,
        "premise_hit": float(bool(prem.get("declared")) and int(prem.get("sign", 0)) == 1),
        "ind_recall": ind["recall"],
        "ind_precision": ind["precision"],
        "ind_f1": ind["f1"],
        "ind_tp": ind["tp"],
        "ind_fp": ind["fp"],
        "ov_precision": ov["precision"],
        "ov_recall": ov["recall"],
        "ov_f1": ov["f1"],
        "sign_acc": ov["sign_acc"],
        "n_declared": sc["n_declared"],
        "far_declared": sc["far_declared"],
        "chain": members,
    }


def k0_check(decl: Decl, rule: Mapping[str, Any] = K0_RULE) -> Check:
    """K0 on the placebo declarations of the primary combination."""
    ps = [
        float(v["p"])
        for v in decl.values()
        if v.get("status") != "undetermined" and np.isfinite(v.get("p", np.nan))
    ]
    m, k = len(ps), int(sum(p <= rule["alpha"] for p in ps))
    tail = float(binom.sf(k - 1, m, rule["alpha"])) if m else 1.0
    nd = int(sum(bool(v.get("declared")) for v in decl.values()))
    return {
        "pass": bool(m > 0 and tail >= rule["binom_level"] and nd <= rule["max_decl"]),
        "m": m,
        "n_reject": k,
        "rate": k / m if m else float("nan"),
        "binom_tail": tail,
        "n_declared": nd,
        "declared": sorted("|".join(h) for h, v in decl.items() if v.get("declared")),
        "rule": dict(rule),
        "min_p": min(ps) if ps else float("nan"),
    }


def k0n_check(
    pvals: Sequence[float],
    fams: Sequence[str],
    var_decl: Sequence[int],
    rule: Mapping[str, Any] = K0N_RULE,
) -> Check:
    """K0n over every used p-value of every null-outcome variant.

    Args:
        pvals: Used p-values, pooled over the variants.
        fams: Family of each p-value.
        var_decl: Number of declarations per variant.
        rule: Thresholds.

    Returns:
        The check with the pooled and per-family rates.
    """
    pv, fam = np.asarray(pvals, float), np.asarray(fams, object)
    rate = float(np.mean(pv <= rule["alpha"])) if len(pv) else float("nan")
    fr = {f: float(np.mean(pv[fam == f] <= rule["alpha"])) for f in sorted(set(fam.tolist()))}
    nvd = int(sum(int(d) > 0 for d in var_decl))
    ok = bool(
        len(pv)
        and len(var_decl)
        and rate <= rule["rate"] + 1e-12
        and all(v <= rule["family_rate"] + 1e-12 for v in fr.values())
        and nvd <= rule["max_var_decl"]
    )
    return {
        "pass": ok,
        "n_variants": len(var_decl),
        "n_p": len(pv),
        "rate05": rate,
        "rate05_family": fr,
        "n_variants_with_decl": nvd,
        "n_decl": int(sum(var_decl)),
        "rule": dict(rule),
    }


def k1_check(
    per_slice: Mapping[str, Mapping[str, int]], rule: Mapping[str, Any] = K1_RULE
) -> Check:
    """K1 over the 120-slices, `{name: {sleep_units, sleep_rejects}}`."""
    rows = {
        k: dict(
            v,
            ok=bool(
                v["sleep_units"] >= rule["min_sleep_units"]
                and v["sleep_rejects"] >= rule["min_sleep_rejects"]
            ),
        )
        for k, v in per_slice.items()
    }
    return {
        "pass": bool(rows) and all(v["ok"] for v in rows.values()),
        "slices": rows,
        "rule": dict(rule),
    }


def g_check(cells: Sequence[Mapping[str, Any]]) -> Check:
    """G on the GT table cells."""
    c = next((c for c in cells if (c["family"], c["relation"], c["kpi"]) == PREMISE), None)
    if c is None:
        return {"pass": False, "cell": None, "rule": "gt_v4 sleep -> nbr pv TRUE(+)"}
    return {
        "pass": bool(c["status"] == "TRUE" and int(c["sign"]) == 1),
        "cell": c,
        "rule": "gt_v4 sleep -> nbr pv TRUE(+) (dir, frozen gt_p rule)",
    }


def p1_check(m120: Sequence[Mapping[str, Any]], rule: Mapping[str, Any] = P1_RULE) -> Check:
    """P1 on the primary combination's `subset_metrics` of each 120-slice (with `name`)."""
    per = []
    for m in m120:
        need = min(rule["chain_min"], int(m["chain_true"]))
        per.append(
            {
                "name": m.get("name"),
                "premise": bool(m["premise_hit"] == 1.0),
                "chain_hits": int(m["chain_hits"]),
                "chain_true": int(m["chain_true"]),
                "need": need,
                "ok": bool(m["premise_hit"] == 1.0 and m["chain_hits"] >= need),
            }
        )
    n_ok = sum(1 for p in per if p["ok"])
    return {
        "pass": bool(n_ok >= rule["min_slices"]),
        "n_ok": n_ok,
        "n_slices": len(per),
        "slices": per,
        "rule": dict(rule),
    }


def _win(m: Any, b: Any) -> bool:
    m = float(m) if m is not None else float("nan")
    b = float(b) if b is not None else float("nan")
    b = 0.0 if not np.isfinite(b) else b
    return bool(np.isfinite(m) and m >= b - 1e-12)


def p2_check(
    f1_60: Sequence[float],
    f1_120: Sequence[float],
    base60: Mapping[str, Sequence[float]],
    base120: Mapping[str, Sequence[float]],
    rule: Mapping[str, Any] = P2_RULE,
) -> Check:
    """P2 from PMRT's indirect F1 per slice and each baseline's, in slice order."""
    per = {}
    for b in base60:
        w60 = [_win(m, x) for m, x in zip(f1_60, base60[b], strict=True)]
        w120 = [_win(m, x) for m, x in zip(f1_120, base120[b], strict=True)]
        per[b] = {
            "wins60": int(sum(w60)),
            "n60": len(w60),
            "wins120": int(sum(w120)),
            "n120": len(w120),
            "win60": w60,
            "win120": w120,
            "ok": bool(sum(w60) >= rule["min_60"] and sum(w120) >= rule["min_120"]),
        }
    return {
        "pass": bool(per) and all(v["ok"] for v in per.values()),
        "per_baseline": per,
        "rule": dict(rule),
        "ties": "count for PMRT; undefined PMRT F1 = loss; undefined baseline F1 = 0",
    }


def p3_check(decl: Decl, ref: Ref, rule: Mapping[str, Any] = P3_RULE) -> Check:
    """P3 on the pooled declarations of the primary combination."""
    m = subset_metrics(decl, ref)
    sc = score_method(decl, ref, HYPOTHESES)
    need = min(rule["chain_min"], int(m["chain_true"]))
    parts = {
        "premise_declared_plus": bool(m["premise_hit"] == 1.0),
        "chain_hits": bool(m["chain_hits"] >= need),
        "overall_precision": _ge(sc["overall"]["precision"], rule["precision"]),
        "sign_accuracy": _ge(sc["overall"]["sign_acc"], rule["sign_acc"]),
    }
    return {
        "pass": all(parts.values()),
        "parts": parts,
        "rule": dict(rule),
        "values": {
            "chain_true": int(m["chain_true"]),
            "chain_need": need,
            "chain_hits": int(m["chain_hits"]),
            "overall_precision": sc["overall"]["precision"],
            "sign_accuracy": sc["overall"]["sign_acc"],
            "far_declared": sc["far_declared"],
            "ind_recall": sc["indirect"]["recall"],
        },
        "chain": m["chain"],
    }


def s_check(
    pooled_sign: float | None, signs120: Sequence[Any], rule: Mapping[str, Any] = S_RULE
) -> Check:
    """S from the pooled sign accuracy and the 120-slices' (undefined ones skipped)."""
    f = [float(s) for s in signs120 if _finite(s)]
    mean = float(np.mean(f)) if f else float("nan")
    ok_p = _ge(pooled_sign, rule["sign_acc"])
    ok_s = bool(f and mean >= rule["sign_acc"] - 1e-12)
    return {
        "pass": ok_p and ok_s,
        "pooled": pooled_sign,
        "mean120": mean,
        "n_defined120": len(f),
        "pooled_ok": ok_p,
        "mean120_ok": ok_s,
        "rule": dict(rule),
    }


def _passes(c: Mapping[str, Any] | None) -> bool:
    return bool(c and c["pass"])


def verdict(
    k0: Check | None,
    k0n: Check | None,
    g: Check | None,
    k1: Check | None,
    p1: Check,
    p2: Check,
    p3: Check,
    s: Check,
    *,
    mode: str,
    frozen: bool,
) -> dict[str, str]:
    """The verdict by precedence and its label.

    The label is "NOT A VERDICT" unless `mode` is "full", and marked provisional unless the
    protocol is frozen.
    """
    if not (_passes(k0) and _passes(k0n)):
        v = "INVALID"
    elif not _passes(g):
        v = "NO-CHAIN"
    elif not _passes(k1):
        v = "UNDERPOWERED"
    elif p1["pass"] and p2["pass"] and p3["pass"] and s["pass"]:
        v = "PASS"
    elif p3["pass"] and s["pass"]:
        v = "PARTIAL"
    else:
        v = "KILL"
    if mode != "full":
        label = f"NOT A VERDICT ({mode}; would-be: {v})"
    elif not frozen:
        label = f"{v} (PROVISIONAL: protocol not frozen)"
    else:
        label = v
    return {"verdict": v, "label": label, "precedence": PRECEDENCE}


def stage_status(
    name: str, seeds: Sequence[int] | None, subs: Sequence[str] = (), smoke: bool = False
) -> dict[str, Any]:
    """Completeness of one v4 record stage against its seed range; None seeds means missing."""
    base, n = SEEDS[name]
    if seeds is None:
        return {"ok": False, "n": 0, "note": "missing"}
    seeds = sorted(int(s) for s in seeds)
    subs = sorted(set(subs))
    ok = seeds == list(range(base, base + n)) and subs == [SUB] and not smoke
    return {
        "ok": bool(ok),
        "n": len(seeds),
        "subs": subs,
        "smoke": smoke,
        "seed_range": [seeds[0], seeds[-1]] if seeds else None,
        "expected": [base, base + n - 1],
    }


def data_status(stages: Mapping[str, dict[str, Any]], gt_seeds: Sequence[int], dry: bool) -> dict:
    """Completeness of the v4 data; mode "full" needs `all_ok`.

    Args:
        stages: `stage_status` of eval, placebo and dev, in that order.
        gt_seeds: Seeds of the GT episodes.
        dry: Dry run (never complete).

    Returns:
        Per stage status, the GT status and `all_ok`.
    """
    st: dict[str, Any] = dict(stages)
    base, n = SEEDS["gt"]
    st["gt"] = {
        "ok": sorted(gt_seeds) == list(range(base, base + n)),
        "n": len(gt_seeds),
        "expected": [base, base + n - 1],
    }
    st["all_ok"] = all(v["ok"] for v in st.values() if isinstance(v, dict)) and not dry
    return st


def mode_of(
    data: Mapping[str, Any],
    k0n: Check | None,
    artifact: Mapping[str, Any],
    complete: bool,
    dry: bool,
) -> str:
    """The analysis mode: "full" when the data, K0n, the artifact and every slice are complete."""
    if (
        data["all_ok"]
        and k0n is not None
        and artifact["sha_ok"]
        and artifact["code_ok"]
        and complete
    ):
        return "full"
    return "dry-run" if dry else "partial"


def criteria(
    slices: Mapping[str, Mapping[str, Any]], pooled_decl: Decl, ref: Ref
) -> dict[str, Check]:
    """K1, P1, P2, P3 and S from the per-slice results.

    Args:
        slices: Per slice name `{kind, index, units: {sleep_units, sleep_rejects}, metrics:
            {method: subset_metrics}}`, in slice order.
        pooled_decl: Pooled declarations of the primary combination.
        ref: GT reference.

    Returns:
        `{K1, P1, P2, P3, S}`.
    """
    s60 = [r for r in slices.values() if r["kind"] == "60"]
    s120 = [r for r in slices.values() if r["kind"] == "120"]
    k1 = k1_check(
        {
            f"s120_{r['index']}": {
                "sleep_units": r["units"]["sleep_units"],
                "sleep_rejects": r["units"]["sleep_rejects"],
            }
            for r in s120
        }
    )
    p1 = p1_check([dict(r["metrics"][PRIMARY_NAME], name=f"s120_{r['index']}") for r in s120])

    def f1(rows: Sequence[Mapping[str, Any]], m: str) -> list[float]:
        return [r["metrics"][m]["ind_f1"] for r in rows]

    p2 = p2_check(
        f1(s60, PRIMARY_NAME),
        f1(s120, PRIMARY_NAME),
        {b: f1(s60, b) for b in BASELINES},
        {b: f1(s120, b) for b in BASELINES},
    )
    p3 = p3_check(pooled_decl, ref)
    s = s_check(
        slices["pooled"]["metrics"][PRIMARY_NAME]["sign_acc"],
        [r["metrics"][PRIMARY_NAME]["sign_acc"] for r in s120],
    )
    return {"K1": k1, "P1": p1, "P2": p2, "P3": p3, "S": s}


def verdict_record(rep: Mapping[str, Any]) -> dict[str, Any]:
    """The `verdict_v4.json` subset of an analysis report."""
    return {k: rep[k] for k in (*VERDICT_KEYS, *VERDICT_KEYS_TAIL)}


def hyp_decl(hyp: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Primary declarations from a stored per-hypothesis table (`placebo_hyp`, `hyp`).

    Each entry is the table's `primary` block plus `z_approx`, the primary statistic's z, for
    tested hypotheses.
    """
    out = {}
    for h, row in hyp.items():
        d = dict(row["primary"])
        if row.get("status") == "tested":
            d["z_approx"] = row[PRIMARY[0]]["z"]
        out[tuple(h.split("|"))] = d
    return out


def recompute(rep: Mapping[str, Any]) -> dict[str, Any]:
    """A stored Study 1 analysis with every criterion, the mode and the verdict recomputed.

    K0 reads the placebo table, G the GT cells, K1 to S the per-slice results and the pooled
    table; K0n, whose raw p-values are not stored, is kept.
    """
    out = dict(rep)
    out["G"] = g_check(rep["gt"]["cells"])
    out["K0"] = k0_check(hyp_decl(rep["placebo_hyp"]))
    pooled_decl = hyp_decl(rep["slices_result"]["pooled"]["hyp"])
    c = criteria(rep["slices_result"], pooled_decl, gt_ref(rep["gt"]["cells"]))
    complete = all(s["complete"] for s in rep["slices"])
    out["mode"] = mode_of(rep["data"], rep["K0n"], rep["artifact"], complete, dry=False)
    out |= c
    out["verdict"] = verdict(
        out["K0"],
        rep["K0n"],
        out["G"],
        c["K1"],
        c["P1"],
        c["P2"],
        c["P3"],
        c["S"],
        mode=out["mode"],
        frozen=rep["protocol"]["frozen"],
    )
    return {k: out[k] for k in rep}


ANALYSIS_KEYS = (
    "schema",
    "protocol",
    "primary",
    "artifact",
    "gt",
    "G",
    "data",
    "slices",
    "baseline_tau",
    "desc_tau",
    "desc_errors",
    "K0",
    "placebo_descriptive",
    "placebo_units",
    "placebo_hyp",
    "slices_result",
    "K0n",
    "K0n_descriptive",
    "mode",
    "K1",
    "P1",
    "P2",
    "P3",
    "S",
    "verdict",
    "timing_s",
)


def placebo_null_blocks(tests: Mapping[str, Any]) -> dict[str, Any]:
    """The placebo and null-outcome blocks of an analysis from the tests (`tests_v4.json`).

    `placebo_descriptive` is K0 of every combination's placebo declarations plus the baselines'
    placebo checks; `K0n` is the primary combination's check over the null-outcome variants and
    `K0n_descriptive` every combination's. Without null-outcome tests K0n is None.
    """
    plc = {
        c: k0_check({tuple(h.split("|")): v for h, v in d.items()})
        for c, d in tests["placebo_decl"].items()
    }
    out: dict[str, Any] = {"placebo_descriptive": plc | dict(tests["placebo_baselines"])}
    raw = tests.get("K0n_raw")
    if raw is None:
        return out | {"K0n": None}
    combos = {c: k0n_check(o["p"], o["fam"], o["var_decl"]) for c, o in raw["combos"].items()}
    return out | {
        "K0n": combos[tests["primary"]],
        "K0n_descriptive": {"variants": raw["variants"], "combos": combos},
    }


def from_tests(tests: Mapping[str, Any]) -> dict[str, Any]:
    """The Study 1 analysis (`analysis_v4.json`) of the discovery tests (`tests_v4.json`)."""
    rep = dict(tests) | placebo_null_blocks(tests)
    return recompute({k: rep.get(k) for k in ANALYSIS_KEYS})

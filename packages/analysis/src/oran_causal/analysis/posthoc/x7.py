"""X7: are the associational referees the static subset rule `sub:ES+PowerES`?

Post hoc and descriptive, on the Study 3 EVAL seeds 191100-191259; nothing frozen is edited. Job
kinds (record arm names, sub `x7`):

* `X7:sub|shadow`: `sub:ES+PowerES` applied, the four referees queried in shadow on a copy of every
  observation.
* `X7:<ref>|own`: the referee `<ref>` applied (the frozen arm, as stored), the subset rule in
  shadow.
* `X7:<ref>|pm`: `<ref>`'s decisions with every non-prot_min deferral replaced by accept.
* `X7:<ref>|other`: `<ref>`'s decisions with every prot_min deferral replaced by accept.

Per-request tallies: `record["agree"][shadow]["<family>|<dir>|<pre|post>"] = [aa, ar, ra, rr]`
(first letter the applied decision, second the shadow decision; a = accept, r = reject).

The run half is `oran_causal.experiments.posthoc.x7`; this module scores its records together
with the stored EVAL records.
"""

import argparse
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc import x4
from oran_causal.analysis.posthoc.common import (
    alias_filled,
    ci_close,
    js_default,
    read_json,
    shard_files,
    sub_filter,
)
from oran_causal.analysis.posthoc.paths import ARTIFACTS, RECORDS, write_text
from oran_causal.analysis.system import stats, study3

SUB = "x7"
REF = "sub:ES+PowerES"
REFS = ("CS:granger@dev", "CS:granger_by", "CS:two_tower@dev", "CS:shap_gbdt@dev")
SHORT = {
    "CS:granger@dev": "granger",
    "CS:granger_by": "granger_by",
    "CS:two_tower@dev": "two_tower",
    "CS:shap_gbdt@dev": "shap",
}
KINDS = ("shadow", "own", "pm", "other")
FAMS = ("prot_min", "ptx", "carrier", "sleep")
SHADOW_ARM = "X7:sub|shadow"
AGREE_BOOT, AGREE_KEY = 2000, (20261005, 7)
SKIP = x4.SKIP | {"agree", "x7"}
DRIVER = "x7_subset_check"
DECLARED = "scratchpad/xmethod/EXTRAS_PROTOCOL.md Amendments 2026-10-05 X7"
RTOL = 1e-9
PerSeed = np.ndarray
Selector = Callable[[str], bool]


def arm_name(ref: str | None, kind: str) -> str:
    """Record arm name of a job kind."""
    return SHADOW_ARM if kind == "shadow" else f"X7:{ref}|{kind}"


def job_list(
    seeds: Sequence[int], kinds: Sequence[str] = KINDS
) -> list[tuple[int, str, str | None]]:
    """`(seed, kind, ref)` seed-major: per seed the shadow job, then per referee own, pm, other."""
    out: list[tuple[int, str, str | None]] = []
    for s in seeds:
        if "shadow" in kinds:
            out.append((s, "shadow", None))
        out += [(s, k, r) for r in REFS for k in ("own", "pm", "other") if k in kinds]
    return out


def _agree_sum(recs: Sequence[Mapping[str, Any]], shadow: str, sel: Selector) -> np.ndarray:
    per = np.zeros((len(recs), 4))
    for j, r in enumerate(recs):
        for key, c in (r.get("agree") or {}).get(shadow, {}).items():
            if sel(key):
                per[j] += c
    return per


def _agree_stats(per: np.ndarray, rng_idx: np.ndarray) -> dict[str, Any]:
    tot = per.sum(0)
    n = tot.sum()
    agr = (tot[0] + tot[3]) / n if n else float("nan")
    b = per[rng_idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ab = (b[:, 0] + b[:, 3]) / b.sum(1)
    return {
        "n": int(n),
        "agree": float(agr),
        "ci90": stats.ci(ab),
        "aa": int(tot[0]),
        "ar": int(tot[1]),
        "ra": int(tot[2]),
        "rr": int(tot[3]),
    }


def _family_selector(wsel: Selector, f: str, d: str | None = None) -> Selector:
    def sel(k: str) -> bool:
        parts = k.split("|")
        return wsel(k) and parts[0] == f and (d is None or parts[1] == d)

    return sel


def agreement(recs: Sequence[Mapping[str, Any]], shadow: str, n_seeds: int) -> dict[str, Any]:
    """Agreement of the applied and the shadow decisions, overall, per family and per direction."""
    idx = np.random.default_rng(list(AGREE_KEY)).integers(0, n_seeds, (AGREE_BOOT, n_seeds))
    out = {}
    for win, wsel in (("all", lambda k: True), ("post", lambda k: k.endswith("|post"))):
        o = {"overall": _agree_stats(_agree_sum(recs, shadow, wsel), idx)}
        for f in FAMS:
            o[f] = _agree_stats(_agree_sum(recs, shadow, _family_selector(wsel, f)), idx)
            for d in ("+1", "-1"):
                o[f"{f}|{d}"] = _agree_stats(
                    _agree_sum(recs, shadow, _family_selector(wsel, f, d)), idx
                )
        out[win] = o
    return out


def same_fields(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Outcome fields on which two records differ."""
    skip = SKIP | {"policy_counts", "key", "kind", "schema", "arm"}
    return [x for x in sorted((set(a) | set(b)) - skip) if a.get(x) != b.get(x)]


def _identity_arm(
    Rx: stats.Records, Rs: stats.Records, seeds: Sequence[int], a: str, src: str
) -> dict[str, Any]:
    nd = pol_eq = n_have = 0
    fields: dict[str, dict[str, Any]] = {}
    dseeds = []
    for s in seeds:
        v, k = Rx.get(s, {}).get(a), Rs[s][src]
        if v is None:
            continue
        n_have += 1
        d = same_fields(v, k)
        nd += not d
        if d:
            dseeds.append(s)
        for x in d:
            f = fields.setdefault(x, {"seeds": 0, "max_abs": 0.0})
            f["seeds"] += 1
            if isinstance(v.get(x), int | float) and isinstance(k.get(x), int | float):
                f["max_abs"] = max(f["max_abs"], abs(v[x] - k[x]))
        pol_eq += (k.get("policy_counts") == v.get("policy_counts")) if src != REF else 1
    return {
        "source": src,
        "n": n_have,
        "identical_outcome_seeds": nd,
        "policy_counts_equal_seeds": pol_eq,
        "diff_fields": fields,
        "diff_seeds": dseeds,
    }


def _relation(Rx: stats.Records, Rs: stats.Records, seeds: Sequence[int]) -> dict[str, Any]:
    rel = {}
    for r in REFS:
        a = arm_name(r, "own")
        both = [s for s in seeds if a in Rx.get(s, {}) and SHADOW_ARM in Rx.get(s, {})]
        k_eq = {s for s in both if not same_fields(Rs[s][r], Rs[s][REF])}
        v_eq = {s for s in both if not same_fields(Rx[s][a], Rx[s][SHADOW_ARM])}
        rel[r] = {
            "n": len(both),
            "stored_identical_to_ref": len(k_eq),
            "vps_identical_to_ref": len(v_eq),
            "relation_same": len(both) - len(k_eq ^ v_eq),
        }
    return rel


def _agreements(Rx: stats.Records, seeds: Sequence[int]) -> dict[str, Any]:
    agr: dict[str, Any] = {"stream_ref": {}, "stream_own": {}}
    if all(SHADOW_ARM in Rx.get(s, {}) for s in seeds):
        recs = [Rx[s][SHADOW_ARM] for s in seeds]
        agr["stream_ref"] = {r: agreement(recs, r, len(seeds)) for r in REFS}
    for r in REFS:
        a = arm_name(r, "own")
        if all(a in Rx.get(s, {}) for s in seeds):
            agr["stream_own"][r] = agreement([Rx[s][a] for s in seeds], REF, len(seeds))
    return agr


def _rows_and_diffs(
    R: stats.Records,
    st: stats.GateA,
    Vb: Mapping[str, np.ndarray],
    a3: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    P, B = st.point, st.boot
    rows: dict[str, Any] = {}
    diffs: dict[str, Any] = {}
    for a in [REF, "noarb", *REFS, *a3]:
        rows[a] = dict(
            P[a],
            R_ci90=stats.ci(B[a]["R"]),
            Rstar_ci90=stats.ci(B[a]["Rstar"]),
            retention_ci90=stats.ci(B[a]["retention"]),
            guard_ratio_ci90={g: stats.ci(v) for g, v in B[a]["guard"].items()},
        )
        if a in Vb and a != REF:
            dv_seed = np.array([R[s][a]["psvr"] - R[s][REF]["psvr"] for s in st.seeds])
            same_seed = sum(not same_fields(R[s][a], R[s][REF]) for s in st.seeds)
            diffs[a] = {
                "dR": P[a]["R"] - P[REF]["R"],
                "dR_ci90": stats.ci(B[a]["R"] - B[REF]["R"]),
                "dV": P[a]["V"] - P[REF]["V"],
                "dV_ci90": stats.ci(Vb[a] - Vb[REF]),
                "seeds_bit_identical_to_ref": int(same_seed),
                "per_seed_dpsvr": {
                    "mean": float(dv_seed.mean()),
                    "sd": float(dv_seed.std(ddof=1)),
                    "zero": int((dv_seed == 0).sum()),
                    "q05_q50_q95": [float(x) for x in np.quantile(dv_seed, [0.05, 0.5, 0.95])],
                },
            }
    return rows, diffs


def _same_platform(
    R: stats.Records,
    st: stats.GateA,
    Vb: Mapping[str, np.ndarray],
    vps: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if SHADOW_ARM not in vps:
        return out
    P, B = st.point, st.boot
    pairs = [(arm_name(r, k), SHADOW_ARM) for r in REFS for k in ("own", "pm", "other")]
    pairs += [(arm_name(r, k), arm_name(r, "own")) for r in REFS for k in ("pm", "other")]
    pairs += [(SHADOW_ARM, REF)] + [(arm_name(r, "own"), r) for r in REFS]
    for a, b in pairs:
        if a in Vb and b in Vb:
            ss = sum(not same_fields(R[s][a], R[s][b]) for s in st.seeds)
            out[f"{a} - {b}"] = {
                "dR": P[a]["R"] - P[b]["R"],
                "dR_ci90": stats.ci(B[a]["R"] - B[b]["R"]),
                "dV": P[a]["V"] - P[b]["V"],
                "dV_ci90": stats.ci(Vb[a] - Vb[b]),
                "seeds_bit_identical": int(ss),
            }
    return out


def analyse(
    stored: Path,
    x7: Path,
    *,
    artifacts: Path = ARTIFACTS,
    cs_eval: Path,
    factors: Path,
    n_boot: int = stats.N_BOOT,
) -> dict[str, Any]:
    """X7 over the stored EVAL records and the X7 shards.

    Args:
        stored: Directory of the stored certified-safe EVAL runs (`*/res_*.jsonl`).
        x7: Directory of the X7 shards (`res_*.jsonl` at any depth).
        artifacts: The frozen artifacts.
        cs_eval: The frozen `cs_eval.json`.
        factors: The speed-factor file of the runtime table.
        n_boot: Bootstrap draws.

    Returns:
        The output tree (`x7_tables.json`).
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    art = read_json(artifacts / "E6P_CERTSAFE.json")
    Rs, _ = alias_filled(
        sorted(stored.glob("*/res_*.jsonl")), study3.job_filter("eval"), art["alias_of"]
    )
    Rx, heads_x = stats.load_jobs(shard_files(x7), sub_filter(SUB))
    seeds = study3.stage_seeds("eval")
    expect = [arm_name(r, k) for _, k, r in job_list(seeds[:1])]
    missing = {a: [s for s in seeds if a not in Rx.get(s, {})] for a in expect}
    ident = {
        a: _identity_arm(Rx, Rs, seeds, a, src)
        for a, src in [(SHADOW_ARM, REF)] + [(arm_name(r, "own"), r) for r in REFS]
    }
    rel = _relation(Rx, Rs, seeds)
    agr = _agreements(Rx, seeds)
    R = {s: dict(r) for s, r in Rs.items()}
    a3 = [arm_name(r, k) for r in REFS for k in ("pm", "other")]
    a3 = [a for a in a3 if all(a in Rx.get(s, {}) for s in seeds)]
    vps = [
        a
        for a in [SHADOW_ARM, *(arm_name(r, "own") for r in REFS)]
        if all(a in Rx.get(s, {}) for s in seeds)
    ]
    a3 = vps + a3
    for s, arms in R.items():
        for a in a3:
            arms[a] = Rx[s][a]
    stored_arms = [a for a in study3.ALL_ARMS if any(a in Rs[s] for s in Rs)]
    st = stats.gate_a(R, [*stored_arms, *a3], n_boot)
    P, B = st.point, st.boot
    E = read_json(cs_eval)
    bad = [
        a
        for a in stored_arms
        if not (
            np.isclose(P[a]["R"], E["arms"][a]["R"], rtol=RTOL)
            and ci_close(stats.ci(B[a]["R"]), E["arms"][a]["R_ci90"], RTOL)
        )
    ]
    n = st.n
    idx = np.random.default_rng([*stats.EVAL_BOOT_KEY, n]).integers(0, n, (n_boot, n))
    Vb = {a: stats.pooled(stats.arrays(R, st.seeds, a), idx)["V"] for a in [REF, *REFS, *a3]}
    rows, diffs = _rows_and_diffs(R, st, Vb, a3)
    same = _same_platform(R, st, Vb, vps)
    share = {
        r: {
            "R_full": P[r]["R"],
            "R_pm": P.get(arm_name(r, "pm"), {}).get("R"),
            "R_other": P.get(arm_name(r, "other"), {}).get("R"),
        }
        for r in REFS
    }
    fac = read_json(factors)
    vk = next((k for k in fac["factors"]["*"] if k.startswith("vps|")), None)
    cpu = sum(Rx[s][a]["cpu_s"] for s in Rx for a in Rx[s]) / 3600
    drivers = [h for h in heads_x if h.get("driver") == DRIVER and not h.get("smoke")]
    return {
        "schema": "x7-subset-check/1",
        "declared": DECLARED,
        "status": "POST HOC, DESCRIPTIVE",
        "ref": REF,
        "refs": list(REFS),
        "n_seeds": n,
        "missing": {a: v for a, v in missing.items() if v},
        "bad_lines": 0,
        "platforms": sorted(
            {json.dumps(stats.platform_key(h.get("platform")), default=str) for h in drivers}
        ),
        "code_commits": sorted(
            {str(h.get("code_commit")) for h in heads_x if h.get("driver") == DRIVER}
        ),
        "repro_cs_eval_mismatch": bad,
        "identity_reruns": ident,
        "identity_relation": rel,
        "agreement": agr,
        "rows": rows,
        "diff_vs_ref": diffs,
        "diff_same_platform": same,
        "attribution": share,
        "cpu": {
            "vps_cpu_h": cpu,
            "factor_star": fac["factors"]["*"].get(vk) if vk else None,
            "kaggle_ref_cpu_h": cpu * fac["factors"]["*"][vk]["f"] if vk else None,
        },
    }


def write(rep: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x7_tables.json`."""
    write_text(Path(out_dir) / "x7_tables.json", json.dumps(rep, indent=1, default=js_default))


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: score the X7 shards against the stored EVAL records."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--x7", default=None, help="X7 shards; default the frozen record set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    a = ap.parse_args(argv)
    rec = Path(a.records)
    rep = analyse(
        rec / "study2_3_decision" / "runs" / "e6p-cs-eval-1",
        Path(a.x7) if a.x7 else rec / "posthoc_x7" / "x7" / "records",
        cs_eval=rec / "system_artifacts" / "decision" / "cs_eval.json",
        factors=rec / "posthoc_x1" / "exp_c" / "calib" / "factors.json",
        n_boot=a.n_boot,
    )
    write(rep, a.out)
    print(json.dumps({"n_seeds": rep["n_seeds"], "mismatch": rep["repro_cs_eval_mismatch"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

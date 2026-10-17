"""X4: a random accept / defer referee on the Study 3 EVAL episodes.

Post hoc and descriptive: no hypothesis, no criterion, no change to any Study 1-3 file.

Policy `X4:rand@<p>` defers every xApp request that reaches the referee ("reject") independently
with probability p, from t = 0, all families and directions alike. p = .352 is MG:PMRT's
request-level deferral rate in the certified-safe EVAL (37890 / 107525), and .25 and .50 bracket
it. The plant, the seeds (191100-191259) and the record schema are those of the certified-safe
driver. The provenance jobs (`noarb` and `never_sleep` on 191100 and 191101) use the frozen
driver's own arbiters, and `analyse` compares them with the stored Kaggle records field by field.

The run half is `oran_causal.experiments.posthoc.x4`; this module scores its records together with
the stored EVAL records on one paired bootstrap.
"""

import argparse
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

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

SUB = "x4"
TAG = 6640
P_PMRT = 0.352
PS = (P_PMRT, 0.25, 0.50)
ARMS = tuple(f"X4:rand@{p:g}" for p in PS)
P_OF = dict(zip(ARMS, PS, strict=True))
PROV_ARMS = ("noarb", "never_sleep")
PROV_SEEDS = (191100, 191101)
COMPARE = ("noarb", "never_sleep", "MG:PMRT", "CS:PMRT", "CS:GT")
SKIP = frozenset({"cpu_s", "secs", "rss_mb", "sub", "aliases", "signature", "alias_of", "x4"})
GUARDS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
DRIVER = "x4_random_defer"
DECLARED = "scratchpad/xmethod/EXTRAS_PROTOCOL.md Amendments 2026-10-05"
RTOL = 1e-9


def provenance(Rx: stats.Records, Rs: stats.Records) -> list[dict[str, Any]]:
    """VPS re-runs of `noarb` and `never_sleep` against the stored records, field by field."""
    prov = []
    for s in PROV_SEEDS:
        for a in PROV_ARMS:
            v, k = Rx.get(s, {}).get(a), Rs.get(s, {}).get(a)
            if v is None or k is None:
                prov.append({"seed": s, "arm": a, "present": False})
                continue
            keys = sorted((set(v) | set(k)) - SKIP - {"policy_counts", "key", "kind", "schema"})
            diff = {x: [k.get(x), v.get(x)] for x in keys if k.get(x) != v.get(x)}
            prov.append(
                {
                    "seed": s,
                    "arm": a,
                    "present": True,
                    "identical": not diff and k.get("policy_counts") == v.get("policy_counts"),
                    "diff": diff,
                    "policy_equal": k.get("policy_counts") == v.get("policy_counts"),
                    "cpu_kaggle": k.get("cpu_s"),
                    "cpu_vps": v.get("cpu_s"),
                }
            )
    return prov


def repro_cs_eval(
    st: stats.GateA, E: Mapping[str, Any], stored_arms: Sequence[str]
) -> dict[str, Any]:
    """The stored arms' values on the joint bootstrap against the frozen `cs_eval.json`."""
    repro = {
        "gate_a": all(
            np.isclose(getattr(st, k), E["gate_a"][k], rtol=1e-12, atol=0)
            for k in ("V_AA", "V_ref", "den")
        )
        and st.ref_arm == E["gate_a"]["ref_arm"]
        and st.n == E["gate_a"]["n"] == 160
    }
    P, B = st.point, st.boot
    bad = []
    for a in stored_arms:
        e, p = E["arms"][a], P[a]
        ok = (
            all(
                np.isclose(p[k], e[k], rtol=RTOL, atol=1e-12)
                for k in ("V", "R", "Rstar", "retention")
            )
            and p["eligible"] == e["eligible"]
            and all(
                np.isclose(p["guard_ratio"][g], e["guard_ratio"][g], rtol=RTOL)
                for g in p["guard_ratio"]
            )
            and ci_close(stats.ci(B[a]["R"]), e["R_ci90"], RTOL)
            and all(
                ci_close(stats.ci(B[a]["guard"][g]), e["guard_ratio_ci90"][g], RTOL)
                for g in p["guard_ratio"]
            )
        )
        if not ok:
            bad.append(a)
    repro["arms"] = not bad
    repro["mismatch"] = bad
    return repro


def _row(R: stats.Records, st: stats.GateA, a: str) -> dict[str, Any]:
    P, B = st.point, st.boot
    rec = [R[s][a] for s in st.seeds]
    req, rej = sum(r["st_req"] for r in rec), sum(r["st_rej"] for r in rec)
    fam: dict[str, int] = {}
    for r in rec:
        for k_, v_ in (r.get("policy_counts") or {}).get("defer", {}).items():
            fam[k_] = fam.get(k_, 0) + v_
    return dict(
        P[a],
        R_ci90=stats.ci(B[a]["R"]),
        Rstar_ci90=stats.ci(B[a]["Rstar"]),
        retention_ci90=stats.ci(B[a]["retention"]),
        guard_ratio_ci90={g: stats.ci(v) for g, v in B[a]["guard"].items()},
        requests=req,
        deferred=rej,
        defer_rate=rej / req if req else float("nan"),
        defer_by_family_phase=dict(sorted(fam.items())),
    )


def cpu_block(
    Rx: stats.Records, prov: Sequence[Mapping[str, Any]], factors: Mapping[str, Any]
) -> dict[str, Any]:
    """CPU hours of the X4 jobs on the VPS and as Kaggle reference seconds."""
    cpu_vps = sum(Rx[s][a]["cpu_s"] for s in Rx for a in Rx[s] if Rx[s][a].get("sub") == SUB) / 3600
    cpu_x4 = sum(Rx[s][a]["cpu_s"] for s in Rx for a in Rx[s] if a in ARMS) / 3600
    fk = [p["cpu_kaggle"] / p["cpu_vps"] for p in prov if p.get("present") and p.get("cpu_vps")]
    vps_key = next((k for k in factors["factors"]["*"] if k.startswith("vps|")), None)
    fstar = factors["factors"]["*"][vps_key] if vps_key else None
    return {
        "vps_cpu_h_all": cpu_vps,
        "vps_cpu_h_x4": cpu_x4,
        "factor_star": fstar,
        "factor_star_host": vps_key,
        "kaggle_ref_cpu_h_x4_star": cpu_x4 * fstar["f"] if fstar else None,
        "kaggle_ref_cpu_h_x4_star_lo_hi": (
            [cpu_x4 * fstar["f_lo"], cpu_x4 * fstar["f_hi"]] if fstar else None
        ),
        "factor_direct_prov": fk,
        "factor_direct_mean": float(np.mean(fk)) if fk else None,
        "kaggle_ref_cpu_h_x4_direct": cpu_x4 * float(np.mean(fk)) if fk else None,
    }


def analyse(
    stored: Path,
    x4: Path,
    *,
    artifacts: Path = ARTIFACTS,
    cs_eval: Path,
    factors: Path,
    n_boot: int = stats.N_BOOT,
) -> dict[str, Any]:
    """X4 over the stored EVAL records and the X4 shards.

    Args:
        stored: Directory of the stored certified-safe EVAL runs (`*/res_*.jsonl`).
        x4: Directory of the X4 shards (`res_*.jsonl` at any depth).
        artifacts: The frozen artifacts.
        cs_eval: The frozen `cs_eval.json`.
        factors: The speed-factor file of the runtime table.
        n_boot: Bootstrap draws.

    Returns:
        The output tree (`x4_tables.json`).
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    art = read_json(artifacts / "E6P_CERTSAFE.json")
    Rs, _ = alias_filled(
        sorted(stored.glob("*/res_*.jsonl")), study3.job_filter("eval"), art["alias_of"]
    )
    Rx, heads_x = stats.load_jobs(shard_files(x4), sub_filter(SUB))
    stored_arms = [a for a in study3.ALL_ARMS if any(a in Rs[s] for s in Rs)]
    prov = provenance(Rx, Rs)
    R = {s: dict(r) for s, r in Rs.items()}
    for s, arms in R.items():
        for a in ARMS:
            if a in Rx.get(s, {}):
                arms[a] = Rx[s][a]
    seeds_eval = study3.stage_seeds("eval")
    x4_arms = [a for a in ARMS if all(a in Rx.get(s, {}) for s in seeds_eval)]
    missing = {a: [s for s in seeds_eval if a not in Rx.get(s, {})] for a in ARMS}
    st = stats.gate_a(R, [*stored_arms, *x4_arms], n_boot)
    repro = repro_cs_eval(st, read_json(cs_eval), stored_arms)
    P, B = st.point, st.boot
    rows = {a: _row(R, st, a) for a in [*COMPARE, *x4_arms] if a in P}
    contrasts = {
        a: {
            c: {"dR": P[a]["R"] - P[c]["R"], "ci90": stats.ci(B[a]["R"] - B[c]["R"])}
            for c in COMPARE
        }
        for a in x4_arms
    }
    drivers = [h for h in heads_x if h.get("driver") == DRIVER and not h.get("smoke")]
    plats = sorted(
        {json.dumps(stats.platform_key(h.get("platform")), default=str) for h in drivers}
    )
    commits = sorted({str(h.get("code_commit")) for h in drivers})
    return {
        "schema": "x4-random-defer/1",
        "declared": DECLARED,
        "status": "POST HOC, DESCRIPTIVE",
        "p": dict(P_OF),
        "tag": TAG,
        "n_seeds": st.n,
        "seeds": [st.seeds[0], st.seeds[-1]] if st.seeds else None,
        "gate_a": {"V_AA": st.V_AA, "V_ref": st.V_ref, "ref_arm": st.ref_arm, "den": st.den},
        "repro_cs_eval": repro,
        "provenance": prov,
        "provenance_identical": all(p.get("identical") for p in prov),
        "x4_missing": missing,
        "x4_bad_lines": 0,
        "x4_platforms": plats,
        "x4_code_commits": commits,
        "rows": rows,
        "contrasts": contrasts,
        "cpu": cpu_block(Rx, prov, read_json(factors)),
        "n_boot": n_boot,
    }


def _f(x: float | None, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:+.{nd}f}" if nd == 3 else f"{x:.{nd}f}"


def table_md(rep: Mapping[str, Any]) -> str:
    """The X4 tables as markdown."""
    head = (
        "| arm | defer rate | R [90 % CI] | R* | retention [90 % CI] | svr | nonprot eMBB | LL | "
        "RLF [90 % CI] | eligible |"
    )
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in rep["rows"].items():
        g = r["guard_ratio"]
        lo, hi = r["R_ci90"]
        rlo, rhi = r["retention_ci90"]
        flo, fhi = r["guard_ratio_ci90"]["rlf"]
        lines.append(
            f"| {a} | {r['defer_rate']:.3f} | {_f(r['R'])} [{_f(lo)}, {_f(hi)}] | "
            f"{_f(r['Rstar'])} | {r['retention']:.3f} [{rlo:.3f}, {rhi:.3f}] | "
            + " | ".join(f"{g[k]:.2f}" for k in GUARDS[:3])
            + f" | {g['rlf']:.2f} [{flo:.2f}, {fhi:.2f}] | {'yes' if r['eligible'] else 'no'} |"
        )
    lines += [
        "",
        "| X4 arm | " + " | ".join(f"dR vs {c} [90 % CI]" for c in COMPARE) + " |",
        "|---|" + "---|" * len(COMPARE),
    ]
    for a, cs in rep["contrasts"].items():
        lines.append(
            f"| {a} | "
            + " | ".join(
                f"{_f(v['dR'])} [{_f(v['ci90'][0])}, {_f(v['ci90'][1])}]" for v in cs.values()
            )
            + " |"
        )
    return "\n".join(lines)


def write(rep: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x4_tables.json`."""
    write_text(Path(out_dir) / "x4_tables.json", json.dumps(rep, indent=1, default=js_default))


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: score the X4 shards against the stored EVAL records."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--x4", default=None, help="X4 shards; default the frozen record set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    a = ap.parse_args(argv)
    rec = Path(a.records)
    rep = analyse(
        rec / "study2_3_decision" / "runs" / "e6p-cs-eval-1",
        Path(a.x4) if a.x4 else rec / "posthoc_x4" / "x4" / "records",
        cs_eval=rec / "system_artifacts" / "decision" / "cs_eval.json",
        factors=rec / "posthoc_x1" / "exp_c" / "calib" / "factors.json",
        n_boot=a.n_boot,
    )
    write(rep, a.out)
    print(table_md(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

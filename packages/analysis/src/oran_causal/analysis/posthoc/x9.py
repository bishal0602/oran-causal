"""X9: the end-to-end decision table of Studies 2 and 3.

Post hoc and descriptive: no verdict, no criterion, no frozen file changes, no new simulation.

Study 3 (certified-safe EVAL, 160 seeds 191100-191259): the stored records are scored with the
frozen Gate A statistics (aliased CS arms filled from their targets, 10 000 bootstrap draws,
`default_rng([6624, 20, 160])`). Every arm must reproduce the stored `cs_eval.json`, else the
analysis stops. Per arm: V, R, R*, retention, the four guard ratios with their 90 % intervals,
eligibility, the request-level rejection rate (sum of `st_rej` over sum of `st_req` over the 160
seeds, the X4 definition) and the seed-paired differences against `never_sleep` on the same
bootstrap matrix (dR as the frozen secondary contrasts, dR* as the frozen D3).

Study 2 (option (a) EVAL, 160 seeds): the same on the conf EVAL records and the maps alias table;
it must reproduce `conf_eval.json`.
"""

import argparse
import csv
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc.common import ci_close
from oran_causal.analysis.posthoc.paths import ARTIFACTS, PROTOCOL_NAME, RECORDS, write_text
from oran_causal.analysis.system import stats, study2, study3

GUARDS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
REF = "never_sleep"
NON_OUTCOME = frozenset(
    {
        "cpu_s",
        "secs",
        "rss_mb",
        "sub",
        "aliases",
        "signature",
        "alias_of",
        "arm",
        "key",
        "kind",
        "schema",
    }
)
GROUP3 = {
    "noarb": "accept-all",
    "never_sleep": "static never-sleep",
    "MG:PMRT": "ungated PMRT",
    "CS:PMRT": "certified PMRT",
    "CS:GT": "certified GT oracle",
    "CS:rand": "gated control",
    "CS:blanket2": "gated control",
}
GROUP2 = {
    "noarb": "accept-all",
    "never_sleep": "static never-sleep",
    "MG:PMRT": "ungated PMRT",
    "MG:GT": "ungated GT oracle",
    "MG:rand": "ungated control",
    "blanket2": "ungated control",
}
QUOTED = [
    ("never_sleep", "R", 0.256, 3),
    ("never_sleep", "V", 160.32, 2),
    ("never_sleep", "retention", 1.146, 3),
    ("MG:PMRT", "R", 0.201, 3),
    ("MG:PMRT", "retention", 1.634, 3),
    ("CS:GT", "R", 0.221, 3),
    ("CS:GT", "retention", 1.294, 3),
    ("noarb", "V", 175.30, 2),
]
DECLARED = f"{PROTOCOL_NAME} amendment 2026-10-11 X9"
CSV_COLUMNS = (
    "study",
    "arm",
    "group",
    "alias_of",
    "V",
    "R",
    "R_lo",
    "R_hi",
    "Rstar",
    "retention",
    "ret_lo",
    "ret_hi",
)
RTOL = 1e-9


class ReproductionError(RuntimeError):
    """The stored records do not reproduce the frozen evaluation; the analysis stops."""


def eval_files(records: Path, run: str) -> list[Path]:
    """The `res_*.jsonl` shards of an EVAL run, in path order."""
    return sorted((records / "study2_3_decision" / "runs" / run).glob("*/res_*.jsonl"))


def repro(st: stats.GateA, frozen: Mapping[str, Any], arms: Sequence[str]) -> dict[str, Any]:
    """Every arm's values, eligibility and intervals against the frozen eval JSON."""
    p_, b_ = st.point, st.boot
    bad = []
    for a in arms:
        e, p = frozen["arms"][a], p_[a]
        ok = (
            all(
                np.isclose(p[k], e[k], rtol=RTOL, atol=1e-12)
                for k in ("V", "R", "Rstar", "retention")
            )
            and p["eligible"] == e["eligible"]
            and all(np.isclose(p["guard_ratio"][g], e["guard_ratio"][g], rtol=RTOL) for g in GUARDS)
            and ci_close(stats.ci(b_[a]["R"]), e["R_ci90"], RTOL)
            and ci_close(stats.ci(b_[a]["Rstar"]), e["Rstar_ci90"], RTOL)
            and ci_close(stats.ci(b_[a]["retention"]), e["retention_ci90"], RTOL)
            and all(
                ci_close(stats.ci(b_[a]["guard"][g]), e["guard_ratio_ci90"][g], RTOL)
                for g in GUARDS
            )
        )
        if not ok:
            bad.append(a)
    ga = all(
        np.isclose(getattr(st, k), frozen["gate_a"][k], rtol=1e-12, atol=0)
        for k in ("V_AA", "V_ref", "den")
    )
    ok = (
        not bad
        and ga
        and st.ref_arm == frozen["gate_a"]["ref_arm"]
        and st.n == frozen["n_seeds"] == 160
    )
    return {"ok": bool(ok), "gate_a": bool(ga), "mismatch": bad, "n_arms": len(arms)}


def rows_of(
    R: stats.Records,
    st: stats.GateA,
    arms: Sequence[str],
    group: Mapping[str, str],
    alias_of: Mapping[str, str],
) -> list[dict[str, Any]]:
    """The table rows of the given arms."""
    p_, b_ = st.point, st.boot
    out = []
    for a in arms:
        p = p_[a]
        req = int(sum(R[s][a]["st_req"] for s in st.seeds))
        rej = int(sum(R[s][a]["st_rej"] for s in st.seeds))
        d_r, d_rs = b_[a]["R"] - b_[REF]["R"], b_[a]["Rstar"] - b_[REF]["Rstar"]
        tgt = alias_of.get(a, a)
        default = (
            "gated associational"
            if a.startswith("CS:")
            else "ungated associational"
            if a.startswith("MG:")
            else "anchor"
        )
        out.append(
            {
                "arm": a,
                "group": group.get(a) or default,
                "alias_of": tgt if tgt != a else None,
                "V": p["V"],
                "R": p["R"],
                "R_ci90": stats.ci(b_[a]["R"]),
                "Rstar": p["Rstar"],
                "retention": p["retention"],
                "retention_ci90": stats.ci(b_[a]["retention"]),
                "guard_ratio": {g: p["guard_ratio"][g] for g in GUARDS},
                "guard_ratio_ci90": {g: stats.ci(b_[a]["guard"][g]) for g in GUARDS},
                "matched": p["matched"],
                "guard_ok": p["guard_ok"],
                "eligible": p["eligible"],
                "requests": req,
                "rejected": rej,
                "rejection_rate": rej / req if req else None,
                "dR_vs_never_sleep": p["R"] - p_[REF]["R"],
                "dR_vs_never_sleep_ci90": stats.ci(d_r),
                "dRstar_vs_never_sleep": p["Rstar"] - p_[REF]["Rstar"],
                "dRstar_vs_never_sleep_ci90": stats.ci(d_rs),
            }
        )
    return out


def identity(R: stats.Records, seeds: Sequence[int], a: str, b: str) -> dict[str, Any]:
    """Seeds on which records a and b agree in every outcome field and in the deferral counts.

    All fields but `NON_OUTCOME` and the decision counter log `policy_counts` are compared, plus
    `policy_counts["defer"]`, as the frozen report's check "CS:PMRT record identical ...".
    """
    skip = NON_OUTCOME | {"policy_counts"}
    same, diff_fields = 0, set()
    for s in seeds:
        ra, rb = R[s][a], R[s][b]
        keys = (set(ra) | set(rb)) - skip
        d = {k for k in keys if ra.get(k) != rb.get(k)}
        if ra["policy_counts"].get("defer") != rb["policy_counts"].get("defer"):
            d.add("policy_counts.defer")
        same += not d
        diff_fields |= d
    return {
        "a": a,
        "b": b,
        "identical_seeds": same,
        "n": len(seeds),
        "differing_fields": sorted(diff_fields),
        "n_fields_compared": len((set(R[seeds[0]][a]) | set(R[seeds[0]][b])) - skip) + 1,
    }


def _quoted(P: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"arm": a, "field": f, "quoted": q, "got": P[a][f], "ok": bool(round(P[a][f], nd) == q)}
        for a, f, q, nd in QUOTED
    ]


def study3_block(records: Path, artifacts: Path, x4_tables: Mapping[str, Any]) -> dict[str, Any]:
    """The Study 3 block of the table."""
    files = eval_files(records, "e6p-cs-eval-1")
    R, _ = stats.load_jobs(files, study3.job_filter("eval"))
    art = json.loads((artifacts / "study3_certsafe.json").read_text(encoding="utf-8"))
    stats.alias_fill(R, art["alias_of"])
    frozen = json.loads(
        (records / "system_artifacts" / "decision" / "cs_eval.json").read_text(encoding="utf-8")
    )
    return study3_of(R, art, frozen, x4_tables, len(files))


def study3_of(
    R: stats.Records,
    art: Mapping[str, Any],
    frozen: Mapping[str, Any],
    x4_tables: Mapping[str, Any],
    n_files: int,
) -> dict[str, Any]:
    """The Study 3 block of alias-filled EVAL records, checked against the evaluation `frozen`."""
    arms = [x for x in study3.ALL_ARMS if any(x in R[s] for s in R)]
    st = stats.gate_a(R, arms, stats.N_BOOT)
    rp = repro(st, frozen, arms)
    quoted = _quoted(st.point)
    ident = identity(R, st.seeds, "CS:PMRT", REF)
    rp["quoted"] = quoted
    rp["cs_pmrt_identical_to_never_sleep"] = ident
    rp["ok"] = bool(rp["ok"] and all(q["ok"] for q in quoted) and ident["identical_seeds"] == 160)
    show = ["noarb", REF, "MG:PMRT", "CS:PMRT", "CS:GT", *study3.CS_ASSOC, "CS:rand", "CS:blanket2"]
    rows = rows_of(R, st, show, GROUP3, art["alias_of"])
    byarm = {r["arm"]: r for r in rows}
    x4chk = {
        a: (v["requests"], v["deferred"]) == (byarm[a]["requests"], byarm[a]["rejected"])
        for a, v in x4_tables["rows"].items()
        if a in byarm
    }
    rp["x4_rejection_counts_equal"] = x4chk
    rp["ok"] = bool(rp["ok"] and all(x4chk.values()))
    return {
        "n_seeds": st.n,
        "seeds": [st.seeds[0], st.seeds[-1]],
        "n_files": n_files,
        "gate_a": {"V_AA": st.V_AA, "V_ref": st.V_ref, "ref_arm": st.ref_arm, "den": st.den},
        "repro": rp,
        "rows": rows,
        "composition_units": frozen.get("composition") or None,
    }


def study2_block(records: Path, artifacts: Path) -> dict[str, Any]:
    """The Study 2 block of the table."""
    files = eval_files(records, "e6p-conf-eval-1")
    art = json.loads((artifacts / "study2_maps.json").read_text(encoding="utf-8"))
    R, _ = stats.load_jobs(files, study2.EVAL_FILTER)
    stats.alias_fill(R, art["alias_of"])
    frozen = json.loads(
        (records / "system_artifacts" / "decision" / "conf_eval.json").read_text(encoding="utf-8")
    )
    return study2_of(R, art, frozen, len(files))


def study2_of(
    R: stats.Records, art: Mapping[str, Any], frozen: Mapping[str, Any], n_files: int
) -> dict[str, Any]:
    """The Study 2 block of alias-filled EVAL records, checked against the evaluation `frozen`."""
    arms = [x for x in study2.ALL_ARMS if any(x in R[s] for s in R)]
    st = stats.gate_a(R, arms, stats.N_BOOT)
    rp = repro(st, frozen, arms)
    sec = frozen["secondary"].get(REF)
    P, B = st.point, st.boot
    if sec:
        d_r = P["MG:PMRT"]["R"] - P[REF]["R"]
        ci = stats.ci(B["MG:PMRT"]["R"] - B[REF]["R"])
        rp["frozen_pmrt_minus_never_sleep"] = {"frozen": sec, "got": {"dR": d_r, "ci90": ci}}
        rp["ok"] = bool(
            rp["ok"] and np.isclose(sec["dR"], d_r, rtol=RTOL) and ci_close(ci, sec["ci90"], RTOL)
        )
    show = ["noarb", REF, "MG:PMRT", "MG:GT", *study2.ASSOC_ARMS, "MG:rand", "blanket2"]
    return {
        "n_seeds": st.n,
        "seeds": [st.seeds[0], st.seeds[-1]],
        "n_files": n_files,
        "gate_a": {"V_AA": st.V_AA, "V_ref": st.V_ref, "ref_arm": st.ref_arm, "den": st.den},
        "repro": rp,
        "rows": rows_of(R, st, show, GROUP2, art["alias_of"]),
        "note": "Study 2 has no certified arm (the certification layer is Study 3); its map "
        "arms are ungated MapGateV2 referees. Rows are the stored option (a) EVAL arms.",
    }


def run(records: Path = RECORDS, artifacts: Path = ARTIFACTS) -> dict[str, Any]:
    """X9 from the frozen record sets.

    Returns:
        The output tree.

    Raises:
        ReproductionError: If the Study 3 records do not reproduce `cs_eval.json` or the quoted
            values.
    """
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    x4 = json.loads((records / "posthoc_x4" / "x4" / "x4_tables.json").read_text(encoding="utf-8"))
    return assemble(study3_block(records, artifacts, x4), study2_block(records, artifacts))


def assemble(s3: Mapping[str, Any], s2: Mapping[str, Any]) -> dict[str, Any]:
    """The output tree of the Study 3 and Study 2 blocks.

    Raises:
        ReproductionError: If the Study 3 block does not reproduce its evaluation.
    """
    if not s3["repro"]["ok"]:
        raise ReproductionError(
            "X9 Study 3: the stored records do not reproduce cs_eval.json / the quoted values: "
            + json.dumps(s3["repro"], indent=1, default=str)
        )
    omitted = {"omitted": "Study 2 records do not reproduce conf_eval.json", "repro": s2["repro"]}
    return {
        "schema": "x9-decision-e2e/1",
        "declared": DECLARED,
        "status": "POST HOC, DESCRIPTIVE",
        "n_boot": stats.N_BOOT,
        "boot_key": "default_rng([6624, 20, 160])",
        "rejection_rate": "sum st_rej / sum st_req over the scored seeds (requests reaching "
        "the referee, all t; X4 definition); aliased arms carry their target's records",
        "paired": "arm minus never_sleep on the one paired seed-bootstrap matrix of the frozen "
        "eval (dR as the frozen secondary contrasts, dR* as D3); 90 % CI = 5th / 95th "
        "percentiles",
        "study3": s3,
        "study2": s2 if s2["repro"]["ok"] else omitted,
    }


def write_csv(path: Path, studies: Mapping[str, Mapping[str, Any]]) -> None:
    """Writes the table rows of the given studies."""
    cols = list(CSV_COLUMNS)
    for g in GUARDS:
        cols += [g, f"{g}_lo", f"{g}_hi"]
    cols += [
        "eligible",
        "requests",
        "rejected",
        "rejection_rate",
        "dR_ns",
        "dR_ns_lo",
        "dR_ns_hi",
        "dRstar_ns",
        "dRstar_ns_lo",
        "dRstar_ns_hi",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for sname, S in studies.items():
            for r in S["rows"]:
                row = [
                    sname,
                    r["arm"],
                    r["group"],
                    r["alias_of"] or "",
                    r["V"],
                    r["R"],
                    *r["R_ci90"],
                    r["Rstar"],
                    r["retention"],
                    *r["retention_ci90"],
                ]
                for g in GUARDS:
                    row += [r["guard_ratio"][g], *r["guard_ratio_ci90"][g]]
                row += [
                    r["eligible"],
                    r["requests"],
                    r["rejected"],
                    r["rejection_rate"],
                    r["dR_vs_never_sleep"],
                    *r["dR_vs_never_sleep_ci90"],
                    r["dRstar_vs_never_sleep"],
                    *r["dRstar_vs_never_sleep_ci90"],
                ]
                w.writerow(row)


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x9_decision_e2e.json` and `x9_decision_e2e.csv`."""
    d = Path(out_dir)
    write_text(d / "x9_decision_e2e.json", json.dumps(out, indent=1, default=float))
    studies = {"study3": out["study3"]}
    if "rows" in out["study2"]:
        studies["study2"] = out["study2"]
    write_csv(d / "x9_decision_e2e.csv", studies)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: recompute X9 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.records), Path(a.artifacts))
    write(out, a.out)
    for nm, key in (("Study 3", "study3"), ("Study 2", "study2")):
        S = out[key]
        print(f"{nm}: repro ok {S['repro']['ok']} (mismatch {S['repro'].get('mismatch')})")
        for r in S.get("rows", []):
            lo, hi = r["R_ci90"]
            print(
                f"  {r['arm']:18s} V {r['V']:7.2f} R {r['R']:+.3f} [{lo:+.3f},{hi:+.3f}] "
                f"ret {r['retention']:.3f} rej {r['rejection_rate'] or 0:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

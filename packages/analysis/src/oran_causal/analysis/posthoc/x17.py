"""X17: the strict centre becomes the implemented centre of the design slope and the gate bounds.

Post hoc, declared in the posthoc protocol (2026-11-24, X17). The design slope of
`core.gate.design_slope_se` now defaults to the strict centre (a unit enters the running centre
only once its 90 s outcome window has closed); the stored Study 2 and 3 artifacts were set with the
frozen centre. This module holds the analysis halves of the declared steps:

* step 1, `analyse`: the frozen setting reproduces the stored map coefficients, gate bounds and
  uncertified set, and the strict setting reproduces the stored X16 strict values within the
  relative tolerance of amendment A1 (exact equality is reported too);
* step 2, `analyse`: the MG:PMRT decision signature and the CS:PMRT certsafe signature under both
  centres, the strict PMRT map and the strict guard-edge bounds;
* step 3, `confirm`: the strict-map MG:PMRT runs on the confirmation seeds against the stored
  frozen records of the same seeds;
* step 4, `rescore`: only after a changed MG:PMRT signature, the two EVAL blocks re-scored with
  the strict MG:PMRT records in place of the frozen ones.

The data are the DISC unit table exactly as X16 builds it. The run half (the referee episodes) is
`oran_causal.experiments.posthoc.x17`, which also writes `study2_maps_strict.json` from the
strict map of `analyse`.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc import x15, x16
from oran_causal.analysis.posthoc.paths import (
    ARTIFACTS,
    PROTOCOL_NAME,
    RECORDS,
    lf_sha256,
    write_text,
)
from oran_causal.analysis.posthoc.refs import Map, RefereeTools
from oran_causal.analysis.system import stats
from oran_causal.analysis.system.study2 import SCHEMA_EVAL
from oran_causal.core.data import UnitTable
from oran_causal.core.design import v_design
from oran_causal.core.gate import FROZEN_WINDOW_S, WINDOW_S, design_slope_se
from oran_causal.core.pmrt.units import unit_order
from oran_causal.sim.system.decision.vocab import FAMILIES

PROTOCOL = f"{PROTOCOL_NAME} 2026-11-24: X17"
SCHEMA = "x17-centre/1"
SCHEMA_BOUNDS = "x17-certsafe-bounds-strict/1"
DRIVER = "x17_strict_mgpmrt"
SUB = "x17"
STAGE_KEY = "x17_stage"
ARM = "MG:PMRT"
BLOCK_SPLIT = x15.BLOCK_SPLIT
CONFIRM_SEEDS = (*range(187000, 187004), *range(191100, 191104))
TIMING_FIELDS = frozenset({"cpu_s", "secs", "rss_mb"})
X16_JSON = Path("posthoc_x9_x16") / "revision" / "x16_centre.json"
CENTRES = {"frozen": FROZEN_WINDOW_S, "strict": WINDOW_S}

Edge = tuple[str, str, str]


def block_of(seed: int) -> str:
    """The EVAL block of a protocol seed: "study2" (187000-187159) or "study3" (191100-191259)."""
    return "study2" if seed < BLOCK_SPLIT else "study3"


def job_filter(stage: str, allow_smoke: bool = False) -> stats.JobFilter:
    """Job and header filter of the X17 run records of `stage` ("confirm" or "full")."""
    return stats.JobFilter(
        schema=SCHEMA_EVAL,
        sub=SUB,
        stage_key=STAGE_KEY,
        stage=stage,
        driver=DRIVER,
        allow_smoke=allow_smoke,
    )


def slope(ud: UnitTable, edge: Edge, window_s: float, levels: np.ndarray) -> dict[str, Any]:
    """PMRT design slope of an edge on the DISC units with the centre of `window_s`."""
    f, rel, kpi = edge
    rows = unit_order(ud, ud.rows_of(f))
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows], levels)
    return design_slope_se(
        v,
        ud.y[(rel, kpi)][rows],
        ud.pre[(rel, kpi)][rows],
        ud.episode[rows],
        ud.sgn[rows],
        ud.t0[rows],
        window_s=window_s,
    )


def strict_map(m: Mapping[Edge, float | None], slopes: Mapping[Edge, Mapping[str, Any]]) -> Map:
    """The PMRT map with strict coefficients: each declared sign times |strict design slope|.

    Raises:
        ValueError: A map edge without a declared value.
    """
    out: Map = {}
    for h, b in sorted(m.items()):
        if b is None:
            raise ValueError(f"map edge {h} has no declared value")
        out[h] = float(np.sign(b) * abs(float(slopes[h]["beta"])))
    return out


def _x16_check(
    slopes: Mapping[Edge, Mapping[str, Any]], x16_stored: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    for r in x16_stored["maps"]:
        h = (r["family"], r["rel"], r["kpi"])
        rows.append([*h, "map_beta", r["beta_strict"], slopes[h]["beta"]])
    for r in x16_stored["gate_bounds"]:
        if r["d"] != 1:
            continue
        h = (r["family"], r["rel"], r["kpi"])
        rows.append([*h, "beta", r["beta_strict"], slopes[h]["beta"]])
        rows.append([*h, "se", r["se_strict"], slopes[h]["se"]])
    diffs = [abs(float(a) - float(b)) for *_, a, b in rows if a is not None and b is not None]
    exact = all(a == b for *_, a, b in rows)
    close = all(
        a == b if a is None or b is None else x16.close(float(a), float(b)) for *_, a, b in rows
    )
    return {
        "rows": rows,
        "exact": bool(exact and rows),
        "close": bool(close and rows),
        "max_abs_diff": max(diffs) if diffs else None,
    }


def analyse(
    ud: UnitTable,
    certsafe_art: Mapping[str, Any],
    maps_art: Mapping[str, Any],
    x16_stored: Mapping[str, Any],
    tools: RefereeTools,
    *,
    stored_units: int | None,
) -> dict[str, Any]:
    """Steps 1 and 2 of X17 on the DISC unit table.

    Args:
        ud: The DISC unit table.
        certsafe_art: The frozen certified-safe artifact.
        maps_art: The frozen Study 2 maps artifact.
        x16_stored: The stored `x16_centre.json`.
        tools: Referee functions.
        stored_units: Unit count of the stored bounds run.

    Returns:
        The result tree (without the provenance stamps `run` adds). `ok` is the step-1 gate;
        `strict_map` and `strict_bounds` feed the strict artifacts.
    """
    stored = tools.bounds_from_json(certsafe_art["bounds"]["pmrt"]) or {}
    tau = tools.tau_from_json(certsafe_art["tau"])
    theta = float(certsafe_art["mapgate"]["theta"])
    m = tools.map_from_json(maps_art["arms"][ARM]["map"])
    if tools.map_from_json(certsafe_art["arms"]["CS:PMRT"]["map"]) != m:
        raise ValueError("the certified-safe PMRT map differs from the Study 2 PMRT map")
    guards = {(f, rel, k) for f in FAMILIES for rel in tools.relations for k in tools.guard_kpis}
    keys = sorted(set(m) | guards)
    slopes = {c: {h: slope(ud, h, w, tools.levels) for h in keys} for c, w in CENTRES.items()}
    maps = {"frozen": m, "strict": strict_map(m, slopes["strict"])}
    bounds = {c: {h: slopes[c][h] for h in stored} for c in CENTRES}
    cert = {c: tools.certify_map(maps[c], bounds[c], tau, theta) for c in CENTRES}
    unc_stored = sorted(map(list, certsafe_art["arms"]["CS:PMRT"]["uncertified"]))
    repro_map = all(
        x16.close(abs(slopes["frozen"][h]["beta"]), abs(b)) for h, b in m.items() if b is not None
    )
    repro_bounds = all(
        x16.close(slopes["frozen"][h]["beta"], sb["beta"])
        and x16.close(slopes["frozen"][h]["se"], sb["se"])
        for h, sb in stored.items()
    )
    x16_check = _x16_check(slopes["strict"], x16_stored)
    check = {
        "map_beta_reproduces": repro_map,
        "bounds_reproduce": repro_bounds,
        "uncertified_reproduces": sorted(map(list, cert["frozen"]["uncertified"])) == unc_stored,
        "n_units": int(ud.n),
        "n_units_stored": stored_units,
        "strict_equals_x16": x16_check,
    }
    check["ok"] = bool(
        repro_map
        and repro_bounds
        and check["uncertified_reproduces"]
        and int(ud.n) == stored_units
        and x16_check["close"]
    )
    mg = {c: tools.signature_key(tools.decision_signature(maps[c], theta)) for c in CENTRES}
    cs = {
        c: tools.signature_key(
            tools.certsafe_signature(maps[c], [tuple(x) for x in cert[c]["uncertified"]], theta)
        )
        for c in CENTRES
    }
    signatures = {
        "MG:PMRT": mg,
        "MG:PMRT_stored": maps_art["arms"][ARM]["signature"],
        "MG:PMRT_equal": mg["frozen"] == mg["strict"],
        "MG:PMRT_frozen_reproduces_stored": mg["frozen"] == maps_art["arms"][ARM]["signature"],
        "CS:PMRT": cs,
        "CS:PMRT_stored": certsafe_art["arms"]["CS:PMRT"]["signature"],
        "CS:PMRT_equal": cs["frozen"] == cs["strict"],
        "CS:PMRT_frozen_reproduces_stored": cs["frozen"]
        == certsafe_art["arms"]["CS:PMRT"]["signature"],
    }
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "centre": {"frozen_window_s": FROZEN_WINDOW_S, "strict_window_s": WINDOW_S},
        "check": check,
        "ok": check["ok"],
        "signatures": signatures,
        "uncertified": {c: cert[c]["uncertified"] for c in CENTRES},
        "strict_map": [[*h, b] for h, b in sorted(maps["strict"].items())],
        "frozen_map": [[*h, b] for h, b in sorted(m.items())],
        "strict_bounds": [[*h, dict(b)] for h, b in sorted(bounds["strict"].items())],
        "reading": reading(signatures),
    }


def reading(sig: Mapping[str, Any]) -> str:
    """The declared reading of the signatures (protocol X17, reading rule)."""
    if not sig["CS:PMRT_equal"]:
        return "CS:PMRT signature changed: stop and ask before simulating Study 3"
    if sig["MG:PMRT_equal"]:
        return (
            "signatures equal: every MG:PMRT and CS:PMRT decision number of Studies 2-3 and X15 "
            "is unchanged; run the step-3 confirmation only"
        )
    return "MG:PMRT signature changed: re-simulate MG:PMRT(strict) on both EVAL blocks (step 4)"


def bounds_doc(out: Mapping[str, Any], certsafe_sha: str) -> dict[str, Any]:
    """The `certsafe_bounds_strict.json` content: the strict PMRT guard-edge bounds."""
    return {
        "schema": SCHEMA_BOUNDS,
        "protocol": PROTOCOL,
        "window_s": WINDOW_S,
        "pmrt": out["strict_bounds"],
        "uncertified_CS:PMRT": out["uncertified"]["strict"],
        "frozen_certsafe_sha256": certsafe_sha,
    }


def run(
    records: Path = RECORDS,
    *,
    tools: RefereeTools,
    cache_dir: Path,
    artifacts: Path = ARTIFACTS,
) -> dict[str, Any]:
    """Steps 1 and 2 from the frozen record sets and the stored X16 output."""
    files = x16.disc_files(records)
    ud = x16.unit_table(files, cache_dir)
    certsafe_art = json.loads((artifacts / x16.CERTSAFE_ART).read_text(encoding="utf-8"))
    maps_art = json.loads((artifacts / x16.MAPS_ART).read_text(encoding="utf-8"))
    x16_stored = json.loads((records / X16_JSON).read_text(encoding="utf-8"))
    bounds = (
        records / "study2_3_decision" / "runs" / "e6p-cs-bounds-1" / "out" / "certsafe_bounds.json"
    )
    stored_units = (
        json.loads(bounds.read_text(encoding="utf-8"))["n_units"] if bounds.exists() else None
    )
    out = analyse(ud, certsafe_art, maps_art, x16_stored, tools, stored_units=stored_units)
    out["data"] = {
        "files": [str(f.relative_to(records)).replace("\\", "/") for f in files],
        "H": x16.H,
        "H_pre": x16.H,
        "stage": "eval",
    }
    out["sha256"] = {
        x16.CERTSAFE_ART: lf_sha256(artifacts / x16.CERTSAFE_ART),
        x16.MAPS_ART: lf_sha256(artifacts / x16.MAPS_ART),
        "x16_centre.json": lf_sha256(records / X16_JSON),
        "oran_causal/analysis/posthoc/x17.py": lf_sha256(Path(__file__)),
    }
    out["env"] = {"numpy": np.__version__, "python": sys.version.split()[0]}
    return out


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x17_centre.json` and `certsafe_bounds_strict.json`."""
    d = Path(out_dir)
    write_text(d / "x17_centre.json", json.dumps(out, indent=1, default=x16._js))
    doc = bounds_doc(out, out["sha256"][x16.CERTSAFE_ART])
    write_text(d / "certsafe_bounds_strict.json", json.dumps(doc, indent=1, default=x16._js))


def _diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    keys = sorted(
        k
        for k in set(a) & set(b)
        if k not in TIMING_FIELDS
        and isinstance(a[k], int | float)
        and not isinstance(a[k], bool)
        and isinstance(b[k], int | float)
    )
    d = max((abs(float(a[k]) - float(b[k])) for k in keys), default=0.0)
    same_counts = (a.get("policy_counts") or {}) == (b.get("policy_counts") or {})
    return {"fields": len(keys), "max_abs_diff": d, "policy_counts_equal": bool(same_counts)}


def stored_blocks(records: Path = RECORDS, artifacts: Path = ARTIFACTS) -> dict[str, stats.Records]:
    """The stored EVAL records of both blocks, aliases filled (X15's loader)."""
    return {b: x15.load(b, records, artifacts)[0] for b in x15.BLOCKS}


def confirm(
    run_files: Sequence[Path],
    stored: Mapping[str, stats.Records],
    signatures: Mapping[str, Any],
    seeds: Sequence[int] = CONFIRM_SEEDS,
) -> dict[str, Any]:
    """Step 3: each strict-map MG:PMRT record against the stored frozen record of its seed.

    Args:
        run_files: The X17 confirmation run files.
        stored: `stored_blocks()`.
        signatures: `analyse(...)["signatures"]`.
        seeds: The confirmation seeds.

    Returns:
        Per seed the field diff; `ok` when the signatures are equal and every record matches.
    """
    R, heads = stats.load_jobs(run_files, job_filter("confirm"))
    rows = {}
    for s in seeds:
        got, ref = R.get(s, {}).get(ARM), stored[block_of(s)].get(s, {}).get(ARM)
        if got is None or ref is None:
            rows[s] = {"missing": "run" if got is None else "stored"}
            continue
        rows[s] = _diff(got, ref)
    complete = all("missing" not in r for r in rows.values())
    identical = complete and all(
        r["max_abs_diff"] == 0.0 and r["policy_counts_equal"] for r in rows.values()
    )
    return {
        "seeds": list(seeds),
        "per_seed": rows,
        "complete": complete,
        "identical": identical,
        "signatures_equal": bool(signatures["MG:PMRT_equal"]),
        "ok": bool(identical and signatures["MG:PMRT_equal"]),
        "headers": len(heads),
    }


def _summary(st: stats.GateA, arms: Sequence[str]) -> dict[str, Any]:
    P, B = st.point, st.boot
    return {
        a: {
            "V": P[a]["V"],
            "R": P[a]["R"],
            "R_ci90": stats.ci(B[a]["R"]),
            "rlf_ratio": P[a]["guard_ratio"]["rlf"],
            "rlf_ci90": stats.ci(B[a]["guard"]["rlf"]),
            "retention": P[a]["retention"],
            "eligible": P[a]["eligible"],
            "dR_vs_never_sleep": P[a]["R"] - P[x15.REF]["R"],
            "dR_vs_never_sleep_ci90": stats.ci(B[a]["R"] - B[x15.REF]["R"]),
        }
        for a in arms
    }


def rescore(
    run_files: Sequence[Path], records: Path = RECORDS, artifacts: Path = ARTIFACTS
) -> dict:
    """Step 4: both EVAL blocks with the strict MG:PMRT records in place of the frozen ones.

    Every seed of both blocks must have a strict record. Per block the frozen eval definitions
    and bootstrap (`stats.gate_a`, N_BOOT); pooled as X15 Q4.

    Raises:
        ValueError: A seed of a block without a strict record.
    """
    strict, _ = stats.load_jobs(run_files, job_filter("full"))
    data: dict[str, Any] = {}
    out: dict[str, Any] = {"schema": "x17-rescore/1", "protocol": PROTOCOL, "blocks": {}}
    for b in x15.BLOCKS:
        R, _, arms, _ = x15.load(b, records, artifacts)
        missing = [s for s in R if ARM not in strict.get(s, {})]
        if missing:
            raise ValueError(f"X17 {b}: no strict MG:PMRT record for seeds {missing[:5]}")
        for s in R:
            R[s][ARM] = strict[s][ARM]
        st = stats.gate_a(R, arms, stats.N_BOOT)
        out["blocks"][b] = {"seeds": st.n, "V_AA": st.V_AA, "arms": _summary(st, arms)}
        data[b] = {"R": R, "seeds": st.seeds}
    out["pooled"] = x15._q4(data)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command line of steps 3 and 4: `confirm` or `rescore` the X17 run records.

    Steps 1 and 2 (`analyse`) also write the strict maps artifact, which needs the referee code:
    `python -m oran_causal.experiments.posthoc x17 analyse`.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("confirm", "rescore"):
        p = sub.add_parser(name)
        p.add_argument("--records", default=str(RECORDS))
        p.add_argument("--runs", nargs="+", required=True)
        p.add_argument("--out", required=True)
        if name == "confirm":
            p.add_argument("--x17", required=True, help="x17_centre.json of steps 1-2")
    a = ap.parse_args(argv)
    runs = [Path(x) for x in a.runs]
    if a.cmd == "confirm":
        x17 = json.loads(Path(a.x17).read_text(encoding="utf-8"))
        res = confirm(runs, stored_blocks(Path(a.records)), x17["signatures"])
        write_text(Path(a.out) / "x17_confirm.json", json.dumps(res, indent=1, default=x16._js))
        print(json.dumps({"ok": res["ok"], "identical": res["identical"]}, indent=1))
        return 0 if res["complete"] else 1
    res = rescore(runs, Path(a.records))
    write_text(Path(a.out) / "x17_rescore.json", json.dumps(res, indent=1, default=x16._js))
    return 0

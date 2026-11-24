"""X18: sensitivity of the decision study to the guard limit, the energy floor and the gate budget.

Post hoc and descriptive (protocol entry 2026-11-24, X18). The frozen guard limit L = 1.10,
energy floor E = 0.90 and gate budget rho = 1.05 stay primary, and no Study 2-3 verdict is
re-judged. The records are the stored Study 2 and Study 3 EVAL records with the TreeSHAP referee
jobs spliced in (`shap_splice.merged_records`), the records the paper reports; the certified-safe
artifact is `study3_certsafe_shap.json`.

Subcommands:

* `cert` (Part B): tau at every rho of the grid from the stored G_k and N(f, d), and the
  certification, uncertified set and certified-safe signature of every gated arm. It first
  reproduces the stored tau, uncertified sets and signatures at rho 1.05 with the frozen bounds,
  then certifies with `--bounds` (the X17 strict bounds) and, optionally, `--maps` (the X17
  strict maps; only the PMRT map is taken from it). A strict signature at rho 1.05 that differs
  from the stored one stops it.
* `plan` (Part C): the signatures that no stored Study 3 job simulates, their cost and the job
  plan of `oran_causal.experiments.posthoc.x18`; more than 8 need `--allow-over-cap`.
* `eligibility` (Part A): eligibility of every stored arm at every (L, E) of both EVAL blocks
  and, with `--cert`, the gated arms at rho(L) and over the rho sweep at L = 1.10, each from the
  stored job of its signature or from the Part C records (`--part-c`, with `--plan`); a gated
  arm whose signature has no complete records is listed as pending. `merge` is `eligibility`
  that requires the Part C records.

Eligibility is held at its point value and R* = R if eligible else min(R, 0), in the point
estimate and in every resample, as the frozen Gate A; contrasts with the best eligible static
rule use the frozen paired seed bootstrap (`default_rng([6624, 20, 160])`, 10 000 draws).
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc import x9
from oran_causal.analysis.posthoc.common import js_default, read_json, shard_files, sub_filter
from oran_causal.analysis.posthoc.paths import (
    ARTIFACTS,
    PROTOCOL_NAME,
    RECORDS,
    lf_sha256,
    write_text,
)
from oran_causal.analysis.posthoc.refs import RefereeTools
from oran_causal.analysis.system import stats, study2, study3
from oran_causal.analysis.system.shap_splice import ShapSources, merged_records
from oran_causal.core.gate import GateSpace, tau_table
from oran_causal.sim.system.decision.vocab import FAMILIES

DECLARED = f"{PROTOCOL_NAME} 2026-11-24: X18"
STATUS = "POST HOC, DESCRIPTIVE"
L_GRID = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30)
E_GRID = (0.80, 0.85, 0.90, 0.95)
RHO_AT_FROZEN = (1.025, 1.05, 1.075, 1.10)
FROZEN_L, FROZEN_E, FROZEN_RHO = 1.10, 0.90, 1.05
STATIC = (*stats.ANCHORS, "never_sleep")
MAP_GATED = ("CS:PMRT", "CS:GT", *study3.CS_ASSOC)
MAP_OVERRIDE = frozenset({"MG:PMRT"})
SIG_ARM = "X18:"
REPRO_ARM = "X18:repro:CS:PMRT"
REPRO_SOURCE = "CS:PMRT"
REPRO_SEEDS = (191100, 191101, 191102, 191103)
EVAL_SEEDS = (191100, 191259)
SUB = "x18"
DRIVER = "x18_certsafe_rho"
CPU_H_PER_SIG = 2.8
CAP = 8
TOL = 1e-9
OUT_CERT = "x18_certification.json"
OUT_PLAN = "x18_plan.json"
OUT_ELIG = "x18_eligibility.json"
OUT_CSV = "x18_eligibility.csv"
OUT_MD = "REVISION_SECTIONS_X18.md"
CERTSAFE_SHAP = "study3_certsafe_shap.json"
SKIP_COMPARE = frozenset(
    {"cpu_s", "secs", "rss_mb", "sub", "aliases", "signature", "alias_of", "arm", "key", "x18"}
)
CSV_COLUMNS = (
    "section",
    "block",
    "L",
    "E",
    "rho",
    "arm",
    "target",
    "eligible",
    "matched",
    "guard_ok",
    "guard_fail",
    "R",
    "R_lo",
    "R_hi",
    "Rstar",
    "best_static",
    "dRstar",
    "dRstar_lo",
    "dRstar_hi",
    "beats",
)


class ReproductionError(RuntimeError):
    """A frozen value does not reproduce; the analysis stops."""


class StrictChangeError(RuntimeError):
    """A strict-bound signature at the frozen rho differs from the stored one."""


class OverCapError(RuntimeError):
    """The plan needs more new signatures than the declared cap."""


def rho_of(limit: float) -> float:
    """The gate budget tied to a guard limit: half its margin."""
    return round(1.0 + (float(limit) - 1.0) / 2.0, 12)


RHO_GRID = tuple(sorted({rho_of(x) for x in L_GRID} | set(RHO_AT_FROZEN)))


def key(x: float) -> str:
    """Grid key of a constant."""
    return f"{float(x):.3f}"


def cell_key(limit: float, floor: float) -> str:
    """Grid key of an (L, E) cell."""
    return f"{key(limit)}|{key(floor)}"


def sig_id(signature: str) -> str:
    """Short stable id of a certified-safe signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]


def _unc(rows: Iterable[Sequence[Any]]) -> list[list[Any]]:
    return sorted([str(f), int(d)] for f, d in rows)


def gate_space(tools: RefereeTools) -> GateSpace:
    """The gate's request classes and guard edges."""
    return GateSpace(FAMILIES, tools.relations, tools.guard_kpis)


def tau_at(art: Mapping[str, Any], tools: RefereeTools, rho: float) -> dict:
    """Tolerances at `rho` from the artifact's accept-all G_k and N(f, d)."""
    n = {(str(f), int(d)): float(v) for f, d, v in art["calib"]["N"]}
    return tau_table(art["calib"]["G"], n, gate_space(tools), rho)


def tau_rows(tau: Mapping) -> list[list[Any]]:
    """Tau as sorted `[f, d, k, value]` rows."""
    return [[f, int(d), k, float(v)] for (f, d, k), v in sorted(tau.items())]


def bound_set(
    art: Mapping[str, Any], bounds: Mapping[str, Any], tools: RefereeTools
) -> dict[str, Any]:
    """Bound records per method: PMRT and OLS from a bounds file, GT from the artifact."""
    return {
        "pmrt": tools.bounds_from_json(bounds["pmrt"]),
        "ols": tools.bounds_from_json(bounds["ols"]),
        "gt": tools.bounds_from_json(art["bounds"]["gt"]),
    }


def maps_of(doc: Mapping[str, Any] | None) -> dict[str, list] | None:
    """Map rows of the `MAP_OVERRIDE` sources in a maps artifact, or None.

    Only the PMRT map takes the X17 strict coefficients; every other arm keeps the certified-safe
    artifact's map (the TreeSHAP maps included).
    """
    if doc is None:
        return None
    return {a: v["map"] for a, v in doc["arms"].items() if a in MAP_OVERRIDE}


def certify_all(
    art: Mapping[str, Any],
    bset: Mapping[str, Any],
    tau: Mapping,
    tools: RefereeTools,
    maps: Mapping[str, list] | None = None,
) -> dict[str, dict[str, Any]]:
    """Certification, uncertified set and certified-safe signature of every artifact arm.

    The map of an arm is its source's map in `maps` when present, else the artifact's.
    """
    theta = float(art["mapgate"]["theta"])
    out = {}
    for arm in sorted(art["arms"]):
        v = art["arms"][arm]
        rows = maps[v["source"]] if maps is not None and v["source"] in maps else v["map"]
        m = tools.map_from_json(rows)
        meth = v["bound_method"]
        cert = tools.certify_map(m, bset[meth] if meth else None, tau, theta)
        unc = _unc(cert["uncertified"])
        sig = tools.signature_key(tools.certsafe_signature(m, [tuple(x) for x in unc], theta))
        out[arm] = {
            "source": v["source"],
            "bound_method": meth,
            "uncertified": unc,
            "signature": sig,
            "classes": cert["classes"],
            "map": [list(r) for r in rows],
        }
    return out


def reproduce_frozen(
    art: Mapping[str, Any], frozen_bounds: Mapping[str, Any], tools: RefereeTools
) -> dict[str, Any]:
    """Tau, uncertified sets and signatures at rho 1.05 with the frozen bounds vs the artifact."""
    tau = tau_at(art, tools, FROZEN_RHO)
    stored = tools.tau_from_json(art["tau"])
    tau_ok = set(tau) == set(stored) and all(
        math.isclose(tau[k], stored[k], rel_tol=1e-12, abs_tol=0.0) for k in tau
    )
    got = certify_all(art, bound_set(art, frozen_bounds, tools), tau, tools)
    bad = [
        a
        for a, t in got.items()
        if t["uncertified"] != _unc(art["arms"][a]["uncertified"])
        or t["signature"] != art["arms"][a]["signature"]
    ]
    rho_ok = math.isclose(float(art["constants"]["rho"]), FROZEN_RHO, rel_tol=0.0, abs_tol=1e-12)
    return {
        "ok": bool(tau_ok and rho_ok and not bad),
        "tau": tau_ok,
        "rho": rho_ok,
        "mismatch": bad,
        "n_arms": len(got),
    }


def stored_signatures(art: Mapping[str, Any]) -> dict[str, str]:
    """`{signature: stored job arm}` of the Study 3 block; the all-accept signature is `noarb`."""
    out = {art["arms"][j]["signature"]: j for j in art["jobs"] if j in art["arms"]}
    if art.get("noarb_alias"):
        out.setdefault(art["all_accept_signature"], "noarb")
    return out


@dataclasses.dataclass(frozen=True)
class CertInputs:
    """Inputs of Part B.

    Attributes:
        art: The certified-safe artifact.
        frozen_bounds: The frozen `certsafe_bounds.json`.
        bounds: The bounds to certify with (X17 strict).
        maps: The maps artifact whose map replaces each arm's source map, or None.
    """

    art: Mapping[str, Any]
    frozen_bounds: Mapping[str, Any]
    bounds: Mapping[str, Any]
    maps: Mapping[str, Any] | None = None


def run_cert(
    inp: CertInputs, tools: RefereeTools, *, allow_strict_change: bool = False
) -> dict[str, Any]:
    """Part B over `RHO_GRID`.

    Raises:
        ReproductionError: The frozen bounds do not reproduce the artifact at rho 1.05.
        StrictChangeError: A signature at rho 1.05 differs from the stored one and
            `allow_strict_change` is False.
    """
    art = inp.art
    rep = reproduce_frozen(art, inp.frozen_bounds, tools)
    if not rep["ok"]:
        raise ReproductionError("X18 Part B: rho 1.05 does not reproduce: " + json.dumps(rep))
    bset = bound_set(art, inp.bounds, tools)
    maps = maps_of(inp.maps)
    stored = stored_signatures(art)
    frozen_unc = {a: _unc(v["uncertified"]) for a, v in art["arms"].items()}
    by_rho: dict[str, Any] = {}
    sigs: dict[str, dict[str, Any]] = {}
    for rho in RHO_GRID:
        tau = tau_at(art, tools, rho)
        arms = {}
        for arm, t in certify_all(art, bset, tau, tools, maps).items():
            reuse = stored.get(t["signature"])
            sid = None
            if reuse is None:
                sid = sig_id(t["signature"])
                e = sigs.setdefault(
                    sid,
                    {
                        "arm": SIG_ARM + sid,
                        "signature": t["signature"],
                        "map": t["map"],
                        "uncertified": t["uncertified"],
                        "used_by": [],
                    },
                )
                e["used_by"].append([arm, key(rho)])
            arms[arm] = {
                "source": t["source"],
                "bound_method": t["bound_method"],
                "uncertified": t["uncertified"],
                "uncertified_frozen": frozen_unc[arm],
                "changed_from_frozen": t["uncertified"] != frozen_unc[arm],
                "signature": t["signature"],
                "reuse": reuse,
                "sig_id": sid,
                "classes": t["classes"],
            }
        by_rho[key(rho)] = {"rho": rho, "tau": tau_rows(tau), "arms": arms}
    at_frozen = by_rho[key(FROZEN_RHO)]["arms"]
    changed = sorted(
        a for a, t in at_frozen.items() if t["signature"] != art["arms"][a]["signature"]
    )
    if changed and not allow_strict_change:
        raise StrictChangeError(f"X18 Part B: strict signatures at rho 1.05 differ for {changed}")
    return {
        "schema": "x18-certification/1",
        "declared": DECLARED,
        "status": STATUS,
        "repro": rep,
        "strict_at_frozen_rho": {"changed": changed, "allowed": bool(allow_strict_change)},
        "grid": {
            "rho": list(RHO_GRID),
            "rho_of_L": {key(x): rho_of(x) for x in L_GRID},
            "rho_at_frozen_L": list(RHO_AT_FROZEN),
        },
        "theta": float(art["mapgate"]["theta"]),
        "stored_signatures": stored,
        "by_rho": by_rho,
        "signatures": dict(sorted(sigs.items())),
    }


def make_plan(
    cert: Mapping[str, Any], art: Mapping[str, Any], *, allow_over_cap: bool = False
) -> dict[str, Any]:
    """Part C job plan: one CertSafeMapGateV2 job per new signature and the reproduction jobs.

    Raises:
        OverCapError: More than `CAP` new signatures and `allow_over_cap` is False.
    """
    jobs = [
        {"sig_id": k, **{f: v[f] for f in ("arm", "signature", "map", "uncertified", "used_by")}}
        for k, v in sorted(cert["signatures"].items())
    ]
    if len(jobs) > CAP and not allow_over_cap:
        raise OverCapError(
            f"X18 Part C: {len(jobs)} new signatures (cap {CAP}, about "
            f"{len(jobs) * CPU_H_PER_SIG:.1f} CPU-h); rerun with --allow-over-cap after approval"
        )
    src = art["arms"][REPRO_SOURCE]
    n_seeds = EVAL_SEEDS[1] - EVAL_SEEDS[0] + 1
    repro_cpu = CPU_H_PER_SIG * len(REPRO_SEEDS) / n_seeds
    return {
        "schema": "x18-plan/1",
        "declared": DECLARED,
        "status": STATUS,
        "theta": float(art["mapgate"]["theta"]),
        "seeds": list(EVAL_SEEDS),
        "n_seeds": n_seeds,
        "jobs": jobs,
        "repro": {
            "arm": REPRO_ARM,
            "source_arm": REPRO_SOURCE,
            "map": [list(r) for r in src["map"]],
            "uncertified": _unc(src["uncertified"]),
            "signature": src["signature"],
            "seeds": list(REPRO_SEEDS),
        },
        "cpu_h_each": CPU_H_PER_SIG,
        "cpu_h": round(CPU_H_PER_SIG * len(jobs) + repro_cpu, 2),
        "n_jobs": n_seeds * len(jobs) + len(REPRO_SEEDS),
        "cap": CAP,
        "over_cap_allowed": bool(allow_over_cap),
    }


def pooled_all(R: stats.Records, seeds: Sequence[int], arms: Iterable[str]) -> dict[str, Any]:
    """Pooled Gate A quantities of the anchors and `arms` on `seeds`."""
    need = dict.fromkeys([*stats.ANCHORS, *arms])
    return {a: stats.pooled(stats.arrays(R, seeds, a)) for a in need}


def eligibility(P: Mapping[str, Any], arm: str, limit: float, floor: float) -> dict[str, Any]:
    """Eligibility of `arm` at guard limit `limit` and energy floor `floor` (frozen rule)."""
    ef, ea = P["freeze"]["E"], P[stats.A_ARM]["E"]
    matched = bool(ef - P[arm]["E"] >= floor * (ef - ea) - TOL)
    fails = [k for k in stats.GUARD_KEYS if not P[arm][k] <= limit * P["noarb"][k] + TOL]
    return {
        "matched": matched,
        "guard_ok": not fails,
        "guard_fail": fails,
        "eligible": bool(matched and not fails),
    }


def check_frozen_eligibility(st: stats.GateA, P: Mapping[str, Any], arms: Sequence[str]) -> list:
    """Arms whose eligibility at (1.10, 0.90) differs from the frozen Gate A."""
    bad = []
    for a in arms:
        e, p = eligibility(P, a, FROZEN_L, FROZEN_E), st.point[a]
        if (e["matched"], e["guard_ok"], e["eligible"]) != (
            p["matched"],
            p["guard_ok"],
            p["eligible"],
        ):
            bad.append(a)
    return bad


def _lb(x: np.ndarray) -> float:
    fin = x[np.isfinite(x)]
    return float(np.quantile(fin, 0.05)) if fin.size else float("nan")


def cell(
    st: stats.GateA,
    P: Mapping[str, Any],
    arms: Sequence[str],
    limit: float,
    floor: float,
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One (L, E) cell: eligibility, R*, the best eligible static rule and arm, and contrasts.

    `names` renames row keys (a virtual arm to its gated arm); static rules keep their names.
    """
    names = names or {}
    el = {a: eligibility(P, a, limit, floor) for a in arms}
    r = {a: float(st.point[a]["R"]) for a in arms}
    rs = {a: r[a] if el[a]["eligible"] else min(r[a], 0.0) for a in arms}

    def boot_rs(a: str) -> np.ndarray:
        b = st.boot[a]["R"]
        return b if el[a]["eligible"] else np.minimum(b, 0.0)

    st_el = [a for a in STATIC if a in el and el[a]["eligible"]]
    bs = max(st_el, key=lambda a: r[a]) if st_el else None
    elig = [a for a in arms if el[a]["eligible"]]
    best = max(elig, key=lambda a: r[a]) if elig else None
    rows: dict[str, Any] = {}
    for a in arms:
        row = {**el[a], "R": r[a], "R_ci90": stats.ci(st.boot[a]["R"]), "Rstar": rs[a]}
        if names.get(a):
            row["virtual"] = a
        if bs is not None:
            d = st.boot[a]["R"] - st.boot[bs]["R"]
            ds = boot_rs(a) - boot_rs(bs)
            lb = _lb(ds)
            dstar = rs[a] - rs[bs]
            row |= {
                "dR": r[a] - r[bs],
                "dR_ci90": stats.ci(d),
                "dRstar": dstar,
                "dRstar_ci90": stats.ci(ds),
                "dRstar_lb90": lb,
                "beats": bool(np.isfinite(dstar) and dstar > 0 and np.isfinite(lb) and lb > 0),
            }
        rows[names.get(a, a)] = row
    return {
        "L": limit,
        "E": floor,
        "best_static": bs,
        "best_static_R": r[bs] if bs else None,
        "best_arm": names.get(best, best) if best else None,
        "best_arm_R": r[best] if best else None,
        "arms": rows,
    }


@dataclasses.dataclass(frozen=True)
class Block:
    """Alias-filled EVAL records of one block with its stored evaluation and artifact.

    Attributes:
        name: `study3` or `study2`.
        R: Records per seed and arm.
        arms: The stored arms of the block.
        frozen: The stored evaluation (`cs_eval.json` / `conf_eval.json`, TreeSHAP).
        art: The block's artifact (certified-safe or maps).
    """

    name: str
    R: stats.Records
    arms: list[str]
    frozen: Mapping[str, Any]
    art: Mapping[str, Any]


def study3_block(records: Path, artifacts: Path) -> Block:
    """The Study 3 EVAL block with the TreeSHAP jobs spliced in."""
    shap = ShapSources(records=records, experiments=artifacts.parent)
    old = read_json(artifacts / "study3_certsafe.json")
    art = read_json(shap.certsafe_artifact)
    m = merged_records(
        x9.eval_files(records, "e6p-cs-eval-1"),
        shap.eval_files(3),
        study3.job_filter("eval"),
        old,
        art,
    )
    arms = [a for a in study3.ALL_ARMS if any(a in m.R[s] for s in m.R)]
    return Block("study3", m.R, arms, read_json(shap.report(3)[0]), art)


def study2_block(records: Path, artifacts: Path) -> Block:
    """The Study 2 EVAL block with the TreeSHAP jobs spliced in."""
    shap = ShapSources(records=records, experiments=artifacts.parent)
    old = read_json(artifacts / "study2_maps.json")
    art = read_json(shap.maps_artifact)
    m = merged_records(
        x9.eval_files(records, "e6p-conf-eval-1"),
        shap.eval_files(2),
        study2.EVAL_FILTER,
        old,
        art,
    )
    arms = [a for a in study2.ALL_ARMS if any(a in m.R[s] for s in m.R)]
    return Block("study2", m.R, arms, read_json(shap.report(2)[0]), art)


def stored_grid(b: Block, extra: Sequence[str] = ()) -> tuple[stats.GateA, dict, dict[str, Any]]:
    """Gate A of the block (plus `extra` arms), the reproduction and the stored-arm grid.

    Raises:
        ReproductionError: The stored arms do not reproduce the block's evaluation, or the
            eligibility rule at (1.10, 0.90) differs from the frozen one.
    """
    st = stats.gate_a(b.R, [*b.arms, *extra], stats.N_BOOT)
    rp = x9.repro(st, b.frozen, b.arms)
    P = pooled_all(b.R, st.seeds, [*b.arms, *extra])
    bad = check_frozen_eligibility(st, P, b.arms)
    rp["eligibility_rule"] = {"ok": not bad, "mismatch": bad}
    if not (rp["ok"] and not bad):
        raise ReproductionError(f"X18 Part A {b.name}: " + json.dumps(rp, default=str))
    grid = {cell_key(x, e): cell(st, P, b.arms, x, e) for x in L_GRID for e in E_GRID}
    return st, P, {"n_seeds": st.n, "seeds": [st.seeds[0], st.seeds[-1]], "repro": rp, "grid": grid}


def virtual_name(arm: str, rho: float) -> str:
    """Name of a gated arm's records at gate budget `rho`."""
    return f"{arm}@rho{key(rho)}"


def add_gated(
    R: stats.Records, cert: Mapping[str, Any], seeds: Sequence[int], rhos: Iterable[float]
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], list[list[Any]]]:
    """Copies, in place, the records of every gated arm at every `rho` from its target arm.

    The target is the stored job of the arm's signature, or its Part C arm. A target without a
    record on every seed leaves the arm pending.

    Returns:
        `{(arm, rho key): virtual arm}`, `{(arm, rho key): target}` and the pending
        `[arm, rho key, target]` rows.
    """
    virt, target, pending = {}, {}, []
    for rho in rhos:
        rk = key(rho)
        for arm, t in cert["by_rho"][rk]["arms"].items():
            tgt = t["reuse"] or SIG_ARM + t["sig_id"]
            target[(arm, rk)] = tgt
            if not all(tgt in R.get(s, {}) for s in seeds):
                pending.append([arm, rk, tgt])
                continue
            v = virtual_name(arm, rho)
            for s in seeds:
                R[s][v] = dict(R[s][tgt], arm=v, x18_of=tgt)
            virt[(arm, rk)] = v
    return virt, target, pending


def gated_section(
    st: stats.GateA,
    P: Mapping[str, Any],
    base: Sequence[str],
    cert: Mapping[str, Any],
    virt: Mapping[tuple[str, str], str],
    target: Mapping[tuple[str, str], str],
    pairs: Iterable[tuple[float, float, float]],
) -> dict[str, Any]:
    """Cells over `(L, E, rho)`: the non-gated stored arms and every gated arm at `rho`."""
    out = {}
    for limit, floor, rho in pairs:
        rk = key(rho)
        arms_rho = cert["by_rho"][rk]["arms"]
        vs = {virt[(a, rk)]: a for a in arms_rho if (a, rk) in virt}
        c = cell(st, P, [*base, *vs], limit, floor, names=vs)
        for a, t in arms_rho.items():
            row = c["arms"].get(a)
            info = {
                "rho": rho,
                "target": target[(a, rk)],
                "uncertified": t["uncertified"],
                "uncertified_frozen": t["uncertified_frozen"],
                "changed_from_frozen": t["changed_from_frozen"],
            }
            if row is None:
                c.setdefault("pending", []).append(a)
            else:
                row |= info
        c["rho"] = rho
        out[f"{cell_key(limit, floor)}|{rk}"] = c
    return out


def part_c_records(
    files: Sequence[Path], plan: Mapping[str, Any], stored: stats.Records
) -> tuple[stats.Records, dict[str, Any]]:
    """Part C records and their checks: the reproduction jobs and per-arm completeness.

    Raises:
        ReproductionError: A reproduction record differs from the stored CS:PMRT record.
    """
    Rc, heads = stats.load_jobs(files, sub_filter(SUB))
    rep = plan["repro"]
    prov = []
    for s in rep["seeds"]:
        a, b = Rc.get(s, {}).get(rep["arm"]), stored.get(s, {}).get(rep["source_arm"])
        if a is None or b is None:
            prov.append({"seed": s, "present": False})
            continue
        keys = sorted((set(a) | set(b)) - SKIP_COMPARE)
        diff = sorted(k for k in keys if a.get(k) != b.get(k))
        prov.append({"seed": s, "present": True, "identical": not diff, "diff": diff})
    seeds = list(range(plan["seeds"][0], plan["seeds"][1] + 1))
    complete = {j["arm"]: sum(j["arm"] in Rc.get(s, {}) for s in seeds) for j in plan["jobs"]}
    ok = all(p.get("identical") for p in prov)
    chk = {
        "repro": prov,
        "repro_ok": bool(ok),
        "complete": complete,
        "n_headers": len(heads),
        "n_files": len(files),
    }
    if not ok:
        raise ReproductionError("X18 Part C: reproduction jobs differ: " + json.dumps(prov))
    return Rc, chk


def _first_beating(sec: Mapping[str, Any], floor: float) -> dict[str, Any]:
    for limit in L_GRID:
        c = sec[f"{cell_key(limit, floor)}|{key(rho_of(limit))}"]
        win = [a for a in MAP_GATED if a in c["arms"] and c["arms"][a].get("beats")]
        if win:
            a = max(win, key=lambda x: c["arms"][x]["dRstar"])
            row = c["arms"][a]
            return {
                "L": limit,
                "status": "found",
                "arm": a,
                "dRstar": row["dRstar"],
                "dRstar_lb90": row["dRstar_lb90"],
                "best_static": c["best_static"],
                "uncertified": row["uncertified"],
                "uncertified_frozen": row["uncertified_frozen"],
            }
        if any(a in MAP_GATED for a in c.get("pending", [])):
            return {"L": limit, "status": "pending", "pending": c["pending"]}
    return {"L": None, "status": "none in grid"}


def summary(stored3: Mapping[str, Any], coupled: Mapping[str, Any]) -> dict[str, Any]:
    """The items of the reading rule."""
    first = {key(e): _first_beating(coupled, e) for e in E_GRID}
    ovp = {}
    for limit in L_GRID:
        for e in E_GRID:
            c = coupled[f"{cell_key(limit, e)}|{key(rho_of(limit))}"]
            p, g = c["arms"].get("CS:PMRT"), c["arms"].get("CS:GT")
            if p is None or g is None:
                ovp[cell_key(limit, e)] = {"status": "pending"}
                continue
            sel = ("eligible", "beats", "uncertified")
            ovp[cell_key(limit, e)] = {
                "pmrt": {k: p.get(k) for k in sel},
                "oracle": {k: g.get(k) for k in sel},
                "differ": any(p.get(k) != g.get(k) for k in ("eligible", "beats")),
            }
    ns = {
        key(x): [
            e for e in E_GRID if stored3["grid"][cell_key(x, e)]["best_static"] == "never_sleep"
        ]
        for x in L_GRID
    }
    return {
        "first_L_gated_map_beats_best_static": first,
        "oracle_vs_pmrt": ovp,
        "never_sleep_best_static_floors": ns,
    }


def _part_c_files(part_c: Path | None) -> list[Path]:
    return shard_files(part_c) if part_c else []


def run_eligibility(
    records: Path = RECORDS,
    artifacts: Path = ARTIFACTS,
    *,
    cert: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
    part_c: Path | None = None,
) -> dict[str, Any]:
    """Part A, and the gated sections with `cert` (merged with the Part C records if given).

    Raises:
        ReproductionError: A block, or the Part C reproduction, does not reproduce.
        ValueError: Part C records without the plan or the certification.
    """
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    if part_c is not None and (plan is None or cert is None):
        raise ValueError("--part-c needs --plan and --cert")
    b3, b2 = study3_block(records, artifacts), study2_block(records, artifacts)
    _, _, s2 = stored_grid(b2)
    out: dict[str, Any] = {
        "schema": "x18-eligibility/1",
        "declared": DECLARED,
        "status": STATUS,
        "grid": {"L": list(L_GRID), "E": list(E_GRID), "rho_at_frozen_L": list(RHO_AT_FROZEN)},
        "frozen": {"L": FROZEN_L, "E": FROZEN_E, "rho": FROZEN_RHO},
        "static_rules": list(STATIC),
        "n_boot": stats.N_BOOT,
        "boot_key": "default_rng([6624, 20, n_seeds])",
        "study2": s2,
    }
    if cert is None:
        _, _, s3 = stored_grid(b3)
        out["study3"] = s3
        return out
    part_c_chk = None
    if part_c is not None and plan is not None:
        Rc, part_c_chk = part_c_records(_part_c_files(part_c), plan, b3.R)
        for s, by_arm in Rc.items():
            if s in b3.R:
                b3.R[s].update({a: r for a, r in by_arm.items() if a.startswith(SIG_ARM)})
    seeds = sorted(s for s in b3.R if set(stats.ANCHORS) | set(b3.arms) <= set(b3.R[s]))
    virt, target, pending = add_gated(b3.R, cert, seeds, RHO_GRID)
    st, P, s3 = stored_grid(b3, extra=sorted(set(virt.values())))
    out["study3"] = s3
    base = [a for a in b3.arms if a not in study3.CS_ARMS]
    coupled = gated_section(
        st, P, base, cert, virt, target, [(x, e, rho_of(x)) for x in L_GRID for e in E_GRID]
    )
    sweep = gated_section(
        st, P, base, cert, virt, target, [(FROZEN_L, e, r) for r in RHO_AT_FROZEN for e in E_GRID]
    )
    out |= {
        "coupled": coupled,
        "rho_sweep": sweep,
        "pending": pending,
        "part_c": part_c_chk,
        "summary": summary(s3, coupled),
    }
    return out


def _csv_rows(section: str, block: str, cells: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for c in cells.values():
        for a, r in c["arms"].items():
            lo, hi = r["R_ci90"]
            dlo, dhi = r.get("dRstar_ci90") or (None, None)
            yield {
                "section": section,
                "block": block,
                "L": c["L"],
                "E": c["E"],
                "rho": c.get("rho", ""),
                "arm": a,
                "target": r.get("target", ""),
                "eligible": r["eligible"],
                "matched": r["matched"],
                "guard_ok": r["guard_ok"],
                "guard_fail": ";".join(r["guard_fail"]),
                "R": r["R"],
                "R_lo": lo,
                "R_hi": hi,
                "Rstar": r["Rstar"],
                "best_static": c["best_static"] or "",
                "dRstar": r.get("dRstar", ""),
                "dRstar_lo": dlo,
                "dRstar_hi": dhi,
                "beats": r.get("beats", ""),
            }


def write_csv(path: Path, out: Mapping[str, Any]) -> None:
    """Writes every cell row of the eligibility output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, CSV_COLUMNS)
        w.writeheader()
        for blk in ("study3", "study2"):
            w.writerows(_csv_rows("stored", blk, out[blk]["grid"]))
        for sec in ("coupled", "rho_sweep"):
            if sec in out:
                w.writerows(_csv_rows(sec, "study3", out[sec]))


def sections_md(out: Mapping[str, Any]) -> str:
    """The X18 section of the revision report."""
    lines = [
        "## X18: guard limit, energy floor and gate budget",
        "",
        f"{DECLARED}. {STATUS}; the frozen L 1.10, E 0.90 and rho 1.05 stay primary.",
        "",
        "Best eligible static rule, Study 3 block (rows L, columns E):",
        "",
        "| L | " + " | ".join(key(e) for e in E_GRID) + " |",
        "|---|" + "---|" * len(E_GRID),
    ]
    g = out["study3"]["grid"]
    for x in L_GRID:
        cells = [str(g[cell_key(x, e)]["best_static"]) for e in E_GRID]
        lines.append(f"| {key(x)} | " + " | ".join(cells) + " |")
    sm = out.get("summary")
    if sm:
        lines += [
            "",
            "Smallest L at which a gated map referee beats the best eligible static rule:",
            "",
        ]
        for e, v in sm["first_L_gated_map_beats_best_static"].items():
            lines.append(f"- E {e}: {json.dumps(v, default=js_default)}")
        if out.get("pending"):
            lines += ["", f"Pending gated arms (no complete records): {len(out['pending'])}."]
    return "\n".join(lines) + "\n"


def write_eligibility(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x18_eligibility.json`, `x18_eligibility.csv` and `REVISION_SECTIONS_X18.md`."""
    d = Path(out_dir)
    write_text(d / OUT_ELIG, json.dumps(out, indent=1, default=js_default))
    write_csv(d / OUT_CSV, out)
    write_text(d / OUT_MD, sections_md(out))


def _load(path: str | None) -> Any:
    return read_json(Path(path)) if path else None


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cert", help="Part B: certification over the rho grid")
    c.add_argument("--artifacts", default=str(ARTIFACTS))
    c.add_argument("--frozen-bounds", required=True)
    c.add_argument("--bounds", required=True)
    c.add_argument("--maps", default=None)
    c.add_argument("--allow-strict-change", action="store_true")
    c.add_argument("--out", required=True)
    p = sub.add_parser("plan", help="Part C: the job plan of the new signatures")
    p.add_argument("--cert", required=True)
    p.add_argument("--artifacts", default=str(ARTIFACTS))
    p.add_argument("--allow-over-cap", action="store_true")
    p.add_argument("--out", required=True)
    for nm in ("eligibility", "merge"):
        e = sub.add_parser(nm, help="Part A" if nm == "eligibility" else "Part A with Part C")
        e.add_argument("--records", default=str(RECORDS))
        e.add_argument("--artifacts", default=str(ARTIFACTS))
        e.add_argument("--cert", required=nm == "merge", default=None)
        e.add_argument("--plan", required=nm == "merge", default=None)
        e.add_argument("--part-c", required=nm == "merge", default=None)
        e.add_argument("--out", required=True)
    return ap


def main(argv: Sequence[str] | None = None, *, tools: RefereeTools | None = None) -> int:
    """Command line: `cert`, `plan`, `eligibility` or `merge`."""
    a = _parser().parse_args(argv)
    if a.cmd == "cert":
        if tools is None:
            raise SystemExit(
                "cert needs the referee tools (python -m oran_causal.experiments.posthoc x18)"
            )
        art_path = Path(a.artifacts) / CERTSAFE_SHAP
        inp = CertInputs(
            read_json(art_path), _load(a.frozen_bounds), _load(a.bounds), _load(a.maps)
        )
        out = run_cert(inp, tools, allow_strict_change=a.allow_strict_change)
        out["sha256"] = {
            CERTSAFE_SHAP: lf_sha256(art_path),
            "frozen_bounds": lf_sha256(a.frozen_bounds),
            "bounds": lf_sha256(a.bounds),
            "maps": lf_sha256(a.maps) if a.maps else None,
            "oran_causal/analysis/posthoc/x18.py": lf_sha256(Path(__file__)),
        }
        write_text(Path(a.out) / OUT_CERT, json.dumps(out, indent=1, default=js_default))
        print(
            json.dumps(
                {
                    "repro": out["repro"]["ok"],
                    "new_signatures": len(out["signatures"]),
                    "strict_changed": out["strict_at_frozen_rho"]["changed"],
                }
            )
        )
        return 0
    if a.cmd == "plan":
        cert = read_json(Path(a.cert))
        art = read_json(Path(a.artifacts) / CERTSAFE_SHAP)
        plan = make_plan(cert, art, allow_over_cap=a.allow_over_cap)
        plan["cert_sha256"] = lf_sha256(a.cert)
        write_text(Path(a.out) / OUT_PLAN, json.dumps(plan, indent=1, default=js_default))
        print(json.dumps({k: plan[k] for k in ("n_jobs", "cpu_h")} | {"new": len(plan["jobs"])}))
        return 0
    out = run_eligibility(
        Path(a.records),
        Path(a.artifacts),
        cert=_load(a.cert),
        plan=_load(a.plan),
        part_c=Path(a.part_c) if a.part_c else None,
    )
    write_eligibility(out, a.out)
    sm = out.get("summary") or {}
    first = sm.get("first_L_gated_map_beats_best_static")
    print(
        json.dumps({"pending": len(out.get("pending", [])), "first_L": first}, default=js_default)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

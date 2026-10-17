"""The frozen referee artifacts of Studies 2 and 3: build and verify.

- Study 2 maps artifact (`E6P_CONF_MAPS.json`, schema `e6p-conf-maps/1`): every map arm of the
  Study 2 discovery output (`disc_conf.json`) with its provenance, decision signature, alias
  target and decision table; the simulated jobs; the MapGateV2 constants.
- Study 3 certsafe artifact (`E6P_CERTSAFE.json`, schema `e6p-certsafe/1`): per certified-safe
  arm its source map, bound method, certification classes, uncertified classes, signature and
  decision table, built from the guard-edge bounds on the Study 2 DISC units
  (`certsafe_bounds.json`), the DEV calibration (`certsafe_calib.json`) and the GT cells.

Both artifacts pin the code that froze them by sha256. That code is CDD-ORAN's; its hashes are
kept here as the frozen `LEGACY_SHA256` values, so verification checks the pins against them.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.data import UnitTable
from oran_causal.core.gate import DIRS, RHO, Z90
from oran_causal.experiments.system import study2, study3
from oran_causal.experiments.system.common import (
    CERTSAFE_CODE,
    CONF_ANALYZER,
    CONF_DRIVER,
    CS_ANALYZER,
    CS_DRIVER,
    LEGACY_SHA256,
    MAPGATE,
    FrozenDoc,
    Record,
    repro_diff,
    sha_lf,
    to_json,
)
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision.vocab import FAMILIES

SCHEMA_DISC = "e6p-conf-disc/1"
SCHEMA_MAPS = "e6p-conf-maps/1"
SCHEMA_BOUNDS = "e6p-certsafe-bounds/1"
SCHEMA_CALIB = "e6p-certsafe-calib/1"
SCHEMA_CERTSAFE = "e6p-certsafe/1"
V4_ARTIFACT_SHA256 = "9991c390df3a53ef12b553990da52c6adab25d4087bb029f1427e20a898e6a91"
TAU_RULE = "tau(f, d, k) = (rho - 1) * G_k / (max(N(f, d), 1) * 3)"
BETA_RTOL = 1e-9


def dir_key(f: str, d: int) -> str:
    """Decision-table key of a request class: family and "+" / "-"."""
    return f"{f}{'+' if d > 0 else '-'}"


def _decision_table(M: rf.Map) -> dict[str, str]:
    return {dir_key(f, d): v for (f, d), v in rf.decision_table_v2(M, rf.THETA).items()}


def _mapgate(**extra: Any) -> dict[str, Any]:
    return {
        "theta": rf.THETA,
        "k_conf": rf.K_CONF,
        "T": study2.COLLECTION.T,
        "open_rule": rf.OPEN_RULE,
        "warmup_s": 0.0,
        **extra,
        "directional": True,
        "sha256": LEGACY_SHA256[MAPGATE],
    }


def write_artifact(art: Mapping[str, Any], out: Path) -> str:
    """Writes an artifact (indent 1, LF) and its `.sha256` sidecar; returns the LF sha256."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(art, fh, indent=1, default=to_json)
    sha = sha_lf(out)
    out.with_name(out.name + ".sha256").write_text(f"{sha}  {out.name}\n", newline="\n")
    return sha


def read_json(path: Path) -> Any:
    """Parses a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_maps(disc: Mapping[str, Any], disc_sha: str, allow_dry: bool = False) -> Record:
    """The Study 2 maps artifact from a `disc_conf.json` dict.

    Raises:
        ValueError: Not a discovery output, discovery INVALID, failed record checks, or (unless
            `allow_dry`) a non-full analysis or missing map arms.
    """
    if disc.get("schema") != SCHEMA_DISC:
        raise ValueError(f"not a disc_conf.json (schema {disc.get('schema')})")
    if disc["label"]["label"] == "INVALID":
        raise ValueError("discovery label INVALID: the study stops before EVAL")
    if any(disc["p_mismatch"].values()) or not all(v["ok"] for v in disc["record_checks"].values()):
        raise ValueError("record checks failed (probs / p_mismatch)")
    if disc["mode"] != "full" and not allow_dry:
        raise ValueError(f"disc analysis mode {disc['mode']!r}: only a full one can be frozen")
    if disc["missing_arms"] and not allow_dry:
        raise ValueError(f"missing map arms {disc['missing_arms']}")
    maps = {arm: rf.map_from_json(v) for arm, v in disc["maps"].items()}
    noarb_alias = bool(disc["repro"]["noarb_alias"])
    al = rf.alias_table(maps, rf.THETA, noarb_alias)
    arms = {
        arm: {
            "map": rf.map_to_json(maps[arm]),
            "provenance": disc["provenance"].get(arm),
            "signature": al["signature"][arm],
            "alias_of": al["alias_of"][arm],
            "decision_table": _decision_table(maps[arm]),
        }
        for arm in rf.MAP_ARMS
        if arm in maps
    }
    return {
        "schema": SCHEMA_MAPS,
        "mode": disc["mode"],
        "dry": disc["mode"] != "full",
        "protocol": disc["protocol"],
        "discovery_label": disc["label"]["label"],
        "K0": disc["K0"]["pass"],
        "K0n": (disc["K0n"] or {}).get("pass"),
        "arms": arms,
        "jobs": al["jobs"],
        "alias_of": al["alias_of"],
        "noarb_alias": noarb_alias,
        "repro": disc["repro"],
        "all_accept_signature": al["all_accept"],
        "mapgate": _mapgate(
            priority=list(rf.PRIORITY_V2), guards=list(rf.GUARDS_V2), rels=list(rf.RELS_V2)
        ),
        "sha256": {
            "disc_conf.json": disc_sha,
            CONF_ANALYZER: LEGACY_SHA256[CONF_ANALYZER],
            CONF_DRIVER: LEGACY_SHA256[CONF_DRIVER],
            "v4_artifact": disc["artifact"]["sha256"],
            "disc_code": disc["code_sha256"],
        },
        "platform": disc["platform"],
    }


def verify_maps(
    path: Path = study2.MAPS.path,
    disc_json: Path | None = None,
    allow_dry: bool = False,
    doc: FrozenDoc = study2.MAPS,
) -> dict[str, Any]:
    """Re-derives the maps artifact's signatures and aliases and checks every pin (`doc`'s)."""
    a = read_json(path)
    sha = sha_lf(path)
    maps = {arm: rf.map_from_json(v["map"]) for arm, v in a["arms"].items()}
    al = rf.alias_table(maps, float(a["mapgate"]["theta"]), bool(a["noarb_alias"]))
    chk = {
        "schema": a.get("schema") == SCHEMA_MAPS,
        "signatures": all(al["signature"][x] == a["arms"][x]["signature"] for x in maps),
        "alias": al["alias_of"] == a["alias_of"] and al["jobs"] == a["jobs"],
        "arms_complete": sorted(maps) == sorted(rf.MAP_ARMS),
        "mapgate_sha": a["mapgate"]["sha256"] == LEGACY_SHA256[MAPGATE],
        "mapgate_consts": (a["mapgate"]["theta"], a["mapgate"]["k_conf"]) == (rf.THETA, rf.K_CONF),
        "analyzer_sha": a["sha256"][CONF_ANALYZER] == LEGACY_SHA256[CONF_ANALYZER],
        "driver_sha": a["sha256"][CONF_DRIVER] == LEGACY_SHA256[CONF_DRIVER],
        "v4_artifact_sha": a["sha256"]["v4_artifact"] == V4_ARTIFACT_SHA256,
        "mode_full": a["mode"] == "full" or allow_dry,
        "not_invalid": a["discovery_label"] != "INVALID",
    }
    if disc_json is not None:
        d = read_json(disc_json)
        chk["disc_sha"] = a["sha256"]["disc_conf.json"] == sha_lf(disc_json)
        chk["disc_maps"] = all(rf.map_from_json(d["maps"][x]) == maps[x] for x in maps)
    return {
        "path": str(path),
        "sha256": sha,
        "checks": chk,
        "ok": all(chk.values()),
        "pinned": doc.want,
        "matches_pin": sha == doc.want,
    }


def maps_from_artifact(path: Path = study2.MAPS.path) -> dict[str, rf.Map]:
    """The source maps {arm: M} of the Study 2 maps artifact."""
    return {arm: rf.map_from_json(v["map"]) for arm, v in read_json(path)["arms"].items()}


def _repro_rows(ud: UnitTable, maps: Mapping[str, rf.Map]) -> dict[str, Any]:
    rows: dict[str, list] = {"pmrt": [], "ols": []}
    for h, b in maps.get("MG:PMRT", {}).items():
        if b is not None:
            got = abs(rf.design_slope_bound(ud, *h)["beta"])
            rows["pmrt"].append([*h, abs(b), got, bool(np.isclose(got, abs(b), rtol=BETA_RTOL))])
    for arm in rf.ASSOC_ARMS:
        for h, b in maps.get(arm, {}).items():
            if b is not None:
                got = abs(rf.naive_slope_se(ud, *h)["beta"])
                close = bool(np.isclose(got, abs(b), rtol=BETA_RTOL))
                rows["ols"].append([arm, *h, abs(b), got, close])
    ok = all(r[-1] for r in rows["pmrt"]) and all(r[-1] for r in rows["ols"]) and bool(rows["pmrt"])
    return dict(rows, ok=ok)


def bounds_tables(ud: UnitTable, maps: Mapping[str, rf.Map]) -> dict[str, Any]:
    """Guard-edge bounds on the DISC units: PMRT design slope and naive OLS slope, each with SE.

    Also checks that the unit table reproduces every frozen map |beta| (relative 1e-9).
    """
    keys = [(f, r, k) for f in rf.FAMILIES for r in rf.RELS_V2 for k in rf.GUARD_KPIS]
    return {
        "pmrt": {h: rf.design_slope_bound(ud, *h) for h in keys},
        "ols": {h: rf.naive_slope_se(ud, *h) for h in keys},
        "repro": _repro_rows(ud, maps),
    }


def bounds_report(
    ud: UnitTable,
    maps: Mapping[str, rf.Map],
    disc: Mapping[str, Any],
    sha: Mapping[str, str],
    full: bool,
    platform: Any = None,
) -> Record:
    """The `certsafe_bounds.json` content of a DISC unit table.

    Args:
        ud: DISC unit table (H 90 / H_pre 90).
        maps: The Study 2 maps artifact's maps.
        disc: The Study 2 discovery output (for the PMRT z comparison).
        sha: sha256 of the maps artifact and of `disc_conf.json` (keys `maps_artifact`,
            `disc_conf.json`).
        full: Whether every freeze precondition holds.
        platform: The host's numeric environment.
    """
    tb = bounds_tables(ud, maps)
    seeds = sorted({int(s) for s in np.unique(ud.seed)})
    z_cmp = []
    for (f, rel, k), b in tb["pmrt"].items():
        hy = disc["disc_hyp"].get(f"{f}|{rel}|{k}", {})
        z_cmp.append(
            [
                f,
                rel,
                k,
                b.get("z"),
                (hy.get("plain_c") or {}).get("z"),
                (hy.get("loadsp_c") or {}).get("z"),
            ]
        )
    return {
        "schema": SCHEMA_BOUNDS,
        "mode": "full" if full else "partial",
        "n_units": int(ud.n),
        "n_episodes": len(seeds),
        "seeds": [seeds[0], seeds[-1]] if seeds else None,
        "p_mismatch": ud.meta.get("p_mismatch"),
        "pmrt": rf.bounds_to_json(tb["pmrt"]),
        "ols": rf.bounds_to_json(tb["ols"]),
        "beta_repro": tb["repro"],
        "z_compare": {
            "cols": ["family", "rel", "kpi", "z_design_slope", "z_plain_c", "z_loadsp_c"],
            "rows": z_cmp,
        },
        "sha256": {
            "maps_artifact": sha["maps_artifact"],
            "disc_conf.json": sha["disc_conf.json"],
            CERTSAFE_CODE: LEGACY_SHA256[CERTSAFE_CODE],
            CS_ANALYZER: LEGACY_SHA256[CS_ANALYZER],
        },
        "platform": platform,
    }


def load_jobs(
    records: Sequence[Mapping[str, Any]], stage: str, allow_smoke: bool = False
) -> dict[int, dict[str, Record]]:
    """{seed: {arm: record}} of Study 3 job records of `stage`."""
    R: dict[int, dict[str, Record]] = {}
    for r in records:
        if (
            r.get("kind") != "job"
            or r.get("schema") != study2.SCHEMA_EVAL
            or r.get("sub") != study3.SUB
            or r.get("cs_stage") != stage
            or (r.get("smoke") and not allow_smoke)
        ):
            continue
        R.setdefault(int(r["seed"]), {})[r["arm"]] = dict(r)
    return R


def _policy_count(r: Mapping[str, Any], key: str) -> float:
    return float(((r.get("policy_counts") or {}).get("policy") or {}).get(key, 0))


def calibrate(R: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """G_k, N(f, d) and the repro check from Study 3 DEV jobs {seed: {arm: record}}.

    G_k is the mean scored guard total of `CAL:allaccept`; N(f, d) its mean unit count per
    request class; the repro check compares it with `noarb` on the repro seeds.
    """
    seeds = sorted(s for s in R if rf.CAL_ARM in R[s])
    cal = [R[s][rf.CAL_ARM] for s in seeds]
    G = (
        {k: float(np.mean([float(r[rf.GUARD_FIELD[k]]) for r in cal])) for k in rf.GUARD_KPIS}
        if cal
        else {}
    )
    N = {
        (f, d): float(np.mean([_policy_count(r, f"dir_{f}_{d:+d}") for r in cal])) if cal else 0.0
        for f in FAMILIES
        for d in DIRS
    }
    rep = {
        s: repro_diff(R[s][rf.CAL_ARM], R[s]["noarb"])
        for s in study3.REPRO_SEEDS
        if s in R and rf.CAL_ARM in R[s] and "noarb" in R[s]
    }
    ok = len(rep) == len(study3.REPRO_SEEDS) and all(v == 0 for v in rep.values())
    return {
        "seeds": seeds,
        "n": len(seeds),
        "G": G,
        "N": [[f, d, v] for (f, d), v in sorted(N.items())],
        "repro": {"per_seed": rep, "ok": ok},
        "noarb_alias": bool(ok),
    }


def build_certsafe(
    bounds: Mapping[str, Any],
    calib: Mapping[str, Any],
    disc: Mapping[str, Any],
    sha: Mapping[str, str],
    maps_doc: FrozenDoc = study2.MAPS,
    protocol: FrozenDoc = study3.PROTOCOL,
    allow_partial: bool = False,
) -> Record:
    """The Study 3 certsafe artifact.

    Args:
        bounds: `certsafe_bounds.json`.
        calib: `certsafe_calib.json`.
        disc: Study 2 `disc_conf.json` (its GT cells give the GT bounds).
        sha: LF sha256 of the three inputs, keyed by their file names.
        maps_doc: The pinned Study 2 maps artifact.
        protocol: The Study 3 protocol document.
        allow_partial: Accept partial inputs (never a frozen artifact).

    Raises:
        ValueError: Wrong inputs, partial inputs, maps or discovery output other than the
            frozen ones, discovery INVALID or an unfrozen protocol.
    """
    if bounds.get("schema") != SCHEMA_BOUNDS or calib.get("schema") != SCHEMA_CALIB:
        raise ValueError("not a certsafe_bounds.json / certsafe_calib.json")
    if not bounds["beta_repro"]["ok"] and not allow_partial:
        raise ValueError("bounds: the DISC unit table does not reproduce the frozen map betas")
    if (bounds["mode"] != "full" or calib["mode"] != "full") and not allow_partial:
        raise ValueError(f"partial bounds ({bounds['mode']}) / calib ({calib['mode']})")
    maps_sha = sha_lf(maps_doc.path)
    pinned = read_json(maps_doc.path)["sha256"]["disc_conf.json"]
    if not allow_partial and (
        maps_sha != maps_doc.want
        or sha["disc_conf.json"] != pinned
        or bounds["sha256"]["maps_artifact"] != maps_sha
    ):
        raise ValueError("maps artifact / disc_conf.json differ from the frozen Study 2 ones")
    if disc["label"]["label"] == "INVALID":
        raise ValueError("discovery INVALID")
    fz = protocol.status()
    if not fz["frozen"] and not allow_partial:
        raise ValueError(f"the protocol {protocol.name} is not frozen")
    tau = rf.tau_from_accept_all(calib["G"], {(f, int(d)): v for f, d, v in calib["N"]})
    bset = {
        "pmrt": rf.bounds_from_json(bounds["pmrt"]),
        "ols": rf.bounds_from_json(bounds["ols"]),
        "gt": rf.gt_ci_bounds(disc["gt"]["cells"]),
    }
    tables = rf.cs_tables(maps_from_artifact(maps_doc.path), bset, tau)
    al = rf.cs_alias_table(tables, rf.THETA, bool(calib["noarb_alias"]))
    arms = {
        arm: {
            "source": t["source"],
            "map": rf.map_to_json(t["map"]),
            "bound_method": t["bound_method"],
            "certification": t["cert"]["classes"],
            "uncertified": [list(x) for x in t["uncertified"]],
            "signature": t["signature"],
            "alias_of": al["alias_of"][arm],
            "decision_table": t["decision_table"],
        }
        for arm, t in tables.items()
    }
    return {
        "schema": SCHEMA_CERTSAFE,
        "mode": "full" if bounds["mode"] == calib["mode"] == "full" else "partial",
        "protocol": fz,
        "maps_artifact": {"path": maps_doc.name, "sha256": maps_sha},
        "constants": {
            "rho": RHO,
            "z90": Z90,
            "guard_kpis": list(rf.GUARD_KPIS),
            "rels": list(rf.RELS_V2),
            "guard_field": rf.GUARD_FIELD,
            "harm_direction": "deferral: h = -d",
            "tau_rule": TAU_RULE,
        },
        "calib": {
            "G": calib["G"],
            "N": calib["N"],
            "seeds": [calib["seeds"][0], calib["seeds"][-1]] if calib["seeds"] else None,
            "repro": calib["repro"],
        },
        "tau": rf.tau_to_json(tau),
        "bounds": {k: rf.bounds_to_json(v) for k, v in bset.items()},
        "arms": arms,
        "jobs": al["jobs"],
        "alias_of": al["alias_of"],
        "noarb_alias": al["noarb_alias"],
        "all_accept_signature": al["all_accept"],
        "mapgate": _mapgate(),
        "sha256": dict(
            sha,
            **{k: LEGACY_SHA256[k] for k in (CERTSAFE_CODE, CS_ANALYZER, CS_DRIVER)},
        ),
        "platform": {"bounds": bounds.get("platform"), "calib": calib.get("platform")},
    }


def verify_certsafe(
    path: Path = study3.CERTSAFE.path,
    maps_doc: FrozenDoc = study2.MAPS,
    doc: FrozenDoc = study3.CERTSAFE,
) -> dict[str, Any]:
    """Re-derives the certsafe artifact's tables, tau and aliases and checks every pin.

    `maps_doc` is the maps artifact it is built on and `doc` its own pin.
    """
    a = read_json(path)
    sha = sha_lf(path)
    loaded = rf.load_certsafe(a)
    tau = rf.tau_from_json(a["tau"])
    bset = {k: rf.bounds_from_json(v) for k, v in a["bounds"].items()}
    maps = {v["source"]: rf.map_from_json(v["map"]) for v in a["arms"].values()}
    re_t = rf.cs_tables(maps, bset, tau)
    calib_n = {(f, int(d)): v for f, d, v in a["calib"]["N"]}
    chk = {
        "schema": a.get("schema") == SCHEMA_CERTSAFE,
        "uncertified": all(
            sorted(map(list, re_t[x]["uncertified"])) == sorted(a["arms"][x]["uncertified"])
            for x in a["arms"]
        ),
        "tau_rule": rf.tau_to_json(rf.tau_from_accept_all(a["calib"]["G"], calib_n)) == a["tau"],
        "arms_complete": sorted(a["arms"]) == sorted(rf.CS_ARMS),
        "maps_artifact": a["maps_artifact"]["sha256"] == maps_doc.want == sha_lf(maps_doc.path),
        "maps_equal": all(
            maps[k] == m for k, m in maps_from_artifact(maps_doc.path).items() if k in maps
        ),
        "mapgate_sha": a["mapgate"]["sha256"] == LEGACY_SHA256[MAPGATE],
        "certsafe_sha": a["sha256"][CERTSAFE_CODE] == LEGACY_SHA256[CERTSAFE_CODE],
        "analyzer_sha": a["sha256"][CS_ANALYZER] == LEGACY_SHA256[CS_ANALYZER],
        "driver_sha": a["sha256"][CS_DRIVER] == LEGACY_SHA256[CS_DRIVER],
        "mode_full": a["mode"] == "full",
        "jobs": loaded["alias"]["jobs"] == a["jobs"],
    }
    return {
        "path": str(path),
        "sha256": sha,
        "checks": chk,
        "ok": all(chk.values()),
        "pinned": doc.want,
        "matches_pin": sha == doc.want,
    }


def platform_key(env: Mapping[str, Any] | None) -> tuple:
    """Numerics-relevant part of a header's numeric env: numpy, scipy, OS / glibc, machine, SIMD."""
    env = env or {}
    plat = str(env.get("platform", ""))
    glibc = plat.rsplit("-with-", maxsplit=1)[-1] if "-with-" in plat else None
    return (
        env.get("numpy"),
        env.get("scipy"),
        plat.split("-", maxsplit=1)[0],
        glibc,
        env.get("machine"),
        tuple(env.get("simd") or ()),
    )


def calib_report(records: Sequence[Mapping[str, Any]], allow_smoke: bool = False) -> Record:
    """The `certsafe_calib.json` content of the Study 3 DEV records (jobs and run headers).

    Full iff every DEV seed ran, nothing is smoke, the repro check holds and every header
    reports the same numerics.
    """
    R = load_jobs(records, "dev", allow_smoke)
    cb = calibrate(R)
    smoke = any(r.get("smoke") for s in R for r in R[s].values())
    plats = {
        platform_key(h.get("platform") or h.get("numeric_env"))
        for h in records
        if h.get("kind") == "header"
        and h.get("driver") in ("e6p_certsafe", "study3")
        and h.get("cs_stage", h.get("stage")) == "dev"
        and not h.get("smoke")
    }
    full = (
        cb["seeds"] == study3.stage_seeds("dev")
        and not smoke
        and cb["repro"]["ok"]
        and len(plats) == 1
    )
    tau = (
        rf.tau_to_json(rf.tau_from_accept_all(cb["G"], {(f, d): v for f, d, v in cb["N"]}))
        if cb["G"]
        else None
    )
    return {
        "schema": SCHEMA_CALIB,
        "mode": "full" if full else "partial",
        **cb,
        "platform": {"keys": sorted(map(list, plats), key=str), "equal": len(plats) == 1},
        "tau_preview": tau,
    }

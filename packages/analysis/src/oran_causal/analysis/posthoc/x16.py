"""X16: a strictly predictable centre for the PMRT design slope and the gate's PMRT bounds.

Post hoc re-analysis of stored data; no frozen file changes. The data are the DISC unit table
exactly as the certified-safe bounds step builds it (episodes 186100-186699, stage eval, windows
H 90 and H_pre 90).

The frozen centre is `core.ridge.predictable_residuals`: unit j enters the running state right
after it is scored. The strict centre is identical, except that unit j enters the state used for
unit u only once its outcome window has closed, `t0_j + H <= t0_u`. Episodes are separate
simulations with their own clocks, taken in the fixed table order, so every unit of an earlier
episode has a closed window when a later episode starts and the `t0` comparison applies within an
episode.

The reproduction check is a hard gate on the reading: the frozen slopes must give the stored map
betas, the stored gate bounds and the stored uncertified set, and the strict centre with a zero
window must equal the frozen one.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.posthoc.paths import ARTIFACTS, RECORDS, lf_sha256, write_text
from oran_causal.analysis.posthoc.refs import RefereeTools
from oran_causal.core.data import UnitTable
from oran_causal.core.design import v_design
from oran_causal.core.gate import DIRS, bound_ub
from oran_causal.core.pmrt.units import unit_order
from oran_causal.core.ridge import ResidualConfig, predictable_residuals
from oran_causal.sim.system.decision.records import build_cache, load_pool
from oran_causal.sim.system.decision.vocab import FAMILIES

H = 90
RTOL = 1e-9
CERTSAFE_ART = "E6P_CERTSAFE.json"
MAPS_ART = "E6P_CONF_MAPS.json"
PROTOCOL = "scratchpad/xmethod/EXTRAS_PROTOCOL.md 2026-10-12: X16"
DISC_SETS = ("e6p-conf-disc-1-a", "e6p-conf-disc-1-b")
CSV_COLUMNS = (
    "table",
    "family",
    "rel",
    "kpi",
    "d",
    "declared_sign",
    "beta_frozen",
    "beta_strict",
    "rel_change",
    "se_frozen",
    "se_strict",
    "z_frozen",
    "z_strict",
    "ub_frozen",
    "ub_strict",
    "tau",
    "dub_over_tau",
    "cert_frozen",
    "cert_strict",
    "sign_agree_strict",
    "sigma_last_frozen",
    "sigma_last_strict",
)

Edge = tuple[str, str, str]


@dataclasses.dataclass
class _State:
    """Running state of the strict centre."""

    strata: dict[tuple[int, float], tuple[int, float, float]] = dataclasses.field(
        default_factory=dict
    )
    pool: dict[float, list[float]] = dataclasses.field(
        default_factory=lambda: {1.0: [0, 0.0, 0.0], -1.0: [0, 0.0, 0.0]}
    )
    cxy: float = 0.0
    cpp: float = 0.0
    r2: float = 0.0
    nr: int = 0


def strict_residuals(
    y: np.ndarray,
    pre: np.ndarray | None,
    episode: np.ndarray,
    sgn: np.ndarray,
    t0: np.ndarray,
    cfg: ResidualConfig,
    h: int = H,
) -> tuple[np.ndarray, np.ndarray]:
    """`predictable_residuals` with delayed entry.

    Unit j joins the running state used for unit u only if `t0_j + h <= t0_u` in the same episode,
    or j is in an earlier episode. Strata, shrinkage, pooled slope, ridge and scale are as in the
    frozen function. Pending units enter in information order, which is monotone in the entry time
    `t0_j + h` within an episode.

    Args:
        y: Outcomes in information order.
        pre: Pre-window values, or None.
        episode: Episode id per unit.
        sgn: Request direction per unit.
        t0: Opening second per unit.
        cfg: Running-residual settings.
        h: Outcome window in seconds.

    Returns:
        (r, sigma), each of shape `[n]`.
    """
    y = np.asarray(y, float)
    pre_v = np.zeros_like(y) if (pre is None or not cfg.use_pre) else np.asarray(pre, float)
    n = len(y)
    res = np.zeros(n)
    sigs = np.full(n, np.nan)
    st = _State()
    k = float(cfg.prior_n)
    pending: list[int] = []

    def enter(j: int) -> None:
        g = 1.0 if sgn[j] > 0 else -1.0
        key = (int(episode[j]), g)
        ns, my, mp = st.strata.get(key, (0, 0.0, 0.0))
        pn, py, pp = st.pool[g]
        if pn >= cfg.burn:
            rc = min(abs(res[j]), 5.0 * np.sqrt(st.r2 / st.nr)) if st.nr >= 5 else abs(res[j])
            st.r2 += rc * rc
            st.nr += 1
        if ns > 0:
            f = ns / (ns + 1.0)
            st.cxy += f * (pre_v[j] - mp) * (y[j] - my)
            st.cpp += f * (pre_v[j] - mp) ** 2
        st.strata[key] = (
            ns + 1,
            my + (y[j] - my) / (ns + 1),
            mp + (pre_v[j] - mp) / (ns + 1),
        )
        st.pool[g] = [pn + 1, py + (y[j] - py) / (pn + 1), pp + (pre_v[j] - pp) / (pn + 1)]

    for u in range(n):
        while pending and (
            int(episode[pending[0]]) != int(episode[u]) or t0[pending[0]] + h <= t0[u]
        ):
            enter(pending.pop(0))
        g = 1.0 if sgn[u] > 0 else -1.0
        ns, my, mp = st.strata.get((int(episode[u]), g), (0, 0.0, 0.0))
        pn, py, pp = st.pool[g]
        if pn == 0:
            a_y, a_p = (my, mp) if ns else (y[u], pre_v[u])
        elif ns + k <= 0:
            a_y, a_p = py, pp
        else:
            a_y = (ns * my + k * py) / (ns + k)
            a_p = (ns * mp + k * pp) / (ns + k)
        cxy, cpp = st.cxy, st.cpp
        b = cxy / (cpp + cfg.ridge * max(cpp, 1e-300) + 1e-300) if cpp > 0 else 0.0
        res[u] = y[u] - a_y - b * (pre_v[u] - a_p)
        if st.nr > 0:
            sigs[u] = np.sqrt(st.r2 / st.nr)
        pending.append(u)
    return res, sigs


def _no_slope(n: int) -> dict[str, Any]:
    return {"kind": "slope", "beta": 0.0, "se": None, "n": n, "z": None}


def slope_se(ud: UnitTable, edge: Edge, *, strict: bool, tools: RefereeTools) -> dict[str, Any]:
    """PMRT design slope of a guard or map edge under the frozen or the strict centre.

    The frozen slope is the referee's own `design_slope`, plus the scale of its last unit.
    """
    f, rel, kpi = edge
    rows = unit_order(ud, ud.rows_of(f))
    cfg = ResidualConfig()
    pre = ud.pre[(rel, kpi)][rows]
    pre_used = pre if np.all(np.isfinite(pre)) else None
    if not strict:
        out = tools.design_slope(ud, f, rel, kpi)
        if len(rows) >= 2:
            _, sig = predictable_residuals(
                ud.y[(rel, kpi)][rows], pre_used, ud.episode[rows], ud.sgn[rows], cfg
            )
            fs = sig[np.isfinite(sig)]
            out = dict(out, sigma_last=float(fs[-1]) if fs.size else None)
        return out
    if len(rows) < 2:
        return _no_slope(len(rows))
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows], tools.levels)
    r, sig = strict_residuals(
        ud.y[(rel, kpi)][rows], pre_used, ud.episode[rows], ud.sgn[rows], ud.t0[rows], cfg
    )
    vv = float(v @ v)
    if vv <= 0:
        return _no_slope(len(rows))
    beta = float(v @ r / vv)
    e = r - beta * v
    se = float(np.sqrt(np.sum(v * v * e * e)) / vv)
    fin = sig[np.isfinite(sig)]
    return {
        "kind": "slope",
        "beta": beta,
        "se": se,
        "n": len(rows),
        "z": beta / se if se > 0 else None,
        "sigma_last": float(fin[-1]) if fin.size else None,
    }


def close(a: Any, b: Any) -> bool:
    """Equal at relative tolerance `RTOL`; False if either is missing."""
    return (
        a is not None
        and b is not None
        and bool(np.isclose(float(a), float(b), rtol=RTOL, atol=0.0))
    )


def rel_change(new: Any, old: Any) -> float | None:
    """(new - old) / |old|, None when old is missing or zero."""
    return (float(new) - float(old)) / abs(float(old)) if old not in (None, 0) else None


def fin(x: float | None) -> float | None:
    """Float, None for a missing or non-finite value."""
    return None if x is None or not math.isfinite(x) else float(x)


def disc_files(records: Path = RECORDS) -> list[Path]:
    """The DISC shard files (`res_*.jsonl`) of the two Study 2 discovery kernels, in order."""
    return [p for s in DISC_SETS for p in sorted((records / "study2_raw" / s).glob("res_*.jsonl"))]


def unit_table(files: Sequence[Path], cache_dir: Path) -> UnitTable:
    """The DISC unit table, built as the bounds step builds it.

    Each shard becomes one cache of its eval episodes (a shard without any is skipped); the pool
    of the caches at H = H_pre = 90 gives the table.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    caches = []
    for i, f in enumerate(files):
        h = hashlib.sha256(str(f.absolute()).encode()).hexdigest()[:10]
        dst = cache_dir / f"disc_{i:03d}_{h}.npz"
        if not dst.exists():
            try:
                build_cache([str(f)], dst, stages={"eval"}, src=f.name)
            except ValueError:
                continue
        caches.append(str(dst))
    pool = load_pool(sorted(caches), H=H, H_pre=H, stages={"eval"})
    return pool.unit_data()


def _check(
    ud: UnitTable,
    keys: Sequence[Edge],
    slopes: Mapping[str, Mapping[Edge, Mapping[str, Any]]],
    m: Mapping[Edge, float | None],
    stored: Mapping[Edge, Mapping[str, Any]],
    unc_stored: list,
    cert_frozen: Mapping[str, Any],
    stored_units: int | None,
    cfg: ResidualConfig,
) -> dict[str, Any]:
    seeds = sorted({int(s) for s in np.unique(ud.seed)})
    rep_map = [
        [
            *h,
            abs(b),
            abs(slopes["frozen"][h]["beta"]),
            close(abs(slopes["frozen"][h]["beta"]), abs(b)),
        ]
        for h, b in sorted(m.items())
        if b is not None
    ]
    rep_bnd = []
    for h, sb in sorted(stored.items()):
        g = slopes["frozen"][h]
        ok = close(g["beta"], sb["beta"]) and close(g["se"], sb["se"])
        rep_bnd.append([*h, sb["beta"], g["beta"], sb["se"], g["se"], ok])
    d0 = 0.0
    for h in keys:
        rows = unit_order(ud, ud.rows_of(h[0]))
        y, pre = ud.y[(h[1], h[2])][rows], ud.pre[(h[1], h[2])][rows]
        pre = pre if np.all(np.isfinite(pre)) else None
        args = (y, pre, ud.episode[rows], ud.sgn[rows])
        r0, _ = predictable_residuals(*args, cfg)
        r1, _ = strict_residuals(*args, ud.t0[rows], cfg, h=0)
        d0 = max(d0, float(np.max(np.abs(r1 - r0))) if len(r0) else 0.0)
    out = {
        "map_beta": rep_map,
        "bounds": rep_bnd,
        "map_ok": all(r[-1] for r in rep_map) and bool(rep_map),
        "bounds_ok": all(r[-1] for r in rep_bnd),
        "n_units": int(ud.n),
        "n_units_stored": stored_units,
        "episodes": [seeds[0], seeds[-1], len(seeds)],
        "p_mismatch": ud.meta.get("p_mismatch"),
        "uncertified_recomputed": sorted(map(list, cert_frozen["uncertified"])),
        "uncertified_stored": unc_stored,
        "strict_h0_max_abs_diff": d0,
    }
    out["ok"] = (
        out["map_ok"]
        and out["bounds_ok"]
        and out["n_units"] == out["n_units_stored"]
        and out["uncertified_recomputed"] == unc_stored
        and d0 == 0.0
    )
    return out


def _map_rows(m: Mapping[Edge, float | None], slopes: Mapping) -> list[dict[str, Any]]:
    rows = []
    for h, b in sorted(m.items()):
        if b is None:
            raise ValueError(f"map edge {h} has no declared value")
        bf, bs = slopes["frozen"][h]["beta"], slopes["strict"][h]["beta"]
        s = int(np.sign(b))
        rows.append(
            {
                "family": h[0],
                "rel": h[1],
                "kpi": h[2],
                "declared_sign": s,
                "map_value": b,
                "beta_frozen": bf,
                "beta_strict": bs,
                "rel_change": rel_change(bs, bf),
                "abs_rel_change_abs": rel_change(abs(bs), abs(bf)),
                "sign_agree_frozen": bool(np.sign(bf) == s),
                "sign_agree_strict": bool(np.sign(bs) == s),
            }
        )
    return rows


def _gate_rows(
    m: Mapping[Edge, float | None], slopes: Mapping, tau: Mapping, tools: RefereeTools
) -> list[dict[str, Any]]:
    rows = []
    for f in FAMILIES:
        for rel in tools.relations:
            for k in tools.guard_kpis:
                h3 = (f, rel, k)
                bf, bs = slopes["frozen"][h3], slopes["strict"][h3]
                for d in DIRS:
                    hh = -d
                    uf, us = bound_ub(bf, hh), bound_ub(bs, hh)
                    t = tau[(f, d, k)]
                    both = math.isfinite(uf) and math.isfinite(us)
                    rows.append(
                        {
                            "family": f,
                            "rel": rel,
                            "kpi": k,
                            "d": d,
                            "h": hh,
                            "declared": h3 in m,
                            "beta_frozen": bf["beta"],
                            "se_frozen": bf["se"],
                            "z_frozen": bf["z"],
                            "beta_strict": bs["beta"],
                            "se_strict": bs["se"],
                            "z_strict": bs["z"],
                            "ub_frozen": fin(uf),
                            "ub_strict": fin(us),
                            "tau": t,
                            "sigma_last_frozen": bf.get("sigma_last"),
                            "sigma_last_strict": bs.get("sigma_last"),
                            "dub_over_tau": fin((us - uf) / t) if both else None,
                            "cert_frozen": uf <= t,
                            "cert_strict": us <= t,
                        }
                    )
    return rows


def _certification(
    m: Mapping[Edge, float | None],
    cert: Mapping[str, Mapping[str, Any]],
    art: Mapping[str, Any],
    theta: float,
    tools: RefereeTools,
) -> tuple[dict[str, Any], list]:
    changes, edge_changes = [], []
    for cd in cert["frozen"]["classes"]:
        a0, a1 = cert["frozen"]["classes"][cd], cert["strict"]["classes"][cd]
        if a0["certified"] != a1["certified"]:
            changes.append(
                [
                    cd,
                    "certified" if a0["certified"] else "uncertified",
                    "certified" if a1["certified"] else "uncertified",
                ]
            )
        for e0, e1 in zip(a0["edges"], a1["edges"], strict=True):
            if e0[2] != e1[2]:
                edge_changes.append(
                    {
                        "class": cd,
                        "rel": e0[0],
                        "kpi": e0[1],
                        "frozen": e0[2],
                        "strict": e1[2],
                        "ub_frozen": e0[3],
                        "ub_strict": e1[3],
                        "tau": e0[4],
                    }
                )
    sig = {
        c: tools.signature_key(
            tools.certsafe_signature(m, [tuple(x) for x in cert[c]["uncertified"]], theta)
        )
        for c in cert
    }
    stored_sig = art["arms"]["CS:PMRT"]["signature"]
    return {
        "status_changes": changes,
        "edge_status_changes": edge_changes,
        "uncertified": {c: cert[c]["uncertified"] for c in cert},
        "signature": sig,
        "signature_stored": stored_sig,
        "signature_equal": sig["frozen"] == sig["strict"],
        "frozen_signature_reproduces_stored": sig["frozen"] == stored_sig,
        "classes": {c: cert[c]["classes"] for c in cert},
    }, changes


def analyse(
    ud: UnitTable,
    certsafe_art: Mapping[str, Any],
    maps_art: Mapping[str, Any],
    tools: RefereeTools,
    *,
    stored_units: int | None,
) -> dict[str, Any]:
    """The X16 comparison of the frozen and the strict centre.

    Args:
        ud: The DISC unit table.
        certsafe_art: The frozen certified-safe artifact.
        maps_art: The frozen Study 2 maps artifact.
        tools: Referee functions.
        stored_units: Unit count of the stored bounds run.

    Returns:
        The result tree (without the provenance stamps `write` adds).
    """
    stored = tools.bounds_from_json(certsafe_art["bounds"]["pmrt"]) or {}
    tau = tools.tau_from_json(certsafe_art["tau"])
    theta = float(certsafe_art["mapgate"]["theta"])
    m = tools.map_from_json(maps_art["arms"]["MG:PMRT"]["map"])
    if tools.map_from_json(certsafe_art["arms"]["CS:PMRT"]["map"]) != m:
        raise ValueError("the certified-safe PMRT map differs from the Study 2 PMRT map")
    guards = {(f, rel, k) for f in FAMILIES for rel in tools.relations for k in tools.guard_kpis}
    keys = sorted(set(m) | guards)
    slopes = {
        c: {h: slope_se(ud, h, strict=(c == "strict"), tools=tools) for h in keys}
        for c in ("frozen", "strict")
    }
    cert_frozen = tools.certify_map(m, {h: slopes["frozen"][h] for h in stored}, tau, theta)
    unc_stored = sorted(map(list, certsafe_art["arms"]["CS:PMRT"]["uncertified"]))
    check = _check(
        ud, keys, slopes, m, stored, unc_stored, cert_frozen, stored_units, ResidualConfig()
    )
    maps_rows = _map_rows(m, slopes)
    gate_rows = _gate_rows(m, slopes, tau, tools)
    dz = [abs(r["dub_over_tau"]) for r in gate_rows if r["dub_over_tau"] is not None]
    und = [
        abs(r["dub_over_tau"])
        for r in gate_rows
        if r["dub_over_tau"] is not None and not r["declared"]
    ]
    summ = {
        "n": len(dz),
        "max_abs_dub_over_tau": max(dz),
        "median_abs_dub_over_tau": float(np.median(dz)),
        "n_undeclared": len(und),
        "max_abs_dub_over_tau_undeclared": max(und) if und else None,
        "median_abs_dub_over_tau_undeclared": float(np.median(und)) if und else None,
    }
    cert = {
        c: tools.certify_map(m, {h: slopes[c][h] for h in stored}, tau, theta)
        for c in ("frozen", "strict")
    }
    certification, changes = _certification(m, cert, certsafe_art, theta, tools)
    mx = max(maps_rows, key=lambda r: abs(r["rel_change"]) if r["rel_change"] is not None else -1)
    reading = (
        "no class changes status: CS:PMRT signature unchanged"
        if not changes and certification["signature_equal"]
        else "status change(s): report which, with the bound change; "
        "re-simulation is a user decision"
    )
    return {
        "schema": "x16-centre/1",
        "protocol": PROTOCOL,
        "strict_rule": "j enters the state for u iff (same episode and t0_j + 90 <= t0_u) "
        "or episode_j < episode_u",
        "check": check,
        "maps": maps_rows,
        "max_map_change": {
            k: mx[k] for k in ("family", "rel", "kpi", "beta_frozen", "beta_strict", "rel_change")
        },
        "gate_bounds": gate_rows,
        "gate_summary": summ,
        "certification": certification,
        "reading": reading,
    }


def _js(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set | tuple):
        return list(o)
    return str(o)


def run(
    records: Path = RECORDS,
    *,
    tools: RefereeTools,
    cache_dir: Path,
    artifacts: Path = ARTIFACTS,
) -> dict[str, Any]:
    """X16 from the frozen record sets: builds the DISC table, then compares the two centres."""
    files = disc_files(records)
    ud = unit_table(files, cache_dir)
    certsafe_art = json.loads((artifacts / CERTSAFE_ART).read_text(encoding="utf-8"))
    maps_art = json.loads((artifacts / MAPS_ART).read_text(encoding="utf-8"))
    bounds = (
        records / "study2_3_decision" / "runs" / "e6p-cs-bounds-1" / "out" / "certsafe_bounds.json"
    )
    stored_units = (
        json.loads(bounds.read_text(encoding="utf-8"))["n_units"] if bounds.exists() else None
    )
    out = analyse(ud, certsafe_art, maps_art, tools, stored_units=stored_units)
    out["data"] = {
        "files": [str(f.relative_to(records)).replace("\\", "/") for f in files],
        "H": H,
        "H_pre": H,
        "stage": "eval",
        "t0_field": "UnitTable.t0",
    }
    out["sha256"] = {
        CERTSAFE_ART: lf_sha256(artifacts / CERTSAFE_ART),
        MAPS_ART: lf_sha256(artifacts / MAPS_ART),
        "oran_causal/analysis/posthoc/x16.py": lf_sha256(Path(__file__)),
    }
    out["env"] = {"numpy": np.__version__, "python": sys.version.split()[0]}
    return out


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x16_centre.json` and `x16_centre.csv`."""
    d = Path(out_dir)
    write_text(d / "x16_centre.json", json.dumps(out, indent=1, default=_js))
    with (d / "x16_centre.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in out["maps"]:
            w.writerow(dict(r, table="map"))
        for r in out["gate_bounds"]:
            w.writerow(dict(r, table="gate"))


def main(argv: Sequence[str] | None = None, *, tools: RefereeTools) -> int:
    """Command line: recompute X16 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.records), tools=tools, cache_dir=Path(a.cache_dir))
    write(out, a.out)
    print(json.dumps({"check_ok": out["check"]["ok"], "reading": out["reading"]}, indent=1))
    return 0

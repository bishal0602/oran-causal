"""Study 2 discovery tests over the confounded episodes, and the referee maps they yield.

Inputs: the dev_conf, disc, placebo and gt collection records (and the Study 1 DEV records for
the descriptive step-1 tau). Steps:

- record checks: every unit of a decision family carries a valid incumbent probability row,
  `pi0_table` is null, collection is directional; the unit tables have no p mismatch;
- repro: the incumbent and all-accept repro jobs against the logging trajectory and `noarb`
  (bit identity decides the noarb alias);
- PMRT (frozen v4 artifact, every combination) on the pooled DISC episodes (split 0), the
  placebo episodes (split 9) and the null-outcome variants (groups of 60 DISC episodes);
- the baselines with the DEV far-FPR tau (`@dev`) and the placebo-calibrated tau (`@plc`;
  granger by the n-free partial r^2 transfer), and granger_by;
- the maps of every referee arm, their signatures and alias table, decision tables, edge
  quality and the offline replay on the DISC logged contexts;
- descriptive competitors: IPW-Wald, Granger with context, MSCR-CRT v2 + BY.

The criteria (K0, K0n, K1, G, X1-X3) and the discovery label are `analysis.system.study2`'s.
"""

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import norm

from oran_causal.analysis.system import study1 as crit
from oran_causal.analysis.system import study2 as crit2
from oran_causal.core.data import UnitTable
from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.experiments.system import study2
from oran_causal.experiments.system.artifacts import platform_key
from oran_causal.experiments.system.common import repro_diff
from oran_causal.experiments.system.study1_discovery import (
    ARTIFACT,
    PRIMARY,
    build_caches,
    decl_record,
    declared_list,
    dump,
    find_records,
    gt_reference,
    hyp_table,
    log,
    null_groups,
    null_tests,
    unit_counts,
)
from oran_causal.methods.system import artifact as pmrt_artifact
from oran_causal.methods.system import baselines as bl
from oran_causal.methods.system import maps as mp
from oran_causal.methods.system import mscr_crt
from oran_causal.methods.system import pmrt as pm
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision import records
from oran_causal.sim.system.decision.collect import INC_FLOOR
from oran_causal.sim.system.decision.vocab import FAMILIES, HYPOTHESES, KPIS, RELATIONS

SCHEMA = "e6p-conf-disc-tests/1"
H = H_PRE = 90
SPLIT = {"pooled": 0, "placebo": 9, "null": 0}
MAX_FP = 1
DISC_BOOT_KEY = (6624, 10)
RAND_TAG, RAND_KEY = 6623, 3
KEEP = (
    "kind",
    "schema",
    "key",
    "stage",
    "sub",
    "conf_stage",
    "seed",
    "j",
    "fold",
    "smoke",
    "policy",
    "pi0_table",
    "n_cells",
    "units",
    "lab_series",
)
STAGE_OF = {"dev": "dev", "disc": "eval", "placebo": "placebo"}

type Decl = Mapping[tuple, Mapping[str, Any]]


def check_unit_row(u: Mapping[str, Any]) -> str | None:
    """None if the unit carries a valid logged incumbent row, else the reason."""
    pr = u.get("probs")
    if pr is None:
        return "missing"
    if set(pr) - {"accept", "reject"}:
        return "keys"
    v = [float(x) for x in pr.values()]
    if abs(sum(v) - 1.0) > 1e-9:
        return "sum"
    if min(v) < INC_FLOOR - 1e-9 or max(v) > 1.0 - INC_FLOOR + 1e-9:
        return "floor"
    if u.get("mode") not in pr or abs(float(pr[u["mode"]]) - float(u["p"])) > 1e-9:
        return "p"
    return None


def check_episode(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Per-unit rows on every decision-family unit and a null `pi0_table` of one record."""
    bad: dict[str, int] = {}
    n = 0
    for u in rec["units"]:
        if u["knob"] not in FAMILIES:
            continue
        n += 1
        why = check_unit_row(u)
        if why is not None:
            bad[why] = bad.get(why, 0) + 1
    return {
        "units": n,
        "bad": bad,
        "pi0_table_null": rec.get("pi0_table") is None,
        "directional": bool(rec.get("directional")),
        "ok": not bad and rec.get("pi0_table") is None,
    }


@dataclasses.dataclass
class Scan:
    """Collection records of one stage: light copies, checks, run headers and job records."""

    records: list[dict]
    headers: list[dict]
    jobs: list[dict]
    check: dict[str, Any]
    outcomes: dict[int, dict]


def _header_row(r: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "conf_stage": r.get("conf_stage", r.get("stage")),
        "smoke": r.get("smoke"),
        "platform": r.get("platform") or r.get("numeric_env"),
    }


def _check_summary(chk: Mapping[tuple, dict]) -> dict[str, Any]:
    rows = [chk[k] for k in sorted(chk)]
    bad: dict[str, int] = {}
    for c in rows:
        for w, v in c["bad"].items():
            bad[w] = bad.get(w, 0) + v
    return {
        "episodes": len(rows),
        "units": int(sum(c["units"] for c in rows)),
        "bad_units": bad,
        "pi0_table_nonnull": int(sum(not c["pi0_table_null"] for c in rows)),
        "not_directional": int(sum(not c["directional"] for c in rows)),
        "seeds": sorted({k[0] for k in chk}),
        "smoke": any(k[1] for k in chk),
        "ok": bool(rows) and all(c["ok"] for c in rows),
    }


def scan_files(files: Sequence[Path], conf_stage: str, keep: bool, allow_smoke: bool) -> Scan:
    """Streams the records of one collection stage: checks every episode, keeps light copies.

    Also collects the run headers, the Study 2 job records and the logging outcomes of the
    repro seeds.
    """
    recs, chk, outcomes = {}, {}, {}
    headers, jobs = [], []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                head = line[:200]
                if '"episode"' not in head:
                    if '"header"' in head or '"job"' in head:
                        r = json.loads(line)
                        if r.get("kind") == "header" and r.get("driver") in ("e6p_conf", "study2"):
                            headers.append(_header_row(r))
                        elif (
                            r.get("kind") == "job"
                            and r.get("sub") == study2.SUB
                            and (allow_smoke or not r.get("smoke"))
                        ):
                            jobs.append(r)
                    continue
                r = json.loads(line)
                if (
                    r.get("kind") != "episode"
                    or r.get("sub") != study2.SUB
                    or r.get("conf_stage") != conf_stage
                    or (r.get("smoke") and not allow_smoke)
                ):
                    continue
                key = (int(r["seed"]), bool(r.get("smoke")))
                chk[key] = check_episode(r)
                if int(r["seed"]) in study2.REPRO_SEEDS and r.get("lab_outcome") is not None:
                    outcomes[int(r["seed"])] = {"lab_outcome": r["lab_outcome"]}
                if keep:
                    for u in r["units"]:
                        u.pop("lab_kpi", None)
                    recs[key] = {k: r[k] for k in KEEP if k in r}
    return Scan([recs[k] for k in sorted(recs)], headers, jobs, _check_summary(chk), outcomes)


def repro_check(coll: Mapping[int, Mapping], arms: Mapping[int, Mapping[str, Mapping]]) -> dict:
    """The repro checks on the repro seeds.

    `coll` {seed: dev_conf collection record}, `arms` {seed: {arm: dev_repro job}}: the
    incumbent arm against the logging trajectory's `lab_outcome`, and all-accept MapGateV2
    against noarb (bit identity decides the noarb alias).
    """
    out: dict[str, Any] = {"incumbent_vs_logging": {}, "allaccept_vs_noarb": {}}
    for s in study2.REPRO_SEEDS:
        a = arms.get(s, {})
        if s in coll and "incumbent" in a:
            out["incumbent_vs_logging"][s] = repro_diff(coll[s]["lab_outcome"], a["incumbent"])
        if "MG:allaccept" in a and "noarb" in a:
            out["allaccept_vs_noarb"][s] = repro_diff(a["MG:allaccept"], a["noarb"])
    for k in ("incumbent_vs_logging", "allaccept_vs_noarb"):
        v = out[k]
        ok = (len(v) == len(study2.REPRO_SEEDS) and all(x == 0 for x in v.values())) if v else None
        out[k] = {"per_seed": v, "ok": ok}
    out["noarb_alias"] = bool(out["allaccept_vs_noarb"]["ok"])
    return out


def pmrt_map(decl: Decl, ud: UnitTable) -> tuple[dict, list]:
    """Declared nonzero-sign edges -> sign * |design slope|, and the sign disagreements."""
    M, dis = {}, []
    for h, v in sorted(decl.items()):
        if not v.get("declared") or int(v.get("sign", 0)) == 0:
            continue
        s = int(v["sign"])
        b = rf.design_slope_bound(ud, *h)["beta"]
        if b != 0 and int(np.sign(b)) != s:
            dis.append([*h, s, b])
        M[tuple(h)] = float(s * abs(b))
    return M, dis


def _granger_fit(post: np.ndarray, A: np.ndarray, df: int) -> tuple[float, int] | None:
    beta, *_ = np.linalg.lstsq(A, post, rcond=None)
    res = post - A @ beta
    s2 = float(res @ res) / df
    se = float(np.sqrt(max((s2 * np.linalg.pinv(A.T @ A))[2, 2], 0.0)))
    if se <= 0:
        return None
    return float((beta[2] / se) ** 2), int(np.sign(beta[2]))


def granger_stats(ud: UnitTable, sup: bl.Support = bl.SUPPORT) -> dict:
    """Granger's regression post = a + b pre + c x per hypothesis.

    Returns:
        Per hypothesis: F, df, the sign and p of c, and the partial r^2 = F / (F + df).
    """
    out = {}
    for f in FAMILIES:
        if not bl.supported(ud, f, sup):
            continue
        rows = ud.rows_of(f)
        for r in RELATIONS:
            for k in KPIS:
                pre = ud.pre[(r, k)][rows]
                post = ud.y[(r, k)][rows] + pre
                n = len(rows)
                A = np.column_stack([np.ones(n), pre, ud.x[rows]])
                if np.std(post) == 0 or np.linalg.matrix_rank(A) < 3:
                    continue
                fit = _granger_fit(post, A, n - 3)
                if fit is None:
                    continue
                F, sign = fit
                df = n - 3
                out[(f, r, k)] = {
                    "F": F,
                    "df": df,
                    "r2": F / (F + df),
                    "sign": sign,
                    "p": float(fdist.sf(F, 1, df)),
                }
    return out


def granger_plc_transfer(
    plc: Mapping, disc: Mapping, relations: Sequence[str] = RELATIONS, max_fp: int = MAX_FP
) -> tuple[dict, dict]:
    """The n-free placebo tau of granger: the (max_fp + 1)-th largest placebo partial r^2.

    DISC declares iff its r^2 exceeds it (sign unchanged). Returns (declarations, tau info).
    """
    s = sorted(
        (v["r2"] for h, v in plc.items() if h[1] in relations and np.isfinite(v["r2"])),
        reverse=True,
    )
    if len(s) <= max_fp:
        tau: dict[str, Any] = {
            "tau_r2": float("-inf"),
            "max_fp": max_fp,
            "n_scorable": len(s),
            "note": "<= max_fp scorable placebo hypotheses: declares every scorable one",
        }
    else:
        tau = {"tau_r2": float(s[max_fp]), "max_fp": max_fp, "n_scorable": len(s)}
    decl = {}
    for h in HYPOTHESES:
        v = disc.get(h)
        ok = (
            v is not None and h[1] in relations and np.isfinite(v["r2"]) and v["r2"] > tau["tau_r2"]
        )
        decl[h] = {
            "declared": bool(ok),
            "sign": int(v["sign"]) if v else 0,
            "score": v["r2"] if v else float("nan"),
        }
    return decl, tau


def n_declared(decl: Decl) -> int:
    """Number of declared edges."""
    return int(sum(bool(v.get("declared")) for v in decl.values() if v))


def decl_list(decl: Decl) -> list[list]:
    """Sorted [hypothesis, sign] rows of the declared edges."""
    return [list(x) for x in declared_list(decl)]


def _by(p: Sequence[float], q: float = 0.05) -> list[bool]:
    return weighted_step_up(np.asarray(p, float), None, q).tolist() if len(p) else []


def _declare_by(out: dict, keys: list[tuple], ps: list[float], q: float) -> dict:
    for h, d in zip(keys, _by(ps, q), strict=True):
        out[h]["declared"] = bool(d)
    return out


def ipw_wald(ud: UnitTable, sup: bl.Support = bl.SUPPORT, q: float = 0.05) -> dict:
    """Descriptive IPW-Wald test with the logged propensities, BY at q.

    psi_u = sgn_u y_u (A_u / m_u - (1 - A_u) / (1 - m_u)), A = the logged mode's level and
    m = probs . levels; estimate = mean psi with an episode-cluster SE, two-sided normal p.
    """
    lv = mscr_crt.LEVEL_V2_ARR
    out = {h: {"declared": False, "sign": 0, "p": float("nan")} for h in HYPOTHESES}
    keys, ps = [], []
    for f in FAMILIES:
        if not bl.supported(ud, f, sup):
            continue
        rows = ud.rows_of(f)
        m = ud.probs[rows] @ lv
        A = lv[ud.mode[rows]]
        ok = (m > 0) & (m < 1)
        w = np.where(ok, A / np.where(ok, m, 1) - (1 - A) / np.where(ok, 1 - m, 1), 0.0)
        ep = ud.episode[rows]
        eps = np.unique(ep)
        for r in RELATIONS:
            for k in KPIS:
                psi = ud.sgn[rows] * ud.y[(r, k)][rows] * w
                if not np.all(np.isfinite(psi)):
                    continue
                est = float(psi.mean())
                cl = np.array([np.sum(psi[ep == e] - est) for e in eps])
                se = float(np.sqrt(np.sum(cl**2)) / len(psi))
                if se <= 0:
                    continue
                z = est / se
                p = float(2 * norm.sf(abs(z)))
                out[(f, r, k)] = {
                    "declared": False,
                    "sign": int(np.sign(est)),
                    "p": p,
                    "est": est,
                    "z": z,
                }
                keys.append((f, r, k))
                ps.append(p)
    return _declare_by(out, keys, ps, q)


def _ctx_matrix(ud: UnitTable, rows: np.ndarray, cols: Sequence[str]) -> np.ndarray:
    Z = []
    for c in cols:
        v = np.asarray(ud.z[c][rows], float)
        fin = np.isfinite(v)
        if fin.sum() < 2 or np.nanstd(v) == 0:
            continue
        Z.append(np.where(fin, (v - np.nanmean(v)) / np.nanstd(v), 0.0))
        if not fin.all():
            Z.append((~fin).astype(float))
    return np.column_stack(Z) if Z else np.zeros((len(rows), 0))


def granger_ctx(ud: UnitTable, sup: bl.Support = bl.SUPPORT, q: float = 0.05) -> dict:
    """Descriptive Granger with context conditioners, BY at q.

    post = a + b pre + c x + G z_ctx over every numeric obs-only context column (standardised,
    NaN -> 0 plus a missing indicator); F test of c.
    """
    out = {h: {"declared": False, "sign": 0, "p": float("nan")} for h in HYPOTHESES}
    keys, ps = [], []
    ctx_cols = [
        c for c in ud.z if c.startswith("ctx_") and c not in ("ctx_step", "ctx_cur", "ctx_prop")
    ]
    for f in FAMILIES:
        if not bl.supported(ud, f, sup):
            continue
        rows = ud.rows_of(f)
        Z = _ctx_matrix(ud, rows, ctx_cols)
        for r in RELATIONS:
            for k in KPIS:
                pre = ud.pre[(r, k)][rows]
                post = ud.y[(r, k)][rows] + pre
                A = np.column_stack([np.ones(len(rows)), pre, ud.x[rows], Z])
                n, p_ = A.shape
                if n - p_ < 5 or np.std(post) == 0:
                    continue
                fit = _granger_fit(post, A, n - p_)
                if fit is None:
                    continue
                F, sign = fit
                p = float(fdist.sf(F, 1, n - p_))
                out[(f, r, k)] = {"declared": False, "sign": sign, "p": p, "F": F}
                keys.append((f, r, k))
                ps.append(p)
    return _declare_by(out, keys, ps, q)


def flip_summary(rep: Mapping[str, Any], n_boot: int = 2000) -> dict[str, Any]:
    """Per replay pair: mean flips per episode with a bootstrap 90 % CI ([6624, 10, pair])."""
    out = {}
    for j, (m, fl) in enumerate(sorted(rep["flips"].items())):
        pe = np.asarray(fl["per_episode"], float)
        ci = [float("nan"), float("nan")]
        if len(pe) >= 2:
            rng = np.random.default_rng([*DISC_BOOT_KEY, j])
            mm = pe[rng.integers(0, len(pe), (int(n_boot), len(pe)))].mean(1)
            ci = [float(np.quantile(mm, 0.05)), float(np.quantile(mm, 0.95))]
        out[m] = {
            "mean_per_episode": float(pe.mean()) if len(pe) else float("nan"),
            "ci90": ci,
            "pre_warm_mean": float(np.mean(fl["pre_warm"])) if fl["pre_warm"] else float("nan"),
            "post_warm_mean": float(np.mean(fl["post_warm"])) if fl["post_warm"] else float("nan"),
            "by_mean": {k: float(np.mean(v)) for k, v in fl["by"].items()},
        }
    return out


@dataclasses.dataclass(frozen=True)
class Settings:
    """Run settings of the Study 2 discovery tests.

    Attributes:
        B: CRT redraws of the pooled DISC and placebo tests.
        B_null: Redraws of the null-outcome variants.
        null_group: Episodes per null-outcome group.
        null_shifts: Cyclic shifts per group.
        workers: Cache-building processes.
        methods: Tuned baselines.
        ann: qacm may fit its neural model.
        v2: Run MSCR-CRT v2 + BY.
        null: Run the null-outcome variants.
        dry_run: Never a full analysis.
        allow_smoke: Accept smoke records.
    """

    B: int = 9999
    B_null: int = 999
    null_group: int = 60
    null_shifts: int = 4
    workers: int = 1
    methods: tuple[str, ...] = bl.TUNED
    ann: bool = True
    v2: bool = True
    null: bool = True
    dry_run: bool = False
    allow_smoke: bool = False


DEFAULT_SETTINGS = Settings()


@dataclasses.dataclass(frozen=True)
class Inputs:
    """Record locations (files, directories or globs) per stage, and the cache directory.

    `dev_step1` holds the Study 1 DEV records (or their caches) of the descriptive step-1 tau.
    """

    dev: tuple[str, ...]
    disc: tuple[str, ...]
    placebo: tuple[str, ...]
    gt: tuple[str, ...]
    cache_dir: Path
    dev_step1: tuple[str, ...] = ()


@dataclasses.dataclass
class _Pools:
    disc: pm.PmrtData
    plc: pm.PmrtData
    dev: UnitTable
    n_disc: int
    n_plc: int


def _pools(caches: Mapping[str, list[Path]], allow_smoke: bool) -> _Pools:
    pool = pm.load_pmrt_pool([str(p) for p in caches["disc"]], stages=["eval"])
    ppool = pm.load_pmrt_pool([str(p) for p in caches["placebo"]], stages=["placebo"])
    dpool = records.load_pool([str(p) for p in caches["dev"]], H=H, H_pre=H_PRE, stages={"dev"})
    for nm, p in (("disc", pool.pool), ("placebo", ppool.pool), ("dev", dpool)):
        subs = sorted({str(s) for s in p.eps["sub"]})
        if subs != [study2.SUB]:
            raise ValueError(f"{nm}: episodes of subs {subs} (only {study2.SUB!r} allowed)")
        if not allow_smoke and bool(np.any(p.eps["smoke"])):
            raise ValueError(f"{nm}: smoke records")
    return _Pools(
        pm.pmrt_data(pool),
        pm.pmrt_data(ppool),
        dpool.unit_data(),
        int(pool.pool.n_eps),
        int(ppool.pool.n_eps),
    )


def _baselines(p: _Pools, s: Settings) -> tuple[dict, dict]:
    res, err = {}, {}
    splits = {"placebo": p.plc.ud, "disc": p.disc.ud}
    for m in s.methods:
        try:
            res.update(bl.run_baselines(p.dev, splits, (m,), sup=bl.Support(), log=log, ann=s.ann))
        except Exception as e:
            err[m] = repr(e)[:400]
            log(f"baseline {m}: FAILED {err[m][:160]}")
    return res, err


@dataclasses.dataclass
class _Maps:
    maps: dict[str, rf.Map] = dataclasses.field(default_factory=dict)
    plc: dict[str, rf.Map] = dataclasses.field(default_factory=dict)
    prov: dict[str, Any] = dataclasses.field(default_factory=dict)
    report: dict[str, Any] = dataclasses.field(default_factory=dict)


def _fixed_maps(mm: _Maps, decl: Decl, decl_p: Decl, p: _Pools, g: Mapping[str, Any]) -> None:
    M_pmrt, disagree = pmrt_map(decl, p.disc.ud)
    mm.maps["MG:PMRT"] = M_pmrt
    mm.prov["MG:PMRT"] = {
        "method": "PMRT " + "+".join(PRIMARY),
        "declared": decl_list(decl),
        "beta": "design-based slope (section 4.3)",
        "sign_disagreements": disagree,
    }
    mm.plc["MG:PMRT"] = mp.map_from_declared(decl_p, p.plc.ud)[0]
    mm.maps["MG:GT"] = rf.map_from_gt(g["cells"], true_only=True)
    mm.prov["MG:GT"] = {
        "method": "knockout GT, TRUE edges ('dir'), beta = the edge mean",
        "gt_seeds": g["seeds"],
    }
    mm.maps["MG:rand"] = rf.random_sized_map(M_pmrt, g["cells"], RAND_TAG, RAND_KEY)
    mm.prov["MG:rand"] = {
        "method": f"random_sized_map(M_PMRT, gt cells, tag {RAND_TAG}, key {RAND_KEY})",
        "n_edges": len(M_pmrt),
    }
    mm.maps["blanket2"] = rf.blanket_saving_map()
    mm.prov["blanket2"] = {"method": "blanket_saving_map() (no-map control)"}


def _granger_by_map(mm: _Maps, r: Mapping[str, Any], p: _Pools) -> None:
    d = r["splits"]["disc"]["declared"]
    mm.maps["MG:granger_by"], drop = mp.map_from_declared(d, p.disc.ud)
    pdcl = r["splits"]["placebo"]["declared"]
    mm.plc["MG:granger_by"] = mp.map_from_declared(pdcl, p.plc.ud)[0]
    mm.prov["MG:granger_by"] = {
        "method": "granger, BY q .05 (untuned)",
        "declared": decl_list(d),
        "dropped_sign0": drop,
        "placebo_declared": decl_list(pdcl),
    }


def _plc_decl(m: str, r: Mapping[str, Any], rels: Sequence[str], p: _Pools, gcache: dict):
    if m == "granger":
        if not gcache:
            gcache["plc"], gcache["disc"] = granger_stats(p.plc.ud), granger_stats(p.disc.ud)
        d_plc, tau = granger_plc_transfer(gcache["plc"], gcache["disc"], rels)
        p_plc, _ = granger_plc_transfer(gcache["plc"], gcache["plc"], rels)
        tau["transfer"] = "n-free: partial r^2 = F / (F + df)"
        return d_plc, p_plc, tau
    tau = mp.placebo_tau(r["splits"]["placebo"]["scores"], rels)
    d_plc = bl.declare(mp.tuple_scores(r["splits"]["disc"]["scores"]), tau["tau"], rels)
    p_plc = bl.declare(mp.tuple_scores(r["splits"]["placebo"]["scores"]), tau["tau"], rels)
    return d_plc, p_plc, tau


def _baseline_maps(mm: _Maps, bres: Mapping[str, Any], p: _Pools) -> None:
    gcache: dict = {}
    for m, r in bres.items():
        if m == "granger_by":
            _granger_by_map(mm, r, p)
            continue
        rels = r.get("decl_relations") or list(RELATIONS)
        d_dev, p_dev = r["splits"]["disc"]["declared"], r["splits"]["placebo"]["declared"]
        arm = f"MG:{m}@dev"
        mm.maps[arm], drop = mp.map_from_declared(d_dev, p.disc.ud)
        mm.plc[arm] = mp.map_from_declared(p_dev, p.plc.ud)[0]
        mm.prov[arm] = {
            "method": f"{m}, far-FPR tau on dev_conf",
            "tau": r["tau"],
            "declared": decl_list(d_dev),
            "dropped_sign0": drop,
            "placebo_declared": decl_list(p_dev),
            "note": r.get("note"),
        }
        d_plc, p_plc, tau = _plc_decl(m, r, rels, p, gcache)
        arm = f"MG:{m}@plc"
        mm.maps[arm], drop = mp.map_from_declared(d_plc, p.disc.ud)
        mm.plc[arm] = mp.map_from_declared(p_plc, p.plc.ud)[0]
        mm.prov[arm] = {
            "method": f"{m}, placebo-calibrated tau (MAX_FP {MAX_FP})",
            "tau": tau,
            "declared": decl_list(d_plc),
            "dropped_sign0": drop,
            "placebo_declared": decl_list(p_plc),
        }
        mm.report[m] = {
            "tau_dev": r["tau"],
            "tau_plc": tau,
            "cpu_s": r.get("cpu_s"),
            "wall_s": r.get("wall_s"),
            "n_placebo_declared": {"dev": n_declared(p_dev), "plc": n_declared(p_plc)},
            "n_disc_declared": {"dev": n_declared(d_dev), "plc": n_declared(d_plc)},
        }


def step1_maps(
    files: Sequence[Path], bres: Mapping[str, Any], ud_disc: UnitTable, cache_dir: Path, s: Settings
) -> dict[str, Any]:
    """Descriptive: the @dev maps with the Study 1 (unconfounded) DEV far-FPR tau."""
    cz = build_caches(files, "dev", cache_dir / "dev_step1", "dev1", s.workers)
    d1 = records.load_pool([str(p) for p in cz], H=H, H_pre=H_PRE, stages={"dev"}).unit_data()
    maps, rep = {}, {}
    for m, r in bres.items():
        if m == "granger_by":
            continue
        fn = bl.SCORERS[m]
        kw: dict[str, Any] = {"seed": 0} if m in bl.SEEDED else {}
        if m == "qacm":
            kw["ann"] = s.ann
        dkw = dict(kw, relations=("own", "far")) if m == "qacm" else kw
        tau = bl.tune_tau(fn(d1, **dkw), "far_fpr")
        rels = r.get("decl_relations") or list(RELATIONS)
        d = bl.declare(mp.tuple_scores(r["splits"]["disc"]["scores"]), tau["tau"], rels)
        maps[f"MG:{m}@dev1"] = mp.map_from_declared(d, ud_disc)[0]
        rep[m] = {"tau": tau, "declared": decl_list(d)}
    return {"maps": maps, "report": rep}


def _replay(mm: _Maps, recs: Sequence[dict], cells: Sequence, extra: Mapping[str, rf.Map]) -> dict:
    all_maps: dict[str, rf.Map] = {"GT": mm.maps["MG:GT"]}
    pairs: dict[str, tuple[str, str]] = {}
    for arm, M in mm.maps.items():
        if arm == "MG:GT":
            continue
        all_maps[arm] = M
        all_maps[f"{arm}|clean"] = mp.clean_map(M, cells)
        pairs[arm] = (arm, "GT")
        pairs[f"{arm}~clean"] = (arm, f"{arm}|clean")
        if arm in mm.plc:
            all_maps[f"GT+plc:{arm}"] = mp.inject(mm.maps["MG:GT"], mm.plc[arm])
            pairs[f"GT+plc:{arm}"] = (f"GT+plc:{arm}", "GT")
    for arm, M in extra.items():
        all_maps[arm] = M
        pairs[arm] = (arm, "GT")
    rp = mp.replay(recs, all_maps, pairs=pairs)
    return {
        "n_units_per_episode": float(np.mean(rp["n_units"])) if rp["n_units"] else float("nan"),
        "defer_rate": rp["defer_rate"],
        "defers_per_episode": {m: float(np.mean(v)) for m, v in rp["decisions"].items()},
        "pairs": pairs,
        "flips": flip_summary(rp),
    }


def _descriptive(p: _Pools, ref: Mapping, cells: Sequence, s: Settings) -> dict[str, Any]:
    desc: dict[str, Any] = {}
    fns: tuple[tuple[str, Callable[[UnitTable], dict]], ...] = (
        ("ipw_wald", ipw_wald),
        ("granger_ctx", granger_ctx),
    )
    for nm, fn in fns:
        try:
            dd, dp = fn(p.disc.ud), fn(p.plc.ud)
            desc[nm] = {
                "declared": decl_list(dd),
                "metrics": crit.subset_metrics(dd, ref),
                "placebo_declared": decl_list(dp),
                "edge_quality": mp.edge_quality(mp.map_from_declared(dd, p.disc.ud)[0], cells),
            }
        except Exception as e:
            desc[nm] = {"error": repr(e)[:300]}
    if s.v2:
        try:
            cfg = mscr_crt.UnitCRTConfigV2(B=s.B)
            e2 = mscr_crt.edges_from_crt_v2(
                mscr_crt.run_crt_units_v2(p.disc.ud, cfg, SPLIT["pooled"])
            )
            desc["v2+by"] = {"declared": decl_list(e2), "metrics": crit.subset_metrics(e2, ref)}
        except Exception as e:
            desc["v2+by"] = {"error": repr(e)[:300]}
    return desc


def _maps_section(
    out: dict, decl: Decl, decl_p: Decl, p: _Pools, g: Mapping, bres: Mapping, noarb_alias: bool
) -> _Maps:
    mm = _Maps()
    _fixed_maps(mm, decl, decl_p, p, g)
    _baseline_maps(mm, bres, p)
    out["baselines"] = mm.report
    out["missing_arms"] = [x for x in rf.MAP_ARMS if x not in mm.maps]
    out["maps"] = {arm: rf.map_to_json(M) for arm, M in mm.maps.items()}
    out["placebo_maps"] = {arm: rf.map_to_json(M) for arm, M in mm.plc.items()}
    out["provenance"] = mm.prov
    out["alias"] = rf.alias_table(mm.maps, rf.THETA, noarb_alias)
    out["decision_tables"] = {
        arm: {
            f"{k[0]}{'+' if k[1] > 0 else '-'}": v
            for k, v in rf.decision_table_v2(M, rf.THETA).items()
        }
        for arm, M in mm.maps.items()
    }
    out["edge_quality"] = {arm: mp.edge_quality(M, g["cells"]) for arm, M in mm.maps.items()}
    return mm


def analyze(
    inputs: Inputs, settings: Settings = DEFAULT_SETTINGS, artifact: Path = ARTIFACT
) -> dict:
    """Runs the Study 2 discovery tests and builds the referee maps (module docstring)."""
    t_all = time.time()
    timing: dict[str, float] = {}
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol": study2.PROTOCOL.status(),
        "primary": "+".join(PRIMARY),
    }
    params, priors = pmrt_artifact.load_artifact(artifact)
    cfg = pmrt_artifact.pmrt_config(params, settings.B)
    t = time.time()
    files = {k: find_records(getattr(inputs, k)) for k in ("dev", "disc", "placebo", "gt")}
    scans = {
        "dev_conf": scan_files(files["dev"], "dev_conf", False, settings.allow_smoke),
        "disc": scan_files(files["disc"], "disc", True, settings.allow_smoke),
        "placebo": scan_files(files["placebo"], "placebo", False, settings.allow_smoke),
    }
    out["record_checks"] = {k: v.check for k, v in scans.items()}
    timing["scan"] = round(time.time() - t, 1)
    bad = {k: v for k, v in out["record_checks"].items() if not v["ok"]}
    if bad:
        raise ValueError(f"records without a valid per-unit row / with a pi0_table: {bad}")
    plats = {
        platform_key(h["platform"]) for s in scans.values() for h in s.headers if not h["smoke"]
    }
    out["platform"] = {
        "keys": sorted(map(list, plats), key=str),
        "equal": len(plats) == 1,
        "linux": all(p[2] == "Linux" for p in plats) and bool(plats),
    }
    arms_r: dict[int, dict] = {}
    for r in scans["dev_conf"].jobs:
        if r.get("conf_stage") == "dev_repro":
            arms_r.setdefault(int(r["seed"]), {})[r["arm"]] = r
    out["repro"] = repro_check(scans["dev_conf"].outcomes, arms_r)
    t = time.time()
    caches = {
        k: build_caches(files[k], STAGE_OF[k], inputs.cache_dir / k, k, settings.workers)
        for k in ("dev", "disc", "placebo")
    }
    timing["caches"] = round(time.time() - t, 1)
    p = _pools(caches, settings.allow_smoke)
    out["units"] = {
        "disc": unit_counts(p.disc.ud),
        "placebo": unit_counts(p.plc.ud),
        "dev": unit_counts(p.dev),
    }
    pmm = {
        "disc": p.disc.ud.meta["p_mismatch"],
        "placebo": p.plc.ud.meta["p_mismatch"],
        "dev": p.dev.meta["p_mismatch"],
    }
    out["p_mismatch"] = pmm
    if any(pmm.values()):
        raise ValueError(f"unit tables with a p mismatch: {pmm}")
    g = gt_reference(files["gt"])
    out["gt"] = {k: v for k, v in g.items() if k != "ref"}
    out["data"] = crit2.data_status(out["record_checks"], g["seeds"], settings.dry_run)
    t = time.time()
    run_p = pm.run_integrated(p.plc, params, cfg, SPLIT["placebo"])
    decl_p_all = pm.declare_all(run_p, priors, p.n_plc)
    out["placebo_hyp"] = hyp_table(run_p, decl_p_all[PRIMARY])
    out["placebo_decl"] = {pm.cname(*k): decl_record(d) for k, d in decl_p_all.items()}
    timing["k0"] = round(time.time() - t, 1)
    t = time.time()
    run = pm.run_integrated(p.disc, params, cfg, SPLIT["pooled"])
    decls_all = pm.declare_all(run, priors, p.n_disc)
    decl = decls_all[PRIMARY]
    out["disc_hyp"] = hyp_table(run, decl)
    out["disc_decl"] = decl_record(decl)
    out["pmrt_combos"] = {
        pm.cname(*k): {"metrics": crit.subset_metrics(d, g["ref"]), "declared": decl_list(d)}
        for k, d in decls_all.items()
    }
    timing["pmrt_disc"] = round(time.time() - t, 1)
    recs = scans["disc"].records
    if settings.null:
        t = time.time()
        seeds = [int(r["seed"]) for r in recs]
        lay = study2.LAYOUT["disc"]
        pseudo = settings.dry_run or not all(lay.lo <= s <= lay.hi for s in seeds)
        groups = _disc_groups(seeds, settings.null_group, pseudo)
        cfg_null = pmrt_artifact.pmrt_config(params, settings.B_null)
        out["K0n_raw"] = null_tests(recs, groups, params, priors, cfg_null, settings.null_shifts)
        timing["k0n"] = round(time.time() - t, 1)
    t = time.time()
    bres, out["baseline_errors"] = _baselines(p, settings)
    timing["baselines"] = round(time.time() - t, 1)
    t = time.time()
    mm = _maps_section(
        out, decl, decl_p_all[PRIMARY], p, g, bres, bool(out["repro"]["noarb_alias"])
    )
    extra: dict[str, rf.Map] = {}
    if inputs.dev_step1:
        s1 = step1_maps(find_records(inputs.dev_step1), bres, p.disc.ud, inputs.cache_dir, settings)
        out["step1_dev_tau"] = s1["report"]
        extra = s1["maps"]
    out["replay"] = _replay(mm, recs, g["cells"], extra)
    timing["maps_replay"] = round(time.time() - t, 1)
    t = time.time()
    out["descriptive"] = _descriptive(p, g["ref"], g["cells"], settings)
    timing["descriptive"] = round(time.time() - t, 1)
    timing["total"] = round(time.time() - t_all, 1)
    out["timing_s"] = timing
    return out


def _disc_groups(seeds: Sequence[int], size: int, pseudo: bool) -> list[list[int]]:
    if pseudo:
        return null_groups(seeds, size, True)
    lay = study2.LAYOUT["disc"]
    n = lay.hi - lay.lo + 1
    return [[i for i, s in enumerate(seeds) if (s - lay.lo) // size == g] for g in range(n // size)]


def run(inputs: Inputs, out_dir: Path, settings: Settings = DEFAULT_SETTINGS) -> Path:
    """Runs `analyze` and writes `tests_conf.json` into `out_dir`; returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyze(inputs, settings)
    path = out_dir / "tests_conf.json"
    dump(res, path)
    log(f"wrote {path}")
    return path

"""The `shap_gbdt` baseline rerun with real TreeSHAP, and the referee arms built on it.

The frozen Study 1 and 2 `shap_gbdt` records took the GBDT + permutation importance path
(`baselines.score_gbdt_perm`): their Kaggle sessions could not import `shap`. This module
recomputes the baseline with `baselines.score_shap` on the frozen inputs, runs the permutation
path next to it as the control that must reproduce the frozen records, and leaves every frozen
file in place. Commands (`python -m oran_causal.experiments.system.shap_rerun <command>`):

- `study1`: DEV tau, placebo and descriptive-slice declarations on the frozen Study 1 caches;
- `study2`: the @dev / @plc / @dev1 maps on the Study 2 discovery records, spliced into a copy of
  the frozen `disc_conf.json` with the alias table and the replay recomputed;
- `maps`, `bounds`, `certsafe`: `E6P_CONF_MAPS_SHAP.json` and `E6P_CERTSAFE_SHAP.json` (own pins);
- `eval`: Study 2 / 3 referee jobs against those artifacts;
- `report`: the Study 2 / 3 evaluation reports on the frozen EVAL records with the arms whose
  decision signature changed replaced by the rerun.
"""

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.system import stats
from oran_causal.analysis.system import study1 as crit
from oran_causal.analysis.system import study2 as astudy2
from oran_causal.analysis.system import study3 as astudy3
from oran_causal.experiments.launch.fingerprint import numeric_env
from oran_causal.experiments.paths import ARTIFACTS
from oran_causal.experiments.system import artifacts as art
from oran_causal.experiments.system import study2, study3
from oran_causal.experiments.system import study2_discovery as sd
from oran_causal.experiments.system.common import FrozenDoc, Part, sha_lf
from oran_causal.experiments.system.study1_discovery import (
    H_PRE,
    H,
    build_caches,
    declared_list,
    dump,
    episodes_of,
    find_records,
    log,
    placebo_check,
    slice_plan,
)
from oran_causal.methods.system import baselines as bl
from oran_causal.methods.system import maps as mp
from oran_causal.methods.system import referees as rf
from oran_causal.sim.system.decision import records
from oran_causal.sim.system.decision.vocab import RELATIONS

SCHEMA = "shap-rerun/1"
RTOL = 1e-6
METHODS = ("shap_gbdt", "gbdt_perm")
FROZEN_METHOD = "shap_gbdt"
DESC_KINDS = frozenset({"pooled", "300", "120"})
SPLICE_KEYS = ("maps", "placebo_maps", "provenance", "decision_tables", "edge_quality")
SHAP_ARMS = {
    2: ("MG:shap_gbdt@dev", "MG:shap_gbdt@plc"),
    3: ("CS:shap_gbdt@dev", "CS:shap_gbdt@plc"),
}
MAPS_SHAP = FrozenDoc(
    "E6P_CONF_MAPS_SHAP.json",
    ARTIFACTS / "E6P_CONF_MAPS_SHAP.json",
    "54c6c4a5da4740e3f4d23cb354273d9844adbd6802f704f87e9cd823b2689685",
    needs_line=False,
)
CERTSAFE_SHAP = FrozenDoc(
    "E6P_CERTSAFE_SHAP.json",
    ARTIFACTS / "E6P_CERTSAFE_SHAP.json",
    "c00901d92ba7b6c65fb90ec8a59f72f448c975e6e23b4e30e402e7547c2291d8",
    needs_line=False,
)


def _pool(cache: Path, stage: str) -> records.Pool:
    files = sorted(str(p) for p in (cache / stage).glob("*.npz"))
    if not files:
        raise ValueError(f"no caches under {cache / stage}")
    return records.load_pool(files, H=H, H_pre=H_PRE, stages={stage})


def study1(cache: Path, gt_cells: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> dict:
    """Study 1 descriptive runs of `methods` on the frozen caches `cache/{dev,eval,placebo}`.

    As `study1_discovery`: DEV far-FPR tau, then declarations on the placebo episodes and on
    the pooled, 300- and 120-episode EVAL slices, scored against the GT cells.
    """
    dev = _pool(cache, "dev").unit_data()
    plc = _pool(cache, "placebo").unit_data()
    pool = _pool(cache, "eval")
    ref = crit.gt_ref(gt_cells)
    plan = [s for s in slice_plan(int(x) for x in pool.eps["seed"]) if s.kind in DESC_KINDS]
    out: dict[str, Any] = {"units": {"dev": int(dev.n), "placebo": int(plc.n)}, "methods": {}}
    for m in methods:
        t = bl.desc_method(m, dev)
        res: dict[str, Any] = {
            "importance": bl.IMPORTANCE[m],
            "tau": t.tau,
            "placebo": placebo_check(t, plc),
            "slices": {},
        }
        log(f"{m}: tau {(t.tau or {}).get('tau')}, placebo {res['placebo']['declared']}")
        for sl in plan:
            ud = pool.unit_data(episodes_of(pool, sl.seeds).tolist())
            d = t(ud, sl.split)
            res["slices"][sl.name] = crit.subset_metrics(d, ref) | {"declared": declared_list(d)}
            log(f"  {sl.name}: {res['slices'][sl.name]['declared']}")
        out["methods"][m] = res
    return out


def _decl_of(pairs: Sequence[Sequence[Any]]) -> dict:
    return {tuple(h.split("|")): {"declared": True, "sign": int(s)} for h, s in pairs}


def _maps_of(
    m: str, p: Any, d1: Any, cells: Sequence[Mapping[str, Any]], log_fn: Any = log
) -> dict[str, Any]:
    bres = bl.run_baselines(
        p.dev, {"placebo": p.plc.ud, "disc": p.disc.ud}, (m,), sup=bl.Support(), log=log_fn
    )
    mm = sd._Maps()
    sd._baseline_maps(mm, bres, p)
    r = bres[m]
    rels = r.get("decl_relations") or list(RELATIONS)
    tau1 = bl.tune_tau(bl.SCORERS[m](d1, seed=0), "far_fpr")
    d = bl.declare(mp.tuple_scores(r["splits"]["disc"]["scores"]), tau1["tau"], rels)
    return {
        "importance": bl.IMPORTANCE[m],
        "baseline": {k: r[k] for k in ("tau", "tau_physics", "note", "importance", "cpu_s")},
        "report": mm.report[m],
        "maps": {a: rf.map_to_json(M) for a, M in mm.maps.items()},
        "placebo_maps": {a: rf.map_to_json(M) for a, M in mm.plc.items()},
        "provenance": mm.prov,
        "decision_tables": {a: _decision_table(M) for a, M in mm.maps.items()},
        "edge_quality": {a: mp.edge_quality(M, cells) for a, M in mm.maps.items()},
        "step1": {"tau": tau1, "declared": sd.decl_list(d)},
        "dev_scores": r["dev_scores"],
        "disc_scores": r["splits"]["disc"]["scores"],
        "placebo_scores": r["splits"]["placebo"]["scores"],
    }


def _decision_table(M: rf.Map) -> dict[str, str]:
    return {
        f"{k[0]}{'+' if k[1] > 0 else '-'}": v for k, v in rf.decision_table_v2(M, rf.THETA).items()
    }


def _renamed(res: Mapping[str, Any], src: str, dst: str) -> dict[str, Any]:
    def ren(d: Mapping[str, Any]) -> dict[str, Any]:
        return {k.replace(f"MG:{src}@", f"MG:{dst}@"): v for k, v in d.items()}

    return dict(res, **{k: ren(res[k]) for k in SPLICE_KEYS})


def splice(
    frozen: Mapping[str, Any], res: Mapping[str, Any], ud_disc: Any, recs: Sequence[dict]
) -> dict[str, Any]:
    """The frozen `disc_conf.json` with the `shap_gbdt` entries of `res` (`_maps_of` output).

    The maps, placebo maps, provenance, decision tables, edge quality, baseline report and
    step-1 entry of both shap arms are replaced; the alias table and the replay are recomputed
    over every arm in the frozen order.
    """
    d = copy.deepcopy(dict(frozen))
    for arm in SHAP_ARMS[2]:
        for k in SPLICE_KEYS:
            d[k][arm] = res[k][arm]
    d["baselines"][FROZEN_METHOD] = res["report"]
    d["step1_dev_tau"][FROZEN_METHOD] = res["step1"]
    maps: dict[str, rf.Map] = {a: rf.map_from_json(v) for a, v in d["maps"].items()}
    alias = rf.alias_table(maps, rf.THETA, bool(d["repro"]["noarb_alias"]))
    d["alias"] = json.loads(json.dumps(alias, default=list))
    plc: dict[str, rf.Map] = {a: rf.map_from_json(v) for a, v in d["placebo_maps"].items()}
    extra = {
        f"MG:{m}@dev1": mp.map_from_declared(_decl_of(v["declared"]), ud_disc)[0]
        for m, v in d["step1_dev_tau"].items()
    }
    d["replay"] = sd._replay(sd._Maps(maps=maps, plc=plc), recs, d["gt"]["cells"], extra)
    return d


def study2_maps(inputs: sd.Inputs, frozen: Mapping[str, Any], methods: Sequence[str]) -> dict:
    """Study 2 maps of `methods` on the discovery records, and the spliced `disc_conf.json`s.

    Returns `{"methods": {m: _maps_of}, "spliced": {m: disc_conf}}`; the `gbdt_perm` splice is
    the control that must equal the frozen file.
    """
    files = {k: find_records(getattr(inputs, k)) for k in ("dev", "disc", "placebo")}
    caches = {k: build_caches(files[k], sd.STAGE_OF[k], inputs.cache_dir / k, k) for k in files}
    p = sd._pools(caches, False)
    c1 = build_caches(find_records(inputs.dev_step1), "dev", inputs.cache_dir / "dev_step1", "dev1")
    d1 = records.load_pool([str(x) for x in c1], H=H, H_pre=H_PRE, stages={"dev"}).unit_data()
    recs = sd.scan_files(files["disc"], "disc", True, False).records
    cells = frozen["gt"]["cells"]
    out: dict[str, Any] = {"methods": {}, "spliced": {}}
    for m in methods:
        res = _maps_of(m, p, d1, cells)
        out["methods"][m] = res
        out["spliced"][m] = splice(frozen, _renamed(res, m, FROZEN_METHOD), p.disc.ud, recs)
    return out


def control_diff(frozen: Any, got: Any, path: str = "", rtol: float = RTOL) -> list[str]:
    """JSON paths where `got` differs from `frozen` (timings and notes skipped).

    Floats match within `rtol` (NaN matches NaN); everything else must be equal.
    """
    skip = ("cpu_s", "wall_s", "note", "importance", "timing_s")
    if isinstance(frozen, Mapping) and isinstance(got, Mapping):
        out = [f"{path}/{k}" for k in set(frozen) ^ set(got) if k not in skip]
        for k in frozen:
            if k in got and k not in skip:
                out += control_diff(frozen[k], got[k], f"{path}/{k}", rtol)
        return out
    if isinstance(frozen, list) and isinstance(got, list):
        if len(frozen) != len(got):
            return [f"{path} (len {len(frozen)} vs {len(got)})"]
        pairs = enumerate(zip(frozen, got, strict=True))
        return [d for i, (a, b) in pairs for d in control_diff(a, b, f"{path}/{i}", rtol)]
    nums = all(isinstance(x, int | float) and not isinstance(x, bool) for x in (frozen, got))
    if nums and (isinstance(frozen, float) or isinstance(got, float)):
        same = bool(np.isclose(frozen, got, rtol=rtol, atol=0.0, equal_nan=True))
    else:
        same = frozen == got
    return [] if same else [path]


def build_maps_shap(disc_path: Path, frozen_disc_sha: str) -> dict[str, Any]:
    """`E6P_CONF_MAPS_SHAP.json` from the spliced `disc_conf_shap.json`."""
    a = art.build_maps(art.read_json(disc_path), sha_lf(disc_path))
    a["shap_rerun"] = {
        "replaces": {"path": study2.MAPS.name, "sha256": study2.MAPS.want},
        "frozen_disc_conf": frozen_disc_sha,
        "arms": list(SHAP_ARMS[2]),
        "importance": bl.IMPORTANCE["shap_gbdt"],
    }
    return a


def bounds_shap(
    frozen_bounds: Mapping[str, Any], disc_cache: Path, disc_path: Path
) -> dict[str, Any]:
    """The frozen `certsafe_bounds.json` with the beta repro check and the pins of the SHAP maps.

    The bound tables cover every guard edge and do not depend on the maps; they are kept, and
    the JSON paths where their recomputation on this host differs (beyond `RTOL`) are stored.
    """
    files = sorted(str(p) for p in disc_cache.glob("*.npz"))
    ud = records.load_pool(files, H=H, H_pre=H_PRE, stages={"eval"}).unit_data()
    maps = art.maps_from_artifact(MAPS_SHAP.path)
    tb = art.bounds_tables(ud, maps)
    b = copy.deepcopy(dict(frozen_bounds))
    b["beta_repro"] = json.loads(json.dumps(tb["repro"], default=list))
    b["sha256"] = dict(
        b["sha256"], maps_artifact=sha_lf(MAPS_SHAP.path), **{"disc_conf.json": sha_lf(disc_path)}
    )
    b["shap_rerun"] = {
        "tables_recomputed_diff": {
            k: control_diff(frozen_bounds[k], rf.bounds_to_json(tb[k])) for k in ("pmrt", "ols")
        },
        "rtol": RTOL,
        "platform": numeric_env(),
    }
    return b


def build_certsafe_shap(bounds: Path, calib: Path, disc_path: Path) -> dict[str, Any]:
    """`E6P_CERTSAFE_SHAP.json` on `E6P_CONF_MAPS_SHAP.json` (pinned)."""
    sha = {
        "certsafe_bounds.json": sha_lf(bounds),
        "certsafe_calib.json": sha_lf(calib),
        "disc_conf.json": sha_lf(disc_path),
    }
    a = art.build_certsafe(
        art.read_json(bounds),
        art.read_json(calib),
        art.read_json(disc_path),
        sha,
        maps_doc=MAPS_SHAP,
    )
    a["shap_rerun"] = {
        "replaces": {"path": study3.CERTSAFE.name, "sha256": study3.CERTSAFE.want},
        "arms": list(SHAP_ARMS[3]),
        "importance": bl.IMPORTANCE["shap_gbdt"],
    }
    return a


def _sig(v: Mapping[str, Any]) -> str:
    return json.dumps(v["signature"])


def frozen_twins(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, str]:
    """{job arm of `new`: job arm of `old`} where only the arm that represents a signature changed.

    The new job's decision signature differs from its own frozen one but equals the signature of
    a frozen job, so the frozen records of that job stand for it.
    """
    by_sig = {_sig(old["arms"][j]): j for j in old["jobs"] if j in old["arms"]}
    out = {}
    for a in new["jobs"]:
        own = old["arms"].get(a)
        if a in old["jobs"] and own is not None and _sig(own) == _sig(new["arms"][a]):
            continue
        twin = by_sig.get(_sig(new["arms"][a]))
        if twin is not None:
            out[a] = twin
    return out


def changed_jobs(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Job arms of `new` whose decision signature no frozen job of `old` simulates."""
    twins = frozen_twins(old, new)
    sig_old = {a: _sig(v) for a, v in old["arms"].items()}
    return [
        a
        for a in new["jobs"]
        if a not in twins and (a not in old["jobs"] or _sig(new["arms"][a]) != sig_old[a])
    ]


def run_eval(
    study: int,
    out: Path,
    part: Part,
    arms: frozenset[str] | None,
    frozen: bool = False,
    smoke: bool = False,
    short: float | None = None,
) -> int:
    """Referee jobs of `arms` against the SHAP artifacts (`frozen`: the frozen ones)."""
    if study == 2:
        doc = study2.MAPS if frozen else MAPS_SHAP
        return study2.run("eval", out, part, smoke, short, None, arms, maps_doc=doc)
    cs = study3.CERTSAFE if frozen else CERTSAFE_SHAP
    mdoc = study2.MAPS if frozen else MAPS_SHAP
    return study3.run("eval", out, part, smoke, short, arms, art_doc=cs, maps_doc=mdoc)


def driver_verify(v: Mapping[str, Any], name: str) -> dict[str, Any]:
    """A `verify_*` report in the stored form (`driver_<name>_SHA256`, `driver_matches`)."""
    out = {k: x for k, x in v.items() if k not in ("pinned", "matches_pin")}
    return out | {f"driver_{name}_SHA256": v["pinned"], "driver_matches": v["matches_pin"]}


def eval_report(
    study: int,
    frozen_files: Sequence[Path],
    new_files: Sequence[Path],
    disc: Mapping[str, Any],
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """The Study 2 / 3 evaluation report with the changed jobs taken from `new_files`.

    Frozen records of map arms the SHAP artifact no longer simulates with the frozen signature
    are dropped; a job whose signature a frozen job already simulates takes that job's frozen
    records (`frozen_twins`); the SHAP artifact's alias table fills the aliased arms.
    """
    mod: Any = astudy2 if study == 2 else astudy3
    filt = astudy2.EVAL_FILTER if study == 2 else astudy3.job_filter("eval")
    old_doc, new_doc = (study2.MAPS, MAPS_SHAP) if study == 2 else (study3.CERTSAFE, CERTSAFE_SHAP)
    old, new = art.read_json(old_doc.path), art.read_json(new_doc.path)
    redo = set(changed_jobs(old, new))
    twins = frozen_twins(old, new)
    map_arms = set(new["arms"])
    R, heads = stats.load_jobs(frozen_files, filt)
    for s in R:
        kept = {
            a: r
            for a, r in R[s].items()
            if a not in map_arms or (a in new["jobs"] and a not in redo and a not in twins)
        }
        R[s] = kept | {a: R[s][j] for a, j in twins.items() if j in R[s]}
    Rn, hn = stats.load_jobs(new_files, filt)
    for s, by_arm in Rn.items():
        R.setdefault(s, {}).update({a: r for a, r in by_arm.items() if a in redo})
    stats.alias_fill(R, new["alias_of"])
    if study == 2:
        ver = driver_verify(art.verify_maps(MAPS_SHAP.path, doc=MAPS_SHAP), "MAPS")
        inp = astudy2.EvalInputs(disc, new, ver, True, argv)
    else:
        v = art.verify_certsafe(CERTSAFE_SHAP.path, maps_doc=MAPS_SHAP, doc=CERTSAFE_SHAP)
        inp = astudy3.EvalInputs(disc, new, driver_verify(v, "CERTSAFE"), True, argv)
    rep = mod.evaluate(R, heads + hn, inp)
    rep["shap_rerun"] = {
        "replaced_jobs": sorted(redo),
        "frozen_twins": twins,
        "new_files": [str(p) for p in new_files],
    }
    return rep


def _paths(xs: Sequence[str]) -> list[Path]:
    return find_records(xs)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line (module docstring)."""
    ap = argparse.ArgumentParser(prog="python -m oran_causal.experiments.system.shap_rerun")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("study1")
    a1.add_argument("--cache", type=Path, required=True)
    a1.add_argument("--analysis", type=Path, required=True, help="frozen analysis_v4.json")
    a1.add_argument("--methods", default=",".join(METHODS))
    a1.add_argument("--out", type=Path, required=True)
    a2 = sub.add_parser("study2")
    for k in ("dev", "disc", "placebo", "dev-step1"):
        a2.add_argument(f"--{k}", nargs="+", required=True)
    a2.add_argument("--frozen", type=Path, required=True, help="frozen disc_conf.json")
    a2.add_argument("--cache", type=Path, required=True)
    a2.add_argument("--methods", default=",".join(METHODS))
    a2.add_argument("--out", type=Path, required=True)
    a3 = sub.add_parser("maps")
    a3.add_argument("--disc", type=Path, required=True, help="disc_conf_shap.json")
    a3.add_argument("--frozen", type=Path, required=True, help="frozen disc_conf.json")
    a4 = sub.add_parser("bounds")
    a4.add_argument("--frozen", type=Path, required=True, help="frozen certsafe_bounds.json")
    a4.add_argument("--disc-cache", type=Path, required=True)
    a4.add_argument("--disc", type=Path, required=True, help="disc_conf_shap.json")
    a4.add_argument("--out", type=Path, required=True)
    a5 = sub.add_parser("certsafe")
    for k in ("bounds", "calib", "disc"):
        a5.add_argument(f"--{k}", type=Path, required=True)
    a6 = sub.add_parser("eval")
    a6.add_argument("--study", type=int, choices=(2, 3), required=True)
    a6.add_argument("--out", type=Path, required=True)
    a6.add_argument("--part", default=None)
    a6.add_argument("--arms", default=None, help="comma list (default: the changed jobs)")
    a6.add_argument("--frozen-artifacts", action="store_true")
    a6.add_argument("--smoke", action="store_true")
    a6.add_argument("--short", type=float, default=None)
    a7 = sub.add_parser("report")
    a7.add_argument("--study", type=int, choices=(2, 3), required=True)
    a7.add_argument("--frozen-runs", nargs="+", required=True)
    a7.add_argument("--new", nargs="+", required=True)
    a7.add_argument("--disc", type=Path, required=True, help="disc_conf_shap.json")
    a7.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "study1":
        frozen = art.read_json(a.analysis)
        res = study1(a.cache, frozen["gt"]["cells"], a.methods.split(","))
        dump({"schema": SCHEMA, "stage": "study1", **res, "platform": numeric_env()}, a.out)
    elif a.cmd == "study2":
        inputs = sd.Inputs(
            tuple(a.dev), tuple(a.disc), tuple(a.placebo), (), a.cache, tuple(a.dev_step1)
        )
        frozen = art.read_json(a.frozen)
        res = study2_maps(inputs, frozen, a.methods.split(","))
        a.out.mkdir(parents=True, exist_ok=True)
        for m, d in res["spliced"].items():
            dump(d, a.out / f"disc_conf_{m}.json")
        ctl = {m: control_diff(frozen, d) for m, d in res["spliced"].items() if m != FROZEN_METHOD}
        dump(
            {
                "schema": SCHEMA,
                "stage": "study2",
                "methods": res["methods"],
                "control_diff": ctl,
                "platform": numeric_env(),
            },
            a.out / "study2_maps.json",
        )
    elif a.cmd == "maps":
        art.write_artifact(build_maps_shap(a.disc, sha_lf(a.frozen)), MAPS_SHAP.path)
    elif a.cmd == "bounds":
        dump(bounds_shap(art.read_json(a.frozen), a.disc_cache, a.disc), a.out)
    elif a.cmd == "certsafe":
        art.write_artifact(build_certsafe_shap(a.bounds, a.calib, a.disc), CERTSAFE_SHAP.path)
    elif a.cmd == "eval":
        if a.arms:
            arms = frozenset(a.arms.split(","))
        else:
            old_doc, new_doc = (
                (study2.MAPS, MAPS_SHAP) if a.study == 2 else (study3.CERTSAFE, CERTSAFE_SHAP)
            )
            arms = frozenset(changed_jobs(art.read_json(old_doc.path), art.read_json(new_doc.path)))
        n = run_eval(a.study, a.out, Part.parse(a.part), arms, a.frozen_artifacts, a.smoke, a.short)
        log(f"{n} jobs")
    else:
        rep = eval_report(
            a.study, _paths(a.frozen_runs), _paths(a.new), art.read_json(a.disc), list(sys.argv)
        )
        key = "conf" if a.study == 2 else "certsafe"
        mod: Any = astudy2 if a.study == 2 else astudy3
        a.out.mkdir(parents=True, exist_ok=True)
        dump(rep, a.out / f"eval_{key}.json")
        dump(mod.verdict_record(rep), a.out / f"verdict_{key}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

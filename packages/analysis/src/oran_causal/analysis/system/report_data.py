"""The Studies 1-3 figure data (`figures_data.json`) of the 2026-10-02 report.

Recomputes the paper's Studies 1-3 numbers from the frozen records and asserts each block equal
to the frozen analyzer outputs; the names of the passed assertion blocks are part of the output.

- `v4`: every method's scores on every Study 1 slice from its declared-edge list and the GT
  cells; K0 (exact binomial tail from the placebo p-values), K1, P1, P2, P3, S and the verdict;
  the protocol and artifact hashes.
- `rename`: leaf-by-leaf diff of the frozen-code and renamed-code Study 1 analyses ("MSCR+" ->
  "PMRT"), against the stored equivalence records.
- `opt_a`: Study 2 discovery inputs, map quality per arm and the 160-seed referee evaluation
  from the raw EVAL records (Gate A, D1, D2, never-sleep contrast, deferral counts).
- `certsafe`: Study 3 calibration from the raw DEV records, the certification table of every
  gated arm and the 160-seed evaluation (D1, D2, D3, S1, per-seed identity of the gated PMRT arm
  with never-sleep).
"""

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.analysis.system import study1, study2, study3
from oran_causal.analysis.system.stats import (
    GUARD_KEYS,
    SUM_FIELDS,
    GateA,
    JobFilter,
    Records,
    alias_fill,
    ci,
    gate_a,
    load_jobs,
    pooled,
    sum_counts,
)
from oran_causal.sim.system.decision.vocab import FAMILIES, HYPOTHESES, RELATIONS

WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
RECORDS = WORKSPACE_ROOT / "data" / "records"
EXPERIMENTS = WORKSPACE_ROOT / "packages" / "experiments"

PRIMARY = study1.PRIMARY_NAME
COMBOS = tuple(
    f"{s}+{lay}" for s in ("plain_c", "loadsp_c", "max") for lay in ("by", "dagger1s", "wby1s")
)
DESC = study1.DESC_METHODS
VOLATILE = frozenset(
    {
        "argv",
        "artifact",
        "protocol",
        "timing",
        "timing_s",
        "wall_s",
        "peak_rss_mb",
        "rss_mb",
        "out",
        "cache_dir",
        "generated",
        "host",
        "elapsed_s",
    }
)
COUNT_GROUPS = ("defer", "defer_units", "units", "policy")
OUTCOME_SKIP = frozenset(
    {"key", "arm", "policy_counts", "signature", "aliases", "cpu_s", "secs", "rss_mb"}
)
RHO, Z90 = 1.05, 1.2815515655446004
FROZEN_NEWLINE = "\r\n"
GUARD_KPIS = ("v", "rlf")
CS_SECONDARY = ("noarb", "incumbent", "never_sleep", "MG:PMRT", "CS:GT")
CONF_COUNT_ARMS = ("MG:PMRT", "MG:GT", "never_sleep", "MG:granger_by", "MG:corr@dev")
COMPOSITION_ARMS = (
    "MG:PMRT",
    "CS:PMRT",
    "CS:GT",
    "CS:corr@dev",
    "CS:granger_by",
    "CS:granger@dev",
    "CS:two_tower@dev",
    "CS:shap_gbdt@dev",
    "CS:qacm@dev",
)
ADDENDUM_CLAIM = {"n_leaves": 44663, "n_diff": 14, "criterion_leaves": 36667}
SLICE_KEYS = (
    "chain_hits",
    "chain_true",
    "premise_hit",
    "ind_recall",
    "ind_precision",
    "ind_f1",
    "ind_tp",
    "ind_fp",
    "ov_precision",
    "ov_recall",
    "ov_f1",
    "sign_acc",
    "n_declared",
    "far_declared",
)


__all__ = [
    "Checks",
    "ReportCheckError",
    "Sources",
    "bound_ub",
    "build",
    "certification",
    "certsafe",
    "cs_eval",
    "figures_json",
    "map_quality",
    "opt_a",
    "opt_a_eval",
    "pooled",
    "rename",
    "rounded",
    "tau_table",
    "tree_diff",
    "v4",
    "write",
]


class ReportCheckError(ValueError):
    """A recomputed number differs from the frozen analyzer output."""


class Checks:
    """Ordered names of the passed assertion blocks."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def require(self, name: str, cond: object, detail: object = "") -> None:
        """Records `name` when `cond` holds.

        Raises:
            ReportCheckError: `cond` is false.
        """
        if not cond:
            raise ReportCheckError(f"{name}: {detail}")
        self.names.append(name)


def ensure(cond: object, detail: object = "") -> None:
    """Raises `ReportCheckError` with `detail` when `cond` is false (an unnamed element check)."""
    if not cond:
        raise ReportCheckError(str(detail))


@dataclasses.dataclass(frozen=True)
class Sources:
    """Where the frozen inputs live.

    Attributes:
        records: The record sets root (`data/records`).
        experiments: The experiments member (frozen protocols and artifacts).
    """

    records: Path = RECORDS
    experiments: Path = EXPERIMENTS

    @property
    def decision(self) -> Path:
        """The frozen analyzer outputs."""
        return self.records / "system_artifacts" / "decision"

    @property
    def study1_runs(self) -> Path:
        """The Study 1 runs."""
        return self.records / "study1_raw"

    @property
    def runs(self) -> Path:
        """The Studies 2 and 3 runs."""
        return self.records / "study2_3_decision" / "runs"

    def artifact(self, name: str) -> Path:
        """A frozen artifact."""
        return self.experiments / "artifacts" / name

    def protocol(self, study: int, name: str) -> Path:
        """A frozen protocol document."""
        return self.experiments / "protocols" / f"study{study}" / name


def read_json(path: Path) -> Any:
    """Parses a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha_lf(path: Path) -> str:
    """sha256 of a file with CRLF normalised to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """Relative closeness with None, NaN and infinities compared exactly."""
    if a is None or b is None:
        return a is b
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(b))


def _closes(xs: Sequence[Any], ys: Sequence[Any]) -> bool:
    return all(close(x, y) for x, y in zip(xs, ys, strict=False))


def binom_sf_ge(k: int, m: int, p: float) -> float:
    """P(Binom(m, p) >= k), summed exactly."""
    return float(sum(math.comb(m, i) * p**i * (1 - p) ** (m - i) for i in range(k, m + 1)))


def decl_from_list(lst: Iterable[Sequence[Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Declarations over every hypothesis from a `[["f|r|k", sign], ...]` list."""
    d: dict[tuple[str, ...], dict[str, Any]] = {
        h: {"declared": False, "sign": 0} for h in HYPOTHESES
    }
    for key, s in lst:
        d[tuple(key.split("|"))] = {"declared": True, "sign": int(s)}
    return d


def _res_files(run: Path, n: int) -> list[Path]:
    files = sorted(run.glob("*/res_*.jsonl"), key=lambda p: p.as_posix())
    ensure(len(files) == n, f"{run.name}: {len(files)} raw shards, want {n}")
    return files


def _metric_table(A: Mapping[str, Any], ref: study1.Ref) -> tuple[dict[str, dict], int]:
    sc: dict[str, dict] = {}
    n = 0
    for sname, sl in A["slices_result"].items():
        for m, st in sl["metrics"].items():
            mine = study1.subset_metrics(decl_from_list(st["declared"]), ref)
            for k in SLICE_KEYS:
                ensure(close(mine[k], st[k]), (sname, m, k, mine[k], st[k]))
            sc.setdefault(m, {})[sname] = mine
            n += 1
    return sc, n


def _k0(A: Mapping[str, Any]) -> dict[str, Any]:
    rows = [v["primary"] for v in A["placebo_hyp"].values()]
    ps = [
        r["p"]
        for r in rows
        if r.get("status") != "undetermined" and r.get("p") is not None and np.isfinite(r["p"])
    ]
    m, k = len(ps), int(sum(p <= 0.05 for p in ps))
    tail = binom_sf_ge(k, m, 0.05)
    nd = sum(bool(r.get("declared")) for r in rows)
    return {
        "m": m,
        "n_reject": k,
        "binom_tail": tail,
        "n_declared": nd,
        "pass": tail >= 0.01 and nd <= 1,
    }


def _slice_names(names: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    s60, s120, s300 = ([s for s in names if s.startswith(f"s{k}_")] for k in ("60", "120", "300"))
    return s60, s120, s300


def _v4_criteria(A: Mapping[str, Any], V: Mapping[str, Any], sc: Mapping, checks: Checks) -> dict:
    names = list(A["slices_result"])
    s60, s120, _ = _slice_names(names)
    P = sc[PRIMARY]
    K0 = _k0(A)
    checks.require(
        "v4: K0 recomputed (m, rejects, tail, declarations)",
        K0["m"] == V["K0"]["m"]
        and K0["n_reject"] == V["K0"]["n_reject"]
        and close(K0["binom_tail"], V["K0"]["binom_tail"], 1e-9)
        and K0["n_declared"] == V["K0"]["n_declared"]
        and K0["pass"] == V["K0"]["pass"],
    )
    units = {s: A["slices_result"][s]["units"] for s in s120}
    K1 = {
        s: {"sleep_units": u["sleep_units"], "sleep_rejects": u["sleep_rejects"]}
        for s, u in units.items()
    }
    k1 = study1.k1_check(K1)
    checks.require(
        "v4: K1",
        k1["pass"] == V["K1"]["pass"]
        and all(K1[s]["sleep_units"] == V["K1"]["slices"][s]["sleep_units"] for s in s120),
    )
    p1 = study1.p1_check([dict(P[s], name=s) for s in s120])
    P1 = [
        {"slice": r["name"], "premise": r["premise"], "chain_hits": r["chain_hits"], "ok": r["ok"]}
        for r in p1["slices"]
    ]
    checks.require("v4: P1", p1["pass"] == V["P1"]["pass"] and p1["n_ok"] == V["P1"]["n_ok"])
    p2 = study1.p2_check(
        [P[s]["ind_f1"] for s in s60],
        [P[s]["ind_f1"] for s in s120],
        {b: [sc[b][s]["ind_f1"] for s in s60] for b in study1.BASELINES},
        {b: [sc[b][s]["ind_f1"] for s in s120] for b in study1.BASELINES},
    )
    P2 = {
        b: {k: v[k] for k in ("wins60", "wins120", "win60", "win120", "ok")}
        for b, v in p2["per_baseline"].items()
    }
    for b in study1.BASELINES:
        st = V["P2"]["per_baseline"][b]
        checks.require(f"v4: P2 vs {b}", all(P2[b][k] == st[k] for k in P2[b]))
    pp = P["pooled"]
    p3 = study1.p3_check(
        decl_from_list(A["slices_result"]["pooled"]["metrics"][PRIMARY]["declared"]),
        study1.gt_ref(A["gt"]["cells"]),
    )
    checks.require(
        "v4: P3",
        p3["parts"] == V["P3"]["parts"]
        and close(pp["ov_precision"], V["P3"]["values"]["overall_precision"])
        and close(pp["ind_recall"], V["P3"]["values"]["ind_recall"])
        and pp["far_declared"] == V["P3"]["values"]["far_declared"],
    )
    s = study1.s_check(pp["sign_acc"], [P[x]["sign_acc"] for x in s120])
    S = {"pooled": s["pooled"], "mean120": s["mean120"]}
    checks.require(
        "v4: S",
        close(S["pooled"], V["S"]["pooled"])
        and close(S["mean120"], V["S"]["mean120"])
        and s["pass"] == V["S"]["pass"],
    )
    g = study1.g_check(A["gt"]["cells"])
    verdict = study1.verdict(
        {"pass": K0["pass"]}, V["K0n"], g, k1, p1, p2, p3, s, mode="full", frozen=True
    )["verdict"]
    checks.require(
        "v4: verdict precedence", verdict == V["verdict"]["verdict"] == "PARTIAL", verdict
    )
    return {
        "K0": K0,
        "K0n": {
            k: V["K0n"][k]
            for k in ("pass", "n_variants", "rate05", "rate05_family", "n_variants_with_decl")
        },
        "K1": K1,
        "G": g["pass"],
        "P1": P1,
        "P1_pass": p1["pass"],
        "P2": P2,
        "P2_pass": p2["pass"],
        "P3_parts": p3["parts"],
        "P3_values": {
            k: pp[k]
            for k in (
                "ov_precision",
                "sign_acc",
                "ind_f1",
                "ind_precision",
                "ind_recall",
                "chain_hits",
                "n_declared",
                "far_declared",
            )
        },
        "S": S,
        "verdict": verdict,
    }


def _ranges(sc: Mapping, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    s60, s120, s300 = _slice_names(names)

    def rng(m: str, key: str, sl: Sequence[str] = names) -> list[float] | None:
        v = [sc[m][s][key] for s in sl if s in sc[m] and np.isfinite(sc[m][s][key])]
        return [min(v), max(v)] if v else None

    return {
        m: {
            "ov_precision_all18": rng(m, "ov_precision"),
            "ind_f1_60": rng(m, "ind_f1", s60),
            "ind_f1_120": rng(m, "ind_f1", s120),
            "ind_f1_300": rng(m, "ind_f1", s300),
            "n_declared_all18": rng(m, "n_declared"),
            "ov_precision_pooled_120_300": rng(m, "ov_precision", ["pooled", *s120, *s300]),
        }
        for m in (PRIMARY, *study1.BASELINES)
    }


def _placebo_declarations(A: Mapping[str, Any], checks: Checks) -> dict[str, int]:
    plc = {m: v["n_declared"] for m, v in A["placebo_descriptive"].items()}
    for m, v in A["placebo_descriptive"].items():
        if "declared" in v:
            ensure(plc[m] == len(v["declared"]), m)
    checks.require("v4: placebo declaration counts consistent with the declared lists", True)
    return plc


def v4(src: Sources, checks: Checks) -> dict[str, Any]:
    """Study 1 figure data, asserted against the frozen Study 1 analysis and verdict."""
    A = read_json(src.decision / "v4_analysis.json")
    V = read_json(src.decision / "v4_verdict.json")
    run = src.study1_runs / "e6p-disc-v4an-1" / "out"
    checks.require(
        "v4: decision copies identical to the run outputs",
        sha_lf(src.decision / "v4_analysis.json") == sha_lf(run / "analysis_v4.json")
        and sha_lf(src.decision / "v4_verdict.json") == sha_lf(run / "verdict_v4.json"),
    )
    out: dict[str, Any] = {
        "protocol_sha256": sha_lf(src.protocol(1, "E6P_DISCOVERY_PROTOCOL_V4.md")),
        "artifact_sha256": {
            "frozen_mscrplus": sha_lf(src.artifact("E6P_MSCRPLUS_V4_FROZEN.json")),
            "pmrt": sha_lf(src.artifact("E6P_PMRT_V4.json")),
        },
    }
    checks.require(
        "v4: protocol LF sha256 = recorded",
        out["protocol_sha256"] == V["protocol"]["sha256"] == A["protocol"]["sha256"],
    )
    checks.require(
        "v4: frozen artifact sha256 = recorded",
        out["artifact_sha256"]["frozen_mscrplus"] == V["artifact"]["sha256"],
    )
    checks.require("v4: primary combination", A["primary"] == PRIMARY)
    ref = study1.gt_ref(A["gt"]["cells"])
    cnt = {
        s: sum(1 for h in HYPOTHESES if ref[h]["status"] == s) for s in ("TRUE", "NULL", "INDET")
    }
    checks.require("v4: GT counts", cnt == A["gt"]["counts"], (cnt, A["gt"]["counts"]))
    out["gt"] = {
        "counts": cnt,
        "nbr_true": sorted(
            "|".join(h) for h in HYPOTHESES if h[1] == "nbr" and ref[h]["status"] == "TRUE"
        ),
        "n_episodes": A["gt"]["n_episodes"],
        "n_labels": A["gt"]["n_labels"],
        "premise": {k: A["G"]["cell"][k] for k in ("mean", "ci", "n", "n_eps", "status", "sign")},
    }
    sc, n_checked = _metric_table(A, ref)
    checks.require(
        f"v4: scores of {n_checked} (method, slice) pairs recomputed from declared lists",
        n_checked > 0,
    )
    names = list(A["slices_result"])
    out["slices"] = {
        "names": names,
        "n_episodes": {s: A["slices_result"][s]["n_episodes"] for s in names},
    }
    out |= _v4_criteria(A, V, sc, checks)
    out["chain_z_pooled"] = {h: c["z"] for h, c in V["P3"]["chain"].items()}
    out["placebo_declarations"] = _placebo_declarations(A, checks)
    pooled_keys = (
        "ind_f1",
        "ind_precision",
        "ind_recall",
        "ov_precision",
        "ov_f1",
        "sign_acc",
        "n_declared",
        "chain_hits",
    )
    out["pooled"] = {
        m: {k: sc[m]["pooled"][k] for k in pooled_keys} for m in sc if "pooled" in sc[m]
    }
    out["ranges"] = _ranges(sc, names)
    out["per_slice"] = {
        m: {
            s: {k: sc[m][s][k] for k in ("ind_f1", "ov_precision", "n_declared", "sign_acc")}
            for s in names
        }
        for m in (PRIMARY, *study1.BASELINES)
    }
    out["desc_tau"] = A["desc_tau"]
    out["baseline_tau"] = A["baseline_tau"]
    out["timing_total_s"] = A["timing_s"]["total"]
    out["pooled_units"] = A["slices_result"]["pooled"]["units"]["units"]
    return out


def _label(x: Any) -> Any:
    return x.replace("MSCR+", "PMRT") if isinstance(x, str) else x


Path_ = tuple[Any, ...]


def _same_leaf(x: Any, y: Any) -> bool:
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    if isinstance(x, str) and isinstance(y, str):
        return _label(x) == _label(y)
    return type(x) is type(y) and x == y


def _dict_diff(x: dict, y: dict, path: Path_) -> list[tuple[Path_, Any, Any]]:
    out = []
    for k in sorted(set(x) | set(y)):
        if k in VOLATILE:
            continue
        if k not in x or k not in y:
            out.append(((*path, k), x.get(k, "<missing>"), y.get(k, "<missing>")))
        else:
            out += tree_diff(x[k], y[k], (*path, k))
    return out


def _list_diff(x: list, y: list, path: Path_) -> list[tuple[Path_, Any, Any]]:
    out = [((*path, "#len"), len(x), len(y))] if len(x) != len(y) else []
    for i, (u, v) in enumerate(zip(x, y, strict=False)):
        out += tree_diff(u, v, (*path, i))
    return out


def tree_diff(x: Any, y: Any, path: Path_ = ()) -> list[tuple[Path_, Any, Any]]:
    """Leaf differences of two JSON trees.

    Volatile keys are skipped, "MSCR+" reads as "PMRT" in strings and NaN equals NaN.
    """
    if isinstance(x, dict) and isinstance(y, dict):
        return _dict_diff(x, y, path)
    if isinstance(x, list) and isinstance(y, list):
        return _list_diff(x, y, path)
    return [] if _same_leaf(x, y) else [(path, x, y)]


def leaves(x: Any, path: Path_ = ()) -> Iterator[Path_]:
    """Paths of every leaf of a JSON tree."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, (*path, k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, (*path, i))
    else:
        yield path


def pstr(p: Path_) -> str:
    """A leaf path as `/key[index]` text."""
    return "".join(
        f"[{k}]" if isinstance(k, int) else ("#len" if k == "#len" else f"/{k}") for k in p
    )


CRITERION_TOPS = frozenset({"K0", "K0n", "G", "K1", "P1", "P2", "P3", "S", "placebo_hyp"})
TOP_CATEGORY = {"desc_tau": "desc_nn", "desc_errors": "desc_nn", "baseline_tau": "assoc"}


def _method_category(m: Any) -> str:
    return "criterion" if m in COMBOS else ("desc_nn" if m in DESC else "assoc")


def _top_category(p: Path_) -> str | None:
    top = p[0]
    if top in CRITERION_TOPS:
        return "criterion"
    if top == "K0n_descriptive":
        return "criterion" if p[1] == "combos" else "other"
    if top == "placebo_descriptive":
        return _method_category(p[1])
    if top == "slices_result" and len(p) > 2 and p[2] in ("hyp", "metrics"):
        return "criterion" if p[2] == "hyp" else _method_category(p[3])
    return TOP_CATEGORY.get(top)


def category(p: Path_) -> str:
    """The rename-diff category of an analysis leaf path."""
    if any(isinstance(k, str) and k in VOLATILE for k in p):
        return "volatile"
    return _top_category(p) or "other"


def _categories(a: Any, d: Sequence[tuple[Path_, Any, Any]]) -> dict[str, dict[str, int]]:
    cats: dict[str, list[int]] = {}
    for p in leaves(a):
        cats.setdefault(category(p), [0, 0])[0] += 1
    for p, _, _ in d:
        cats.setdefault(category(p if p[-1] != "#len" else p[:-1]), [0, 0])[1] += 1
    return {c: {"leaves": v[0], "diffs": v[1]} for c, v in cats.items()}


def _rename_one(fa: Path, fb: Path, name: str, checks: Checks) -> dict[str, Any]:
    a, b = read_json(fa / name), read_json(fb / name)
    d = tree_diff(a, b)
    n_leaf = sum(1 for _ in leaves(a))
    out: dict[str, Any] = {
        "n_leaves": n_leaf,
        "n_diff": len(d),
        "diffs": [[pstr(p), repr(x)[:60], repr(y)[:60]] for p, x, y in d],
    }
    if name == "analysis_v4.json":
        out["by_category"] = _categories(a, d)
        out["diff_paths_top"] = sorted({"/".join(str(k) for k in p[:4]) for p, _, _ in d})
        out["verdict_equal"] = a["verdict"] == b["verdict"]
        out["v_a"], out["v_b"] = a["verdict"]["verdict"], b["verdict"]["verdict"]
    st = read_json(fb / ("equivalence_" + name.split("_", 1)[0] + ".json"))
    checks.require(
        f"rename: {name} leaf count / diff count / diff paths = stored equivalence record",
        n_leaf == st["n_leaves"]
        and len(d) == st["n_diff"]
        and [pstr(p) for p, _, _ in d[:20]] == [x[0] for x in st["first_diffs"]],
        (n_leaf, len(d), st["n_leaves"], st["n_diff"]),
    )
    return out


def rename(src: Sources, checks: Checks) -> dict[str, Any]:
    """The frozen-code vs renamed-code Study 1 diff, asserted against the equivalence records."""
    fa = src.study1_runs / "e6p-disc-v4an-1" / "out"
    fb = src.study1_runs / "e6p-disc-v4an-pmrt-1" / "out"
    out: dict[str, Any] = {
        name: _rename_one(fa, fb, name, checks) for name in ("analysis_v4.json", "verdict_v4.json")
    }
    ana = out["analysis_v4.json"]
    cats = ana["by_category"]
    checks.require(
        "rename: category leaf counts sum to the tree size",
        sum(v["leaves"] for v in cats.values()) == ana["n_leaves"],
    )
    checks.require(
        "rename: zero differences in PMRT / criterion leaves", cats["criterion"]["diffs"] == 0
    )
    out["criterion_leaves"] = cats["criterion"]["leaves"]
    out["addendum_claim"] = dict(ADDENDUM_CLAIM)
    out["addendum_agrees"] = (
        ana["n_leaves"] == ADDENDUM_CLAIM["n_leaves"]
        and ana["n_diff"] == ADDENDUM_CLAIM["n_diff"]
        and cats["criterion"]["leaves"] == ADDENDUM_CLAIM["criterion_leaves"]
    )
    out["artifact_sha256"] = {
        n: sha_lf(src.artifact(n)) for n in ("E6P_MSCRPLUS_V4_FROZEN.json", "E6P_PMRT_V4.json")
    }
    checks.require(
        "rename: artifact sha256 as in the addendum",
        out["artifact_sha256"]["E6P_MSCRPLUS_V4_FROZEN.json"].startswith("4735a85a")
        and out["artifact_sha256"]["E6P_PMRT_V4.json"].startswith("9991c390"),
    )
    out["pmrt_supersedes"] = read_json(src.artifact("E6P_PMRT_V4.json")).get("supersedes")
    return out


def eval_filter(sub: str, stage_key: str, stage: str) -> JobFilter:
    """Referee job filter by stage only (any header of the stage)."""
    return JobFilter(schema=study2.SCHEMA_EVAL, sub=sub, stage_key=stage_key, stage=stage)


def run_ledger(heads: Sequence[Mapping[str, Any]], R: Records) -> dict[str, Any]:
    """Parts, first header time, code heads, platforms and CPU hours of a referee run."""
    jobs = [r for arms in R.values() for r in arms.values() if not r.get("alias_of")]
    return {
        "parts": len(heads),
        "first_header_utc": min(h["utc"] for h in heads),
        "git_head": sorted({h["code"]["git_head"][:7] for h in heads}),
        "e6_dirty": sorted({str(h["code"]["e6_dirty"]) for h in heads}),
        "platform": sorted({h["platform"]["platform"] for h in heads}),
        "cpu_h": sum(r["cpu_s"] for r in jobs) / 3600.0,
        "n_jobs": len(jobs),
    }


def fd_arms(st: GateA) -> dict[str, dict[str, Any]]:
    """Per arm point statistics and 90 % CIs in the figure-data layout."""
    out = {}
    for a, p in st.point.items():
        b = st.boot[a]
        out[a] = {
            **{
                k: p[k]
                for k in (
                    "V",
                    "R",
                    "Rstar",
                    "retention",
                    "matched",
                    "guard_ok",
                    "eligible",
                    "guard_ratio",
                )
            },
            "R_ci90": ci(b["R"]),
            "retention_ci90": ci(b["retention"]),
            "guard_ratio_ci90": {k: ci(b["guard"][k]) for k in GUARD_KEYS},
        }
    return out


def check_arms(
    tag: str, arms: Mapping[str, Mapping[str, Any]], stored: Mapping[str, Mapping[str, Any]]
) -> None:
    """Asserts every arm's point statistics and CIs equal the stored evaluation."""
    for a, p in arms.items():
        s = stored[a]
        for k in ("V", "R", "Rstar", "retention"):
            ensure(close(p[k], s[k]), (tag, a, k, p[k], s[k]))
        for k in GUARD_KEYS:
            ensure(close(p["guard_ratio"][k], s["guard_ratio"][k]), (tag, a, k))
            ensure(_closes(p["guard_ratio_ci90"][k], s["guard_ratio_ci90"][k]), (tag, a, k))
        ensure(p["eligible"] == s["eligible"] and p["matched"] == s["matched"], (tag, a))
        ensure(_closes(p["R_ci90"], s["R_ci90"]), (tag, a, p["R_ci90"], s["R_ci90"]))
        ensure(_closes(p["retention_ci90"], s["retention_ci90"]), (tag, a))


def _fd_d1(d1: Mapping[str, Any]) -> dict[str, Any]:
    return {k: d1[k] for k in ("pass", "delta1", "best_point", "lb90", "ci90")}


def _fd_d2(d2: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pass": d2["pass"],
        "n_hypotheses": d2["n_hypotheses"],
        "hypotheses": {
            b: {k: h[k] for k in ("members", "p", "rejected")} for b, h in d2["hypotheses"].items()
        },
    }


def _check_d1(checks: Checks, tag: str, d1: Mapping[str, Any], vd: Mapping[str, Any]) -> None:
    checks.require(
        f"{tag}: D1",
        d1["pass"] == vd["pass"]
        and close(d1["delta1"], vd["delta1"])
        and close(d1["lb90"], vd["lb90"])
        and d1["best_point"] == vd["best_point"],
    )


def _counts(
    R: Records, seeds: Sequence[int], arms: Sequence[str], stored: Mapping[str, Any]
) -> dict:
    counts = {a: {g: sum_counts(R, seeds, a, g) for g in COUNT_GROUPS} for a in arms}
    for a in arms:
        for g in COUNT_GROUPS:
            ensure(counts[a][g] == stored[a]["counts"][g], (a, g))
    return counts


def _edge_counts(edges: Iterable[Sequence[Any]], ref: study1.Ref) -> dict[str, Any]:
    c: dict[str, Any] = dict.fromkeys(
        ("edges", "true_right", "true_wrong", "null", "indet", "unknown"), 0
    )
    for f, r, k, beta in edges:
        c["edges"] += 1
        g = ref.get((f, r, k))
        if g is None:
            c["unknown"] += 1
        elif g["status"] == "TRUE":
            c["true_right" if int(np.sign(beta)) == g["sign"] else "true_wrong"] += 1
        elif g["status"] == "NULL":
            c["null"] += 1
        else:
            c["indet"] += 1
    return c


def map_quality(disc: Mapping[str, Any], maps_art: Mapping[str, Any]) -> dict[str, dict]:
    """Edge counts of every map against the GT cells, asserted against the stored edge quality."""
    ref = study1.gt_ref(disc["gt"]["cells"])
    mq = {}
    for arm, edges in disc["maps"].items():
        c = _edge_counts(edges, ref)
        eq = disc["edge_quality"][arm]
        ensure(
            (c["edges"], c["true_right"], c["true_wrong"], c["null"], c["indet"])
            == (eq["n_edges"], eq["n_true_right"], eq["n_true_wrong"], eq["n_null"], eq["n_indet"]),
            (arm, c, eq),
        )
        pl = disc["placebo_maps"].get(arm)
        c["placebo_edges"] = None if pl is None else len(pl)
        mq[arm] = c
        art_arm = maps_art["arms"].get(arm, {})
        if "map" in art_arm:
            ensure([list(e) for e in art_arm["map"]] == [list(e) for e in edges], arm)
    return mq


def _map_quality(
    src: Sources, disc: Mapping[str, Any], maps_art: Mapping[str, Any], checks: Checks
) -> dict:
    mq = map_quality(disc, maps_art)
    checks.require(
        "opt(a): map quality of all 17 maps recomputed from maps + GT cells", len(mq) == 17
    )
    txt = {}
    lines = (src.decision / "conf_map_quality.txt").read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        left, right = line.split("|")
        t = left.split()
        txt[t[0]] = (*map(int, t[1:6]), None if right.strip() == "-" else int(right))
    checks.require(
        "opt(a): conf_map_quality.txt agrees",
        all(
            txt[a]
            == (
                v["edges"],
                v["true_right"],
                v["true_wrong"],
                v["null"],
                v["indet"],
                v["placebo_edges"],
            )
            for a, v in mq.items()
        ),
    )
    return mq


def _opt_a_disc(disc: Mapping[str, Any], checks: Checks) -> dict[str, Any]:
    X2 = disc["X"]["X2"]
    scored = [e for e in X2["edges"] if e[2] != "INDET"]
    tr = [e for e in scored if e[2] == "TRUE"]
    prec = len(tr) / len(scored)
    sign = sum(e[1] == e[3] for e in tr) / len(tr)
    checks.require(
        "opt(a): X2 precision and sign accuracy from the scored edge list",
        close(prec, X2["precision"])
        and close(sign, X2["sign_accuracy"])
        and len(scored) == X2["n_scored"],
    )
    return {
        "label": disc["label"]["label"],
        "K0": {k: disc["K0"][k] for k in ("pass", "n_reject", "m", "n_declared", "min_p")},
        "K0n": {k: disc["K0n"][k] for k in ("pass", "rate05", "n_variants_with_decl")},
        "K1": {k: disc["K1"][k] for k in ("pass", "sleep_units", "sleep_rejects")},
        "G": {k: disc["G"]["cell"][k] for k in ("mean", "ci", "status", "sign")},
        "X1": disc["X"]["X1"],
        "X2": {
            "precision": prec,
            "sign_accuracy": sign,
            "n_scored": len(scored),
            "n_true": len(tr),
        },
        "X3": {k: disc["X"]["X3"][k] for k in ("chain_hits", "chain_true")},
        "gt_counts": disc["gt"]["counts"],
        "units": disc["units"]["disc"]["units"],
        "n_episodes": disc["data"],
    }


def _ptx_nbr_rlf(disc: Mapping[str, Any]) -> tuple[dict, dict]:
    ref = {(c["family"], c["relation"], c["kpi"]): c for c in disc["gt"]["cells"]}
    gt_rlf = {
        "|".join(h): {"status": ref[h]["status"], "mean": ref[h]["mean"], "ci": ref[h]["ci"]}
        for h in HYPOTHESES
        if h[2] == "rlf" and h[1] in ("nbr", "far") and h[0] in ("sleep", "ptx")
    }
    hy = disc["disc_hyp"]["ptx|nbr|rlf"]
    return gt_rlf, {
        "plain_c_z": hy["plain_c"]["z"],
        "plain_c_p2": hy["plain_c"]["p2"],
        "loadsp_c_z": hy["loadsp_c"]["z"],
        "loadsp_c_p2": hy["loadsp_c"]["p2"],
        "primary_p": hy["primary"]["p"],
        "primary_status": hy["primary"]["status"],
        "gt": gt_rlf["ptx|nbr|rlf"],
    }


def _has_edge(edges: Iterable[Sequence[Any]], h: tuple[str, str, str]) -> bool:
    return any((e[0], e[1], e[2]) == h for e in edges)


def _defer_units(counts: Mapping[str, Any], fam: str) -> dict[str, float]:
    return {
        a: counts[a]["defer_units"].get(f"{fam}_post", 0)
        + counts[a]["defer_units"].get(f"{fam}_pre", 0)
        for a in ("MG:PMRT", "MG:GT")
    }


def _opt_a_eval(src: Sources, maps_art: Mapping[str, Any], checks: Checks) -> dict[str, Any]:
    files = _res_files(src.runs / "e6p-conf-eval-1", 12)
    checks.require("opt(a): 12 raw EVAL shards present", len(files) == 12, len(files))
    R, heads = load_jobs(files, eval_filter(study2.SUB, "conf_stage", "eval"))
    led = run_ledger(heads, R)
    alias_fill(R, maps_art["alias_of"])
    E = read_json(src.decision / "conf_eval.json")
    Vd = read_json(src.decision / "conf_verdict.json")
    return opt_a_eval(R, led, maps_art, E, Vd, checks)


def opt_a_eval(
    R: Records,
    led: Mapping[str, Any],
    maps_art: Mapping[str, Any],
    E: Mapping[str, Any],
    Vd: Mapping[str, Any],
    checks: Checks,
) -> dict[str, Any]:
    """Study 2 referee evaluation of alias-filled EVAL records, asserted against `E` and `Vd`.

    Args:
        R: EVAL records per seed and arm, aliases filled.
        led: The run ledger (`run_ledger`).
        maps_art: The maps artifact (decision signatures).
        E: The stored evaluation (`conf_eval.json`).
        Vd: The stored verdict (`conf_verdict.json`).
        checks: Collects the passed assertion blocks.
    """
    arms = [a for a in study2.ALL_ARMS if any(a in R[s] for s in R)]
    st = gate_a(R, arms)
    checks.require(
        "opt(a): 160 complete seeds, every arm", st.n == E["n_seeds"] == 160 and len(arms) == 26
    )
    checks.require(
        "opt(a): Gate A constants",
        close(st.V_AA, E["gate_a"]["V_AA"])
        and close(st.den, E["gate_a"]["den"])
        and st.ref_arm == E["gate_a"]["ref_arm"],
    )
    P, B = fd_arms(st), st.boot
    check_arms("opt(a)", P, E["arms"])
    checks.require(
        "opt(a): every arm's V, R, R*, retention, guard ratios, eligibility and 90% CIs = "
        "conf_eval.json",
        True,
    )
    assoc = list(study2.ASSOC_ARMS)
    sig = {a: v["signature"] for a, v in maps_art["arms"].items()}
    D1 = _fd_d1(study2.d1_check(st.point, B, "MG:PMRT", assoc))
    D2 = _fd_d2(study2.d2_check(B, "MG:PMRT", assoc, sig))
    _check_d1(checks, "opt(a)", D1, Vd["D1"])
    checks.require(
        "opt(a): D2 (Holm)",
        D2["pass"] == Vd["D2"]["pass"]
        and D2["n_hypotheses"] == Vd["D2"]["n_hypotheses"]
        and all(
            close(h["p"], Vd["D2"]["hypotheses"][b]["p"])
            and h["rejected"] == Vd["D2"]["hypotheses"][b]["rejected"]
            for b, h in D2["hypotheses"].items()
        ),
    )
    sec = study2.secondary(st, "MG:PMRT", study2.SECONDARY)
    ns = sec["never_sleep"]
    checks.require(
        "opt(a): PMRT - never_sleep",
        close(ns["dR"], Vd["never_sleep"]["dR"]) and _closes(ns["ci90"], Vd["never_sleep"]["ci90"]),
    )
    E_pass = P["MG:PMRT"]["eligible"]
    verdict = study2.eval_verdict(k0=True, k0n=True, e=E_pass, d1=D1["pass"], d2=D2["pass"])[
        "verdict"
    ]
    checks.require(
        "opt(a): verdict", verdict == Vd["verdict"]["verdict"] == "NOT ELIGIBLE", verdict
    )
    counts = _counts(R, st.seeds, arms, E["arms"])
    checks.require("opt(a): deferral / unit / policy counts = conf_eval.json", True)
    return {
        "n": st.n,
        "seeds": [st.seeds[0], st.seeds[-1]],
        "V_AA": st.V_AA,
        "V_ref": st.V_ref,
        "ref_arm": st.ref_arm,
        "den": st.den,
        "arms": P,
        "D1": D1,
        "D2": D2,
        "never_sleep": ns,
        "E": E_pass,
        "verdict": verdict,
        "ptx_up_defer_units": _defer_units(counts, "ptx"),
        "sleep_defer_units": _defer_units(counts, "sleep"),
        "counts": {a: counts[a] for a in CONF_COUNT_ARMS},
        "secondary": sec,
        "run": led,
    }


def opt_a(src: Sources, checks: Checks) -> dict[str, Any]:
    """Study 2 figure data, asserted against the frozen discovery, maps and evaluation outputs."""
    disc = read_json(src.decision / "conf_disc_conf.json")
    maps_path = src.artifact("E6P_CONF_MAPS.json")
    maps_art = read_json(maps_path)
    out: dict[str, Any] = {
        "protocol_sha256": sha_lf(src.protocol(2, "E6P_CONFOUNDED_PROTOCOL.md")),
        "maps_sha256": sha_lf(maps_path),
        "disc_json_sha256": sha_lf(src.decision / "conf_disc_conf.json"),
    }
    checks.require(
        "opt(a): protocol sha256 = disc record",
        out["protocol_sha256"] == disc["protocol"]["sha256"],
    )
    checks.require(
        "opt(a): maps artifact sha256 = .sha256 file",
        out["maps_sha256"] == src.artifact("E6P_CONF_MAPS.json.sha256").read_text().split()[0],
    )
    checks.require(
        "opt(a): disc_conf.json = the copy the maps artifact pins",
        out["disc_json_sha256"] == maps_art["sha256"]["disc_conf.json"],
    )
    out["disc"] = _opt_a_disc(disc, checks)
    out["map_quality"] = _map_quality(src, disc, maps_art, checks)
    out["gt_rlf"], out["ptx_nbr_rlf"] = _ptx_nbr_rlf(disc)
    out["pmrt_has_ptx_nbr_rlf"] = _has_edge(disc["maps"]["MG:PMRT"], ("ptx", "nbr", "rlf"))
    out["gt_has_ptx_nbr_rlf"] = _has_edge(disc["maps"]["MG:GT"], ("ptx", "nbr", "rlf"))
    out["decision_tables"] = {a: disc["decision_tables"][a] for a in ("MG:PMRT", "MG:GT")}
    out["eval"] = _opt_a_eval(src, maps_art, checks)
    return out


def tau_table(
    G: Mapping[str, float], N: Mapping[tuple[str, int], float]
) -> dict[tuple[str, int, str], float]:
    """Harm tolerance per (family, direction, guard KPI): (rho - 1) G_k / (max(N(f, d), 1) 3)."""
    return {
        (f, d, k): (RHO - 1.0) * G[k] / (max(N.get((f, d), 0.0), 1.0) * 3)
        for f in FAMILIES
        for d in (1, -1)
        for k in GUARD_KPIS
    }


def bound_ub(b: Mapping[str, Any] | None, h: float) -> float:
    """One-sided 90 % upper bound of the harm of a step `h` from a slope or interval bound."""
    if not b:
        return math.inf
    if b["kind"] == "slope":
        if b.get("beta") is None or b.get("se") is None:
            return math.inf
        return h * b["beta"] + Z90 * b["se"]
    return max(h * b["lo"], h * b["hi"])


def _calibration(
    src: Sources, cal: Mapping[str, Any], checks: Checks
) -> tuple[dict, dict, dict, int]:
    Rd, _ = load_jobs(
        [src.runs / "e6p-cs-dev-1" / "all.jsonl"], eval_filter(study3.SUB, "cs_stage", "dev")
    )
    seeds = sorted(s for s in Rd if study3.CAL_ARM in Rd[s])
    calr = [Rd[s][study3.CAL_ARM] for s in seeds]
    G = {
        "v": float(np.mean([r["viol_ue_s"] for r in calr])),
        "rlf": float(np.mean([r["rlf"] for r in calr])),
    }

    def n_dir(r: Mapping[str, Any], f: str, d: int) -> float:
        return float(((r.get("policy_counts") or {}).get("policy") or {}).get(f"dir_{f}_{d:+d}", 0))

    N = {(f, d): float(np.mean([n_dir(r, f, d) for r in calr])) for f in FAMILIES for d in (1, -1)}
    rep = {
        s: max(abs(float(Rd[s][study3.CAL_ARM][f]) - float(Rd[s]["noarb"][f])) for f in SUM_FIELDS)
        for s in (191000, 191001)
    }
    checks.require(
        "2b: G_k, N(f, d) recomputed from the raw DEV records = cs_calib.json",
        len(seeds) == 40 == cal["n"]
        and all(close(G[k], cal["G"][k]) for k in GUARD_KPIS)
        and all(close(N[(f, int(d))], v) for f, d, v in cal["N"])
        and all(v == 0 for v in rep.values()),
    )
    return G, N, rep, len(seeds)


def _certify_class(
    cls: str, c: Mapping[str, Any], mkeys: set, b: Mapping | None, tau: Mapping, arm: str
) -> bool:
    f, ds = cls.split("|")
    d = int(ds)
    ok = True
    for i, rel in enumerate(RELATIONS):
        for j, k in enumerate(GUARD_KPIS):
            stored = c["edges"][i * 2 + j]
            if (f, rel, k) in mkeys:
                stt = "DECLARED"
            else:
                ub = bound_ub((b or {}).get((f, rel, k)), -d)
                stt = "CERTIFIED" if ub <= tau[(f, d, k)] else "UNRESOLVED"
                ensure(
                    (stored[3] is None and not math.isfinite(ub)) or close(stored[3], ub),
                    (arm, cls, rel, k),
                )
            ensure(stored[:3] == [rel, k, stt], (arm, cls, stored, stt))
            ok &= stt != "UNRESOLVED"
    return ok


def certification(art: Mapping[str, Any], tau: Mapping, checks: Checks) -> dict[str, Any]:
    """Certification of every certsafe arm recomputed from its map, bounds and `tau`."""
    bset = {m: {(f, r, k): v for f, r, k, v in rows} for m, rows in art["bounds"].items()}
    cert = {}
    for arm, a in art["arms"].items():
        mkeys = {(e[0], e[1], e[2]) for e in a["map"]}
        b = bset.get(a["bound_method"]) if a["bound_method"] else None
        unc = []
        for cls, c in a["certification"].items():
            if not c["can_defer"]:
                continue
            ok = _certify_class(cls, c, mkeys, b, tau, arm)
            ensure(ok == c["certified"], (arm, cls))
            if not ok:
                f, ds = cls.split("|")
                unc.append([f, int(ds)])
        ensure(sorted(unc) == sorted(a["uncertified"]), (arm, unc, a["uncertified"]))
        cert[arm] = {
            "uncertified": unc,
            "bound_method": a["bound_method"],
            "alias_of": art["alias_of"][arm],
            "decision_table": a["decision_table"],
            "classes": {cls: c for cls, c in a["certification"].items() if c["can_defer"]},
        }
    checks.require(
        "2b: certification (edge statuses, upper bounds, uncertified classes) of all 17 CS arms "
        "= artifact",
        len(cert) == 17,
    )
    return cert


def _identity(R: Records, seeds: Sequence[int], checks: Checks) -> tuple[int, list[str], int]:
    same = same_defer = 0
    fields: list[str] = []
    for s in seeds:
        a, b = R[s]["CS:PMRT"], R[s]["never_sleep"]
        fields = sorted(k for k in a if k not in OUTCOME_SKIP)
        same += all(a[k] == b[k] for k in fields) and set(fields) == {
            k for k in b if k not in OUTCOME_SKIP
        }
        same_defer += a["policy_counts"]["defer"] == b["policy_counts"]["defer"]
    first = R[seeds[0]]["CS:PMRT"]
    num = [
        k for k in fields if isinstance(first[k], int | float) and not isinstance(first[k], bool)
    ]
    checks.require(
        "2b: CS:PMRT record identical to never_sleep on every seed (all outcome fields, deferred "
        "requests)",
        same == 160 and same_defer == 160,
        (same, same_defer),
    )
    return same, num, len(fields)


def _composition(counts: Mapping[str, Any]) -> dict[str, Any]:
    comp: dict[str, Any] = {}
    for a in COMPOSITION_ARMS:
        p = counts[a]["policy"]
        comp[a] = {
            "defer": {f: p.get(f"defer_{f}", 0) for f in FAMILIES},
            "withheld": {f: p.get(f"uncert_{f}", 0) for f in FAMILIES},
        }
    comp["never_sleep"] = {"defer_requests": counts["never_sleep"]["defer"]}
    comp["CS:PMRT"]["defer_requests"] = counts["CS:PMRT"]["defer"]
    return comp


def _cs_load(src: Sources, art: Mapping[str, Any], checks: Checks) -> tuple[Records, dict]:
    files = _res_files(src.runs / "e6p-cs-eval-1", 20)
    checks.require("2b: all 20 raw EVAL shards present", len(files) == 20, len(files))
    closes = sum(
        json.loads(f.read_text(encoding="utf-8").splitlines()[-1]).get("kind") == "close"
        for f in files
    )
    checks.require("2b: every shard ends with a close record", closes == 20, closes)
    R, heads = load_jobs(files, eval_filter(study3.SUB, "cs_stage", "eval"))
    led = run_ledger(heads, R)
    checks.require(
        "2b: eval headers carry the frozen certsafe.py sha256",
        {h.get("certsafe_sha256") for h in heads} == {art["sha256"][study3.CERTSAFE_KEY]},
    )
    alias_fill(R, art["alias_of"])
    return R, led


def _cs_criteria(st: GateA, art: Mapping[str, Any], Vd: Mapping[str, Any], checks: Checks) -> dict:
    B = st.boot
    sig = {a: v["signature"] for a, v in art["arms"].items()}
    assoc = list(study3.CS_ASSOC)
    D1 = _fd_d1(study2.d1_check(st.point, B, "CS:PMRT", assoc))
    D2 = _fd_d2(study2.d2_check(B, "CS:PMRT", assoc, sig))
    d3 = study3.d3_check(st.point, B)
    D3 = {k: d3[k] for k in ("pass", "delta3", "lb90", "ci90")}
    s1 = study3.s1_check(B)
    S1 = {k: s1[k] for k in ("pass", "rlf_ratio_ub90")}
    _check_d1(checks, "2b", D1, Vd["D1"])
    checks.require(
        "2b: D2 (Holm, 7 signature groups)",
        D2["pass"] == Vd["D2"]["pass"]
        and D2["n_hypotheses"] == Vd["D2"]["n_hypotheses"]
        and all(
            close(h["p"], Vd["D2"]["hypotheses"][b]["p"])
            and h["members"] == Vd["D2"]["hypotheses"][b]["members"]
            for b, h in D2["hypotheses"].items()
        ),
    )
    checks.require(
        "2b: D3",
        D3["pass"] == Vd["D3"]["pass"]
        and close(D3["delta3"], Vd["D3"]["delta3"])
        and close(D3["lb90"], Vd["D3"]["lb90"]),
    )
    checks.require(
        "2b: S1",
        S1["pass"] == Vd["S1"]["pass"] and close(S1["rlf_ratio_ub90"], Vd["S1"]["rlf_ratio_ub90"]),
    )
    return {"D1": D1, "D2": D2, "D3": D3, "S1": S1}


def _cs_eval(src: Sources, art: Mapping[str, Any], checks: Checks) -> dict[str, Any]:
    R, led = _cs_load(src, art, checks)
    E = read_json(src.decision / "cs_eval.json")
    Vd = read_json(src.decision / "cs_verdict.json")
    return cs_eval(R, led, art, E, Vd, checks)


def cs_eval(
    R: Records,
    led: Mapping[str, Any],
    art: Mapping[str, Any],
    E: Mapping[str, Any],
    Vd: Mapping[str, Any],
    checks: Checks,
) -> dict[str, Any]:
    """Study 3 referee evaluation of alias-filled EVAL records, asserted against `E` and `Vd`.

    Args:
        R: EVAL records per seed and arm, aliases filled.
        led: The run ledger (`run_ledger`).
        art: The certsafe artifact (decision signatures).
        E: The stored evaluation (`cs_eval.json`).
        Vd: The stored verdict (`cs_verdict.json`).
        checks: Collects the passed assertion blocks.
    """
    arms = [a for a in study3.ALL_ARMS if any(a in R[s] for s in R)]
    st = gate_a(R, arms)
    checks.require(
        "2b: 160 complete seeds 191100-191259, every arm",
        st.n == 160 == E["n_seeds"] and len(arms) == 26 and st.seeds == list(range(191100, 191260)),
    )
    checks.require(
        "2b: Gate A constants",
        close(st.V_AA, E["gate_a"]["V_AA"]) and close(st.den, E["gate_a"]["den"]),
    )
    P = fd_arms(st)
    check_arms("2b", P, E["arms"])
    checks.require(
        "2b: every arm's V, R, R*, retention, guard ratios, eligibility and 90% CIs = cs_eval.json",
        True,
    )
    crit = _cs_criteria(st, art, Vd, checks)
    E_pass = P["CS:PMRT"]["eligible"]
    verdict = study3.eval_verdict(
        k0=True,
        k0n=True,
        e=E_pass,
        d1=crit["D1"]["pass"],
        d2=crit["D2"]["pass"],
        d3=crit["D3"]["pass"],
    )["verdict"]
    checks.require("2b: verdict", verdict == Vd["verdict"]["verdict"] == "PARTIAL", verdict)
    counts = _counts(R, st.seeds, arms, E["arms"])
    checks.require("2b: deferral / unit / policy counts = cs_eval.json", True)
    same, num_fields, n_fields = _identity(R, st.seeds, checks)
    pc = counts["CS:PMRT"]["policy"]
    checks.require(
        "2b: CS:PMRT deferred only sleep-up units",
        {k for k in pc if k.startswith("defer_")} == {"defer_sleep"}
        and pc["defer_sleep"] == pc["dir_sleep_+1"],
    )
    return {
        "n": st.n,
        "seeds": [st.seeds[0], st.seeds[-1]],
        "V_AA": st.V_AA,
        "V_ref": st.V_ref,
        "ref_arm": st.ref_arm,
        "den": st.den,
        "arms": P,
        **crit,
        "E": E_pass,
        "verdict": verdict,
        "identical_seeds": same,
        "identical_fields": num_fields,
        "n_identical_fields": n_fields,
        "composition": _composition(counts),
        "secondary": study2.secondary(st, "CS:PMRT", CS_SECONDARY),
        "run": led,
        "cpu_h_eval_stored": E["cpu_h_eval"],
    }


def certsafe(src: Sources, checks: Checks) -> dict[str, Any]:
    """Study 3 figure data, asserted against the certsafe artifact, calibration and evaluation."""
    art_path = src.artifact("E6P_CERTSAFE.json")
    art = read_json(art_path)
    cal = read_json(src.decision / "cs_calib.json")
    out: dict[str, Any] = {
        "protocol_sha256": sha_lf(src.protocol(3, "E6P_CERTSAFE_PROTOCOL.md")),
        "artifact_sha256": sha_lf(art_path),
    }
    checks.require(
        "2b: artifact sha256 = .sha256 file",
        out["artifact_sha256"] == src.artifact("E6P_CERTSAFE.json.sha256").read_text().split()[0],
    )
    checks.require(
        "2b: protocol sha256 = artifact record", out["protocol_sha256"] == art["protocol"]["sha256"]
    )
    G, N, rep, n_dev = _calibration(src, cal, checks)
    tau = tau_table(G, N)
    checks.require(
        "2b: tau = .05 G_k / (max(N, 1) * 3) = artifact tau",
        all(close(tau[(f, int(d), k)], v) for f, d, k, v in art["tau"]),
    )
    out["calib"] = {
        "G": G,
        "N": {f"{f}|{d:+d}": v for (f, d), v in N.items()},
        "n_dev": n_dev,
        "repro": rep,
    }
    out["tau"] = {f"{f}|{d:+d}|{k}": v for (f, d, k), v in tau.items()}
    out["cert"] = certification(art, tau, checks)
    out["jobs"] = art["jobs"]
    bp = {(f, r, k): v for f, r, k, v in art["bounds"]["pmrt"]}[("ptx", "nbr", "rlf")]
    out["pmrt_bound_ptx_nbr_rlf"] = {
        "beta": bp["beta"],
        "se": bp["se"],
        "z": bp["z"],
        "ub90_defer_up": bound_ub(bp, -1),
        "tau": tau[("ptx", 1, "rlf")],
    }
    out["eval"] = _cs_eval(src, art, checks)
    return out


def _round_float(x: float, n: int) -> float | str | None:
    if math.isnan(x):
        return None
    return str(x) if math.isinf(x) else float(f"{float(x):.{n + 2}g}")


def rounded(x: Any, n: int = 5) -> Any:
    """JSON-ready copy: floats at n + 2 significant digits, NaN as None, infinities as text."""
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return _round_float(x, n)
    if isinstance(x, dict):
        return {str(k): rounded(v, n) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, n) for v in x]
    return x


def build(src: Sources | None = None) -> dict[str, Any]:
    """The figure data `{v4, rename, option_a, certsafe, assertions}`, unrounded.

    Raises:
        ReportCheckError: A recomputed block differs from the frozen analyzer outputs.
    """
    src = src or Sources()
    checks = Checks()
    data = {
        "v4": v4(src, checks),
        "rename": rename(src, checks),
        "option_a": opt_a(src, checks),
        "certsafe": certsafe(src, checks),
    }
    return data | {"assertions": checks.names}


def figures_json(data: Mapping[str, Any]) -> str:
    """The `figures_data.json` text of `build` output."""
    return json.dumps(rounded(data), indent=1, ensure_ascii=False)


def write(
    path: str | Path, src: Sources | None = None, newline: str = FROZEN_NEWLINE
) -> dict[str, Any]:
    """Builds the figure data and writes `figures_data.json` (UTF-8) to `path`.

    Args:
        path: Output file.
        src: Input locations; the workspace layout by default.
        newline: Line end; the frozen record has CRLF (it was written in Windows text mode).

    Returns:
        The unrounded figure data.
    """
    data = build(src)
    Path(path).write_text(figures_json(data), encoding="utf-8", newline=newline)
    return data

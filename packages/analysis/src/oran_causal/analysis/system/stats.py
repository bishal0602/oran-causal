"""Gate A statistics of the system-level referee studies, record loading and the JSON writer.

An arm's episode records are pooled over the seeds that have every anchor arm: V = 3600 sum
prot_viol / sum prot_ue_s, R = (V_AA - V) / (V_AA - V_ref) with V_ref the lowest pooled V of the
single-xApp arms, energy retention = (Ef - E) / (Ef - EA), and a guard ratio per guard KPI
against accept-all. An arm is eligible when it keeps 90 % of the energy saving and every guard
ratio is at most 1.10. Confidence intervals come from one paired seed bootstrap shared by every
arm, with V_ref re-minimised per resample.
"""

import dataclasses
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Record = dict[str, Any]
Records = dict[int, dict[str, Record]]

SINGLES = ("sub:ES", "sub:PowerES", "sub:SliceGuarantee")
A_ARM = "sub:ES+PowerES"
ANCHORS = ("freeze", A_ARM, "noarb", *SINGLES)
SUM_FIELDS = (
    "prot_viol",
    "prot_ue_s",
    "energy_j",
    "viol_ue_s",
    "ue_s",
    "nonprot_embb_viol",
    "ll_viol",
    "rlf",
)
GUARD_KEYS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
X_MATCH = 0.90
GUARD = 1.10
N_BOOT = 10_000
BOOT_TAG = 6618
EVAL_BOOT_KEY = (6624, 20)
COUNT_GROUPS = ("defer", "defer_units", "units", "collateral", "policy", "passed")


def arrays(R: Records, seeds: Sequence[int], arm: str) -> dict[str, np.ndarray]:
    """Per-seed values of the summed outcome fields of one arm."""
    return {f: np.array([float(R[s][arm][f]) for s in seeds]) for f in SUM_FIELDS}


def pooled(A: Mapping[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, Any]:
    """Pooled Gate A quantities of one arm.

    Args:
        A: Per-seed arrays from `arrays`.
        idx: Bootstrap index matrix `(B, n)`; None pools the seeds once.

    Returns:
        V, E, svr and the raw guard totals, scalars without `idx` and `(B,)` arrays with it.
    """
    s = {f: v.sum() if idx is None else v[idx].sum(1) for f, v in A.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "V": 3600.0 * s["prot_viol"] / np.maximum(s["prot_ue_s"], 1e-9),
            "E": s["energy_j"],
            "svr": 3600.0 * s["viol_ue_s"] / np.maximum(s["ue_s"], 1e-9),
            "nonprot_embb_viol": s["nonprot_embb_viol"],
            "ll_viol": s["ll_viol"],
            "rlf": s["rlf"],
        }


def ratio(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Returns a / b where b > 0; otherwise inf when a > 0 and 1 when both are 0."""
    a, b = np.asarray(a), np.asarray(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.where(a > 0, np.inf, 1.0))


def ci(x: npt.ArrayLike) -> list[float | None]:
    """The 5th and 95th percentiles of the finite values, or `[None, None]`."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))] if v.size else [None, None]


def sum_counts(R: Records, seeds: Iterable[int], arm: str, grp: str) -> dict[str, float]:
    """Totals of the numeric `policy_counts[grp]` entries of `arm` over `seeds`, sorted by key."""
    tot: dict[str, float] = {}
    for s in seeds:
        for k, v in ((R[s][arm].get("policy_counts") or {}).get(grp) or {}).items():
            if isinstance(v, int | float):
                tot[k] = tot.get(k, 0) + v
    return dict(sorted(tot.items()))


def _read_jsonl(path: Path) -> Iterable[Record]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


@dataclasses.dataclass(frozen=True)
class JobFilter:
    """Which job and header records of a referee run belong to one stage.

    Attributes:
        schema: Job record schema.
        sub: Study tag of the job records.
        stage_key: Record field naming the stage (`conf_stage`, `cs_stage`).
        stage: Stage value to keep.
        driver: Header `driver` to require; None accepts any header of the stage.
        allow_smoke: Keep smoke jobs.
    """

    schema: str
    sub: str
    stage_key: str
    stage: str
    driver: str | None = None
    allow_smoke: bool = False

    def header(self, r: Record) -> bool:
        """True for a header record of this stage."""
        return (
            r.get("kind") == "header"
            and (self.driver is None or r.get("driver") == self.driver)
            and r.get(self.stage_key) == self.stage
        )

    def job(self, r: Record) -> bool:
        """True for a job record of this stage."""
        return (
            r.get("kind") == "job"
            and r.get("schema") == self.schema
            and r.get("sub") == self.sub
            and r.get(self.stage_key) == self.stage
            and not (r.get("smoke") and not self.allow_smoke)
        )


def load_jobs(paths: Iterable[str | Path], filt: JobFilter) -> tuple[Records, list[Record]]:
    """Job records `{seed: {arm: record}}` (the last record wins) and the stage headers."""
    R: Records = {}
    heads: list[Record] = []
    for path in paths:
        for r in _read_jsonl(Path(path)):
            if filt.header(r):
                heads.append(r)
            if filt.job(r):
                R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    return R, heads


def alias_fill(R: Records, alias_of: Mapping[str, str]) -> None:
    """Adds every aliased arm as a copy of its simulated target, tagged `alias_of`, in place."""
    for s in R:
        for arm, tgt in alias_of.items():
            if tgt != arm and tgt in R[s] and arm not in R[s]:
                R[s][arm] = dict(R[s][tgt], arm=arm, alias_of=tgt)


@dataclasses.dataclass(frozen=True)
class GateA:
    """Gate A statistics of a set of arms on their common seeds.

    Attributes:
        seeds: Seeds with every anchor and every arm.
        V_AA: Pooled V of accept-all.
        V_ref: Lowest pooled V of the single-xApp arms.
        ref_arm: The single-xApp arm attaining `V_ref`.
        den: V_AA - V_ref.
        point: Per arm point statistics.
        boot: Per arm bootstrap arrays `R`, `Rstar`, `retention`, `guard`.
    """

    seeds: list[int]
    V_AA: float = float("nan")
    V_ref: float = float("nan")
    ref_arm: str = ""
    den: float = float("nan")
    point: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    boot: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of common seeds."""
        return len(self.seeds)

    def summary(self) -> dict[str, Any]:
        """The constants block `{V_AA, V_ref, ref_arm, den, n}`."""
        return {
            "V_AA": self.V_AA,
            "V_ref": self.V_ref,
            "ref_arm": self.ref_arm,
            "den": self.den,
            "n": self.n,
        }


def _eligibility(p: Mapping[str, Any], P: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    Ef, EA = P["freeze"]["E"], P[A_ARM]["E"]
    matched = bool(Ef - p["E"] >= X_MATCH * (Ef - EA) - 1e-9)
    guard_ok = bool(all(p[k] <= GUARD * P["noarb"][k] + 1e-9 for k in GUARD_KEYS))
    return {
        "matched": matched,
        "guard_ratio": {
            k: float(ratio(np.float64(p[k]), np.float64(P["noarb"][k]))) for k in GUARD_KEYS
        },
        "guard_ok": guard_ok,
        "eligible": matched and guard_ok,
    }


def _point(p: Mapping[str, Any], P: Mapping[str, Mapping[str, Any]], den: float, n: int) -> dict:
    V_AA = P["noarb"]["V"]
    Ef, EA = P["freeze"]["E"], P[A_ARM]["E"]
    Rv = float((V_AA - p["V"]) / den) if abs(den) > 1e-12 else float("nan")
    ret = float((Ef - p["E"]) / (Ef - EA)) if abs(Ef - EA) > 1e-12 else float("nan")
    el = _eligibility(p, P)
    return {
        "V": float(p["V"]),
        "R": Rv,
        "Rstar": Rv if el["eligible"] else min(Rv, 0.0),
        "retention": ret,
        **el,
        "E_kj_per_ep": float(p["E"] / n / 1e3),
    }


def _boot(
    A: Mapping[str, Mapping[str, np.ndarray]],
    point: Mapping[str, Mapping[str, Any]],
    idx: np.ndarray,
) -> dict[str, dict[str, Any]]:
    Pb = {a: pooled(A[a], idx) for a in A}
    den_b = Pb["noarb"]["V"] - np.min([Pb[x]["V"] for x in SINGLES], 0)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for a in A:
            Rb = (Pb["noarb"]["V"] - Pb[a]["V"]) / den_b
            out[a] = {
                "R": Rb,
                "Rstar": Rb if point[a]["eligible"] else np.minimum(Rb, 0.0),
                "retention": (Pb["freeze"]["E"] - Pb[a]["E"])
                / (Pb["freeze"]["E"] - Pb[A_ARM]["E"]),
                "guard": {k: ratio(Pb[a][k], Pb["noarb"][k]) for k in GUARD_KEYS},
            }
    return out


def gate_a(
    R: Records,
    arms: Iterable[str],
    n_boot: int = N_BOOT,
    key: tuple[int, int] = EVAL_BOOT_KEY,
) -> GateA:
    """Gate A statistics of the anchors and `arms` on ONE bootstrap index matrix.

    Eligibility is held at its point value, and R* = R if eligible else min(R, 0), in the point
    estimate and in every resample.

    Args:
        R: Records `{seed: {arm: record}}`.
        arms: Arms to evaluate besides the anchors.
        n_boot: Resamples; 0 skips the bootstrap.
        key: Bootstrap stream key; the generator is `default_rng([*key, n_seeds])`.

    Returns:
        The statistics; an empty `GateA` when no seed has every arm.
    """
    need = list(dict.fromkeys([*ANCHORS, *arms]))
    seeds = sorted(s for s in R if set(need) <= set(R[s]))
    if not seeds:
        return GateA(seeds=[])
    n = len(seeds)
    A = {a: arrays(R, seeds, a) for a in need}
    P = {a: pooled(A[a]) for a in need}
    ref = min(SINGLES, key=lambda x: P[x]["V"])
    den = P["noarb"]["V"] - P[ref]["V"]
    point = {a: _point(P[a], P, den, n) for a in need}
    boot = {}
    if n_boot:
        idx = np.random.default_rng([*key, n]).integers(0, n, (int(n_boot), n))
        boot = _boot(A, point, idx)
    return GateA(
        seeds=seeds,
        V_AA=float(P["noarb"]["V"]),
        V_ref=float(P[ref]["V"]),
        ref_arm=ref,
        den=float(den),
        point=point,
        boot=boot,
    )


def arm_stats(
    R: Records,
    arm: str,
    comps: Sequence[str] = (),
    n_boot: int = N_BOOT,
    tag: int = BOOT_TAG,
) -> dict[str, Any] | None:
    """Gate A statistics of one arm with paired contrasts, as the option (a) kill test K-A.

    Args:
        R: Records `{seed: {arm: record}}`.
        arm: The arm.
        comps: Contrast arms; `dR[c]` = R(c) - R(arm) on the same seeds and resamples.
        n_boot: Resamples; 0 skips the bootstrap.
        tag: Bootstrap tag; the generator is `default_rng([tag, n_seeds])`.

    Returns:
        The statistics, or None when no seed has every anchor, the arm and every contrast.
    """
    need = set(ANCHORS) | {arm} | set(comps)
    seeds = sorted(s for s in R if need <= set(R[s]))
    if not seeds:
        return None
    n = len(seeds)
    A = {a: arrays(R, seeds, a) for a in need}
    P = {a: pooled(A[a]) for a in need}
    ref = min(SINGLES, key=lambda a: P[a]["V"])
    den = P["noarb"]["V"] - P[ref]["V"]
    Ef, EA = P["freeze"]["E"], P[A_ARM]["E"]
    p = P[arm]
    el = _eligibility(p, P)
    out: dict[str, Any] = {
        "arm": arm,
        "n_seeds": n,
        "V": float(p["V"]),
        "V_AA": float(P["noarb"]["V"]),
        "V_ref": float(P[ref]["V"]),
        "ref_arm": ref,
        "den": float(den),
        "R": float((P["noarb"]["V"] - p["V"]) / den),
        "retention": float((Ef - p["E"]) / (Ef - EA)),
        "matched": el["matched"],
        "guard_ratio": el["guard_ratio"],
        "guard_ok": el["guard_ok"],
        "eligible": el["eligible"],
        "E_kj_per_ep": float(p["E"] / n / 1e3),
        "dR": {c: float((p["V"] - P[c]["V"]) / den) for c in comps},
    }
    if n_boot:
        idx = np.random.default_rng([tag, n]).integers(0, n, (n_boot, n))
        out |= _arm_boot(A, arm, comps, idx)
    return out


def _arm_boot(
    A: Mapping[str, Mapping[str, np.ndarray]], arm: str, comps: Sequence[str], idx: np.ndarray
) -> dict[str, Any]:
    Pb = {a: pooled(A[a], idx) for a in A}
    den_b = Pb["noarb"]["V"] - np.min([Pb[a]["V"] for a in SINGLES], 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "R_ci90": ci((Pb["noarb"]["V"] - Pb[arm]["V"]) / den_b),
            "retention_ci90": ci(
                (Pb["freeze"]["E"] - Pb[arm]["E"]) / (Pb["freeze"]["E"] - Pb[A_ARM]["E"])
            ),
            "guard_ratio_ci90": {k: ci(ratio(Pb[arm][k], Pb["noarb"][k])) for k in GUARD_KEYS},
            "dR_ci90": {c: ci((Pb[arm]["V"] - Pb[c]["V"]) / den_b) for c in comps},
        }


_HEADER_FIELDS: dict[str, Callable[[Record], Any]] = {
    "load_factor": lambda h: h["consts"]["load_factor"],
    "pair_stratum": lambda h: [h["consts"]["pair"], h["consts"]["stratum"]],
    "scenario_version": lambda h: h["scenario_version"],
    "e6p_spec_commit": lambda h: h["e6p_spec_commit"],
    "frozen_sha256": lambda h: h["frozen_sha256"],
    "state_load_factor": lambda h: h["state"]["load_factor"],
    "numpy_scipy_platform": lambda h: [
        h["numeric_env"].get(x) for x in ("numpy", "scipy", "platform")
    ],
}


def config_match(
    R: Records,
    new_heads: Sequence[Record],
    ref_heads: Sequence[Record],
    repro_seeds: Sequence[int],
) -> dict[str, Any]:
    """Checks that reused reference runs and new runs share plant, config and numeric platform.

    Args:
        R: Records `{seed: {arm: record}}` of both runs.
        new_heads: Headers of the new runs.
        ref_heads: Headers of the reused reference runs.
        repro_seeds: Seeds on which `noarb_repro` must equal the reused `noarb` exactly.

    Returns:
        Per header field `{ref, new, ok}`, the noarb reproduction and the job constants.
    """

    def one(hs: Sequence[Record], f: Callable[[Record], Any]) -> list[str]:
        return sorted({json.dumps(f(h), sort_keys=True, default=str) for h in hs})

    out: dict[str, Any] = {}
    for k, f in _HEADER_FIELDS.items():
        a, b = one(ref_heads, f), one(new_heads, f)
        out[k] = (
            {"ref": a, "new": b, "ok": len(a) == 1 and a == b}
            if new_heads
            else {"ref": a, "new": None, "ok": None}
        )
    rep = {
        s: max(abs(float(R[s]["noarb"][f]) - float(R[s]["noarb_repro"][f])) for f in SUM_FIELDS)
        for s in repro_seeds
        if s in R and "noarb" in R[s] and "noarb_repro" in R[s]
    }
    out["noarb_repro_max_abs_diff"] = {
        "per_seed": rep,
        "ok": all(v == 0 for v in rep.values()) if rep else None,
    }
    recs = [r for arms in R.values() for r in arms.values()]
    job_lf = {float(r["load_factor"]) for r in recs}
    job_ep = {(float(r["warmup_s"]), float(r["episode_s"])) for r in recs if not r.get("smoke")}
    out["jobs"] = {
        "load_factor": sorted(job_lf),
        "warmup_episode_s": sorted(job_ep),
        "ok": len(job_lf) == 1 and len(job_ep) == 1,
    }
    return out


def check_gt_map(cells: Sequence[Mapping[str, Any]], table: Sequence[Sequence[Any]]) -> bool:
    """True when the GT cells equal a `(family, relation, kpi, status, mean)` table, in order."""
    return [(c["family"], c["relation"], c["kpi"], c["status"], c["mean"]) for c in cells] == [
        tuple(r) for r in table
    ]


def platform_key(env: Mapping[str, Any] | None) -> tuple:
    """Numerics-relevant part of a header's numeric env: numpy, scipy, OS, glibc, machine, SIMD."""
    env = env or {}
    plat = str(env.get("platform", ""))
    return (
        env.get("numpy"),
        env.get("scipy"),
        plat.split("-", 1)[0],
        plat.rsplit("-with-", 1)[-1] if "-with-" in plat else None,
        env.get("machine"),
        tuple(env.get("simd") or ()),
    )


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


def _keys(d: Any) -> Any:
    if isinstance(d, dict):
        return {
            ("|".join(map(str, k)) if isinstance(k, tuple) else k): _keys(v) for k, v in d.items()
        }
    if isinstance(d, list | tuple):
        return [_keys(v) for v in d]
    return d


def dumps(obj: Any) -> str:
    """The analyzers' JSON text: tuple keys joined with `|`, indent 1, numpy scalars as Python."""
    return json.dumps(_keys(obj), indent=1, default=_js)


def dump(obj: Any, path: str | Path) -> None:
    """Writes `dumps(obj)` to `path` with LF line ends."""
    Path(path).write_text(dumps(obj), encoding="utf-8", newline="\n")

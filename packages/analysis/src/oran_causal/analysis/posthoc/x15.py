"""X15: why did the ungated PMRT referee's contrast with never-sleep not replicate?

Post hoc and diagnostic: no verdict changes, no frozen file changes, no new simulation.

Data: the stored per-seed EVAL records, loaded as X9 does: Study 2 is the option (a) EVAL
(seeds 187000-187159) and Study 3 the certified-safe EVAL (seeds 191100-191259).

Reproduction comes first and is a hard stop: the frozen Gate A statistics (10 000 draws,
`default_rng([6624, 20, 160])`) must reproduce `conf_eval.json` and `cs_eval.json` for `noarb`,
`never_sleep` and `MG:PMRT` (and `MG:GT`, Study 2 only: Study 3 has no `MG:GT` arm) and the frozen
`MG:PMRT - never_sleep` contrasts, and the values quoted in the X15 declaration.

Questions: Q1 plant-level exchangeability of the two blocks; Q2 the per-seed contrasts and what
they depend on; Q3 the mechanism (rejections, accept-all violation level); Q4 the pooled
320-seed best estimate. The reading rule picks branch A (chance plus heavy tails) or B (seed-block
sensitivity of the referee).
"""

import argparse
import csv
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from oran_causal.analysis.posthoc.paths import ARTIFACTS, RECORDS, write_text
from oran_causal.analysis.posthoc.x9 import REF, eval_files, repro
from oran_causal.analysis.system import stats, study2, study3

N_PERM = stats.N_BOOT
KEY = (stats.EVAL_BOOT_KEY[0], stats.EVAL_BOOT_KEY[1])
BLOCKS = ("study2", "study3")
BLOCK_SPLIT = 190000
RTOL = 1e-9
DECLARED = "scratchpad/xmethod/EXTRAS_PROTOCOL.md amendment 2026-10-11 X15"
QUOTED = {
    "study2": {
        "MG:PMRT.R": (0.470, 3),
        "MG:PMRT.R_ci90": ((0.346, 0.594), 3),
        "dR": (0.249, 3),
        "dR_ci90": ((0.139, 0.362), 3),
        "rlf": (1.31, 2),
        "V_AA": (157.67, 2),
        "never_sleep.R": (0.221, 3),
    },
    "study3": {
        "MG:PMRT.R": (0.201, 3),
        "MG:PMRT.R_ci90": ((0.025, 0.363), 3),
        "dR": (-0.055, 3),
        "dR_ci90": ((-0.216, 0.091), 3),
        "rlf": (1.05, 2),
        "V_AA": (175.30, 2),
        "never_sleep.R": (0.256, 3),
    },
}
DEFINITIONS = (
    "(frozen metric definitions). prot_ue_s is constant across arms within a seed "
    "(checked), so the\n"
    "frozen pooled V of an arm is the prot_ue_s-weighted mean of the per-seed V_s "
    "= 3600 prot_viol / prot_ue_s, and\n"
    "  dR_s(a - b) = (V_s(b) - V_s(a)) / den_block      (den_block = the block's "
    "frozen V_AA - V_ref)\n"
    "has prot_ue_s-weighted mean equal to the frozen pooled R(a) - R(b) exactly. "
    "R(a) is dR(a - noarb). Per-seed\n"
    "contribution c_s = w_s dR_s / sum w (sums to the pooled contrast). sd / skew "
    "/ quantiles are of the unweighted dR_s.\n"
    "Tests and intervals (RNG convention of the frozen eval, key (6624, 20, n), a "
    "fourth sub-key per X15 test):\n"
    "  permutation of the block labels, 10 000 draws, two-sided, statistic = "
    "difference of the block (weighted) means;\n"
    "  KS = scipy.stats.ks_2samp (two-sided, exact/asymp auto) on the per-seed "
    "values;\n"
    "  pooled (Q4) = the frozen pooled formulas on all 320 seeds; 90 % CI from a "
    "block-stratified seed bootstrap\n"
    "  (each draw resamples 160 seeds within each block), 10 000 draws, "
    "default_rng([6624, 20, 320]).\n"
    "Leave-top-5-out: the frozen pooled contrast (den recomputed with the frozen "
    "rule) on the block without the 5 seeds\n"
    "with the largest c_s, and (descriptive) without the 5 with the largest |c_s|."
)

CONTRASTS = {
    "MG:PMRT-never_sleep": ("MG:PMRT", REF, BLOCKS),
    "MG:GT-never_sleep": ("MG:GT", REF, ("study2",)),
    "never_sleep-noarb": (REF, "noarb", BLOCKS),
    "MG:PMRT-noarb": ("MG:PMRT", "noarb", BLOCKS),
}


class ReproductionError(RuntimeError):
    """The stored records do not reproduce the frozen eval."""


def load(
    block: str, records: Path, artifacts: Path
) -> tuple[stats.Records, dict[str, Any], list[str], int]:
    """The records, frozen eval JSON, compared arms and file count of one block."""
    dec = records / "system_artifacts" / "decision"
    if block == "study2":
        files = eval_files(records, "e6p-conf-eval-1")
        art = json.loads((artifacts / "E6P_CONF_MAPS.json").read_text(encoding="utf-8"))
        R, _ = stats.load_jobs(files, study2.EVAL_FILTER)
        stats.alias_fill(R, art["alias_of"])
        frozen = json.loads((dec / "conf_eval.json").read_text(encoding="utf-8"))
        arms = ["noarb", REF, "MG:PMRT", "MG:GT"]
    else:
        files = eval_files(records, "e6p-cs-eval-1")
        R, _ = stats.load_jobs(files, study3.job_filter("eval"))
        art = json.loads((artifacts / "E6P_CERTSAFE.json").read_text(encoding="utf-8"))
        stats.alias_fill(R, art["alias_of"])
        frozen = json.loads((dec / "cs_eval.json").read_text(encoding="utf-8"))
        arms = ["noarb", REF, "MG:PMRT"]
    return R, frozen, arms, len(files)


def _contrast_ok(block: str, frozen: Mapping[str, Any], d_r: float, dci: list) -> tuple[Any, bool]:
    if block == "study2":
        sec = frozen["secondary"][REF]
        return sec, bool(
            np.isclose(sec["dR"], d_r, rtol=RTOL) and np.allclose(sec["ci90"], dci, rtol=RTOL)
        )
    sec = frozen["secondary"]["MG:PMRT"]
    return sec, bool(
        np.isclose(-sec["dR"], d_r, rtol=RTOL)
        and np.allclose([-sec["ci90"][1], -sec["ci90"][0]], dci, rtol=RTOL, atol=1e-12)
    )


def reproduce(
    block: str, R: stats.Records, frozen: Mapping[str, Any], arms: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduction of one block and its frozen summary."""
    st = stats.gate_a(R, arms, stats.N_BOOT)
    rp = repro(st, frozen, arms)
    P, B = st.point, st.boot
    d_r = P["MG:PMRT"]["R"] - P[REF]["R"]
    dci = stats.ci(B["MG:PMRT"]["R"] - B[REF]["R"])
    sec, ok_c = _contrast_ok(block, frozen, d_r, dci)
    got: dict[str, Any] = {
        "MG:PMRT.R": P["MG:PMRT"]["R"],
        "MG:PMRT.R_ci90": stats.ci(B["MG:PMRT"]["R"]),
        "dR": d_r,
        "dR_ci90": dci,
        "rlf": P["MG:PMRT"]["guard_ratio"]["rlf"],
        "V_AA": st.V_AA,
        "never_sleep.R": P[REF]["R"],
    }
    q = []
    for k, (v, nd) in QUOTED[block].items():
        g = got[k]
        ok = (
            (tuple(round(x, nd) for x in g) == tuple(v))
            if isinstance(v, tuple)
            else round(g, nd) == v
        )
        q.append({"field": k, "quoted": v, "got": g, "ok": bool(ok)})
    rp["contrast_pmrt_minus_never_sleep"] = {
        "frozen": sec,
        "got": {"dR": d_r, "ci90": dci},
        "ok": ok_c,
    }
    rp["quoted"] = q
    rp["ok"] = bool(rp["ok"] and ok_c and all(x["ok"] for x in q))
    summ = {
        a: {
            "V": P[a]["V"],
            "R": P[a]["R"],
            "R_ci90": stats.ci(B[a]["R"]),
            "rlf_ratio": P[a]["guard_ratio"]["rlf"],
            "rlf_ci90": stats.ci(B[a]["guard"]["rlf"]),
            "dR_vs_never_sleep": P[a]["R"] - P[REF]["R"],
            "dR_vs_never_sleep_ci90": stats.ci(B[a]["R"] - B[REF]["R"]),
        }
        for a in arms
    }
    return rp, {
        "seeds": st.seeds,
        "V_AA": st.V_AA,
        "V_ref": st.V_ref,
        "ref_arm": st.ref_arm,
        "den": st.den,
        "arms": summ,
    }


def vseed(R: stats.Records, seeds: Sequence[int], arm: str) -> np.ndarray:
    """Per-seed protected violation V_s of an arm."""
    return np.array([3600.0 * R[s][arm]["prot_viol"] / R[s][arm]["prot_ue_s"] for s in seeds])


def wseed(R: stats.Records, seeds: Sequence[int]) -> np.ndarray:
    """Per-seed weights prot_ue_s, checked constant across arms within a seed."""
    w = np.array([float(R[s]["noarb"]["prot_ue_s"]) for s in seeds])
    if not all(len({R[s][a]["prot_ue_s"] for a in R[s]}) == 1 for s in seeds):
        raise ReproductionError("prot_ue_s varies across arms")
    return w


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean."""
    return float(np.sum(w * x) / np.sum(w))


def describe(x: np.ndarray, w: np.ndarray, seeds: Sequence[int], k: int = 5) -> dict[str, Any]:
    """Moments, quantiles and the largest contributions of per-seed values."""
    c = w * x / w.sum()
    o = np.argsort(-c)
    pos = c[c > 0].sum()
    return {
        "n": int(x.size),
        "mean_w": wmean(x, w),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "skew": float(sps.skew(x, bias=False)),
        "kurtosis_excess": float(sps.kurtosis(x, bias=False)),
        "q": {str(p): float(np.quantile(x, p)) for p in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)},
        "top5_pos": [
            {"seed": int(seeds[i]), "value": float(x[i]), "contribution": float(c[i])}
            for i in o[:k]
        ],
        "top5_neg": [
            {"seed": int(seeds[i]), "value": float(x[i]), "contribution": float(c[i])}
            for i in o[::-1][:k]
        ],
        "top5_pos_contribution": float(c[o[:k]].sum()),
        "top5_neg_contribution": float(c[o[::-1][:k]].sum()),
        "share_of_positive_mass_top5": float(c[o[:k]].sum() / pos) if pos > 0 else None,
    }


def perm_test(
    xa: np.ndarray, wa: np.ndarray, xb: np.ndarray, wb: np.ndarray, sub: int, n: int = N_PERM
) -> dict[str, Any]:
    """Two-sided permutation test of the difference of the (prot_ue_s-weighted) block means."""
    x, w = np.r_[xa, xb], np.r_[wa, wb]
    na = xa.size
    obs = wmean(xa, wa) - wmean(xb, wb)
    rng = np.random.default_rng([*KEY, x.size, sub])
    cnt = 0
    for _ in range(n):
        p = rng.permutation(x.size)
        ia, ib = p[:na], p[na:]
        d = wmean(x[ia], w[ia]) - wmean(x[ib], w[ib])
        cnt += abs(d) >= abs(obs) - 1e-12
    pv = (cnt + 1) / (n + 1)
    return {
        "diff": obs,
        "p": pv,
        "p_mc_se": float(np.sqrt(pv * (1 - pv) / n)),
        "n_perm": n,
        "rng": f"default_rng([6624, 20, {x.size}, {sub}])",
    }


def ks(xa: np.ndarray, xb: np.ndarray) -> dict[str, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p."""
    r = sps.ks_2samp(xa, xb)
    return {"D": float(r.statistic), "p": float(r.pvalue)}


def drop_top(
    R: stats.Records, seeds: Sequence[int], a: str, b: str, c: np.ndarray, k: int = 5
) -> dict[str, Any]:
    """The pooled contrast a - b after dropping the k seeds of largest c and of largest |c|."""
    out = {}
    for nm, order in (("largest_c", np.argsort(-c)), ("largest_abs_c", np.argsort(-np.abs(c)))):
        dropped = set(order[:k].tolist())
        keep = [s for i, s in enumerate(seeds) if i not in dropped]
        st = stats.gate_a({s: R[s] for s in keep}, [a, b], 0)
        out[nm] = {
            "dropped": [int(seeds[i]) for i in order[:k]],
            "n": len(keep),
            "dR": st.point[a]["R"] - st.point[b]["R"],
            "den": st.den,
        }
    return out


def rejections(
    R: stats.Records, seeds: Sequence[int], arm: str = "MG:PMRT"
) -> dict[str, np.ndarray]:
    """Per-seed deferral counts of an arm: ptx, sleep, and the request-level st_rej and st_req."""

    def g(s: int, pre: str) -> float:
        d = R[s][arm]["policy_counts"].get("defer", {})
        return sum(v for k, v in d.items() if k.startswith(pre))

    return {
        "ptx": np.array([g(s, "ptx") for s in seeds], float),
        "sleep": np.array([g(s, "sleep") for s in seeds], float),
        "st_rej": np.array([R[s][arm]["st_rej"] for s in seeds], float),
        "st_req": np.array([R[s][arm]["st_req"] for s in seeds], float),
    }


def _q1(
    S2: Mapping[str, Any], S3: Mapping[str, Any], data: Mapping[str, Any], sub: Any
) -> dict[str, Any]:
    q1: dict[str, Any] = {}
    for arm in ("noarb", REF):
        a, b = S2["V"][arm], S3["V"][arm]
        q1[arm] = {
            "study2": describe(a, S2["w"], S2["seeds"]),
            "study3": describe(b, S3["w"], S3["seeds"]),
            "perm_mean": perm_test(a, S2["w"], b, S3["w"], next(sub)),
            "ks": ks(a, b),
        }
    lf = {
        k: np.array([d["R"][s]["noarb"]["load_factor"] for s in d["seeds"]])
        for k, d in data.items()
    }
    q1["load_factor"] = {
        "study2": describe(lf["study2"], S2["w"], S2["seeds"]),
        "study3": describe(lf["study3"], S3["w"], S3["seeds"]),
        "perm_mean": perm_test(lf["study2"], S2["w"], lf["study3"], S3["w"], next(sub)),
        "ks": ks(lf["study2"], lf["study3"]),
    }
    q1["prot_ue_s"] = {
        "study2_mean": float(S2["w"].mean()),
        "study3_mean": float(S3["w"].mean()),
        "ks": ks(S2["w"], S3["w"]),
        "note": "per-seed protected UE-seconds (the stored load proxy; load_factor is constant)",
    }
    q1_ps = [q1[k][t]["p"] for k in ("noarb", REF) for t in ("perm_mean", "ks")]
    q1["block_difference_p_lt_01"] = bool(min(q1_ps) < 0.01)
    q1["min_p"] = float(min(q1_ps))
    return q1


def _q2(
    S2: Mapping[str, Any], S3: Mapping[str, Any], data: Mapping[str, Any], sub: Any
) -> tuple[dict[str, Any], dict[tuple, np.ndarray]]:
    q2: dict[str, Any] = {}
    dr: dict[tuple, np.ndarray] = {}
    for nm, (a, b, blocks) in CONTRASTS.items():
        q2[nm] = {}
        for bl in blocks:
            d = data[bl]
            x = (d["V"][b] - d["V"][a]) / d["den"]
            dr[(nm, bl)] = x
            c = d["w"] * x / d["w"].sum()
            q2[nm][bl] = {
                **describe(x, d["w"], d["seeds"]),
                "leave_top5_out": drop_top(d["R"], d["seeds"], a, b, c),
                "dV_mean_w": wmean(d["V"][b] - d["V"][a], d["w"]),
            }
        if len(blocks) == 2:
            xa, xb = dr[(nm, "study2")], dr[(nm, "study3")]
            q2[nm]["perm_block_diff"] = perm_test(xa, S2["w"], xb, S3["w"], next(sub))
            va = S2["V"][b] - S2["V"][a]
            vb = S3["V"][b] - S3["V"][a]
            q2[nm]["perm_block_diff_dV"] = perm_test(va, S2["w"], vb, S3["w"], next(sub))
            q2[nm]["ks"] = ks(xa, xb)
        else:
            q2[nm]["note"] = "Study 3 has no MG:GT arm (E6P_CERTSAFE_PROTOCOL.md s.4); Study 2 only"
    return q2, dr


def _q3_block(d: Mapping[str, Any], x: np.ndarray, r_ns: np.ndarray) -> tuple[dict, dict]:
    vaa = d["V"]["noarb"]
    rej = rejections(d["R"], d["seeds"])
    qs = np.quantile(r_ns, [0.25, 0.5, 0.75])
    quart = np.searchsorted(qs, r_ns, side="right")
    qv = np.quantile(vaa, [0.25, 0.5, 0.75])
    quart_v = np.searchsorted(qv, vaa, side="right")
    c = d["w"] * x / d["w"].sum()

    def sp(u: np.ndarray, v: np.ndarray) -> dict[str, float]:
        return dict(zip(("rho", "p"), map(float, sps.spearmanr(u, v)), strict=True))

    out = {
        "spearman_dR_vs_V_AA": sp(x, vaa),
        "load_factor": "constant 1.7555 on every seed of both blocks (no per-seed information)",
        "spearman_dR_vs_prot_ue_s": sp(x, d["w"]),
        "ue_s": "constant on every seed of both blocks (no per-seed information)",
        "spearman_dR_vs_never_sleep_R_s": sp(x, r_ns),
        "ols_dR_on_V_AA": {"slope": float(np.polyfit(vaa, x, 1)[0])},
        "by_V_AA_quartile": [
            {
                "quartile": q + 1,
                "n": int((quart_v == q).sum()),
                "mean_w_dR": wmean(x[quart_v == q], d["w"][quart_v == q]),
                "share_of_block_dR": float(c[quart_v == q].sum()),
            }
            for q in range(4)
        ],
        "by_never_sleep_R_s_quartile": [
            {
                "quartile": q + 1,
                "n": int((quart == q).sum()),
                "mean_w_dR": wmean(x[quart == q], d["w"][quart == q]),
                "mean_never_sleep_R_s": float(r_ns[quart == q].mean()),
                "share_of_block_dR": float(c[quart == q].sum()),
            }
            for q in range(4)
        ],
        "block_dR": float(c.sum()),
        "mgpmrt_rejections_per_seed": {
            k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(np.median(v))}
            for k, v in rej.items()
        },
        "mgpmrt_rejection_rate": float(rej["st_rej"].sum() / rej["st_req"].sum()),
    }
    return out, rej


def _q3(
    S2: Mapping[str, Any],
    S3: Mapping[str, Any],
    data: dict[str, Any],
    dr: Mapping[tuple, np.ndarray],
    sub: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    q3: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for bl in BLOCKS:
        d = data[bl]
        x = dr[("MG:PMRT-never_sleep", bl)]
        r_ns = dr[("never_sleep-noarb", bl)]
        q3[bl], rej = _q3_block(d, x, r_ns)
        d["rej"] = rej
        for i, s in enumerate(d["seeds"]):
            rows.append(
                {
                    "block": bl,
                    "seed": s,
                    "w_prot_ue_s": d["w"][i],
                    "ue_s": d["R"][s]["noarb"]["ue_s"],
                    **{f"V_{a}": d["V"][a][i] for a in d["arms"]},
                    **{f"dR_{nm}": dr[(nm, bl)][i] for nm in CONTRASTS if (nm, bl) in dr},
                    **{f"rej_{k}": v[i] for k, v in rej.items()},
                }
            )
    ones = np.ones(160)
    q3["rejections_block_diff"] = {
        k: {
            "perm_mean": perm_test(S2["rej"][k], ones, S3["rej"][k], ones, next(sub)),
            "ks": ks(S2["rej"][k], S3["rej"][k]),
        }
        for k in ("ptx", "sleep", "st_rej", "st_req")
    }
    q3["note"] = (
        "dR(PMRT - never_sleep)_s = R_s(PMRT) - R_s(never_sleep), so its correlation with "
        "never-sleep's own per-seed R_s is partly mechanical (shared term); read the quartile "
        "table with that in mind."
    )
    return q3, rows


def _q4(data: Mapping[str, Any]) -> dict[str, Any]:
    RR: stats.Records = {}
    for bl in BLOCKS:
        for s in data[bl]["seeds"]:
            RR[s] = data[bl]["R"][s]
    arms4 = ["noarb", REF, "MG:PMRT"]
    st = stats.gate_a(RR, arms4, 0)
    if st.n != 320:
        raise ReproductionError(f"pooled seeds {st.n} != 320")
    need = list(dict.fromkeys([*stats.ANCHORS, *arms4]))
    seeds = st.seeds
    A = {a: stats.arrays(RR, seeds, a) for a in need}
    blk = np.array([0 if s < BLOCK_SPLIT else 1 for s in seeds])
    i2, i3 = np.where(blk == 0)[0], np.where(blk == 1)[0]
    rng = np.random.default_rng([*KEY, 320])
    idx = np.concatenate(
        [
            i2[rng.integers(0, i2.size, (N_PERM, i2.size))],
            i3[rng.integers(0, i3.size, (N_PERM, i3.size))],
        ],
        axis=1,
    )
    Pb = {a: stats.pooled(A[a], idx) for a in need}
    den_b = Pb["noarb"]["V"] - np.min([Pb[x]["V"] for x in stats.SINGLES], 0)
    Rb = {a: (Pb["noarb"]["V"] - Pb[a]["V"]) / den_b for a in arms4}
    rlf_b = stats.ratio(Pb["MG:PMRT"]["rlf"], Pb["noarb"]["rlf"])
    P = st.point
    return {
        "n": 320,
        "V_AA": st.V_AA,
        "V_ref": st.V_ref,
        "ref_arm": st.ref_arm,
        "den": st.den,
        "R": {a: {"R": P[a]["R"], "ci90": stats.ci(Rb[a])} for a in arms4},
        "dR_MG:PMRT-never_sleep": {
            "dR": P["MG:PMRT"]["R"] - P[REF]["R"],
            "ci90": stats.ci(Rb["MG:PMRT"] - Rb[REF]),
        },
        "rlf_ratio_MG:PMRT": {"ratio": P["MG:PMRT"]["guard_ratio"]["rlf"], "ci90": stats.ci(rlf_b)},
        "rlf_ratio_never_sleep": {
            "ratio": P[REF]["guard_ratio"]["rlf"],
            "ci90": stats.ci(stats.ratio(Pb[REF]["rlf"], Pb["noarb"]["rlf"])),
        },
        "MG:GT": "not pooled: Study 3 has no MG:GT arm; Study 2 value in frozen_summary.study2",
        "bootstrap": "block-stratified seed bootstrap, 10 000 draws, default_rng([6624, 20, 320]); "
        "160 + 160",
    }


def _reading(q1: Mapping[str, Any], q2: Mapping[str, Any]) -> dict[str, Any]:
    p2 = q2["MG:PMRT-never_sleep"]["perm_block_diff"]["p"]
    if not q1["block_difference_p_lt_01"] and p2 >= 0.01:
        branch = (
            "A: chance plus heavy tails; the pooled 320-seed estimate is the reported best estimate"
        )
    elif q1["block_difference_p_lt_01"]:
        branch = "B: seed-block sensitivity of the referee (mechanism from Q3)"
    else:
        branch = (
            "neither as worded: no Q1 block difference but Q2 permutation p < .01 (the rule is "
            "silent); reported as is"
        )
    return {
        "q1_block_difference": q1["block_difference_p_lt_01"],
        "q1_rule": "block difference = any of the four Q1 tests (permutation of the mean, KS; "
        "accept-all V and never-sleep V) with p < .01",
        "q2_perm_p": p2,
        "branch": branch,
    }


def run(records: Path = RECORDS, artifacts: Path = ARTIFACTS) -> dict[str, Any]:
    """X15 from the frozen record sets.

    Returns:
        `{"result", "rows"}`: the JSON output and the per-(block, seed) CSV rows.

    Raises:
        ReproductionError: If a block does not reproduce its frozen eval.
    """
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    data: dict[str, Any] = {}
    rep: dict[str, Any] = {
        "schema": "x15-replication/1",
        "declared": DECLARED,
        "status": "POST HOC, DIAGNOSTIC (no verdict changes)",
        "repro": {},
        "frozen_summary": {},
    }
    for b in BLOCKS:
        R, frozen, arms, nf = load(b, records, artifacts)
        rp, summ = reproduce(b, R, frozen, arms)
        rep["repro"][b] = {**rp, "n_files": nf}
        rep["frozen_summary"][b] = {k: v for k, v in summ.items() if k != "seeds"}
        if not rp["ok"]:
            raise ReproductionError(
                f"X15 {b}: the stored records do not reproduce the frozen eval: "
                + json.dumps(rp, indent=1, default=str)
            )
        seeds = summ["seeds"]
        data[b] = {
            "R": R,
            "seeds": seeds,
            "den": summ["den"],
            "w": wseed(R, seeds),
            "V": {a: vseed(R, seeds, a) for a in arms},
            "arms": arms,
        }
    S2, S3 = data["study2"], data["study3"]
    sub = iter(range(1, 100))
    q1 = _q1(S2, S3, data, sub)
    q2, dr = _q2(S2, S3, data, sub)
    q3, rows = _q3(S2, S3, data, dr, sub)
    q4 = _q4(data)
    rep["reading_rule"] = _reading(q1, q2)
    rep.update({"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "definitions": DEFINITIONS})
    return {"result": rep, "rows": rows}


def write(out: Mapping[str, Any], out_dir: str | Path) -> None:
    """Writes `x15_replication.json` and `x15_replication.csv`."""
    d = Path(out_dir)
    write_text(d / "x15_replication.json", json.dumps(out["result"], indent=1, default=float))
    cols = list(dict.fromkeys(k for r in out["rows"] for k in r))
    with (d / "x15_replication.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(out["rows"])


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: recompute X15 from the frozen record sets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.records), Path(a.artifacts))
    write(out, a.out)
    rep = out["result"]
    print("branch:", rep["reading_rule"]["branch"])
    print(json.dumps({"Q1min_p": rep["Q1"]["min_p"], "Q4": rep["Q4"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

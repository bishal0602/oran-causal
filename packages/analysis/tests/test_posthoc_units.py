"""Post hoc analyses on small synthetic inputs: the rules each module encodes."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from oran_causal.analysis.posthoc import (
    x1_runtime as rt,
)
from oran_causal.analysis.posthoc import (
    x2x3,
    x5x6,
    x8,
    x9,
    x11,
    x13,
    x14,
    x15,
    x16,
)
from oran_causal.core.ridge import ResidualConfig, predictable_residuals


def _variant(group: int, shift: int, p_primary: list[float], declared: list[bool]) -> dict:
    decl = {
        "loadsp_c+wby1s": {
            f"h{i}": [p, d, 1 if d else 0]
            for i, (p, d) in enumerate(zip(p_primary, declared, strict=True))
        },
        "plain_c+by": {"h0": [0.5, False, 0]},
    }
    hyp = {"h0": {"status": "tested", "stats": {"plain_c": {"p2": 0.04, "z_raw": 2.0}}}}
    return {
        "group": group,
        "shift": shift,
        "B": 9999,
        "n_units": 10,
        "wall_s": 1.0,
        "hyp": hyp,
        "decl": decl,
    }


def _write_variants(d: Path, rows: list[dict]) -> None:
    for i in range(2):
        text = "\n".join(json.dumps(r) for r in rows[i::2])
        (d / f"res_{i}.jsonl").write_text(text + "\n{torn", encoding="utf-8")


def test_x8_summary_rates_and_checks(tmp_path):
    rows = [_variant(0, k, [0.5, 0.0005, 0.02], [False, False, False]) for k in range(1, 5)]
    rows.append(_variant(1, 1, [0.5, 0.0001, 0.00009], [False, False, True]))
    rows.append(rows[0])
    _write_variants(tmp_path, rows)
    s = x8.summarize(tmp_path)
    assert s["n_variants"] == 5
    assert s["B"] == [9999]
    c = s["combos"]["loadsp_c+wby1s"]
    assert c["n_p"] == 15
    assert c["n_decl"] == 1
    assert c["variants_with_decl"] == 1
    assert c["var_decl_frac"] == pytest.approx(0.2)
    assert c["n_le_0.001"] == 6
    assert c["rate_le_0.001"] == pytest.approx(0.4)
    assert s["primary"] == "loadsp_c+wby1s"
    assert s["checks"]["X8a_var_decl"]["pass"] is False
    assert s["checks"]["X8b_rate_1e3"]["pass"] is False
    norm = s["arms_p2_vs_normal"]["plain_c"]
    assert norm["n"] == 5
    assert norm["redraw_le_0.05"] == 1.0
    assert norm["normal_le_0.05"] == pytest.approx(1.0)
    assert norm["normal_le_0.01"] == 0.0


def test_x8_summary_of_nothing(tmp_path):
    assert x8.summarize(tmp_path) == {
        "n_variants": 0, "B": [], "criteria": x8.CRIT, "combos": {},
    }  # fmt: skip


def _rec(arm: str, n: int, seed: int, cpu: float, host: str = "kaggle", **extra) -> dict:
    return {
        "key": f"{arm}|E2|R2|k0.25|n{n}|s{seed}",
        "arm": arm,
        "method": arm,
        "role": "timing",
        "status": "ok",
        "cpu_s": cpu,
        "wall_s": cpu,
        "peak_rss_mb": 100.0,
        "rss_scope": "job",
        "job": {"world": "E2", "regime": "R2", "lam": None, "kappa": 0.25, "n": n, "seed": seed},
        "host": {"platform": host, "cpu_model": "Xeon", "node": host},
        "spec_name": "s",
        "code": {"commit": "abcdef0123"},
        **extra,
    }


def _write(path: Path, recs: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return path


def test_x1_table_t3_marks_the_first_over_budget_n_and_larger(tmp_path):
    ref = "kaggle|Xeon"
    recs = [_rec("a", 500, s, 10.0) for s in range(3)] + [
        _rec("a", 1000, s, 8000.0) for s in range(3)
    ]
    recs += [_rec("a", 4000, 0, 9000.0)] + [_rec("b", 500, s, 1.0) for s in range(3)]
    path = _write(tmp_path / "r.jsonl", recs)
    loaded, info = rt.load([path])
    assert info["lines"] == 10
    assert info["duplicates"] == 0
    out = rt.build(loaded, info, ref=ref, generated_utc="t")
    st = out["feasibility"]["a"]
    assert st["first_infeasible_n"] == 1000
    assert st["first_infeasible_cost"] == 8000.0
    assert st["by_n"]["500"]["status"] == "feasible"
    assert st["by_n"]["1000"]["status"] == "infeasible"
    assert st["by_n"]["4000"]["status"] == "infeasible (smaller n infeasible)"
    assert out["feasibility"]["b"]["first_infeasible_n"] is None
    assert out["spec_t3_fields"] == {"a": {"max_n": 500, "t3_cost_cpu_s": {"1000": 8000.0}}}
    assert out["generated_utc"] == "t"
    md = rt.render_md(out, "title")
    assert md.startswith("# Experiment C: runtime per method (title)")
    assert "**INF**" in md


def test_x1_converts_other_hosts_with_calibration_factors(tmp_path):
    ref = "kaggle|Xeon"
    pair = [_rec("a", 1000, s, 10.0) for s in range(3)] + [
        _rec("a", 1000, s, 20.0, host="vps") for s in range(3)
    ]
    path = _write(tmp_path / "p.jsonl", pair)
    _recs, info = rt.load([path])
    assert info["conflicts"] == 3
    assert len(info["copies"]) == 3
    allc = [c for cs in info["copies"].values() for c in cs]
    cal = rt.calibrate(allc, ref)
    f = cal["factors"]["a"]["vps|Xeon"]
    assert f["f"] == pytest.approx(0.5)
    assert f["n_units"] == 3
    assert cal["source"] == "paired_units"
    assert cal["factors"]["*"]["vps|Xeon"]["arms"] == ["a"]
    only_host = _write(tmp_path / "h.jsonl", [_rec("a", 1000, 9, 40.0, host="vps")])
    recs2, info2 = rt.load([only_host])
    fac = {("a", "vps|Xeon"): {**f, "source": "calib"}}
    out = rt.build(recs2, info2, ref=ref, factors=fac, observational=False)
    assert out["table"][0]["cost_ref"]["max"] == pytest.approx(20.0)
    assert out["table"][0]["factor_sources"] == ["calib"]
    unconverted = rt.build(recs2, info2, ref=ref, observational=False)
    assert unconverted["table"][0]["n_unconverted"] == 1


def test_x1_load_labels_the_inputs_and_hashes_them(tmp_path):
    path = _write(tmp_path / "r.jsonl", [_rec("a", 500, 0, 1.0)])
    _, info = rt.load([path], {str(path.absolute()): "old/path/r.jsonl"})
    assert info["files"][0]["path"] == "old/path/r.jsonl"
    assert info["files"][0]["sha256"] == rt.file_sha256(path)
    assert info["files"][0]["lines"] == 1


def test_x1_gpu_arms_are_read_in_wall_seconds():
    assert rt.cost_field({"budget_wall_s": 3600}) == "wall_s"
    assert rt.cost_field({}) == "cpu_s"
    s = rt.slim({**_rec("g", 500, 0, 5.0), "wall_s": 9.0})
    assert rt.unit_cost(s, "wall_s") == 9.0
    assert rt.unit_cost(s, "cpu_s") == 5.0


def test_x2x3_told_variants_are_validated():
    assert x2x3.check_told({"width": 0.8}) == ("width", 0.8)
    assert x2x3.check_told({"width": 2, "centre": "true", "redraw": "told"}) == ("width", 2.0)
    for bad in (
        {},
        {"width": 1, "shift": 1},
        {"shift": 1, "centre": "told"},
        {"lam": -1},
        {"x": 1},
    ):
        with pytest.raises(ValueError, match="told"):
            x2x3.check_told(bad)


def test_x2x3_design_labels():
    x2 = {
        "extras": {"experiment": "X2", "design": {"delta": 0.1, "n_blocks": 20}},
        "arms": {"a": {}},
    }
    assert x2x3.design_label(x2, "a") == "delta 0.1 / 20 blocks"
    assert x2x3.design_key(x2, "a") == [0.1, 20]
    x3 = {
        "extras": {"experiment": "X3"},
        "arms": {"e": {}, "w": {"told": {"width": 0.8}}, "s": {"told": {"shift": 0.02}}},
    }
    assert [x2x3.design_label(x3, a) for a in "ews"] == [
        "exact",
        "width x0.8",
        "switch +2 % of a block",
    ]
    assert x2x3.design_key(x3, "e") == ["exact", 1.0]


def test_x2x3_trend_words():
    assert x2x3.trend_word([0.1, 0.2, 0.3]) == "rises"
    assert x2x3.trend_word([0.3, 0.2, 0.2]) == "falls"
    assert x2x3.trend_word([0.2, 0.2]) == "does not change"
    assert x2x3.trend_word([0.1, 0.3, 0.2]) == "is not monotone"
    assert "not reported" in x2x3.trend_word([0.1, None])


def _rate(rate: float, validity: str = "VALID") -> dict:
    return {"rate": rate, "hits": 1, "n": 10, "ci": [0.0, 0.5], "validity": validity}


def test_x2_trend_and_x3_curve_read_the_pooled_rows():
    pooled = [
        {
            "experiment": "X2",
            "arm": "a",
            "design_key": [d, 20],
            "null_decl": _rate(r),
            "null_raw": None,
            "plac_raw": None,
        }
        for d, r in ((0.02, 0.1), (0.05, 0.2), (0.10, 0.3), (0.20, 0.4))
    ]
    pooled.append(
        {
            "experiment": "X2",
            "arm": "a",
            "design_key": [0.1, 5],
            "null_decl": _rate(0.6),
            "null_raw": None,
            "plac_raw": None,
        }
    )
    t = x2x3.x2_trend(pooled)
    assert [x["rate"] for x in t] == ["null_decl"]
    assert t[0]["trend"] == "rises"
    assert t[0]["blocks_5"]["rate"] == 0.6
    assert t[0]["blocks_80"] is None
    row = dict.fromkeys(x2x3.X3_RATE_KEYS)
    x3 = [
        {
            "experiment": "X3",
            "base_arm": "m",
            "regime": "R2",
            "design_key": ["exact", 1.0],
            **row,
            "null_raw": _rate(0.04),
        },
        {
            "experiment": "X3",
            "base_arm": "m",
            "regime": "R2",
            "design_key": ["width", 0.8],
            **row,
            "null_raw": _rate(0.06),
        },
        {
            "experiment": "X3",
            "base_arm": "m",
            "regime": "R2",
            "design_key": ["width", 0.5],
            **row,
            "null_raw": _rate(0.3, "INVALID"),
        },
    ]
    c = x2x3.x3_curve(x3)
    assert len(c) == 1
    assert c[0]["family"] == "width narrower"
    assert c[0]["first_invalid"] == "width x0.5"
    assert [s["variant"] for s in c[0]["steps"]] == ["exact", "width x0.8", "width x0.5"]


def test_x2x3_pooled_rate_clusters_by_world_and_seed():
    rows = [{"world": "E1", "seed": s, "cnt": {"k": [h, 10]}} for s, h in ((1, 0), (2, 5), (3, 10))]
    r = x2x3.pooled_rate(rows, "k")
    assert r is not None
    assert r["hits"] == 15
    assert r["n"] == 30
    assert r["n_clusters"] == 3
    assert r["rate"] == 0.5
    assert x2x3.pooled_rate([], "k") is None


def test_x5_holm_and_clopper_pearson():
    assert x5x6.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert x5x6.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    lo, hi = x5x6.cp_ci(3, 20)
    assert (lo, hi) == pytest.approx((sps.beta.ppf(0.025, 3, 18), sps.beta.ppf(0.975, 4, 17)))
    assert x5x6.cp_ci(0, 10)[0] == 0.0
    assert x5x6.cp_ci(10, 10)[1] == 1.0


def test_x5_label_probability_is_a_probability_and_reproducible():
    a = x5x6.label_prob(40, nsim=500)
    assert a == x5x6.label_prob(40, nsim=500)
    assert 0 <= a["p_invalid"] <= 1
    assert 0 <= a["p_valid"] <= 1
    assert a["p_invalid"] + a["p_valid"] <= 1
    assert (a["S"] == 40 and a["min_hits_invalid"] is None) or a["min_hits_invalid"] > 2


def _arm(centre: str, redraw: str, label: str, ci_hi: float, bias: dict | None = None) -> dict:
    row = {
        "base": "pmrt_nl_eq", "width": 2.0, "centre": centre, "redraw": redraw,
        "null_raw": {"validity": label, "ci": [0.0, ci_hi]},
    }  # fmt: skip
    if bias:
        row["bias"] = {"null": bias}
    return row


def test_x6_predictions_follow_the_declared_rule():
    b_big = {"mean": 0.8, "ci": [0.5, 1.0], "top_minus_bottom_ci": [0.1, 0.3]}
    b_small = {"mean": 0.1, "ci": [-0.1, 0.3], "top_minus_bottom_ci": [-0.1, 0.1]}
    arms = [
        _arm("told", "told", "INVALID", 0.4, b_big),
        _arm("told", "true", "VALID", 0.06),
        _arm("true", "told", "VALID", 0.07, b_small),
        _arm("true", "true", "VALID", 0.05),
    ]
    p = x5x6._predictions(arms)
    assert (p["P1"], p["P2"], p["P3"], p["P4"]) == (True, True, True, True)
    assert p["outcome"] == "SUPPORTED"
    arms[2] = _arm("true", "told", "INVALID", 0.4, b_small)
    p = x5x6._predictions(arms)
    assert p["P3"] is False
    assert p["outcome"] == "REFUTED"
    assert x5x6._predictions(arms[:3])["outcome"] is None


def test_x5_primary_outcomes_use_holm_and_the_valid_bound():
    cells = [
        {
            "arm": "pmrt_eq",
            "lam": lam,
            "n": n,
            kind: {"hits": h, "n": 100, "p_binom_gt05": p, "cp_ci": ci},
        }
        for (lam, n, key), (h, p, ci) in zip(
            x5x6.X5_PRIMARY,
            [(20, 1e-6, [0.1, 0.3]), (4, 0.7, [0.01, 0.06]), (9, 0.1, [0.04, 0.16])],
            strict=True,
        )
        for kind in [key.split("_")[0]]
    ]
    prim, answer = x5x6._x5_primary(cells)
    assert [r["outcome"] for r in prim] == ["EXCESS", "CHANCE", "UNRESOLVED"]
    assert answer == "real excess"
    assert prim[0]["p_holm"] < 0.05 <= prim[1]["p_holm"]
    assert x5x6._x5_primary([])[1] is None


def test_x9_identity_compares_outcomes_and_deferral_counts():
    base = {
        "arm": "a",
        "psvr": 1.0,
        "cpu_s": 3.0,
        "policy_counts": {"defer": {"ptx": 2}, "policy": {"x": 1}},
    }
    R = {
        1: {
            "a": base,
            "b": {
                **base,
                "arm": "b",
                "cpu_s": 9.0,
                "policy_counts": {"defer": {"ptx": 2}, "policy": {}},
            },
        },
        2: {"a": base, "b": {**base, "arm": "b", "psvr": 2.0}},
        3: {"a": base, "b": {**base, "arm": "b", "policy_counts": {"defer": {"ptx": 3}}}},
    }
    r = x9.identity(R, [1, 2, 3], "a", "b")
    assert r["identical_seeds"] == 1
    assert r["n"] == 3
    assert r["differing_fields"] == ["policy_counts.defer", "psvr"]


def test_x15_perm_test_and_descriptives_are_reproducible():
    rng = np.random.default_rng(3)
    xa, xb = rng.normal(0.5, 1, 40), rng.normal(0, 1, 40)
    w = np.ones(40)
    a = x15.perm_test(xa, w, xb, w, 1, n=300)
    assert a == x15.perm_test(xa, w, xb, w, 1, n=300)
    assert a["p"] != x15.perm_test(xa, w, xb, w, 2, n=300)["p"]
    assert a["diff"] == pytest.approx(xa.mean() - xb.mean())
    assert 0 < a["p"] <= 1
    d = x15.describe(xa, w, list(range(40)))
    assert d["n"] == 40
    assert d["mean_w"] == pytest.approx(xa.mean())
    assert d["top5_pos"][0]["value"] == pytest.approx(xa.max())
    assert x15.ks(xa, xa) == {"D": 0.0, "p": 1.0}


def _cells() -> list[dict]:
    spec = [
        ("carrier", "own", "pv", "TRUE", 1), ("carrier", "nbr", "pv", "NULL", 0),
        ("sleep", "own", "v", "TRUE", -1), ("sleep", "far", "e", "INDET", 0),
    ]  # fmt: skip
    return [
        {"family": f, "relation": r, "kpi": k, "status": s, "sign": g, "mean": 2.0 * g}
        for f, r, k, s, g in spec
    ]


def test_x11_scores_follow_the_frozen_convention_and_indet_bounds():
    cells = _cells()
    ref = {
        (c["family"], c["relation"], c["kpi"]): {"status": c["status"], "sign": c["sign"]}
        for c in cells
    }
    decl = x11.decl_from_list([["carrier|own|pv", 1], ["carrier|nbr|pv", 1], ["sleep|far|e", 1]])
    ref = {**{h: {"status": "NULL", "sign": 0} for h in decl}, **ref}
    s = x11.scores(decl, ref, cells)
    assert (s["tp"], s["fp"], s["fn"], s["indet_declared"]) == (1, 1, 1, 1)
    assert s["precision"] == 0.5
    assert s["recall"] == 0.5
    assert s["precision_indet_wrong"] == pytest.approx(1 / 3)
    assert s["precision_indet_right"] == pytest.approx(2 / 3)
    assert (s["n_true_right"], s["n_null"], s["n_indet"]) == (1, 1, 1)
    assert s["far_declared"] == 1


def test_x11_variants_move_one_factor_at_a_time():
    assert x11.VARIANTS["frozen"] == (0.05, 0.5, 0.95)
    assert x11.VARIANTS["delta_x2"][0] == 0.1
    assert x11.VARIANTS["floor_x0.5"][1] == 0.25
    assert x11.VARIANTS["ci99"][2] == 0.99


def test_x11_check_cells_is_a_hard_stop():
    c = {
        "family": "f", "relation": "r", "kpi": "k", "status": "TRUE", "sign": 1, "n": 5,
        "n_eps": 3, "mean": 1.0, "ci": [0.5, 1.5], "delta": 0.2,
    }  # fmt: skip
    x11.check_cells([c], [dict(c)], "same")
    with pytest.raises(x11.ReproductionError, match="REPRODUCTION FAILED"):
        x11.check_cells([c], [{**c, "status": "NULL"}], "diff")


def _analysis13() -> dict:
    cells = [
        {
            "family": "carrier",
            "relation": rel,
            "kpi": "pv",
            "status": st,
            "sign": sg,
            "mean": float(sg),
        }
        for rel, st, sg in (("own", "TRUE", 1), ("nbr", "NULL", 0), ("far", "TRUE", -1))
    ]
    return {
        "gt": {"cells": cells},
        "slices_result": {
            "pooled": {
                "kind": "pooled",
                "metrics": {
                    "m": {
                        "declared": [
                            ["carrier|own|pv", 1], ["carrier|nbr|pv", 1], ["carrier|far|pv", 1],
                        ],
                        "n_declared": 3, "far_declared": 1,
                    },
                },
            }
        },
    }  # fmt: skip


def test_x13_flags_own_attributable_declarations_and_counts_far_labels():
    out, rows = x13.analyse_study("S1", _analysis13())
    assert [r["own_attributable"] for r in rows] == [True, False]
    assert [r["declared_sign_matches_gt"] for r in rows] == [False, False]
    c = out["methods"]["m"]["slices"]["pooled"]
    assert c["nbr_declared"] == 1
    assert c["nbr_own_attributable"] == 1
    assert c["far_gt_TRUE"] == 1
    assert out["far_hypotheses_by_gt_label"]["all_20"] == {"TRUE": 1}
    bad = _analysis13()
    bad["slices_result"]["pooled"]["metrics"]["m"]["far_declared"] = 2
    with pytest.raises(x13.ReproductionError, match="REPRODUCTION FAILED"):
        x13.analyse_study("S1", bad)


def test_x14_existence_rule_ignores_magnitude():
    stats = {
        ("carrier", "own", "pv"): {"mean": 0.01, "n": 50, "n_eps": 10, "ci": {0.95: (0.005, 0.02)}},
        ("carrier", "nbr", "pv"): {"mean": 3.0, "n": 50, "n_eps": 10, "ci": {0.95: (-1.0, 7.0)}},
        ("carrier", "far", "pv"): {"mean": 3.0, "n": 5, "n_eps": 10, "ci": {0.95: (1.0, 7.0)}},
        ("sleep", "own", "pv"): {"mean": -2.0, "n": 50, "n_eps": 10, "ci": {0.95: (-3.0, -1.0)}},
    }
    ex = x14.existence_labels(stats, 0.95)
    assert [ex[h]["status"] for h in stats] == [
        "EXISTS",
        "NOT_ESTABLISHED",
        "UNSUPPORTED",
        "EXISTS",
    ]
    assert [ex[h]["sign"] for h in stats] == [1, 0, 0, -1]
    decl = x11.decl_from_list(
        [["carrier|own|pv", 1], ["carrier|nbr|pv", 1], ["carrier|far|pv", 1], ["sleep|own|pv", 1]]
    )
    full = {h: ex.get(h, {"status": "NOT_ESTABLISHED", "sign": 0}) for h in decl}
    s = x14.existence_scores(decl, full)
    assert (
        s["ex_confirmed"],
        s["ex_wrong_sign"],
        s["ex_not_established"],
        s["ex_unsupported"],
    ) == (1, 1, 1, 1)
    assert s["ex_precision"] == pytest.approx(1 / 3)


def _units(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    episode = np.repeat([0, 1], n // 2)
    t0 = np.concatenate([np.sort(rng.uniform(0, 600, n // 2)) for _ in range(2)])
    sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    y = rng.normal(size=n) + 0.5 * sgn
    pre = rng.normal(size=n)
    return y, pre, episode, sgn, t0


def test_x16_strict_centre_with_a_zero_window_is_the_frozen_centre():
    y, pre, episode, sgn, t0 = _units()
    cfg = ResidualConfig()
    r0, s0 = predictable_residuals(y, pre, episode, sgn, cfg)
    r1, s1 = x16.strict_residuals(y, pre, episode, sgn, t0, cfg, h=0)
    assert np.array_equal(r0, r1)
    assert np.array_equal(s0, s1, equal_nan=True)


@pytest.mark.claim("ASM-SLOPE-CENTRE")
def test_x16_strict_centre_never_uses_an_open_window():
    y, pre, episode, sgn, t0 = _units()
    cfg = ResidualConfig()
    r, _ = x16.strict_residuals(y, pre, episode, sgn, t0, cfg, h=90)
    u = 30
    open_units = [j for j in range(u) if episode[j] == episode[u] and t0[j] + 90 > t0[u]]
    assert open_units, "fixture must have units whose window is still open"
    y2 = y.copy()
    y2[open_units] += 1000.0
    r2, _ = x16.strict_residuals(y2, pre, episode, sgn, t0, cfg, h=90)
    assert r2[u] == r[u]
    r3, _ = predictable_residuals(y2, pre, episode, sgn, cfg)
    assert r3[u] != predictable_residuals(y, pre, episode, sgn, cfg)[0][u]


def test_x16_small_helpers():
    assert x16.rel_change(1.1, 1.0) == pytest.approx(0.1)
    assert x16.rel_change(1.0, 0) is None
    assert x16.fin(float("inf")) is None
    assert x16.fin(2) == 2.0
    assert x16.close(1.0, 1.0 + 1e-12)
    assert not x16.close(None, 1.0)

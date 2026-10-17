import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import binom

from oran_causal.analysis.system import study1, study2, study3
from oran_causal.sim.system.decision.vocab import HYPOTHESES


def ref_all(status: str = "NULL") -> dict:
    ref = {h: {"status": status, "sign": 0} for h in HYPOTHESES}
    for f, r, k, s in study1.CHAIN:
        ref[(f, r, k)] = {"status": "TRUE", "sign": s}
    return ref


def decl_of(pairs: dict) -> dict:
    d: dict[Any, dict[str, Any]] = {h: {"declared": False, "sign": 0} for h in HYPOTHESES}
    for h, s in pairs.items():
        d[h] = {"declared": True, "sign": s, "p": 1e-4, "z_approx": 3.0 * s}
    return d


CHAIN_DECL = {(f, r, k): s for f, r, k, s in study1.CHAIN}


def test_subset_metrics_on_the_chain() -> None:
    ref = ref_all()
    m = study1.subset_metrics(decl_of(CHAIN_DECL | {("carrier", "own", "pv"): -1}), ref)
    assert (m["chain_hits"], m["chain_true"], m["premise_hit"]) == (4, 4, 1.0)
    assert (m["ind_tp"], m["ind_fp"], m["ind_f1"]) == (4, 0, 1.0)
    assert m["ov_precision"] == 0.8
    assert m["n_declared"] == 5
    assert m["chain"]["sleep|nbr|pv"] == {"gt_true": True, "hit": True, "p": 1e-4, "z": 3.0}
    flipped = study1.subset_metrics(decl_of(CHAIN_DECL | {study1.PREMISE: -1}), ref)
    assert flipped["premise_hit"] == 0.0
    assert flipped["sign_acc"] == 0.75


def test_k0_binomial_tail_and_declarations() -> None:
    decl: dict[Any, dict[str, Any]] = {
        ("h", str(i)): {"p": 0.04 if i < 7 else 0.5, "declared": False} for i in range(60)
    }
    decl[("x", "u")] = {"p": float("nan"), "status": "undetermined"}
    k0 = study1.k0_check(decl)
    assert (k0["m"], k0["n_reject"]) == (60, 7)
    assert k0["binom_tail"] == pytest.approx(float(binom.sf(6, 60, 0.05)))
    assert k0["pass"]
    decl[("h", "0")] |= {"declared": True}
    decl[("h", "1")] |= {"declared": True}
    assert not study1.k0_check(decl)["pass"]


def test_k0n_rates_and_variants() -> None:
    p = [0.01] * 5 + [0.5] * 95
    fams = ["sleep"] * 50 + ["ptx"] * 50
    ok = study1.k0n_check(p, fams, [0] * 40)
    assert ok["pass"]
    assert ok["rate05"] == 0.05
    assert ok["rate05_family"] == {"ptx": 0.0, "sleep": 0.1}
    assert not study1.k0n_check(p, fams, [1] * 6)["pass"]
    assert not study1.k0n_check([], [], [])["pass"]


def test_k1_g_p1() -> None:
    k1 = study1.k1_check({"s120_0": {"sleep_units": 60, "sleep_rejects": 15}})
    assert k1["pass"]
    assert not study1.k1_check({"s": {"sleep_units": 59, "sleep_rejects": 99}})["pass"]
    cell = {"family": "sleep", "relation": "nbr", "kpi": "pv", "status": "TRUE", "sign": 1}
    assert study1.g_check([cell])["pass"]
    assert not study1.g_check([cell | {"sign": -1}])["pass"]
    assert study1.g_check([])["cell"] is None
    m = {"premise_hit": 1.0, "chain_hits": 2, "chain_true": 2}
    p1 = study1.p1_check([m, m, m, m | {"premise_hit": 0.0}, m])
    assert (p1["pass"], p1["n_ok"]) == (True, 4)
    assert not study1.p1_check([m, m, m | {"chain_hits": 1}])["pass"]


def test_p2_ties_and_undefined_scores() -> None:
    nan = float("nan")
    p2 = study1.p2_check([0.5, nan, 0.4], [0.5], {"b": [0.5, 0.1, nan]}, {"b": [0.6]})
    assert p2["per_baseline"]["b"]["win60"] == [True, False, True]
    assert p2["per_baseline"]["b"]["win120"] == [False]
    assert not p2["pass"]


def test_p3_and_s() -> None:
    ref = ref_all()
    p3 = study1.p3_check(decl_of(CHAIN_DECL), ref)
    assert p3["pass"]
    assert p3["values"]["overall_precision"] == 1.0
    worse = study1.p3_check(
        decl_of(CHAIN_DECL | {("carrier", "own", h[2]): 1 for h in HYPOTHESES[:5]}), ref
    )
    assert not worse["parts"]["overall_precision"]
    assert study1.s_check(1.0, [1.0, None, float("nan"), 0.9])["pass"]
    assert not study1.s_check(1.0, [None])["pass"]
    assert not study1.s_check(0.85, [1.0])["pooled_ok"]


def ok(v: bool = True) -> dict:
    return {"pass": v}


@pytest.mark.parametrize(
    ("args", "want"),
    [
        ((ok(False), ok(), ok(), ok(), ok(), ok(), ok(), ok()), "INVALID"),
        ((ok(), None, ok(), ok(), ok(), ok(), ok(), ok()), "INVALID"),
        ((ok(), ok(), ok(False), ok(), ok(), ok(), ok(), ok()), "NO-CHAIN"),
        ((ok(), ok(), ok(), ok(False), ok(), ok(), ok(), ok()), "UNDERPOWERED"),
        ((ok(), ok(), ok(), ok(), ok(), ok(), ok(), ok()), "PASS"),
        ((ok(), ok(), ok(), ok(), ok(), ok(False), ok(), ok()), "PARTIAL"),
        ((ok(), ok(), ok(), ok(), ok(), ok(), ok(), ok(False)), "KILL"),
    ],
)
def test_study1_verdict_precedence(args: tuple, want: str) -> None:
    assert study1.verdict(*args, mode="full", frozen=True)["verdict"] == want


def test_study1_verdict_labels() -> None:
    args = (ok(),) * 8
    assert study1.verdict(*args, mode="partial", frozen=True)["label"].startswith("NOT A VERDICT")
    assert study1.verdict(*args, mode="full", frozen=False)["label"].endswith("not frozen)")


def test_study1_data_status_and_mode() -> None:
    base, n = study1.SEEDS["placebo"]
    good = study1.stage_status("placebo", range(base, base + n), ["v4"])
    assert good["ok"]
    assert not study1.stage_status("placebo", range(base, base + n), ["v4"], smoke=True)["ok"]
    assert study1.stage_status("dev", None) == {"ok": False, "n": 0, "note": "missing"}
    b, k = study1.SEEDS["gt"]
    gt = list(range(b, b + k))
    st = study1.data_status({"placebo": good}, gt, dry=False)
    assert st["all_ok"]
    assert not study1.data_status({"placebo": good}, gt, dry=True)["all_ok"]
    art = {"sha_ok": True, "code_ok": True}
    assert study1.mode_of(st, ok(), art, complete=True, dry=False) == "full"
    assert study1.mode_of(st, None, art, complete=True, dry=False) == "partial"
    assert study1.mode_of(st, None, art, complete=True, dry=True) == "dry-run"


def test_x_checks_scoring() -> None:
    ref = ref_all()
    ref[("ptx", "own", "e")] = {"status": "INDET", "sign": 0}
    decl = decl_of(CHAIN_DECL | {("ptx", "own", "e"): 1, ("ptx", "own", "load"): 1})
    x = study2.x_checks(decl, ref)
    assert x["X1"]["pass"]
    assert (x["X2"]["n_scored"], x["X2"]["precision"], x["X2"]["sign_accuracy"]) == (2, 1.0, 1.0)
    assert [e[0] for e in x["X2"]["edges"]] == ["ptx|own|e", "sleep|nbr|pv", "sleep|nbr|v"]
    assert x["X3"]["pass"]
    wrong = study2.x_checks(decl_of({study1.PREMISE: -1, ("sleep", "nbr", "v"): 1}), ref)
    assert not wrong["X1"]["pass"]
    assert wrong["X2"]["precision"] == 0.5
    assert not wrong["X2"]["pass"]


def test_disc_label_precedence() -> None:
    x = {"X1": ok(), "X2": ok(), "X3": ok()}
    k1 = study2.k1_check({"sleep_units": 500, "sleep_rejects": 75})
    assert study2.disc_label(ok(), ok(), ok(), k1, x)["label"] == "DISC-PASS"
    assert study2.disc_label(ok(), ok(), ok(), k1, x | {"X1": ok(False)})["label"] == "DISC-PARTIAL"
    assert study2.disc_label(ok(), ok(), ok(), k1, x | {"X2": ok(False)})["label"] == "DISC-FAIL"
    small = study2.k1_check({"sleep_units": 499, "sleep_rejects": 75})
    assert study2.disc_label(ok(), ok(), ok(), small, x)["label"] == "UNDERPOWERED"
    assert study2.disc_label(ok(), None, ok(), k1, x)["stops_before_eval"]


def test_holm_step_down() -> None:
    assert study2.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert study2.holm({"a": 0.01, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    assert study2.holm({"a": 0.2}) == {"a": False}


def boot_of(**cols: list[float]) -> dict[str, dict[str, Any]]:
    return {a: {"Rstar": np.array(v), "R": np.array(v)} for a, v in cols.items()}


def test_d1_reselects_best_map() -> None:
    point = {"P": {"Rstar": 0.5}, "a": {"Rstar": 0.3}, "b": {"Rstar": 0.35}}
    boot = boot_of(P=[0.5, 0.5, 0.5], a=[0.3, 0.45, 0.2], b=[0.2, 0.3, 0.6])
    d1 = study2.d1_check(point, boot, "P", ["a", "b"])
    assert d1["best_point"] == "b"
    assert d1["delta1"] == pytest.approx(0.15)
    assert d1["best_selected_share"] == pytest.approx({"a": 2 / 3, "b": 1 / 3})
    assert not d1["pass"]


def test_d2_groups_signatures_and_counts_nan_against() -> None:
    boot = boot_of(P=[1.0, 1.0, np.nan], a=[0.0, 0.0, 0.0], b=[0.0, 0.0, 0.0], c=[0.0, 2.0, 0.0])
    d2 = study2.d2_check(boot, "P", ["a", "b", "c"], {"a": 1, "b": 1, "c": 2, "P": 3})
    assert d2["n_hypotheses"] == 2
    assert d2["hypotheses"]["a"]["members"] == ["a", "b"]
    assert d2["hypotheses"]["a"]["p"] == 0.5
    assert d2["hypotheses"]["c"]["p"] == 0.75
    assert not d2["pass"]


@pytest.mark.parametrize(
    ("flags", "want"),
    [
        ({"k0": False}, "INVALID"),
        ({"e": False}, "NOT ELIGIBLE"),
        ({}, "PASS"),
        ({"d2": False}, "PARTIAL"),
        ({"d1": False, "d2": False}, "FAIL"),
    ],
)
def test_study2_eval_verdict(flags: dict, want: str) -> None:
    base: dict[str, Any] = {"k0": True, "k0n": True, "e": True, "d1": True, "d2": True}
    base |= flags
    v = study2.eval_verdict(**base)
    assert v["verdict"] == want
    assert v["complete"]
    assert "NOT A VERDICT" in study2.eval_verdict(**base, why=["x"])["label"]


def test_d3_and_s1() -> None:
    point = {"CS:PMRT": {"Rstar": 0.26}, "never_sleep": {"Rstar": 0.26}}
    boot = boot_of(**{"CS:PMRT": [0.2, 0.3], "never_sleep": [0.2, 0.3]})
    d3 = study3.d3_check(point, boot)
    assert (d3["pass"], d3["delta3"], d3["lb90"], d3["ci90"]) == (False, 0.0, 0.0, [0.0, 0.0])
    boot["CS:PMRT"]["guard"] = {"rlf": np.array([0.9, 1.1, math.inf])}
    s1 = study3.s1_check(boot)
    assert s1["pass"]
    assert s1["rlf_ratio_ub90"] == pytest.approx(1.09)


@pytest.mark.parametrize(
    ("flags", "want"),
    [
        ({}, "PASS"),
        ({"d3": False}, "PARTIAL"),
        ({"d1": False, "d2": False, "d3": False}, "FAIL"),
        ({"e": False}, "NOT ELIGIBLE"),
        ({"k0n": False}, "INVALID"),
    ],
)
def test_study3_eval_verdict(flags: dict, want: str) -> None:
    args: dict[str, Any] = {"k0": True, "k0n": True, "e": True, "d1": True, "d2": True, "d3": True}
    args |= flags
    assert study3.eval_verdict(**args)["verdict"] == want


def test_arm_sets() -> None:
    assert len(study2.ALL_ARMS) == 26
    assert len(study2.ASSOC_ARMS) == 13
    assert len(study3.ALL_ARMS) == 26
    assert len(study3.CS_ARMS) == 17
    assert study3.CAL_ARM not in study3.ALL_ARMS
    assert study2.stage_seeds("eval") == list(range(187000, 187160))
    assert study3.stage_seeds("eval") == list(range(191100, 191260))


def test_placebo_null_blocks_from_the_tests() -> None:
    plc = {
        "carrier|own|e": {"declared": True, "p": 0.001, "status": "tested"},
        "ptx|own|e": {"declared": False, "p": 0.4, "status": "tested"},
    }
    raw = {
        "variants": ["g0:shift1", "g0:shift2"],
        "combos": {
            study1.PRIMARY_NAME: {
                "p": [0.01, 0.5, 0.6, 0.7],
                "fam": list("aabb"),
                "var_decl": [0, 1],
            },
            "other": {"p": [0.9], "fam": ["a"], "var_decl": [0]},
        },
    }
    tests = {
        "primary": study1.PRIMARY_NAME,
        "placebo_decl": {study1.PRIMARY_NAME: plc},
        "placebo_baselines": {"corr": {"n_declared": 3}},
        "K0n_raw": raw,
    }
    out = study1.placebo_null_blocks(tests)
    k0 = out["placebo_descriptive"][study1.PRIMARY_NAME]
    assert (k0["m"], k0["n_reject"], k0["declared"]) == (2, 1, ["carrier|own|e"])
    assert out["placebo_descriptive"]["corr"] == {"n_declared": 3}
    assert out["K0n"] == study1.k0n_check([0.01, 0.5, 0.6, 0.7], list("aabb"), [0, 1])
    assert out["K0n"]["rate05_family"] == {"a": 0.5, "b": 0.0}
    assert out["K0n_descriptive"]["variants"] == raw["variants"]
    assert set(out["K0n_descriptive"]["combos"]) == {study1.PRIMARY_NAME, "other"}
    assert study1.placebo_null_blocks(tests | {"K0n_raw": None})["K0n"] is None


def test_disc_from_tests_adds_the_placebo_and_null_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study2, "recompute_disc", dict)
    raw = {
        "variants": ["g0:shift1"],
        "groups": 10,
        "combos": {study1.PRIMARY_NAME: {"p": [0.2], "fam": ["a"], "var_decl": [0]}},
    }
    tests = {
        "primary": study1.PRIMARY_NAME,
        "placebo_decl": {study1.PRIMARY_NAME: {"a|own|e": {"declared": False, "p": 0.3}}},
        "disc_decl": {},
        "K0n_raw": raw,
        "units": {},
    }
    rep = study2.disc_from_tests(tests)
    assert set(rep) == {"primary", "units", "placebo_combos", "K0n", "K0n_descriptive"}
    assert rep["placebo_combos"][study1.PRIMARY_NAME]["m"] == 1
    assert rep["K0n"]["n_variants"] == 1
    assert rep["K0n_descriptive"]["groups"] == 10

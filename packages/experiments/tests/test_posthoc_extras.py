import copy
import json

import numpy as np
import pytest

from oran_causal.core.pmrt import nonlinear
from oran_causal.core.pmrt.nonlinear import PmrtNl
from oran_causal.experiments.campaign import extras
from oran_causal.experiments.campaign.__main__ import main as campaign_main
from oran_causal.experiments.campaign.integrity import accept_rule
from oran_causal.experiments.campaign.run import plain_execute
from oran_causal.experiments.campaign.spec import Unit, expand, is_eval, spec_sha256
from oran_causal.experiments.harness.records import Job
from oran_causal.experiments.posthoc import extras_run as xr
from oran_causal.experiments.posthoc import extras_specs as xs
from oran_causal.sim.controlled.datasets import generate
from oran_causal.sim.controlled.datasets.generate import dataset_hash, generate_dataset

DUMMY = "dummy"
SEED = 3_200_060


def small_x3() -> dict:
    arm = {"ref": DUMMY, "declare": "by"}
    return {
        "name": "extras_t",
        "budget_cpu_s": None,
        "extras": {"experiment": "X3", "design": None, "protocol": "p"},
        "arms": {
            "d": arm,
            "d.w200": {**arm, "base": "d", "told": {"width": 2.0}},
            "d.s05": {**arm, "base": "d", "told": {"shift": 0.05}},
        },
        "blocks": [
            {
                "role": "measure",
                "worlds": ["E1"],
                "regimes": ["R2"],
                "ns": [300],
                "kappas": [0.25],
                "seeds": [SEED, SEED + 1],
                "arms": ["d", "d.w200", "d.s05"],
            }
        ],
    }


def small_x2() -> dict:
    spec = small_x3()
    spec["extras"] = {"experiment": "X2", "design": {"delta": 0.05, "n_blocks": 10}}
    spec["arms"] = {"d": spec["arms"]["d"]}
    spec["blocks"][0]["arms"] = ["d"]
    spec["blocks"][0]["seeds"] = [3_200_020, 3_200_021]
    return spec


def test_the_diff_reports_a_changed_float_and_a_missing_key():
    specs = xs.build_specs()
    specs["x2_d05"]["extras"]["design"]["delta"] = 0.06
    specs["x6_gbm"]["extra_key"] = 1
    problems = xs.diff_frozen(specs)
    assert any("delta: 0.06 != 0.05" in p for p in problems)
    assert any("extra_key" in p for p in problems)
    assert xs.diff_frozen({"nope": {}}) == ["nope: no frozen file"]


def test_the_cost_table_gives_a_told_arm_its_base_arm_entries():
    specs = xs.build_specs()
    cost = {"pmrt_eq|E1|R2|n1000": 3.0, "corr|E1|R2|n1000": 1.0}
    table = xs.build_cost_agg({"x3_told": specs["x3_told"]}, cost)["cost"]
    assert table["pmrt_eq.w050|E1|R2|n1000"] == 3.0
    assert table["pmrt_eq|E1|R2|n1000"] == 3.0
    assert "corr|E1|R2|n1000" not in table


@pytest.mark.parametrize(
    ("experiment", "seeds"),
    [
        ("X2", [3_000_000, 3_000_010]),
        ("X3", [3_200_200, 3_200_201]),
        ("X5", [3_200_000, 3_200_005]),
    ],
)
def test_seeds_outside_the_experiments_block_are_refused(experiment, seeds):
    spec = small_x3()
    spec["extras"]["experiment"] = experiment
    with pytest.raises(ValueError, match="seeds"):
        extras.seed_list(seeds, spec)


def test_diag_experiments_take_only_their_own_block():
    spec = small_x3()
    spec["extras"]["experiment"] = "X6"
    assert extras.seed_list([3_302_000, 3_302_002], spec) == [3_302_000, 3_302_001, 3_302_002]
    with pytest.raises(ValueError, match="XMETHOD_DIAG"):
        extras.seed_list([3_300_000, 3_300_001], spec)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s["extras"].update(design={"delta": 0.1, "n_blocks": 20}), "no extras.design"),
        (lambda s: s["extras"].update(experiment="X2"), "needs extras.design"),
        (lambda s: s["extras"].update(experiment="X5"), "frozen arms only"),
        (lambda s: s["arms"]["d.w200"]["told"].update(centre="true"), "X6 only"),
        (lambda s: s["arms"]["d.w200"].update(told={"width": -1}), "factor"),
        (lambda s: s["extras"].update(experiment="X9"), "extras.experiment"),
    ],
)
def test_malformed_extras_specs_are_refused(mutate, message):
    spec = small_x3()
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        extras.validate(spec)


def test_dither_design_must_stay_inside_the_setpoint_envelope():
    assert extras.check_design({"delta": 0.1, "n_blocks": 20}) == (0.1, 20)
    for bad in ({"delta": 0.3, "n_blocks": 20}, {"delta": 0.1, "n_blocks": 0}):
        with pytest.raises(ValueError, match=r"dither delta|n_blocks"):
            extras.check_design(bad)


def test_an_extras_spec_is_dev_mode_and_accepts_only_its_own_records():
    spec = small_x3()
    assert not is_eval(spec)
    accept = accept_rule(spec)
    other = copy.deepcopy(spec)
    other["arms"]["d.w200"]["told"] = {"width": 3.0}
    assert not accept({"integrity": {"spec_sha256": "0" * 64}})
    assert accept({"run_mode": {"mode": "dev"}, "integrity": {"spec_sha256": spec_sha256(spec)}})
    assert not accept(
        {"run_mode": {"mode": "dev"}, "integrity": {"spec_sha256": spec_sha256(other)}}
    )
    assert len(expand(spec)) == 6


def test_dither_constants_change_the_data_and_are_restored():
    frozen = dataset_hash(generate_dataset("E1", "R2", 400, SEED, kappa=0.25)[0])
    with xr.dither_constants(0.05, 10):
        assert (generate.DITHER_DELTA, generate.DITHER_BLOCKS) == (0.05, 10)
        changed = dataset_hash(generate_dataset("E1", "R2", 400, SEED, kappa=0.25)[0])
    assert (generate.DITHER_DELTA, generate.DITHER_BLOCKS) == (0.10, 20)
    assert changed != frozen
    with xr.dither_constants(0.10, 20):
        again = dataset_hash(generate_dataset("E1", "R2", 400, SEED, kappa=0.25)[0])
    assert again == frozen


def test_the_design_source_refuses_fewer_rows_than_blocks():
    source = xr.design_dataset({"delta": 0.1, "n_blocks": 80})
    with pytest.raises(ValueError, match="n_blocks"):
        source(Unit("E1", "R2", None, 50, 0.25, SEED, "a", "measure"))


def test_shift_rows_rounds_half_up():
    assert xr.shift_rows(0.05, 1000, 20) == 3
    assert xr.shift_rows(0.02, 1000, 20) == 1
    assert xr.shift_rows(0.01, 1000, 20) == 1
    assert xr.shift_rows(0.0, 1000, 20) == 0


@pytest.fixture(scope="module")
def e1():
    return generate_dataset("E1", "R2", 400, SEED, kappa=0.25)[0]


def arr(x):
    assert x is not None
    return np.asarray(x, float)


def law(d):
    assert d.dist is not None
    return d.dist


def test_telling_a_width_scales_every_dither_law_and_keeps_the_data(e1):
    told = xr.tell(e1, {"width": 2.0})
    assert told.X_action is e1.X_action
    for a, b in zip(e1.designs, told.designs, strict=True):
        if a.kind == "dither":
            assert law(b)["hi"] == 2.0 * law(a)["hi"]
            assert law(b)["lo"] == 2.0 * law(a)["lo"]
            np.testing.assert_array_equal(arr(b.fixed_part), arr(a.fixed_part))
    assert dataset_hash(e1) == dataset_hash(generate_dataset("E1", "R2", 400, SEED, kappa=0.25)[0])


def test_telling_a_shift_moves_the_setpoint_and_keeps_the_realised_action(e1):
    told = xr.tell(e1, {"shift": 0.5})
    s = xr.shift_rows(0.5, e1.n, e1.meta["dither"]["n_blocks"])
    for j, (a, b) in enumerate(zip(e1.designs, told.designs, strict=True)):
        if a.kind != "dither":
            continue
        np.testing.assert_array_equal(arr(b.fixed_part)[s:], arr(a.fixed_part)[:-s])
        np.testing.assert_allclose(arr(b.fixed_part) + arr(b.random_part), arr(e1.X_action)[:, j])
        same = arr(b.fixed_part) == arr(a.fixed_part)
        np.testing.assert_array_equal(arr(b.random_part)[same], arr(a.random_part)[same])


def test_telling_a_lambda_recomputes_the_logged_propensity():
    ds = generate_dataset("E4", "R3", 300, SEED, lam=1.0, kappa=0.25)[0]
    told = xr.tell(ds, {"lam": 0.5})
    logged = [(a, b) for a, b in zip(ds.designs, told.designs, strict=True) if a.kind == "logged"]
    assert logged
    for a, b in logged:
        assert law(b)["lam"] == 0.5
        assert arr(b.propensity).shape == arr(a.propensity).shape
        assert not np.allclose(arr(a.propensity), arr(b.propensity))
        np.testing.assert_allclose(arr(b.propensity).sum(1), 1.0)


def test_a_variant_that_changes_no_design_is_refused(e1):
    with pytest.raises(ValueError, match="changes no design"):
        xr.tell(e1, {"lam": 0.5})


def run_part_main(tmp_path, spec, kit, part="0/1", name="out.jsonl"):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    out = tmp_path / name
    argv = ["--root", str(tmp_path), "run", "--spec", str(path), "--part", part, "--out", str(out)]
    argv += ["--no-isolate"]
    messages: list[str] = []
    providers = kit.quiet_providers(log=messages.append)
    assert campaign_main(argv, out=messages.append, providers=providers) == 0
    return kit.read_jsonl(out)


def test_the_campaign_cli_runs_an_x3_spec_with_told_designs(tmp_path, kit):
    recs = run_part_main(tmp_path, small_x3(), kit)
    assert len(recs) == 6
    assert {r["status"] for r in recs} == {"ok"}
    by_arm = {r["arm"]: r for r in recs if r["key"].endswith(f"s{SEED}")}
    assert "told" not in by_arm["d"]
    assert by_arm["d.w200"]["told"] == {"width": 2.0}
    assert by_arm["d.s05"]["told"] == {"shift": 0.05}
    assert by_arm["d"]["dataset_sha256"] == by_arm["d.w200"]["dataset_sha256_generated"]
    assert by_arm["d.w200"]["dataset_sha256"] != by_arm["d"]["dataset_sha256"]
    r = recs[0]
    assert r["run_mode"]["mode"] == "dev"
    assert r["integrity"]["extras"]["experiment"] == "X3"
    assert r["integrity"]["extras"]["seed_block"] == "XMETHOD_EXTRAS"
    assert r["integrity"]["extras"]["version"] == "xm-extras/1"
    assert r["integrity"]["part"] == 0


def test_an_x2_spec_generates_under_its_dither_design(tmp_path, kit):
    recs = run_part_main(tmp_path, small_x2(), kit)
    assert len(recs) == 2
    assert recs[0]["integrity"]["extras"]["design"] == {"delta": 0.05, "n_blocks": 10}
    with xr.dither_constants(0.05, 10):
        want = dataset_hash(generate_dataset("E1", "R2", 300, 3_200_020, kappa=0.25)[0])
    assert recs[0]["dataset_sha256"] == want
    assert (generate.DITHER_DELTA, generate.DITHER_BLOCKS) == (0.10, 20)


def test_extras_runs_are_resumable_and_split_by_part(tmp_path, kit):
    first = run_part_main(tmp_path, small_x3(), kit, "0/2", "p0.jsonl")
    second = run_part_main(tmp_path, small_x3(), kit, "1/2", "p1.jsonl")
    assert {r["key"] for r in first}.isdisjoint(r["key"] for r in second)
    assert len(first) + len(second) == 6
    assert len(run_part_main(tmp_path, small_x3(), kit, "0/2", "p0.jsonl")) == len(first)


def test_a_dev_spec_with_extras_seeds_is_not_a_plain_campaign_spec():
    spec = small_x3()
    del spec["extras"]
    with pytest.raises(ValueError, match="EVAL seeds"):
        expand(spec)


def test_x6_hooks_swap_the_frozen_functions_and_restore_them():
    frozen = nonlinear.design_law, nonlinear.gbm_profiles, nonlinear.GbmStat
    session = xr.X6Session()
    with xr.x6_hooks(session):
        assert nonlinear.design_law is not frozen[0]
        assert nonlinear.gbm_profiles is not frozen[1]
        assert issubclass(nonlinear.GbmStat, frozen[2])
    assert (nonlinear.design_law, nonlinear.gbm_profiles, nonlinear.GbmStat) == frozen
    with pytest.raises(RuntimeError), xr.x6_hooks(session):
        raise RuntimeError
    assert nonlinear.design_law is frozen[0]


@pytest.fixture(scope="module")
def gbm_job():
    ds, truth = generate_dataset("E1", "R2", 300, 3_302_000, kappa=0.25)
    config = {"covariates": "eq", "statistic": "gbm", "gbm_trees": 20, "B": 50}
    return Job("k", ds, truth, config, 0.0)


def x6_record(gbm_job, told):
    session = xr.X6Session()
    execute = xr.x6_execute(plain_execute, session, {"a": told} if told else {}, {"a"})
    unit = Unit("E1", "R2", None, 300, 0.25, 3_302_000, "a", "measure")
    with xr.x6_hooks(session):
        rec = execute(PmrtNl(), gbm_job, unit)
    assert session.unit is None
    return rec


def test_x6_logs_the_centring_bias_of_a_told_width(gbm_job):
    exact = x6_record(gbm_job, None)
    wide = x6_record(gbm_job, {"width": 2.0})
    assert exact["error"] is None
    assert exact["x6"]["centre"] == exact["x6"]["redraw"] == "told"
    assert set(exact["x6"]["bias"]) == {"P0", "P1", "P2", "P3", "P_placebo"}
    assert all(v == 0.0 for b in exact["x6"]["bias"].values() for v in b["rms_c"])
    assert any(v > 0.0 for b in wide["x6"]["bias"].values() for v in b["rms_c"])
    assert wide["told"] == {"width": 2.0}
    assert wide["dataset_sha256_generated"] == dataset_hash(gbm_job.ds)


def test_x6_ablation_centres_and_redraws_under_the_true_law(gbm_job):
    told = x6_record(gbm_job, {"width": 2.0})
    true = x6_record(gbm_job, {"width": 2.0, "centre": "true", "redraw": "true"})
    assert (true["x6"]["centre"], true["x6"]["redraw"]) == ("true", "true")
    assert true["error"] is None
    shift = [b["mean_c"] for b in told["x6"]["bias"].values()]
    assert shift == [b["mean_c"] for b in true["x6"]["bias"].values()]
    assert told["edges"] != true["edges"] or told["notes"] != true["notes"]


def test_the_x6_executor_leaves_untold_non_gbm_arms_alone(gbm_job):
    session = xr.X6Session()
    execute = xr.x6_execute(plain_execute, session, {}, set())
    unit = Unit("E1", "R2", None, 300, 0.25, 3_302_000, "b", "measure")
    with xr.x6_hooks(session):
        rec = execute(PmrtNl(), gbm_job, unit)
    assert "x6" not in rec
    assert "told" not in rec

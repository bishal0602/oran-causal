import numpy as np
import pytest

from oran_causal.sim.controlled.datasets.generate import (
    KAPPA_DEFAULT,
    NOISE_SIGMA,
    WARMUP,
    dataset_hash,
    generate_dataset,
)
from oran_causal.sim.controlled.datasets.truth import (
    ENV_CLASSES,
    PLACEBO,
    PLACEBO_CONF,
    REGIMES_OF,
    candidates_for,
    roles_for,
    truth_for,
)

SEED = 3_000_000
CELLS = [(w, r) for w, regimes in REGIMES_OF.items() for r in regimes]
pytestmark = pytest.mark.claim("DEF-ENV")


@pytest.mark.parametrize(("world", "regime"), CELLS)
def test_generation_is_a_pure_function_of_its_arguments(world: str, regime: str) -> None:
    first, _ = generate_dataset(world, regime, 200, SEED, lam=1.5)
    second, _ = generate_dataset(world, regime, 200, SEED, lam=1.5)
    assert dataset_hash(first) == dataset_hash(second)
    other, _ = generate_dataset(world, regime, 200, SEED + 1, lam=1.5)
    assert dataset_hash(first) != dataset_hash(other)


@pytest.mark.parametrize(("world", "regime"), CELLS)
def test_rows_are_aligned_and_shapes_consistent(world: str, regime: str) -> None:
    ds, _ = generate_dataset(world, regime, 200, SEED, lam=1.0, kappa=0.0)
    env_cls = ENV_CLASSES[world]
    assert ds.Y.shape == ds.X_kpi_lag.shape == (200, env_cls.num_kpis)
    assert ds.X_action.shape == (200, len(ds.action_names))
    assert len(ds.designs) == len(ds.action_names)
    assert np.array_equal(ds.Y[:-1], ds.X_kpi_lag[1:])
    assert ds.time_index is not None
    assert ds.time_index[0] == WARMUP
    assert ds.candidates == candidates_for(world, regime)


@pytest.mark.parametrize(("world", "regime"), CELLS)
def test_placebo_columns_follow_the_regime(world: str, regime: str) -> None:
    ds, _ = generate_dataset(world, regime, 100, SEED, lam=1.0)
    assert ds.action_names[-1 if regime not in ("R3", "R4") else -2] == PLACEBO
    assert (PLACEBO_CONF in ds.action_names) == (regime in ("R3", "R4"))
    bare, _ = generate_dataset(world, regime, 100, SEED, lam=1.0, with_placebo=False)
    assert PLACEBO not in bare.action_names
    assert np.array_equal(bare.X_action, ds.X_action[:, : bare.X_action.shape[1]])
    assert np.array_equal(bare.Y, ds.Y)


@pytest.mark.parametrize("world", sorted(REGIMES_OF))
def test_design_kinds_per_regime(world: str) -> None:
    kinds = {
        regime: {d.kind for d in generate_dataset(world, regime, 100, SEED)[0].designs}
        for regime in REGIMES_OF[world]
    }
    assert kinds["R1"] == {"iid"}
    assert kinds["R2"] == {"dither"}
    if world == "E4":
        assert kinds["R3"] == {"logged", "iid"}
        assert kinds["R4"] == {"none", "iid"}


def test_logged_propensities_are_distributions() -> None:
    ds, _ = generate_dataset("E4", "R3", 300, SEED, lam=1.5)
    table = ds.designs[0].propensity
    assert table is not None
    assert table.shape == (300, 101)
    assert np.allclose(table.sum(axis=1), 1.0)
    assert ds.context is not None
    assert ds.context.shape == (300, 1)
    assert np.all(ds.meta["realised_propensity"]["P0"] > 0.0)


def test_dither_never_leaves_the_parameter_range() -> None:
    ds, _ = generate_dataset("E2", "R2", 400, SEED)
    for column, (lo, hi) in enumerate(ENV_CLASSES["E2"].id_ranges):
        assert ds.X_action[:, column].min() >= lo
        assert ds.X_action[:, column].max() <= hi
        design = ds.designs[column]
        assert design.fixed_part is not None
        assert design.random_part is not None
        assert np.array_equal(ds.X_action[:, column], design.fixed_part + design.random_part)


@pytest.mark.parametrize(("world", "regime"), CELLS)
def test_truth_edges_are_candidates_and_nulls_are_the_rest(world: str, regime: str) -> None:
    truth = truth_for(world, regime)
    candidates = set(candidates_for(world, regime))
    assert truth.edges <= candidates
    assert truth.null_edges == candidates - truth.edges
    assert set(truth.signs) <= truth.edges
    num_kpis = ENV_CLASSES[world].num_kpis
    assert all((PLACEBO, f"K{j}") in truth.null_edges for j in range(num_kpis))


def test_roles_are_returned_as_a_copy() -> None:
    roles = roles_for("E5")
    roles.clear()
    assert roles_for("E5")


@pytest.mark.parametrize(("world", "regime"), CELLS)
def test_default_kappa_is_the_world_default_and_explicit_kappa_is_uniform(
    world: str, regime: str
) -> None:
    default, _ = generate_dataset(world, regime, 300, SEED, lam=1.5)
    explicit, _ = generate_dataset(world, regime, 300, SEED, lam=1.5, kappa=KAPPA_DEFAULT[world])
    assert dataset_hash(default) == dataset_hash(explicit)
    zero, truth_zero = generate_dataset(world, regime, 300, SEED, lam=1.5, kappa=0.0)
    noisy, truth_noisy = generate_dataset(world, regime, 300, SEED, lam=1.5, kappa=0.2)
    assert "obs_noise" not in zero.meta
    assert noisy.meta["obs_noise"] == {"kappa": 0.2, "sigma": list(NOISE_SIGMA[world])}
    assert ("obs_noise" in default.meta) == (KAPPA_DEFAULT[world] > 0)
    assert np.array_equal(zero.X_action, noisy.X_action)
    assert truth_zero == truth_noisy
    assert not np.array_equal(zero.Y, noisy.Y)
    assert np.array_equal(noisy.Y[:-1], noisy.X_kpi_lag[1:])


@pytest.mark.parametrize("world", sorted(REGIMES_OF))
def test_realised_noise_sd_and_common_random_numbers(world: str) -> None:
    base, _ = generate_dataset(world, "R1", 4000, SEED, kappa=0.0)
    series_base = np.concatenate([base.X_kpi_lag, base.Y[-1:]])
    noise = {}
    for kappa in (0.1, 0.5):
        ds, _ = generate_dataset(world, "R1", 4000, SEED, kappa=kappa)
        noise[kappa] = np.concatenate([ds.X_kpi_lag, ds.Y[-1:]]) - series_base
        scale = kappa * np.array(NOISE_SIGMA[world])
        assert np.all(np.abs(noise[kappa].std(0) / scale - 1) < 0.05)
        assert np.all(np.abs(noise[kappa].mean(0)) < 4 * scale / np.sqrt(4001))
    np.testing.assert_allclose(
        noise[0.5] / 0.5, noise[0.1] / 0.1, rtol=1e-6, atol=1e-9 * max(NOISE_SIGMA[world])
    )


def test_e5_has_a_single_noise_layer() -> None:
    noisy, _ = generate_dataset("E5", "R1", 2000, SEED)
    clean, _ = generate_dataset("E5", "R1", 2000, SEED, kappa=0.0)
    assert noisy.meta["obs_noise"]["kappa"] == 0.3
    assert np.max(np.abs(clean.Y[:, 2] - clean.X_action[:, 0])) == 0.0
    residual = noisy.Y[:, 2] - noisy.X_action[:, 0]
    assert abs(residual.std() / (0.3 * NOISE_SIGMA["E5"][2]) - 1) < 0.05


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_invalid_kappa_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="kappa"):
        generate_dataset("E1", "R1", 50, SEED, kappa=bad)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("E9", "R1", 50), "unknown world"),
        (("E1", "R3", 50), "not defined"),
        (("E1", "R1", 1), "n must"),
    ],
)
def test_invalid_arguments_are_rejected(args: tuple[str, str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_dataset(*args, SEED)


def test_e4_lambda_must_be_in_the_frozen_sweep() -> None:
    with pytest.raises(ValueError, match="lambda"):
        generate_dataset("E4", "R3", 50, SEED, lam=0.7)


def test_lambda_changes_only_the_confounded_regimes() -> None:
    r1_a, _ = generate_dataset("E4", "R1", 100, SEED, lam=0.0)
    r1_b, _ = generate_dataset("E4", "R1", 100, SEED, lam=1.5)
    assert np.array_equal(r1_a.X_action, r1_b.X_action)
    r4_a, _ = generate_dataset("E4", "R4", 100, SEED, lam=0.0)
    r4_b, _ = generate_dataset("E4", "R4", 100, SEED, lam=1.5)
    assert not np.array_equal(r4_a.X_action, r4_b.X_action)


def test_confounded_regimes_have_a_biased_pooled_slope() -> None:
    ds, _ = generate_dataset("E4", "R4", 6000, SEED, lam=1.0, kappa=0.0)
    a = ds.X_action[:, 0]
    pooled = np.cov(a, ds.Y[:, 0])[0, 1] / a.var(ddof=1)
    assert pooled > 0.0
    interventional, _ = generate_dataset("E4", "R1", 6000, SEED, lam=1.0, kappa=0.0)
    a_do = interventional.X_action[:, 0]
    assert abs(np.cov(a_do, interventional.Y[:, 0])[0, 1] / a_do.var(ddof=1) + 1.0) < 0.2

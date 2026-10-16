from typing import Any, cast

import numpy as np

from oran_causal.sim.controlled.e1 import E1Env


def make_env(noise: float = 0.0, seed: int = 7, episode: int = 0) -> E1Env:
    return E1Env(env_seed=seed, obs_noise_scale=noise, episode=episode)


def tape_noise(env: E1Env, scale: float) -> np.ndarray:
    return np.array(
        [env._coord_rng(env._NS_OBS, v, env.time).normal(0.0, scale) for v in range(env.num_kpis)]
    )


def test_repeated_observation_at_one_coordinate_is_identical() -> None:
    env = make_env(noise=0.5)
    env.step(0, 0.3)
    env.step(1, 0.6)
    first = env.observed_kpis()
    for _ in range(5):
        assert np.array_equal(env.observed_kpis(), first)


def test_initial_params_are_coordinate_keyed_and_stable() -> None:
    a = make_env(seed=3, episode=2)
    b = make_env(seed=3, episode=2)
    c = make_env(seed=3, episode=5)
    assert np.array_equal(a.params, b.params)
    assert not np.array_equal(a.params, c.params)


def test_arms_with_different_actions_share_byte_identical_noise() -> None:
    scale = 0.75
    arm_a = make_env(noise=scale, seed=11)
    arm_b = make_env(noise=scale, seed=11)
    for t in range(4):
        arm_a.step(0, 0.1 * t)
        arm_b.step(1, 0.9 - 0.1 * t)
        arm_b.observed_kpis()
        assert arm_a.time == arm_b.time
        eps_a, eps_b = tape_noise(arm_a, scale), tape_noise(arm_b, scale)
        assert np.array_equal(eps_a, eps_b)
        assert np.array_equal(arm_a.observed_kpis(), arm_a.latent_kpis() + eps_a)
        assert np.array_equal(arm_b.observed_kpis(), arm_b.latent_kpis() + eps_b)


def test_noise_off_latent_equals_observed() -> None:
    env = make_env()
    env.step(0, 0.4)
    env.step(2, 0.8)
    assert np.array_equal(env.latent_kpis(), env.observed_kpis())


def test_noise_on_leaves_latent_unchanged_and_uses_declared_scale() -> None:
    scale = 0.6
    env = make_env(noise=scale)
    env.step(0, 0.4)
    env.step(1, 0.7)
    latent_before = env.latent_kpis()
    for _ in range(10):
        observed = env.observed_kpis()
    expected = tape_noise(env, scale)
    assert np.array_equal(env.latent_kpis(), latent_before)
    assert np.array_equal(observed, latent_before + expected)
    assert np.any(expected != 0.0)


def test_distinct_variables_draw_independently() -> None:
    env = make_env(noise=0.5)
    env.step(0, 0.5)
    eps = env.observed_kpis() - env.latent_kpis()
    assert len(set(np.round(eps, 12))) > 1


def test_global_numpy_state_is_untouched_by_a_rollout() -> None:
    before = cast("tuple[Any, ...]", np.random.get_state())  # noqa: NPY002
    env = make_env(noise=0.5)
    for t in range(6):
        env.step(t % env.num_params, 0.1 * t)
        env.observed_kpis()
    after = cast("tuple[Any, ...]", np.random.get_state())  # noqa: NPY002
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_snapshot_diverge_restore_reproduces_continuation() -> None:
    scale = 0.4
    env = make_env(noise=scale, seed=21)
    env.step(0, 0.2)
    env.step(1, 0.5)
    snap = env.snapshot()

    reference = make_env(noise=scale, seed=21)
    reference.restore(snap)
    reference.step(2, 0.3)
    reference.step(3, 0.9)

    env.step(2, 0.99)
    env.observed_kpis()
    env.step(3, 0.01)
    env.observed_kpis()
    env.restore(snap)
    env.step(2, 0.3)
    env.step(3, 0.9)
    assert np.array_equal(env.latent_kpis(), reference.latent_kpis())
    assert np.array_equal(env.observed_kpis(), reference.observed_kpis())

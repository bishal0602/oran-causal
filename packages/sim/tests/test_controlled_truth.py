import math

import numpy as np
import pytest

from oran_causal.sim.controlled.base import ControlledEnv
from oran_causal.sim.controlled.e1 import E1Env
from oran_causal.sim.controlled.e2 import E2Env
from oran_causal.sim.controlled.e3 import E3Env
from oran_causal.sim.controlled.e4 import LATENT_EDGES, MU_OUT, SIGMA_OUT, E4Env
from oran_causal.sim.controlled.e5 import E5Env

pytestmark = pytest.mark.claim("DEF-ENV")

TOL = 1e-12
DO_HIGH, DO_LOW = 0.75, 0.25


def primed[E: ControlledEnv](env: E, steps: int = 3) -> E:
    for _ in range(steps):
        env.neutral_step()
    return env


def do_two_steps(env: ControlledEnv, snap: tuple, param_id: int, value: float) -> np.ndarray:
    env.restore(snap)
    env.apply_action(param_id, value)
    env.advance()
    return env.advance()


def test_e1_analytic_mechanism() -> None:
    env = primed(E1Env(env_seed=5))
    p, k = env.prev_params.copy(), env.prev_kpis.copy()
    expected = np.array([p[0], p[1], p[2] + 0.5 * k[0], p[3] + 0.5 * k[1]])
    assert np.allclose(env.advance(), expected, atol=TOL)


def test_e1_action_effect_is_delayed_by_one_step() -> None:
    env = primed(E1Env(env_seed=3))
    snap = env.snapshot()
    env.apply_action(0, 0.9)
    warm_up = env.advance()
    env.restore(snap)
    neutral = env.advance()
    assert np.allclose(warm_up, neutral, atol=TOL)
    env.restore(snap)
    env.apply_action(0, 0.9)
    env.advance()
    assert abs(env.advance()[0] - 0.9) <= TOL


def test_e1_direct_effects_match_locked_values() -> None:
    env = primed(E1Env(env_seed=0))
    snap = env.snapshot()
    effect = abs(do_two_steps(env, snap, 0, DO_HIGH)[0] - do_two_steps(env, snap, 0, DO_LOW)[0])
    assert abs(effect - 0.5) <= TOL


def test_e1_shadow_zeroing_removes_exactly_one_edge() -> None:
    shadow = E1Env(env_seed=0, b2=0.0)
    primed(shadow)
    snap = shadow.snapshot()
    low = do_two_steps(shadow, snap, 0, DO_LOW)
    high = do_two_steps(shadow, snap, 0, DO_HIGH)
    assert abs(high[2] - low[2]) <= TOL


def test_e1_adjacency_matrix() -> None:
    matrix = E1Env(env_seed=0).true_adj_matrix()
    assert matrix.shape == (8, 8)
    assert int(matrix.sum()) == 6
    assert matrix[6, 4] == 1.0
    assert matrix[7, 5] == 1.0


def test_e2_latency_and_decoy() -> None:
    env = primed(E2Env(env_seed=3))
    snap = env.snapshot()
    committed_p0 = float(snap[1][0])
    env.apply_action(0, committed_p0 + 60.0)
    warm_up = env.advance()
    env.restore(snap)
    assert np.allclose(warm_up, env.advance())

    true_env, decoy_env = E2Env(env_seed=5), E2Env(env_seed=5, decoy_omit_p0_k5=True)
    for e in (true_env, decoy_env):
        primed(e)

    def terminal_k5(e: E2Env, value: float) -> float:
        snap = e.snapshot()
        e.apply_action(0, value)
        e.advance()
        result = float(e.advance()[5])
        e.restore(snap)
        return result

    p0 = float(true_env.prev_params[0])
    assert terminal_k5(decoy_env, p0 - 40.0) == terminal_k5(decoy_env, p0 + 40.0)
    assert terminal_k5(true_env, p0 - 40.0) != terminal_k5(true_env, p0 + 40.0)
    assert (5, 0) in true_env.adjacency_edges
    assert (5, 0) not in decoy_env.adjacency_edges
    assert len(true_env.adjacency_edges) == 16


def test_e3_standardization_is_exact() -> None:
    expected = np.array(
        [
            math.sqrt(1 / 12),
            math.sqrt(1 / 12),
            math.sqrt(2 / 12),
            math.sqrt(2 / 12),
            math.sqrt(1 / 12),
        ]
    )
    assert np.array_equal(E3Env.sigma, expected)
    assert np.array_equal(E3Env.mu, np.array([0.5, 0.5, 1.0, 1.0, 0.5]))
    assert E3Env.kpi_thresholds == (0.5, 0.5, 1.0, 1.0, 0.5)


def test_e3_steady_state_after_three_neutral_advances() -> None:
    env = E3Env(env_seed=7)
    p = env.prev_params.copy()
    primed(env)
    expected = np.array([p[0], p[0], p[1] + p[0], p[2] + p[0], p[3]])
    assert np.allclose(env.prev_kpis, expected, atol=TOL)


def test_e3_cascade_timing() -> None:
    env = primed(E3Env(env_seed=0))
    snap = env.snapshot()

    def rollout(value: float) -> list[np.ndarray]:
        env.restore(snap)
        env.apply_action(0, value)
        env.advance()
        return [env.advance() for _ in range(3)]

    hi, lo = rollout(DO_HIGH), rollout(DO_LOW)

    def moved(lag: int, kpi: int) -> bool:
        return abs(hi[lag][kpi] - lo[lag][kpi]) > TOL

    assert moved(0, 0)
    assert not moved(0, 1)
    assert moved(1, 1)
    assert not moved(1, 2)
    assert not moved(1, 3)
    assert moved(2, 2)
    assert moved(2, 3)
    assert not any(moved(lag, 4) for lag in range(3))


def test_e3_structure_and_truncated_fanout() -> None:
    full = E3Env(env_seed=0)
    truncated = E3Env(env_seed=0, truncate_fanout=True)
    assert set(full.adjacency_edges) == {(0, 0), (1, 4), (2, 1), (2, 5), (3, 2), (3, 5), (4, 3)}
    assert set(full.adjacency_edges) - set(truncated.adjacency_edges) == {(2, 5), (3, 5)}
    matrix = full.true_adj_matrix()
    assert matrix.shape == (9, 9)
    assert int(matrix.sum()) == 7
    assert matrix[5, 4] == 1.0
    assert matrix[6, 1] == 1.0
    assert matrix[6, 5] == 1.0


def test_e4_policy_and_do_action_semantics() -> None:
    obs = E4Env(env_seed=3, lam=1.0, mode="obs")
    expected = float(np.clip(0.5 + obs.lam * obs.Z + obs._draw_eta(obs.time), 0.0, 1.0))
    assert obs.behavior_action() == expected == obs.generate_action()

    do = E4Env(env_seed=3, lam=1.0, mode="do")
    first = do.generate_action()
    do.Z += 123.456
    assert do.generate_action() == first
    assert first in set(np.round(np.arange(101) / 100.0, 2))


def test_e4_same_latent_enters_outcome_under_both_modes() -> None:
    do_env, obs_env = E4Env(env_seed=7, mode="do"), E4Env(env_seed=7, mode="obs")
    assert do_env.Z == obs_env.Z
    z0 = do_env.Z

    def roll(env: E4Env) -> float:
        env.reset(episode=0)
        env.apply_action(0, 0.3)
        env.advance()
        return float(env.advance()[0])

    assert roll(do_env) == roll(obs_env)
    assert math.isclose(roll(do_env), -0.3 + 2.5 * z0, abs_tol=TOL)


def test_e4_snapshot_restore_carries_latents() -> None:
    env = primed(E4Env(env_seed=5, mode="do"))
    snap = env.snapshot()
    z_pending, z_committed = float(env.Z), float(env.prev_Z)
    assert snap[5] == z_pending
    assert snap[6] == z_committed
    env.apply_action(0, 0.8)
    env.advance()
    after = float(env.advance()[0])
    env.restore(snap)
    assert (env.Z, env.prev_Z) == (z_pending, z_committed)
    env.apply_action(0, 0.8)
    env.advance()
    assert float(env.advance()[0]) == after


def test_e4_constants_and_adjacency() -> None:
    assert E4Env.mu[0] == MU_OUT == -0.5
    assert E4Env.sigma[0] == SIGMA_OUT == math.sqrt(1.0 / 12.0)
    assert E4Env(env_seed=0).adjacency_edges == [(0, 0)]
    assert LATENT_EDGES == [("Z", "A_behavior"), ("Z", "K_out")]
    matrix = E4Env(env_seed=0).true_adj_matrix()
    assert matrix.shape == (2, 2)
    assert int(matrix.sum()) == 1
    assert matrix[1, 0] == 1.0


def test_e4_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        E4Env(mode="other")


def test_e4_interventional_mean_is_confound_free() -> None:
    assert E4Env(env_seed=0, lam=0.5).score_interventional_mean(0.25) == -0.25


def test_e5_default_confound_masks_the_true_effect() -> None:
    env = E5Env()
    assert env.theta < 0.0 < env.lam
    assert env.lam * env.theta < 0.0 < env.subdom


def test_e5_behavior_policy_scale_and_do_grid() -> None:
    env = E5Env(mode="obs", lam=1.0)
    env.Z = 1.3
    assert abs(env.behavior_p0() - 30.0 * 1.3) < 3.0
    do = E5Env(env_seed=0, mode="do")
    draws = [do._draw_do_p0(t) for t in range(200)]
    assert np.all(np.isin(draws, E5Env._DO_GRID))


def test_e5_logged_p0_tracks_z_only_in_obs_mode() -> None:
    def correlation(mode: str) -> float:
        env = E5Env(env_seed=0, mode=mode)
        p0, z = [], []
        for _ in range(400):
            z.append(env.Z)
            p0.append(env.generate_p0())
            env.advance()
        return abs(np.corrcoef(p0, z)[0, 1])

    assert correlation("obs") > 0.9
    assert correlation("do") < 0.2


def test_e5_layers_toggle_their_edges() -> None:
    full = set(E5Env().adjacency_edges)
    assert (E5Env.K_HARM, E5Env.P0) in full
    assert (E5Env.K_HARM, 4 + E5Env.K_MID) in full
    no_subdom = set(E5Env(subdom=0.0).adjacency_edges)
    no_chain = set(E5Env(chain_gamma=0.0).adjacency_edges)
    assert full - no_subdom == {(E5Env.K_HARM, E5Env.P0)}
    assert full - no_chain == {(E5Env.K_HARM, 4 + E5Env.K_MID)}
    assert E5Env(theta=0.0).latent_edges == []
    assert E5Env(mode="obs").latent_edges == [("Z", "K_harm"), ("Z", "P0_behavior")]
    assert E5Env().latent_edges == [("Z", "K_harm")]


def test_e5_gated_effect_is_inert_outside_the_gate() -> None:
    env = E5Env(env_seed=0, theta=0.0)
    env.reset(episode=0)

    def terminal_harm(g1: float, g2: float, p0: float) -> float:
        env.reset(episode=0)
        for index, value in ((E5Env.G1, g1), (E5Env.G2, g2), (E5Env.P0, p0)):
            env.apply_action(index, value)
        env.advance()
        return float(env.advance()[E5Env.K_HARM])

    in_gate = terminal_harm(2.0, 25.0, 50.0) - terminal_harm(2.0, 25.0, -50.0)
    outside = terminal_harm(3.5, 25.0, 50.0) - terminal_harm(3.5, 25.0, -50.0)
    assert in_gate > 0.0
    assert outside == 0.0

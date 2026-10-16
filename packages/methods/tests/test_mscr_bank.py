import numpy as np
import pytest

from oran_causal.core.multiplicity import weighted_step_up
from oran_causal.methods.controlled.mscr_bank import (
    Execution,
    MSCRConfig,
    build_banks,
    discover_mscr,
    resolve_device,
    rowsums_cpu,
    strata_plan,
)


def _by(p, q=0.05):
    return weighted_step_up(np.asarray(p, float), None, q, reshape=True)


@pytest.mark.claim("THM-BY-FIN")
def test_by_cutoffs_m8():
    assert _by([1 / 3000] + [1.0] * 7)[0]
    assert not _by([0.00231] + [1.0] * 7).any()
    assert _by([0.001, 0.004, 0.0068] + [1.0] * 5).sum() == 3


def test_lagged_columns_are_conditioners_not_tested():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(400, 5))
    y = np.column_stack([x[:, 0] + 0.1 * rng.normal(size=400), rng.normal(size=400)])
    res = discover_mscr(x, y, 3, 1, MSCRConfig(n_perm=199))
    assert res.pvals.shape == res.declared.shape == (2, 3)
    assert res.declared[0, 0]
    assert not res.declared[1].any()


def test_gated_edge_found_via_single_conditioner():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(3000, 4))
    y = 5.0 * x[:, 2] + 2.0 * x[:, 0] * (x[:, 1] > 0.83) + 0.1 * rng.normal(size=3000)
    res = discover_mscr(x, y, 4, 0, MSCRConfig(n_perm=299))
    assert res.param_edges() >= {(0, 0), (0, 1), (0, 2)}


def test_rejects_bad_shapes():
    x = np.zeros((50, 4))
    with pytest.raises(ValueError, match="row mismatch"):
        discover_mscr(x, np.zeros((49, 2)), 2, 0)
    with pytest.raises(ValueError, match="n_params"):
        discover_mscr(x, np.zeros((50, 2)), 5, 0)


def test_execution_settings_never_change_the_bank():
    rng = np.random.default_rng(4)
    x = rng.uniform(size=(500, 3))
    y = rng.normal(size=(500, 2))
    cfg = MSCRConfig(n_perm=120)
    ref = build_banks(
        x, y, cfg, [np.random.default_rng([1, j]) for j in range(2)], Execution(1, "cpu")
    )
    for run in (Execution(2, "cpu", 1 << 12), Execution(3, "cpu", 777)):
        got = build_banks(x, y, cfg, [np.random.default_rng([1, j]) for j in range(2)], run)
        for (d0, n0, _), (d1, n1, _) in zip(ref, got, strict=True):
            assert np.array_equal(d0, d1)
            assert np.array_equal(n0, n1)


def test_parallel_draws_leave_generator_where_serial_draws_do():
    rng = np.random.default_rng(5)
    x = rng.uniform(size=(300, 3))
    y = rng.normal(size=(300, 1))
    cfg = MSCRConfig(n_perm=50)
    plan = strata_plan(x, cfg)
    ys = [[[y[rows, 0] for rows, _, _ in kept] for kept in plan]]
    serial, parallel = np.random.default_rng(9), np.random.default_rng(9)
    rowsums_cpu(ys, plan, cfg, [serial], Execution(1, "cpu", 1 << 10))
    rowsums_cpu(ys, plan, cfg, [parallel], Execution(2, "cpu", 1 << 10))
    assert serial.random() == parallel.random()


def test_resolve_device_rejects_unknown():
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="MSCR device"):
        resolve_device("tpu")


def test_torch_kernel_is_bit_identical_on_cpu():
    pytest.importorskip("torch")
    from oran_causal.methods.controlled.mscr_torch import device_selfcheck  # noqa: PLC0415

    assert device_selfcheck("torch:cpu")
    rng = np.random.default_rng(6)
    x = rng.uniform(size=(700, 3))
    y = rng.normal(size=(700, 2))
    cfg = MSCRConfig(n_perm=64)
    ref = build_banks(
        x, y, cfg, [np.random.default_rng([2, j]) for j in range(2)], Execution(1, "cpu")
    )
    got = build_banks(
        x, y, cfg, [np.random.default_rng([2, j]) for j in range(2)], Execution(2, "torch:cpu")
    )
    for (_, n0, _), (_, n1, _) in zip(ref, got, strict=True):
        assert np.array_equal(n0, n1)

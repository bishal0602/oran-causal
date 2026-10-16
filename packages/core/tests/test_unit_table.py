import dataclasses

import numpy as np
import pytest

from oran_causal.core.data import UnitTable

pytestmark = pytest.mark.claim("DEF-SYS-HYP")


def make_table() -> UnitTable:
    n, cells = 5, 3
    y = {("own", "load"): np.arange(n, dtype=float)}
    return UnitTable(
        episode=np.array([0, 0, 0, 1, 1]),
        seed=np.array([10, 10, 10, 11, 11]),
        fold=np.full(n, -1),
        family=np.array([1, 0, 1, 1, 1]),
        xapp=np.array(["ES"] * n, dtype=object),
        c=np.array([2, 0, 2, 2, 2]),
        t0=np.array([100, 110, 170, 100, 160]),
        mode=np.array([0, 2, 1, 2, 0]),
        applied=np.array([0, 2, 1, 2, 0]),
        p=np.full(n, 0.5),
        probs=np.tile([0.5, 0.0, 0.5, 0.0], (n, 1)),
        step=np.array([1.0, -1.0, 0.0, 1.0, -1.0]),
        sgn=np.array([1.0, -1.0, 1.0, 1.0, -1.0]),
        y=y,
        pre={k: v + 1 for k, v in y.items()},
        ycell={"load": np.zeros((n, cells))},
        rel_mask={"own": np.eye(cells, dtype=bool)[[2, 0, 2, 2, 2]]},
        z={"pre_own_load": y[("own", "load")] + 1},
        prev=np.array([-1, -1, 0, -1, 3]),
        n_cells=cells,
        H=90,
        H_pre=90,
        relations=("own",),
        kpis=("load",),
        families=("carrier", "sleep"),
        levels=np.array([1.0, 0.5, 0.0, 1.0]),
        meta={"n_units": n},
    )


def test_level_and_oriented_treatment() -> None:
    t = make_table()
    assert t.n == 5
    assert np.array_equal(t.level, [1.0, 0.0, 0.5, 0.0, 1.0])
    assert np.array_equal(t.x, [1.0, -0.0, 0.5, 0.0, -1.0])
    assert np.array_equal(t.rows_of("sleep"), [0, 2, 3, 4])
    assert np.array_equal(t.rows_of("carrier"), [1])


def test_subset_remaps_prev_and_keeps_shared_fields() -> None:
    t = make_table()
    s = t.subset(np.array([True, False, True, False, True]))
    assert np.array_equal(s.t0, [100, 170, 160])
    assert np.array_equal(s.prev, [-1, 0, -1])
    assert np.array_equal(s.y[("own", "load")], [0.0, 2.0, 4.0])
    assert np.array_equal(s.levels, t.levels)
    assert s.families == t.families
    assert s.meta == t.meta
    assert s.meta is not t.meta
    assert np.array_equal(t.subset([3, 4]).prev, [-1, 0])


def test_table_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_table().H = 60  # ty: ignore[invalid-assignment]

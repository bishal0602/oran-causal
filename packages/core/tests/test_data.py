import dataclasses

import numpy as np
import pytest

from oran_causal.core import data


def make_dataset(n: int = 4) -> data.Dataset:
    return data.Dataset(
        world="E1",
        regime="R1",
        n=n,
        seed=0,
        action_names=("P0",),
        kpi_names=("K0",),
        X_action=np.zeros((n, 1)),
        X_kpi_lag=np.full((n, 1), np.nan),
        Y=np.zeros((n, 1)),
        designs=(data.Design(kind="iid", dist={"name": "uniform", "lo": 0.0, "hi": 1.0}),),
        candidates=(("P0", "K0"),),
    )


def test_dataset_defaults_and_immutability() -> None:
    dataset = make_dataset()
    assert dataset.time_index is None
    assert dataset.context is None
    assert dataset.meta == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataset.__setattr__("n", 5)


def test_meta_default_is_not_shared() -> None:
    first, second = make_dataset(), make_dataset()
    first.meta["k"] = 1
    assert second.meta == {}

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


def test_world_and_regime_ids() -> None:
    assert data.WORLDS == ("E1", "E2", "E3", "E4", "E5")
    assert data.REGIMES == ("R1", "R2", "R3", "R4")


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


def test_truth_edges_are_hashable_sets() -> None:
    truth = data.Truth(
        world="E1",
        regime="R1",
        edges=frozenset({("P0", "K0")}),
        signs={("P0", "K0"): 1},
        null_edges=frozenset(),
    )
    assert ("P0", "K0") in truth.edges


def test_result_holds_edge_results() -> None:
    edge = data.EdgeResult(source="P0", target="K0", score=1.0, p=0.01, sign=1, declared=True)
    result = data.Result(method="m", version="1", edges=(edge,), cpu_s=0.1, config={})
    assert result.edges[0].declared
    assert result.notes == {}

import json

from oran_causal.experiments.campaign.shard import (
    assignment,
    complete_dataset_keys,
    partition,
    unit_cost,
)
from oran_causal.experiments.campaign.spec import expand


def test_partition_whole_datasets_exact_cover_deterministic(kit, shipped):
    units = expand(kit.make_spec(), shipped)
    parts = partition(units, 5)
    assert sorted(u.key for p in parts for u in p) == sorted(u.key for u in units)
    owner: dict[tuple, int] = {}
    for i, p in enumerate(parts):
        for u in p:
            assert owner.setdefault(u.dataset, i) == i
    assert [[u.key for u in p] for p in partition(units, 5)] == [[u.key for u in p] for p in parts]
    ct = {f"d_by|E1|R1|n{n}": 100.0 for n in (300, 400)}
    loads = [sum(unit_cost(u, ct) for u in p) for p in partition(units, 3, ct)]
    assert max(loads) - min(loads) <= 100.0 + 0.4 * 2


def test_parts_start_with_small_n(kit, shipped):
    for p in partition(expand(kit.make_spec(), shipped), 2):
        assert [u.n for u in p] == sorted(u.n for u in p)


def test_assignment_maps_every_key_to_its_part(kit, shipped):
    units = expand(kit.make_spec(), shipped)
    parts = partition(units, 4)
    a = assignment(units, 4)
    assert a == {u.key: i for i, p in enumerate(parts) for u in p}


def test_relaunch_reruns_whole_incomplete_datasets(kit, shipped, tmp_path):
    units = expand(kit.small_spec(), shipped)
    by_ds: dict[tuple, list] = {}
    for u in units:
        by_ds.setdefault(u.dataset, []).append(u)
    complete, partial = list(by_ds.values())[:2]
    recs = [{"key": u.key, "status": "ok"} for u in complete]
    recs += [
        {"key": partial[0].key, "status": "infeasible"},
        {"key": partial[1].key, "status": "error"},
    ]
    path = tmp_path / "old.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert complete_dataset_keys(units, [path]) == {u.key for u in complete}
    assert complete_dataset_keys(units, [path], accept=lambda r: False) == set()

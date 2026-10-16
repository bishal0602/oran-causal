import json

import pytest

from oran_causal.experiments.campaign.merge import (
    dataset_of,
    iter_jsonl,
    merge,
    merge_records,
    platform_of,
    summary_path,
)
from oran_causal.experiments.campaign.run import Shard, run_part
from oran_causal.experiments.campaign.spec import expand, spec_sha256

ONE = Shard(0, 1)


def write(path, recs, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs) + tail, encoding="utf-8")


def test_merge_missing_duplicate_broken(kit, opts, tmp_path):
    spec = kit.small_spec()
    full = tmp_path / "full.jsonl"
    run_part(spec, ONE, full, opts())
    recs = kit.read_jsonl(full)
    p1, p2 = tmp_path / "p1.jsonl", tmp_path / "p2.jsonl"
    err = dict(recs[0], status="error", error="boom", edges=[])
    inf = dict(recs[1], status="infeasible", reason="x", edges=[])
    write(p1, [err, inf, *recs[2:5]], "{broken\n")
    write(p2, recs[:4])
    s = merge(spec, [p1, p2], tmp_path / "m.jsonl")
    merged = {r["key"]: r for r in kit.read_jsonl(tmp_path / "m.jsonl")}
    assert (s["n_records"], s["bad_lines"], s["duplicates"]) == (5, 1, 4)
    assert merged[recs[0]["key"]]["status"] == "ok"
    assert merged[recs[1]["key"]]["status"] == "ok"
    assert set(s["missing"]) == {r["key"] for r in recs[5:]}
    assert not s["errors"]
    assert not s["conflicts"]
    flip = json.loads(json.dumps(recs[2]))
    flip["edges"][0]["declared"] = not flip["edges"][0]["declared"]
    p3 = tmp_path / "p3.jsonl"
    write(p3, [inf, err, flip])
    best, st = merge_records([p1, p3])
    assert best[recs[1]["key"]]["status"] == "infeasible"
    assert st["conflicts"] == [recs[2]["key"]]


def test_dev_merge_drops_eval_records(kit, opts, tmp_path):
    spec = kit.small_spec()
    good = tmp_path / "g.jsonl"
    run_part(spec, ONE, good, opts())
    recs = kit.read_jsonl(good)
    mixed = tmp_path / "mixed.jsonl"
    write(mixed, [dict(recs[0], run_mode={"mode": "eval", "protocol_sha256": "a" * 64})])
    s = merge(spec, [good, mixed], tmp_path / "m.jsonl")
    assert (s["foreign"], s["n_records"], s["mode"]) == (1, len(recs), "dev")


def test_merge_one_platform_per_dataset(kit, opts, tmp_path):
    spec = kit.small_spec()
    out = tmp_path / "r.jsonl"
    run_part(spec, ONE, out, opts())
    recs = kit.read_jsonl(out)
    ds0 = dataset_of(recs[0])
    same = [r for r in recs if dataset_of(r) == ds0]
    assert len(same) == 2
    a = [dict(r, host={"platform": "kaggle"}) for r in recs]
    b = [dict(same[0], host={"platform": "colab"})]
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(pa, a)
    write(pb, b)
    best, st = merge_records([pa, pb])
    assert {platform_of(best[r["key"]]) for r in same} == {"kaggle"}
    assert st["platform_switched_datasets"] == 1
    write(pa, [r for r in a if r["key"] != same[0]["key"]])
    _, st = merge_records([pa, pb])
    assert len(st["mixed_platform_datasets"]) == 1


def test_gzip_merge_is_sorted_with_summary(kit, opts, tmp_path):
    spec = kit.small_spec()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_part(spec, Shard(0, 2), a, opts())
    run_part(spec, Shard(1, 2), b, opts())
    out = tmp_path / "m.jsonl.gz"
    s = merge(spec, [a, b], out)
    assert s["n_records"] == s["n_expected"] == len(expand(spec))
    assert not s["missing"]
    assert summary_path(out) == tmp_path / "m.jsonl.summary.json"
    assert summary_path(out).exists()
    keys = [r["key"] for r in iter_jsonl(out)]
    assert keys == sorted(keys)


def test_eval_merge_refuses_mixed_partitions(kit, opts, tmp_path, monkeypatch):
    texts, sha = kit.frozen_texts()
    spec = kit.eval_spec(sha)
    stamp = {"run_mode": {"mode": "eval", "protocol_sha256": sha}}
    sha_spec = spec_sha256(spec)
    recs = [
        {
            "key": u.key,
            "status": "ok",
            **stamp,
            "integrity": {"spec_sha256": sha_spec, "parts": parts},
        }
        for u, parts in zip(expand(spec, texts)[:2], (2, 3), strict=True)
    ]
    p = tmp_path / "p.jsonl"
    write(p, recs)
    out = tmp_path / "x.jsonl"
    with pytest.raises(ValueError, match="merge refused: EVAL records mix parts"):
        merge(spec, [p], out, texts)
    assert not out.exists()
    assert summary_path(out).exists()

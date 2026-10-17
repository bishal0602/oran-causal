import argparse
import json
from pathlib import Path

import pytest

from oran_causal.experiments.campaign import __main__ as C
from oran_causal.experiments.campaign import run as R
from oran_causal.experiments.campaign.spec import ProtocolTexts, expand
from oran_causal.experiments.launch.bundle import FROZEN_COPY_REL, PROTOCOL_REL


def write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_part_parsing() -> None:
    assert C.parse_part("2/5") == (2, 5)
    for bad in ("5/5", "x/2", "1"):
        with pytest.raises(argparse.ArgumentTypeError):
            C.parse_part(bad)


def test_list_prints_units_and_part_costs(kit, tmp_path) -> None:
    spec = kit.small_spec()
    lines: list[str] = []
    assert (
        C.main(["list", "--spec", str(write_spec(tmp_path, spec)), "--parts", "2"], lines.append)
        == 0
    )
    summary = json.loads(lines[0])
    units = expand(spec)
    assert summary["mode"] == "dev"
    assert summary["units"] == len(units)
    assert summary["datasets"] == len({u.dataset for u in units})
    assert sum(summary["by_arm"].values()) == len(units)
    assert len(summary["part_est_cpu_s"]) == 2


def test_run_then_merge_through_the_cli(kit, tmp_path) -> None:
    spec_path = write_spec(tmp_path, kit.small_spec())
    providers = kit.quiet_providers()
    for i in (0, 1):
        argv = ["run", "--spec", str(spec_path), "--part", f"{i}/2", "--no-isolate"]
        argv += ["--out", str(tmp_path / f"res_{i}.jsonl")]
        assert C.main(argv, providers=providers) == 0
    lines: list[str] = []
    argv = ["merge", "--spec", str(spec_path), "--inputs", str(tmp_path / "res_*.jsonl")]
    assert C.main([*argv, "--out", str(tmp_path / "m.jsonl")], lines.append) == 0
    summary = json.loads(lines[0])
    assert summary["n_ok"] == len(expand(kit.small_spec()))
    assert summary["missing"] == 0
    recs = kit.read_jsonl(tmp_path / "res_0.jsonl")
    assert recs[0]["integrity"]["part"] == 0
    assert recs[0]["integrity"]["parts"] == 2


def test_run_skips_datasets_complete_in_earlier_shards(kit, tmp_path) -> None:
    spec_path = write_spec(tmp_path, kit.small_spec())
    providers = kit.quiet_providers()
    first = tmp_path / "old" / "res_0.jsonl"
    base = ["run", "--spec", str(spec_path), "--no-isolate"]
    C.main([*base, "--out", str(first)], providers=providers)
    again = tmp_path / "new.jsonl"
    skip = ["--skip-complete-from", str(tmp_path / "old" / "*.jsonl")]
    C.main([*base, "--out", str(again), *skip], providers=providers)
    assert not again.exists() or kit.read_jsonl(again) == []


def test_protocol_texts_come_from_the_bundle_root(tmp_path) -> None:
    (tmp_path / PROTOCOL_REL).parent.mkdir(parents=True)
    (tmp_path / PROTOCOL_REL).write_bytes(b"live")
    (tmp_path / FROZEN_COPY_REL).parent.mkdir(parents=True)
    (tmp_path / FROZEN_COPY_REL).write_bytes(b"copy")
    assert C.protocol_texts(tmp_path) == ProtocolTexts(live=b"live", frozen_copy=b"copy")
    assert C.protocol_texts(tmp_path / "empty").live is not None


def test_aggregate_cost_tables_are_converted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(C, "cost_table_from_agg", lambda agg: {"converted": 1.0})
    path = tmp_path / "agg.json"
    path.write_text(json.dumps({"cost": {}}), encoding="utf-8")
    assert C.read_cost_table(path) == {"converted": 1.0}
    path.write_text(json.dumps({"a|E1|R1|n300": 2.0}), encoding="utf-8")
    assert C.read_cost_table(path) == {"a|E1|R1|n300": 2.0}
    assert C.read_cost_table(None) is None


def test_dataset_provider_replaces_the_generator(kit, opts, tmp_path) -> None:
    seen: list[str] = []

    def source(u):
        seen.append(u.dataset)
        return R.generated_dataset(u)

    spec = kit.small_spec()
    units = expand(spec)
    R.run_units(spec, units, tmp_path / "r.jsonl", opts(providers={"dataset": source}))
    assert set(seen) == {u.dataset for u in units}
    assert len(seen) == len(set(seen))

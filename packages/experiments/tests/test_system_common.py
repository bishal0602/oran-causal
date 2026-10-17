import json
from pathlib import Path

import numpy as np
import pytest

from oran_causal.experiments.system.common import (
    Block,
    FrozenDoc,
    Job,
    Part,
    SeedRegistry,
    append_record,
    cfg_seed,
    check_layout,
    check_seed,
    read_records,
    run_jobs,
    sha_lf,
    to_json,
)


def test_block_and_seed_checks() -> None:
    b = Block.sized(100, 10)
    assert (b.lo, b.hi) == (100, 109)
    assert 105 in b
    assert 110 not in b
    assert b.within(Block(0, 200))
    assert not b.within(Block(101, 200))
    assert b.overlaps(Block(109, 120))
    assert not b.overlaps(Block(110, 120))
    assert check_seed(105, b, [Block(0, 50)]) == 105
    with pytest.raises(ValueError, match="outside"):
        check_seed(99, b)
    with pytest.raises(ValueError, match="forbidden"):
        check_seed(105, b, [Block(104, 106)])


@pytest.mark.claim("PRO-SEEDS")
def test_check_layout() -> None:
    check_layout({"a": Block(0, 9), "b": Block(10, 19)}, Block(0, 99), [Block(50, 60)])
    with pytest.raises(ValueError, match="overlap"):
        check_layout({"a": Block(0, 10), "b": Block(10, 19)}, Block(0, 99), [])
    with pytest.raises(ValueError, match="forbidden"):
        check_layout({"a": Block(0, 9)}, Block(0, 99), [Block(5, 6)])


def test_part() -> None:
    p = Part.parse("1/3")
    assert [u for u in range(7) if p.owns(u)] == [1, 4]
    assert str(p) == "1/3"
    assert Part.parse(None) == Part()
    with pytest.raises(ValueError, match="0 <= i < k"):
        Part.parse("3/3")


def test_cfg_seed() -> None:
    assert cfg_seed(188000, False) == 188000
    assert cfg_seed(188000, True) == 188000 % 31


def test_frozen_doc(tmp_path: Path) -> None:
    doc = tmp_path / "P.md"
    doc.write_bytes(b"# P\r\nFROZEN: yes (today)\r\n")
    want = sha_lf(doc)
    assert want == sha_lf(_lf_copy(tmp_path, doc))
    st = FrozenDoc("docs/P.md", doc, want).status()
    assert st == {
        "doc": "docs/P.md",
        "exists": True,
        "line": "FROZEN: yes (today)",
        "sha256": want,
        "want": want,
        "frozen": True,
    }
    assert not FrozenDoc("docs/P.md", doc, "0" * 64).status()["frozen"]
    doc.write_text("FROZEN: no\n")
    assert not FrozenDoc("docs/P.md", doc, sha_lf(doc)).status()["frozen"]
    assert FrozenDoc("docs/P.md", doc, sha_lf(doc), needs_line=False).status()["frozen"]
    assert not FrozenDoc("x", tmp_path / "none.md", want).status()["exists"]


def _lf_copy(tmp: Path, doc: Path) -> Path:
    out = tmp / "lf.md"
    out.write_bytes(doc.read_bytes().replace(b"\r\n", b"\n"))
    return out


def test_registry(tmp_path: Path) -> None:
    p = tmp_path / "seeds.json"
    p.write_text(
        json.dumps(
            {"E6": {"a": [[10, 20]], "b": {"c": [[30, 40], [50, 60]]}}, "rng_stream_tags": [1, 2]}
        )
    )
    reg = SeedRegistry(p)
    assert sorted(reg.ranges()) == [(10, 20), (30, 40), (50, 60)]
    assert reg.covers(Block(31, 39))
    assert not reg.covers(Block(35, 55))
    assert reg.has_tags([1])
    assert not reg.has_tags([1, 3])
    assert reg.e6("a") == [[10, 20]]
    assert reg.e6("zz", 0) == 0


def test_records_io(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    append_record(
        out, {"a": np.int64(3), "b": np.float32(0.5), "c": np.arange(2), "d": np.bool_(True)}
    )
    with out.open("a") as fh:
        fh.write("not json\n")
    assert read_records(out) == [{"a": 3, "b": 0.5, "c": [0, 1], "d": True}]
    assert read_records(tmp_path / "missing.jsonl") == []
    assert to_json(Path("x")) == "x"


def _jobs(calls: list) -> list[Job]:
    def make(seed: int, x: str) -> Job:
        def run() -> dict:
            calls.append((seed, x))
            return {"kind": "job", "key": [seed, x], "seed": seed, "smoke": False, "cpu_s": 0}

        return Job((seed, x), seed, run)

    return [make(s, x) for s in (1, 2) for x in ("a", "b")]


def test_run_jobs_shards_and_resumes(tmp_path: Path) -> None:
    out, calls = tmp_path / "o.jsonl", []
    head = {"kind": "header", "stage": "eval"}
    assert run_jobs(out, head, _jobs(calls), Part(0, 2), False) == 2
    assert calls == [(1, "a"), (2, "a")]
    assert run_jobs(out, head, _jobs(calls), Part(), False) == 2
    assert calls[2:] == [(1, "b"), (2, "b")]
    assert run_jobs(out, head, _jobs(calls), Part(), False) == 0
    recs = read_records(out)
    jobs = [r for r in recs if r["kind"] == "job"]
    assert len(jobs) == 4
    assert all("secs" in r and "rss_mb" in r for r in jobs)
    closes = [r for r in recs if r["kind"] == "close"]
    assert [c["n_jobs"] for c in closes] == [2, 2, 0]
    assert all(c["n_total"] == 4 for c in closes)


def test_run_jobs_smoke_runs_first_seed_only(tmp_path: Path) -> None:
    calls: list = []
    run_jobs(tmp_path / "o.jsonl", {"kind": "header", "stage": "s"}, _jobs(calls), Part(1, 2), True)
    assert calls == [(1, "a"), (1, "b")]

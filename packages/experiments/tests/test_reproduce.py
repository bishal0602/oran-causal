import dataclasses
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from oran_causal.experiments.campaign.spec import expand
from oran_causal.experiments.paths import SPECS
from oran_causal.experiments.reproduce import checks, data, plan, t0, t0_figures, t1, verify
from oran_causal.experiments.reproduce.checks import Check, Status
from oran_causal.experiments.reproduce.context import Context, guarded, missing


def ctx_at(tmp_path: Path, **kw: Any) -> Context:
    return Context(tmp_path / "records", tmp_path / "out", log=lambda _: None, **kw)


def test_passed_counts_skips_as_passing() -> None:
    rows = [Check("a", "x", Status.PASS), Check("a", "y", Status.SKIP)]
    assert checks.passed(rows)
    assert not checks.passed([*rows, Check("a", "z", Status.DIFF)])
    assert not checks.passed([Check("a", "z", Status.ERROR)])


def test_table_has_one_row_per_check() -> None:
    text = checks.table([Check("a", "x", Status.PASS, "ok"), Check("b", "y", Status.DIFF)])
    assert text.splitlines() == [
        "| Item | Target | Status | Detail |",
        "|---|---|---|---|",
        "| a | x | pass | ok |",
        "| b | y | diff |  |",
    ]


def test_compare_bytes(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x\ny\n")
    b.write_bytes(b"x\r\ny\r\n")
    assert checks.compare_bytes("i", a, b).status is Status.DIFF
    assert checks.compare_bytes("i", a, b, eol=True).status is Status.PASS
    assert checks.compare_bytes("i", tmp_path / "c", b).detail == "not regenerated"
    assert checks.compare_bytes("i", a, tmp_path / "c").status is Status.ERROR


def test_close_tree_reports_paths() -> None:
    a = {"x": 1.0, "y": [1, {"z": "s"}]}
    assert checks.close_tree(a, {"x": 1.0 + 1e-13, "y": [1, {"z": "s"}]}, 1e-12) == []
    assert checks.close_tree(a, {"x": 1.1, "y": [1, {"z": "t"}]}, 1e-12) == [
        "/x: 1.0 != 1.1",
        "/y[1]/z: 's' != 't'",
    ]
    assert checks.close_tree({"x": 1}, {"w": 1}, 0.0) == [": keys ['w', 'x']"]


@pytest.fixture
def record_set(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "records"
    files = {"a.txt": b"alpha", "sub/b.txt": b"beta", "sub/c.pyc": b"gamma"}
    for rel, body in files.items():
        (root / "s1" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "s1" / rel).write_bytes(body)
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    lines = [f"{hashlib.sha256(b).hexdigest()}  {rel}" for rel, b in files.items()]
    (manifests / "s1.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zenodo = tmp_path / "data" / "zenodo.yaml"
    zenodo.write_text(
        "sets:\n"
        "  - {name: s1, manifest: manifests/s1.sha256, files: 3, bytes: 14, doi: null}\n"
        "  - {name: s2, manifest: manifests/s1.sha256, files: 3, bytes: 14, doi: null}\n",
        encoding="utf-8",
    )
    return root, zenodo


def test_verify_set_hashes_every_entry(record_set: tuple[Path, Path]) -> None:
    root, zenodo = record_set
    rs = data.record_sets(zenodo)["s1"]
    assert data.verify_set(rs, root).detail == "3 files verified"
    (root / "s1" / "sub" / "c.pyc").unlink()
    assert data.verify_set(rs, root).status is Status.DIFF
    assert data.verify_set(rs, root, (".pyc",)).detail == "2 files verified, 1 known missing"
    (root / "s1" / "a.txt").write_bytes(b"changed")
    row = data.verify_set(rs, root, (".pyc",))
    assert (row.status, row.detail) == (Status.DIFF, "0 missing, 1 differ (first a.txt)")


def test_fetch_skips_absent_sets_until_they_have_a_doi(record_set: tuple[Path, Path]) -> None:
    root, zenodo = record_set
    rows = data.fetch(None, root, zenodo=zenodo)
    assert [(r.target, r.status) for r in rows] == [("s1", Status.PASS), ("s2", Status.SKIP)]
    assert "no Zenodo DOI" in rows[1].detail
    only = data.fetch(["s2"], root, verify_only=True, zenodo=zenodo)
    assert only[0].detail.startswith("not present")
    with pytest.raises(KeyError, match="s3"):
        data.fetch(["s3"], root, zenodo=zenodo)


def zenodo_with_doi(tmp_path: Path, sha: str | None) -> Path:
    zenodo = tmp_path / "data" / "zenodo.yaml"
    zenodo.parent.mkdir(exist_ok=True)
    pin = "null" if sha is None else sha
    zenodo.write_text(
        "sets:\n"
        "  - {name: s1, manifest: manifests/s1.sha256, files: 3, bytes: 14, "
        f"doi: 10.5281/zenodo.42, archive_sha256: {pin}}}\n",
        encoding="utf-8",
    )
    return zenodo


@pytest.fixture
def served(tmp_path: Path, record_set: tuple[Path, Path]) -> tuple[Path, Path, str]:
    root, zenodo = record_set
    rs = data.record_sets(zenodo)["s1"]
    record = tmp_path / "served" / "42" / "files"
    record.mkdir(parents=True)
    data.build_archive(rs, record / "s1.tar.zst", root)
    sha = data.file_sha256(record / "s1.tar.zst")
    return root, zenodo_with_doi(tmp_path, sha), (tmp_path / "served").as_uri()


def test_download_extracts_a_served_archive_and_fetch_verifies_it(
    served: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, zenodo, base = served
    target = tmp_path / "fresh"
    rows = data.fetch(["s1"], target, zenodo=zenodo, base_url=base)
    assert [(r.status, r.detail) for r in rows] == [(Status.PASS, "3 files verified")]
    assert (target / "s1" / "sub" / "b.txt").read_bytes() == b"beta"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".fetch-")] == []


def test_download_refuses_a_checksum_mismatch(
    served: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, _, base = served
    zenodo = zenodo_with_doi(tmp_path, "0" * 64)
    rs = data.record_sets(zenodo)["s1"]
    row = data.download(rs, tmp_path / "fresh", base)
    assert row.status is Status.DIFF
    assert not (tmp_path / "fresh" / "s1").exists()


def test_download_needs_a_pinned_checksum_and_a_reachable_record(
    served: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, _, base = served
    unpinned = data.record_sets(zenodo_with_doi(tmp_path, None))["s1"]
    assert "pins no archive_sha256" in data.download(unpinned, tmp_path / "fresh", base).detail
    pinned = data.record_sets(zenodo_with_doi(tmp_path, "0" * 64))["s1"]
    absent = (tmp_path / "nowhere").as_uri()
    assert data.download(pinned, tmp_path / "fresh", absent).status is Status.ERROR


def test_download_rejects_a_path_traversal_member(tmp_path: Path) -> None:
    evil = tmp_path / "served" / "42" / "files"
    evil.mkdir(parents=True)
    payload = tmp_path / "payload"
    payload.write_bytes(b"x")
    with tarfile.open(evil / "s1.tar.zst", "w:zst") as tar:
        tar.add(payload, arcname="s1/ok.txt")
        tar.add(payload, arcname="../escaped.txt")
    zenodo = zenodo_with_doi(tmp_path, data.file_sha256(evil / "s1.tar.zst"))
    rs = data.record_sets(zenodo)["s1"]
    root = tmp_path / "fresh"
    row = data.download(rs, root, (tmp_path / "served").as_uri())
    assert (row.status, "unsafe" in row.detail) == (Status.ERROR, True)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (root / "s1").exists()


def test_archive_url_needs_a_zenodo_doi(record_set: tuple[Path, Path]) -> None:
    rs = data.record_sets(record_set[1])["s1"]
    with pytest.raises(ValueError, match="no Zenodo DOI"):
        data.archive_url(rs)
    with_doi = dataclasses.replace(rs, doi="10.5281/zenodo.42")
    assert data.archive_url(with_doi) == "https://zenodo.org/records/42/files/s1.tar.zst?download=1"


def test_pack_is_deterministic_and_verifies(record_set: tuple[Path, Path], tmp_path: Path) -> None:
    root, zenodo = record_set
    first = data.pack(["s1"], tmp_path / "a", root, known_missing={"s1": (".pyc",)}, zenodo=zenodo)
    again = data.pack(["s1"], tmp_path / "b", root, known_missing={"s1": (".pyc",)}, zenodo=zenodo)
    assert first == again
    assert first[0].detail == "3 members verified"
    sums = (tmp_path / "a" / "SHA256SUMS").read_text(encoding="utf-8")
    assert sums == f"{data.file_sha256(tmp_path / 'a' / 's1.tar.zst')}  s1.tar.zst\n"
    assert (tmp_path / "a" / "s1.tar.zst").read_bytes() == (
        tmp_path / "b" / "s1.tar.zst"
    ).read_bytes()
    assert (tmp_path / "a" / "s1.sha256").is_file()


def test_pack_skips_excused_files_flags_tampering_and_rejects_unknown_sets(
    record_set: tuple[Path, Path], tmp_path: Path
) -> None:
    root, zenodo = record_set
    (root / "s1" / "sub" / "c.pyc").unlink()
    rs = data.record_sets(zenodo)["s1"]
    with pytest.raises(FileNotFoundError):
        data.build_archive(rs, tmp_path / "x.tar.zst", root)
    rows = data.pack(["s1"], tmp_path / "out", root, known_missing={"s1": (".pyc",)}, zenodo=zenodo)
    assert rows[0].detail == "2 members verified"
    (root / "s1" / "a.txt").write_bytes(b"changed")
    data.build_archive(rs, tmp_path / "t.tar.zst", root, (".pyc",))
    assert data.verify_archive(rs, tmp_path / "t.tar.zst", (".pyc",)).status is Status.DIFF
    with pytest.raises(KeyError, match="s9"):
        data.pack(["s9"], tmp_path / "o2", root, zenodo=zenodo)


def test_shipped_sets_have_manifests() -> None:
    sets = data.record_sets()
    assert {"study4_eval", "study1_raw", "posthoc_x9_x16"} <= set(sets)
    for rs in sets.values():
        assert len(data.read_manifest(rs.manifest)) == rs.files


def test_guarded_turns_an_exception_into_an_error_row(tmp_path: Path) -> None:
    logs: list[str] = []

    def boom(ctx: Context) -> list[Check]:
        raise ValueError("bad input")

    rows = guarded("item", boom, Context(tmp_path, tmp_path, log=logs.append))
    assert [(r.item, r.status) for r in rows] == [("item", Status.ERROR)]
    assert "ValueError" in rows[0].detail
    assert logs[0].startswith("[item] 1 checks in ")


def test_missing_lists_absent_record_paths(tmp_path: Path) -> None:
    ctx = ctx_at(tmp_path)
    (ctx.records / "here").mkdir(parents=True)
    assert [r.target for r in missing("i", ctx, "here", "gone")] == ["gone"]


def test_t0_items_skip_without_records(tmp_path: Path) -> None:
    ctx = ctx_at(tmp_path)
    rows = t0.run(ctx, ["study4", "system", "bridging"])
    assert rows
    assert all(r.status is Status.SKIP for r in rows)


def test_t0_refuses_an_unknown_item(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="nope"):
        t0.run(ctx_at(tmp_path), ["nope"])


def test_paper_includes_are_parsed(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "sections" / "s1.tex").write_text(
        "\\includegraphics[width=\\textwidth]{figures/fig_a.pdf}\n"
        "% \\includegraphics{figures/fig_b.png}\n\\includegraphics{figures/fig_c.pdf}\n",
        encoding="utf-8",
    )
    assert t0_figures.included(paper) == {"fig_a.pdf", "fig_c.pdf"}


def test_stored_figures_not_included_may_be_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = tmp_path / "paper"
    (paper / "figures").mkdir(parents=True)
    for name in ("fig_used.pdf", "fig_old.pdf"):
        (paper / "figures" / name).write_bytes(b"%PDF")
    (paper / "s.tex").write_text("\\includegraphics{figures/fig_used.pdf}", encoding="utf-8")
    monkeypatch.setattr(
        t0_figures,
        "compare_pdf",
        lambda mine, stored, work: Check("figures", stored.name, Status.PASS),
    )
    rows = t0_figures.compare_stored(ctx_at(tmp_path, paper=paper), tmp_path / "out")
    assert [(r.target, r.status) for r in rows] == [
        ("fig_old.pdf", Status.SKIP),
        ("fig_used.pdf", Status.PASS),
    ]


def test_unbuilt_figure_is_an_error(tmp_path: Path) -> None:
    row = t0_figures.compare_pdf(tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path)
    assert (row.status, row.detail) == (Status.ERROR, "not regenerated")


def test_t1_smoke_spec_has_one_dataset_per_environment() -> None:
    spec = t1.smoke_spec(json.loads((SPECS / "eval" / "full.json").read_text(encoding="utf-8")))
    units = expand(spec)
    assert len(units) == 16
    assert {u.world for u in units if u.arm in t1.LINEAR_ARMS} == set(t1.WORLDS)
    assert {(u.world, u.n, u.regime, u.seed) for u in units if u.arm == t1.GBM_ARM} == {
        ("E1", 500, "R2", t1.SEED)
    }


def test_t1_agreement_and_stored_records(tmp_path: Path) -> None:
    def rec(key: str, *flags: bool) -> dict[str, Any]:
        edges = [{"source": f"s{i}", "target": "t", "declared": f} for i, f in enumerate(flags)]
        return {"key": key, "edges": edges}

    assert t1.agreement(rec("k", True, False), rec("k", True, False)) == 1.0
    assert t1.agreement(rec("k", True, False), rec("k", True, True)) == 0.5
    assert t1.agreement({}, {}) == 1.0
    path = tmp_path / "m.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for k in ("a|1", "b|2", "c|3"):
            fh.write(json.dumps(rec(k, True)) + "\n")
    assert sorted(t1.stored_records(path, ["a|1", "c|3", "z"])) == ["a|1", "c|3"]


def test_t1_wall_time_is_checked(tmp_path: Path) -> None:
    ticks = iter([0.0, 601.0])
    steps = {"one": lambda ctx: [Check("t1 one", "x", Status.PASS)]}
    rows = t1.run(ctx_at(tmp_path), steps, clock=lambda: next(ticks))
    assert rows[-1].status is Status.DIFF
    assert rows[-1].detail == "601 s (limit 600 s)"


def test_interval_misses_walks_nested_ci90() -> None:
    old = {
        "arms": {
            "A": {"R": 0.5, "R_ci90": [0.4, 0.6], "g": {"rlf": 1.2}, "g_ci90": {"rlf": [1.1, 1.3]}}
        },
        "never_sleep": {"dR": 0.2, "ci90": [0.1, 0.3]},
    }
    same = {"arms": {"A": {"R": 0.45, "g": {"rlf": 1.25}}}, "never_sleep": {"dR": 0.15}}
    assert verify.interval_misses(same, old) == (3, [])
    off = {"arms": {"A": {"R": 0.7, "g": {"rlf": 1.0}}}, "never_sleep": {"dR": 0.15}}
    assert verify.interval_misses(off, old) == (3, ["/arms/A/R", "/arms/A/g.rlf"])


def _v4(c1: str, n_invalid: int, rate: float) -> dict[str, Any]:
    pooled = {"null_raw": {"rate": rate, "ci": [0.04, 0.05], "hits": 1, "n": 2}}
    block = {"verdict": "SUPPORTED", "f_max": 2, "n_invalid": n_invalid, "pooled": pooled}
    return {
        "components": {"C1": c1, "C2a": "SUPPORTED", "C2b": "SUPPORTED", "C3": "SUPPORTED"},
        "claim": "SUPPORTED",
        "C2a": block,
        "pmrt_secondary": {"pmrt_r3": {"C2a": block, "label": None}},
    }


def test_study4_rules() -> None:
    frozen = _v4("SUPPORTED", 1, 0.045)
    assert verify.study4_verdicts(_v4("SUPPORTED", 2, 0.046), frozen) == {}
    assert verify.study4_verdicts(_v4("FAILURE", 2, 0.046), frozen) == {
        "C1": ("FAILURE", "SUPPORTED")
    }
    assert verify.study4_bounds(_v4("SUPPORTED", 2, 0.046), frozen) == ([], [], 2, 2)
    over, outside, _, _ = verify.study4_bounds(_v4("SUPPORTED", 3, 0.06), frozen)
    assert over == ["/C2a", "/pmrt_secondary/pmrt_r3/C2a"]
    assert outside == ["/C2a/pooled/null_raw", "/pmrt_secondary/pmrt_r3/C2a/pooled/null_raw"]


def test_verify_refuses_an_unknown_study(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="study9"):
        verify.verify("study9", tmp_path, ctx_at(tmp_path))


def small_manifest() -> dict[str, Any]:
    return {
        "factors": {"kaggle": 1.0, "vps": 2.0},
        "platforms": {
            "kaggle": {"sessions": 2, "procs": 2, "session_h": 12},
            "vps": {"sessions": 1, "procs": 4, "session_h": 72},
        },
        "items": [
            {
                "name": "a",
                "platform": "kaggle",
                "tiers": ["t2", "t3"],
                "stages": [
                    {"name": "run", "ref_cpu_h": 8, "basis": "M"},
                    {"name": "gt", "ref_cpu_h": 4, "basis": "M", "frozen_in": ["t2"]},
                ],
            },
            {
                "name": "b",
                "platform": "vps",
                "tiers": ["t3"],
                "stages": [{"name": "x", "ref_cpu_h": [2, 6], "basis": "E"}],
            },
        ],
    }


def test_plan_prices_stages_per_tier() -> None:
    m = small_manifest()
    t2 = plan.plan("t2", m)
    assert [(r.item, r.stage) for r in t2.rows] == [("a", "run")]
    assert t2.frozen == (("a", "gt"),)
    assert plan.totals(t2) == {"kaggle": (8.0, 8.0), "all": (8.0, 8.0)}
    t3 = plan.plan("t3", m)
    assert plan.totals(t3) == {"kaggle": (12.0, 12.0), "vps": (2.0, 6.0), "all": (14.0, 18.0)}
    assert plan.plan("t3", m, platform="vps").rows == (plan.Row("b", "x", "vps", 2.0, 6.0, "E"),)
    assert plan.wall_h(12.0, t3.platforms["kaggle"]) == 3.0
    lines = plan.render(t3)
    assert "| vps | 2.00-6.00 | 2.0 | 1.0-3.0 | 4 | 0.2-0.8 |" in lines
    assert lines[-1] == "| all | 14.00-18.00 | | | | |"


def test_plan_rejects_bad_tier_and_platform() -> None:
    with pytest.raises(ValueError, match="t1"):
        plan.plan("t1", small_manifest())
    with pytest.raises(ValueError, match="colab"):
        plan.plan("t2", small_manifest(), platform="colab")


def test_t2_t3_totals_equal_the_cost_table_and_the_plan() -> None:
    t2, t3 = plan.plan("t2"), plan.plan("t3")
    tot2, tot3 = plan.totals(t2), plan.totals(t3)
    table = json.loads((SPECS.parent / "artifacts" / "eval_cost_table.json").read_text())
    units = expand(json.loads((SPECS / "eval" / "full.json").read_text(encoding="utf-8")))
    full = sum(table[f"{u.arm}|{u.world}|{u.regime}|n{u.n}"] for u in units) / 3600
    assert tot3["vps"] == pytest.approx((full, full))
    assert round(full, 2) == 503.64
    assert tot2["vps"][0] == pytest.approx(296.5337, abs=1e-4)
    assert (t3.shards and t3.shards.units, t2.shards and t2.shards.units) == (150960, 123960)
    assert t3.shards is not None
    assert sum(t3.shards.part_ref_h) == pytest.approx(full)
    by_item: dict[str, float] = {}
    for r in t3.rows:
        by_item[r.item] = by_item.get(r.item, 0.0) + r.low
    assert {k: round(by_item[k], 2) for k in ("study1", "study2", "study3")} == {
        "study1": 64.0,
        "study2": 122.04,
        "study3": 59.26,
    }
    assert tot2["kaggle"] == pytest.approx((161.3, 161.3))
    assert tot3["all"] == pytest.approx((972.84, 1013.84), abs=1e-2)

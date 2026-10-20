import copy
import csv
import dataclasses
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from oran_causal.analysis.controlled.integrity import IntegrityInputs
from oran_causal.analysis.controlled.report import (
    DEV_MARK,
    ChunkOptions,
    ReportRequest,
    analyse,
    chunked_analyse,
    csv_tables,
    dev_spec,
    main,
    report_md,
    run_report,
    sub_spec,
)
from oran_causal.analysis.controlled.spec import expected_units
from oran_causal.analysis.controlled.tables import clean
from oran_causal.analysis.records import file_sha256, load_many

DATA = Path(__file__).parent / "data"


def canonical(x: Any) -> str:
    return json.dumps(clean(x), sort_keys=True)


def write_records(records: list, path: Path, *, gz: bool = False) -> Path:
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    return path


def inputs(synth: Any) -> IntegrityInputs:
    return IntegrityInputs(
        freeze_commit=synth.commit, spec_sha=synth.spec_sha, protocol={"ok": True}
    )


@pytest.fixture(scope="module")
def eval_spec() -> dict[str, Any]:
    return json.loads((DATA / "eval_full_spec.json").read_text(encoding="utf-8"))


def test_chunked_equals_analyse(synth: Any, tmp_path: Path) -> None:
    records = synth.records()
    first = write_records(records[::2], tmp_path / "a.jsonl.gz", gz=True)
    second = write_records([*records[1::2], *records[:3]], tmp_path / "b.jsonl")
    reference = analyse(load_many([first, second]), synth.spec, inputs(synth))
    options = ChunkOptions(tmp_dir=tmp_path, log=lambda _: None)
    got, counts = chunked_analyse([first, second], synth.spec, inputs(synth), options)
    assert canonical(got) == canonical(reference)
    assert sum(counts.values()) == len(records) + 3
    assert got["V11_integrity"]["n_duplicates"] == 3
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("xm_eval_report_")]


def test_chunked_equals_analyse_with_defects(synth: Any, tmp_path: Path) -> None:
    records = copy.deepcopy(synth.records())
    records[5]["edges"] = records[5]["edges"][:-1]
    records[7]["code"] = {"commit": "other", "dirty": True}
    records[9]["dataset_sha256"] = "x"
    records = [r for i, r in enumerate(records) if i != 11]
    ghost = {
        **copy.deepcopy(records[0]),
        "key": "ghost|" + records[0]["key"].split("|", 1)[1],
        "arm": "ghost",
    }
    records.append(ghost)
    path = write_records(records, tmp_path / "d.jsonl")
    reference = analyse(load_many([path]), synth.spec, inputs(synth))
    got, _ = chunked_analyse(
        [path], synth.spec, inputs(synth), ChunkOptions(tmp_dir=tmp_path, log=lambda _: None)
    )
    assert canonical(got) == canonical(reference)
    v11 = got["V11_integrity"]
    assert v11["label"] == "PROVISIONAL"
    expected = {"candidates_complete", "commits", "clean", "dataset_hash", "missing", "unexpected"}
    assert expected <= set(v11["failed"])


def test_dev_spec_fields_and_t10_fallback(eval_spec: dict[str, Any]) -> None:
    fields = ("ref", "config", "declare")
    names = ("pmrt_eq", "corr", "mscr_native", "mscr_eq", "pcorr_eq")
    dev = {
        "name": "dev_x",
        "arms": {a: {k: eval_spec["arms"][a][k] for k in fields} for a in names},
        "blocks": [
            {
                "role": "measure",
                "worlds": ["E1"],
                "regimes": ["R1"],
                "ns": [500],
                "kappas": [0.25],
                "seeds": [3000100, 3000119],
            }
        ],
    }
    dev["arms"]["pdcor_eq"] = {"ref": "x:Y", "config": {}, "declare": "by"}
    spec, notes = dev_spec(eval_spec, [dev])
    assert spec["focal"] == "pmrt_eq"
    assert spec["arms"]["pmrt_eq"]["analysis"] == "primary"
    assert "label" not in spec["arms"]["pmrt_eq"]
    assert "pdcor_eq" not in spec["arms"]
    assert spec["arms"]["mscr_native"]["c1_sensitivity_drop"]
    assert spec["arms"]["mscr_eq"]["c2b"] is False
    assert spec["blocks"][0]["arms"] == list(names)
    assert any("T10 fallback" in x for x in notes)
    assert any("pdcor_eq" in x for x in notes)
    dev["arms"]["pmrt_nl_eq"] = {k: eval_spec["arms"]["pmrt_nl_eq"][k] for k in fields}
    spec2, notes2 = dev_spec(eval_spec, [dev])
    assert spec2["focal"] == "pmrt_nl_eq"
    assert not any("T10 fallback" in x for x in notes2)
    only = sub_spec(spec2, "corr")
    assert only["arms"].keys() == {"corr"}
    assert only["focal"] == "pmrt_nl_eq"


def test_dev_without_any_pmrt_arm_is_refused(eval_spec: dict[str, Any]) -> None:
    dev = {
        "name": "d",
        "arms": {"corr": {"ref": "x:Y", "config": {}, "declare": "by"}},
        "blocks": [],
    }
    with pytest.raises(SystemExit, match="neither the EVAL focal arm"):
        dev_spec(eval_spec, [dev])


def test_dev_report_is_watermarked_and_complete(synth: Any, tmp_path: Path) -> None:
    eval_like = copy.deepcopy(synth.spec)
    eval_like["arms"]["ci_native"]["c1_sensitivity_drop"] = True
    dev = {
        "name": "dev_synth",
        "arms": {
            a: {k: d[k] for k in ("ref", "config", "declare")}
            for a, d in synth.spec["arms"].items()
        },
        "blocks": synth.spec["blocks"],
    }
    eval_path, dev_path = tmp_path / "eval.json", tmp_path / "dev.json"
    eval_path.write_text(json.dumps(eval_like), encoding="utf-8")
    dev_path.write_text(json.dumps(dev), encoding="utf-8")
    merged = write_records(synth.records(), tmp_path / "m.jsonl.gz", gz=True)
    out = tmp_path / "rep"
    code = main(
        [
            "--dev",
            "--spec",
            str(eval_path),
            "--dev-spec",
            str(dev_path),
            "--merged",
            str(merged),
            "--out",
            str(out),
            "--tmp-dir",
            str(tmp_path),
        ]
    )
    assert code == 2
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    assert md.startswith("# [DEV, NOT EVAL]")
    assert md.rstrip().endswith(f"> **{DEV_MARK}**")
    sections = md.split("\n## ")[1:]
    assert len(sections) >= 9
    assert all(DEV_MARK in s for s in sections)
    headings = (
        "1. Claim",
        "2. C1",
        "3. C2a",
        "4. C2b",
        "5. C3",
        "6. Power",
        "7. kappa sweep",
        "8. Not in grid",
        "9. Figure-ready",
    )
    for heading in headings:
        assert f"\n## {heading}" in md
    assert "C1 sensitivity (R-56, no effect): without ci_native" in md
    assert "Every eq arm, same rule on R1 + R2, unfiltered (R-56)" in md
    assert "Pre-registered disclosure" in md
    assert "### V0 like-for-like" in md
    assert "### V11 integrity" in md
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["meta"]["mode"] == "DEV"
    assert payload["meta"]["watermark"] == DEV_MARK
    assert payload["tables"]["V4_verdicts"]["C1"]["sensitivity_without"]["dropped"] == ["ci_native"]
    names = {
        "validity_cells",
        "recall_vs_n",
        "like_for_like",
        "paired_diff",
        "information_levels_R2",
        "e4_by_lambda",
        "kappa_sweep",
        "cost",
        "eq_arms",
        "not_in_grid",
    }
    assert {p.stem for p in (out / "csv").iterdir()} == names
    with (out / "csv" / "validity_cells.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(payload["tables"]["V1_validity"])
    assert {"arm", "null_raw_rate", "null_raw_lo"} <= set(rows[0])
    assert next(iter(rows[0])) == "data_mode"
    assert {r["data_mode"] for r in rows} == {"DEV-NOT-EVAL"}
    with (out / "csv" / "eq_arms.csv").open(encoding="utf-8") as handle:
        assert {r["arm"] for r in csv.DictReader(handle)} == {"ci_eq", "pc_eq"}


def test_r4_not_applicable_rows_are_listed(synth: Any, tmp_path: Path) -> None:
    spec = copy.deepcopy(synth.spec)
    for block in spec["blocks"]:
        block.update(worlds=["E4"], regimes=["R4"], ns=[500], e4_lams={"R4": [1.0]})
    records = [
        synth.record(
            u["arm"], u["world"], u["regime"], u["lam"], u["n"], u["seed"], u["role"], liberal=False
        )
        for u in expected_units(spec).values()
    ]
    path = write_records(records, tmp_path / "r4.jsonl")
    out, _ = chunked_analyse(
        [path], spec, inputs(synth), ChunkOptions(tmp_dir=tmp_path, log=lambda _: None)
    )
    meta = {
        "mode": "EVAL",
        "spec_sha256": synth.spec_sha,
        "utc": "t",
        "inputs": [],
        "records_by_arm": {},
        "csv": {},
    }
    md = report_md(out, meta)
    section = md.split("\n## 6. ")[1].split("\n## ")[0]
    assert "| pmrt_eq | E4 R4 lam1 k0.25 n500 | NA (" in section
    assert not [r for r in out["V2_recall"] if r["arm"] == "pmrt_eq" and r["recall"] is not None]
    assert "DEV" not in md.split("\n", 1)[0]
    assert DEV_MARK not in md


def test_report_not_in_grid_section(synth: Any, tmp_path: Path) -> None:
    spec = copy.deepcopy(synth.spec)
    others = [a for a in spec["arms"] if a != "ci_native"]
    spec["blocks"] = [{**b, "arms": others} for b in synth.spec["blocks"]] + [
        {**b, "ns": [500], "arms": ["ci_native"]} for b in synth.spec["blocks"]
    ]
    spec["not_in_grid"] = [
        {
            "ruling": "R-x",
            "arms": ["ci_native"],
            "regimes": None,
            "ns": [1000],
            "kappas": None,
            "reason": "test trim",
        }
    ]
    records = [
        r for r in synth.records() if not (r["arm"] == "ci_native" and r["job"]["n"] == 1000)
    ]
    path = write_records(records, tmp_path / "g.jsonl")
    out, _ = chunked_analyse(
        [path], spec, inputs(synth), ChunkOptions(tmp_dir=tmp_path, log=lambda _: None)
    )
    meta = {
        "mode": "EVAL",
        "spec_sha256": synth.spec_sha,
        "utc": "t",
        "inputs": [],
        "records_by_arm": {},
        "csv": {},
    }
    section = report_md(out, meta).split("\n## 8. Not in grid")[1].split("\n## ")[0]
    assert "| R-x | ci_native | ns [1000] | " in section
    assert "test trim" in section
    assert "unexplained gaps: none" in section
    assert out["V4_verdicts"]["C1"]["arms"]["ci_native"]["complete"]
    assert out["V11_integrity"]["checks"]["grid_gaps_ruled"]
    grid = csv_tables(out)["not_in_grid"]
    assert grid
    assert {r["arm"] for r in grid} == {"ci_native"}
    assert {r["n"] for r in grid} == {1000}


def test_eval_mode_refuses_dev_seeds(synth: Any, tmp_path: Path) -> None:
    spec_path = tmp_path / "s.json"
    spec_path.write_text(json.dumps(synth.spec), encoding="utf-8")
    merged = write_records(synth.records()[:5], tmp_path / "m.jsonl")
    with pytest.raises(SystemExit, match="EVAL spec"):
        main(["--spec", str(spec_path), "--merged", str(merged), "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="--dev-spec needs --dev"):
        main(
            [
                "--spec",
                str(spec_path),
                "--dev-spec",
                str(spec_path),
                "--merged",
                "x",
                "--out",
                str(tmp_path),
            ]
        )
    with pytest.raises(SystemExit, match="--dev needs --dev-spec"):
        main(["--dev", "--spec", str(spec_path), "--merged", "x", "--out", str(tmp_path)])


def stamped_study(synth: Any, tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text("# P\n\nFROZEN: yes (test)\n", encoding="utf-8")
    sha = file_sha256(protocol)
    spec = {
        **synth.spec,
        "protocol_sha256": sha,
        "blocks": [
            dict(b, seeds=[3_100_000 + s for s in b["seeds"]]) for b in synth.spec["blocks"]
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    spec_sha = file_sha256(spec_path)
    records = synth.records(spec)
    for r in records:
        r["run_mode"] = {"mode": "eval", "protocol_sha256": sha}
        r["integrity"] = {**r["integrity"], "protocol_sha256": sha, "spec_file_sha256": spec_sha}
    return spec, spec_path, protocol, write_records(records, tmp_path / "m.jsonl")


def test_eval_report_is_final_deterministic_and_pinnable(synth: Any, tmp_path: Path) -> None:
    _, spec_path, protocol, merged = stamped_study(synth, tmp_path)

    def request(out: str, **overrides: Any) -> ReportRequest:
        base = ReportRequest(
            spec=spec_path,
            merged=[merged],
            out_dir=tmp_path / out,
            protocol=protocol,
            freeze_commit=synth.commit,
            tmp_dir=tmp_path,
            generated_utc="2026-10-05T07:33:00Z",
        )
        return dataclasses.replace(base, **overrides)

    quiet = lambda _: None  # noqa: E731
    assert run_report(request("o1"), quiet) == 0
    md = (tmp_path / "o1" / "REPORT.md").read_text(encoding="utf-8")
    assert md.startswith("# Study A EVAL report")
    assert "integrity **FINAL**" in md
    assert "generated 2026-10-05T07:33:00Z" in md
    assert run_report(request("o2", freeze_commit="zzz"), quiet) == 2
    assert run_report(request("o3"), quiet) == 0
    for name in ("report.json", "REPORT.md"):
        assert (tmp_path / "o3" / name).read_bytes() == (tmp_path / "o1" / name).read_bytes()
    labelled = request("o4", input_labels={str(spec_path): "specs/eval/full.json"})
    assert run_report(labelled, quiet) == 0
    meta = json.loads((tmp_path / "o4" / "report.json").read_text(encoding="utf-8"))["meta"]
    assert meta["inputs"][0][0] == "specs/eval/full.json"
    protocol.write_text("# P\n\nFROZEN: no\n", encoding="utf-8")
    assert run_report(request("o5"), quiet) == 2
    eval_named = json.loads(spec_path.read_text(encoding="utf-8")) | {"name": "eval_full"}
    spec_path.write_text(json.dumps(eval_named), encoding="utf-8")
    with pytest.raises(SystemExit, match="DEV testing only"):
        run_report(request("o6", min_flag_seeds=1), quiet)

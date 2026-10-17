import gzip
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from oran_causal.trace import pointers, symbols
from oran_causal.trace.cli import Paths, build_context
from oran_causal.trace.rules import Context, Status, code_symbols, record_pointers

type Workspace = Callable[..., Path]
type MakeClaim = Callable[..., dict[str, Any]]

SHA = "c" * 64
SELF = "oran_causal.trace.rules"
RECORD = {
    "a": {"n": 28, "precision": 0.9166, "ok": True, "name": "x", "rows": [1, 2.5]},
    "plus+key": {"~0": 7},
}


def context_for(root: Path, **changes: Any) -> Context:
    return replace(build_context(Paths.under(root)), **changes)


def evidence(pointer: str | None, expect: Any = None, record: str = "setx/r.json") -> dict:
    item: dict[str, Any] = {"record": record, "sha256": SHA}
    if pointer is not None:
        item["pointer"] = pointer
    if expect is not None:
        item["expect"] = expect
    return item


@pytest.fixture
def records(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "records" / "setx"
    root.mkdir(parents=True)
    (root / "r.json").write_text(json.dumps(RECORD), encoding="utf-8")
    (root / "z.json.gz").write_bytes(gzip.compress(json.dumps(RECORD).encode()))
    return tmp_path


def f7(workspace: Workspace, new_claim: MakeClaim, records: Path, *items: dict, **claim: Any):
    root = workspace([new_claim("EMP-A", "empirical", evidence=list(items), **claim)])
    return record_pointers(
        context_for(records if False else root, records_root=records / "data" / "records")
    )


def test_f7_passes_when_pointers_resolve_and_expect_holds(
    workspace: Workspace, new_claim: MakeClaim, records: Path
) -> None:
    items = (
        evidence("/a", {"n": 28, "precision": 0.92, "ok": True}),
        evidence("/a/rows", [1, 2.5]),
        evidence("/a/name", "x"),
        evidence("/plus+key/~00", 7),
        evidence("/a/n"),
        evidence("/a", {"n": 28}, record="setx/z.json.gz"),
        {"record": "setx/r.json", "sha256": SHA, "derive": "a / b"},
    )
    assert f7(workspace, new_claim, records, *items).status is Status.PASS


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (evidence("/a", {"n": 29}), "/a/n: expected 29, found 28"),
        (evidence("/a", {"precision": 0.91}), "/a/precision: expected 0.91, found 0.9166"),
        (evidence("/a", {"ok": 1}), "/a/ok: expected 1, found True"),
        (evidence("/a", {"missing": 1}), "/a/missing: missing"),
        (evidence("/a/rows", [1, 2]), "/a/rows[1]: expected 2, found 2.5"),
        (evidence("/a/nope"), "pointer /a/nope does not resolve"),
        (evidence("/a/rows/9"), "does not resolve"),
        (evidence("/a/*", {"n": 1}), "not a plain JSON pointer"),
        (evidence("/a", record="setx/gone.json"), "setx/gone.json not found under data/records"),
    ],
)
def test_f7_reports_each_kind_of_failure(
    item: dict, message: str, workspace: Workspace, new_claim: MakeClaim, records: Path
) -> None:
    finding = f7(workspace, new_claim, records, item)
    assert finding.status is Status.FAIL
    assert message in " ".join(finding.messages)
    assert finding.messages[0].startswith("EMP-A: ")


def test_f7_is_not_checkable_without_data_records(
    workspace: Workspace, new_claim: MakeClaim, tmp_path: Path
) -> None:
    root = workspace([new_claim("EMP-A", "empirical", evidence=[evidence("/a")])])
    for missing in (None, tmp_path / "no" / "records"):
        finding = record_pointers(context_for(root, records_root=missing))
        assert finding.status is Status.NOT_CHECKABLE
        assert "data/records not found" in finding.messages[0]


def test_f7_skips_retracted_claims(
    workspace: Workspace, new_claim: MakeClaim, records: Path
) -> None:
    claim = new_claim("EMP-A", "empirical", status="retracted", evidence=[evidence("/a/nope")])
    context = context_for(workspace([claim]), records_root=records / "data" / "records")
    assert record_pointers(context).status is Status.PASS


def test_f12_passes_for_importable_symbols(workspace: Workspace, new_claim: MakeClaim) -> None:
    claim = new_claim("DEF-A", code=[f"{SELF}:Status.PASS", f"{SELF}:record_pointers"])
    assert code_symbols(context_for(workspace([claim]))).status is Status.PASS


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("oran_causal.trace.nope:X", "module oran_causal.trace.nope does not import"),
        (f"{SELF}:no_such_name", f"{SELF} has no attribute no_such_name"),
        (f"{SELF}:Status.NOPE", f"{SELF} has no attribute Status.NOPE"),
    ],
)
def test_f12_names_the_broken_symbol(
    reference: str, message: str, workspace: Workspace, new_claim: MakeClaim
) -> None:
    finding = code_symbols(context_for(workspace([new_claim("DEF-A", code=[reference])])))
    assert finding.status is Status.FAIL
    assert finding.messages[0].startswith(f"DEF-A: {reference}: {message}")


@pytest.mark.parametrize("reference", ["json", "json:", ":x"])
def test_a_reference_without_module_and_attribute_is_malformed(reference: str) -> None:
    with pytest.raises(ValueError, match="not of the form module:attribute"):
        symbols.resolve(reference)


def test_f12_skips_retracted_claims(workspace: Workspace, new_claim: MakeClaim) -> None:
    claim = new_claim("DEF-A", status="retracted", code=[f"{SELF}:no_such_name"])
    assert code_symbols(context_for(workspace([claim]))).status is Status.PASS


def test_a_module_that_raises_on_import_is_reported(tmp_path: Path) -> None:
    def boom(_: str) -> object:
        raise RuntimeError("broken")

    with pytest.raises(ValueError, match=r"RuntimeError: broken"):
        symbols.resolve("pkg:x", boom)


def test_pointer_steps_unescape_and_index_lists() -> None:
    document = {"a/b": {"c~d": [10, 20]}}
    assert pointers.follow(document, "/a~1b/c~0d/1") == 20
    assert pointers.follow(document, "") == document
    with pytest.raises(pointers.PointerError):
        pointers.follow(document, "/a~1b/c~0d/x")


def test_unreadable_records_are_pointer_errors(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    for name in ("bad.json", "a.csv"):
        with pytest.raises(pointers.PointerError):
            pointers.read_json(tmp_path / name)


@pytest.mark.parametrize(
    ("found", "expected", "ok"),
    [
        (0.92, 0.92, True),
        (0.9166, 0.92, True),
        (0.9166, 0.917, True),
        (0.9166, 0.91, False),
        (19953.6501, 19953.65, True),
        (3, 3.0, True),
        (2.5, 2, False),
        (0.125, 0.13, True),
        (0.1249, 0.13, False),
        (True, 1, False),
        (None, None, True),
        ("a", "a", True),
        ("a", "b", False),
        ({"a": 1, "b": 2}, {"a": 1}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
    ],
)
def test_expect_comparison(found: Any, expected: Any, ok: bool) -> None:
    assert (pointers.mismatches(found, expected) == []) is ok

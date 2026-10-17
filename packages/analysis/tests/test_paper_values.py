import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from oran_causal.analysis import paper_values as pv

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("value", "fmt", "text"),
    [
        (0.625, ".2f|up", "0.63"),
        (0.875, ".2f|up", "0.88"),
        (0.625, ".2f", "0.62"),
        (0.0654, "+.1f|x100", "+6.5"),
        (-0.029, "+.1f|x100|math", "$-2.9$"),
        (9999, ",d|thousands", r"9\,999"),
        (28, "d", "28"),
        (0.00079, ".1e|sci", r"7.9 \times 10^{-4}"),
    ],
)
def test_format_value(value: float, fmt: str, text: str) -> None:
    assert pv.format_value(value, fmt) == text


def test_up_filter_needs_a_fixed_spec() -> None:
    with pytest.raises(pv.SpecError):
        pv.format_value(1.5, "d|up")


def test_unknown_filter_is_rejected() -> None:
    with pytest.raises(pv.SpecError):
        pv.format_value(1.5, ".1f|sparkle")


def test_evaluate_arithmetic_and_functions() -> None:
    assert pv.evaluate("round(r * n * (1 - s))", {"r": 0.5, "n": 10, "s": 0.8}) == 1
    assert pv.evaluate("-abs(a) + max(a, b)", {"a": -2, "b": 5}) == 3
    assert pv.evaluate("dot(a, b)", {"a": [1, 2], "b": [3, 4]}) == 11


def test_evaluate_rejects_unsupported_syntax() -> None:
    with pytest.raises(pv.SpecError):
        pv.evaluate("__import__('os')", {})


ROWS = {"rows": [{"arm": "a", "n": 1}, {"arm": "b", "n": 2}], "k": {"x/y": 7}}


@pytest.mark.parametrize(
    ("pointer", "expected"),
    [
        ("/rows/1/n", 2),
        ("/rows/arm=b/n", 2),
        ("/rows/[*]/n", [1, 2]),
        ("/rows/[arm=a]/n", [1]),
        ("/k/x~1y", 7),
    ],
)
def test_resolve_pointer(pointer: str, expected: object) -> None:
    assert pv.resolve_pointer(ROWS, pointer) == expected


@pytest.mark.parametrize("pointer", ["/rows/arm=c/n", "/missing", "/rows/9"])
def test_resolve_pointer_failures_name_the_step(pointer: str) -> None:
    with pytest.raises(pv.SpecError, match="pointer"):
        pv.resolve_pointer(ROWS, pointer)


def make_workspace(root: Path, record: Mapping[str, object], toml: str) -> None:
    data = json.dumps(record).encode()
    target = root / "data" / "records" / "set_a" / "r.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    manifests = root / "data" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "set_a.sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  r.json\n", encoding="utf-8"
    )
    (root / "spec").mkdir()
    (root / "spec" / "values.toml").write_text(toml, encoding="utf-8")


TOML = """
[records]
a = "set_a/r.json"

[[values]]
key = "t.share"
from = "a:/hits/rate"
fmt = ".2f|up"

[[values]]
key = "t.ratio"
fmt = ".1f"
expr = "x / y"

[values.args]
x = "a:/hits/n"
y = "a:/misses/n"
"""


def test_generate_is_deterministic_and_sorted(tmp_path: Path) -> None:
    record = {"hits": {"rate": 0.625, "n": 5}, "misses": {"n": 2}}
    make_workspace(tmp_path, record, TOML)
    tex, sidecar = pv.generate(tmp_path)
    assert pv.generate(tmp_path) == (tex, sidecar)
    assert tex.splitlines()[2:] == [r"\defval{t.ratio}{2.5}", r"\defval{t.share}{0.63}"]
    entry = json.loads(sidecar)["values"]["t.share"]
    assert entry["args"]["v"]["pointer"] == "/hits/rate"
    assert entry["args"]["v"]["value"] == 0.625
    assert len(entry["args"]["v"]["sha256"]) == 64


def test_generate_refuses_a_record_that_differs_from_its_manifest(tmp_path: Path) -> None:
    make_workspace(tmp_path, {"hits": {"rate": 0.5, "n": 1}, "misses": {"n": 1}}, TOML)
    (tmp_path / "data" / "records" / "set_a" / "r.json").write_text("{}", encoding="utf-8")
    with pytest.raises(pv.SpecError, match="differs from manifest"):
        pv.generate(tmp_path)


def test_main_writes_then_check_passes_and_detects_staleness(tmp_path: Path) -> None:
    make_workspace(tmp_path, {"hits": {"rate": 0.5, "n": 1}, "misses": {"n": 1}}, TOML)
    assert pv.main(["--root", str(tmp_path), "--check"]) == 1
    assert pv.main(["--root", str(tmp_path)]) == 0
    assert pv.main(["--root", str(tmp_path), "--check"]) == 0
    values = tmp_path / pv.VALUES_TEX
    values.write_text(values.read_text(encoding="utf-8") + "% edited\n", encoding="utf-8")
    assert pv.main(["--root", str(tmp_path), "--check"]) == 1


@pytest.mark.parametrize(
    "toml",
    [
        TOML.replace("t.share", "Bad Key"),
        TOML.replace('from = "a:/hits/rate"', 'from = "zz:/hits/rate"'),
        TOML + '\n[[values]]\nkey = "t.share"\nfrom = "a:/hits/rate"\nfmt = "d"\n',
    ],
)
def test_load_spec_rejects_malformed_entries(tmp_path: Path, toml: str) -> None:
    path = tmp_path / "values.toml"
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(pv.SpecError):
        pv.load_spec(path)


def test_manuscript_claims_style_is_a_copy_of_the_spec() -> None:
    spec = (REPO / "spec" / "claims.sty").read_bytes()
    assert (REPO / "paper" / "claims.sty").read_bytes() == spec


def test_committed_values_cover_the_spec_keys() -> None:
    keys = {spec.key for spec in pv.load_spec(REPO / pv.SPEC_PATH)}
    sidecar = json.loads((REPO / pv.VALUES_JSON).read_text(encoding="utf-8"))["values"]
    tex = (REPO / pv.VALUES_TEX).read_text(encoding="utf-8")
    assert keys == set(sidecar)
    assert all(rf"\defval{{{key}}}{{" in tex for key in keys)


@pytest.mark.skipif(
    not (REPO / "data" / "records" / "system_artifacts").is_dir(), reason="frozen records absent"
)
def test_regeneration_from_the_frozen_records_is_byte_identical() -> None:
    tex, sidecar = pv.generate(REPO)
    assert (REPO / pv.VALUES_TEX).read_bytes().decode("utf-8") == tex
    assert (REPO / pv.VALUES_JSON).read_bytes().decode("utf-8") == sidecar

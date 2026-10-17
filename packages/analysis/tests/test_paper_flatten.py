import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "paper"


def load_flatten() -> ModuleType:
    spec = importlib.util.spec_from_file_location("paper_flatten", PAPER / "tools" / "flatten.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flatten = load_flatten()


def test_strip_macros_expands_values_and_drops_claim_tags() -> None:
    values = {"a.b": "0.92", "c.d": "$-2.9$"}
    line = r"rate \val{a.b} and \val{c.d}.\claim{EMP-A}\claim{EMP-B} Next."
    assert flatten.strip_macros(line, values) == "rate 0.92 and $-2.9$. Next."


def test_strip_macros_rejects_an_unknown_key() -> None:
    with pytest.raises(KeyError):
        flatten.strip_macros(r"\val{no.such}", {})


def test_load_values_reads_defval_lines(tmp_path: Path) -> None:
    path = tmp_path / "values.tex"
    path.write_text("% header\n\\defval{a.b}{1\\,000}\n\\defval{c.d}{$x$}\n", encoding="utf-8")
    assert flatten.load_values(path) == {"a.b": "1\\,000", "c.d": "$x$"}


def test_expand_inlines_inputs_and_removes_the_claims_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text(
        "\\usepackage{claims}%\n\\showclaimsfalse\n\\input{generated/values}\n"
        "\\input{sections/a}\n% comment\ntext % trailing\n",
        encoding="utf-8",
    )
    (tmp_path / "sections" / "a.tex").write_text(
        "\\label{x}\\claim{EMP-A}\nvalue \\val{k.v}\n", encoding="utf-8"
    )
    monkeypatch.setattr(flatten, "ROOT", tmp_path)
    assert flatten.expand(tmp_path / "main.tex", {"k.v": "7"}) == [
        "\\label{x}",
        "value 7",
        "",
        "text %",
        "",
    ]


def test_the_manuscript_flattens_without_custom_macros() -> None:
    values = flatten.load_values(PAPER / "generated" / "values.tex")
    body = "\n".join(flatten.expand(PAPER / "sn-article.tex", values))
    for macro in (r"\val{", r"\claim{", r"\defval{", r"\showclaims", r"\usepackage{claims}"):
        assert macro not in body
    assert r"\input{" not in body

import pytest

from oran_causal.figures import paths


def test_record_resolves_below_records(tmp_path, monkeypatch):
    (tmp_path / "set").mkdir()
    (tmp_path / "set" / "a.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "RECORDS", tmp_path)
    assert paths.record("set", "a.json") == tmp_path / "set" / "a.json"


def test_record_refuses_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RECORDS", tmp_path)
    with pytest.raises(FileNotFoundError, match="record missing"):
        paths.record("set", "a.json")


@pytest.mark.parametrize(
    "parts",
    [
        ("study2", "E6P_CONFOUNDED_PROTOCOL.md"),
        ("study3", "E6P_CERTSAFE_PROTOCOL.md"),
        ("study4", "PROTOCOL_A.md"),
    ],
)
def test_protocols_are_the_frozen_copies(parts):
    assert paths.protocol(*parts).is_file()


def test_protocol_refuses_a_missing_file():
    with pytest.raises(FileNotFoundError, match="protocol missing"):
        paths.protocol("study9", "none.md")


def test_outputs_default_to_the_journal(monkeypatch):
    monkeypatch.delenv(paths.OUT_ENV, raising=False)
    assert paths.figures_dir() == paths.JOURNAL / "figures"
    assert paths.tables_dir() == paths.JOURNAL / "tables"
    assert paths.preview_dir() == paths.PREVIEW


def test_outputs_follow_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.OUT_ENV, str(tmp_path))
    assert paths.figures_dir() == tmp_path / "figures"
    assert paths.tables_dir() == tmp_path / "tables"
    assert paths.preview_dir() == tmp_path / "preview"

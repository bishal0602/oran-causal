import pytest

from oran_causal.figures import paths


def test_record_resolves_below_records(tmp_path, monkeypatch):
    (tmp_path / "set").mkdir()
    (tmp_path / "set" / "a.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "RECORDS", tmp_path)
    assert paths.record("set", "a.json") == tmp_path / "set" / "a.json"


def test_record_reads_the_overlay_copy_of_a_shap_file(tmp_path, monkeypatch):
    rel = "system_artifacts/decision/cs_eval.json"
    for d in ("system_artifacts/decision", "shap_rerun/overlay/system_artifacts/decision"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "cs_eval.json").write_text("{}", encoding="utf-8")
        (tmp_path / d / "other.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "RECORDS", tmp_path)
    assert paths.record(*rel.split("/")) == tmp_path / "shap_rerun" / "overlay" / rel
    assert paths.record("system_artifacts", "decision", "other.json") == (
        tmp_path / "system_artifacts" / "decision" / "other.json"
    )


def test_record_refuses_a_shap_file_without_its_overlay_copy(tmp_path, monkeypatch):
    (tmp_path / "system_artifacts" / "decision").mkdir(parents=True)
    (tmp_path / "system_artifacts" / "decision" / "cs_eval.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "RECORDS", tmp_path)
    with pytest.raises(FileNotFoundError, match="shap_rerun"):
        paths.record("system_artifacts", "decision", "cs_eval.json")


def test_record_refuses_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RECORDS", tmp_path)
    with pytest.raises(FileNotFoundError, match="record missing"):
        paths.record("set", "a.json")


@pytest.mark.parametrize(
    "parts",
    [
        ("study2", "PROTOCOL.md"),
        ("study3", "PROTOCOL.md"),
        ("study4", "PROTOCOL.md"),
    ],
)
def test_protocols_are_the_frozen_copies(parts):
    assert paths.protocol(*parts).is_file()


def test_protocol_refuses_a_missing_file():
    with pytest.raises(FileNotFoundError, match="protocol missing"):
        paths.protocol("study9", "none.md")


def test_outputs_default_to_the_journal(monkeypatch):
    monkeypatch.delenv(paths.OUT_ENV, raising=False)
    assert paths.figures_dir() == paths.PAPER / "figures"
    assert paths.tables_dir() == paths.PAPER / "tables"
    assert paths.preview_dir() == paths.PREVIEW


def test_outputs_follow_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.OUT_ENV, str(tmp_path))
    assert paths.figures_dir() == tmp_path / "figures"
    assert paths.tables_dir() == tmp_path / "tables"
    assert paths.preview_dir() == tmp_path / "preview"

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_claim(claim_id: str, kind: str = "definition", **extra: Any) -> dict[str, Any]:
    return {
        "id": claim_id,
        "kind": kind,
        "statement": f"Statement of {claim_id}.",
        "paper": ["sec:x"],
        "spec": "spec:a",
        **extra,
    }


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def schema_path() -> Path:
    return REPO_ROOT / "spec" / "claims.schema.yaml"


@pytest.fixture
def new_claim() -> Callable[..., dict[str, Any]]:
    return make_claim


@pytest.fixture
def workspace(tmp_path: Path, schema_path: Path) -> Callable[..., Path]:
    def build(claims: list[dict[str, Any]], labels: tuple[str, ...] = ("spec:a",)) -> Path:
        spec = tmp_path / "spec"
        (spec / "sections").mkdir(parents=True)
        shutil.copy(schema_path, spec / "claims.schema.yaml")
        document = {"version": 1, "claims": claims}
        (spec / "claims.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        anchors = "\n".join(f"\\label{{{label}}}" for label in labels)
        (spec / "sections" / "01-a.tex").write_text(anchors + "\n", encoding="utf-8")
        return tmp_path

    return build

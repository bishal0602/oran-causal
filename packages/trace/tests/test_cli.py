from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from oran_causal.trace.cli import find_root, main

type Workspace = Callable[..., Path]
type MakeClaim = Callable[..., dict[str, Any]]


def test_check_is_non_blocking_without_strict(
    workspace: Workspace, new_claim: MakeClaim, capsys: pytest.CaptureFixture[str]
) -> None:
    root = workspace([new_claim("DEF-A"), new_claim("DEF-A")])
    assert main(["--root", str(root), "check"]) == 0
    output = capsys.readouterr().out
    assert "F1   warn" in output
    assert "(non-strict)" in output


def test_check_strict_fails_on_any_failure(workspace: Workspace, new_claim: MakeClaim) -> None:
    root = workspace([new_claim("DEF-A"), new_claim("DEF-A")])
    assert main(["--root", str(root), "check", "--strict"]) == 1


def test_matrix_then_strict_check_passes(
    workspace: Workspace, new_claim: MakeClaim, capsys: pytest.CaptureFixture[str]
) -> None:
    root = workspace([new_claim("DEF-A")])
    assert main(["--root", str(root), "matrix"]) == 0
    assert (root / "docs" / "TRACEABILITY.md").is_file()
    assert main(["--root", str(root), "check", "--strict"]) == 0
    assert "F13  pass" in capsys.readouterr().out


def test_long_message_lists_are_truncated(
    workspace: Workspace, new_claim: MakeClaim, capsys: pytest.CaptureFixture[str]
) -> None:
    claims = [new_claim(f"DEF-A{n}", depends_on=["DEF-X"]) for n in range(10)]
    main(["--root", str(workspace(claims)), "check"])
    assert "... 2 more (use --verbose)" in capsys.readouterr().out


def test_find_root_walks_up(workspace: Workspace, new_claim: MakeClaim) -> None:
    root = workspace([new_claim("DEF-A")])
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    assert find_root(nested) == root


def test_find_root_falls_back_to_start(tmp_path: Path) -> None:
    assert find_root(tmp_path) == tmp_path

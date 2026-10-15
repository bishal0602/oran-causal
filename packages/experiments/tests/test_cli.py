from collections.abc import Mapping, Sequence

import pytest

from oran_causal.experiments import cli


def recording_runner(
    codes: Mapping[tuple[str, ...], int], calls: list[tuple[str, ...]]
) -> cli.Runner:
    def runner(args: Sequence[str]) -> int:
        key = tuple(args)
        calls.append(key)
        return codes.get(key, 0)

    return runner


def test_checks_run_in_declared_order() -> None:
    calls: list[tuple[str, ...]] = []
    assert cli.run_checks(recording_runner({}, calls)) == 0
    assert calls == list(cli.CHECKS)


def test_first_failure_stops_the_run() -> None:
    calls: list[tuple[str, ...]] = []
    codes: dict[tuple[str, ...], int] = {("ruff", "format", "--check"): 3}
    assert cli.run_checks(recording_runner(codes, calls)) == 3
    assert calls == [("ruff", "check"), ("ruff", "format", "--check")]


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["unknown"])
    assert exit_info.value.code == 2


def test_check_command_dispatches_to_run_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_checks", lambda runner: 7)
    assert cli.main(["check"]) == 7

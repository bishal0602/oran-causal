from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from oran_causal.experiments import cli
from oran_causal.experiments.reproduce import t0, t1
from oran_causal.experiments.reproduce.checks import Check, Status

PARSED_HELP = (
    [],
    ["check"],
    ["data"],
    ["data", "fetch"],
    ["paper-values"],
    ["reproduce"],
    ["verify"],
    ["run"],
    ["run", "system"],
)
DELEGATED_HELP = (
    ["campaign"],
    ["campaign", "list"],
    ["campaign", "run"],
    ["campaign", "merge"],
    ["bridging"],
    ["bridging", "run"],
    ["launch"],
    ["colab"],
    ["kaggle"],
    ["vps"],
    ["dispatch"],
    ["fingerprint"],
)


def recording_runner(
    codes: Mapping[tuple[str, ...], int], calls: list[tuple[str, ...]]
) -> cli.Runner:
    def runner(args: Sequence[str]) -> int:
        key = tuple(args)
        calls.append(key)
        return codes.get(key, 0)

    return runner


def collect() -> tuple[list[str], cli.Out]:
    lines: list[str] = []
    return lines, lines.append


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
    seen: list[object] = []
    monkeypatch.setattr(cli, "run_checks", lambda runner, checks: seen.append(checks) or 7)
    assert cli.main(["check", "--jobs", "2"]) == 7
    assert seen == [cli.with_jobs(cli.CHECKS, 2)]


def test_jobs_only_reaches_the_pytest_check() -> None:
    assert cli.with_jobs(cli.CHECKS, None) == cli.CHECKS
    jobbed = cli.with_jobs(cli.CHECKS, 2)
    assert ("pytest", "-n", "2") in jobbed
    assert [c for c in jobbed if c[0] != "pytest"] == [c for c in cli.CHECKS if c[0] != "pytest"]


@pytest.mark.parametrize("args", PARSED_HELP + DELEGATED_HELP, ids=" ".join)
def test_every_command_has_help(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*args, "--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_top_level_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    for name in ("check", "data", "reproduce", "verify", "run", *cli.DELEGATES):
        assert name in text


@pytest.mark.parametrize("args", [["posthoc"], ["posthoc", "--help"]])
def test_posthoc_help_lists_the_items(args: list[str]) -> None:
    lines, out = collect()
    assert cli.main(args, out=out) == 0
    assert "x16" in lines[0]


@pytest.mark.parametrize(
    ("args", "module", "forwarded"),
    [
        (["campaign", "list", "--spec", "s.json"], "campaign", ["list", "--spec", "s.json"]),
        (["bridging", "costs"], "bridging", ["costs"]),
        (["launch", "--platform", "vps"], "launch", ["launch", "--platform", "vps"]),
        (["dispatch", "tick", "q"], "launch", ["dispatch", "tick", "q"]),
        (["fingerprint"], "launch", ["fingerprint"]),
    ],
)
def test_delegates_forward_their_arguments(
    monkeypatch: pytest.MonkeyPatch, args: list[str], module: str, forwarded: list[str]
) -> None:
    seen: list[list[str]] = []

    def fake(argv: Sequence[str], **_: Any) -> int:
        seen.append(list(argv))
        return 5

    monkeypatch.setattr(f"oran_causal.experiments.{module}.__main__.main", fake)
    assert cli.main(args) == 5
    assert seen == [forwarded]


def test_posthoc_item_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    from oran_causal.experiments.posthoc import __main__ as posthoc

    seen: list[list[str]] = []
    monkeypatch.setattr(posthoc, "main", lambda argv: seen.append(list(argv)) or 0)
    assert cli.main(["posthoc", "x9", "--out", "o"]) == 0
    assert seen == [["x9", "--out", "o"]]


def test_run_system_builds_the_stage_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from oran_causal.experiments.system import stages

    calls: list[tuple[Any, ...]] = []

    def fake(study: int, stage: str, part: Any, out: Path, inputs: Any, **opts: Any) -> int:
        calls.append((study, stage, part, out, inputs, opts))
        return 2

    monkeypatch.setattr(stages, "run_stage", fake)
    lines, out = collect()
    args = ["run", "system", "--study", "3", "--stage", "eval", "--part", "0/4"]
    args += ["--out", str(tmp_path / "x.jsonl"), "--smoke", "--short", "120", "--arms", "A,B"]
    args += ["--input", "eval=a.jsonl", "--input", "eval=b.jsonl", "--input", "gt=g.jsonl"]
    assert cli.main(args, out=out) == 0
    inputs = {"eval": ["a.jsonl", "b.jsonl"], "gt": ["g.jsonl"]}
    opts = {"smoke": True, "short": 120, "arms": ["A", "B"]}
    assert calls == [(3, "eval", "0/4", tmp_path / "x.jsonl", inputs, opts)]
    assert lines == ["study 3 eval: 2"]


def test_run_system_rejects_a_malformed_input() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "system", "--study", "1", "--stage", "eval", "--input", "eval"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize("tier", ["t2", "t3"])
def test_plan_tiers_only_dry_run(tier: str) -> None:
    lines, out = collect()
    assert cli.main(["reproduce", "--tier", tier], out=out) == 2
    assert "--dry-run" in lines[0]


def test_plan_dry_run_prints_the_platform_table() -> None:
    lines, out = collect()
    assert (
        cli.main(["reproduce", "--tier", "t2", "--dry-run", "--platform", "kaggle"], out=out) == 0
    )
    assert lines[0] == "# Tier t2 dry run (reference CPU-h; no launch)"
    assert "| all | 161.30 | | | | |" in lines


@pytest.mark.parametrize("tier", ["t0", "t1"])
def test_unknown_item_is_refused(tier: str, tmp_path: Path) -> None:
    lines, out = collect()
    args = ["reproduce", "--tier", tier, "--item", "nope", "--out", str(tmp_path)]
    assert cli.main(args, out=out) == 2
    assert "nope" in lines[0]


@pytest.mark.parametrize(("status", "code"), [(Status.PASS, 0), (Status.DIFF, 1)])
def test_t0_writes_its_result_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: Status, code: int
) -> None:
    seen: list[Sequence[str]] = []

    def fake_run(ctx: Any, items: Sequence[str]) -> list[Check]:
        seen.append(items)
        return [Check("system", "a.json", status, "x")]

    monkeypatch.setattr(t0, "run", fake_run)
    _lines, out = collect()
    args = ["reproduce", "--tier", "t0", "--item", "system", "--out", str(tmp_path)]
    assert cli.main([*args, "--records", str(tmp_path)], out=out) == code
    assert seen == [["system"]]
    text = (tmp_path / "t0_results.md").read_text(encoding="utf-8")
    assert f"| system | a.json | {status} | x |" in text


def test_t1_runs_the_chosen_steps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def fake_run(ctx: Any, steps: Mapping[str, Any]) -> list[Check]:
        seen.append(list(steps))
        return [Check("t1", "wall time", Status.PASS)]

    monkeypatch.setattr(t1, "run", fake_run)
    args = ["reproduce", "--tier", "t1", "--item", "host", "--item", "pmrt"]
    assert cli.main([*args, "--out", str(tmp_path)], out=lambda _: None) == 0
    assert seen == [["host", "pmrt"]]
    assert (tmp_path / "t1_results.md").exists()


def test_verify_passes_the_frozen_and_rerun_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from oran_causal.experiments.reproduce import verify

    seen: list[tuple[str, Path, Path]] = []

    def fake(study: str, records: Path, ctx: Any) -> list[Check]:
        seen.append((study, records, ctx.records))
        return [Check(study, "verdict", Status.PASS)]

    monkeypatch.setattr(verify, "verify", fake)
    new, frozen = tmp_path / "new", tmp_path / "frozen"
    args = ["verify", "study3", "--records", str(new), "--frozen", str(frozen)]
    assert cli.main([*args, "--out", str(tmp_path)], out=lambda _: None) == 0
    assert seen == [("study3", new.resolve(), frozen.resolve())]
    assert (tmp_path / "verify_study3.md").exists()


def test_data_fetch_reports_each_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from oran_causal.experiments.reproduce import data

    def fake(names: Any, root: Path, *, verify_only: bool, base_url: str) -> list[Check]:
        return [Check("data", f"{n}:{verify_only}", Status.PASS) for n in names]

    monkeypatch.setattr(data, "fetch", fake)
    lines, out = collect()
    args = ["data", "fetch", "--set", "freeze", "--verify-only", "--records", str(tmp_path)]
    assert cli.main(args, out=out) == 0
    assert "| data | freeze:True | pass |  |" in lines[0]


def test_data_pack_passes_its_flags_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from oran_causal.experiments.reproduce import data

    seen: list[tuple[Any, Path, Path]] = []

    def fake(names: Any, out: Path, root: Path) -> list[Check]:
        seen.append((names, out, root))
        return [Check("data", "freeze", Status.PASS)]

    monkeypatch.setattr(data, "pack", fake)
    args = [
        "data",
        "pack",
        "--set",
        "freeze",
        "--out",
        str(tmp_path / "o"),
        "--records",
        str(tmp_path),
    ]
    assert cli.main(args, out=lambda _: None) == 0
    assert seen == [(["freeze"], (tmp_path / "o").resolve(), tmp_path.resolve())]


def test_paper_values_command_passes_its_flags_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from oran_causal.analysis import paper_values

    seen: list[list[str]] = []
    monkeypatch.setattr(paper_values, "main", lambda argv: seen.append(list(argv)) or 0)
    assert cli.main(["paper-values"]) == 0
    assert cli.main(["paper-values", "--check", "--root", "somewhere"]) == 0
    assert seen == [[], ["--check", "--root", "somewhere"]]


def test_check_runs_the_paper_values_step_before_pytest() -> None:
    assert cli.PAPER_VALUES_CHECK in cli.CHECKS
    assert cli.CHECKS.index(cli.PAPER_VALUES_CHECK) < cli.CHECKS.index(("pytest",))


def test_check_ends_with_the_strict_trace_check() -> None:
    assert cli.CHECKS[-1] == cli.TRACE_CHECK
    assert "--strict" in cli.TRACE_CHECK


def test_paper_values_check_is_skipped_without_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "run_module", lambda args: 9)
    assert cli.run_check(cli.PAPER_VALUES_CHECK, root=tmp_path) == 0
    assert "skipped" in capsys.readouterr().out
    (tmp_path / "data" / "records").mkdir(parents=True)
    assert cli.run_check(cli.PAPER_VALUES_CHECK, root=tmp_path) == 9
    assert cli.run_check(("ruff", "check"), root=tmp_path / "absent") == 9

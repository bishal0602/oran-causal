"""Command line entry point for the oran-causal workspace.

`check`, `data verify`, `paper-values`, `reproduce`, `verify` and `run system` are parsed here.
`campaign`, `bridging`, `posthoc` and the launcher commands (`launch`, `colab`, `kaggle`, `vps`,
`dispatch`, `fingerprint`) hand their arguments to the command line of their package.
"""

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from oran_causal.experiments.reproduce.checks import Check, passed, table

Runner = Callable[[Sequence[str]], int]
Out = Callable[[str], None]

PAPER_VALUES_CHECK = ("oran_causal.analysis.paper_values", "--check")
TRACE_CHECK = ("oran_causal.trace.cli", "check", "--strict")
CHECKS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check"),
    ("ruff", "format", "--check"),
    ("ty", "check"),
    PAPER_VALUES_CHECK,
    ("pytest",),
    TRACE_CHECK,
)
TIERS = ("t0", "t1", "t2", "t3")
STUDIES = ("study1", "study2", "study3", "study4")
DEFAULT_OUT = Path("build") / "reproduce"
LAUNCH_COMMANDS = ("launch", "colab", "kaggle", "vps", "dispatch", "fingerprint")
DELEGATES: dict[str, str] = {
    "campaign": "Study 4 campaign runner: list, run, merge",
    "bridging": "bridging analysis (Appendix B): prep, counts, list, run, merge, table, costs",
    "posthoc": "post hoc analyses and drivers X1-X16",
    "launch": "plan and start a session on Kaggle, Colab or the VPS",
    "colab": "Colab session commands",
    "kaggle": "Kaggle kernel commands",
    "vps": "VPS session commands",
    "dispatch": "unattended queue across platforms",
    "fingerprint": "numeric fingerprint of this host",
}


def run_module(args: Sequence[str]) -> int:
    """Runs `python -m <args>` with the current interpreter.

    Args:
        args: Module name followed by its arguments.

    Returns:
        The process exit code.
    """
    return subprocess.run([sys.executable, "-m", *args], check=False).returncode


def run_check(args: Sequence[str], root: Path | None = None) -> int:
    """Runs one check; the paper values check is skipped when `data/records` is absent.

    Args:
        args: Module name followed by its arguments.
        root: Workspace root; defaults to this checkout.

    Returns:
        The process exit code.
    """
    from oran_causal.analysis.paper_values import WORKSPACE_ROOT

    base = root or WORKSPACE_ROOT
    if tuple(args) == PAPER_VALUES_CHECK and not (base / "data" / "records").is_dir():
        print("paper_values --check skipped: data/records not found")
        return 0
    return run_module(args)


def with_jobs(checks: Sequence[Sequence[str]], jobs: int | None) -> tuple[tuple[str, ...], ...]:
    """Adds `-n <jobs>` to the pytest check; other checks and `jobs=None` are unchanged.

    Args:
        checks: Ordered module invocations.
        jobs: pytest-xdist worker count, or None to keep pytest's own default.

    Returns:
        The invocations with the worker option applied.
    """
    return tuple(
        (*c, "-n", str(jobs)) if jobs is not None and tuple(c) == ("pytest",) else tuple(c)
        for c in checks
    )


def run_checks(runner: Runner, checks: Sequence[Sequence[str]] = CHECKS) -> int:
    """Runs each check in order and stops at the first failure.

    Args:
        runner: Executes one check and returns its exit code.
        checks: Ordered module invocations.

    Returns:
        0 when every check passes, otherwise the exit code of the first failure.
    """
    for check in checks:
        code = runner(check)
        if code != 0:
            return code
    return 0


def _input(text: str) -> tuple[str, str]:
    key, sep, path = text.partition("=")
    if not sep or not key or not path:
        raise argparse.ArgumentTypeError(f"--input {text!r}: expected KEY=PATH")
    return key, path


def _add_data(commands: argparse._SubParsersAction) -> None:
    data = commands.add_parser("data", help="record sets: verify against data/manifests")
    sub = data.add_subparsers(dest="data_command", required=True)
    verify = sub.add_parser("verify", help="hash the present sets against their manifests")
    verify.add_argument("--set", dest="sets", action="append", help="record set (repeatable)")
    verify.add_argument("--records", type=Path, default=None, help="records directory")


def _add_paper_values(commands: argparse._SubParsersAction) -> None:
    values = commands.add_parser(
        "paper-values", help="regenerate paper/generated/values.{tex,json} from spec/values.toml"
    )
    values.add_argument("--check", action="store_true", help="exit 1 when the files are stale")
    values.add_argument("--root", type=Path, default=None, help="workspace root")


def _add_reproduce(commands: argparse._SubParsersAction) -> None:
    rep = commands.add_parser("reproduce", help="reproduce tiers t0 to t3")
    rep.add_argument("--tier", required=True, choices=TIERS)
    rep.add_argument("--item", action="append", help="t0 item or t1 step (repeatable)")
    rep.add_argument("--records", type=Path, default=None, help="frozen records directory")
    rep.add_argument("--out", type=Path, default=DEFAULT_OUT, help="scratch output directory")
    rep.add_argument("--platform", choices=("kaggle", "colab", "vps"), help="t2/t3: one lane")
    rep.add_argument("--dry-run", action="store_true", help="t2/t3: print the plan and cost")


def _add_verify(commands: argparse._SubParsersAction) -> None:
    ver = commands.add_parser("verify", help="same-conclusion rules on a re-run's records")
    ver.add_argument("study", choices=STUDIES)
    ver.add_argument("--records", required=True, type=Path, help="re-run records directory")
    ver.add_argument("--frozen", type=Path, default=None, help="frozen records directory")
    ver.add_argument("--out", type=Path, default=DEFAULT_OUT, help="scratch output directory")


def _add_run(commands: argparse._SubParsersAction) -> None:
    run = commands.add_parser("run", help="run one stage of an experiment")
    sub = run.add_subparsers(dest="run_command", required=True)
    system = sub.add_parser("system", help="one stage of Study 1, 2 or 3")
    system.add_argument("--study", required=True, type=int, choices=(1, 2, 3))
    system.add_argument("--stage", required=True)
    system.add_argument("--part", default=None, help="episode stages: shard i/k")
    system.add_argument("--out", type=Path, default=Path("out"))
    system.add_argument(
        "--input", dest="inputs", action="append", type=_input, default=[], help="KEY=PATH"
    )
    system.add_argument("--smoke", action="store_true")
    system.add_argument("--short", type=int, default=None, help="episode length in s")
    system.add_argument("--max-labels", type=int, default=None)
    system.add_argument("--arms", default=None, help="comma-separated arms")


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(prog="oran-causal", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="ruff, ty, paper values, pytest, then the strict trace check"
    )
    check.add_argument("-n", "--jobs", type=int, default=None, help="pytest-xdist workers")
    _add_data(commands)
    _add_paper_values(commands)
    _add_reproduce(commands)
    _add_verify(commands)
    _add_run(commands)
    for name, text in DELEGATES.items():
        commands.add_parser(name, help=text, add_help=False)
    return parser


def _delegate(argv: Sequence[str], out: Out) -> int:
    name, rest = argv[0], list(argv[1:])
    if name in LAUNCH_COMMANDS:
        from oran_causal.experiments.launch.__main__ import main as launch_main

        return launch_main([name, *rest], out=out)
    if name == "campaign":
        from oran_causal.experiments.campaign.__main__ import main as campaign_main

        return campaign_main(rest, out=out)
    if name == "bridging":
        from oran_causal.experiments.bridging.__main__ import main as bridging_main

        return bridging_main(rest, out=out)
    from oran_causal.experiments.posthoc import __main__ as posthoc

    if not rest or rest[0] in ("-h", "--help"):
        out(posthoc.__doc__ or "")
        return 0
    return posthoc.main(rest)


def _records(path: Path | None) -> Path:
    from oran_causal.experiments.reproduce.data import RECORDS

    return (path or RECORDS).resolve()


def _data(a: argparse.Namespace, out: Out) -> int:
    from oran_causal.experiments.reproduce import data

    rows = data.verify(a.sets, _records(a.records))
    out(table(rows))
    return 0 if passed(rows) else 1


def _paper_values(a: argparse.Namespace) -> int:
    from oran_causal.analysis import paper_values

    argv = ["--check"] if a.check else []
    if a.root is not None:
        argv += ["--root", str(a.root)]
    return paper_values.main(argv)


def _report(rows: Sequence[Check], path: Path, out: Out) -> int:
    text = table(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    out(text)
    out(f"wrote {path}")
    return 0 if passed(rows) else 1


def _plan(a: argparse.Namespace, out: Out) -> int:
    from oran_causal.experiments.reproduce import plan

    if not a.dry_run:
        out(f"tier {a.tier} launches sessions: run with --dry-run, then `oran-causal launch`")
        return 2
    for line in plan.render(plan.plan(a.tier, platform=a.platform)):
        out(line)
    return 0


def _reproduce(a: argparse.Namespace, out: Out) -> int:
    if a.tier in ("t2", "t3"):
        return _plan(a, out)
    from oran_causal.experiments.reproduce import t0, t1
    from oran_causal.experiments.reproduce.context import Context

    registry = t0.ITEMS if a.tier == "t0" else t1.STEPS
    unknown = sorted(set(a.item or ()) - set(registry))
    if unknown:
        out(f"unknown {a.tier} items {unknown}; known: {', '.join(registry)}")
        return 2
    ctx = Context(_records(a.records), a.out.resolve(), log=out)
    if a.tier == "t0":
        rows = t0.run(ctx, a.item or tuple(t0.ITEMS))
    else:
        rows = t1.run(ctx, {k: t1.STEPS[k] for k in a.item or t1.STEPS})
    return _report(rows, ctx.out / f"{a.tier}_results.md", out)


def _verify(a: argparse.Namespace, out: Out) -> int:
    from oran_causal.experiments.reproduce import verify
    from oran_causal.experiments.reproduce.context import Context

    ctx = Context(_records(a.frozen), a.out.resolve(), log=out)
    rows = verify.verify(a.study, a.records.resolve(), ctx)
    return _report(rows, ctx.out / f"verify_{a.study}.md", out)


def _run_system(a: argparse.Namespace, out: Out) -> int:
    from oran_causal.experiments.system import stages

    inputs: dict[str, list[str]] = {}
    for key, path in a.inputs:
        inputs.setdefault(key, []).append(path)
    opts = {
        k: v
        for k, v in (
            ("smoke", a.smoke or None),
            ("short", a.short),
            ("max_labels", a.max_labels),
            ("arms", a.arms.split(",") if a.arms else None),
        )
        if v is not None
    }
    result = stages.run_stage(a.study, a.stage, a.part, a.out, inputs or None, **opts)
    out(f"study {a.study} {a.stage}: {result}")
    return 0


def main(argv: Sequence[str] | None = None, out: Out = print) -> int:
    """Runs the oran-causal CLI.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.
        out: Output sink.

    Returns:
        The process exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in DELEGATES:
        return _delegate(args, out)
    a = build_parser().parse_args(args)
    if a.command == "check":
        return run_checks(run_check, with_jobs(CHECKS, a.jobs))
    handlers: dict[str, Callable[[], int]] = {
        "data": lambda: _data(a, out),
        "paper-values": lambda: _paper_values(a),
        "reproduce": lambda: _reproduce(a, out),
        "verify": lambda: _verify(a, out),
        "run": lambda: _run_system(a, out),
    }
    return handlers[a.command]()


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point for the oran-causal workspace."""

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence

Runner = Callable[[Sequence[str]], int]

CHECKS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check"),
    ("ruff", "format", "--check"),
    ("ty", "check"),
    ("pytest",),
)


def run_module(args: Sequence[str]) -> int:
    """Runs `python -m <args>` with the current interpreter.

    Args:
        args: Module name followed by its arguments.

    Returns:
        The process exit code.
    """
    return subprocess.run([sys.executable, "-m", *args], check=False).returncode


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


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(prog="oran-causal")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="ruff check, ruff format --check, ty check, pytest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the oran-causal CLI.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        The process exit code.
    """
    build_parser().parse_args(argv)
    return run_checks(run_module)

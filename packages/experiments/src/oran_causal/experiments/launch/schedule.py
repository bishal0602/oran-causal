"""Recurring ticks (pull, keep-alive, dispatch) run by the OS scheduler."""

import dataclasses
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from oran_causal.experiments.paths import WORKSPACE_ROOT

LAUNCH_MODULE = "oran_causal.experiments.launch"
TASK_PREFIX = "oran-causal"

type Schtasks = Callable[[Sequence[str]], tuple[int, str]]

TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>{description}</Description></RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition><Interval>PT{every}M</Interval><StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>{start}</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType>
    <RunLevel>LeastPrivilege</RunLevel></Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{conhost}</Command>
      <Arguments>--headless cmd.exe /c "{cmd}"</Arguments>
      <WorkingDirectory>{root}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def tick_argv(kind: str, name: str) -> tuple[str, ...]:
    """Arguments after the interpreter that run one tick of a job or queue.

    Args:
        kind: `colab` (a Colab job) or `dispatch` (a queue).
        name: Job or queue name.
    """
    return ("-m", LAUNCH_MODULE, kind, "tick", name)


def task_name(kind: str, name: str) -> str:
    """OS scheduler task name of a tick."""
    return f"{TASK_PREFIX}-{kind}-{name}"


class Scheduler(Protocol):
    """Registers and removes a recurring tick."""

    def schedule(self, task: str, argv: Sequence[str], every_min: int, workdir: Path) -> str:
        """Registers `task` to run `python <argv>` every `every_min` minutes."""
        ...

    def unschedule(self, task: str) -> bool:
        """Removes `task`; True when it is gone."""
        ...


def run_schtasks(args: Sequence[str]) -> tuple[int, str]:
    """`schtasks <args>`; returns (rc, output)."""
    p = subprocess.run(
        ["schtasks", *args],
        capture_output=True,
        text=True,
        timeout=120,
        errors="replace",
        check=False,
    )
    return p.returncode, (p.stdout + p.stderr).strip()


@dataclasses.dataclass(frozen=True)
class TaskScheduler:
    """Windows Task Scheduler (current user, no admin, no console flash).

    Attributes:
        schtasks: Runs `schtasks` with arguments.
        python: Interpreter the tick runs with.
        root: Working directory of the tick.
        env: Variables the tick command file sets before starting.
    """

    schtasks: Schtasks = run_schtasks
    python: str = sys.executable
    root: Path = WORKSPACE_ROOT
    env: tuple[tuple[str, str], ...] = (("PYTHONIOENCODING", "utf-8"), ("MSYS_NO_PATHCONV", "1"))

    def command_file(self, task: str, argv: Sequence[str], workdir: Path) -> str:
        """Contents of the `.cmd` file the task runs."""
        out = workdir / f"{task}.out"
        lines = ["@echo off", *(f'set "{k}={v}"' for k, v in self.env)]
        lines += [
            f'cd /d "{self.root}"',
            f'echo ==== %DATE% %TIME% >> "{out}"',
            f'"{self.python}" {" ".join(argv)} >> "{out}" 2>&1',
        ]
        return "\n".join(lines) + "\n"

    def schedule(self, task: str, argv: Sequence[str], every_min: int, workdir: Path) -> str:
        """Writes the command file and task XML and registers the task.

        Raises:
            RuntimeError: If `schtasks /create` fails.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        cmd = workdir / f"{task}.cmd"
        cmd.write_text(self.command_file(task, argv, workdir), newline="\r\n")
        user = "\\".join(x for x in (os.environ.get("USERDOMAIN"), os.environ.get("USERNAME")) if x)
        start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 60 * every_min))
        conhost = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "conhost.exe"
        xml = workdir / f"{task}.xml"
        xml.write_text(
            TASK_XML.format(
                description=f"oran-causal {' '.join(argv[2:])}",
                every=every_min,
                start=start,
                user=user,
                conhost=conhost,
                cmd=cmd,
                root=self.root,
            ),
            encoding="utf-16",
        )
        rc, out = self.schtasks(["/create", "/tn", task, "/xml", str(xml), "/f"])
        if rc != 0:
            raise RuntimeError(f"schtasks /create failed rc={rc}: {out}")
        return self.schtasks(["/query", "/tn", task, "/fo", "list"])[1]

    def unschedule(self, task: str) -> bool:
        """Deletes the task and reports whether a query still finds it."""
        self.schtasks(["/delete", "/tn", task, "/f"])
        return self.schtasks(["/query", "/tn", task])[0] != 0


def cron_line(argv: Sequence[str], every_min: int, root: Path, log: Path, python: str) -> str:
    """A crontab entry that runs the tick every `every_min` minutes on a POSIX host."""
    command = f"cd {shlex.quote(str(root))} && {shlex.quote(python)} {shlex.join(argv)}"
    return f"*/{every_min} * * * * {command} >> {shlex.quote(str(log))} 2>&1"

"""Resource accounting and process isolation of one job."""

import ctypes
import dataclasses
import json
import math
import os
import select
import signal
import sys
import time
from collections.abc import Callable
from ctypes import wintypes
from types import TracebackType
from typing import Any

type Record = dict[str, Any]

if sys.platform != "win32":
    import resource

SIGXCPU = getattr(signal, "SIGXCPU", 24)
SIGKILL = getattr(signal, "SIGKILL", 9)
_READ_CHUNK = 1 << 20
NO_FORK = "fork isolation needs a POSIX platform"


def _linux_peak_mb() -> float | None:
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return None


def _windows_peak_mb() -> float | None:
    if sys.platform != "win32":
        return None
    counters = ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage")
    counters += ("QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage")
    counters += ("PagefileUsage", "PeakPagefileUsage")

    class Pmc(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in counters
        ]

    pmc = Pmc()
    pmc.cb = ctypes.sizeof(Pmc)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Pmc), wintypes.DWORD]
    if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return pmc.PeakWorkingSetSize / 2**20
    return None


class PeakRSS:
    """Peak resident memory over a block.

    Linux resets the high-water mark through `/proc/self/clear_refs` and reads `VmHWM`
    afterwards (scope `"job"`). Windows reads the process-lifetime `PeakWorkingSetSize`
    (scope `"process"`). Other platforms report `None`.
    """

    def __init__(self) -> None:
        self.scope: str | None = None
        self.mb: float | None = None

    def __enter__(self) -> PeakRSS:
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/clear_refs", "w", encoding="utf-8") as fh:
                    fh.write("5")
                self.scope = "job"
            except OSError:
                self.scope = "process"
        elif sys.platform == "win32":
            self.scope = "process"
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if sys.platform.startswith("linux"):
                self.mb = _linux_peak_mb()
            elif sys.platform == "win32":
                self.mb = _windows_peak_mb()
        except Exception:
            self.mb = None


def classify_kill(
    sig: int | None, cpu_s: float, limit_s: float | None, wall_hit: bool = False
) -> str:
    """Why an isolated child died.

    Returns:
        `"wall_limit"` when the parent killed it at the wall budget, `"cpu_limit"` on
        SIGXCPU or on SIGKILL at the hard CPU limit, otherwise `"killed"` (for example the
        kernel OOM killer), which is an error and never an infeasible unit.
    """
    if wall_hit:
        return "wall_limit"
    if sig == SIGXCPU:
        return "cpu_limit"
    if sig == SIGKILL and limit_s and cpu_s >= limit_s - 1.0:
        return "cpu_limit"
    return "killed"


@dataclasses.dataclass(frozen=True)
class IsolatedOutcome:
    """What the parent learns about one forked job."""

    record: Record | None
    cpu_s: float
    maxrss_mb: float
    signal: int | None
    wall_s: float
    wall_hit: bool


def _child(task: Callable[[], Record], write_fd: int, limit_s: float | None) -> None:
    if sys.platform == "win32":
        raise OSError(NO_FORK)
    code = 0
    try:
        if limit_s:
            hard = math.ceil(limit_s)
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard + 10))
        data = json.dumps(task(), allow_nan=False).encode()
        with os.fdopen(write_fd, "wb") as fh:
            fh.write(data)
    except BaseException:
        code = 1
    os._exit(code)


def _collect(read_fd: int, pid: int, started: float, wall_s: float | None) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    with os.fdopen(read_fd, "rb") as fh:
        while True:
            left = None if not wall_s else wall_s - (time.perf_counter() - started)
            if left is not None and left <= 0:
                os.kill(pid, SIGKILL)
                return b"".join(chunks), True
            ready, _, _ = select.select([fh], [], [], left)
            if not ready:
                continue
            block = fh.read1(_READ_CHUNK)
            if not block:
                return b"".join(chunks), False
            chunks.append(block)


def run_isolated(
    task: Callable[[], Record], limit_s: float | None, wall_s: float | None = None
) -> IsolatedOutcome:
    """Runs `task` in a forked child under `RLIMIT_CPU = limit_s` (Linux only).

    The parent kills the child after `wall_s` seconds of wall clock.

    Args:
        task: Produces the job record inside the child.
        limit_s: CPU-second limit, or None.
        wall_s: Wall-second limit, or None.

    Returns:
        The child's record (None if it died) with its CPU, memory and wall figures.
    """
    if sys.platform == "win32":
        raise OSError(NO_FORK)
    read_fd, write_fd = os.pipe()
    started = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        _child(task, write_fd, limit_s)
    os.close(write_fd)
    data, hit = _collect(read_fd, pid, started, wall_s)
    _, status, usage = os.wait4(pid, 0)
    sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
    record = json.loads(data) if data and sig is None and not hit else None
    return IsolatedOutcome(
        record=record,
        cpu_s=usage.ru_utime + usage.ru_stime,
        maxrss_mb=usage.ru_maxrss / 1024.0,
        signal=sig,
        wall_s=time.perf_counter() - started,
        wall_hit=hit,
    )


def can_isolate() -> bool:
    """Whether fork isolation is available (Linux with `os.fork`)."""
    return hasattr(os, "fork") and sys.platform.startswith("linux")

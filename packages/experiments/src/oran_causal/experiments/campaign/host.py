"""Host description and load around a unit: CPU model, vCPU quota, concurrent campaign processes."""

import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

PLATFORM_ENV = "ORAN_CAUSAL_PLATFORM"
CAMPAIGN_MARKER = b"oran_causal.experiments"


def _cpu_model() -> str | None:
    cpu = platform.processor() or None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return cpu


def _gpus() -> list[str] | None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if p.returncode != 0:
        return None
    return [x.strip() for x in p.stdout.splitlines() if x.strip()] or None


def host_info() -> dict[str, Any]:
    """Platform tag, CPU model and count, node name and GPUs of this host."""
    return {
        "platform": os.environ.get(PLATFORM_ENV, "local"),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "node": platform.node(),
        "gpu": _gpus(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _own_cgroup(self_cgroup: Path) -> str | None:
    own = None
    try:
        for line in self_cgroup.read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                own = line.strip()[3:]
    except OSError:
        pass
    return own


def _cgroup_chain(root: Path, own: str | None) -> list[Path]:
    root_s = os.path.normpath(root)
    if not own or own == "/":
        return [Path(root_s)]
    d = os.path.normpath(os.path.join(root_s, own.lstrip("/")))
    chain = []
    while d.startswith(root_s) and len(d) >= len(root_s):
        chain.append(Path(d))
        if d == root_s:
            break
        d = os.path.dirname(d)
    return chain


def _v2_quotas(dirs: list[Path]) -> list[float]:
    qs = []
    for d in dirs:
        try:
            a, b = (d / "cpu.max").read_text(encoding="utf-8").split()[:2]
            if a != "max":
                qs.append(float(a) / float(b))
        except OSError, ValueError:
            continue
    return qs


def _v1_quotas(root: Path) -> list[float]:
    for name in ("cpu", "cpu,cpuacct"):
        try:
            a = float((root / name / "cpu.cfs_quota_us").read_text(encoding="utf-8"))
            b = float((root / name / "cpu.cfs_period_us").read_text(encoding="utf-8"))
        except OSError, ValueError:
            continue
        return [a / b] if a > 0 else []
    return []


def cpu_quota(
    cgroup: Path = Path("/sys/fs/cgroup"), self_cgroup: Path = Path("/proc/self/cgroup")
) -> int:
    """VCPUs this process may use.

    The tightest cgroup v2 `cpu.max` quota from this process's own cgroup up to the root, else
    the v1 CFS quota, capped by the affinity mask; the visible CPU count without a quota.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    qs = _v2_quotas(_cgroup_chain(cgroup, _own_cgroup(self_cgroup))) or _v1_quotas(cgroup)
    q = min(qs) if qs else None
    return max(1, min(n, math.ceil(q - 1e-9))) if q else max(1, n)


def _cmdline(proc: Path, pid: str) -> list[bytes] | None:
    try:
        return (proc / pid / "cmdline").read_bytes().split(bytes(1))
    except OSError:
        return None


def campaign_procs(proc: Path = Path("/proc"), marker: bytes = CAMPAIGN_MARKER) -> int | None:
    """Concurrent campaign `run` processes: distinct `--out` shard files among their command lines.

    Returns:
        The count, or None without `/proc`.
    """
    if not proc.is_dir():
        return None
    outs = set()
    for d in (d for d in os.listdir(proc) if d.isdigit()):
        argv = _cmdline(proc, d)
        if argv is None or not any(marker in a for a in argv) or b"run" not in argv:
            continue
        outs.add(argv[argv.index(b"--out") + 1] if b"--out" in argv[:-1] else d.encode())
    return len(outs)


def proc_visible(proc: Path = Path("/proc")) -> int | None:
    """Processes listed in `/proc` (a restricted `/proc` hides other processes)."""
    return sum(1 for d in os.listdir(proc) if d.isdigit()) if proc.is_dir() else None


def load_info(proc: Path = Path("/proc")) -> dict[str, Any]:
    """Load averages, running / total processes, campaign processes and the vCPU quota."""
    out: dict[str, Any] = {
        "loadavg": None,
        "procs_running": None,
        "procs_total": None,
        "campaign_procs": campaign_procs(proc),
        "proc_visible": proc_visible(proc),
        "cpu_quota": cpu_quota(),
    }
    try:
        f = (proc / "loadavg").read_text(encoding="utf-8").split()
        r, t = f[3].split("/")
        out.update(loadavg=[float(x) for x in f[:3]], procs_running=int(r), procs_total=int(t))
    except OSError, ValueError, IndexError:
        if hasattr(os, "getloadavg"):
            out["loadavg"] = list(os.getloadavg())
    return out

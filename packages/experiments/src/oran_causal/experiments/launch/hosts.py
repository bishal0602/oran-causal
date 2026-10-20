"""Resource profiles of the platforms a campaign shard can run on."""

import dataclasses

from oran_causal.experiments.launch import settings

UNIT_PREFIX = "oran-causal-"
GIB_KB = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class LoadGuard:
    """Refusals checked before a launch on a shared host.

    Attributes:
        min_mem_available_kb: Refuse below this `MemAvailable`.
        max_other_load: Refuse when the 1-minute load, minus our own processes, exceeds this.
        one_job_at_a_time: Refuse while a scope of ours is active.
    """

    min_mem_available_kb: int
    max_other_load: float
    one_job_at_a_time: bool


@dataclasses.dataclass(frozen=True)
class HostProfile:
    """Caps a platform places on one campaign job.

    Attributes:
        name: Profile name.
        max_procs: Single-threaded processes one job may run.
        mem_max: systemd `MemoryMax` of the job scope, or None for no cap.
        swap_max: systemd `MemorySwapMax`, or None.
        load_guard: Pre-launch refusals, or None.
        wall_s: Default per-process timeout so the session ends before the platform's own limit.
        base_dir: Directory every remote file of a job lives under; None reads `base_setting`.
        base_setting: Environment variable of the base directory (see `settings`), or None for a
            platform without one.
        unit_prefix: Prefix of the job's systemd scope unit.
        user_scope: Run scopes in the login user's systemd instance (`systemd-run --user`), for
            a host where the SSH user is not root.
    """

    name: str
    max_procs: int
    mem_max: str | None
    swap_max: str | None
    load_guard: LoadGuard | None
    wall_s: int
    base_dir: str | None = None
    base_setting: str | None = None
    unit_prefix: str = UNIT_PREFIX
    user_scope: bool = False

    @property
    def base(self) -> str | None:
        """Directory every remote file of a job lives under, or None."""
        return self.base_dir or (settings.base(self.base_setting) if self.base_setting else None)

    @property
    def systemctl(self) -> str:
        """`systemctl` for the scopes of this host."""
        return "systemctl --user" if self.user_scope else "systemctl"

    @property
    def systemd_run(self) -> str:
        """`systemd-run` for the scopes of this host."""
        return "systemd-run --user" if self.user_scope else "systemd-run"

    def check_procs(self, procs: int) -> int:
        """Returns `procs` when it is within 1..`max_procs`.

        Raises:
            ValueError: Otherwise.
        """
        if not 1 <= procs <= self.max_procs:
            raise ValueError(f"--procs must be 1..{self.max_procs}")
        return procs

    def scope_limits(self, procs: int) -> tuple[str, ...]:
        """`systemd-run -p` arguments of a job of `procs` processes.

        CPUQuota is `procs` x 100 % so the one-process-per-vCPU gate of an EVAL run sees exactly
        `procs` vCPUs in the scope's cgroup.
        """
        limits = ["-p", f"CPUQuota={100 * self.check_procs(procs)}%"]
        if self.mem_max:
            limits += ["-p", f"MemoryMax={self.mem_max}"]
        if self.swap_max:
            limits += ["-p", f"MemorySwapMax={self.swap_max}"]
        return tuple(limits)


KAGGLE_CPU = HostProfile("kaggle-cpu", 4, None, None, None, 11 * 3600)
COLAB_CPU = HostProfile("colab-cpu", 4, None, None, None, 3 * 3600)
VPS_DEDICATED = HostProfile(
    "vps-dedicated", 16, None, None, None, 23 * 3600, base_setting=settings.VPS_BASE
)
VPS_SHARED = HostProfile(
    "vps-shared",
    7,
    "11G",
    "0",
    LoadGuard(12 * GIB_KB, 1.0, True),
    23 * 3600,
    base_setting=settings.VPS_BASE,
)

INSTABOX = HostProfile(
    "instabox",
    4,
    "6500M",
    "0",
    LoadGuard(6 * GIB_KB, 1.0, True),
    23 * 3600,
    base_setting=settings.INSTABOX_BASE,
    user_scope=True,
)

PROFILES: dict[str, HostProfile] = {
    p.name: p for p in (KAGGLE_CPU, COLAB_CPU, VPS_DEDICATED, VPS_SHARED, INSTABOX)
}

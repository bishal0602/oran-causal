"""Colab one-shot jobs: launch, and the recurring tick that pulls results and ends the runtime.

A job gets a new runtime, an uploaded bundle and one shell command under `nohup`. Ticks keep the
runtime alive, pack and download its outputs, and on the exit code stop the runtime and remove the
schedule. A job that loses its runtime is recorded as lost and is not relaunched: start a new one.
"""

import ctypes
import dataclasses
import json
import os
import shutil
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oran_causal.experiments.campaign.provenance import GitRunner
from oran_causal.experiments.launch import bundle, schedule
from oran_causal.experiments.launch.colab_session import (
    JOB_FILES,
    JOB_LAUNCH,
    JOB_PULL,
    JOB_SETUP,
    JOB_TICK,
    WORK,
    AssignmentError,
    Colab,
    fill,
)
from oran_causal.experiments.launch.runner import Dirs

EXEC_FAILURE_LIMIT = 3
LOCK_STALE_S = 3 * 3600
FINISH_WAIT_S = 600
FINISH_POLL_S = 20
STATE_FILE = "colab.json"
ENDED_KEYS = ("finished", "lost", "failed")
SETUP_TIMEOUT_S = 2700

type PidAlive = Callable[[int], bool]


def pid_alive(pid: int) -> bool:
    """Whether process `pid` exists (POSIX signal 0; Windows exit-code query)."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return code.value == 259


def utc(now: float | None = None) -> str:
    """ISO UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


@dataclasses.dataclass(frozen=True)
class ColabJob:
    """One Colab job.

    Attributes:
        name: Session and job name; new per launch.
        cmd: The command run by bash from the bundle root.
        extra: Workspace-relative files or directories added to the bundle.
        out_dir: Bundle-relative directory whose contents each tick downloads, or None.
        threads: BLAS/OpenMP threads; None means the runtime's CPU quota.
        frozen_protocol: Verified frozen protocol text bundled for the EVAL guard.
        every_min: Tick interval.
        scheduled: Register the recurring tick after launch.
    """

    name: str
    cmd: str
    extra: tuple[str, ...] = ()
    out_dir: str | None = None
    threads: int | None = None
    frozen_protocol: bytes | None = None
    every_min: int = 10
    scheduled: bool = True

    def __post_init__(self) -> None:
        """Rejects an `out_dir` that is absolute, climbs out of the bundle or names a job file."""
        if self.out_dir is None:
            return
        parts = self.out_dir.replace("\\", "/").strip("/").split("/")
        if (
            os.path.isabs(self.out_dir)
            or self.out_dir.startswith(("/", "\\"))
            or ".." in parts
            or self.out_dir.strip("/") in JOB_FILES
            or not self.out_dir.strip("/")
        ):
            raise ValueError(f"out_dir must be a bundle-relative directory, got {self.out_dir!r}")
        object.__setattr__(self, "out_dir", "/".join(parts))


@dataclasses.dataclass(frozen=True)
class ColabEnv:
    """What the job functions act through.

    Attributes:
        colab: The Colab account.
        dirs: Local directories; the job's state lives in `dirs.runs/<name>`.
        scheduler: Registers the recurring tick.
        git: Git runner for the manifest.
        alive: Process liveness for the tick lock.
        log: Output sink for progress lines.
    """

    colab: Colab
    dirs: Dirs
    scheduler: schedule.Scheduler
    git: GitRunner | None = None
    alive: PidAlive = pid_alive
    log: Callable[[str], None] = print

    def run_dir(self, name: str) -> Path:
        """Local directory of job `name`."""
        return self.dirs.runs / name

    def load(self, name: str) -> dict[str, Any]:
        """The job's state record."""
        return json.loads((self.run_dir(name) / STATE_FILE).read_text())

    def save(self, name: str, meta: dict[str, Any]) -> None:
        """Writes the state record atomically."""
        d = self.run_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(d / STATE_FILE)

    def event(self, name: str, event: str, info: dict[str, Any] | None = None) -> None:
        """Appends one line to the job's `tick.log` and echoes it."""
        d = self.run_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        line = f"{utc(self.colab.clock.now())} {event} {json.dumps(info or {}, default=str)}"
        with (d / "tick.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.log(line)

    def arm(self, name: str, every_min: int) -> str:
        """Registers the recurring tick of job `name`."""
        return self.scheduler.schedule(
            schedule.task_name("colab", name),
            schedule.tick_argv("colab", name),
            every_min,
            self.run_dir(name),
        )

    def disarm(self, name: str) -> bool:
        """Removes the recurring tick of job `name`; True when it is gone."""
        gone = self.scheduler.unschedule(schedule.task_name("colab", name))
        self.event(name, "unscheduled", {"gone": gone})
        return gone


@dataclasses.dataclass
class TickLock:
    """`tick.lock` of a job: one tick at a time; stale when its pid is gone or it is old."""

    path: Path
    alive: PidAlive
    now: Callable[[], float] = time.time
    held: bool = False

    def __enter__(self) -> bool:
        """Takes the lock; False when another live tick holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._stale():
                    return False
                self.path.unlink(missing_ok=True)
                continue
            os.write(fd, json.dumps({"pid": os.getpid(), "t": self.now()}).encode())
            os.close(fd)
            self.held = True
            return True
        return False

    def __exit__(self, *exc: object) -> None:
        """Releases the lock if this tick took it."""
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def _stale(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
            return not self.alive(info["pid"]) or self.now() - info["t"] > LOCK_STALE_S
        except OSError, ValueError, KeyError:
            return True


def launch_job(job: ColabJob, env: ColabEnv, root: Path | None = None) -> dict[str, Any]:
    """Bundles the files, creates a runtime, uploads, sets up and starts the command.

    Args:
        job: The job.
        env: Account, directories and scheduler.
        root: Workspace to bundle; default `env.dirs.root`.

    Returns:
        The job's state record.

    Raises:
        FileExistsError: If the job name already has a record.
        BaseException: Any failure after the runtime exists stops the runtime first and is
            recorded in the state as `failed_utc` / `error`.
    """
    d = env.run_dir(job.name)
    if (d / STATE_FILE).exists():
        raise FileExistsError(f"{d / STATE_FILE} exists: pick a new job name")
    held = env.colab.active_assignments()
    if held:
        env.log(f"WARNING: {held} active assignment(s) already; a new runtime may be refused")
    req = bundle.BundleRequest(
        job.name, root or env.dirs.root, job.extra, job.frozen_protocol, job.cmd
    )
    built = bundle.build_bundle(req, env.dirs.work / job.name, env.git)
    meta: dict[str, Any] = {
        "kind": "job",
        "session": job.name,
        "cmd": job.cmd,
        "out_dir": job.out_dir,
        "threads": job.threads,
        "git_head": built.manifest["git_head"],
        "dirty": built.manifest["dirty"],
        "n_files": len(built.files),
        "bundle_bytes": built.zip_path.stat().st_size,
        "created_utc": utc(env.colab.clock.now()),
    }
    env.save(job.name, meta)
    env.log(
        f"bundle {meta['n_files']} files, {meta['bundle_bytes'] / 1e6:.1f} MB, "
        f"head {meta['git_head']}, dirty {meta['dirty']}"
    )
    t0 = env.colab.clock.now()
    env.colab.new_session(job.name)
    meta["t_new"] = round(env.colab.clock.now() - t0, 1)
    env.save(job.name, meta)
    try:
        _start(job, env, built.zip_path, meta)
    except BaseException as e:
        meta["failed_utc"], meta["error"] = utc(env.colab.clock.now()), repr(e)[-1500:]
        env.save(job.name, meta)
        env.event(job.name, "job-setup-failed", {"err": repr(e)[-800:]})
        env.colab.stop(job.name)
        raise
    finally:
        built.zip_path.unlink(missing_ok=True)
    meta["launched_utc"] = utc(env.colab.clock.now())
    env.save(job.name, meta)
    launched = meta["launched"]
    env.event(
        job.name,
        "job-launch",
        {"pid": launched["pid"], "alive_5s": launched["alive_5s"], "cmd": job.cmd},
    )
    if not launched["alive_5s"]:
        env.log(
            "NOTE: the job had exited 5 s after launch; log head:\n" + str(launched["log_head"])
        )
    if job.scheduled:
        env.arm(job.name, job.every_min)
        meta.update(task=schedule.task_name("colab", job.name), every_min=job.every_min)
        env.save(job.name, meta)
    return meta


def _start(job: ColabJob, env: ColabEnv, zip_path: Path, meta: dict[str, Any]) -> None:
    colab = env.colab
    t1 = colab.clock.now()
    remote_zip = f"/content/{job.name}_job.zip"
    colab.upload(job.name, zip_path, remote_zip)
    meta["t_upload"] = round(colab.clock.now() - t1, 1)
    setup = fill(JOB_SETUP, W=WORK, BUNDLE=remote_zip, THREADS=job.threads, FILES=list(JOB_FILES))
    start = colab.remote(job.name, setup, timeout=SETUP_TIMEOUT_S, tag="JOBSTART")
    meta.update(t_setup=round(colab.clock.now() - t1, 1), startup=start)
    (env.run_dir(job.name) / "startup.json").write_text(json.dumps(start, indent=1))
    launch = fill(JOB_LAUNCH, W=WORK, CMD=job.cmd, OUT=job.out_dir)
    meta["launched"] = colab.remote(job.name, launch, timeout=300, tag="JOBLAUNCHED")


def remote_state(name: str, env: ColabEnv, pack: bool = True) -> dict[str, Any]:
    """Runs the tick cell in the session: process state, log tail, and the packed outputs."""
    meta = env.load(name)
    code = fill(
        JOB_TICK,
        W=WORK,
        OUT=meta.get("out_dir"),
        DST=JOB_PULL,
        FILES=list(JOB_FILES),
        PACK=pack,
    )
    return env.colab.remote(name, code, timeout=600, tag="JOBTICK")


def download(name: str, env: ColabEnv) -> dict[str, list[str]]:
    """Downloads the packed tarball into `runs/<name>` (the out dir to `runs/<name>/<out_dir>`).

    Returns:
        `{"pulled": [...]}`: the job files and the out dir (with a trailing slash) that arrived.
    """
    meta = env.load(name)
    d = env.run_dir(name)
    tgz, tmp = d / "pull.tar.gz", d / ".pull"
    env.colab.download(name, JOB_PULL, tgz)
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tgz) as t:
        t.extractall(tmp, filter="data")
    tgz.unlink()
    got: list[str] = []
    for src in sorted(tmp.iterdir()):
        if src.name == "outdir" and src.is_dir() and meta.get("out_dir"):
            shutil.copytree(src, d.joinpath(*meta["out_dir"].split("/")), dirs_exist_ok=True)
            got.append(meta["out_dir"] + "/")
        elif src.name in JOB_FILES and src.is_file():
            shutil.copyfile(src, d / src.name)
            got.append(src.name)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"pulled": got}


def brief(state: dict[str, Any]) -> dict[str, Any]:
    """The fields of a tick state worth showing."""
    keys = (
        "alive",
        "exit_code",
        "elapsed_s",
        "live_cpu_s",
        "live_rss_mb",
        "live_procs",
        "mem_mb",
        "log_lines",
        "traceback",
        "n_out",
        "times",
        "load",
    )
    return {k: state.get(k) for k in keys}


def status(name: str, env: ColabEnv) -> dict[str, Any]:
    """Process state of the job without packing outputs."""
    return remote_state(name, env, pack=False)


def pull(name: str, env: ColabEnv) -> dict[str, Any]:
    """Packs and downloads the job's outputs now; returns its state plus what arrived."""
    state = remote_state(name, env)
    return {**brief(state), **download(name, env)}


def stop(name: str, env: ColabEnv) -> int:
    """Stops the runtime, verifies it is unassigned and removes the schedule.

    Returns:
        The other assignments still active.
    """
    others = env.colab.stop(name)
    env.disarm(name)
    return others


def lost(name: str, env: ColabEnv, why: str, info: dict[str, Any] | None = None) -> None:
    """Records the job as lost (never relaunched) and removes the schedule."""
    meta = env.load(name)
    meta["lost_utc"], meta["lost_reason"] = utc(env.colab.clock.now()), why
    env.save(name, meta)
    env.event(
        name,
        "job-lost",
        {**(info or {}), "reason": why, "action": "not resumable: re-run under a new name"},
    )
    env.disarm(name)


def finish(name: str, env: ColabEnv, info: dict[str, Any], runtime_alive: bool = True) -> bool:
    """After the final pull: stops the runtime, verifies it is unassigned, marks finished.

    The schedule stays on when the runtime cannot be verified gone, so the next tick retries.

    Returns:
        True when the job is marked finished.
    """
    env.event(name, "final", info)
    try:
        if runtime_alive:
            env.colab.stop(name)
        else:
            ok, n = env.colab.wait_gone(name, FINISH_WAIT_S)
            if not ok:
                raise AssignmentError(f"STILL {n} ACTIVE COLAB ASSIGNMENT(S)")
    except AssignmentError as e:
        env.event(name, "stop-failed", {"err": str(e)})
        return False
    meta = env.load(name)
    meta["finished_utc"], meta["final"] = utc(env.colab.clock.now()), info
    env.save(name, meta)
    env.event(name, "done", {"finished_utc": meta["finished_utc"]})
    env.disarm(name)
    return True


def tick(name: str, env: ColabEnv) -> str:
    """One tick: keep the runtime alive, pull, and finish or give up.

    Returns:
        What the tick did: `skip`, `noop`, `probe-error`, `exec-error`, `lost`, `running`,
        `finished` or `finish-pending`.
    """
    with TickLock(env.run_dir(name) / "tick.lock", env.alive, env.colab.clock.now) as got:
        if not got:
            env.event(name, "skip", {"reason": "another tick holds tick.lock"})
            return "skip"
        meta = env.load(name)
        ended = {k: meta[k + "_utc"] for k in ENDED_KEYS if meta.get(k + "_utc")}
        if ended:
            env.event(name, "noop", ended)
            env.disarm(name)
            return "noop"
        try:
            probe = env.colab.probe(name, True)
        except Exception as e:
            env.event(name, "probe-error", {"err": str(e)[-400:]})
            return "probe-error"
        keepalive = {k: probe.get(k) for k in ("keepalive_http", "keepalive_err", "kernels_err")}
        if probe["alive"]:
            return _tick_alive(name, env, meta, keepalive)
        return _tick_gone(name, env, meta, probe)


def _tick_alive(name: str, env: ColabEnv, meta: dict[str, Any], keepalive: dict[str, Any]) -> str:
    try:
        state = remote_state(name, env)
        got = download(name, env)
    except Exception as e:
        meta["exec_failures"] = meta.get("exec_failures", 0) + 1
        env.save(name, meta)
        env.event(
            name, "exec-error", {"n": meta["exec_failures"], "err": str(e)[-400:], **keepalive}
        )
        if meta["exec_failures"] < EXEC_FAILURE_LIMIT:
            return "exec-error"
        try:
            env.colab.stop(name)
        except AssignmentError as stop_err:
            env.event(name, "stop-failed", {"err": str(stop_err)})
            return "exec-error"
        lost(name, env, f"{EXEC_FAILURE_LIMIT} consecutive failed execs (runtime stopped)")
        return "lost"
    meta["exec_failures"] = 0
    info = {
        **brief(state),
        "log_tail": state["log_tail"][-3:],
        "uptime_min": round(state["uptime_s"] / 60, 1),
        **got,
        **keepalive,
    }
    if state["exit_code"] is None and state["alive"]:
        env.save(name, meta)
        env.event(name, "tick", info)
        return "running"
    meta["exit_code"] = state["exit_code"]
    info["result"] = (
        f"exit {state['exit_code']}"
        if state["exit_code"] is not None
        else "DIED without an exit code (wrapper killed?)"
    )
    info["out_files"] = state["out_files"]
    env.save(name, meta)
    return "finished" if finish(name, env, info) else "finish-pending"


def _tick_gone(name: str, env: ColabEnv, meta: dict[str, Any], probe: dict[str, Any]) -> str:
    code = env.run_dir(name) / "exit_code"
    if code.exists():
        meta["exit_code"] = int(code.read_text().strip() or -1)
        env.save(name, meta)
        note = {"exit_code": meta["exit_code"], "note": "already complete locally"}
        return "finished" if finish(name, env, note, runtime_alive=False) else "finish-pending"
    held = probe.get("n_assignments")
    lost(
        name,
        env,
        "session gone before the exit code was pulled",
        {
            "n_assignments": held,
            "orphans": probe.get("orphans"),
            "known": probe.get("known"),
            "warning": "orphan assignment(s): stop them by hand (colab --auth adc sessions)"
            if held
            else None,
        },
    )
    return "lost"

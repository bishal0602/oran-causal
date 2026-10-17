"""Unattended dispatcher for a campaign spec split into parts.

Each tick pulls finished jobs, checks which of their parts are incomplete (those are requeued, at
most three attempts per part), and launches at most one chunk of parts on the first platform with
room, except that every free Kaggle slot is filled in one tick. Order: Kaggle (long unattended),
the SSH hosts (the VPS, then instabox; one job at a time each), Colab. State is `dispatch.json`
and `dispatch.log` in `runs/<queue>/`.
"""

import dataclasses
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from oran_causal.experiments.campaign.merge import iter_jsonl
from oran_causal.experiments.campaign.shard import complete_dataset_keys, partition
from oran_causal.experiments.campaign.spec import expand
from oran_causal.experiments.launch import colab, plan, schedule
from oran_causal.experiments.launch.runner import Clock, Dirs

KAGGLE_MAX = 5
KAGGLE_MINE_MAX = 4
COLAB_MAX = 2
KAGGLE_CHUNK = 8
COLAB_CHUNK = 4
VPS_CHUNK = 7
INSTABOX_CHUNK = 4
COLAB_WALL_S = 39600
COLAB_BACKOFF_S = 7200
VPS_BACKOFF_S = 1800
MAX_ATTEMPTS = 3
LOCK_STALE_S = 7200
MINE_PREFIXES = ("xm-dev-", "xm-eval-")
COLAB_REFUSED = "Allocation refused"
VPS_REFUSED = "refused:"
PLATFORMS: tuple[plan.Platform, ...] = ("kaggle", "vps", "instabox", "colab")
STATE_FILE = "dispatch.json"
LOG_FILE = "dispatch.log"
DONE_DIR = "done"

type Outcome = plan.Platform | Literal["priority"] | None


class Backend(Protocol):
    """What the dispatcher needs from the platforms; replaced by a fake in tests."""

    def kaggle_counts(self, prefixes: Sequence[str]) -> tuple[int, int]:
        """(sessions RUNNING or QUEUED on the account, how many of them match `prefixes`)."""
        ...

    def kaggle_state(self, job: str) -> str:
        """Kernel status text of `job`."""
        ...

    def kaggle_pull(self, job: str) -> None:
        """Downloads the kernel output into `runs/<job>`."""
        ...

    def ssh_pull(self, platform: plan.Platform, job: str) -> None:
        """Merges the outputs of `job` on the SSH host `platform` into `runs/<job>`."""
        ...

    def launch(self, request: plan.LaunchRequest) -> None:
        """Plans and launches; raises on any failure (the message decides the back-off)."""
        ...


@dataclasses.dataclass(frozen=True)
class Chunk:
    """Parts one launch runs.

    Attributes:
        platform: Where.
        ids: The part indexes.
        job: The job name.
    """

    platform: plan.Platform
    ids: list[int]
    job: str


@dataclasses.dataclass
class Queue:
    """Persistent state of one dispatch queue.

    Attributes:
        spec: Workspace-relative spec path.
        cost_table: Workspace-relative cost table path.
        parts: Number of parts.
        kaggle_chunk: Parts per Kaggle session.
        kaggle_mine_max: Own Kaggle sessions at once (the account allows `KAGGLE_MAX`).
        colab_chunk: Parts per Colab job.
        skip_from: Globs of earlier shard files whose complete datasets are skipped.
        vps: The VPS lane is enabled.
        instabox: The instabox lane is enabled.
        vps_chunk: Parts (one process each) per VPS job.
        instabox_chunk: Parts (one process each) per instabox job.
        only: Restrict the queue to these platforms; None means all.
        mine_prefixes: Slug prefixes that count as this account's own sessions.
        next: First part not yet launched.
        jobs: Launched jobs (`job`, `platform`, `parts`, `utc`, `pulled`, ...).
        attempts: Name counter per job kind.
        requeue: Parts waiting to be relaunched.
        part_attempts: Launch attempts per part (as a string key).
        priority: Jobs of another spec that launch first on Kaggle.
        parts_end: Last part (exclusive) to launch, or None for `parts`.
        vps_backoff_until: Epoch seconds before which the VPS is not tried.
        colab_backoff_until: Epoch seconds before which Colab is not tried.
        instabox_backoff_until: Epoch seconds before which instabox is not tried.
    """

    spec: str
    cost_table: str
    parts: int
    kaggle_chunk: int = KAGGLE_CHUNK
    kaggle_mine_max: int = KAGGLE_MINE_MAX
    colab_chunk: int = COLAB_CHUNK
    skip_from: list[str] = dataclasses.field(default_factory=list)
    vps: bool = False
    instabox: bool = False
    vps_chunk: int = VPS_CHUNK
    instabox_chunk: int = INSTABOX_CHUNK
    only: list[str] | None = None
    mine_prefixes: list[str] = dataclasses.field(default_factory=lambda: list(MINE_PREFIXES))
    next: int = 0
    jobs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    attempts: dict[str, int] = dataclasses.field(default_factory=dict)
    requeue: list[int] = dataclasses.field(default_factory=list)
    part_attempts: dict[str, int] = dataclasses.field(default_factory=dict)
    priority: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    parts_end: int | None = None
    vps_backoff_until: float = 0.0
    colab_backoff_until: float = 0.0
    instabox_backoff_until: float = 0.0

    @property
    def end(self) -> int:
        """Exclusive end of the parts to launch."""
        return self.parts if self.parts_end is None else self.parts_end

    def allows(self, platform: str) -> bool:
        """Whether the queue may use `platform`."""
        return platform in (self.only or PLATFORMS)

    def ssh_enabled(self, platform: plan.Platform) -> bool:
        """Whether the lane of the SSH host `platform` is enabled."""
        return self.instabox if platform == "instabox" else self.vps

    def ssh_chunk(self, platform: plan.Platform) -> int:
        """Parts per job on the SSH host `platform`."""
        return self.instabox_chunk if platform == "instabox" else self.vps_chunk

    def backoff_until(self, platform: plan.Platform) -> float:
        """Epoch seconds before which the SSH host `platform` is not tried."""
        return self.instabox_backoff_until if platform == "instabox" else self.vps_backoff_until

    def back_off(self, platform: plan.Platform, until: float) -> None:
        """Pauses the SSH host `platform` until `until`."""
        if platform == "instabox":
            self.instabox_backoff_until = until
        else:
            self.vps_backoff_until = until

    def exhausted(self) -> bool:
        """No part left to launch, requeue or run with priority."""
        return self.next >= self.end and not self.requeue and not self.priority


def utc(now: float) -> str:
    """ISO UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


@dataclasses.dataclass
class DispatchLock:
    """`dispatch.lock` of a queue: one tick at a time; stale after two hours."""

    path: Path
    now: Callable[[], float] = time.time
    held: bool = False

    def __enter__(self) -> bool:
        """Takes the lock; False when another tick holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.now() - self.path.stat().st_mtime > LOCK_STALE_S:
            self.path.unlink(missing_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self.held = True
        return True

    def __exit__(self, *exc: object) -> None:
        """Releases the lock if this tick took it."""
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


@dataclasses.dataclass(frozen=True)
class Dispatcher:
    """A queue's tick logic over a backend.

    Attributes:
        dirs: Local directories; the queue's state lives in `dirs.runs/<name>`.
        backend: The platforms.
        clock: Time source.
        scheduler: Registers the recurring dispatch tick.
        dry_run: Print launches, pull nothing, launch nothing.
        kaggle_now: Dry run: assume these Kaggle counts instead of asking Kaggle.
        log: Output sink for dry-run lines.
    """

    dirs: Dirs
    backend: Backend
    clock: Clock = dataclasses.field(default_factory=Clock)
    scheduler: schedule.Scheduler | None = None
    dry_run: bool = False
    kaggle_now: tuple[int, int] | None = None
    log: Callable[[str], None] = print

    def qdir(self, name: str) -> Path:
        """Directory of queue `name`."""
        return self.dirs.runs / name

    def load(self, name: str) -> Queue:
        """The queue's state."""
        return Queue(**json.loads((self.qdir(name) / STATE_FILE).read_text(encoding="utf-8")))

    def save(self, name: str, queue: Queue) -> None:
        """Writes the queue's state."""
        d = self.qdir(name)
        d.mkdir(parents=True, exist_ok=True)
        (d / STATE_FILE).write_text(
            json.dumps(dataclasses.asdict(queue), indent=1), encoding="utf-8", newline="\n"
        )

    def event(self, name: str, event: str, info: Any) -> None:
        """Appends one line to the queue's log."""
        d = self.qdir(name)
        d.mkdir(parents=True, exist_ok=True)
        line = f"{utc(self.clock.now())} {event} {json.dumps(info, default=str)}\n"
        with (d / LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(line)

    def init(self, name: str, queue: Queue) -> None:
        """Creates the queue."""
        queue.vps = queue.vps or "vps" in (queue.only or ())
        queue.instabox = queue.instabox or "instabox" in (queue.only or ())
        queue.kaggle_mine_max = min(queue.kaggle_mine_max, KAGGLE_MAX)
        self.save(name, queue)
        self.event(name, "init", dataclasses.asdict(queue))

    def colab_active(self) -> int:
        """Own Colab jobs launched and neither finished, lost nor fully pulled."""
        n = 0
        for f in sorted(self.dirs.runs.glob("*/colab.json")):
            meta = json.loads(f.read_text(encoding="utf-8"))
            done = (f.parent / "exit_code").exists()
            ended = meta.get("finished_utc") or meta.get("lost_utc")
            n += bool(meta.get("kind") == "job" and meta.get("launched") and not ended and not done)
        return n

    def outputs(self, job: dict[str, Any]) -> list[Path]:
        """Shard files pulled for `job` (DEV `res_*`, EVAL `eval_res_*`)."""
        subs = (plan.COLAB_OUT, plan.COLAB_EVAL_OUT) if job["platform"] == "colab" else ("out",)
        base = self.dirs.runs / job["job"]
        return sorted(f for sub in subs for f in (base / sub).glob("**/*res_*.jsonl"))

    def check_job(self, name: str, queue: Queue, job: dict[str, Any]) -> list[int]:
        """After `job` is pulled: slim record copies go to `done/` and incomplete parts requeue.

        The slim copies (key, status, error, run_mode, spec hash) are the `--skip-complete-from`
        input of requeues. A priority job (another spec) is only reported.

        Returns:
            The job's parts whose datasets are not all complete.
        """
        done_dir = self.qdir(name) / DONE_DIR
        done_dir.mkdir(parents=True, exist_ok=True)
        for f in self.outputs(job):
            with (done_dir / f"{job['job']}__{f.name}").open(
                "w", encoding="utf-8", newline="\n"
            ) as fo:
                for r in iter_jsonl(f):
                    slim = {
                        "key": r.get("key"),
                        "status": r.get("status"),
                        "error": r.get("error"),
                        "run_mode": r.get("run_mode"),
                        "integrity": {"spec_sha256": (r.get("integrity") or {}).get("spec_sha256")},
                    }
                    fo.write(json.dumps(slim) + "\n")
        root = self.dirs.root
        if job.get("spec"):
            units = partition(expand(plan.spec_of(root / job["spec"])), job["nparts"], None)
            got = {
                r.get("key")
                for f in self.outputs(job)
                for r in iter_jsonl(f)
                if r.get("status") in ("ok", "infeasible")
            }
            job["incomplete_parts"] = [i for i in job["parts"] if {u.key for u in units[i]} - got]
            self.event(
                name,
                "checked",
                {"job": job["job"], "incomplete_parts": job["incomplete_parts"], "priority": True},
            )
            return job["incomplete_parts"]
        spec = plan.spec_of(root / queue.spec)
        table = json.loads((root / queue.cost_table).read_text(encoding="utf-8"))
        units = partition(expand(spec), queue.parts, table)
        files = sorted(done_dir.glob("*.jsonl"))
        incomplete = [
            i
            for i in job["parts"]
            if {u.key for u in units[i]} - complete_dataset_keys(units[i], files)
        ]
        for i in incomplete:
            tries = queue.part_attempts.get(str(i), 1) + 1
            queue.part_attempts[str(i)] = tries
            if tries <= MAX_ATTEMPTS and i not in queue.requeue:
                queue.requeue.append(i)
        job["incomplete_parts"] = incomplete
        self.event(
            name,
            "checked",
            {"job": job["job"], "incomplete_parts": incomplete, "requeue": queue.requeue},
        )
        return incomplete

    def pull_finished(self, name: str, queue: Queue) -> bool:
        """Pulls every finished job once; True when every launched job is finished and pulled."""
        all_done = True
        for job in queue.jobs:
            if job.get("pulled"):
                continue
            state = self._pull_one(name, queue, job)
            if state:
                job["pulled"] = state
                self.event(name, "pulled", {"job": job["job"], "state": state})
                self.check_job(name, queue, job)
            else:
                all_done = False
        self.save(name, queue)
        return all_done

    def _pull_one(self, name: str, queue: Queue, job: dict[str, Any]) -> str | None:
        run = self.dirs.runs / job["job"]
        if job["platform"] in plan.SSH_PLATFORMS:
            self.backend.ssh_pull(job["platform"], job["job"])
            code = run / "exit_code"
            return f"{job['platform']} exit {code.read_text().strip()}" if code.exists() else None
        if job["platform"] == "colab":
            path = run / colab.STATE_FILE
            meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if meta.get("finished_utc") or meta.get("lost_utc") or (run / "exit_code").exists():
                return "colab " + ("lost" if meta.get("lost_utc") else "finished")
            return None
        text = self.backend.kaggle_state(job["job"])
        if "RUNNING" in text or "QUEUED" in text or "Status" not in text:
            return None
        try:
            self.backend.kaggle_pull(job["job"])
        except Exception as e:
            self.event(name, "pull-failed", {"job": job["job"], "err": str(e)[-300:]})
            return None
        return text.strip().split()[-1].strip('"')

    def job_name(self, name: str, queue: Queue, kind: str) -> str:
        """A slug never used before: attempt counter and UTC day-hour-minute."""
        queue.attempts[kind] = queue.attempts.get(kind, 0) + 1
        stamp = time.strftime("%d%H%M", time.gmtime(self.clock.now()))
        return f"{name}-{kind[0]}{queue.attempts[kind]}-{stamp}"

    def relative(self, glob: str) -> str:
        """A skip glob as a workspace-relative posix pattern."""
        path = Path(glob)
        if path.is_absolute():
            return path.relative_to(self.dirs.root).as_posix()
        return glob.replace("\\", "/")

    def request(
        self, name: str, queue: Queue, chunk: Chunk, requeue: bool = False
    ) -> plan.LaunchRequest:
        """The launch request of a chunk of the queue's own spec.

        A requeue also skips the datasets already complete in the queue's `done/` copies.
        """
        root = self.dirs.root
        platform, ids, job = chunk.platform, chunk.ids, chunk.job
        skip_from = [self.relative(g) for g in queue.skip_from]
        if requeue:
            done = self.qdir(name) / DONE_DIR / "*.jsonl"
            skip_from.append(done.relative_to(root).as_posix())
        return plan.LaunchRequest(
            platform,
            job,
            root / queue.spec,
            queue.parts,
            ids,
            root / queue.cost_table,
            skip_from=skip_from,
            wall_s=COLAB_WALL_S if platform == "colab" else None,
            selftest=True,
            procs=min(queue.ssh_chunk(platform), len(ids))
            if platform in plan.SSH_PLATFORMS
            else None,
            root=root,
        )

    def launch(self, request: plan.LaunchRequest) -> tuple[int, str]:
        """(rc, message tail) of one launch; a dry run prints and succeeds."""
        if self.dry_run:
            self.log(
                f"[dry-run] {request.platform} {request.name} parts {list(request.part_ids or [])}"
            )
            return 0, "dry-run"
        try:
            self.backend.launch(request)
        except Exception as e:
            return 1, str(e)[-3000:]
        return 0, "launched"

    def tick(self, name: str) -> str:
        """One tick of queue `name`.

        Returns:
            `skipped` (another tick holds the lock), `done` (all parts launched, schedule removed
            when all pulled), or the platform of the launch (`none` when nothing launched).
        """
        with DispatchLock(self.qdir(name) / "dispatch.lock", self.clock.now) as mine:
            if not mine:
                self.event(name, "tick-skipped", {"why": "another tick holds dispatch.lock"})
                return "skipped"
            return self._tick(name)

    def _tick(self, name: str) -> str:
        queue = self.load(name)
        for job in queue.jobs:
            if job.get("pulled") and "incomplete_parts" not in job:
                self.check_job(name, queue, job)
        self.save(name, queue)
        pulled = True if self.dry_run else self.pull_finished(name, queue)
        if queue.exhausted():
            self.event(name, "all-launched", {"all_pulled": pulled})
            if pulled and not self.dry_run:
                self.unschedule(name)
            return "done"
        running, mine = self.kaggle_now or self.backend.kaggle_counts(queue.mine_prefixes)
        colab_n = self.colab_active()
        last = "none"
        for _ in range(KAGGLE_MAX + 1):
            last = self.launch_one(name, queue, running, mine, colab_n) or "none"
            if last != "kaggle":
                break
            running, mine = running + 1, mine + 1
        return last

    def launch_one(self, name: str, queue: Queue, running: int, mine: int, colab_n: int) -> Outcome:
        """At most one launch: priority jobs, requeues, then Kaggle, the VPS, Colab."""
        if queue.exhausted():
            return None
        info = {
            "kaggle_running": running,
            "kaggle_mine": mine,
            "colab_active": colab_n,
            "next": queue.next,
            "requeue": queue.requeue,
        }
        kaggle_room = running < KAGGLE_MAX and mine < queue.kaggle_mine_max
        if queue.priority:
            return self._priority(name, queue, info, kaggle_room)
        now = self.clock.now()
        ssh_free = [p for p in plan.SSH_PLATFORMS if self._ssh_free(queue, p, now)]
        if queue.requeue:
            kag = kaggle_room and queue.allows("kaggle")
            if kag or ssh_free:
                return self._requeue(name, queue, info, "kaggle" if kag else ssh_free[0])
            if queue.next >= queue.end:
                self.event(name, "no-room-requeue", info)
                return None
        lanes = (
            ("kaggle", kaggle_room, queue.kaggle_chunk),
            *((p, p in ssh_free, queue.ssh_chunk(p)) for p in plan.SSH_PLATFORMS),
            ("colab", colab_n < COLAB_MAX and now >= queue.colab_backoff_until, queue.colab_chunk),
        )
        for platform, room, chunk in lanes:
            if room and queue.allows(platform):
                return self._chunk(name, queue, info, platform, chunk)
        self.event(name, "no-room", info)
        return None

    def _ssh_free(self, queue: Queue, platform: plan.Platform, now: float) -> bool:
        busy = any(j["platform"] == platform and not j.get("pulled") for j in queue.jobs)
        return queue.ssh_enabled(platform) and not busy and now >= queue.backoff_until(platform)

    def _priority(
        self, name: str, queue: Queue, info: dict[str, Any], kaggle_room: bool
    ) -> Outcome:
        if not kaggle_room:
            self.event(name, "no-room-priority", info)
            return None
        pri = queue.priority[0]
        job = self.job_name(name, queue, pri.get("kind", "pilot"))
        self.save(name, queue)
        ids = list(range(pri["nparts"]))
        root = self.dirs.root
        request = plan.LaunchRequest(
            "kaggle",
            job,
            root / pri["spec"],
            pri["nparts"],
            ids,
            selftest=True,
            root=root,
        )
        rc, tail = self.launch(request)
        self.event(
            name,
            "priority-launch",
            {**info, "job": job, "spec": pri["spec"], "rc": rc, "tail": tail[-300:]},
        )
        if rc:
            return None
        queue.jobs.append(
            {
                "job": job,
                "platform": "kaggle",
                "parts": ids,
                "spec": pri["spec"],
                "nparts": pri["nparts"],
                "utc": utc(self.clock.now()),
            }
        )
        queue.priority = queue.priority[1:]
        self.save(name, queue)
        return "priority"

    def _requeue(
        self, name: str, queue: Queue, info: dict[str, Any], platform: plan.Platform
    ) -> Outcome:
        kaggle = platform == "kaggle"
        ids = queue.requeue[: queue.kaggle_chunk if kaggle else queue.ssh_chunk(platform)]
        job = self.job_name(name, queue, "requeue")
        self.save(name, queue)
        rc, tail = self.launch(self.request(name, queue, Chunk(platform, ids, job), requeue=True))
        self.event(
            name,
            "requeue-launch",
            {**info, "job": job, "parts": ids, "rc": rc, "tail": tail[-300:]},
        )
        if rc:
            return None
        queue.jobs.append(
            {
                "job": job,
                "platform": platform,
                "parts": ids,
                "requeue": True,
                "utc": utc(self.clock.now()),
            }
        )
        queue.requeue = [i for i in queue.requeue if i not in ids]
        self.save(name, queue)
        return platform

    def _chunk(
        self, name: str, queue: Queue, info: dict[str, Any], lane: plan.Platform, chunk: int
    ) -> Outcome:
        ids = list(range(queue.next, min(queue.next + chunk, queue.end)))
        job = self.job_name(name, queue, lane)
        self.save(name, queue)
        rc, tail = self.launch(self.request(name, queue, Chunk(lane, ids, job)))
        self.event(
            name,
            "launch",
            {
                **info,
                "platform": lane,
                "job": job,
                "parts": [ids[0], ids[-1]],
                "rc": rc,
                "tail": tail[-300:] if rc == 0 else tail,
            },
        )
        if rc and lane == "colab" and COLAB_REFUSED in tail:
            queue.colab_backoff_until = self.clock.now() + COLAB_BACKOFF_S
            self.save(name, queue)
        if rc and lane in plan.SSH_PLATFORMS and VPS_REFUSED in tail:
            queue.back_off(lane, self.clock.now() + VPS_BACKOFF_S)
            self.save(name, queue)
        if rc:
            return None
        queue.jobs.append(
            {"job": job, "platform": lane, "parts": ids, "utc": utc(self.clock.now())}
        )
        queue.next = ids[-1] + 1
        self.save(name, queue)
        return lane

    def arm(self, name: str, every_min: int = 15) -> str:
        """Registers the recurring dispatch tick.

        Raises:
            ValueError: If the dispatcher has no scheduler.
        """
        if self.scheduler is None:
            raise ValueError("no scheduler")
        out = self.scheduler.schedule(
            schedule.task_name("dispatch", name),
            schedule.tick_argv("dispatch", name),
            every_min,
            self.qdir(name),
        )
        self.event(name, "scheduled", {"every_min": every_min})
        return out

    def unschedule(self, name: str) -> bool:
        """Removes the recurring dispatch tick; True when it is gone."""
        if self.scheduler is None:
            return False
        gone = self.scheduler.unschedule(schedule.task_name("dispatch", name))
        self.event(name, "unscheduled", {"gone": gone})
        return gone

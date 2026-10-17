import dataclasses
import io
import json
import tarfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from oran_causal.experiments.launch import colab, colab_session
from oran_causal.experiments.launch.runner import Dirs
from oran_causal.experiments.launch.schedule import task_name

SESSION_ROW = "[job-x] m-s-1 | Hardware: CPU | Shape: Standard | Variant: DEFAULT"


class FakeScheduler:
    def __init__(self):
        self.tasks: dict[str, tuple] = {}

    def schedule(self, task, argv, every_min, workdir):
        self.tasks[task] = (tuple(argv), every_min)
        return f"scheduled {task}"

    def unschedule(self, task):
        self.tasks.pop(task, None)
        return True


class FakeColab:
    """A scripted `colab` CLI: one session, tick answers taken from `ticks`."""

    def __init__(self):
        self.calls: list[tuple[str, ...]] = []
        self.session = False
        self.ticks: list[dict | Exception] = []
        self.pack: dict[str, bytes] = {"outdir/res_0.jsonl": b"{}\n", "exit_code": b"0"}
        self.execs: list[str] = []
        self.setup_error: Exception | None = None
        self.download_error: Exception | None = None

    def __call__(self, argv: Sequence[str], timeout: float, check: bool) -> str:
        self.calls.append(tuple(argv))
        verb = argv[0]
        if verb == "usage":
            return f"Active assignments: {int(self.session)}"
        if verb == "new":
            self.session = True
            return "READY"
        if verb == "sessions":
            return SESSION_ROW if self.session else "[colab] No active sessions found on server."
        if verb == "stop":
            self.session = False
            return "stopped"
        if verb == "upload":
            return "uploaded"
        if verb == "download":
            if self.download_error is not None:
                raise self.download_error
            self.write_tar(Path(argv[-1]))
            return "downloaded"
        if verb == "exec":
            code = Path(argv[argv.index("-f") + 1]).read_text()
            self.execs.append(code)
            return self.exec_answer(code)
        raise AssertionError(argv)

    def write_tar(self, path: Path) -> None:
        with tarfile.open(path, "w:gz") as tf:
            for name, data in self.pack.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    def exec_answer(self, code: str) -> str:
        if self.setup_error is not None:
            raise self.setup_error
        if "JOBSTART" in code:
            return 'JOBSTART {"python": "3.14"}'
        if "JOBLAUNCHED" in code:
            return 'JOBLAUNCHED {"pid": 7, "alive_5s": true, "log_head": ""}'
        step = self.ticks.pop(0)
        if isinstance(step, Exception):
            raise step
        return "JOBTICK " + json.dumps(step)

    def probe(self, name: str, ping: bool) -> dict:
        return {"alive": self.session, "known": self.session, "n_assignments": int(self.session)}


def tick_state(exit_code=None, alive=True) -> dict:
    return {
        "alive": alive,
        "exit_code": exit_code,
        "elapsed_s": 60,
        "log_tail": ["a", "b", "c", "d"],
        "uptime_s": 120,
        "out_files": ["res_0.jsonl"],
    }


@pytest.fixture
def world(tmp_path: Path, launch_kit):
    cli = FakeColab()
    clock = launch_kit.FakeClock()
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    sched = FakeScheduler()
    logs: list[str] = []
    env = colab.ColabEnv(
        colab_session.Colab(cli, cli.probe, clock.clock()),
        Dirs(tmp_path / "work", tmp_path / "runs", ws),
        sched,
        launch_kit.fake_git(),
        alive=lambda pid: False,
        log=logs.append,
    )
    return env, cli, sched, logs


JOB = colab.ColabJob("job-x", "bash run.sh", out_dir="xm_out", every_min=5)


def test_launch_uploads_sets_up_starts_and_schedules(world):
    env, cli, sched, _ = world
    meta = colab.launch_job(JOB, env)
    verbs = [c[0] for c in cli.calls]
    assert verbs[:2] == ["usage", "new"]
    assert "upload" in verbs
    assert len(cli.execs) == 2
    assert "bash run.sh" in cli.execs[1]
    assert meta["launched"]["pid"] == 7
    assert sched.tasks[task_name("colab", "job-x")][1] == 5
    assert json.loads((env.run_dir("job-x") / "colab.json").read_text())["git_head"]


def test_a_second_launch_under_the_same_name_is_refused(world):
    env, *_ = world
    colab.launch_job(JOB, env)
    with pytest.raises(FileExistsError):
        colab.launch_job(JOB, env)


def test_a_failed_setup_stops_the_runtime_and_is_recorded(world):
    env, cli, sched, _ = world
    cli.setup_error = colab_session.ColabError("boom")
    with pytest.raises(colab_session.ColabError):
        colab.launch_job(JOB, env)
    assert not cli.session
    assert "error" in env.load("job-x")
    assert not sched.tasks


@pytest.mark.parametrize("bad", ["/abs", "../up", "a/../b", "", "exit_code"])
def test_out_dir_must_stay_inside_the_bundle(bad: str):
    with pytest.raises(ValueError, match="out_dir"):
        colab.ColabJob("n", "c", out_dir=bad)


def test_ticks_keep_running_then_finish_and_stop_the_runtime(world):
    env, cli, sched, _ = world
    colab.launch_job(JOB, env)
    cli.ticks = [tick_state(), tick_state(exit_code=0)]
    assert colab.tick("job-x", env) == "running"
    assert (env.run_dir("job-x") / "xm_out" / "res_0.jsonl").exists()
    assert sched.tasks
    assert colab.tick("job-x", env) == "finished"
    assert not cli.session
    assert not sched.tasks
    meta = env.load("job-x")
    assert meta["exit_code"] == 0
    assert meta["finished_utc"]
    assert colab.tick("job-x", env) == "noop"


def test_three_failed_execs_stop_the_runtime_and_mark_the_job_lost(world):
    env, cli, sched, _ = world
    colab.launch_job(JOB, env)
    cli.ticks = [colab_session.ColabError("x")] * 3
    assert [colab.tick("job-x", env) for _ in range(3)] == ["exec-error", "exec-error", "lost"]
    meta = env.load("job-x")
    assert "lost_utc" in meta
    assert not cli.session
    assert not sched.tasks


def test_a_good_tick_resets_the_failure_count(world):
    env, cli, _, _ = world
    colab.launch_job(JOB, env)
    cli.ticks = [colab_session.ColabError("x"), tick_state(), colab_session.ColabError("x")]
    for _ in range(3):
        colab.tick("job-x", env)
    assert env.load("job-x")["exec_failures"] == 1


def test_a_vanished_runtime_is_lost_unless_the_exit_code_was_pulled(world):
    env, cli, sched, _ = world
    colab.launch_job(JOB, env)
    cli.session = False
    assert colab.tick("job-x", env) == "lost"
    assert not sched.tasks
    assert "lost_utc" in env.load("job-x")


def test_a_vanished_runtime_after_the_exit_code_is_finished(world):
    env, cli, _, _ = world
    colab.launch_job(JOB, env)
    (env.run_dir("job-x") / "exit_code").write_text("0\n")
    cli.session = False
    assert colab.tick("job-x", env) == "finished"
    assert env.load("job-x")["exit_code"] == 0


def test_a_probe_error_changes_nothing(world):
    env, _cli, sched, _ = world
    colab.launch_job(JOB, env)

    def broken(name, ping):
        raise colab_session.ColabError("net")

    object.__setattr__(env.colab, "probe", broken)
    assert colab.tick("job-x", env) == "probe-error"
    assert sched.tasks


def test_the_tick_lock_excludes_a_live_holder_and_reclaims_a_dead_one(tmp_path: Path):
    path = tmp_path / "tick.lock"
    with colab.TickLock(path, lambda pid: True) as first:
        assert first
        with colab.TickLock(path, lambda pid: True) as second:
            assert not second
    assert not path.exists()
    path.write_text(json.dumps({"pid": 1, "t": 0}))
    with colab.TickLock(path, lambda pid: False) as reclaimed:
        assert reclaimed


def test_an_old_lock_is_stale_even_if_its_pid_lives(tmp_path: Path):
    path = tmp_path / "tick.lock"
    path.write_text(json.dumps({"pid": 1, "t": 0}))
    with colab.TickLock(path, lambda pid: True, now=lambda: colab.LOCK_STALE_S + 1) as got:
        assert got


def test_stop_disarms_and_reports_other_assignments(world):
    env, cli, sched, _ = world
    colab.launch_job(JOB, env)
    assert colab.stop("job-x", env) == 0
    assert not sched.tasks
    assert not cli.session


def tick_events(env) -> list[tuple[str, dict]]:
    lines = (env.run_dir("job-x") / "tick.log").read_text().splitlines()
    return [(ln.split(" ", 2)[1], json.loads(ln.split(" ", 2)[2])) for ln in lines]


def test_a_reclaimed_runtime_fails_the_download_then_the_job_is_lost(world):
    env, cli, sched, _ = world
    colab.launch_job(JOB, env)
    cli.ticks = [tick_state(exit_code=0, alive=False)]
    cli.download_error = colab_session.ColabError(
        "colab download -s job-x rc=1: File or directory not found: " + colab_session.JOB_PULL
    )
    assert colab.tick("job-x", env) == "exec-error"
    event, info = tick_events(env)[-1]
    assert event == "exec-error"
    assert info["step"] == "download"
    assert info["state"]["exit_code"] == 0
    assert sched.tasks
    cli.session = False
    assert colab.tick("job-x", env) == "lost"
    assert env.load("job-x")["lost_reason"] == "session gone before the exit code was pulled"
    assert not sched.tasks


def test_a_failed_exec_is_logged_without_a_state(world):
    env, cli, _, _ = world
    colab.launch_job(JOB, env)
    cli.ticks = [colab_session.ColabError("x")]
    colab.tick("job-x", env)
    _, info = tick_events(env)[-1]
    assert (info["step"], info["state"]) == ("exec", None)


@pytest.mark.parametrize("every", [0, colab.IDLE_RECLAIM_MIN, 15])
def test_a_tick_slower_than_the_idle_reclaim_is_refused(world, every: int):
    env, *_ = world
    with pytest.raises(ValueError, match="reclaimed"):
        colab.ColabJob("n", "c", every_min=every)
    with pytest.raises(ValueError, match="reclaimed"):
        env.arm("job-x", every)


def test_the_default_tick_beats_the_idle_reclaim(world):
    env, _, sched, _ = world
    colab.launch_job(dataclasses.replace(JOB, every_min=colab.TICK_EVERY_MIN), env)
    assert sched.tasks[task_name("colab", "job-x")][1] == colab.TICK_EVERY_MIN
    assert 2 * colab.TICK_EVERY_MIN <= colab.IDLE_RECLAIM_MIN
    assert colab.ColabJob("n", "c").every_min == colab.TICK_EVERY_MIN


def test_assignment_rows_skip_cli_messages_and_keep_orphans():
    out = "\n".join(
        [
            "[colab] Pruned 1 stale local session(s).",
            SESSION_ROW,
            "[?] m-s-2 | Hardware: CPU | Shape: Standard | Variant: DEFAULT",
        ]
    )
    assert colab_session.assignment_rows(out) == ["job-x", "?"]
    assert colab_session.assignment_rows("[colab] No active sessions found on server.") == []
    with pytest.raises(colab_session.ColabError):
        colab_session.assignment_rows("[colab] Error: 503")


def test_gone_ignores_the_lagging_usage_count(launch_kit):
    answers = {
        "usage": "Active assignments: 1",
        "sessions": "[colab] No active sessions found on server.",
    }
    clock = launch_kit.FakeClock().clock()
    c = colab_session.Colab(lambda argv, t, check: answers[argv[0]], clock=clock)
    assert c.gone("job-x") == (True, 0)
    answers["sessions"] = SESSION_ROW
    assert c.gone("job-x") == (False, 0)
    assert c.gone("other") == (True, 1)

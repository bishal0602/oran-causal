import json
import shutil
from pathlib import Path

import pytest

from oran_causal.experiments.campaign.shard import partition
from oran_causal.experiments.campaign.spec import expand
from oran_causal.experiments.launch import __main__ as cli
from oran_causal.experiments.launch import dispatch, plan
from oran_causal.experiments.launch.runner import Clock, Dirs
from oran_causal.experiments.paths import WORKSPACE_ROOT

SMOKE = "packages/experiments/specs/dev/smoke.json"
EVAL = "packages/experiments/specs/eval/full.json"


class DryScheduler:
    def __init__(self):
        self.tasks: dict[str, tuple] = {}

    def schedule(self, task, argv, every_min, workdir):
        self.tasks[task] = (tuple(argv), every_min)
        return f"scheduled {task}"

    def unschedule(self, task):
        self.tasks.pop(task, None)
        return True


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "packages/experiments/specs/dev").mkdir(parents=True)
    shutil.copy(WORKSPACE_ROOT / SMOKE, root / SMOKE)
    (root / "cost.json").write_text("{}")
    return root


def request(ws: Path, platform: plan.Platform = "kaggle", **kw) -> plan.LaunchRequest:
    kw.setdefault("parts", 4)
    return plan.LaunchRequest(platform, "job-a", ws / SMOKE, root=ws, **kw)


def make(ws: Path, platform: plan.Platform = "kaggle", launch_kit=None, **kw) -> plan.SessionPlan:
    spec = json.loads((ws / SMOKE).read_text())
    git = launch_kit.fake_git() if launch_kit else None
    return plan.make_plan(request(ws, platform, **kw), spec, plan.launcher_texts(ws), git)


def test_a_dev_kaggle_plan_sets_up_then_runs_every_part(ws: Path, launch_kit):
    p = make(ws, "kaggle", launch_kit, part_ids=[0, 1])
    assert p.out_dir == "$JOB_OUT"
    assert p.extra == (SMOKE,)
    assert p.frozen_protocol is None
    assert not p.eval
    assert p.cmd.index("uv sync --frozen") < p.cmd.index("--part 0/4")
    assert p.cmd.count("--part ") == 2
    assert "--wall-s 39600" in p.cmd or "39600" in p.cmd
    assert p.procs == 2


def test_the_platform_default_wall_is_the_profiles(ws: Path, launch_kit):
    colab = make(ws, "colab", launch_kit, part_ids=[0])
    assert str(3 * 3600) in colab.cmd
    assert colab.out_dir == plan.COLAB_OUT
    vps = make(ws, "vps", launch_kit, part_ids=[0])
    assert "uv sync" not in vps.cmd
    assert vps.out_dir == "{OUT}"
    assert str(23 * 3600) in vps.cmd


def test_vps_refuses_more_parts_than_processes(ws: Path, launch_kit):
    with pytest.raises(plan.LaunchRefused, match="one process per part"):
        make(ws, "vps", launch_kit, parts=8, procs=2)


def test_procs_beyond_the_profile_are_refused(ws: Path, launch_kit):
    with pytest.raises(ValueError, match=r"--procs must be 1\.\.7"):
        make(ws, "vps", launch_kit, part_ids=[0], procs=9)


def test_paths_and_skip_globs_join_the_bundle(ws: Path, launch_kit):
    (ws / "d").mkdir()
    (ws / "d" / "a.jsonl").write_text("{}\n")
    (ws / "d" / "b.jsonl").write_text("{}\n")
    p = make(ws, "kaggle", launch_kit, part_ids=[0], paths=["d/a.jsonl"], skip_from=["d/b*.jsonl"])
    assert p.extra == (SMOKE, "d/a.jsonl", "d/b.jsonl")
    assert "--skip-complete-from 'd/b*.jsonl'" in p.cmd
    with pytest.raises(FileNotFoundError):
        make(ws, "kaggle", launch_kit, part_ids=[0], paths=["nothing/*.x"])


def test_selftest_bundles_the_tests_for_a_cloud_session(ws: Path, launch_kit):
    p = make(ws, "kaggle", launch_kit, part_ids=[0], selftest=True)
    assert "packages/experiments/tests" in p.extra
    assert "packages/trace/src" in p.extra
    assert "pytest" in p.cmd


def test_an_eval_spec_is_refused_without_authorisation(ws: Path, launch_kit):
    spec = json.loads((ws / SMOKE).read_text())
    spec["blocks"][0]["seeds"] = [3100000]
    (ws / SMOKE).write_text(json.dumps(spec))
    with pytest.raises(plan.LaunchRefused, match="EVAL refused"):
        make(ws, "kaggle", launch_kit, part_ids=[0])


def test_describe_lists_the_steps_of_each_platform(ws: Path, launch_kit):
    k = "\n".join(plan.describe(make(ws, "kaggle", launch_kit, part_ids=[0])))
    assert "kaggle datasets create job-a-code" in k
    c = "\n".join(plan.describe(make(ws, "colab", launch_kit, part_ids=[0], gpu=True)))
    assert "colab new job-a (gpu:T4)" in c
    v = "\n".join(plan.describe(make(ws, "vps", launch_kit, part_ids=[0, 1])))
    assert "vps launch job-a --procs 2" in v
    assert "CPUQuota=200%" in v


def test_cli_dry_run_prints_the_plan_and_starts_nothing(capsys, launch_kit):
    lines: list[str] = []

    def no_targets():
        raise AssertionError("a dry run must not build clients")

    rc = cli.main(
        [
            "launch",
            "kaggle",
            "--name",
            "x",
            "--spec",
            SMOKE,
            "--parts",
            "4",
            "--part-set",
            "0,1",
            "--dry-run",
        ],
        no_targets,
        lines.append,
    )
    assert rc == 0
    text = "\n".join(lines)
    assert "kaggle datasets create x-code" in text
    assert "--part 0/4" in text


def test_cli_reports_a_refusal_with_exit_code_2():
    lines: list[str] = []

    def no_targets():
        raise AssertionError("a refusal must not build clients")

    rc = cli.main(
        ["launch", "vps", "--name", "x", "--spec", SMOKE, "--parts", "9", "--dry-run"],
        no_targets,
        lines.append,
    )
    assert rc == 2
    assert "one process per part" in lines[0]


class FakeBackend:
    def __init__(self):
        self.launched: list[plan.LaunchRequest] = []
        self.kaggle = (0, 0)
        self.kaggle_states: dict[str, str] = {}
        self.pulled: list[str] = []
        self.fail: dict[str, str] = {}

    def kaggle_counts(self, prefixes):
        return self.kaggle

    def kaggle_state(self, job):
        return self.kaggle_states.get(job, 'Status "complete"')

    def kaggle_pull(self, job):
        self.pulled.append(job)

    def vps_pull(self, job):
        self.pulled.append(job)

    def launch(self, request):
        if request.platform in self.fail:
            raise RuntimeError(self.fail[request.platform])
        self.launched.append(request)


def dispatcher(ws: Path, tmp_path: Path, **kw):
    backend = FakeBackend()
    clock = Clock(now=lambda: 1_700_000_000.0, sleep=lambda s: None)
    sched = DryScheduler()
    d = dispatch.Dispatcher(
        Dirs(tmp_path / "w", ws / "records" / "runs", ws), backend, clock, sched, **kw
    )
    return d, backend, sched


def queue(**kw) -> dispatch.Queue:
    return dispatch.Queue(SMOKE, "cost.json", kw.pop("parts", 20), **kw)


def test_a_tick_fills_every_free_kaggle_slot_then_stops(ws: Path, tmp_path: Path):
    d, backend, _ = dispatcher(ws, tmp_path)
    d.init("q", queue(parts=100))
    assert d.tick("q") == "colab"
    assert [r.platform for r in backend.launched] == ["kaggle"] * 4 + ["colab"]
    assert [len(r.part_ids) for r in backend.launched] == [8, 8, 8, 8, 4]
    q = d.load("q")
    assert q.next == 36
    assert len(q.jobs) == 5


def test_the_next_lane_is_used_when_kaggle_is_full(ws: Path, tmp_path: Path):
    d, backend, _ = dispatcher(ws, tmp_path)
    backend.kaggle = (5, 0)
    d.init("q", queue(vps=True))
    assert d.tick("q") == "vps"
    assert backend.launched[0].procs == 7
    assert d.tick("q") == "colab"
    assert len(backend.launched[1].part_ids) == dispatch.COLAB_CHUNK
    assert d.tick("q") == "colab"
    assert [j["platform"] for j in d.load("q").jobs] == ["vps", "colab", "colab"]


def test_a_refused_lane_backs_off(ws: Path, tmp_path: Path):
    d, backend, _ = dispatcher(ws, tmp_path)
    backend.kaggle = (5, 0)
    backend.fail = {"vps": "refused: load", "colab": "Allocation refused"}
    d.init("q", queue(vps=True))
    assert d.tick("q") == "none"
    assert d.load("q").vps_backoff_until == 1_700_000_000.0 + dispatch.VPS_BACKOFF_S
    assert d.tick("q") == "none"
    q = d.load("q")
    assert q.colab_backoff_until == 1_700_000_000.0 + dispatch.COLAB_BACKOFF_S
    assert q.next == 0
    backend.launched.clear()
    assert d.tick("q") == "none"
    assert not backend.launched


def test_only_restricts_the_lanes(ws: Path, tmp_path: Path):
    d, backend, _ = dispatcher(ws, tmp_path)
    backend.kaggle = (5, 0)
    d.init("q", queue(only=["kaggle"]))
    assert d.tick("q") == "none"


def test_the_tick_lock_skips_a_concurrent_tick(ws: Path, tmp_path: Path):
    d, backend, _ = dispatcher(ws, tmp_path)
    d.init("q", queue())
    with dispatch.DispatchLock(d.qdir("q") / "dispatch.lock") as held:
        assert held
        assert d.tick("q") == "skipped"
    assert not backend.launched


def write_results(d: dispatch.Dispatcher, job: str, keys: list[str]) -> None:
    out = d.dirs.runs / job / "out"
    out.mkdir(parents=True)
    rows = [{"key": k, "status": "ok", "run_mode": "DEV"} for k in keys]
    (out / "res_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_an_incomplete_part_is_requeued_up_to_three_attempts(ws: Path, tmp_path: Path):
    d, backend, _sched = dispatcher(ws, tmp_path)
    d.init("q", queue(parts=4, only=["kaggle"]))
    d.arm("q", 15)
    d.tick("q")
    job = d.load("q").jobs[0]["job"]
    units = partition(expand(json.loads((ws / SMOKE).read_text())), 4, {})
    write_results(d, job, [u.key for i in (0, 1, 2) for u in units[i]] + [units[3][0].key])
    d.tick("q")
    q = d.load("q")
    assert q.jobs[0]["incomplete_parts"] == [3]
    assert backend.launched[-1].part_ids == [3]
    assert any("done" in g for g in backend.launched[-1].skip_from)
    assert q.requeue == []


def test_a_finished_queue_removes_its_schedule(ws: Path, tmp_path: Path):
    d, _backend, sched = dispatcher(ws, tmp_path)
    d.init("q", queue(parts=2, only=["kaggle"]))
    d.arm("q", 15)
    d.tick("q")
    job = d.load("q").jobs[0]["job"]
    units = partition(expand(json.loads((ws / SMOKE).read_text())), 2, {})
    write_results(d, job, [u.key for part in units for u in part])
    assert d.tick("q") == "done"
    assert not sched.tasks


def test_a_running_kaggle_job_is_not_pulled(ws: Path, tmp_path: Path):
    d, backend, _ = dispatcher(ws, tmp_path)
    d.init("q", queue(parts=2, only=["kaggle"]))
    d.tick("q")
    job = d.load("q").jobs[0]["job"]
    backend.kaggle_states[job] = 'Status "RUNNING"'
    d.tick("q")
    assert not backend.pulled


def test_dry_run_changes_nothing(ws: Path, tmp_path: Path):
    lines: list[str] = []
    d, backend, _ = dispatcher(ws, tmp_path, dry_run=True, log=lines.append)
    d.init("q", queue())
    d.tick("q")
    assert not backend.launched
    assert any("[dry-run] kaggle" in line for line in lines)

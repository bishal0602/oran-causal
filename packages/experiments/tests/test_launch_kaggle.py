import json
import zipfile
from pathlib import Path

import pytest

from oran_causal.experiments.launch import kaggle
from oran_causal.experiments.launch.runner import Dirs


class FakeCli:
    def __init__(self, answers: dict[str, list[str]] | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.answers = answers or {}

    def __call__(self, argv, cwd, timeout):
        key = " ".join(argv[1:3])
        self.calls.append((key, *argv[3:]))
        queue = self.answers.get(key)
        if queue:
            return queue.pop(0) if len(queue) > 1 else queue[0]
        return "ready" if "status" in key else "ok"


def client(cli: FakeCli, clock) -> kaggle.KaggleClient:
    return kaggle.KaggleClient(runner=cli, executable="kaggle", owner="me", clock=clock.clock())


def test_stage_writes_dataset_and_kernel(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    job = kaggle.KaggleJob("j1", "bash run.sh", sources=("me/other",), gpu=True)
    kaggle.stage(job, tmp_path / "out", ws, launch_kit.fake_git(), owner="me")
    ds, kd = tmp_path / "out" / "dataset", tmp_path / "out" / "kernel"
    meta = json.loads((ds / "dataset-metadata.json").read_text())
    assert meta["id"] == "me/j1-code"
    km = json.loads((kd / "kernel-metadata.json").read_text())
    assert km["enable_gpu"] is True
    assert km["kernel_sources"] == ["me/other"]
    assert km["dataset_sources"] == ["me/j1-code"]
    src = (kd / "kernel.py").read_text()
    assert "'j1'" in src
    assert "'bash run.sh'" in src
    assert "__NAME__" not in src
    names = zipfile.ZipFile(ds / "j1_bundle.zip").namelist()
    assert "MANIFEST.json" in names
    assert "uv.lock" in names
    assert "_job_j1.json" in names


def test_launch_creates_waits_pushes_and_records(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    cli, clock = FakeCli({"datasets status": ["pending", "ready"]}), launch_kit.FakeClock()
    dirs = Dirs(tmp_path / "work", tmp_path / "runs", ws)
    rec = kaggle.launch(
        kaggle.KaggleJob("j2", "echo hi"), client(cli, clock), dirs, launch_kit.fake_git()
    )
    keys = [c[0] for c in cli.calls]
    assert keys == ["datasets create", "datasets status", "datasets status", "kernels push"]
    assert clock.slept == [10]
    assert rec["git_head"] == "f" * 40
    assert json.loads((dirs.runs / "j2" / "launch.json").read_text())["name"] == "j2"


def test_push_retries_while_the_session_limit_blocks(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    blocked = "Maximum batch GPU session count reached"
    cli, clock = FakeCli({"kernels push": [blocked, blocked, "pushed"]}), launch_kit.FakeClock()
    dirs = Dirs(tmp_path / "work", tmp_path / "runs", ws)
    kaggle.launch(kaggle.KaggleJob("j3", "x"), client(cli, clock), dirs, launch_kit.fake_git())
    assert [c[0] for c in cli.calls].count("kernels push") == 3
    assert clock.slept == [60, 60]


def test_push_gives_up_after_wait_min(launch_kit):
    cli = FakeCli({"kernels push": ["Maximum batch session limit"]})
    clock = launch_kit.FakeClock()
    with pytest.raises(TimeoutError, match="blocked"):
        client(cli, clock).push_kernel(Path("."), wait_min=3)


def test_dataset_error_stops_the_launch(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    cli = FakeCli({"datasets create": ["Error: slug taken"]})
    dirs = Dirs(tmp_path / "work", tmp_path / "runs", ws)
    with pytest.raises(RuntimeError, match="dataset create failed"):
        kaggle.launch(
            kaggle.KaggleJob("j4", "x"),
            client(cli, launch_kit.FakeClock()),
            dirs,
            launch_kit.fake_git(),
        )
    assert [c[0] for c in cli.calls] == ["datasets create"]


def test_dataset_wait_times_out(launch_kit):
    cli = FakeCli({"datasets status": ["pending"]})
    with pytest.raises(TimeoutError, match="not ready"):
        client(cli, launch_kit.FakeClock()).wait_dataset("j5")


def test_pull_reads_job_json_and_skips_npz_by_default(tmp_path: Path):
    seen: list[list[str]] = []

    def runner(argv, cwd, timeout):
        dst = Path(cwd) / argv[argv.index("-p") + 1]
        (dst / "job.json").write_text(json.dumps({"rc": 0}))
        seen.append(list(argv))
        return ""

    cl = kaggle.KaggleClient(runner=runner, executable="kaggle", owner="me")
    assert kaggle.pull("j6", cl, tmp_path) == {"rc": 0}
    assert seen[-1][seen[-1].index("--file-pattern") + 1] == kaggle.DEFAULT_PULL
    kaggle.pull("j6", cl, tmp_path, pattern=None)
    assert "--file-pattern" not in seen[-1]

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from oran_causal.experiments.launch import bundle, hosts, vps
from oran_causal.experiments.launch.commands import SetupPlan
from oran_causal.experiments.launch.runner import Dirs

GB = 1024 * 1024
STATE = {"load1": 0.4, "mem_available_kb": 20 * GB, "our_procs": 0, "units": ""}


class FakeTransport:
    def __init__(self, state: dict | None = None):
        self.state = dict(STATE, **(state or {}))
        self.scripts: list[str] = []
        self.puts: list[tuple[bytes, str]] = []
        self.fetched: list[str] = []
        self.tar = b""

    def script(self, body: str, timeout: int = 600, *, check: bool = True) -> str:
        self.scripts.append(body)
        return json.dumps(self.state) if "/proc/loadavg" in body else "ok"

    def put(self, data: bytes, remote_cmd: str, timeout: int = 1800) -> str:
        self.puts.append((data, remote_cmd))
        return "pushed 9 files"

    def fetch(self, remote_cmd: str, timeout: int = 1800) -> bytes:
        self.fetched.append(remote_cmd)
        return self.tar


def host(state: dict | None = None) -> tuple[vps.Vps, FakeTransport]:
    t = FakeTransport(state)
    return vps.Vps(t, hosts.VPS_SHARED), t


def request(ws: Path) -> bundle.BundleRequest:
    return bundle.BundleRequest("job-a", ws)


def test_state_parses_the_probe():
    v, _ = host({"units": "oran-causal-x.scope", "our_procs": 3, "load1": 3.5})
    s = v.state()
    assert s.units == ("oran-causal-x.scope",)
    assert s.other_load == 0.5


@pytest.mark.parametrize(
    ("state", "msg"),
    [
        ({"units": "oran-causal-x.scope"}, "one job at a time"),
        ({"mem_available_kb": 8 * GB}, "MemAvailable"),
        ({"load1": 2.5}, "load from others"),
    ],
)
def test_the_shared_profile_refuses(state: dict, msg: str):
    v, _ = host(state)
    with pytest.raises(vps.Refused, match=msg):
        vps.check_launch(v.profile, v.state())


def test_the_dedicated_profile_never_refuses():
    t = FakeTransport({"load1": 9.0, "units": "oran-causal-x.scope", "mem_available_kb": 1})
    v = vps.Vps(t, hosts.VPS_DEDICATED)
    vps.check_launch(v.profile, v.state())


def test_a_cleared_load_is_not_refused():
    v, _ = host({"load1": 7.9, "our_procs": 7})
    vps.check_launch(v.profile, v.state())


def test_job_names_are_validated():
    v, _ = host()
    assert v.unit("job-1") == "oran-causal-job-1"
    for bad in ("A", "-x", "a b", "a;rm"):
        with pytest.raises(ValueError, match="bad job name"):
            v.job_dir(bad)


def test_the_run_script_substitutes_out_and_procs():
    v, _ = host()
    sh = v.run_script("job-a", "run --out {OUT} --procs {PROCS}", 4)
    jd = "/opt/oran-causal/jobs/job-a"
    assert f"run --out {jd}/out --procs 4" in sh
    assert "{OUT}" not in sh
    assert f"mv {jd}/exit_code.tmp {jd}/exit_code" in sh
    assert "NUMBA_NUM_THREADS=1" in sh
    assert "TMPDIR=/opt/oran-causal/tmp" in sh


def test_the_launch_script_runs_in_a_capped_scope():
    v, _ = host()
    sh = v.launch_script("job-a", "#!/bin/bash\necho x", 5)
    assert "systemd-run --scope --quiet --unit oran-causal-job-a" in sh
    assert "-p CPUQuota=500%" in sh
    assert "-p MemoryMax=11G" in sh
    assert "-p MemorySwapMax=0" in sh
    assert "systemctl is-active oran-causal-job-a.scope" in sh


def test_the_venv_step_syncs_the_lock_and_checks_the_interpreter():
    v, _ = host()
    sh = v.venv_script("job-a", SetupPlan("/o", gpu=False, torch=True), "3.14")
    assert "UV_PROJECT_ENVIRONMENT=/opt/oran-causal/jobs/job-a/venv" in sh
    assert "uv sync --frozen" in sh
    assert 'case "$v" in "Python 3.14"*' in sh
    assert "pin_check.json" in sh


def test_procs_beyond_the_cap_are_refused(tmp_path: Path):
    v, t = host()
    with pytest.raises(ValueError, match=r"--procs must be 1\.\.7"):
        v.launch("job-a", "x", 8, Dirs(tmp_path, tmp_path, tmp_path))
    assert not t.scripts


def test_launch_needs_a_push_of_the_clean_current_commit(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    dirs = Dirs(tmp_path / "w", tmp_path / "runs", ws)
    v, t = host()
    with pytest.raises(vps.Refused, match="push first"):
        v.launch("job-a", "x", 2, dirs, launch_kit.fake_git())
    v.push("job-a", request(ws), dirs, launch_kit.fake_git())
    with pytest.raises(vps.Refused, match="clean current commit"):
        v.launch("job-a", "x", 2, dirs, launch_kit.fake_git(" M x\n"))
    assert not any("systemd-run" in s for s in t.scripts)


def test_push_then_launch_records_both(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    dirs = Dirs(tmp_path / "w", tmp_path / "runs", ws)
    v, t = host()
    pushed = v.push("job-a", request(ws), dirs, launch_kit.fake_git())
    data, cmd = t.puts[0]
    assert "tar -xzf - -C /opt/oran-causal/jobs/job-a/b" in cmd
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        names = tf.getnames()
    assert "MANIFEST.json" in names
    assert pushed["commit"] == launch_kit.HEAD
    rec = v.launch("job-a", "echo {OUT}", 3, dirs, launch_kit.fake_git())
    assert rec["limits"] == ["-p", "CPUQuota=300%", "-p", "MemoryMax=11G", "-p", "MemorySwapMax=0"]
    assert json.loads((dirs.runs / "job-a" / "launch.json").read_text())["procs"] == 3
    assert "systemd-run" in t.scripts[-1]


def tar_of(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_pull_keeps_the_larger_local_file_and_ignores_escapes(tmp_path: Path):
    v, t = host()
    dirs = Dirs(tmp_path, tmp_path / "runs", tmp_path)
    dst = dirs.runs / "job-a" / "out"
    dst.mkdir(parents=True)
    (dst / "res_0.jsonl").write_bytes(b"1\n2\n3\n")
    t.tar = tar_of({"out/res_0.jsonl": b"1\n", "out/res_1.jsonl": b"1\n", "../escape.txt": b"x"})
    assert v.pull("job-a", dirs) == 1
    assert (dst / "res_0.jsonl").read_bytes() == b"1\n2\n3\n"
    assert (dst / "res_1.jsonl").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_ssh_runs_scripts_on_stdin_and_raises_on_failure():
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw["input"]))
        return subprocess.CompletedProcess(argv, 3, b"", b"boom")

    ssh = vps.Ssh("export A=1", host="h", run=run)
    with pytest.raises(vps.RemoteError, match="rc 3"):
        ssh.script("echo hi")
    argv, data = calls[0]
    assert argv[0] == "ssh"
    assert argv[-3:] == ["h", "bash", "-s"]
    assert data == b"set -eu\nexport A=1\necho hi\n"
    assert ssh.script("echo hi", check=False).endswith("boom")


def test_status_and_stop_touch_only_the_jobs_own_scope():
    v, t = host()
    v.status("job-a")
    v.stop("job-a")
    assert "systemctl stop oran-causal-job-a.scope" in t.scripts[-1]
    assert "pkill" not in "".join(t.scripts)


def test_instabox_scopes_run_in_the_users_systemd_instance():
    t = FakeTransport()
    v = vps.Vps(t, hosts.INSTABOX)
    sh = v.launch_script("job-a", "#!/bin/bash\necho x", 4)
    assert "systemd-run --user --scope --quiet --unit oran-causal-job-a" in sh
    assert "-p CPUQuota=400%" in sh
    assert "-p MemoryMax=6500M" in sh
    assert "systemctl --user is-active oran-causal-job-a.scope" in sh
    assert v.job_dir("job-a") == "/data/home/oran-causal/jobs/job-a"
    v.state()
    v.stop("job-a")
    assert "systemctl --user list-units" in t.scripts[0]
    assert "systemctl --user stop oran-causal-job-a.scope" in t.scripts[-1]


def test_instabox_caps_procs_at_four(tmp_path: Path):
    v = vps.Vps(FakeTransport(), hosts.INSTABOX)
    with pytest.raises(ValueError, match=r"--procs must be 1\.\.4"):
        v.launch("job-a", "x", 5, Dirs(tmp_path, tmp_path, tmp_path))

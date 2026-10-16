import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from oran_causal.experiments.campaign.host import PLATFORM_ENV
from oran_causal.experiments.campaign.provenance import COMMIT_ENV, DIRTY_ENV, LOCK_ENV
from oran_causal.experiments.launch import commands, hosts

CODE = {"commit": "abc", "dirty": False}


def parts(**changes) -> commands.PartsPlan:
    base: dict[str, Any] = {
        "spec": "spec.json",
        "parts": 2,
        "part_ids": [0, 1],
        "out_dir": "$OUT",
        "platform": "kaggle",
        "wall_s": 3600,
        "code": CODE,
    }
    return commands.PartsPlan(**{**base, **changes})


def git_bash() -> str | None:
    if os.name == "nt":
        p = r"C:\Program Files\Git\bin\bash.exe"
        return p if os.path.exists(p) else None
    return shutil.which("bash")


def test_part_environment_uses_the_oran_causal_names():
    env = commands.part_env(parts())
    assert f"{PLATFORM_ENV}=kaggle" in env
    assert f"{COMMIT_ENV}=abc" in env
    assert f"{DIRTY_ENV}=0" in env
    assert f"{LOCK_ENV}=uv.lock" in env
    assert "OMP_NUM_THREADS=1" in env
    assert "XM_" not in env


def test_gpu_cloud_and_setup_commands():
    cmd = commands.cloud_cmd(parts(registry=True, gpu=True))
    assert "NG=$(nvidia-smi -L" in cmd
    assert "CUDA_VISIBLE_DEVICES=$(( 1 % NG ))" in cmd
    assert cmd.count("--registry $OUT/registry.jsonl") == 2
    assert "PYTORCH_NVML_BASED_CUDA_CHECK=1" in cmd
    cpu = commands.cloud_cmd(parts(part_ids=[0], parts=1))
    assert "CUDA_VISIBLE_DEVICES" not in cpu
    assert "--registry" not in cpu
    relaunch = commands.cloud_cmd(parts(platform="colab", skip_from=["d/l1_*.jsonl"]))
    assert relaunch.count("--skip-complete-from 'd/l1_*.jsonl'") == 2
    assert "--extra cu128" in commands.sync_command(commands.SetupPlan("$OUT", gpu=True))
    assert "--extra cpu" in commands.sync_command(commands.SetupPlan("$OUT"))
    assert "--extra" not in commands.sync_command(commands.SetupPlan("$OUT", torch=False))


def test_setup_installs_the_bundled_lock_and_checks_the_pins():
    setup = commands.setup_cmd(commands.SetupPlan("$OUT"))
    assert "uv sync --frozen --no-dev --package oran-causal-experiments" in setup
    assert f"UV_PROJECT_ENVIRONMENT={commands.VENV_DIR}" in setup
    assert "pin_check(read_lock(), workspace_python())" in setup
    assert "$OUT/pin_check.json" in setup
    assert "pytest" not in setup
    selftest = commands.setup_cmd(commands.SetupPlan("$OUT", selftest=True))
    assert "uv sync --frozen --group dev" in selftest
    assert "packages/experiments/tests/test_campaign_*.py" in selftest


def test_eval_cloud_cmd_caps_processes_at_the_cpu_quota():
    ev = commands.cloud_cmd(
        parts(parts=6, part_ids=list(range(6)), prefix="eval_res", one_per_cpu=True)
    )
    assert "from oran_causal.experiments.campaign.host import cpu_quota" in ev
    assert ev.count("while [ $(jobs -rp | wc -l) -ge $NP ]; do wait -n; done;") == 6
    assert ev.index("NP=$(") < ev.index("--part 0/6")
    dev = commands.cloud_cmd(parts())
    assert "wait -n" not in dev
    assert "NP=" not in dev


@pytest.mark.parametrize("gate", [False, True])
def test_session_exit_is_nonzero_if_any_part_fails(tmp_path: Path, gate: bool):
    bash = git_bash()
    if not bash:
        pytest.skip("no bash")
    out = tmp_path.as_posix()

    def run(fail: bool) -> subprocess.CompletedProcess[str]:
        plan = parts(parts=3, part_ids=[0, 1, 2], out_dir=out, wall_s=60, one_per_cpu=gate)
        cmd = commands.cloud_cmd(plan)
        stand_in = (
            'bash -c \'case "$4" in ' + ("1/*) exit 3;; " if fail else "") + "*) exit 0;; esac' x"
        )
        cmd = cmd.replace("python -u -m oran_causal.experiments.campaign run", stand_in)
        cmd = cmd.replace(
            'NP=$(python -c "from oran_causal.experiments.campaign.host import cpu_quota; '
            'print(cpu_quota())")',
            "NP=1",
        )
        return subprocess.run(
            [bash, "-c", cmd], capture_output=True, text=True, timeout=120, check=False
        )

    failed = run(fail=True)
    assert failed.returncode != 0
    assert "[xm-c] part 1 exit 3" in failed.stdout
    codes = [(tmp_path / f"rc_{i}.txt").read_text().strip() for i in range(3)]
    assert codes == ["0", "3", "0"]
    assert run(fail=False).returncode == 0


def test_the_shared_production_profile_holds_the_old_caps():
    p = hosts.VPS_SHARED
    assert (p.max_procs, p.mem_max, p.swap_max) == (7, "11G", "0")
    assert p.load_guard == hosts.LoadGuard(12 * 1024 * 1024, 1.0, True)
    assert p.scope_limits(4) == (
        "-p",
        "CPUQuota=400%",
        "-p",
        "MemoryMax=11G",
        "-p",
        "MemorySwapMax=0",
    )
    assert "CPUQuota=700%" in p.scope_limits(7)
    with pytest.raises(ValueError, match=r"1\.\.7"):
        p.scope_limits(8)
    assert hosts.KAGGLE_CPU.wall_s == 11 * 3600
    assert hosts.COLAB_CPU.wall_s == 3 * 3600
    assert hosts.PROFILES["vps-dedicated"].mem_max is None


def test_eval_launch_needs_an_authorised_protocol_and_a_clean_tree(kit):
    texts, sha = kit.frozen_texts()
    ev = kit.eval_spec(sha)
    clean = {"commit": "a" * 40, "dirty": False, "dirty_paths": []}
    assert commands.launch_checks(ev, clean, texts) == []
    wrong = kit.eval_spec("0" * 64)
    assert "EVAL refused" in commands.launch_checks(wrong, clean, texts)[0]
    dirty = {"commit": "a" * 40, "dirty": True, "dirty_paths": ["README.md"]}
    assert "tree not clean" in commands.launch_checks(ev, dirty, texts)[0]
    assert commands.launch_checks(kit.small_spec(), dirty, texts) == []

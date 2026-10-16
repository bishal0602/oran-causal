import json
import os
import types
from typing import Any, cast

import pytest

from oran_causal.experiments.campaign import provenance as P
from oran_causal.experiments.campaign.host import campaign_procs, cpu_quota, load_info
from oran_causal.experiments.campaign.integrity import check_one_per_cpu


def lock(text: str) -> P.LockFile:
    return P.LockFile(text, "uv.lock", "0" * 64)


def test_dirty_paths_scope():
    st = "\n".join(
        [
            " M packages/core/src/oran_causal/core/design.py",
            "?? packages/experiments/src/oran_causal/experiments/new.py",
            "!! packages/core/src/oran_causal/core/__pycache__/a.pyc",
            "!! packages/experiments/specs/eval/full.json",
            "!! data/records/study4_eval/eval/report.json",
            "?? notes.txt",
            " M README.md",
            "!! packages/x/__pycache__/",
        ]
    )
    assert P.dirty_paths(st) == [
        "packages/core/src/oran_causal/core/design.py",
        "packages/experiments/src/oran_causal/experiments/new.py",
        "packages/experiments/specs/eval/full.json",
        "README.md",
    ]
    assert P.dirty_paths("") == []


def test_code_info_prefers_env_then_git_then_bundle(tmp_path):
    env = {P.COMMIT_ENV: "abc123", P.DIRTY_ENV: "0"}
    code = P.code_info(env, git=lambda args: pytest.fail("git called"), root=tmp_path)
    assert code == {
        "commit": "abc123",
        "dirty": False,
        "source": "env (launcher git check)",
        "dirty_paths": None,
    }
    answers = {"rev-parse": "f" * 40 + "\n", "status": " M README.md\n"}
    code = P.code_info({}, git=lambda args: answers[args[0]], root=tmp_path)
    assert (code["commit"], code["dirty"], code["dirty_paths"]) == ("f" * 40, True, ["README.md"])
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"git_head": "b" * 40, "dirty": False}))
    code = P.code_info({}, git=lambda args: None, root=tmp_path)
    assert (code["commit"], code["dirty"]) == ("b" * 40, False)
    (tmp_path / "MANIFEST.json").unlink()
    assert P.code_info({}, git=lambda args: None, root=tmp_path)["commit"] is None


def uv_lock(*packages: tuple[str, str, str]) -> str:
    head = 'version = 1\nrequires-python = ">=3.14"\n'
    body = "".join(
        f'\n[[package]]\nname = "{n}"\nversion = "{v}"\nsource = {{ {src} = "x" }}\n'
        for n, v, src in packages
    )
    return head + body


def test_pin_check_against_lock():
    text = uv_lock(
        ("numpy", "2.5.4", "registry"),
        ("scipy", "0.0.1", "registry"),
        ("absent", "1.0", "registry"),
        ("oran-causal-core", "0.1.0", "editable"),
    )
    installed = {"numpy": "2.5.4", "scipy": "1.18.0", "oran-causal-core": "9.9", "extra": "1"}
    pc = P.pin_check(lock(text), installed=installed)
    assert pc["n_pins"] == 3
    assert pc["mismatches"] == {"scipy": ["0.0.1", "1.18.0"]}
    assert P.pin_check(None)["mismatches"] is None


def test_pin_check_torch_build_exception(monkeypatch):
    text = uv_lock(
        ("torch", "2.11.0", "registry"),
        ("torch", "2.11.0+cpu", "registry"),
        ("torch", "2.11.0+cu128", "registry"),
        ("triton", "3.6.0", "registry"),
        ("nvidia-cudnn-cu12", "9.10.2.21", "registry"),
        ("nvidia-nccl-cu13", "2.32.3", "registry"),
        ("numpy", "2.5.4", "registry"),
    )
    ok = {
        "torch": "2.11.0+cu126",
        "triton": "3.5.0",
        "nvidia-nccl-cu13": "2.32.3",
        "numpy": "2.5.4",
    }
    pc = P.pin_check(lock(text), installed=ok)
    assert pc["mismatches"] == {}
    assert pc["torch_build"] == "2.11.0+cu126"
    assert pc["torch_build_exempt"] == {"triton": "3.5.0"}
    old_torch = {**ok, "torch": "2.9.1+cpu"}
    assert set(P.pin_check(lock(text), installed=old_torch)["mismatches"]) == {"torch"}
    bad_nccl = {**ok, "nvidia-nccl-cu13": "2.30.0"}
    assert set(P.pin_check(lock(text), installed=bad_nccl)["mismatches"]) == {"nvidia-nccl-cu13"}
    monkeypatch.setattr(P.platform, "python_version", lambda: "3.15.0")
    pin = P.PythonPin("3.14", ">=3.14")
    pc = P.pin_check(lock(text), python=pin, installed=ok)
    assert pc["mismatches"] == {"python": ["3.14 (>=3.14)", "3.15.0"]}
    assert pc["python_required"] == "3.14 (>=3.14)"
    assert P.pin_check(lock(text), installed=ok)["mismatches"] == {}


def test_python_pin_admits_the_pinned_release_only():
    pin = P.PythonPin("3.14", ">=3.14")
    assert pin.admits("3.14.0")
    assert pin.admits("3.14.4")
    assert not pin.admits("3.15.0")
    assert not pin.admits("3.12.14")
    assert not P.PythonPin("3.14.4", ">=3.14").admits("3.14.3")
    assert not P.PythonPin("3.14", ">=3.14.5").admits("3.14.4")


def test_workspace_python_reads_python_version_and_lock(tmp_path):
    (tmp_path / ".python-version").write_text("3.14\n")
    (tmp_path / "uv.lock").write_text(uv_lock(("numpy", "2.5.4", "registry")))
    assert P.workspace_python(tmp_path) == P.PythonPin("3.14", ">=3.14")


def test_the_running_interpreter_is_the_workspace_pin():
    assert P.workspace_python().admits(P.platform.python_version())


def test_installed_versions_first_copy_wins():
    dists = [
        types.SimpleNamespace(metadata={"Name": "NumPy"}, version="2.5.4"),
        types.SimpleNamespace(metadata={"Name": "numpy"}, version="1.26.0"),
    ]
    assert P.installed_versions(cast(Any, dists)) == {"numpy": "2.5.4"}


def test_read_lock_hashes_with_lf(tmp_path, monkeypatch):
    path = tmp_path / "uv.lock"
    path.write_bytes(b"version = 1\r\n")
    got = P.read_lock(path)
    assert got is not None
    assert got.sha256 == P.sha_lf(b"version = 1\n")
    monkeypatch.delenv(P.LOCK_ENV, raising=False)
    found = P.read_lock(None, root=tmp_path)
    assert found is not None
    assert found.sha256 == got.sha256
    assert P.read_lock(None, root=tmp_path / "missing") is None
    assert P.read_lock(tmp_path / "missing.lock") is None


def test_cpu_quota_procs_and_load(tmp_path):
    cap = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("400000 100000\n")
    none = tmp_path / "none"
    assert cpu_quota(v2, none) == min(4, cap)
    (v2 / "cpu.max").write_text("150000 100000\n")
    assert cpu_quota(v2, none) == min(2, cap)
    (v2 / "cpu.max").write_text("max 100000\n")
    assert cpu_quota(v2, none) >= 1
    v1 = tmp_path / "v1" / "cpu"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("200000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert cpu_quota(tmp_path / "v1", none) == min(2, cap)
    host = tmp_path / "host"
    scope = host / "system.slice" / "oran-causal-j.scope"
    scope.mkdir(parents=True)
    (host / "cpu.max").write_text("max 100000\n")
    (scope / "cpu.max").write_text("700000 100000\n")
    me = tmp_path / "self_cgroup"
    me.write_text("0::/system.slice/oran-causal-j.scope\n")
    assert cpu_quota(host, me) == min(7, cap)
    (host / "system.slice" / "cpu.max").write_text("300000 100000\n")
    assert cpu_quota(host, me) == min(3, cap)
    me.write_text("0::/\n")
    assert cpu_quota(host, me) == cap


def test_campaign_procs_count_shard_files(tmp_path):
    proc = tmp_path / "proc"
    run = "python -u -m oran_causal.experiments run --part"
    for pid, cmd in (
        (9, f"timeout 99 {run} 0/2 --out o/r_0.jsonl"),
        (10, f"{run} 0/2 --out o/r_0.jsonl"),
        (11, f"{run} 0/2 --out o/r_0.jsonl"),
        (12, f"{run} 1/2 --out o/r_1.jsonl"),
        (13, "bash -c sleep 9"),
        (14, "python -m oran_causal.experiments merge --spec s --out m.jsonl"),
    ):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
    (proc / "loadavg").write_text("3.50 2.25 1.00 4/180 999\n")
    assert campaign_procs(proc) == 2
    li = load_info(proc)
    assert li["loadavg"] == [3.5, 2.25, 1.0]
    assert (li["procs_running"], li["procs_total"]) == (4, 180)
    assert (li["campaign_procs"], li["proc_visible"]) == (2, 6)
    assert campaign_procs(tmp_path / "missing") is None
    check_one_per_cpu(4, 4)
    check_one_per_cpu(None, 1)
    with pytest.raises(ValueError, match="R-55"):
        check_one_per_cpu(5, 4)

import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from oran_causal.experiments.launch import bundle
from oran_causal.experiments.paths import WORKSPACE_ROOT


def test_the_base_set_is_exactly_what_a_run_needs(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path)
    got = set(bundle.collect(ws))
    assert got == {
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "packages/core/pyproject.toml",
        "packages/core/src/oran_causal/core/__init__.py",
        "packages/figures/pyproject.toml",
        "packages/experiments/protocols/study4/PROTOCOL_A.md",
        "packages/experiments/manifests/seeds.json",
        "packages/experiments/artifacts/table.json",
    }


def test_extras_add_files_and_directories_but_never_leave_the_workspace(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    extra = bundle.collect(ws, ["packages/experiments/specs/dev/s.json", "records/runs/old/done"])
    assert "packages/experiments/specs/dev/s.json" in extra
    assert "records/runs/old/done/a.jsonl" in extra
    with pytest.raises(ValueError, match="outside the workspace"):
        bundle.collect(ws, ["../elsewhere"])
    with pytest.raises(FileNotFoundError):
        bundle.collect(ws, ["nothing/here.json"])


def test_the_manifest_hashes_every_member_and_text_is_lf(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    req = bundle.BundleRequest("job-1", ws, cmd="echo hi", sources=("me/kernel",))
    built = bundle.build_bundle(req, tmp_path / "out", launch_kit.fake_git(" M README.md\n"))
    with zipfile.ZipFile(built.zip_path) as z:
        man = json.loads(z.read(bundle.MANIFEST))
        assert sorted(n for n in z.namelist() if n != bundle.MANIFEST) == list(built.files)
        for rel in built.files:
            assert man["sha256"][rel] == bundle.sha256(z.read(rel))
        assert b"\r" not in z.read("pyproject.toml")
        assert z.read("packages/core/src/oran_causal/core/__init__.py") == b"A = 1\n"
    assert man["git_head"] == launch_kit.HEAD
    assert man["dirty"] is True
    assert man["dirty_paths"] == ["README.md"]
    assert man["cmd"] == "echo hi"
    assert man["sources"] == ["me/kernel"]
    assert set(man["local_env"]) >= {"python"}
    assert "local_fp" not in man


def test_the_fingerprint_is_recorded_only_on_request(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    fp = {"env": {"numpy": "9"}, "hash": "h"}
    req = bundle.BundleRequest("job-2", ws, with_fingerprint=True)
    built = bundle.build_bundle(req, tmp_path / "out", launch_kit.fake_git(), lambda: fp)
    assert built.manifest["local_fp"] == fp
    assert built.manifest["local_env"] == {"numpy": "9"}


def test_the_frozen_protocol_copy_and_markers_ride_along(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    req = bundle.BundleRequest(
        "job-3", ws, frozen_protocol=b"FROZEN: yes\nold\n", markers={"_job_job-3.json": b"{}"}
    )
    built = bundle.build_bundle(req, tmp_path / "out", launch_kit.fake_git())
    assert bundle.FROZEN_COPY_REL in built.files
    with zipfile.ZipFile(built.zip_path) as z:
        assert z.read(bundle.FROZEN_COPY_REL) == b"FROZEN: yes\nold\n"
        assert "_job_job-3.json" in z.namelist()
        assert "_job_job-3.json" not in built.manifest["sha256"]
    unpacked = tmp_path / "unpacked"
    zipfile.ZipFile(built.zip_path).extractall(unpacked)
    texts = bundle.bundled_texts(unpacked)
    assert texts.live == b"FROZEN: yes\n"
    assert texts.frozen_copy == b"FROZEN: yes\nold\n"


def test_kaggle_dot_names_survive_the_dataset_round_trip(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    built = bundle.build_bundle(
        bundle.BundleRequest("job-4", ws, dot_encode=True), tmp_path / "out", launch_kit.fake_git()
    )
    with zipfile.ZipFile(built.zip_path) as z:
        names = z.namelist()
    assert "_dot_python-version" in names
    assert ".python-version" not in names
    assert all(bundle.decode_dots(n) in {*built.files, bundle.MANIFEST} for n in names)
    assert bundle.encode_dots("a/.b/c.d") == "a/_dot_b/c.d"


def test_the_ssh_tarball_carries_the_same_members(tmp_path: Path, launch_kit):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    req = bundle.BundleRequest("job-5", ws)
    blobs = bundle.build_blobs(req)
    man = bundle.build_manifest(req, blobs, launch_kit.fake_git())
    with tarfile.open(fileobj=io.BytesIO(bundle.tar_bytes(blobs, man)), mode="r:gz") as tf:
        names = tf.getnames()
        assert sorted(names) == sorted([*blobs, bundle.MANIFEST])
        member = tf.extractfile("uv.lock")
        assert member is not None
        assert member.read() == blobs["uv.lock"]


def test_the_frozen_text_comes_from_the_freeze_commit_else_the_live_file(
    tmp_path: Path, launch_kit
):
    ws = launch_kit.mini_workspace(tmp_path / "ws")
    seen: list[tuple[str, str]] = []

    def reader(commit: str, rel: str) -> bytes | None:
        seen.append((commit, rel))
        return b"frozen at commit"

    assert bundle.frozen_protocol({"freeze_commit": "abc"}, ws, reader) == b"frozen at commit"
    assert seen == [("abc", bundle.PROTOCOL_REL)]
    assert (
        bundle.frozen_protocol({"freeze_commit": "abc"}, ws, lambda c, r: None) == b"FROZEN: yes\n"
    )
    assert bundle.frozen_protocol({}, ws, reader) == b"FROZEN: yes\n"
    assert len(seen) == 1


def test_the_real_workspace_bundle_has_the_run_files_and_no_dev_tooling():
    files = set(bundle.collect(WORKSPACE_ROOT))
    for need in (
        "uv.lock",
        ".python-version",
        "pyproject.toml",
        bundle.PROTOCOL_REL,
        "packages/core/src/oran_causal/core/pmrt/rows.py",
        "packages/experiments/src/oran_causal/experiments/campaign/run.py",
        "packages/experiments/src/oran_causal/experiments/launch/commands.py",
        "packages/methods/src/oran_causal/methods/controlled/mscr.py",
    ):
        assert need in files, need
    assert not [f for f in files if "/tests/" in f or "__pycache__" in f or f.endswith(".pyc")]
    assert not [f for f in files if f.startswith(("records/", "docs/", "figures/"))]


@pytest.mark.slow
def test_a_bundle_imports_every_arm_without_the_workspace(tmp_path: Path, launch_kit):
    built = bundle.build_bundle(
        bundle.BundleRequest("job-imports", WORKSPACE_ROOT), tmp_path / "out", launch_kit.fake_git()
    )
    root = tmp_path / "unpacked"
    zipfile.ZipFile(built.zip_path).extractall(root)
    src = [str(p) for p in sorted((root / "packages").glob("*/src"))]
    probe = (
        "import importlib, json, sys\n"
        "from oran_causal.methods import registry\n"
        "bad = {}\n"
        "for name, (module, _cls) in registry.PATHS.items():\n"
        "    try:\n"
        "        importlib.import_module(module)\n"
        "    except Exception as e:\n"
        "        bad[name] = type(e).__name__ + ': ' + str(e)[:200]\n"
        "print(json.dumps({'file': registry.__file__, 'bad': bad}))\n"
    )

    def run(cwd: Path, paths: list[str]) -> dict:
        env = {"PYTHONPATH": ";".join(paths) if sys.platform == "win32" else ":".join(paths)}
        p = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=cwd,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            check=False,
        )
        assert p.returncode == 0, p.stderr[-2000:]
        return json.loads(p.stdout.strip().splitlines()[-1])

    in_bundle = run(root, src)
    assert Path(in_bundle["file"]).is_relative_to(root)
    local = run(
        WORKSPACE_ROOT, [str(p) for p in sorted((WORKSPACE_ROOT / "packages").glob("*/src"))]
    )
    assert set(in_bundle["bad"]) <= set(local["bad"]), in_bundle["bad"]

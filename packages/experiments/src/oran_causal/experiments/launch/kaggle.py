"""Kaggle launcher: a code dataset plus a private script kernel that runs one session command.

The dataset `<owner>/<name>-code` holds the bundle zip; the kernel `<owner>/<name>` unpacks it to
`/tmp/job_root` and runs the command with bash from there. Names are new per launch, because a
kernel may not share a dataset's slug.
"""

import dataclasses
import json
import re
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from oran_causal.experiments.campaign.provenance import GitRunner
from oran_causal.experiments.launch import bundle, settings
from oran_causal.experiments.launch.runner import RUNS_DIR, Clock, Dirs, Runner, run_process

DEFAULT_PULL = r"^(?!.*\.npz$)"
DATASET_WAIT_S = 900
PUSH_RETRY_S = 60
PUSH_BLOCKED = "maximum batch"

KERNEL = r"""
import glob, json, os, shutil, subprocess, sys, time, zipfile
NAME, CMD, SOURCES = __NAME__, __CMD__, __SOURCES__
W = "/kaggle/working"
OUT = os.path.join(W, "out")
os.makedirs(OUT, exist_ok=True)
logf = open(os.path.join(W, "job.log"), "a")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


t0 = time.time()
root = "/tmp/job_root"
mk = glob.glob(f"/kaggle/input/**/_job_{NAME}.json", recursive=True)
if mk:
    shutil.copytree(os.path.dirname(mk[0]), root, dirs_exist_ok=True)
    how = "unzipped dataset at " + os.path.dirname(mk[0])
else:
    z = glob.glob(f"/kaggle/input/**/{NAME}_bundle.zip", recursive=True)
    zipfile.ZipFile(z[0]).extractall(root)
    how = "zip " + z[0]
for dp, dn, fn in os.walk(root, topdown=False):
    for x in dn + fn:
        if x.startswith("_dot_"):
            os.rename(os.path.join(dp, x), os.path.join(dp, "." + x[5:]))
log(f"[job] bundle: {how} -> {root}")
lines = []
for dp, dn, fn in os.walk("/kaggle/input"):
    depth = dp.count("/") - 2
    if "/bundle" in dp or "__pycache__" in dp:
        continue
    if depth <= 5:
        lines.append(f"D {dp} ({len(fn)} files)")
    for f in fn:
        if f.endswith((".jsonl", ".npz", ".json", ".zip")) and "/bundle/" not in dp:
            p = os.path.join(dp, f)
            lines.append(f"F {p} {os.path.getsize(p)}")
open(os.path.join(W, "job_inputs.txt"), "w").write("\n".join(lines) + "\n")
log(f"[job] /kaggle/input: {len(lines)} listing lines -> job_inputs.txt")
SRC = "/tmp/src"
os.makedirs(SRC, exist_ok=True)
found = {}
for s in SOURCES:
    slug = s.split("/")[-1]
    hits = sorted((dp for dp, dn, fn in os.walk("/kaggle/input") if os.path.basename(dp) == slug
                   and dp.count("/") <= 6), key=len)
    if hits:
        found[slug] = hits[0]
        if not os.path.exists(os.path.join(SRC, slug)):
            os.symlink(hits[0], os.path.join(SRC, slug))
missing = [s for s in SOURCES if s.split("/")[-1] not in found]
log(f"[job] sources -> {SRC}/<slug>: {found}; missing {missing}")
man = json.load(open(os.path.join(root, "MANIFEST.json")))
env = dict(os.environ, KAGGLE_SOURCES_DIR="/kaggle/input", JOB_SRC=SRC, JOB_OUT=OUT, JOB_NAME=NAME,
           PYTHONUNBUFFERED="1")
log(f"[job] git {man.get('git_head')} dirty {man.get('dirty')}; cpus {os.cpu_count()}; CMD: {CMD}")
p = subprocess.Popen(["bash", "-c", CMD], cwd=root, env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    log(line.rstrip("\n"))
rc = p.wait()
wall = round(time.time() - t0, 1)
log(f"[job] exit code {rc}; wall {wall} s")
done = {"name": NAME, "cmd": CMD, "rc": rc, "wall_s": wall, "git_head": man.get("git_head"),
        "dirty": man.get("dirty"), "cpus": os.cpu_count()}
json.dump(done, open(os.path.join(W, "job.json"), "w"), indent=1)
"""


@dataclasses.dataclass(frozen=True)
class KaggleJob:
    """One Kaggle session.

    Attributes:
        name: New slug per launch.
        cmd: The session command, run by bash from the bundle root.
        extra: Workspace-relative files added to the bundle.
        sources: `owner/kernel` outputs mounted as inputs (read through `$JOB_SRC/<slug>`).
        gpu: Request a GPU.
        wait_min: Minutes to retry a push blocked by the account's concurrent-session limit.
        frozen_protocol: Verified frozen protocol text bundled for the EVAL guard.
    """

    name: str
    cmd: str
    extra: Sequence[str] = ()
    sources: Sequence[str] = ()
    gpu: bool = False
    wait_min: int = 240
    frozen_protocol: bytes | None = None


def kernel_source(job: KaggleJob) -> str:
    """The kernel script with the job's parameters substituted."""
    return (
        KERNEL.replace("__NAME__", repr(job.name))
        .replace("__CMD__", repr(job.cmd))
        .replace("__SOURCES__", repr(list(job.sources)))
    )


def dataset_metadata(job: KaggleJob, owner: str) -> dict[str, Any]:
    """`dataset-metadata.json` of the code dataset."""
    return {
        "title": f"{job.name}-code",
        "id": f"{owner}/{job.name}-code",
        "licenses": [{"name": "CC0-1.0"}],
    }


def kernel_metadata(job: KaggleJob, owner: str) -> dict[str, Any]:
    """`kernel-metadata.json`: private script kernel with internet (the session runs `uv sync`)."""
    return {
        "id": f"{owner}/{job.name}",
        "title": job.name,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": job.gpu,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{job.name}-code"],
        "kernel_sources": list(job.sources),
    }


def stage(
    job: KaggleJob,
    out: Path,
    root: Path = bundle.WORKSPACE_ROOT,
    git: GitRunner | None = None,
    owner: str | None = None,
) -> bundle.Bundle:
    """Writes `out/dataset` (bundle zip, marker, metadata) and `out/kernel` (script, metadata).

    `owner` defaults to the `ORAN_CAUSAL_KAGGLE_OWNER` setting.
    """
    owner = owner or settings.kaggle_owner()
    ds, kd = out / "dataset", out / "kernel"
    kd.mkdir(parents=True, exist_ok=True)
    marker = json.dumps({"name": job.name})
    req = bundle.BundleRequest(
        job.name,
        root,
        job.extra,
        job.frozen_protocol,
        job.cmd,
        job.sources,
        markers={f"_job_{job.name}.json": marker.encode()},
        dot_encode=True,
    )
    built = bundle.build_bundle(req, ds, git)
    (ds / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(job, owner)))
    (kd / "kernel.py").write_text(kernel_source(job), newline="\n")
    (kd / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(job, owner), indent=1))
    return built


@dataclasses.dataclass(frozen=True)
class KaggleClient:
    """The `kaggle` CLI behind an injectable runner.

    Attributes:
        runner: Runs an argv in a directory with a timeout and returns its output.
        executable: The CLI; None reads ORAN_CAUSAL_KAGGLE_CLI (else `kaggle` on PATH) when used.
        owner: Account the datasets and kernels belong to; None reads ORAN_CAUSAL_KAGGLE_OWNER when
            used.
        clock: Time source for the waiting loops.
    """

    runner: Runner = run_process
    executable: str | None = None
    owner: str | None = None
    clock: Clock = dataclasses.field(default_factory=Clock)

    @property
    def cli(self) -> str:
        """The Kaggle CLI."""
        return self.executable or settings.kaggle_cli()

    @property
    def account(self) -> str:
        """The Kaggle account."""
        return self.owner or settings.kaggle_owner()

    def run(self, *args: str, cwd: Path | None = None, timeout: float = 900) -> str:
        """One CLI call."""
        return self.runner([self.cli, *args], cwd, timeout)

    def create_dataset(self, out: Path) -> str:
        """Creates the code dataset from `out/dataset` (relative `-p`)."""
        return self.run("datasets", "create", "-p", "dataset", cwd=out)

    def wait_dataset(self, name: str) -> float:
        """Waits until the code dataset is ready (a new one answers 403 until then).

        Returns:
            Seconds waited.

        Raises:
            TimeoutError: If it is not ready after `DATASET_WAIT_S`.
        """
        t0 = self.clock.now()
        while True:
            status = self.run("datasets", "status", f"{self.account}/{name}-code", timeout=120)
            if "ready" in status.lower():
                return self.clock.now() - t0
            if self.clock.now() - t0 > DATASET_WAIT_S:
                raise TimeoutError(f"dataset not ready after 15 min: {status[-300:]}")
            self.clock.sleep(10)

    def push_kernel(self, out: Path, wait_min: int) -> str:
        """Pushes the kernel, retrying every minute while the concurrent-session limit blocks it.

        Raises:
            TimeoutError: If still blocked after `wait_min` minutes.
        """
        t0 = self.clock.now()
        while True:
            result = self.run("kernels", "push", "-p", "kernel", cwd=out)
            if PUSH_BLOCKED not in result.lower():
                return result
            if self.clock.now() - t0 > 60 * wait_min:
                raise TimeoutError(f"push still blocked after {wait_min} min: {result[-200:]}")
            self.clock.sleep(PUSH_RETRY_S)

    def status(self, name: str) -> str:
        """Kernel status line."""
        return self.run("kernels", "status", f"{self.account}/{name}", timeout=60)

    def output(self, name: str, dst: Path, pattern: str | None) -> str:
        """Downloads the kernel output into `dst` (`pattern` None = everything)."""
        args = ["kernels", "output", f"{self.account}/{name}", "-p", dst.name, "-o"]
        if pattern is not None:
            args += ["--file-pattern", pattern]
        return self.run(*args, cwd=dst.parent, timeout=3600)


def launch(
    job: KaggleJob, client: KaggleClient, dirs: Dirs, git: GitRunner | None = None
) -> dict[str, Any]:
    """Stages, creates the dataset, waits for it, pushes the kernel and records the launch.

    Raises:
        RuntimeError: If the dataset or the kernel push reports an error.
    """
    out = dirs.work / job.name
    if out.exists():
        shutil.rmtree(out)
    built = stage(job, out, dirs.root, git, client.account)
    created = client.create_dataset(out)
    if re.search("error", created, re.IGNORECASE):
        raise RuntimeError(f"dataset create failed (the name must be new per launch): {created}")
    waited = client.wait_dataset(job.name)
    pushed = client.push_kernel(out, job.wait_min)
    if re.search("error", pushed, re.IGNORECASE):
        raise RuntimeError(f"kernel push failed: {pushed}")
    record = {
        "name": job.name,
        "cmd": job.cmd,
        "sources": list(job.sources),
        "extra": list(job.extra),
        "gpu": job.gpu,
        "git_head": built.manifest["git_head"],
        "dirty": built.manifest["dirty"],
        "files": len(built.files),
        "dataset_wait_s": round(waited),
        "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(client.clock.now())),
    }
    dst = dirs.runs / job.name
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "launch.json").write_text(json.dumps(record, indent=1))
    return record


def pull(
    name: str, client: KaggleClient, runs: Path = RUNS_DIR, pattern: str | None = DEFAULT_PULL
) -> dict[str, Any]:
    """Downloads the kernel output to `runs/<name>` and returns its `job.json` (or {}).

    The default skips `*.npz` caches; pass `pattern=None` to pull everything.
    """
    dst = runs / name
    dst.mkdir(parents=True, exist_ok=True)
    client.output(name, dst, pattern)
    job = dst / "job.json"
    return json.loads(job.read_text()) if job.exists() else {}

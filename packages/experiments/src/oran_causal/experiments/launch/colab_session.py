"""The Colab session layer: the `colab` CLI, remote scripts run in the session's kernel.

The campaign uses one-shot jobs: a new runtime, an uploaded bundle, a shell command started under
`nohup`, and ticks that pack and download its outputs. Sessions are never resumed.
"""

import dataclasses
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from oran_causal.experiments.launch.runner import Clock

WORK = "/content/oran-causal-run"
JOB_PULL = "/content/oran-causal-pull.tar.gz"
HTTP_TIMEOUT = int(os.environ.get("COLAB_HTTP_TIMEOUT", "900"))
JOB_FILES = (
    "job.log",
    "job.json",
    "job.sh",
    "startup.json",
    "exit_code",
    "times.txt",
    "ended",
    "wrapper.log",
)
ACCELERATORS = {"tpu": "--tpu", "gpu": "--gpu", "cpu": None}
NEW_TRIES = 3
NEW_RETRY_S = 60
STOP_WAIT_S = 600
STOP_POLL_S = 15
SESSION_ROW = re.compile(r"^\[(?P<name>[^\]]+)\] \S+ \| Hardware: ")
NO_SESSIONS = "No active sessions found on server"

WRAP = (
    "import sys\n"
    "import google.auth.transport.requests as R\n"
    "_o = R.AuthorizedSession.request\n"
    "def _req(self, method, url, *a, **k):\n"
    f"    k.setdefault('timeout', {HTTP_TIMEOUT})\n"
    "    return _o(self, method, url, *a, **k)\n"
    "R.AuthorizedSession.request = _req\n"
    "from colab_cli.cli import main\n"
    "sys.argv = ['colab'] + sys.argv[1:]\n"
    "main()\n"
)
TERMIOS_STUB = (
    "TCSANOW=0\nTCSADRAIN=1\nTCSAFLUSH=2\n"
    "def tcgetattr(fd):\n    raise OSError('termios unavailable on Windows')\n"
    "def tcsetattr(*a):\n    raise OSError('termios unavailable on Windows')\n"
)

JOB_SETUP = r"""
import json, os, shutil, subprocess, time, zipfile
W, BUNDLE, THREADS = __W__, __BUNDLE__, __THREADS__
t0 = time.time()
os.makedirs(W, exist_ok=True)
root = os.path.join(W, "b")
shutil.rmtree(root, ignore_errors=True)
for f in __FILES__:
    try:
        os.remove(os.path.join(W, f))
    except FileNotFoundError:
        pass
shutil.move(BUNDLE, os.path.join(W, "bundle.zip"))
zipfile.ZipFile(os.path.join(W, "bundle.zip")).extractall(root)
man = json.load(open(os.path.join(root, "MANIFEST.json")))
try:
    q, per = open("/sys/fs/cgroup/cpu.max").read().split()
    quota = None if q == "max" else int(q) / int(per)
except Exception:
    quota = None
threads = int(THREADS or max(1, int(quota or os.cpu_count() or 1)))
try:
    host = subprocess.run(
        ["bash", "-c", "free -g | sed -n 2p; lscpu | grep -E 'Model name|Thread|Core|Socket'"],
        capture_output=True, text=True).stdout
except Exception as e:
    host = repr(e)
start = {"git_head": man.get("git_head"), "dirty": man.get("dirty"),
         "created_utc": man.get("created_utc"), "n_files": len(man.get("sha256", {})),
         "cpus": os.cpu_count(), "cpu_quota": quota, "threads": threads, "host": host,
         "t_setup_s": round(time.time() - t0, 1)}
json.dump(start, open(os.path.join(W, "startup.json"), "w"), indent=1)
print("JOBSTART " + json.dumps(start), flush=True)
"""

JOB_LAUNCH = r"""
import json, os, shlex, subprocess, time
W, CMD, OUT = __W__, __CMD__, __OUT__
root = os.path.join(W, "b")
st = json.load(open(os.path.join(W, "startup.json")))
n = str(st["threads"])
env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
           **{k: n for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
if OUT:
    os.makedirs(os.path.join(root, OUT), exist_ok=True)
sh = os.path.join(W, "job.sh")
open(sh, "w").write("set -o pipefail\n" + CMD + "\n")
q = lambda x: shlex.quote(os.path.join(W, x))
wrap = (f"bash {q('job.sh')} > {q('job.log')} 2>&1 < /dev/null; rc=$?; times > {q('times.txt')}; "
        f"date +%s > {q('ended')}; echo $rc > {q('exit_code.tmp')}; "
        f"mv {q('exit_code.tmp')} {q('exit_code')}")
p = subprocess.Popen(["nohup", "bash", "-c", wrap], env=env, cwd=root, stdin=subprocess.DEVNULL,
                     stdout=open(os.path.join(W, "wrapper.log"), "a"), stderr=subprocess.STDOUT,
                     start_new_session=True)
J = {"pid": p.pid, "cmd": CMD, "out": OUT, "cwd": root, "started": time.time()}
json.dump(J, open(os.path.join(W, "job.json"), "w"))
time.sleep(5)
J["alive_5s"] = p.poll() is None
try:
    J["log_head"] = open(os.path.join(W, "job.log"), errors="replace").read()[:600]
except Exception:
    J["log_head"] = None
print("JOBLAUNCHED " + json.dumps(J), flush=True)
"""

JOB_TICK = r"""
import json, os, tarfile, time
W, OUT, DST, FILES, PACK = __W__, __OUT__, __DST__, __FILES__, __PACK__
root = os.path.join(W, "b")
jp = os.path.join(W, "job.json")
J = json.load(open(jp)) if os.path.exists(jp) else {}
def stat(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    except Exception:
        return None
def alive(pid):
    s = stat(pid)
    return bool(s) and s[0] != "Z"
def rd(n):
    try:
        return open(os.path.join(W, n), errors="replace").read().strip()
    except Exception:
        return None
tck = os.sysconf("SC_CLK_TCK")
cpu_s, rss, nproc = 0.0, 0, 0
if J:
    for d in os.listdir("/proc"):
        s = stat(d) if d.isdigit() else None
        if s and int(s[3]) == J["pid"] and s[0] != "Z":
            nproc += 1
            cpu_s += (int(s[11]) + int(s[12])) / tck
            rss += int(s[21]) * os.sysconf("SC_PAGE_SIZE")
rc, ended, log = rd("exit_code"), rd("ended"), rd("job.log") or ""
lines = log.splitlines()
od = os.path.join(root, OUT) if OUT else None
outs = []
if od and os.path.isdir(od):
    for dp, _, fns in os.walk(od):
        for fn in sorted(fns):
            f = os.path.join(dp, fn)
            outs.append([os.path.relpath(f, od), os.path.getsize(f)])
size = None
if PACK:
    with tarfile.open(DST, "w:gz") as t:
        for n in FILES:
            if os.path.exists(os.path.join(W, n)):
                t.add(os.path.join(W, n), arcname=n)
        if od and os.path.isdir(od):
            t.add(od, arcname="outdir")
    size = os.path.getsize(DST)
now = time.time()
mem = {ln.split(":")[0]: int(ln.split()[1]) // 1024 for ln in open("/proc/meminfo")
       if ln.split(":")[0] in ("MemTotal", "MemAvailable")}
elapsed = (float(ended) if ended else now) - J.get("started", now)
print("JOBTICK " + json.dumps({
    "launched": bool(J), "alive": bool(J) and alive(J["pid"]), "exit_code": int(rc) if rc else None,
    "elapsed_s": round(elapsed, 1), "times": rd("times.txt"),
    "live_cpu_s": round(cpu_s, 1), "live_rss_mb": rss >> 20, "live_procs": nproc, "mem_mb": mem,
    "log_lines": len(lines), "log_tail": lines[-8:], "traceback": "Traceback" in log,
    "n_out": len(outs), "out_files": outs[:50], "bytes": size,
    "uptime_s": float(open("/proc/uptime").read().split()[0]),
    "load": open("/proc/loadavg").read().split()[:3]}), flush=True)
"""

SPROBE = r"""
import json, sys
import google.auth.transport.requests as R
_o = R.AuthorizedSession.request
def _req(self, method, url, *a, **k):
    k.setdefault("timeout", 120)
    return _o(self, method, url, *a, **k)
R.AuthorizedSession.request = _req
import requests
from colab_cli.auth import AuthProvider
from colab_cli.common import state
state.auth_provider = AuthProvider("adc")
name, ping = sys.argv[1], sys.argv[2] == "1"
s = state.store.get(name)
asg = state.client.list_assignments()
by = {a.endpoint: a for a in asg}
out = {"known": s is not None, "n_assignments": len(asg), "endpoint": s.endpoint if s else None,
       "alive": bool(s) and s.endpoint in by,
       "orphans": [e for e in by if not s or e != s.endpoint]}
if out["alive"] and ping:
    try:
        r = state.client.session.request(
            "GET", f"https://colab.research.google.com/tun/m/{s.endpoint}/keep-alive/",
            params={"authuser": "0"},
            headers={"X-Colab-Tunnel": "Google", "X-Colab-Client-Agent": "colab-cli",
                     "Accept": "application/json"})
        out["keepalive_http"] = r.status_code
    except Exception as e:
        out["keepalive_err"] = repr(e)[:300]
if out["alive"]:
    try:
        ri = by[s.endpoint].runtime_proxy_info
        h = {"X-Colab-Runtime-Proxy-Token": ri.token, "X-Colab-Client-Agent": "colab-cli"}
        r = requests.get(ri.url.rstrip("/") + "/api/status", headers=h, timeout=60)
        if r.ok:
            out["server"] = r.json()
        r = requests.get(ri.url.rstrip("/") + "/api/kernels", headers=h, timeout=60)
        out["kernels_http"] = r.status_code
        if r.ok:
            out["kernels"] = [{k: x.get(k) for k in ("id", "last_activity", "execution_state",
                                                      "connections")} for x in r.json()]
    except Exception as e:
        out["kernels_err"] = repr(e)[:300]
print("SPROBE " + json.dumps(out), flush=True)
"""


def fill(src: str, **kw: Any) -> str:
    """Substitutes `__KEY__` placeholders with the `repr` of the values."""
    for k, v in kw.items():
        src = src.replace(f"__{k}__", repr(v))
    return src


class ColabError(RuntimeError):
    """A `colab` CLI call failed."""


class AssignmentError(RuntimeError):
    """Colab assignments are in a state that needs a human."""


type Invoke = Callable[[Sequence[str], float, bool], str]
type Probe = Callable[[str, bool], dict[str, Any]]


def termios_stub(directory: Path | None = None) -> Path:
    """Directory holding a `termios.py` stub the Windows `colab` CLI needs on PYTHONPATH."""
    d = directory or Path(
        os.environ.get("COLAB_STUB") or Path(tempfile.gettempdir()) / "colab_stub"
    )
    stub = d / "termios.py"
    if not stub.exists():
        d.mkdir(parents=True, exist_ok=True)
        stub.write_text(TERMIOS_STUB)
    return d


def tool_python() -> Path:
    """Python of the `google-colab-cli` uv tool (its `colab_cli` package lives there)."""
    override = os.environ.get("COLAB_TOOL_PY")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / "uv" / "tools" / "google-colab-cli" / "Scripts" / "python.exe"


def cli_invoke(argv: Sequence[str], timeout: float, check: bool) -> str:
    """Runs the `colab` CLI with ADC auth.

    The CLI runs through the tool's python with google-auth's per-request timeout raised to
    `COLAB_HTTP_TIMEOUT`: the TPU assignment POST takes about 130 s, so the stock CLI times out.
    """
    env = dict(
        os.environ,
        PYTHONPATH=str(termios_stub()),
        MSYS_NO_PATHCONV="1",
        PYTHONIOENCODING="utf-8",
        COLUMNS="200",
    )
    py = tool_python()
    exe = [str(py), "-c", WRAP] if py.exists() else [shutil.which("colab") or "colab"]
    p = subprocess.run(
        [*exe, "--auth", "adc", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        raise ColabError(f"colab {' '.join(argv[:3])} rc={p.returncode}: {out[-1500:]}")
    return out


def session_probe(name: str, ping: bool = True) -> dict[str, Any]:
    """Assignment state of session `name`, a keep-alive ping and the kernels' activity.

    One tool-python process; no kernel traffic.
    """
    env = dict(os.environ, PYTHONPATH=str(termios_stub()), PYTHONIOENCODING="utf-8")
    p = subprocess.run(
        [str(tool_python()), "-c", SPROBE, name, "1" if ping else "0"],
        capture_output=True,
        text=True,
        timeout=400,
        env=env,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith("SPROBE "):
            return json.loads(line[7:])
    raise ColabError(f"session probe rc={p.returncode}: {(p.stdout + p.stderr)[-1500:]}")


def assignment_rows(out: str) -> list[str]:
    """Session names of the server's assignments in `colab sessions` output (`?` for an orphan).

    Message lines such as `[colab] Pruned 1 stale local session(s).` are not rows.

    Raises:
        ColabError: If the output has neither a row nor the no-sessions message.
    """
    rows = [m["name"] for ln in out.splitlines() if (m := SESSION_ROW.match(ln.strip()))]
    if not rows and NO_SESSIONS not in out:
        raise ColabError(f"cannot parse `colab sessions`: {out[-1500:]}")
    return rows


def tagged(out: str, tag: str) -> dict[str, Any]:
    """The JSON after the last line starting with `tag `."""
    for line in reversed(out.splitlines()):
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1 :])
    raise ColabError(f"no {tag} line in remote output:\n{out[-3000:]}")


@dataclasses.dataclass(frozen=True)
class Colab:
    """A Colab account driven through the CLI.

    Attributes:
        invoke: Runs one CLI call (argv, timeout, check) and returns its output.
        probe: Session probe (name, ping) -> state; replaceable in tests.
        clock: Time source for the waits.
        accelerator: `tpu:v5e1`, `gpu:T4` or `cpu`.
    """

    invoke: Invoke = cli_invoke
    probe: Probe = session_probe
    clock: Clock = dataclasses.field(default_factory=Clock)
    accelerator: str = "cpu"

    def call(self, *args: str, timeout: float = 600, check: bool = True) -> str:
        """One CLI call."""
        return self.invoke(args, timeout, check)

    def remote(self, name: str, code: str, timeout: int = 600, tag: str | None = None) -> Any:
        """Executes python `code` in the session's kernel.

        Returns:
            The full output, or the JSON after `tag` when given.
        """
        fd, path = tempfile.mkstemp(suffix=".py", prefix="colab_exec_")
        with os.fdopen(fd, "w", newline="\n") as f:
            f.write(code)
        try:
            out = self.call(
                "exec", "-s", name, "-f", path, "--timeout", str(timeout), timeout=timeout + 120
            )
        finally:
            Path(path).unlink()
        return out if tag is None else tagged(out, tag)

    def upload(self, name: str, local: Path, remote_path: str, timeout: int = 1800) -> str:
        """Uploads a file into the session."""
        return self.call("upload", "-s", name, str(local), remote_path, timeout=timeout)

    def download(self, name: str, remote_path: str, local: Path, timeout: int = 1800) -> str:
        """Downloads a file from the session."""
        return self.call("download", "-s", name, remote_path, str(local), timeout=timeout)

    def active_assignments(self) -> int:
        """Number of active assignments on the account.

        Raises:
            ColabError: If `colab usage` cannot be parsed.
        """
        out = self.call("usage", timeout=120)
        for line in out.splitlines():
            if "Active assignments" in line:
                return int(line.split(":")[1].strip())
        raise ColabError(f"cannot parse `colab usage`: {out}")

    def new_flags(self) -> list[str]:
        """CLI flags that select the runtime of `accelerator`."""
        kind, _, variant = self.accelerator.partition(":")
        flag = ACCELERATORS[kind]
        return [flag, variant] if flag else []

    def new_session(self, name: str, tries: int = NEW_TRIES) -> None:
        """`colab new -s name` with retries.

        After a failed attempt it refuses to continue when the server holds an assignment the CLI
        does not know (an orphan: stop it by hand).

        Raises:
            AssignmentError: On an orphan assignment.
            ColabError: If no runtime could be created.
        """
        for _ in range(tries):
            try:
                out = self.call("new", "-s", name, *self.new_flags(), timeout=HTTP_TIMEOUT + 300)
                if "READY" in out:
                    return
            except ColabError, subprocess.TimeoutExpired:
                pass
            sessions = self.call("sessions", timeout=120)
            rows = assignment_rows(sessions)
            if name in rows:
                return
            if "?" in rows:
                raise AssignmentError(
                    f"{rows.count('?')} orphan assignment(s) (not {name!r}): {sessions}"
                )
            self.clock.sleep(NEW_RETRY_S)
        raise ColabError(f"could not create a {self.accelerator} runtime after {tries} attempts")

    def gone(self, name: str) -> tuple[bool, int]:
        """(session `name` is gone, the other active assignments).

        Both come from the server's assignment list (`colab sessions`), so a finished job is
        recognised while another one runs. `colab usage` is not used: it shows an unassigned
        runtime for about 5 more minutes.
        """
        rows = assignment_rows(self.call("sessions", timeout=120))
        return name not in rows, sum(1 for r in rows if r != name)

    def wait_gone(self, name: str, wait_s: float = STOP_WAIT_S) -> tuple[bool, int]:
        """Polls `gone` until true or `wait_s` passed (unassigning shows in usage after minutes)."""
        t0, n, ok = self.clock.now(), 0, False
        while self.clock.now() - t0 < wait_s:
            ok, n = self.gone(name)
            if ok:
                break
            self.clock.sleep(STOP_POLL_S)
        return ok, n

    def stop(self, name: str, wait_s: float = STOP_WAIT_S) -> int:
        """Stops the runtime and verifies it is unassigned.

        Returns:
            The other assignments still active.

        Raises:
            AssignmentError: If the session is still assigned after `wait_s`.
        """
        try:
            self.call("stop", "-s", name, timeout=300, check=False)
        finally:
            ok, n = self.wait_gone(name, wait_s)
        if not ok:
            raise AssignmentError(
                f"STILL {n} ACTIVE COLAB ASSIGNMENT(S): stop them by hand "
                "(colab --auth adc sessions)"
            )
        return n

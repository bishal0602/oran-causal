"""SSH lane: run a bundled job on a Linux host inside a resource-capped systemd scope.

Everything lives under the profile's base directory: uv, the managed Python, caches, one venv and
one bundle per job. Compute runs only as `systemd-run --scope` with the profile's limits, after the
profile's pre-launch refusals. Host-level changes (packages, services, containers, reboots) are
never made, and `stop` stops only the job's own scope.
"""

import dataclasses
import io
import json
import re
import subprocess
import tarfile
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from oran_causal.experiments.campaign.provenance import GitRunner, code_info, run_git
from oran_causal.experiments.launch import bundle
from oran_causal.experiments.launch.commands import PIN_CHECK, SetupPlan, sync_command
from oran_causal.experiments.launch.hosts import VPS_SHARED, HostProfile
from oran_causal.experiments.launch.runner import Dirs

HOST = "root@vps.brihatech.com"
UV_TARGET = "uv-x86_64-unknown-linux-gnu"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")
OUT_PLACEHOLDER = "{OUT}"
PROCS_PLACEHOLDER = "{PROCS}"
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20")


class Refused(RuntimeError):
    """A safety refusal: the launch must not go ahead."""


class RemoteError(RuntimeError):
    """A remote command failed."""


class Transport(Protocol):
    """How scripts and bytes reach the host."""

    def script(self, body: str, timeout: int = 600, *, check: bool = True) -> str:
        """Runs a bash script on the host (fed on stdin) and returns its output."""
        ...

    def put(self, data: bytes, remote_cmd: str, timeout: int = 1800) -> str:
        """Streams `data` into a remote command."""
        ...

    def fetch(self, remote_cmd: str, timeout: int = 1800) -> bytes:
        """Returns the stdout of a remote command."""
        ...


type Run = Callable[..., subprocess.CompletedProcess[bytes]]


@dataclasses.dataclass(frozen=True)
class Ssh:
    """`Transport` over the `ssh` client.

    Attributes:
        env: Shell prelude that confines the remote environment to the base directory.
        host: `user@host`.
        run: `subprocess.run` replacement (tests).
    """

    env: str
    host: str = HOST
    run: Run = dataclasses.field(default=subprocess.run)

    def _ssh(self, remote: Sequence[str], data: bytes | None, timeout: int) -> tuple[int, bytes]:
        p = self.run(
            ["ssh", *SSH_OPTS, self.host, *remote],
            input=data,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout + (p.stderr if p.returncode else b"")

    def script(self, body: str, timeout: int = 600, *, check: bool = True) -> str:
        """Runs `set -eu`, the confined environment and `body` with bash on the host."""
        full = f"set -eu\n{self.env}\n{body}\n"
        code, out = self._ssh(["bash", "-s"], full.encode(), timeout)
        text = out.decode(errors="replace")
        if check and code != 0:
            raise RemoteError(f"remote rc {code}:\n{text[-3000:]}")
        return text

    def put(self, data: bytes, remote_cmd: str, timeout: int = 1800) -> str:
        """Streams `data` into `remote_cmd`."""
        code, out = self._ssh([f"set -eu; {self.env}; {remote_cmd}"], data, timeout)
        text = out.decode(errors="replace")
        if code != 0:
            raise RemoteError(f"remote rc {code}:\n{text[-3000:]}")
        return text

    def fetch(self, remote_cmd: str, timeout: int = 1800) -> bytes:
        """Stdout of `remote_cmd`."""
        code, out = self._ssh([remote_cmd], None, timeout)
        if code != 0:
            raise RemoteError(out.decode(errors="replace")[-2000:])
        return out


def remote_env(base: str) -> str:
    """Shell prelude: HOME, XDG, uv and temp directories all inside `base`."""
    return (
        f"export HOME={base}/home XDG_CONFIG_HOME={base}/home/.config XDG_CACHE_HOME={base}/cache "
        f"XDG_DATA_HOME={base}/home/.local/share UV_CACHE_DIR={base}/cache/uv "
        f"UV_PYTHON_INSTALL_DIR={base}/python UV_PYTHON_BIN_DIR={base}/bin "
        f"UV_TOOL_DIR={base}/tools UV_NO_CONFIG=1 UV_NO_MODIFY_PATH=1 "
        f"TMPDIR={base}/tmp PATH={base}/bin:$PATH"
    )


def check_name(name: str) -> str:
    """Returns `name` when it is a valid job name.

    Raises:
        ValueError: Otherwise.
    """
    if not NAME_RE.match(name):
        raise ValueError(f"bad job name {name!r} (lowercase letters, digits, '-')")
    return name


@dataclasses.dataclass(frozen=True)
class HostState:
    """Read-only state of the host.

    Attributes:
        load1: 1-minute load average.
        mem_available_kb: `MemAvailable`.
        our_procs: Processes in our active scopes.
        units: Our active scope units.
    """

    load1: float
    mem_available_kb: int
    our_procs: int
    units: tuple[str, ...]

    @property
    def other_load(self) -> float:
        """Load not caused by our own processes."""
        return round(self.load1 - self.our_procs, 2)


def check_launch(profile: HostProfile, state: HostState) -> None:
    """Refuses a launch the profile's guard does not allow.

    Raises:
        Refused: With the reason.
    """
    guard = profile.load_guard
    if guard is None:
        return
    if guard.one_job_at_a_time and state.units:
        raise Refused(f"an oran-causal job is active ({' '.join(state.units)}); one job at a time")
    if state.mem_available_kb < guard.min_mem_available_kb:
        have, need = state.mem_available_kb / 2**20, guard.min_mem_available_kb / 2**20
        raise Refused(f"MemAvailable {have:.1f} GB < {need:.0f} GB")
    if state.other_load > guard.max_other_load:
        raise Refused(f"1-min load from others {state.other_load:.2f} > {guard.max_other_load}")


STATE_SCRIPT = """
read l1 l5 l15 rest < /proc/loadavg
ma=$(awk '/MemAvailable/ {{print $2}}' /proc/meminfo)
units=$(systemctl list-units --type=scope --state=active --no-legend --plain '{prefix}*' \
| awk '{{print $1}}')
ours=0
for u in $units; do
  cg=/sys/fs/cgroup$(systemctl show -p ControlGroup --value $u)/cgroup.procs
  if [ -f $cg ]; then ours=$((ours + $(wc -l < $cg))); fi
done
echo "{{\\"load1\\": $l1, \\"mem_available_kb\\": $ma, \\"our_procs\\": $ours, \
\\"units\\": \\"$(echo $units)\\"}}"
"""


@dataclasses.dataclass(frozen=True)
class Vps:
    """A host reached through a `Transport`, under a `HostProfile`.

    Attributes:
        transport: SSH or a test double.
        profile: Caps and refusals; its `base` is the directory everything lives under.
    """

    transport: Transport
    profile: HostProfile = VPS_SHARED

    @property
    def base(self) -> str:
        """Base directory on the host."""
        if not self.profile.base:
            raise ValueError(f"profile {self.profile.name} has no base directory")
        return self.profile.base

    def job_dir(self, name: str) -> str:
        """Directory of job `name` on the host."""
        return f"{self.base}/jobs/{check_name(name)}"

    def unit(self, name: str) -> str:
        """Systemd scope unit of job `name`."""
        return self.profile.unit_prefix + check_name(name)

    def state(self) -> HostState:
        """Host load, memory and our running scopes."""
        out = self.transport.script(STATE_SCRIPT.format(prefix=self.profile.unit_prefix), 120)
        raw = json.loads(out.strip().splitlines()[-1])
        return HostState(
            float(raw["load1"]),
            int(raw["mem_available_kb"]),
            int(raw["our_procs"]),
            tuple(raw["units"].split()),
        )

    def probe(self) -> str:
        """Read-only host check."""
        body = (
            "grep -m1 'model name' /proc/cpuinfo; nproc; df -h /opt | tail -1; "
            f"du -sh {self.base} 2>/dev/null || echo '{self.base} absent'"
        )
        return (
            json.dumps(dataclasses.asdict(self.state())) + "\n" + self.transport.script(body, 120)
        )

    def setup(self, uv_version: str, python: str) -> str:
        """Base tree, the uv release binary (sha256-checked) and the managed Python, all in base."""
        b = self.base
        url = f"https://github.com/astral-sh/uv/releases/download/{uv_version}/{UV_TARGET}.tar.gz"
        return self.transport.script(
            f"""
mkdir -p {b}/bin {b}/python {b}/cache {b}/home {b}/tmp {b}/jobs {b}/tools
cd {b}/tmp
if ! {b}/bin/uv --version 2>/dev/null | grep -q " {uv_version}"; then
  curl -fsSL -o uv.tgz {url}
  curl -fsSL -o uv.tgz.sha256 {url}.sha256
  echo "$(cut -d' ' -f1 uv.tgz.sha256)  uv.tgz" | sha256sum -c -
  tar -xzf uv.tgz
  install -m 0755 {UV_TARGET}/uv {UV_TARGET}/uvx {b}/bin/
  rm -rf uv.tgz uv.tgz.sha256 {UV_TARGET}
fi
{b}/bin/uv --version
nice -n 19 ionice -c3 {b}/bin/uv python install {python}
{b}/bin/uv python find {python}
du -sh {b}
""",
            1800,
        )

    def push(
        self, name: str, req: bundle.BundleRequest, dirs: Dirs, git: GitRunner | None = None
    ) -> dict[str, Any]:
        """Uploads the bundle of `req` to `<job>/b` and records `push.json` locally.

        Returns:
            The bundle manifest.
        """
        jd = self.job_dir(name)
        blobs = bundle.build_blobs(req)
        man = bundle.build_manifest(req, blobs, git)
        data = bundle.tar_bytes(blobs, man)
        out = self.transport.put(
            data,
            f"rm -rf {jd}/b && mkdir -p {jd}/b {jd}/out && tar -xzf - -C {jd}/b && "
            f"echo pushed $(find {jd}/b -type f | wc -l) files",
        )
        record = {
            "name": name,
            "root": str(req.root),
            "commit": man["git_head"],
            "dirty": man["dirty"],
            "n_files": len(blobs),
            "bundle_bytes": len(data),
            "extra": list(req.extra),
        }
        dst = dirs.runs / name
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "push.json").write_text(json.dumps(record, indent=1))
        return {"remote": out.strip(), **record}

    def venv_script(self, name: str, plan: SetupPlan, python: str) -> str:
        """The per-job venv step: `uv sync --frozen` of the bundle's lock, then the pin check.

        The venv is per job, because workspace members are installed editable and point into the
        job's bundle; the uv cache makes the installs hardlinks.
        """
        jd = self.job_dir(name)
        vd = f"{jd}/venv"
        return f"""
cd {jd}/b
rm -rf {vd}
export UV_PROJECT_ENVIRONMENT={vd}
nice -n 19 ionice -c3 {sync_command(plan)}
v=$({vd}/bin/python --version 2>&1); echo "$v"
case "$v" in "Python {python}"*) ;; *) echo "interpreter mismatch: $v != {python}"; exit 4;; esac
grep '^home' {vd}/pyvenv.cfg
PATH={vd}/bin:$PATH {PIN_CHECK} > {jd}/out/pin_check.json 2>&1; cat {jd}/out/pin_check.json
du -sh {vd}
"""

    def venv(self, name: str, python: str) -> str:
        """Creates the job's venv and runs the pin check into `<job>/out/pin_check.json`."""
        plan = SetupPlan(f"{self.job_dir(name)}/out", gpu=False, torch=True)
        return self.transport.script(self.venv_script(name, plan, python), 3600)

    def run_script(self, name: str, cmd: str, procs: int) -> str:
        """`run.sh` of a job: confined environment, venv on PATH, the command, its exit code."""
        jd, b = self.job_dir(name), self.base
        full = cmd.replace(OUT_PLACEHOLDER, f"{jd}/out").replace(PROCS_PLACEHOLDER, str(procs))
        threads = " ".join(
            f"{v}_NUM_THREADS=1" for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR", "NUMBA")
        )
        return "\n".join(
            [
                "#!/bin/bash",
                "set -u",
                remote_env(b),
                f"cd {jd}/b",
                f"export PATH={jd}/venv/bin:$PATH {threads}",
                f"echo started $(date -u +%FT%TZ) >> {jd}/run.out",
                full,
                "rc=$?",
                f"echo $rc > {jd}/exit_code.tmp && mv {jd}/exit_code.tmp {jd}/exit_code",
                f"echo finished rc $rc $(date -u +%FT%TZ) >> {jd}/run.out",
                "",
            ]
        )

    def launch_script(self, name: str, run_sh: str, procs: int) -> str:
        """Writes `run.sh` and starts it in a capped scope; fails if the scope is not active."""
        jd, unit = self.job_dir(name), self.unit(name)
        limits = " ".join(self.profile.scope_limits(procs))
        return f"""
cat > {jd}/run.sh <<'ORAN_CAUSAL_EOF'
{run_sh}
ORAN_CAUSAL_EOF
chmod 0755 {jd}/run.sh
rm -f {jd}/exit_code
cd {jd}
nohup setsid systemd-run --scope --quiet --unit {unit} {limits} nice -n 19 ionice -c3 {jd}/run.sh \\
  >> {jd}/run.out 2>&1 < /dev/null &
sleep 5
systemctl is-active {unit}.scope || (echo NOT-ACTIVE; tail -20 {jd}/run.out; exit 1)
systemctl show {unit}.scope -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax
"""

    def launch(
        self,
        name: str,
        cmd: str,
        procs: int,
        dirs: Dirs,
        git: GitRunner | None = None,
    ) -> dict[str, Any]:
        """Checks the pushed bundle and the host, then starts the job.

        Raises:
            Refused: If nothing was pushed, the pushed bundle is not the clean current commit, or
                the host guard refuses.
        """
        self.profile.check_procs(procs)
        pj = dirs.runs / name / "push.json"
        if not pj.exists():
            raise Refused("push first")
        pushed = json.loads(pj.read_text())
        code = code_info({}, git or (lambda a: run_git(a, dirs.root)), dirs.root)
        if pushed["commit"] != code["commit"] or pushed["dirty"] or code["dirty"]:
            raise Refused("the pushed bundle is not the clean current commit (commit, push again)")
        state = self.state()
        check_launch(self.profile, state)
        out = self.transport.script(
            self.launch_script(name, self.run_script(name, cmd, procs), procs), 300
        )
        record = {
            "name": name,
            "unit": self.unit(name),
            "cmd": cmd,
            "procs": procs,
            "limits": list(self.profile.scope_limits(procs)),
            "commit": code["commit"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preflight": dataclasses.asdict(state),
            "remote": out.strip(),
        }
        (dirs.runs / name / "launch.json").write_text(json.dumps(record, indent=1))
        return record

    def status(self, name: str) -> str:
        """Scope state, exit code, shard line counts and log tails of a job."""
        jd, unit = self.job_dir(name), self.unit(name)
        body = f"""
echo "scope: $(systemctl is-active {unit}.scope 2>/dev/null || true)"
[ -f {jd}/exit_code ] && echo "exit_code: $(cat {jd}/exit_code)" || echo "exit_code: none"
for f in {jd}/out/*res_*.jsonl; do [ -f "$f" ] && echo "$(basename $f) $(wc -l < $f)"; done
for f in {jd}/out/rc_*.txt; do [ -f "$f" ] && echo "$(basename $f .txt) exit $(cat $f)"; done
tail -n 3 {jd}/run.out 2>/dev/null || true
for f in {jd}/out/log_*.txt; do [ -f "$f" ] && echo "-- $(basename $f): $(tail -n 1 $f)"; done
"""
        return (
            json.dumps(dataclasses.asdict(self.state()))
            + "\n"
            + self.transport.script(body, 120, check=False)
        )

    def pull(self, name: str, dirs: Dirs) -> int:
        """Merges the job's out dir into `runs/<name>`.

        A local file is never replaced by a smaller one.

        Returns:
            Number of files written.
        """
        jd = self.job_dir(name)
        data = self.transport.fetch(
            f"cd {jd} && tar -czf - out run.out $(ls exit_code 2>/dev/null)"
        )
        dst = dirs.runs / name
        written = 0
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                target = (dst / m.name).resolve()
                if not m.isfile() or not target.is_relative_to(dst.resolve()):
                    continue
                if target.exists() and target.stat().st_size > m.size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                member = tf.extractfile(m)
                if member is not None:
                    target.write_bytes(member.read())
                    written += 1
        return written

    def stop(self, name: str) -> str:
        """Stops the job's own scope."""
        unit = self.unit(name)
        return self.transport.script(
            f"systemctl stop {unit}.scope && echo stopped {unit} || echo 'not active: {unit}'",
            300,
            check=False,
        )

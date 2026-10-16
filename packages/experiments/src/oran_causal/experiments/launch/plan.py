"""From a launch request to the session a platform runs: command, bundle files, limits."""

import dataclasses
import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from oran_causal.experiments.campaign.provenance import GitRunner, code_info, run_git
from oran_causal.experiments.campaign.spec import ProtocolTexts, is_eval
from oran_causal.experiments.launch import bundle, colab, commands, hosts, kaggle, vps
from oran_causal.experiments.launch.colab_session import Colab
from oran_causal.experiments.launch.runner import Dirs
from oran_causal.experiments.paths import WORKSPACE_ROOT

type Platform = Literal["kaggle", "colab", "vps"]

KAGGLE_OUT = "$JOB_OUT"
COLAB_OUT = "xm_out"
COLAB_EVAL_OUT = "xm_eval_out"
EVAL_SUBDIR = "eval"
SELFTEST_EXTRA = ("packages/experiments/tests",)
PROFILES: dict[str, hosts.HostProfile] = {
    "kaggle": hosts.KAGGLE_CPU,
    "colab": hosts.COLAB_CPU,
    "vps": hosts.VPS_SHARED,
}


class LaunchRefused(RuntimeError):
    """A launch check failed; nothing was started."""


@dataclasses.dataclass(frozen=True)
class LaunchRequest:
    """What the user asked to launch.

    Attributes:
        platform: Where the session runs.
        name: Job name; new per launch.
        spec: Spec path (absolute or relative to `root`).
        parts: Number of parts the spec is split into.
        part_ids: The parts this session runs; None means all.
        cost_table: Cost table path, or None.
        paths: Extra workspace-relative files, directories or globs added to the bundle.
        skip_from: Workspace-relative globs of earlier shard files: complete datasets are skipped.
        wall_s: Per-process timeout; None means the platform default.
        selftest: Run the campaign tests after the install.
        gpu: GPU session with the CUDA torch build.
        procs: VPS processes; None means the profile maximum.
        torch: Install torch.
        root: The workspace.
    """

    platform: Platform
    name: str
    spec: Path
    parts: int = 4
    part_ids: Sequence[int] | None = None
    cost_table: Path | None = None
    paths: Sequence[str] = ()
    skip_from: Sequence[str] = ()
    wall_s: int | None = None
    selftest: bool = False
    gpu: bool = False
    procs: int | None = None
    torch: bool = True
    root: Path = WORKSPACE_ROOT


@dataclasses.dataclass(frozen=True)
class SessionPlan:
    """One session, ready to launch.

    Attributes:
        request: The request it came from.
        cmd: The session command.
        extra: Workspace-relative files added to the base bundle.
        out_dir: Output directory in the session.
        eval: Whether the spec is an EVAL spec.
        frozen_protocol: Verified frozen protocol bytes for the EVAL guard, or None.
        procs: Processes the session runs.
        profile: The platform's caps.
        code: Launcher's `commit` / `dirty` check.
    """

    request: LaunchRequest
    cmd: str
    extra: tuple[str, ...]
    out_dir: str
    eval: bool
    frozen_protocol: bytes | None
    procs: int
    profile: hosts.HostProfile
    code: dict[str, Any]


def rel(path: Path | str, root: Path) -> str:
    """Workspace-relative posix path of `path`.

    Raises:
        ValueError: If `path` is outside the workspace.
    """
    p = Path(path)
    p = p if p.is_absolute() else root / p
    return p.resolve().relative_to(root.resolve()).as_posix()


def expand_globs(patterns: Sequence[str], root: Path) -> list[str]:
    """Workspace files matching the globs (a plain path matches itself).

    Raises:
        FileNotFoundError: If a pattern matches nothing.
    """
    out: set[str] = set()
    for pattern in patterns:
        hits = [p for p in sorted(root.glob(pattern.replace("\\", "/"))) if p.is_file()]
        if not hits:
            raise FileNotFoundError(f"nothing at {pattern!r}")
        out.update(p.relative_to(root).as_posix() for p in hits)
    return sorted(out)


def session_out(platform: Platform, is_eval_spec: bool) -> str:
    """Output directory of a session."""
    if platform == "kaggle":
        return f"{KAGGLE_OUT}/{EVAL_SUBDIR}" if is_eval_spec else KAGGLE_OUT
    if platform == "colab":
        return COLAB_EVAL_OUT if is_eval_spec else COLAB_OUT
    return vps.OUT_PLACEHOLDER


def make_plan(
    req: LaunchRequest,
    spec: dict[str, Any],
    texts: ProtocolTexts,
    git: GitRunner | None = None,
) -> SessionPlan:
    """The session of `req`.

    Args:
        req: The request.
        spec: The parsed spec.
        texts: The protocol texts the EVAL guard reads.
        git: Git runner for the launcher's commit and dirty check.

    Raises:
        LaunchRefused: If the spec may not be launched (EVAL authorisation, dirty tree) or the
            parts do not fit the platform.
    """
    root = req.root
    code = code_info({}, git or (lambda args: run_git(args, root)), root)
    errors = commands.launch_checks(spec, code, texts)
    if errors:
        raise LaunchRefused("launch refused:\n  " + "\n  ".join(errors))
    ev = is_eval(spec)
    ids = list(req.part_ids) if req.part_ids is not None else list(range(req.parts))
    profile = PROFILES[req.platform]
    procs = profile.check_procs(req.procs or profile.max_procs)
    if req.platform == "vps" and len(ids) > procs and not ev:
        raise LaunchRefused(f"{len(ids)} parts > --procs {procs}: one process per part on the VPS")
    spec_rel = rel(req.spec, root)
    cost_rel = rel(req.cost_table, root) if req.cost_table else None
    out = session_out(req.platform, ev)
    skip = [x.replace("\\", "/") for x in req.skip_from]
    parts = commands.PartsPlan(
        spec_rel,
        req.parts,
        ids,
        out,
        req.platform,
        req.wall_s or profile.wall_s,
        code,
        cost_rel,
        "eval_res" if ev else "res",
        registry=not ev,
        gpu=req.gpu and req.platform != "vps",
        skip_from=skip,
        one_per_cpu=ev,
    )
    run = commands.cloud_cmd(parts)
    if req.platform == "vps":
        cmd = run
    else:
        setup = commands.SetupPlan(out, req.gpu, req.torch, req.selftest)
        cmd = commands.setup_cmd(setup) + run
    extra = [spec_rel, *([cost_rel] if cost_rel else []), *expand_globs(req.paths, root)]
    extra += expand_globs(skip, root) if skip else []
    extra += SELFTEST_EXTRA if req.selftest and req.platform != "vps" else ()
    frozen = bundle.frozen_protocol(spec, root) if ev else None
    return SessionPlan(
        req,
        cmd,
        tuple(dict.fromkeys(extra)),
        out,
        ev,
        frozen,
        min(procs, len(ids)),
        profile,
        code,
    )


@dataclasses.dataclass(frozen=True)
class Targets:
    """The clients a plan is executed with; only the one for its platform is needed.

    Attributes:
        dirs: Local directories.
        kaggle_client: Kaggle CLI client.
        colab_env: Colab job environment (account, scheduler).
        vps_host: The VPS.
        git: Git runner for the manifests.
    """

    dirs: Dirs
    kaggle_client: kaggle.KaggleClient | None = None
    colab_env: colab.ColabEnv | None = None
    vps_host: vps.Vps | None = None
    git: GitRunner | None = None


def kaggle_job(plan: SessionPlan) -> kaggle.KaggleJob:
    """The Kaggle job of a plan."""
    r = plan.request
    return kaggle.KaggleJob(
        r.name, plan.cmd, plan.extra, (), r.gpu, frozen_protocol=plan.frozen_protocol
    )


def colab_job(plan: SessionPlan, every_min: int = 10) -> colab.ColabJob:
    """The Colab job of a plan (one thread per process)."""
    r = plan.request
    return colab.ColabJob(
        r.name,
        plan.cmd,
        plan.extra,
        plan.out_dir,
        1,
        plan.frozen_protocol,
        every_min,
    )


def colab_accelerator(plan: SessionPlan) -> str:
    """Runtime of a Colab plan."""
    return "gpu:T4" if plan.request.gpu else "cpu"


def vps_request(plan: SessionPlan) -> bundle.BundleRequest:
    """The bundle a VPS plan pushes."""
    r = plan.request
    return bundle.BundleRequest(r.name, r.root, plan.extra, plan.frozen_protocol, plan.cmd)


def describe(plan: SessionPlan) -> list[str]:
    """The steps a launch would take, one line each (the dry run)."""
    r = plan.request
    steps = [f"bundle {r.name}: base + {len(plan.extra)} extra: {' '.join(plan.extra)}"]
    if r.platform == "vps":
        steps += [
            f"vps push {r.name}",
            f"vps venv {r.name}",
            f"vps launch {r.name} --procs {plan.procs} "
            f"limits {' '.join(plan.profile.scope_limits(plan.procs))}",
        ]
    elif r.platform == "kaggle":
        steps += [
            f"kaggle datasets create {r.name}-code",
            f"kaggle kernels push {r.name} (gpu={r.gpu})",
        ]
    else:
        steps += [f"colab new {r.name} ({colab_accelerator(plan)})", f"colab job {r.name}"]
    return [*steps, f"cmd: {shlex.quote(plan.cmd)}"]


def execute(plan: SessionPlan, targets: Targets) -> dict[str, Any]:
    """Launches the plan on its platform.

    Raises:
        ValueError: If `targets` lacks the client for the plan's platform.
    """
    r = plan.request
    if r.platform == "kaggle":
        if targets.kaggle_client is None:
            raise ValueError("no Kaggle client")
        return kaggle.launch(kaggle_job(plan), targets.kaggle_client, targets.dirs, targets.git)
    if r.platform == "colab":
        if targets.colab_env is None:
            raise ValueError("no Colab environment")
        env = dataclasses.replace(targets.colab_env, colab=colab_for(plan, targets.colab_env.colab))
        return colab.launch_job(colab_job(plan), env, r.root)
    if targets.vps_host is None:
        raise ValueError("no VPS")
    host = targets.vps_host
    pushed = host.push(r.name, vps_request(plan), targets.dirs, targets.git)
    python = (r.root / ".python-version").read_text().strip()
    venv = host.venv(r.name, python)
    launched = host.launch(r.name, plan.cmd, plan.procs, targets.dirs, targets.git)
    return {"push": pushed, "venv": venv, "launch": launched}


def spec_of(path: Path) -> dict[str, Any]:
    """The parsed spec file."""
    return json.loads(path.read_bytes())


def colab_for(plan: SessionPlan, base: Colab | None = None) -> Colab:
    """The `Colab` driver with the accelerator the plan needs."""
    return dataclasses.replace(base or Colab(), accelerator=colab_accelerator(plan))


def launcher_texts(root: Path) -> ProtocolTexts:
    """Protocol texts for the launcher's EVAL guard: the live file and the freeze commit's blob."""
    live = root / bundle.PROTOCOL_REL
    return ProtocolTexts(
        live=live.read_bytes() if live.exists() else None,
        at_commit=lambda commit: bundle.git_blob(commit, bundle.PROTOCOL_REL, root),
    )

"""Command line of the cloud launchers: `python -m oran_causal.experiments.launch <command>`."""

import argparse
import dataclasses
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from oran_causal.experiments.launch import colab, dispatch, fingerprint, kaggle, live, plan
from oran_causal.experiments.launch.colab_session import Colab
from oran_causal.experiments.launch.runner import Dirs
from oran_causal.experiments.launch.schedule import TaskScheduler

PLATFORMS = ("kaggle", "colab", "vps")
COLAB_ACTIONS = ("tick", "status", "pull", "stop", "schedule", "unschedule")
DISPATCH_ACTIONS = ("init", "tick", "show", "schedule", "unschedule")
VPS_ACTIONS = ("status", "pull", "stop", "probe")
WORK_DIR = Path(tempfile.gettempdir()) / "oran-causal-launch"

type TargetsFactory = Callable[[], plan.Targets]


def default_targets() -> plan.Targets:
    """Real clients: Kaggle CLI, Colab CLI with Task Scheduler, the shared VPS over SSH."""
    dirs = Dirs(WORK_DIR)
    return plan.Targets(
        dirs,
        kaggle.KaggleClient(),
        colab.ColabEnv(Colab(), dirs, TaskScheduler()),
        live.default_ssh_vps(),
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    ap = argparse.ArgumentParser(prog="python -m oran_causal.experiments.launch")
    sub = ap.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="launch one campaign session")
    launch.add_argument("platform", choices=PLATFORMS)
    launch.add_argument("--name", required=True)
    launch.add_argument("--spec", required=True, type=Path)
    launch.add_argument("--parts", type=int, default=4)
    launch.add_argument("--part-set", default=None, help="'i,j,...' subset of the parts")
    launch.add_argument("--cost-table", type=Path, default=None)
    launch.add_argument("--paths", nargs="*", default=[])
    launch.add_argument("--skip-complete-from", nargs="*", default=[])
    launch.add_argument("--wall-s", type=int, default=None)
    launch.add_argument("--selftest", action="store_true")
    launch.add_argument("--gpu", action="store_true")
    launch.add_argument("--procs", type=int, default=None)
    launch.add_argument("--dry-run", action="store_true")

    for platform, actions in (("colab", COLAB_ACTIONS), ("vps", VPS_ACTIONS)):
        p = sub.add_parser(platform)
        p.add_argument("action", choices=actions)
        p.add_argument("name", nargs="?")
        p.add_argument("--every", type=int, default=10)
    k = sub.add_parser("kaggle")
    k.add_argument("action", choices=("status", "pull"))
    k.add_argument("name")
    k.add_argument("--all", action="store_true", help="pull every file, including *.npz caches")

    d = sub.add_parser("dispatch")
    d.add_argument("action", choices=DISPATCH_ACTIONS)
    d.add_argument("name")
    d.add_argument("--spec")
    d.add_argument("--cost-table")
    d.add_argument("--parts", type=int)
    d.add_argument("--kaggle-chunk", type=int, default=dispatch.KAGGLE_CHUNK)
    d.add_argument("--kaggle-mine-max", type=int, default=dispatch.KAGGLE_MINE_MAX)
    d.add_argument("--colab-chunk", type=int, default=dispatch.COLAB_CHUNK)
    d.add_argument("--skip-complete-from", nargs="*", default=[])
    d.add_argument("--vps", action="store_true")
    d.add_argument("--only", nargs="+", choices=PLATFORMS, default=None)
    d.add_argument("--every", type=int, default=15)
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--kaggle-now", nargs=2, type=int, default=None, metavar=("RUNNING", "MINE"))

    sub.add_parser("fingerprint", help="print this machine's numeric fingerprint")
    return ap


def request_of(a: argparse.Namespace) -> plan.LaunchRequest:
    """The launch request of the parsed `launch` arguments."""
    ids = [int(x) for x in a.part_set.split(",")] if a.part_set else None
    return plan.LaunchRequest(
        a.platform,
        a.name,
        a.spec,
        a.parts,
        ids,
        a.cost_table,
        a.paths,
        a.skip_complete_from,
        a.wall_s,
        a.selftest,
        a.gpu,
        a.procs,
    )


def run_launch(
    a: argparse.Namespace, make_targets: TargetsFactory, out: Callable[[str], None]
) -> int:
    """`launch <platform>`: plan, refuse or describe (dry run), else launch."""
    request = request_of(a)
    root = request.root
    spec = plan.spec_of(request.spec if request.spec.is_absolute() else root / request.spec)
    request = (
        request
        if request.spec.is_absolute()
        else dataclasses.replace(request, spec=root / request.spec)
    )
    try:
        session = plan.make_plan(request, spec, plan.launcher_texts(root))
    except plan.LaunchRefused as e:
        out(f"[launch] {e}")
        return 2
    for line in plan.describe(session):
        out(line)
    if a.dry_run:
        return 0
    result = plan.execute(session, make_targets())
    out(json.dumps(result, indent=1, default=str))
    return 0


def run_colab(
    a: argparse.Namespace, make_targets: TargetsFactory, out: Callable[[str], None]
) -> int:
    """`colab <action> NAME`: tick, status, pull, stop, schedule, unschedule."""
    env = make_targets().colab_env
    if env is None or not a.name:
        raise SystemExit("colab: a job NAME is required")
    result: Any
    match a.action:
        case "tick":
            result = colab.tick(a.name, env)
        case "status":
            state = colab.status(a.name, env)
            result = {**colab.brief(state), "log_tail": state["log_tail"]}
        case "pull":
            result = colab.pull(a.name, env)
        case "stop":
            result = colab.stop(a.name, env)
        case "schedule":
            result = env.arm(a.name, a.every)
        case _:
            result = env.disarm(a.name)
    out(json.dumps(result, default=str))
    return 0


def run_kaggle(
    a: argparse.Namespace, make_targets: TargetsFactory, out: Callable[[str], None]
) -> int:
    """`kaggle status|pull NAME`."""
    targets = make_targets()
    client = targets.kaggle_client
    if client is None:
        raise SystemExit("no Kaggle client")
    if a.action == "status":
        out(client.status(a.name))
        return 0
    pattern = None if a.all else kaggle.DEFAULT_PULL
    out(json.dumps(kaggle.pull(a.name, client, targets.dirs.runs, pattern), default=str))
    return 0


def run_vps(a: argparse.Namespace, make_targets: TargetsFactory, out: Callable[[str], None]) -> int:
    """`vps probe | status|pull|stop NAME`."""
    targets = make_targets()
    host = targets.vps_host
    if host is None:
        raise SystemExit("no VPS")
    if a.action == "probe":
        out(host.probe())
    elif not a.name:
        raise SystemExit("vps: a job NAME is required")
    elif a.action == "status":
        out(host.status(a.name))
    elif a.action == "pull":
        out(f"{host.pull(a.name, targets.dirs)} files written")
    else:
        out(host.stop(a.name))
    return 0


def run_dispatch(
    a: argparse.Namespace, make_targets: TargetsFactory, out: Callable[[str], None]
) -> int:
    """`dispatch init|tick|show|schedule|unschedule NAME`."""
    targets = make_targets()
    scheduler = targets.colab_env.scheduler if targets.colab_env else TaskScheduler()
    disp = dispatch.Dispatcher(
        targets.dirs,
        live.LiveBackend(targets),
        scheduler=scheduler,
        dry_run=a.dry_run,
        kaggle_now=tuple(a.kaggle_now) if a.kaggle_now else None,
        log=out,
    )
    match a.action:
        case "init":
            if not (a.spec and a.cost_table and a.parts):
                raise SystemExit("dispatch init: --spec, --cost-table and --parts are required")
            disp.init(
                a.name,
                dispatch.Queue(
                    a.spec,
                    a.cost_table,
                    a.parts,
                    a.kaggle_chunk,
                    a.kaggle_mine_max,
                    a.colab_chunk,
                    a.skip_complete_from,
                    a.vps,
                    a.only,
                ),
            )
        case "show":
            out(json.dumps(dataclasses.asdict(disp.load(a.name)), indent=1, default=str))
        case "tick":
            out(disp.tick(a.name))
        case "schedule":
            out(disp.arm(a.name, a.every))
        case _:
            out(str(disp.unschedule(a.name)))
    return 0


def main(
    argv: Sequence[str] | None = None,
    make_targets: TargetsFactory = default_targets,
    out: Callable[[str], None] = print,
) -> int:
    """Runs one command.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.
        make_targets: Builds the platform clients (replaced in tests).
        out: Output sink.

    Returns:
        The process exit code.
    """
    a = build_parser().parse_args(argv)
    match a.command:
        case "launch":
            return run_launch(a, make_targets, out)
        case "colab":
            return run_colab(a, make_targets, out)
        case "kaggle":
            return run_kaggle(a, make_targets, out)
        case "vps":
            return run_vps(a, make_targets, out)
        case "dispatch":
            return run_dispatch(a, make_targets, out)
        case _:
            out(json.dumps(fingerprint.fingerprint()))
            return 0


if __name__ == "__main__":
    sys.exit(main())

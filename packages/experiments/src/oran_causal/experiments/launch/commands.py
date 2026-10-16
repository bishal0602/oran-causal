"""Shell commands a cloud session runs: environment setup and the campaign parts."""

import dataclasses
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from oran_causal.experiments.campaign.host import PLATFORM_ENV
from oran_causal.experiments.campaign.provenance import COMMIT_ENV, DIRTY_ENV, LOCK_ENV
from oran_causal.experiments.campaign.spec import ProtocolTexts, eval_authorised, is_eval

CAMPAIGN_MODULE = "oran_causal.experiments.campaign"
LOCK_REL = "uv.lock"
PYTHON_VERSION_REL = ".python-version"
VENV_DIR = "/tmp/oran-causal-venv"
EXPERIMENTS_PACKAGE = "oran-causal-experiments"
METHODS_PACKAGE = "oran-causal-methods"
SELFTEST_TESTS = ("packages/experiments/tests/test_campaign_*.py",)
THREADS = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
PIN_CHECK = (
    'python -c "import json; from oran_causal.experiments.campaign.provenance import '
    "pin_check, read_lock, workspace_python; "
    'print(json.dumps(pin_check(read_lock(), workspace_python())))"'
)
QUOTA_CHECK = (
    'NP=$(python -c "from oran_causal.experiments.campaign.host import cpu_quota; '
    'print(cpu_quota())"); '
)
GPU_PREAMBLE = (
    'NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1; nvidia-smi -L > '
    "{out}/gpus.txt 2>&1; "
)
GATE = "while [ $(jobs -rp | wc -l) -ge $NP ]; do wait -n; done; "


@dataclasses.dataclass(frozen=True)
class PartsPlan:
    """The campaign parts one session runs.

    Attributes:
        spec: Repo-relative spec path.
        parts: Number of parts the spec is split into.
        part_ids: The parts this session runs.
        out_dir: Output directory in the session.
        platform: Value of `ORAN_CAUSAL_PLATFORM`.
        wall_s: Per-process timeout, so the session ends (and saves) before the platform limit.
        code: `commit` and `dirty` of the launcher's git check, passed to every process.
        cost_table: Repo-relative cost table, or None.
        prefix: Shard file prefix (`res` for DEV, `eval_res` for EVAL).
        registry: DEV: the processes share `<out_dir>/registry.jsonl`.
        gpu: Process i sees GPU i mod the GPU count; CUDA is not initialised before the fork.
        skip_from: Globs of earlier shard files: datasets complete there are skipped.
        one_per_cpu: EVAL: at most `cpu_quota()` processes at once.
        module: Module whose `run` command executes one part.
    """

    spec: str
    parts: int
    part_ids: Sequence[int]
    out_dir: str
    platform: str
    wall_s: int
    code: Mapping[str, Any]
    cost_table: str | None = None
    prefix: str = "res"
    registry: bool = False
    gpu: bool = False
    skip_from: Sequence[str] = ()
    one_per_cpu: bool = False
    module: str = CAMPAIGN_MODULE


def part_env(plan: PartsPlan) -> str:
    """Environment assignments every process of the plan starts with."""
    code = plan.code
    return (
        f"{PLATFORM_ENV}={plan.platform} {COMMIT_ENV}={code['commit']} "
        f"{DIRTY_ENV}={int(bool(code['dirty']))} {LOCK_ENV}={LOCK_REL} {THREADS}"
    )


def _part_args(plan: PartsPlan) -> str:
    out = plan.out_dir
    extra = f" --cost-table {plan.cost_table}" if plan.cost_table else ""
    if plan.registry:
        extra += f" --registry {out}/registry.jsonl"
    if plan.skip_from:
        extra += " --skip-complete-from " + " ".join(shlex.quote(x) for x in plan.skip_from)
    return extra


def _part_command(plan: PartsPlan, i: int) -> str:
    out = plan.out_dir
    gpu = (
        f"PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=$(( {i} % NG )) " if plan.gpu else ""
    )
    gate = GATE if plan.one_per_cpu else ""
    return (
        f"{gate}( {gpu}{part_env(plan)} timeout {plan.wall_s} python -u -m {plan.module} run "
        f"--spec {plan.spec} --part {i}/{plan.parts} --out {out}/{plan.prefix}_{i}.jsonl"
        f"{_part_args(plan)} > {out}/log_{i}.txt 2>&1; echo $? > {out}/rc_{i}.txt ) &"
    )


def _exit_check(plan: PartsPlan) -> str:
    ids = " ".join(map(str, plan.part_ids))
    out = plan.out_dir
    return (
        f"RC=0; for i in {ids}; do r=$(cat {out}/rc_$i.txt 2>/dev/null || echo none); "
        f'[ "$r" = 0 ] || {{ echo "[xm-c] part $i exit $r"; RC=1; }}; done; [ $RC -eq 0 ]'
    )


def cloud_cmd(plan: PartsPlan) -> str:
    """Runs the plan's parts as background processes and waits for them.

    Each part's exit status goes to `rc_<i>.txt`, robust to the gate's `wait -n` reaping; the
    command's own status (the session's exit code) is non-zero if any part failed or left none.
    A part cut by the timeout resumes on relaunch, since records are appended per unit.
    """
    pre = GPU_PREAMBLE.format(out=plan.out_dir) if plan.gpu else ""
    if plan.one_per_cpu:
        pre += QUOTA_CHECK + f'echo "vCPU quota $NP" > {plan.out_dir}/cpu_quota.txt; '
    runs = " ".join(_part_command(plan, i) for i in plan.part_ids)
    out = plan.out_dir
    return f"mkdir -p {out} && {pre}{runs} wait; tail -n 3 {out}/log_*.txt; {_exit_check(plan)}"


@dataclasses.dataclass(frozen=True)
class SetupPlan:
    """The environment a session needs before the parts start.

    Attributes:
        out_dir: Output directory in the session (holds `install.log`, `pin_check.json`).
        gpu: Install the CUDA 12.8 torch build instead of the CPU one.
        torch: Install torch at all.
        selftest: Run the campaign tests after the install.
    """

    out_dir: str
    gpu: bool = False
    torch: bool = True
    selftest: bool = False


def sync_command(plan: SetupPlan) -> str:
    """`uv sync` of the bundle's lock into a venv outside the bundle.

    The torch extra belongs to the methods package, so it is selected as a second package.
    """
    pkgs = f"--package {EXPERIMENTS_PACKAGE}"
    if plan.torch:
        extra = "cu128" if plan.gpu else "cpu"
        pkgs += f" --package {METHODS_PACKAGE} --extra {extra}"
    group = "--group dev" if plan.selftest else "--no-dev"
    return f"uv sync --frozen {group} {pkgs}"


def setup_cmd(plan: SetupPlan) -> str:
    """Installs the bundle's `uv.lock` for its `.python-version`, then checks the pins.

    The sync is `--frozen`, so the lock decides every version; a fresh uv venv outside the bundle
    keeps the image's interpreter and packages untouched. The pin check lands in `pin_check.json`.
    """
    out = plan.out_dir
    log = f"{out}/install.log"
    install = (
        f"(command -v uv || pip install uv) > {log} 2>&1; unset UV_SYSTEM_PYTHON; "
        f"export UV_PROJECT_ENVIRONMENT={VENV_DIR}; {sync_command(plan)} >> {log} 2>&1; "
        f"source {VENV_DIR}/bin/activate; python --version >> {log} 2>&1; "
    )
    tests = " ".join(SELFTEST_TESTS)
    test = (
        f"python -m pytest -q -p no:cacheprovider {tests} > {out}/selftest.log 2>&1; "
        if plan.selftest
        else ""
    )
    return (
        f"mkdir -p {out}; {install}{PIN_CHECK} > {out}/pin_check.json 2>&1; nproc >> {log}; {test}"
    )


def launch_checks(
    spec: Mapping[str, Any], code: Mapping[str, Any], texts: ProtocolTexts
) -> list[str]:
    """Refusals before any cloud launch.

    EVAL needs an authorised protocol and a clean tree.

    Returns:
        One message per refusal; empty when the launch may go ahead.
    """
    errs: list[str] = []
    if is_eval(dict(spec)):
        ok, why = eval_authorised(dict(spec), texts)
        if not ok:
            errs.append(f"EVAL refused: {why}")
        if not code.get("commit") or code.get("dirty") is not False:
            errs.append(f"EVAL refused: tree not clean at launch ({code.get('dirty_paths')})")
    return errs

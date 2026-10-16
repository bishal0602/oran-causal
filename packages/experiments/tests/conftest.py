import copy
import dataclasses
import hashlib
import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from oran_causal.experiments.campaign.run import Providers, RunOptions
from oran_causal.experiments.campaign.spec import ProtocolTexts, shipped_protocol
from oran_causal.experiments.launch.runner import Clock

DATA = Path(__file__).parent / "data"
DUMMY = "cdd_oran.xmethod.runner:DummyMethod"
COMMIT = {"commit": "0" * 40, "dirty": False, "source": "test", "dirty_paths": None}
PROTOCOL_TEXT = "# Protocol\n\nFROZEN: yes (2026-10-04)\nbody\n"


def quiet_providers(**changes: Any) -> Providers:
    base = Providers(
        code=lambda: dict(COMMIT),
        pins=lambda python: {"lock": None, "n_pins": 0, "mismatches": None, "python": python},
        python_pin=lambda: None,
        host=lambda: {"platform": "test", "cpu_model": None, "cpu_count": 1, "node": "t"},
        load=dict,
        pkgs=lambda: {"numpy": None},
        procs=lambda: 1,
        quota=lambda: 1,
        log=lambda _msg: None,
    )
    return dataclasses.replace(base, **changes)


def make_spec(**changes: Any) -> dict[str, Any]:
    spec = {
        "name": "t",
        "budget_cpu_s": None,
        "arms": {
            "d_by": {"ref": DUMMY, "declare": "by"},
            "d_tau": {"ref": DUMMY, "declare": "tau"},
            "d_e3": {"ref": DUMMY, "declare": "by", "worlds": ["E3"], "max_n": 300},
        },
        "blocks": [
            {
                "role": "tune",
                "worlds": ["E1", "E4"],
                "regimes": ["R1", "R3"],
                "ns": [300, 400],
                "kappas": [0.25],
                "seeds": [3000000, 3000002],
            },
            {
                "role": "measure",
                "worlds": ["E1", "E4"],
                "regimes": ["R1", "R3"],
                "ns": [300, 400],
                "kappas": [0.25],
                "seeds": [3000100, 3000103],
            },
        ],
    }
    spec.update(copy.deepcopy(changes))
    return spec


def small_spec() -> dict[str, Any]:
    return make_spec(
        blocks=[
            {
                "role": "tune",
                "worlds": ["E4"],
                "regimes": ["R1"],
                "ns": [300],
                "kappas": [0.25],
                "seeds": [3000000, 3000001],
            },
            {
                "role": "measure",
                "worlds": ["E4"],
                "regimes": ["R1"],
                "ns": [300, 400],
                "kappas": [0.25],
                "seeds": [3000100, 3000101],
            },
        ]
    )


def frozen_texts(text: str = PROTOCOL_TEXT, crlf: bool = False) -> tuple[ProtocolTexts, str]:
    data = (text.replace("\n", "\r\n") if crlf else text).encode()
    return ProtocolTexts(live=data), hashlib.sha256(text.encode()).hexdigest()


def eval_spec(sha: str | None) -> dict[str, Any]:
    spec = small_spec()
    spec["blocks"][1]["seeds"] = [3100000, 3100001]
    if sha is not None:
        spec["protocol_sha256"] = sha
    return spec


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@pytest.fixture
def opts() -> Callable[..., RunOptions]:
    def build(**changes: Any) -> RunOptions:
        providers = quiet_providers(**changes.pop("providers", {}))
        return dataclasses.replace(RunOptions(isolate=False, providers=providers), **changes)

    return build


@pytest.fixture
def shipped() -> ProtocolTexts:
    return shipped_protocol()


@pytest.fixture
def kit() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        DATA=DATA,
        DUMMY=DUMMY,
        COMMIT=COMMIT,
        make_spec=make_spec,
        small_spec=small_spec,
        eval_spec=eval_spec,
        frozen_texts=frozen_texts,
        read_jsonl=read_jsonl,
        quiet_providers=quiet_providers,
    )


HEAD = "f" * 40


def fake_git(status: str = ""):
    answers = {"rev-parse": HEAD + "\n", "status": status}
    return lambda args: answers[args[0]]


def mini_workspace(root: Path) -> Path:
    files = {
        "pyproject.toml": "[tool.uv]\r\n",
        "uv.lock": "version = 1\n",
        ".python-version": "3.14\n",
        "packages/a/pyproject.toml": "[project]\nname = 'a'\n",
        "packages/a/src/oran_causal/a/__init__.py": "A = 1\r\n",
        "packages/a/src/oran_causal/a/__pycache__/m.cpython-314.pyc": "x",
        "packages/a/tests/test_a.py": "def test(): ...\n",
        "packages/experiments/protocols/study4/PROTOCOL_A.md": "FROZEN: yes\n",
        "packages/experiments/manifests/seeds.json": "{}\n",
        "packages/experiments/artifacts/table.json": "[1]\n",
        "packages/experiments/specs/dev/s.json": "{}\n",
        "records/runs/old/done/a.jsonl": "{}\n",
        "scratch/notes.md": "x\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    return root


@dataclasses.dataclass
class FakeClock:
    t: float = 1_000_000.0
    slept: list[float] = dataclasses.field(default_factory=list)

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s

    def clock(self) -> Clock:
        return Clock(now=lambda: self.t, sleep=self.sleep)


@pytest.fixture
def launch_kit() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        HEAD=HEAD, fake_git=fake_git, mini_workspace=mini_workspace, FakeClock=FakeClock
    )

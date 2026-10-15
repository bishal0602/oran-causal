import json
import subprocess
import sys

import pytest

ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "sim": frozenset({"core"}),
    "methods": frozenset({"core", "sim"}),
    "analysis": frozenset({"core", "sim"}),
    "figures": frozenset({"analysis", "sim"}),
    "trace": frozenset(),
    "experiments": frozenset({"core", "sim", "methods", "analysis", "figures"}),
}
TORCH_FREE = frozenset({"core", "sim", "analysis", "figures", "trace"})

PROBE = (
    "import importlib, json, sys;"
    "importlib.import_module('oran_causal.' + sys.argv[1]);"
    "print(json.dumps(sorted(sys.modules)))"
)


def loaded_modules(member: str) -> set[str]:
    output = subprocess.run(
        [sys.executable, "-c", PROBE, member],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(json.loads(output))


def loaded_members(modules: set[str]) -> set[str]:
    return {name.split(".")[1] for name in modules if name.startswith("oran_causal.")}


@pytest.mark.parametrize("member", sorted(ALLOWED))
def test_member_imports_only_allowed_members(member: str) -> None:
    extra = loaded_members(loaded_modules(member)) - {member} - ALLOWED[member]
    assert not extra, sorted(extra)


@pytest.mark.parametrize("member", sorted(TORCH_FREE))
def test_member_does_not_load_torch(member: str) -> None:
    assert "torch" not in loaded_modules(member)

import importlib
import json
import pkgutil
import subprocess
import sys

import oran_causal.core

PROBE = (
    "import importlib, json, pkgutil, sys;"
    "import oran_causal.core as c;"
    "names = [m.name for m in pkgutil.walk_packages(c.__path__, 'oran_causal.core.')];"
    "[importlib.import_module(n) for n in names];"
    "print(json.dumps(sorted(sys.modules)))"
)


def test_member_imports() -> None:
    module = importlib.import_module("oran_causal.core")
    assert module.__name__ == "oran_causal.core"


def test_every_module_imports() -> None:
    names = [m.name for m in pkgutil.walk_packages(oran_causal.core.__path__, "oran_causal.core.")]
    assert "oran_causal.core.pmrt.engine" in names
    for name in names:
        importlib.import_module(name)


def test_core_loads_no_other_member_and_no_torch() -> None:
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    ).stdout
    modules = set(json.loads(out))
    members = {m.split(".")[1] for m in modules if m.startswith("oran_causal.")}
    assert members == {"core"}
    assert "torch" not in modules

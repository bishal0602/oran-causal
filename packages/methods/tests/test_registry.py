import json
import subprocess
import sys

import numpy as np
import pytest

from oran_causal.core.pmrt.nonlinear import PmrtNl
from oran_causal.core.pmrt.rows import PmrtCore
from oran_causal.methods.registry import ARMS, PATHS, resolve

HEAVY = ("torch", "shap", "causallearn", "statsmodels", "xgboost")


def _fresh(code: str) -> list[str]:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.claim("DEF-ENV-ARMS")
def test_pmrt_refs_bind_to_core():
    assert resolve("pmrt_core") is PmrtCore
    assert resolve("pmrt_nl") is PmrtNl


def test_registry_names():
    assert set(ARMS) == set(PATHS)
    assert len(ARMS) == len(PATHS)
    for name in PATHS:
        cls = ARMS[name]
        assert cls().name == name
    with pytest.raises(KeyError, match="unknown arm"):
        resolve("nope")


def test_registry_import_is_lazy():
    code = (
        "import json, sys; import oran_causal.methods.registry as r; a = sorted(sys.modules);"
        "r.ARMS['corr']; r.ARMS['pcorr']; r.ARMS['dummy']; b = sorted(sys.modules);"
        "r.resolve('pc'); c = sorted(sys.modules);"
        "print(json.dumps([a, b, c]))"
    )
    first, light, after_pc = _fresh(code)
    for mods in (first, light):
        assert not {m.split(".")[0] for m in mods} & set(HEAVY)
    assert "causallearn" in {m.split(".")[0] for m in after_pc}


def test_methods_import_only_core_and_sim():
    code = (
        "import json, sys, importlib, pkgutil; import oran_causal.methods.controlled as c;"
        "p = 'oran_causal.methods.controlled.';"
        "names = [m.name for m in pkgutil.walk_packages(c.__path__, p)];"
        "[importlib.import_module(n) for n in names]; import oran_causal.methods.registry;"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('oran_causal.'))))"
    )
    members = {m.split(".")[1] for m in _fresh(code)}
    assert members <= {"core", "sim", "methods"}


def test_dummy_arm_reproduces_runner_stream(make_ci):
    ds, _ = make_ci("planted", 50, 3_000_000)
    ds = type(ds)(**{**ds.__dict__, "world": "E2", "regime": "R2", "meta": {"lam": 1.5}})
    r = resolve("dummy")().run(ds, {"q": 0.05})
    rng = np.random.default_rng(np.random.SeedSequence([7899, 3_000_000, 2, 2, 1500, 50]))
    p = rng.uniform(size=len(ds.candidates))
    sign = rng.choice([-1, 1], size=len(ds.candidates))
    assert [e.p for e in r.edges] == p.tolist()
    assert [e.sign for e in r.edges] == sign.tolist()
    assert [e.score for e in r.edges] == (1 - p).tolist()
    assert resolve("dummy")().tune([]) == {"q": 0.05}

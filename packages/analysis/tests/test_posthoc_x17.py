"""X17 analysis half on small synthetic inputs: strict map, reading rule and the confirmation."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oran_causal.analysis.posthoc import x17
from oran_causal.analysis.system.study2 import SCHEMA_EVAL
from oran_causal.core.gate import FROZEN_WINDOW_S, WINDOW_S


def test_centres_are_the_declared_windows():
    assert x17.CENTRES == {"frozen": FROZEN_WINDOW_S, "strict": WINDOW_S}
    assert WINDOW_S == 90.0
    assert len(x17.CONFIRM_SEEDS) == 8


def test_strict_map_keeps_the_declared_sign():
    m = {("sleep", "own", "pv"): -2.0, ("ptx", "nbr", "rlf"): 0.5}
    slopes = {("sleep", "own", "pv"): {"beta": 1.5}, ("ptx", "nbr", "rlf"): {"beta": -0.25}}
    assert x17.strict_map(m, slopes) == {("ptx", "nbr", "rlf"): 0.25, ("sleep", "own", "pv"): -1.5}
    with pytest.raises(ValueError, match="declared"):
        x17.strict_map({("sleep", "own", "pv"): None}, slopes)


@pytest.mark.parametrize(
    ("mg", "cs", "want"),
    [(True, True, "signatures equal"), (False, True, "re-simulate"), (True, False, "stop")],
)
def test_reading_rule(mg: bool, cs: bool, want: str):
    assert want in x17.reading({"MG:PMRT_equal": mg, "CS:PMRT_equal": cs})


def test_block_of_splits_the_eval_blocks():
    assert x17.block_of(187159) == "study2"
    assert x17.block_of(191100) == "study3"


def _job(seed: int, **kw) -> dict:
    return {
        "kind": "job",
        "schema": SCHEMA_EVAL,
        "sub": x17.SUB,
        x17.STAGE_KEY: "confirm",
        "seed": seed,
        "arm": x17.ARM,
        "prot_viol": 10.0,
        "rlf": 3,
        "policy_counts": {"defer": 2},
        "cpu_s": 5.0,
        **kw,
    }


def _write(path: Path, recs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


def _stored(seeds, **kw) -> dict:
    out: dict = {"study2": {}, "study3": {}}
    for s in seeds:
        out[x17.block_of(s)][s] = {
            x17.ARM: {
                "seed": s,
                "prot_viol": 10.0,
                "rlf": 3,
                "policy_counts": {"defer": 2},
                "cpu_s": 9.0,
                **kw,
            }
        }
    return out


def test_confirm_ignores_timing_and_flags_a_difference(tmp_path):
    seeds = x17.CONFIRM_SEEDS
    runs = _write(tmp_path / "r.jsonl", [_job(s) for s in seeds])
    sig = {"MG:PMRT_equal": True}
    res = x17.confirm([runs], _stored(seeds), sig)
    assert res["identical"]
    assert res["ok"]
    bad = _stored(seeds)
    bad["study3"][191100][x17.ARM]["rlf"] = 4
    res = x17.confirm([runs], bad, sig)
    assert not res["identical"]
    assert res["per_seed"][191100]["max_abs_diff"] == 1.0
    res = x17.confirm([runs], _stored(seeds[:-1]), sig)
    assert not res["complete"]
    assert res["per_seed"][seeds[-1]] == {"missing": "stored"}
    assert not x17.confirm([runs], _stored(seeds), {"MG:PMRT_equal": False})["ok"]


def test_slope_uses_the_window(monkeypatch):
    calls: list = []

    def fake(*args, window_s):
        calls.append((len(args), window_s))
        return {"beta": 0.0}

    monkeypatch.setattr(x17, "design_slope_se", fake)

    z = np.zeros(3)
    ud: Any = SimpleNamespace(
        mode=z,
        probs=z,
        sgn=z,
        episode=z,
        t0=z,
        y={("own", "pv"): z},
        pre={("own", "pv"): z},
        rows_of=lambda f: np.arange(3),
    )
    monkeypatch.setattr(x17, "unit_order", lambda ud, rows: rows)
    monkeypatch.setattr(x17, "v_design", lambda *a: z)
    x17.slope(ud, ("sleep", "own", "pv"), WINDOW_S, np.ones(2))
    x17.slope(ud, ("sleep", "own", "pv"), FROZEN_WINDOW_S, np.ones(2))
    assert calls == [(6, WINDOW_S), (6, FROZEN_WINDOW_S)]

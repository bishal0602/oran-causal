import dataclasses
import random
from collections.abc import Callable
from typing import Any

import pytest

from oran_causal.analysis.controlled.integrity import IntegrityInputs, by_declare
from oran_causal.analysis.controlled.protocol import PRIMARY_KAPPA
from oran_causal.analysis.controlled.report import analyse
from oran_causal.analysis.controlled.spec import expected_units
from oran_causal.analysis.records import job_key
from oran_causal.core.data import EdgeResult, Result
from oran_causal.core.scoring import PLACEBO, PLACEBO_CONF
from oran_causal.sim.controlled.datasets.truth import truth_for

TUNE = list(range(3_000_000, 3_000_020))
MEAS = list(range(3_000_100, 3_000_112))
PSHA, SSHA, COMMIT = "p" * 64, "s" * 64, "abc"
PKGS = {"numpy": "2.0.2"}
CI = "cdd_oran.xmethod.methods.ci:CI"
LIBERAL = {("corr", "R2"), ("ci_native", "R2")}

SPEC: dict[str, Any] = {
    "name": "test_synth",
    "protocol_sha256": PSHA,
    "budget_cpu_s": 7200,
    "arms": {
        "pmrt_eq": {
            "ref": "cdd_oran.xmethod.methods.pmrt_core:PmrtCore",
            "config": {"covariates": "eq"},
            "declare": "by",
        },
        "corr": {
            "ref": "cdd_oran.xmethod.methods.corr:Corr",
            "config": {"arm": "native"},
            "declare": "by",
            "analysis": "primary",
        },
        "ci_eq": {
            "ref": CI,
            "config": {"arm": "eq"},
            "declare": "by",
            "native_partner": "ci_native",
        },
        "ci_native": {"ref": CI, "config": {"arm": "native"}, "declare": "by"},
        "pcorr_hac": {
            "ref": "cdd_oran.xmethod.methods.hac:HAC",
            "config": {"arm": "native"},
            "declare": "by",
        },
        "pc_eq": {
            "ref": "cdd_oran.xmethod.methods.pc:PC",
            "config": {"arm": "eq"},
            "declare": "tau",
        },
        "pc_native": {
            "ref": "cdd_oran.xmethod.methods.pc:PC",
            "config": {"arm": "native"},
            "declare": "tau",
        },
    },
    "blocks": [
        {
            "role": "tune",
            "worlds": ["E1", "E4"],
            "regimes": ["R1", "R2", "R3"],
            "ns": [500, 1000],
            "kappas": [0.25],
            "seeds": TUNE,
            "e4_lams": {"R1": [], "R2": [], "R3": [1.0]},
        },
        {
            "role": "measure",
            "worlds": ["E1", "E4"],
            "regimes": ["R1", "R2", "R3"],
            "ns": [500, 1000],
            "kappas": [0.25],
            "seeds": MEAS,
            "e4_lams": {"R1": [], "R2": [], "R3": [1.0]},
        },
    ],
}


def make_record(  # noqa: PLR0917
    arm: str,
    world: str,
    regime: str,
    lam: float | None,
    n: int,
    seed: int,
    role: str,
    *,
    liberal: bool,
    commit: str = COMMIT,
) -> dict[str, Any]:
    truth = truth_for(world, regime)
    rng = random.Random(f"{arm}|{world}|{regime}|{lam}|{n}|{seed}")
    edges: list[dict[str, Any]] = []
    for source, target in sorted(truth.edges | truth.null_edges):
        if (source, target) in truth.edges:
            p = 1e-5
        elif liberal and rng.random() < 0.4:
            p = rng.uniform(0, 0.01)
        else:
            p = rng.uniform(0.06, 1.0)
        sign = truth.signs.get((source, target), 1)
        edges.append(
            {
                "source": source,
                "target": target,
                "score": 1 - p,
                "p": p,
                "sign": sign,
                "declared": False,
            }
        )
    family = [e for e in edges if not e["source"].startswith("K") and e["source"] != PLACEBO_CONF]
    for edge, flag in zip(family, by_declare([e["p"] for e in family]), strict=True):
        edge["declared"] = flag
    return {
        "key": job_key(arm, world, regime, lam, n, seed, PRIMARY_KAPPA),
        "method": arm,
        "arm": arm,
        "role": role,
        "status": "ok",
        "error": None,
        "job": {"world": world, "regime": regime, "lam": lam, "n": n, "seed": seed, "kappa": 0.25},
        "edges": edges,
        "notes": {},
        "config": {},
        "cpu_s": 0.1,
        "peak_rss_mb": 10.0,
        "dataset_sha256": f"{world}|{regime}|{lam}|{n}|{seed}",
        "code": {"commit": commit, "dirty": False},
        "run_mode": {"mode": "eval", "protocol_sha256": PSHA},
        "integrity": {
            "code_commit": commit,
            "code_dirty": False,
            "protocol_sha256": PSHA,
            "spec_sha256": "c" * 64,
            "spec_file_sha256": SSHA,
        },
        "pkgs": PKGS,
        "host": {"platform": "kaggle"},
    }


def synth_records(spec: dict[str, Any] = SPEC) -> list[dict[str, Any]]:
    return [
        make_record(
            u["arm"],
            u["world"],
            u["regime"],
            u["lam"],
            u["n"],
            u["seed"],
            u["role"],
            liberal=(u["arm"], u["regime"]) in LIBERAL,
        )
        for u in expected_units(spec).values()
    ]


def run(
    records: list[dict[str, Any]], spec: dict[str, Any] = SPEC, **overrides: Any
) -> dict[str, Any]:
    inputs = IntegrityInputs(
        freeze_commit=COMMIT, spec_sha=SSHA, protocol={"ok": True}, **overrides
    )
    return analyse(sorted(records, key=lambda r: r["key"]), spec, inputs)


def placebo_result(scores: list[float]) -> Result:
    edges = tuple(
        EdgeResult(PLACEBO, f"K{i}", score, None, 0, declared=False)
        for i, score in enumerate(scores)
    )
    return Result("m", "v", edges, 0.0, {})


@dataclasses.dataclass(frozen=True)
class Synth:
    """The synthetic study: spec, record factory and analysis runner."""

    spec: dict[str, Any]
    record: Callable[..., dict[str, Any]]
    records: Callable[..., list[dict[str, Any]]]
    run: Callable[..., dict[str, Any]]
    commit: str = COMMIT
    spec_sha: str = SSHA
    meas: tuple[int, ...] = tuple(MEAS)
    tune: tuple[int, ...] = tuple(TUNE)


@pytest.fixture(scope="session")
def synth() -> Synth:
    return Synth(SPEC, make_record, synth_records, run)


@pytest.fixture(scope="module")
def study(synth: Synth) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = synth.records()
    return records, synth.run(records)

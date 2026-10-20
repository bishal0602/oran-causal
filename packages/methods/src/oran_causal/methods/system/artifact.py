"""The frozen PMRT artifact of the Study 1 fresh-seed re-test (`study1_pmrt.json`).

One pure-data JSON file holds every learned piece the integrated PMRT statistic and its
declaration layers use, all fitted before any Study 1 record existed: the statistic settings,
the receiver tables, the training context keys and, per hypothesis, the ridge adjustment, the
`plain` and `loadsp` kernels, slot count, training plain z, kernel signs, running-centre
intercepts and the log-variance scale model; and the layer's prior (signed effect z on the 480
training episodes per statistic) with its level, floor, cap, threshold, directions and weights.
Its sha256 (LF bytes) is pinned by the Study 1 protocol and recorded in the `.sha256` sidecar.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from oran_causal.core.declare import Prior, WeightConfig, prior_directions, prior_weights
from oran_causal.core.multiplicity import W_CAP
from oran_causal.core.pmrt.units import PMRT_VERSION, PmrtConfig
from oran_causal.methods.system.pmrt import COL_ARMS, FLOOR, MAX_OF, N_PRIOR, STATS, THR, Q
from oran_causal.sim.system.decision.vocab import HYPOTHESES

ARTIFACT_NAME = "study1_pmrt.json"
ARTIFACT_SHA256 = "6f8e543145ad0e76afca653b17c4ea13b010c70d6b3134b109cd45fa0bcc99eb"
ARTIFACT_SCHEMA = "e6p-pmrt/1"
MSCRPLUS_SCHEMA = "e6p-mscrplus-frozen/1"
SCHEMAS = (ARTIFACT_SCHEMA, MSCRPLUS_SCHEMA)
MSCRPLUS_NAME = "study1_mscr_plus.json"
N_PLANNED = (60, 120, 600)
LEARNED_KEYS = (
    "config",
    "n_train_units",
    "n_train_eps",
    "ctx_keys",
    "rho",
    "hyp",
    "statistic_arms",
    "max_of",
)
LAYER_KEYS = ("q", "floor", "thr", "cap", "n_prior", "z", "directions", "weights")


def sha_file(path: str | os.PathLike[str]) -> str:
    """Sha256 of a file's bytes with CRLF read as LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sidecar_sha(path: str | os.PathLike[str]) -> str:
    """The hash recorded in the `<path>.sha256` sidecar."""
    return Path(f"{path}.sha256").read_text(encoding="utf-8").split()[0]


def _read(path: str | os.PathLike[str]) -> dict[str, Any]:
    a = json.loads(Path(path).read_text(encoding="utf-8"))
    if a.get("schema") not in SCHEMAS:
        raise ValueError(f"{path}: schema {a.get('schema')}")
    return a


def _core(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "fill": np.array(c["fill"], float),
        "mx": np.array(c["mx"], float),
        "sx": np.array(c["sx"], float),
        "keep": np.array(c["keep"], bool),
        "B": np.array(c["B"], float),
        "lam": c["lam"],
        "S": int(c["S"]),
        "z_plain": float(c["z_plain"]),
        "sdir": dict(c["sdir"]),
        "K": {v: (np.array(k[0], float), np.array(k[1], float)) for v, k in c["K"].items()},
        "prior": {v: {1: p["1"], -1: p["-1"]} for v, p in c["prior"].items()},
    }


def _hmodel(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "w": np.array(m["w"], float),
        "xm": np.array(m["xm"], float),
        "ym": float(m["ym"]),
        "med": float(m["med"]),
        "kappa": float(m["kappa"]),
    }


def _priors(layer: dict[str, Any]) -> dict[str, Prior]:
    out = {}
    for s in STATS:
        d = {
            h: ({"z": z, "status": "tested"} if z is not None else {"status": "undetermined"})
            for h, z in layer["z"][s].items()
        }
        out[s] = Prior.from_stats(d, n_prior=layer["n_prior"], source="frozen artifact")
    return out


def load_artifact(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Prior]]:
    """Loads a frozen PMRT artifact (PMRT or MSCR+ schema).

    Returns:
        (params usable by `lean_columns` and `run_integrated`, the layer prior per statistic
        of `STATS`; empty when the artifact has no layer).

    Raises:
        ValueError: Unknown schema.
    """
    a = _read(path)
    hyp = {
        h: {
            "core": _core(hp["core"]),
            "h": {v: _hmodel(m) for v, m in hp["h"].items()},
            "gb": None,
        }
        for h, hp in a["hyp"].items()
    }
    params = {
        "version": a["version"],
        "config": a["config"],
        "ctx_keys": a["ctx_keys"],
        "rho": dict(a["rho"]),
        "hyp": hyp,
    }
    return params, (_priors(a["layer"]) if "layer" in a else {})


def pmrt_config(params: dict[str, Any], B: int) -> PmrtConfig:
    """The statistic settings of `params` with `B` redraws."""
    fields = PmrtConfig.__dataclass_fields__
    c = {k: v for k, v in params["config"].items() if k in fields}
    c["B"] = int(B)
    c["best_exclude"] = tuple(c.get("best_exclude", ()))
    return PmrtConfig(**c)


def learned_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Learned fields (`LEARNED_KEYS`, layer `LAYER_KEYS`) that differ between two artifacts."""
    bad = [k for k in LEARNED_KEYS if a.get(k) != b.get(k)]
    la, lb = a.get("layer"), b.get("layer")
    if la is None or lb is None:
        return bad + (["layer"] if (la is None) != (lb is None) else [])
    return bad + [f"layer.{k}" for k in LAYER_KEYS if la.get(k) != lb.get(k)]


def _layer_problems(layer: dict[str, Any], priors: dict[str, Prior]) -> list[str]:
    bad = [
        k
        for k, v in (("q", Q), ("floor", FLOOR), ("thr", THR), ("cap", W_CAP), ("n_prior", N_PRIOR))
        if layer[k] != v
    ]
    for s in STATS:
        dirs = {
            "|".join(h): d for h, d in prior_directions(priors[s], HYPOTHESES, THR).items() if d
        }
        if dirs != layer["directions"][s]:
            bad.append(f"directions.{s}")
        for n in N_PLANNED:
            w = prior_weights(priors[s], HYPOTHESES, WeightConfig(n, Q, FLOOR))
            stored = layer["weights"][s][str(n)]
            if any(abs(w[tuple(h.split("|"))] - x) > 1e-12 for h, x in stored.items()):
                bad.append(f"weights.{s}.{n}")
    return bad


def verify(
    path: str | os.PathLike[str], superseded: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Checks a frozen artifact against its pinned hash and the code's settings.

    Checks: sha256 = sidecar = `ARTIFACT_SHA256`; schema and version; the stored config builds
    the default `PmrtConfig`; statistic arms; the layer's level, floor, threshold, cap and prior
    size, and its directions and planned-size weights recomputed from its prior; with `superseded`,
    the superseded artifact's hash and identical learned content.

    Returns:
        `{"sha256", "hyp", "problems"}`; `problems` is empty when every check passes.
    """
    a = _read(path)
    sha = sha_file(path)
    bad = [] if sha == sidecar_sha(path) == ARTIFACT_SHA256 else ["sha256"]
    if a["schema"] != ARTIFACT_SCHEMA or a["version"] != PMRT_VERSION:
        bad.append("schema")
    params, priors = load_artifact(path)
    if pmrt_config(params, PmrtConfig().B) != PmrtConfig():
        bad.append("config")
    if tuple(a["statistic_arms"]) != COL_ARMS or tuple(a["max_of"]) != MAX_OF:
        bad.append("arms")
    bad += [f"layer.{k}" for k in _layer_problems(a["layer"], priors)]
    if superseded is not None:
        if a.get("supersedes", {}).get("sha256") != sha_file(superseded):
            bad.append("supersedes")
        bad += [f"superseded.{k}" for k in learned_diff(a, _read(superseded))]
    return {"sha256": sha, "hyp": len(a["hyp"]), "problems": bad}

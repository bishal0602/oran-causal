"""Numeric fingerprint of a machine: which numerical layers are bit-identical to another one."""

import hashlib
import importlib
import json
import platform
import sys
from typing import Any

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from oran_causal.sim.system.config import E6Config
from oran_causal.sim.system.env import E6Env

SIMD_FLAGS = ("AVX2", "FMA3", "AVX512F", "AVX512_SKX", "AVX512_SPR")
LAYERS = ("rng", "ufunc", "scipy_fp", "traj")
TRAJ_SEED = 100020


def numeric_env() -> dict[str, Any]:
    """Versions of numpy, scipy and Python, the OS and the SIMD extensions numpy dispatches to."""
    features = getattr(
        importlib.import_module("numpy._core._multiarray_umath"), "__cpu_features__", None
    )
    simd = sorted(k for k in SIMD_FLAGS if features.get(k)) if features else None
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "simd": simd,
    }


def _hash(a: Any) -> str:
    data = np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()
    return hashlib.sha256(data).hexdigest()[:12]


def _rng_layer() -> dict[str, str]:
    def gen() -> np.random.Generator:
        return np.random.default_rng([7, 6600, 3, 11])

    return {
        "uniform": _hash(gen().uniform(0, 1, 5000)),
        "integers": _hash(gen().integers(0, 37, 5000)),
        "normal": _hash(gen().normal(0, 1, 5000)),
        "poisson": _hash(gen().poisson(np.linspace(0, 400, 5000))),
        "choice": _hash(gen().choice(500, 40, replace=False)),
    }


def _stream() -> tuple[np.ndarray, np.ndarray]:
    raw = np.random.PCG64(12345).random_raw(100_000) >> np.uint64(11)
    x = raw.astype(np.float64) / 2.0**53
    return x, x * 200 - 100


def _ufunc_layer(x: np.ndarray, y: np.ndarray) -> dict[str, str]:
    layers = {
        "log10": np.log10(x + 1e-3),
        "log": np.log(x + 1e-3),
        "log2": np.log2(1 + 10 ** (y / 10)),
        "pow": 10 ** (y / 10),
        "exp": np.exp(-y / 7),
        "sin": np.sin(y),
        "cos": np.cos(y),
        "arctan2": np.arctan2(y, x - 0.5),
        "sum": [np.sum(y), np.mean(y)],
    }
    return {k: _hash(v) for k, v in layers.items()}


def _scipy_layer(y: np.ndarray) -> dict[str, str]:
    z = np.random.default_rng(1).normal(size=(128, 128))
    return {
        "gaussian_filter": _hash(gaussian_filter(z, 4.0, mode="wrap")),
        "norm_cdf": _hash(norm.cdf(y / 30)),
    }


def _trajectory(seconds: int) -> list[str]:
    cfg = E6Config(
        seed=TRAJ_SEED,
        load="medium",
        mobility="mixed",
        mix="M4",
        warmup_s=120.0,
        scored_s=600.0,
        scenario="base",
    )
    env = E6Env(cfg, log=False, wg3=True)
    traj = []
    while env.sec < seconds:
        obs = env.step_propose()
        env.step_apply(
            {"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []}
        )
        sla = sorted(env.plant.sla.items())
        traj.append(_hash([float(v) for _, v in sla if np.isscalar(v)]))
    return traj


def fingerprint(seconds: int = 300) -> dict[str, Any]:
    """Hashes of the numeric primitives the plant uses and of a short accept-all episode.

    Layers: `rng` (Generator methods), `ufunc` (transcendentals on a bit-stable PCG64 stream),
    `scipy_fp` (gaussian filter, normal cdf) and `traj` (a per-second hash of `plant.sla`). Equal
    hashes in a layer mean bit-identical numerics for it.

    Args:
        seconds: Length of the accept-all episode in plant seconds.
    """
    x, y = _stream()
    return {
        "env": numeric_env(),
        "rng": _rng_layer(),
        "ufunc": _ufunc_layer(x, y),
        "scipy_fp": _scipy_layer(y),
        "traj": _trajectory(seconds),
    }


def compare(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, bool]:
    """Per-layer equality of two fingerprints; empty when either lacks the data."""
    if not local or "env" not in remote:
        return {}
    return {k: remote.get(k) == local.get(k) for k in LAYERS}


def main(argv: list[str]) -> None:
    """Prints `fingerprint(*argv)` as one JSON line."""
    print(json.dumps(fingerprint(*[int(a) for a in argv])))


if __name__ == "__main__":
    main(sys.argv[1:])

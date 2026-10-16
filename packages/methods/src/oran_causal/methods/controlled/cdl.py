"""Causal Dynamics Learning (Wang et al. 2022), the authors' NaNA 2026 conference method, offline.

State nodes: the actions at t (min-max scaled), non-NaN lagged KPIs (z-scored) and context
(min-max); targets: the KPIs at t+1 (z-scored); the action input is a constant 0 and the
training mask a uniformly random node. Gradient steps = min(`max_gradient_steps`,
ceil(`epochs` n / `batch_size`)); each step is one Adam step on rows drawn with replacement,
every `eval_steps`-th step adds one CMI batch, and the EMA (tau .99, started at 0) moves
every `eval_steps` CMI batches. Score = final EMA CMI of (source node, target); `native` =
the conference threshold CMI >= .16. Sign: partial correlation given the native set. The
primary fit has no `P_placebo_conf` node; its candidates come from a second fit with every
action. torch is seeded from `default_rng([7804, seed, fit index])` with its global RNG
state restored and deterministic algorithms on during the fit.
"""

import math
import time
from typing import Any, ClassVar

import numpy as np
import torch

from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._cdl_model import CDL, CdlSettings
from oran_causal.methods.controlled._classic import (
    PLACEBO_CONF,
    ClassicBase,
    Columns,
    Scored,
    columns,
    resolve,
    sign_pcorr_given_z,
    target_index,
)

RNG_TAG_CDL = 7804
SOURCE = "refactor/codebase@65b56f3 cdd_oran/models/cdl.py"


def fit_seed(data: Dataset, fit_idx: int) -> int:
    """Torch seed of fit `fit_idx` on `data`."""
    return int(
        np.random.default_rng([RNG_TAG_CDL, int(data.seed), int(fit_idx)]).integers(0, 2**31 - 1)
    )


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(0), a.max(0)
    return (a - lo) / np.where(hi > lo, hi - lo, 1.0)


def _z(a: np.ndarray) -> np.ndarray:
    sd = a.std(0)
    return (a - a.mean(0)) / np.where(sd > 0, sd, 1.0)


def gradient_steps(n: int, config: dict[str, Any]) -> int:
    """min(max_gradient_steps, ceil(epochs * n / batch_size))."""
    cap = int(config["max_gradient_steps"])
    return int(min(cap, math.ceil(float(config["epochs"]) * n / int(config["batch_size"]))))


def _model(s: np.ndarray, y: np.ndarray, config: dict[str, Any], dev: torch.device) -> CDL:
    settings = CdlSettings(
        state_dim=s.shape[1],
        action_dim=1,
        target_dim=y.shape[1],
        feature_dims=tuple(config["feature_fc_dims"]),
        generative_dims=tuple(config["generative_fc_dims"]),
        lr=config["lr"],
        eval_tau=config["eval_tau"],
        eval_steps=config["eval_steps"],
        grad_clip=config["grad_clip"],
        informed_drop=False,
    )
    return CDL(settings, dev)


def _train(
    model: CDL, s: np.ndarray, y: np.ndarray, config: dict[str, Any], dev: torch.device
) -> float:
    s_all = torch.as_tensor(s, dtype=torch.float32, device=dev)
    y_all = torch.as_tensor(y, dtype=torch.float32, device=dev)
    n, bs = s.shape[0], int(config["batch_size"])
    a0 = torch.zeros(bs, 1, device=dev)
    every = int(config["eval_steps"])
    loss: Any = float("nan")
    for step in range(gradient_steps(n, config)):
        idx = torch.randint(n, (bs,)).to(dev)
        loss = model.fit_step(s_all[idx], y_all[idx], a0)
        if step % every == 0:
            idx = torch.randint(n, (bs,)).to(dev)
            model.cmi_step(s_all[idx], y_all[idx], a0)
    return float(loss)


def fit_cmi(s: np.ndarray, y: np.ndarray, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Trains CDL offline on state nodes `s` `[n, d]` and targets `y` `[n, k]`.

    Returns:
        {cmi `[k, d + 1]` (last column: the constant action node), final_loss, wall_s,
        device, torch}.
    """
    dev = torch.device(config.get("device", "cpu"))
    det = torch.are_deterministic_algorithms_enabled()
    t0 = time.perf_counter()
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.use_deterministic_algorithms(True, warn_only=dev.type == "cuda")
        try:
            torch.manual_seed(seed)
            model = _model(s, y, config, dev)
            loss = _train(model, s, y, config, dev)
            cmi = model.mask_CMI.detach().cpu().numpy().astype(float)
        finally:
            torch.use_deterministic_algorithms(det)
    return {
        "cmi": cmi,
        "final_loss": loss,
        "wall_s": time.perf_counter() - t0,
        "device": str(dev),
        "torch": torch.__version__,
    }


def _fit(
    data: Dataset, cols: Columns, config: dict[str, Any], ok_y: list[int], fit_idx: int
) -> dict[str, Any]:
    with_conf = fit_idx == 1
    use = [j for j, a in enumerate(data.action_names) if with_conf or a != PLACEBO_CONF]
    s = np.column_stack([_minmax(cols.actions[:, use]), _z(cols.lags), _minmax(cols.context)])
    seed = fit_seed(data, fit_idx)
    r = fit_cmi(s, _z(cols.Y[:, ok_y]), seed, config)
    r.update(use=use, seed=seed)
    return r


def _facts(data: Dataset, cols: Columns, r: dict[str, Any], ok_y: list[int]) -> dict[str, Any]:
    names = [
        *(data.action_names[j] for j in r["use"]),
        *cols.lag_names,
        *(f"ctx:{c}" for c in range(cols.context.shape[1])),
        "action_node",
    ]
    cmi = {
        data.kpi_names[k]: dict(zip(names, map(float, r["cmi"][i]), strict=True))
        for i, k in enumerate(ok_y)
    }
    keys = ("seed", "final_loss", "wall_s", "device", "torch")
    return {**{k: r[k] for k in keys}, "cmi": cmi}


class CDLMethod(ClassicBase):
    """CDL arm (native only)."""

    name = "cdl"
    version = f"{SOURCE} port, offline 130-epoch rule (cap 16k steps)"
    uses_p = False
    method_idx = 7
    defaults: ClassVar[dict[str, Any]] = {
        "lr": 0.001,
        "batch_size": 128,
        "grad_clip": 10.0,
        "feature_fc_dims": [64, 64],
        "generative_fc_dims": [64, 64],
        "eval_tau": 0.99,
        "eval_steps": 10,
        "epochs": 130,
        "max_gradient_steps": 16000,
        "cmi_threshold": 0.16,
        "device": "cpu",
    }

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        ok_y = [k for k in range(cols.Y.shape[1]) if np.std(cols.Y[:, k]) > 0]
        prim = _fit(data, cols, config, ok_y, 0)
        has_conf = any(s == PLACEBO_CONF for s, _ in data.candidates)
        diag = _fit(data, cols, config, ok_y, 1) if has_conf else None
        thr = float(config["cmi_threshold"])
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            r = diag if s == PLACEBO_CONF and diag is not None else prim
            if j is None or ti not in ok_y or (fam == "action" and j not in r["use"]):
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            node = r["use"].index(j) if fam == "action" else len(r["use"]) + j
            v = float(r["cmi"][ok_y.index(ti), node])
            out.append(
                Scored(
                    s, t, fam, v, sign_pcorr_given_z(data, cols, (fam, j, ti)), None, bool(v >= thr)
                )
            )
        notes: dict[str, Any] = {
            "sign_rule": "pcorr_given_Z",
            "source": SOURCE,
            "gradient_steps": gradient_steps(data.n, config),
            "native_rule": f"conference fixed threshold CMI >= {thr} (secondary)",
            "fit": _facts(data, cols, prim, ok_y),
        }
        if diag is not None:
            notes["diagnostic_fit"] = {
                "rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                "action; primary fit without it",
                **_facts(data, cols, diag, ok_y),
            }
        return out, notes

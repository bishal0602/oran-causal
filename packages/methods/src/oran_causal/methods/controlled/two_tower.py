"""Two-tower interaction learning (arXiv:2601.13213), self-supervised E2 port.

An adaptation, not the published supervised model: the paper trains with BCE against the
ground-truth label matrix, which a truth-free method cannot use. Param tower = the actions at
t, KPI tower = the KPIs at t+1, inputs z-scored; gate a[k, p] = softplus(<U_k, V_p>); the
gate-weighted parameter embeddings feed one head per KPI, trained by MSE plus an L1 penalty
on the gate. Score: the row share a[k, p] / sum_p' a[k, p'] (invariant to the per-head
rescaling of a gate row); `native` = a >= tau_rel * max_p a. Lagged-KPI candidates are not
scorable. Sign: partial correlation given the native set. The primary fit has no
`P_placebo_conf` input; its own candidates come from a second fit with every action.
"""

import dataclasses
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from oran_causal.core.data import Dataset
from oran_causal.methods.controlled._classic import (
    PLACEBO_CONF,
    ClassicBase,
    Columns,
    Scored,
    columns,
    int_seed,
    resolve,
    sign_pcorr_given_z,
    target_index,
)

LABEL = "adaptation (self-supervised E2 port), not the published supervised model"


@dataclasses.dataclass(frozen=True)
class TowerSizes:
    """Sizes of a `TwoTower` model.

    Attributes:
        num_params: Parameter (action) inputs.
        num_kpis: KPI outputs.
        d: Embedding width.
        r: Tower rank.
        hidden: Hidden width of the encoder and heads.
    """

    num_params: int
    num_kpis: int
    d: int = 16
    r: int = 8
    hidden: int = 32


class TwoTower(nn.Module):
    """Two-tower interaction gate with a per-KPI nonlinear reconstruction head."""

    def __init__(self, sizes: TowerSizes):
        """Builds the towers; parameter creation order fixes the seeded initialisation."""
        super().__init__()
        p, k, d, hidden = sizes.num_params, sizes.num_kpis, sizes.d, sizes.hidden
        self.psi = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, d))
        self.id_emb = nn.Parameter(torch.zeros(p, d))
        self.U = nn.Parameter(torch.randn(k, sizes.r) * 0.3)
        self.V = nn.Parameter(torch.randn(p, sizes.r) * 0.3)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1)) for _ in range(k)]
        )
        self.num_params, self.num_kpis, self.d = p, k, d

    def gate(self) -> torch.Tensor:
        """a[k, p] = softplus(<U_k, V_p>), shape `[K, P]`."""
        return torch.nn.functional.softplus(self.U @ self.V.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructed KPIs, shape `[N, K]`."""
        n = x.shape[0]
        e = self.psi(x.reshape(n * self.num_params, 1)).reshape(n, self.num_params, self.d)
        e = e + self.id_emb[None, :, :]
        z = torch.einsum("kp,npd->nkd", self.gate(), e)
        return torch.cat([self.heads[k](z[:, k, :]) for k in range(self.num_kpis)], dim=1)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Training settings of `fit_two_tower`.

    Attributes:
        epochs: Full-batch Adam steps.
        lr: Learning rate.
        l1: Weight of the mean |gate| penalty.
        seed: torch seed.
        device: torch device.
    """

    epochs: int = 800
    lr: float = 0.01
    l1: float = 1e-3
    seed: int = 0
    device: str = "cpu"


def fit_two_tower(
    x: np.ndarray, y: np.ndarray, sizes: TowerSizes, train: TrainConfig | None = None
) -> tuple[np.ndarray, float]:
    """Trains the two-tower reconstruction on z-scored data.

    Returns:
        (gate S[k, p], pseudo-R^2 of the standardized fit).
    """
    train = train or TrainConfig()
    torch.manual_seed(train.seed)
    xz = (x - x.mean(0)) / (x.std(0) + 1e-9)
    yz = (y - y.mean(0)) / (y.std(0) + 1e-9)
    xt = torch.tensor(xz, dtype=torch.float32, device=train.device)
    yt = torch.tensor(yz, dtype=torch.float32, device=train.device)
    model = TwoTower(sizes).to(train.device)
    opt = torch.optim.Adam(model.parameters(), lr=train.lr)
    lossfn = nn.MSELoss()
    for _ in range(train.epochs):
        opt.zero_grad()
        loss = lossfn(model(xt), yt) + train.l1 * model.gate().abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = model.gate().cpu().numpy()
        resid = float(((model(xt) - yt) ** 2).mean())
    return s, float(1.0 - resid)


def _ok_targets(cols: Columns) -> list[int]:
    return [k for k in range(cols.Y.shape[1]) if np.std(cols.Y[:, k]) > 0]


def _fit(
    data: Dataset, cols: Columns, config: dict[str, Any], seed: int, with_conf: bool
) -> tuple[list[int], np.ndarray, float]:
    ok_y = _ok_targets(cols)
    use = [j for j, a in enumerate(data.action_names) if with_conf or a != PLACEBO_CONF]
    x = cols.actions[:, use]
    sizes = TowerSizes(x.shape[1], len(ok_y), config["d"], config["r"], config["hidden"])
    train = TrainConfig(config["epochs"], config["lr"], config["l1"], seed)
    s, r2 = fit_two_tower(x, cols.Y[:, ok_y], sizes, train)
    return use, s, r2


def _gate(
    data: Dataset, use: list[int], s: np.ndarray, ok_y: list[int]
) -> dict[str, dict[str, float]]:
    return {
        data.kpi_names[k]: {data.action_names[a]: float(s[i, p]) for p, a in enumerate(use)}
        for i, k in enumerate(ok_y)
    }


class TwoTowerM(ClassicBase):
    """Two-tower arm (native only)."""

    name = "two_tower"
    version = "scripts/e2_baseline_gnn.fit_two_tower (E2 port), row-share score"
    uses_p = False
    method_idx = 4
    defaults: ClassVar[dict[str, Any]] = {
        "d": 16,
        "r": 8,
        "hidden": 32,
        "epochs": 500,
        "lr": 0.01,
        "l1": 1e-3,
        "tau_rel_native": 0.10,
    }

    def _score(self, data: Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        cols = columns(data)
        ok_y = _ok_targets(cols)
        seed = (
            int_seed(data, self.method_idx) if config.get("seed") is None else int(config["seed"])
        )
        use, s_main, r2 = _fit(data, cols, config, seed, False)
        has_conf = any(s == PLACEBO_CONF for s, _ in data.candidates)
        diag = _fit(data, cols, config, seed, True) if has_conf else None
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            u, sx, _ = diag if s == PLACEBO_CONF and diag is not None else (use, s_main, r2)
            if fam != "action" or ti not in ok_y or j not in u:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            row = sx[ok_y.index(ti)]
            v = float(row[u.index(j)])
            tot = float(row.sum())
            native = bool(row.max() > 0 and v >= config["tau_rel_native"] * float(row.max()))
            share = v / tot if tot > 0 else float("nan")
            out.append(
                Scored(s, t, fam, share, sign_pcorr_given_z(data, cols, (fam, j, ti)), None, native)
            )
        notes: dict[str, Any] = {
            "sign_rule": "pcorr_given_Z",
            "pseudo_r2": float(r2),
            "seed": seed,
            "label": LABEL,
            "score": "row share a[k, p] / sum_p' a[k, p'] (F7 fix; raw gate in notes['gate'])",
            "gate": _gate(data, use, s_main, ok_y),
        }
        if diag is not None:
            notes["diagnostic_fit"] = {
                "rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                "action; primary fit without it",
                "pseudo_r2": float(diag[2]),
                "gate": _gate(data, diag[0], diag[1], ok_y),
            }
        return out, notes

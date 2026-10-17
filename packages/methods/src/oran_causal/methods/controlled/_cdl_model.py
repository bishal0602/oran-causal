"""CDL model: port of the authors' conference implementation (NaNA 2026, Causal Dynamics Learning).

Source: CDD-ORAN branch `refactor/codebase`, `cdd_oran/models/cdl.py` at commit 65b56f3, via
CDD-ORAN@0c89106 `cdd_oran/xmethod/methods/_cdl_model.py`. Changes against the source: no
`CausalModel` base; `target_dim` decouples the predicted variables from the input nodes;
`informed_drop=False` replaces the changed-parameter mask by the uniformly random mask while
drawing the same torch random numbers; `fit_step` and `cmi_step` take (s_t, targets, a).
Only the training and CMI methods the offline arm calls are kept.
"""

import dataclasses
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Normal
from torch.func import functional_call, vmap


def masked_max_from_top_two(features: torch.Tensor) -> torch.Tensor:
    """Each single-source-ablated max pool, without materializing masks."""
    values, indices = torch.topk(features, k=2, dim=2)
    largest = values[:, :, 0, :]
    second = values[:, :, 1, :]
    largest_indices = indices[:, :, 0, :]
    drop_indices = torch.arange(features.shape[2], device=features.device).view(-1, 1, 1, 1)
    return torch.where(
        largest_indices.unsqueeze(0) == drop_indices,
        second.unsqueeze(0),
        largest.unsqueeze(0),
    )


class MLP(nn.Module):
    """Fully connected ReLU network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int]):
        """Builds the layers."""
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Network output."""
        return self.net(x)


class StatePredictor(nn.Module):
    """Per-node linear feature extractors, max-pool and a Gaussian head for one target."""

    def __init__(self, state_dim: int, action_dim: int, feature_dim: int, pred_hidden: list[int]):
        """Builds the extractors and head."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_feature_extractors = nn.ModuleList(
            [MLP(1, feature_dim, []) for _ in range(state_dim)]
        )
        self.action_feature_extractor = MLP(action_dim, feature_dim, [])
        self.predictor = MLP(feature_dim, 2, pred_hidden)

    def features(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Node features, shape `[bs, state_dim + 1, feature_dim]`."""
        feats = [self.state_feature_extractors[i](s[:, i : i + 1]) for i in range(self.state_dim)]
        feats.append(self.action_feature_extractor(a))
        return torch.stack(feats, dim=1)

    def head(self, feats: torch.Tensor, pooled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian mean and std, each `[..., 1]`."""
        h = feats if pooled else feats.max(dim=-2).values
        out = self.predictor(h)
        mu = out[..., 0:1]
        log_std = out[..., 1:2]
        std = torch.exp(torch.clamp(log_std, -5, 2)) + 1e-4
        return mu, std

    def forward(
        self,
        s: Any,
        a: Any,
        mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        return_features: bool = False,
        pooled_features: torch.Tensor | None = None,
    ) -> Any:
        """Features, or (mu, std) from inputs, features or pooled features."""
        if pooled_features is not None:
            return self.head(pooled_features, pooled=True)
        feats = self.features(s, a) if features is None else features
        if return_features:
            return feats
        if mask is not None:
            feats = feats.masked_fill(mask.unsqueeze(-1), float("-inf"))
        return self.head(feats)


@dataclasses.dataclass(frozen=True)
class CdlSettings:
    """Model and optimizer settings of `CDL`.

    Attributes:
        state_dim: Input nodes (the action node comes on top).
        action_dim: Width of the action input.
        target_dim: Predicted variables; None predicts every state node.
        feature_dims: Feature extractor widths; the last is the feature width.
        generative_dims: Hidden widths of the prediction head.
        lr: Adam learning rate.
        eval_tau: EMA factor of the CMI.
        eval_steps: CMI batches averaged per EMA update.
        grad_clip: Gradient-norm clip.
        informed_drop: Drop the changed parameter half of the time (False: random node only).
    """

    state_dim: int
    action_dim: int
    target_dim: int | None
    feature_dims: tuple[int, ...]
    generative_dims: tuple[int, ...]
    lr: float
    eval_tau: float
    eval_steps: int = 10
    grad_clip: float = 10.0
    informed_drop: bool = True


class CDL:
    """One masked predictor per target; full and one-node-masked NLL; EMA of the CMI."""

    def __init__(self, settings: CdlSettings, device: torch.device):
        """Builds the predictors, their vmapped forwards and the Adam optimizer."""
        st = settings
        self.target_dim = st.state_dim if st.target_dim is None else int(st.target_dim)
        self.informed_drop = bool(st.informed_drop)
        self.state_dim = st.state_dim
        self.eval_tau = st.eval_tau
        self.eval_steps = st.eval_steps
        self.grad_clip = st.grad_clip
        self.device = torch.device(device)
        feature_dim = st.feature_dims[-1]
        self.models = nn.ModuleList(
            [
                StatePredictor(st.state_dim, st.action_dim, feature_dim, list(st.generative_dims))
                for _ in range(self.target_dim)
            ]
        ).to(self.device)
        prototype = self.models[0]
        parameter_names = tuple(dict(prototype.named_parameters()))
        parameter_maps = [dict(model.named_parameters()) for model in self.models]
        self._parameter_names = parameter_names
        self._parameter_refs = tuple(
            tuple(parameters[name] for parameters in parameter_maps) for name in parameter_names
        )

        def forward_one(
            parameters: Any,
            model_features: Any,
            model_pooled: Any,
            s: Any,
            a: Any,
            mask: Any,
            return_features: bool,
        ) -> Any:
            return functional_call(
                prototype, parameters, (s, a, mask, model_features, return_features, model_pooled)
            )

        self._vmap_parameters = vmap(forward_one, in_dims=(0, None, None, None, None, None, None))
        self._vmap_features = vmap(forward_one, in_dims=(0, 0, None, None, None, None, None))
        self._vmap_pooled_drop = vmap(forward_one, in_dims=(0, None, 1, None, None, None, None))
        fused = True if self.device.type == "cuda" else None
        self.opt = optim.Adam(self.models.parameters(), lr=st.lr, fused=fused)
        fd, td = st.state_dim, self.target_dim
        self.mask_CMI = torch.zeros(td, fd + 1, device=self.device)
        self._eval_cmi_acc = torch.zeros(td, fd + 1, device=self.device)
        self._eval_step_count = 0

    def _stacked_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.stack(parameters)
            for name, parameters in zip(self._parameter_names, self._parameter_refs, strict=True)
        }

    def _input_features(self, parameters: dict, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._vmap_parameters(parameters, None, None, s, a, None, True)

    def _from_features(self, parameters: dict, feats: torch.Tensor) -> Any:
        return self._vmap_features(parameters, feats, None, None, None, None, False)

    def _from_pooled(self, parameters: dict, pooled: torch.Tensor) -> Any:
        return self._vmap_pooled_drop(parameters, None, pooled, None, None, None, False)

    def _nll(self, mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -Normal(mu, std).log_prob(target)

    def _drop_mask(self, a_batch: torch.Tensor) -> torch.Tensor:
        bs, fd = a_batch.shape[0], self.state_dim
        changed_param_ids = a_batch[:, 0].long()
        use_informed = torch.rand(bs, device=self.device) > 0.5
        if not self.informed_drop:
            use_informed = torch.zeros_like(use_informed)
        random_drop = torch.randint(fd + 1, (bs,), device=self.device)
        drop_idx = torch.where(use_informed, changed_param_ids, random_drop)
        return F.one_hot(drop_idx, fd + 1).bool()

    def fit_step(
        self, s_t: torch.Tensor, s_tp1: torch.Tensor, a_batch: torch.Tensor
    ) -> torch.Tensor:
        """One Adam step on full plus one-node-masked NLL; `s_tp1` holds `[bs, target_dim]`."""
        self.opt.zero_grad()
        mask = self._drop_mask(a_batch)
        self.models.train()
        parameters = self._stacked_parameters()
        feats = self._input_features(parameters, s_t, a_batch)
        mu, std = self._from_features(parameters, feats)
        targets = s_tp1.transpose(0, 1).unsqueeze(-1)
        full_loss = self._nll(mu, std, targets).mean()
        masked_feats = feats.masked_fill(mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
        mu_m, std_m = self._from_features(parameters, masked_feats)
        masked_loss = self._nll(mu_m, std_m, targets).mean()
        loss = full_loss + masked_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.models.parameters(), self.grad_clip)
        self.opt.step()
        return loss.detach()

    def cmi_step(self, s_t: torch.Tensor, s_tp1: torch.Tensor, a_batch: torch.Tensor) -> None:
        """One CMI batch; every `eval_steps` batches the average moves the EMA `mask_CMI`."""
        with torch.no_grad():
            self.models.eval()
            parameters = self._stacked_parameters()
            feats = self._input_features(parameters, s_t, a_batch)
            mu, std = self._from_features(parameters, feats)
            targets = s_tp1.transpose(0, 1).unsqueeze(-1)
            full_nll = self._nll(mu, std, targets)
            mu_m, std_m = self._from_pooled(parameters, masked_max_from_top_two(feats))
            masked_nll = self._nll(mu_m, std_m, targets.unsqueeze(1))
            step_cmi = (masked_nll - full_nll.unsqueeze(1)).mean(dim=(2, 3))
        self._eval_cmi_acc += step_cmi
        self._eval_step_count += 1
        if self._eval_step_count >= self.eval_steps:
            avg_cmi = self._eval_cmi_acc / self.eval_steps
            self.mask_CMI = self.eval_tau * self.mask_CMI + (1 - self.eval_tau) * avg_cmi
            self._eval_cmi_acc.zero_()
            self._eval_step_count = 0

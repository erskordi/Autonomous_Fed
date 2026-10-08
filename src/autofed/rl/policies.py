# MIT License
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Policies: reference rules, actor networks, the tabular policy, and trained-cell records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import Bounded
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, NormalParamExtractor, ProbabilisticActor, TanhNormal

from ..rules import PolicyRule
from .config import normalised_to_rate, rate_to_normalised

__all__ = [
    "ACTION_SPEC", "OBS_DIM", "DeterministicPolicy", "LinearActorNet", "NonlinearActorNet", "ReferenceRule",
    "TabularQPolicy", "TrainedCell", "as_env_policy", "make_ddpg_actor", "make_sac_actor",
]

#: Observation width of each specification: x1 = (y, pi), x2 = (y, y_lag, pi, pi_lag).
OBS_DIM: Final[dict[str, int]] = {"x1": 2, "x2": 4}
#: Unbatched action spec handed to exploration and actor modules.
ACTION_SPEC: Final = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32)


class ReferenceRule(nn.Module):
    """A benchmark rule as a policy module: ``i = max(0, alpha0 + beta_pi * pi + beta_y * y)``."""

    def __init__(self, rule: PolicyRule, action_high: float) -> None:
        super().__init__()
        self.rule = rule
        self.action_high = float(action_high)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        y = obs[..., 0]
        pi = obs[..., 1] if obs.shape[-1] == 2 else obs[..., 2]
        rate = (self.rule.alpha0 + self.rule.beta_pi * pi + self.rule.beta_y * y).clamp_min(0.0)
        return rate_to_normalised(rate, self.action_high).unsqueeze(-1).to(torch.float32)


class LinearActorNet(nn.Module):
    """H&T (2021) linear actor: ``f(x) = alpha_0 + beta' x``, no output squashing."""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(obs_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(obs)


class NonlinearActorNet(nn.Module):
    """H&T (2021) non-linear actor: ``f(x) = alpha_0 + sum_j delta_j tanh(beta_j' x + alpha_j)``."""

    def __init__(self, obs_dim: int, hidden_nodes: int) -> None:
        super().__init__()
        if hidden_nodes < 1:
            raise ValueError(f"hidden_nodes must be positive, got {hidden_nodes}")
        self.hidden = nn.Linear(obs_dim, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.hidden(obs)))


def make_ddpg_actor(obs_dim: int, policy_variant: str, actor_nodes: int) -> TensorDictModule:
    """Deterministic actor of the requested variant, reading ``observation`` and writing ``action``.

    Raises:
        ValueError: If ``policy_variant`` is neither ``"linear"`` nor ``"nonlinear"``.
    """
    if policy_variant not in ("linear", "nonlinear"):
        raise ValueError(f"policy_variant must be 'linear' or 'nonlinear', got {policy_variant!r}")
    net = LinearActorNet(obs_dim) if policy_variant == "linear" else NonlinearActorNet(obs_dim, actor_nodes)
    return TensorDictModule(net, in_keys=["observation"], out_keys=["action"])


def make_sac_actor(obs_dim: int, hidden: Sequence[int]) -> ProbabilisticActor:
    """Tanh-Gaussian actor on a ReLU network."""
    net = nn.Sequential(
        MLP(in_features=obs_dim, out_features=2, num_cells=list(hidden), activation_class=nn.ReLU),
        NormalParamExtractor(),
    )
    return ProbabilisticActor(
        module=TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"], spec=ACTION_SPEC, distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0}, default_interaction_type=ExplorationType.RANDOM,
        return_log_prob=False,
    )


class DeterministicPolicy(nn.Module):
    """Evaluation view of a TorchRL actor: observations to normalised actions, no exploration.

    For a deterministic actor this is the actor itself; for a stochastic one it is the distribution's
    deterministic sample. Outputs are clipped to the action spec, as the environment does.
    """

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.actor = actor

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"observation": obs.to(torch.float32)}, batch_size=obs.shape[:-1])
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            return self.actor(td)["action"].clamp(-1.0, 1.0)


class TabularQPolicy(nn.Module):
    """Q-table over a discretised ``(y, pi)`` grid and a discrete rate grid; ``forward`` is greedy.

    Args:
        action_high: Upper bound of the policy rate used by the environment.
        n_bins: Number of bins along ``(y, pi)``.
        action_grid: Candidate policy rates in percent; 0 to 10 in steps of 0.5 when omitted.
        obs_low: Lower edge of the grid; observations below fall in the first bin.
        obs_high: Upper edge of the grid; observations above fall in the last bin.

    Raises:
        ValueError: If the action grid or the bin edges are malformed.
    """

    def __init__(self, action_high: float, n_bins: tuple[int, int] = (6, 7),
                 action_grid: Optional[Sequence[float]] = None, obs_low: tuple[float, float] = (-6.0, -2.0),
                 obs_high: tuple[float, float] = (6.0, 6.0)) -> None:
        super().__init__()
        grid = (torch.arange(0.0, 10.5, 0.5, dtype=torch.float64) if action_grid is None
                else torch.as_tensor(action_grid, dtype=torch.float64))
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("action_grid must be one-dimensional with at least two rates")
        if any(hi <= lo for lo, hi in zip(obs_low, obs_high)):
            raise ValueError("obs_high must exceed obs_low in every dimension")
        self.action_high = float(action_high)
        self.register_buffer("action_grid", grid)
        self.register_buffer("obs_low", torch.tensor(obs_low, dtype=torch.float64))
        self.register_buffer("obs_high", torch.tensor(obs_high, dtype=torch.float64))
        self.register_buffer("bins", torch.tensor(n_bins, dtype=torch.long))
        self.register_buffer("Q", torch.zeros(*n_bins, grid.numel(), dtype=torch.float64))

    def discretise(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bin indices of ``(y, pi)`` for a batch of observations."""
        x = (obs[..., :2].to(torch.float64) - self.obs_low) / (self.obs_high - self.obs_low)
        idx = torch.minimum(torch.floor(x * self.bins).long().clamp_min(0), self.bins - 1)
        return idx[..., 0], idx[..., 1]

    def greedy_index(self, obs: torch.Tensor) -> torch.Tensor:
        """Index of the value-maximising rate."""
        iy, ipi = self.discretise(obs)
        return self.Q[iy, ipi].argmax(dim=-1)

    def action_for(self, index: torch.Tensor) -> torch.Tensor:
        """Normalised action of a rate-grid index, shaped for the environment."""
        return rate_to_normalised(self.action_grid[index], self.action_high).unsqueeze(-1).to(torch.float32)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_for(self.greedy_index(obs))


def as_env_policy(policy: nn.Module) -> TensorDictModule:
    """Wrap an observation-to-action module so TorchRL rollouts can call it."""
    return TensorDictModule(policy, in_keys=["observation"], out_keys=["action"])


@dataclass
class TrainedCell:
    """One trained (algorithm, economy, observation specification, policy variant) cell.

    Attributes:
        algo: ``"DDPG"``, ``"SAC"`` or ``"Q-learning"``.
        env_name: Economy the policy was trained in.
        obs_spec: ``"x1"`` or ``"x2"``.
        policy_variant: ``"linear"``, ``"nonlinear"`` or ``None`` for the tabular policy.
        critic_nodes: Hidden size of the selected critic.
        actor_nodes: Hidden size of the selected actor.
        steady_state_reward: Selection statistic of the winning agent.
        policy: Deterministic observation-to-normalised-action module of the selected agent.
        action_high: Upper bound of the policy rate the action is scaled by.
        training_rewards: Undiscounted return of every training episode, in order.
    """

    algo: str
    env_name: str
    obs_spec: str
    policy_variant: Optional[str]
    critic_nodes: Optional[int]
    actor_nodes: Optional[int]
    steady_state_reward: float
    policy: nn.Module
    action_high: float
    training_rewards: list[float] = field(default_factory=list)

    @torch.no_grad()
    def rate(self, obs: torch.Tensor) -> torch.Tensor:
        """Policy rate in percent for a batch of observations, ``(..., obs_dim) -> (...)``."""
        return normalised_to_rate(self.policy(obs.to(torch.float32))[..., 0].to(torch.float64), self.action_high)

    @property
    def variant(self) -> str:
        """Policy variant as a label; ``"tabular"`` for the Q-learning policy."""
        return self.policy_variant if self.policy_variant else "tabular"

    @property
    def label(self) -> str:
        """Short label within one economy, e.g. ``"DDPG x1 linear"``."""
        return f"{self.algo} {self.obs_spec} {self.variant}"

    @property
    def key(self) -> str:
        """File-safe identifier, e.g. ``"ddpg_svar_x1_linear"``."""
        return f"{self.algo}_{self.env_name}_{self.obs_spec}_{self.variant}".lower().replace("-", "_")

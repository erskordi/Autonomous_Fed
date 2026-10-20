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

import math
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

from ..rules import TR93, PolicyRule
from .config import normalised_to_rate, rate_to_normalised

__all__ = [
    "ACTION_SPEC", "OBS_DIM", "DeterministicPolicy", "LinearActorNet", "NonlinearActorNet", "ReferenceRule",
    "PI_EDGES", "TabularQPolicy", "TrainedCell", "Y_EDGES", "ZLB_LEAK", "as_env_policy", "make_ddpg_actor", "make_sac_actor",
    "obs_dim", "zero_lower_bound",
]

#: Macro width of each observation specification: x1 = (y, pi), x2 = (y, y_lag, pi, pi_lag).
OBS_DIM: Final[dict[str, int]] = {"x1": 2, "x2": 4}
#: Unbatched action spec handed to exploration and actor modules.
ACTION_SPEC: Final = Bounded(low=-1.0, high=1.0, shape=(1,), dtype=torch.float32)


def obs_dim(obs_spec: str, lagged_rate: bool) -> int:
    """Width of a policy observation: the macro block of ``obs_spec`` plus ``i_{t-1}`` if observed.

    Raises:
        ValueError: If ``obs_spec`` is not ``"x1"`` or ``"x2"``.
    """
    if obs_spec not in OBS_DIM:
        raise ValueError(f"obs_spec must be one of {sorted(OBS_DIM)}, got {obs_spec!r}")
    return OBS_DIM[obs_spec] + (1 if lagged_rate else 0)


def _macro(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(y, pi)`` of an observation of any width (x1 or x2, with or without the lagged rate)."""
    return obs[..., 0], (obs[..., 1] if obs.shape[-1] in (2, 3) else obs[..., 2])


class ReferenceRule(nn.Module):
    """A benchmark rule as a policy module: ``i = max(0, alpha0 + beta_pi * pi + beta_y * y)``."""

    def __init__(self, rule: PolicyRule, action_high: float) -> None:
        super().__init__()
        self.rule = rule
        self.action_high = float(action_high)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        y, pi = _macro(obs)
        rate = (self.rule.alpha0 + self.rule.beta_pi * pi + self.rule.beta_y * y).clamp_min(0.0)
        return rate_to_normalised(rate, self.action_high).unsqueeze(-1).to(torch.float32)


#: Slope of the actor output below the zero lower bound. H&T's ReLU has none, which leaves an actor that
#: has drifted below the bound everywhere without a gradient; a small slope keeps one alive, and the
#: environment's clip still makes the realised rate exactly zero.
ZLB_LEAK: Final = 0.01


def zero_lower_bound(f: torch.Tensor) -> torch.Tensor:
    """H&T (2021, eq. 12) ``i = max{f, 0}`` in normalised units, ``a = max{f, -1}``, with a leaky slope below."""
    return torch.nn.functional.leaky_relu(f + 1.0, ZLB_LEAK) - 1.0


class LinearActorNet(nn.Module):
    """H&T (2021) linear actor, eq. 12-13: ``i = max{alpha_0 + beta' x, 0}``.

    The zero lower bound sits inside the actor, as in H&T eq. 12, through :func:`zero_lower_bound`; the
    upper bound of the rate is left to the environment.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(obs_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return zero_lower_bound(self.mu(obs))


class NonlinearActorNet(nn.Module):
    """H&T (2021) non-linear actor, eq. 12-13: ``i = max{alpha_0 + sum_j delta_j tanh(beta_j' x + alpha_j), 0}``."""

    def __init__(self, obs_dim: int, hidden_nodes: int) -> None:
        super().__init__()
        if hidden_nodes < 1:
            raise ValueError(f"hidden_nodes must be positive, got {hidden_nodes}")
        self.hidden = nn.Linear(obs_dim, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return zero_lower_bound(self.out(torch.tanh(self.hidden(obs))))


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


#: Interior bin edges of the output gap, percent: finer around the closed gap and the stopping band.
Y_EDGES: Final[tuple[float, ...]] = (-4.0, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, 4.0)
#: Interior bin edges of inflation, percent: finer around the 2 percent target and its stopping band.
PI_EDGES: Final[tuple[float, ...]] = (0.0, 1.0, 1.7, 2.3, 3.0, 4.0, 6.0)


class TabularQPolicy(nn.Module):
    """Q-table over a binned ``(y, pi)`` state and a discrete rate grid; ``forward`` is greedy.

    Bins are defined by their interior edges, so they can be fine around the target and coarse in the
    tails; observations beyond the outer edges fall in the outer bins. Only the first two columns of an
    observation (``y``, ``pi``) are used. The table also counts visits, and the greedy choice is taken
    over visited actions only: every reward is non-positive, so an unvisited action's initial value of
    zero would otherwise beat every visited one. A state the agent never visited falls back to
    ``fallback``, the TR93 rule by default.

    Args:
        action_high: Upper bound of the policy rate used by the environment.
        action_grid: Candidate policy rates in percent; ``0`` to ``action_high`` in steps of ``0.5``
            when omitted.
        y_edges: Interior bin edges of the output gap, increasing.
        pi_edges: Interior bin edges of inflation, increasing.
        fallback: Rule applied in states without any visit.

    Raises:
        ValueError: If the action grid or the bin edges are malformed.
    """

    action_grid: torch.Tensor
    y_edges: torch.Tensor
    pi_edges: torch.Tensor
    Q: torch.Tensor
    N: torch.Tensor

    def __init__(self, action_high: float, action_grid: Optional[Sequence[float]] = None,
                 y_edges: Sequence[float] = Y_EDGES, pi_edges: Sequence[float] = PI_EDGES,
                 fallback: PolicyRule = TR93) -> None:
        super().__init__()
        grid = (torch.arange(0.0, float(action_high) + 0.25, 0.5, dtype=torch.float64) if action_grid is None
                else torch.as_tensor(action_grid, dtype=torch.float64))
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("action_grid must be one-dimensional with at least two rates")
        y_e = torch.as_tensor(list(y_edges), dtype=torch.float64)
        pi_e = torch.as_tensor(list(pi_edges), dtype=torch.float64)
        for label, edges in (("y_edges", y_e), ("pi_edges", pi_e)):
            if edges.ndim != 1 or edges.numel() < 1 or bool((edges[1:] <= edges[:-1]).any()):
                raise ValueError(f"{label} must be a non-empty increasing sequence")
        self.action_high = float(action_high)
        self.fallback = fallback
        self.register_buffer("action_grid", grid)
        self.register_buffer("y_edges", y_e)
        self.register_buffer("pi_edges", pi_e)
        self.register_buffer("Q", torch.zeros(y_e.numel() + 1, pi_e.numel() + 1, grid.numel(), dtype=torch.float64))
        self.register_buffer("N", torch.zeros(y_e.numel() + 1, pi_e.numel() + 1, grid.numel(), dtype=torch.long))

    @property
    def bins(self) -> tuple[int, int]:
        """Number of bins along ``(y, pi)``."""
        return int(self.Q.shape[0]), int(self.Q.shape[1])

    def discretise(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bin indices of ``(y, pi)`` for a batch of observations."""
        y, pi = obs[..., 0].to(torch.float64), obs[..., 1].to(torch.float64)
        return torch.bucketize(y, self.y_edges), torch.bucketize(pi, self.pi_edges)

    def greedy_index(self, obs: torch.Tensor) -> torch.Tensor:
        """Index of the value-maximising visited rate; the fallback rule's nearest rate in unvisited states."""
        iy, ipi = self.discretise(obs)
        q, n = self.Q[iy, ipi], self.N[iy, ipi]
        greedy = q.masked_fill(n == 0, -math.inf).argmax(dim=-1)
        y, pi = obs[..., 0].to(torch.float64), obs[..., 1].to(torch.float64)
        fallback_rate = (self.fallback.alpha0 + self.fallback.beta_pi * pi + self.fallback.beta_y * y).clamp_min(0.0)
        fallback = (fallback_rate.unsqueeze(-1) - self.action_grid).abs().argmin(dim=-1)
        return torch.where(n.sum(dim=-1) > 0, greedy, fallback)

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
        steady_state_reward: Selection statistic of the winning agent: the reward at the deterministic
            closed-loop steady state (H&T 2021), ``-inf`` if the closed loop has none.
        policy: Deterministic observation-to-normalised-action module of the selected agent.
        action_high: Upper bound of the policy rate the action is scaled by.
        training_rewards: Undiscounted return of every training episode, in order.
        lagged_rate: Whether the policy observes ``i_{t-1}`` after the macro block.
        eval_reward: Mean per-step reward of the selected agent over the stochastic evaluation
            episodes; ``nan`` if not evaluated.
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
    lagged_rate: bool = False
    eval_reward: float = math.nan

    @property
    def obs_dim(self) -> int:
        """Width of the observation the policy reads."""
        return obs_dim(self.obs_spec, self.lagged_rate)

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

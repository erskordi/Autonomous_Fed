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

"""Hyper-parameters of the reinforcement-learning exercise, and the reward they define."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from ..exceptions import ArtifactError

__all__ = ["RLConfig", "cb_reward", "is_terminal", "normalised_to_rate", "rate_to_normalised"]


@dataclass(frozen=True)
class RLConfig:
    """Every setting of the reinforcement-learning notebooks, in one place.

    Attributes:
        pi_star: Inflation target, percent.
        omega_pi: Reward weight on squared inflation deviations (H&T 2021, eq. 16).
        omega_y: Reward weight on the squared output gap.
        omega_di: Sack-Wieland (2000) weight on squared policy-rate changes; ``0`` gives H&T's reward.
        gamma: Discount factor.
        t_max: Episode length at which an episode is truncated, in quarters.
        action_high: Upper bound of the policy rate, percent; the lower bound is zero.
        penalty_band: Absolute deviation beyond which the H&T footnote-13 penalty applies.
        penalty_weight: Weight of that penalty.
        stop_pi_band: Half-width of the inflation stopping band.
        stop_y_band: Half-width of the output-gap stopping band.
        macro_clamp_pi: Safety clamp on ``|pi - pi_star|``; ``inf`` disables it.
        macro_clamp_y: Safety clamp on ``|y|``; ``inf`` disables it.
        select_er_es_min: H&T's original agent-selection threshold on episode reward per step.
        select_er_es_min_search: Looser threshold used while collecting candidates.
        total_timesteps: Environment frames per training run.
        critic_node_grid: Critic hidden sizes searched by DDPG.
        actor_node_grid: Actor hidden sizes searched by the non-linear DDPG actor.
        ss_rollout_steps: Length of the steady-state-reward rollout.
        candidate_save_every: Keep every n-th qualifying episode's checkpoint.
        q_episodes: Episodes of tabular Q-learning.
        num_envs: Parallel copies of the economy per training run.
        learning_starts: Frames of uniformly random actions before updates begin.
        batch_size: Replay minibatch size.
        tau: Polyak coefficient of the target networks.
        ddpg_lr: Adam learning rate of the DDPG actor and critic.
        ddpg_noise_sigma: Gaussian exploration noise, in normalised action units.
        sac_lr: Adam learning rate of the SAC actor, critics and entropy coefficient.
        sac_hidden: Hidden layer sizes of the SAC actor and critics.
        q_alpha: Q-learning step size.
        q_epsilon: Initial exploration probability of Q-learning.
        q_epsilon_min: Floor of the exploration probability.
        q_epsilon_decay: Per-episode decay of the exploration probability.
        seed: Seed of every training run and of every environment's shock generator.
    """

    pi_star: float = 2.0
    omega_pi: float = 0.5
    omega_y: float = 0.5
    omega_di: float = 0.5
    gamma: float = 0.99
    t_max: int = 50
    action_high: float = 15.0
    penalty_band: float = 2.0
    penalty_weight: float = 10.0
    stop_pi_band: float = 0.3
    stop_y_band: float = 0.3
    macro_clamp_pi: float = 10.0
    macro_clamp_y: float = 10.0
    select_er_es_min: float = -4.0
    select_er_es_min_search: float = -50.0
    total_timesteps: int = 12_500
    critic_node_grid: tuple[int, ...] = (1, 2)
    actor_node_grid: tuple[int, ...] = (1, 4, 8)
    ss_rollout_steps: int = 100
    candidate_save_every: int = 5
    q_episodes: int = 2_000
    num_envs: int = 1
    learning_starts: int = 500
    batch_size: int = 64
    tau: float = 0.005
    ddpg_lr: float = 1e-3
    ddpg_noise_sigma: float = 0.3
    sac_lr: float = 3e-4
    sac_hidden: tuple[int, int] = (64, 64)
    q_alpha: float = 0.2
    q_epsilon: float = 0.2
    q_epsilon_min: float = 0.02
    q_epsilon_decay: float = 0.999
    seed: int = 2026

    def __post_init__(self) -> None:
        for name in ("critic_node_grid", "actor_node_grid", "sac_hidden"):
            object.__setattr__(self, name, tuple(int(v) for v in getattr(self, name)))
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.action_high <= 0 or self.t_max < 1 or self.total_timesteps < 1 or self.num_envs < 1:
            raise ValueError("action_high, t_max, total_timesteps and num_envs must be positive")
        if not self.critic_node_grid or not self.actor_node_grid or min(self.critic_node_grid + self.actor_node_grid) < 1:
            raise ValueError("node grids must be non-empty and positive")

    def with_full_grid(self) -> RLConfig:
        """The paper's full architecture grid, hidden sizes 1 to 10 for critic and actor."""
        return replace(self, critic_node_grid=tuple(range(1, 11)), actor_node_grid=tuple(range(1, 11)))

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the configuration to JSON and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: ("inf" if isinstance(v, float) and math.isinf(v) else v) for k, v in asdict(self).items()}
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> RLConfig:
        """Read a configuration written by :meth:`save`.

        Raises:
            ArtifactError: If the file is missing or malformed.
        """
        source = Path(path)
        if not source.is_file():
            raise ArtifactError(f"{source} not found; run notebook 07_rl_environments first")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            return cls(**{k: (math.inf if v == "inf" else v) for k, v in raw.items()})
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"{source} is not a valid RL configuration: {exc}") from exc


def cb_reward(config: RLConfig, pi_next: torch.Tensor, y_next: torch.Tensor, rate: torch.Tensor,
              rate_prev: torch.Tensor) -> torch.Tensor:
    """Central-bank reward, element-wise: H&T (2021, eq. 16 and fn. 13) plus Sack-Wieland smoothing.

    ``r = -omega_pi (pi - pi*)^2 - omega_y y^2 - omega_di (i_t - i_{t-1})^2``, with an additional
    ``-penalty_weight * deviation^2`` for each of ``pi`` and ``y`` deviating by more than
    ``penalty_band``.
    """
    dev_pi = pi_next - config.pi_star
    base = -config.omega_pi * dev_pi**2 - config.omega_y * y_next**2 - config.omega_di * (rate - rate_prev) ** 2
    zero = torch.zeros_like(dev_pi)
    penalty = -config.penalty_weight * (
        torch.where(dev_pi.abs() > config.penalty_band, dev_pi**2, zero)
        + torch.where(y_next.abs() > config.penalty_band, y_next**2, zero)
    )
    return base + penalty


def is_terminal(config: RLConfig, pi_next: torch.Tensor, y_next: torch.Tensor) -> torch.Tensor:
    """True where the macro state has reached the H&T stopping band."""
    return ((pi_next - config.pi_star).abs() < config.stop_pi_band) & (y_next.abs() < config.stop_y_band)


def normalised_to_rate(action: torch.Tensor, action_high: float) -> torch.Tensor:
    """Map a normalised action in ``[-1, 1]`` to a policy rate in ``[0, action_high]``; clips first."""
    return 0.5 * (action.clamp(-1.0, 1.0) + 1.0) * action_high


def rate_to_normalised(rate: torch.Tensor, action_high: float) -> torch.Tensor:
    """Inverse of :func:`normalised_to_rate`; rates outside ``[0, action_high]`` are clipped."""
    return (2.0 * rate / action_high - 1.0).clamp(-1.0, 1.0)

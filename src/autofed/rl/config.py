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

__all__ = ["RLConfig", "cb_penalty", "cb_reward", "is_terminal", "normalised_to_rate", "rate_to_normalised"]


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
        observe_lagged_rate: Append ``i_{t-1}`` to the policy observation, so the smoothing term of the
            reward is a function of the state the agent sees. Ignored (never appended) when ``omega_di``
            is zero.
        economies: Economies the algorithm notebooks train in; notebook 07 sets this to the economies
            that pass the transmission check.
        transmission_horizon: Quarters over which the transmission check holds the rate up.
        select_er_es_min: Agent-selection threshold on episode reward per step (H&T 2021: -4).
        total_timesteps: Environment frames per training run.
        critic_node_grid: Critic hidden sizes searched by DDPG.
        actor_node_grid: Actor hidden sizes searched by the non-linear DDPG actor.
        ss_horizon: Quarters the deterministic closed loop is run to find its steady state.
        ss_tol: Largest quarter-to-quarter change, in percentage points, still counted as converged.
        eval_episodes: Episodes of the stochastic evaluation of a selected agent.
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
        q_alpha: Initial Q-learning step size; the step on a pair visited ``n`` times is
            ``q_alpha / n ** q_alpha_decay``.
        q_alpha_decay: Exponent of the visit-count step-size decay; ``0`` keeps the step constant.
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
    observe_lagged_rate: bool = True
    economies: tuple[str, ...] = ("SVAR", "TVP-SVAR-SV", "NARX", "NSSM")
    transmission_horizon: int = 20
    select_er_es_min: float = -4.0
    total_timesteps: int = 12_500
    critic_node_grid: tuple[int, ...] = (1, 2)
    actor_node_grid: tuple[int, ...] = (1, 4, 8, 10)
    ss_horizon: int = 800
    ss_tol: float = 1e-4
    eval_episodes: int = 50
    candidate_save_every: int = 1
    q_episodes: int = 20_000
    num_envs: int = 1
    learning_starts: int = 500
    batch_size: int = 64
    tau: float = 0.005
    ddpg_lr: float = 1e-3
    ddpg_noise_sigma: float = 0.3
    sac_lr: float = 3e-4
    sac_hidden: tuple[int, int] = (64, 64)
    q_alpha: float = 0.5
    q_alpha_decay: float = 0.6
    q_epsilon: float = 0.2
    q_epsilon_min: float = 0.02
    q_epsilon_decay: float = 0.9999
    seed: int = 2026

    def __post_init__(self) -> None:
        for name in ("critic_node_grid", "actor_node_grid", "sac_hidden"):
            object.__setattr__(self, name, tuple(int(v) for v in getattr(self, name)))
        object.__setattr__(self, "economies", tuple(str(e) for e in self.economies))
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.action_high <= 0 or self.t_max < 1 or self.total_timesteps < 1 or self.num_envs < 1:
            raise ValueError("action_high, t_max, total_timesteps and num_envs must be positive")
        if not self.critic_node_grid or not self.actor_node_grid or min(self.critic_node_grid + self.actor_node_grid) < 1:
            raise ValueError("node grids must be non-empty and positive")
        if self.ss_horizon < 2 or self.ss_tol <= 0 or self.eval_episodes < 1 or self.transmission_horizon < 1:
            raise ValueError("ss_horizon, ss_tol, eval_episodes and transmission_horizon must be positive")
        if self.q_alpha <= 0 or self.q_alpha_decay < 0 or self.candidate_save_every < 1:
            raise ValueError("q_alpha and candidate_save_every must be positive, q_alpha_decay non-negative")

    @property
    def observes_rate(self) -> bool:
        """Whether policies see ``i_{t-1}``: requested and relevant (``omega_di > 0``)."""
        return bool(self.observe_lagged_rate and self.omega_di > 0.0)

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
            raise ArtifactError(f"{source} is not a valid RL configuration ({exc}); re-run notebook 07_rl_environments") from exc


def cb_penalty(config: RLConfig, pi_next: torch.Tensor, y_next: torch.Tensor) -> torch.Tensor:
    """The H&T (2021, fn. 13) penalty, element-wise and non-negative.

    ``penalty_weight * dev^2`` for each of ``pi - pi*`` and ``y`` whose absolute deviation exceeds
    ``penalty_band``; the term is discontinuous at the band edge by construction.
    """
    dev_pi = pi_next - config.pi_star
    zero = torch.zeros_like(dev_pi)
    return config.penalty_weight * (
        torch.where(dev_pi.abs() > config.penalty_band, dev_pi**2, zero)
        + torch.where(y_next.abs() > config.penalty_band, y_next**2, zero)
    )


def cb_reward(config: RLConfig, pi_next: torch.Tensor, y_next: torch.Tensor, rate: torch.Tensor,
              rate_prev: torch.Tensor) -> torch.Tensor:
    """Central-bank reward, element-wise: H&T (2021, eq. 16 and fn. 13) plus Sack-Wieland smoothing.

    ``r = -omega_pi (pi - pi*)^2 - omega_y y^2 - omega_di (i_t - i_{t-1})^2 - penalty``, with the
    penalty of :func:`cb_penalty`.
    """
    dev_pi = pi_next - config.pi_star
    base = -config.omega_pi * dev_pi**2 - config.omega_y * y_next**2 - config.omega_di * (rate - rate_prev) ** 2
    return base - cb_penalty(config, pi_next, y_next)


def is_terminal(config: RLConfig, pi_next: torch.Tensor, y_next: torch.Tensor) -> torch.Tensor:
    """True where the macro state has reached the H&T stopping band."""
    return ((pi_next - config.pi_star).abs() < config.stop_pi_band) & (y_next.abs() < config.stop_y_band)


def normalised_to_rate(action: torch.Tensor, action_high: float) -> torch.Tensor:
    """Map a normalised action in ``[-1, 1]`` to a policy rate in ``[0, action_high]``; clips first."""
    return 0.5 * (action.clamp(-1.0, 1.0) + 1.0) * action_high


def rate_to_normalised(rate: torch.Tensor, action_high: float) -> torch.Tensor:
    """Inverse of :func:`normalised_to_rate`; rates outside ``[0, action_high]`` are clipped."""
    return (2.0 * rate / action_high - 1.0).clamp(-1.0, 1.0)

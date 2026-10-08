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

"""Training: the shared off-policy loop, DDPG and SAC agents, tabular Q-learning, agent selection."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, RandomSampler, TensorDictReplayBuffer
from torchrl.envs import EnvBase
from torchrl.modules import MLP, AdditiveGaussianModule, ValueOperator
from torchrl.objectives import DDPGLoss, SACLoss, SoftUpdate, ValueEstimators

from .config import RLConfig
from .lab import PolicyLab
from .policies import ACTION_SPEC, OBS_DIM, DeterministicPolicy, TabularQPolicy, TrainedCell, make_ddpg_actor, make_sac_actor
from .transitions import ENV_DEVICE

__all__ = [
    "AgentCandidate", "EpisodeTracker", "OffPolicyAgent", "build_ddpg_agent", "build_sac_agent",
    "select_best_candidate", "train_ddpg_cell", "train_off_policy", "train_q_cell", "train_sac_cell",
]

_REPLAY_KEYS: Final = ("observation", "action", ("next", "observation"), ("next", "reward"), ("next", "done"),
                       ("next", "terminated"))


@dataclass
class AgentCandidate:
    """Checkpoint of an actor taken when a training episode passed the selection filter."""

    state_dict: dict[str, torch.Tensor]
    episode: int
    ER: float
    ES: int


class EpisodeTracker:
    """H&T (2021) agent-selection bookkeeping over a stream of collected frames.

    Reads the episode return and length that the ``RewardSum`` and ``StepCounter`` transforms attach
    to every frame, and checkpoints the actor on every n-th episode with reward per step above the
    search threshold and a length in ``(1, t_max]``.

    Args:
        actor: Module whose ``state_dict`` is checkpointed.
        config: Thresholds and the checkpoint cadence.
    """

    def __init__(self, actor: nn.Module, config: RLConfig) -> None:
        self.actor = actor
        self.config = config
        self.candidates: list[AgentCandidate] = []
        self.episode_rewards: list[float] = []
        self._qualifying_seen = 0

    def observe(self, frames: TensorDictBase) -> None:
        """Record the episodes that ended in ``frames``."""
        done = frames["next", "done"].reshape(-1)
        if not bool(done.any()):
            return
        cfg = self.config
        returns = frames["next", "episode_reward"].reshape(-1)[done].tolist()
        lengths = frames["next", "step_count"].reshape(-1)[done].tolist()
        for ER, ES in zip(returns, lengths):
            self.episode_rewards.append(float(ER))
            if not (ER / max(ES, 1) > cfg.select_er_es_min_search and 1 < ES <= cfg.t_max):
                continue
            self._qualifying_seen += 1
            if self._qualifying_seen % cfg.candidate_save_every == 0:
                self.candidates.append(AgentCandidate(
                    {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
                    len(self.episode_rewards), float(ER), int(ES)))


@dataclass
class OffPolicyAgent:
    """Everything :func:`train_off_policy` needs.

    Attributes:
        actor: The TorchRL actor (checkpointed, and evaluated through ``DeterministicPolicy``).
        behaviour: Exploration policy handed to the collector.
        loss_module: TorchRL loss returning one tensor per entry of ``loss_keys``.
        loss_keys: Loss terms to sum and back-propagate.
        optimisers: One optimiser per parameter group.
        target_updater: Polyak updater of the target networks.
    """

    actor: nn.Module
    behaviour: nn.Module
    loss_module: nn.Module
    loss_keys: tuple[str, ...]
    optimisers: list[torch.optim.Optimizer]
    target_updater: SoftUpdate


def build_ddpg_agent(obs_dim: int, policy_variant: str, actor_nodes: int, critic_nodes: int,
                     config: RLConfig) -> OffPolicyAgent:
    """DDPG: unsquashed actor, single-hidden-layer ReLU critic, target copies of both, Gaussian exploration."""
    actor = make_ddpg_actor(obs_dim, policy_variant, actor_nodes)
    critic = ValueOperator(
        MLP(in_features=obs_dim + 1, out_features=1, num_cells=[critic_nodes], activation_class=nn.ReLU),
        in_keys=["observation", "action"],
    )
    loss_module = DDPGLoss(actor_network=actor, value_network=critic, loss_function="l2", delay_actor=True, delay_value=True)
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config.gamma)
    noise = AdditiveGaussianModule(spec=ACTION_SPEC, sigma_init=config.ddpg_noise_sigma, sigma_end=config.ddpg_noise_sigma,
                                   annealing_num_steps=1)
    return OffPolicyAgent(
        actor=actor, behaviour=TensorDictSequential(actor, noise), loss_module=loss_module,
        loss_keys=("loss_actor", "loss_value"),
        optimisers=[
            torch.optim.Adam(list(loss_module.actor_network_params.flatten_keys().values()), lr=config.ddpg_lr),
            torch.optim.Adam(list(loss_module.value_network_params.flatten_keys().values()), lr=config.ddpg_lr),
        ],
        target_updater=SoftUpdate(loss_module, tau=config.tau),
    )


def build_sac_agent(obs_dim: int, config: RLConfig) -> OffPolicyAgent:
    """SAC: tanh-Gaussian actor, twin Q critics, learned entropy coefficient, target entropy -1."""
    actor = make_sac_actor(obs_dim, config.sac_hidden)
    qvalue = ValueOperator(
        MLP(in_features=obs_dim + 1, out_features=1, num_cells=list(config.sac_hidden), activation_class=nn.ReLU),
        in_keys=["observation", "action"],
    )
    loss_module = SACLoss(actor_network=actor, qvalue_network=qvalue, num_qvalue_nets=2, loss_function="l2",
                          alpha_init=1.0, target_entropy=-1.0, delay_qvalue=True)
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config.gamma)
    return OffPolicyAgent(
        actor=actor, behaviour=actor, loss_module=loss_module, loss_keys=("loss_actor", "loss_qvalue", "loss_alpha"),
        optimisers=[
            torch.optim.Adam(list(loss_module.actor_network_params.flatten_keys().values()), lr=config.sac_lr),
            torch.optim.Adam(list(loss_module.qvalue_network_params.flatten_keys().values()), lr=config.sac_lr),
            torch.optim.Adam([loss_module.log_alpha], lr=config.sac_lr),
        ],
        target_updater=SoftUpdate(loss_module, tau=config.tau),
    )


def train_off_policy(agent: OffPolicyAgent, env: EnvBase, config: RLConfig) -> EpisodeTracker:
    """Collect, store and update for ``total_timesteps`` frames; return the episode bookkeeping.

    A TorchRL ``Collector`` steps the environment with the behaviour policy (uniformly random actions
    for the first ``learning_starts`` frames). Every frame goes into a replay buffer, and after the
    warm-up each collected frame is followed by one gradient step on a uniformly sampled minibatch
    and one Polyak update of the target networks.
    """
    tracker = EpisodeTracker(agent.actor, config)
    num_envs = int(env.batch_size.numel())
    collector = Collector(env, agent.behaviour, frames_per_batch=num_envs, total_frames=config.total_timesteps,
                          init_random_frames=config.learning_starts, device=ENV_DEVICE)
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(config.total_timesteps + num_envs),
                                    sampler=RandomSampler(), batch_size=config.batch_size)
    collected = 0
    try:
        for batch in collector:
            frames = batch.reshape(-1)
            buffer.extend(frames.select(*_REPLAY_KEYS))
            tracker.observe(frames)
            collected += frames.numel()
            if collected <= config.learning_starts:
                continue
            for _ in range(frames.numel()):                      # one gradient step per collected frame
                losses = agent.loss_module(buffer.sample())
                for optimiser in agent.optimisers:
                    optimiser.zero_grad(set_to_none=True)
                sum(losses[key] for key in agent.loss_keys).backward()
                for optimiser in agent.optimisers:
                    optimiser.step()
                agent.target_updater.step()
    finally:
        collector.shutdown()
    return tracker


def select_best_candidate(lab: PolicyLab, actor: nn.Module, policy: nn.Module, candidates: list[AgentCandidate],
                          env_name: str, *, observation_lag: bool) -> tuple[Optional[AgentCandidate], float]:
    """Load the candidate with the highest steady-state reward into ``actor``.

    Returns:
        ``(candidate, reward)``; ``(None, -inf)`` if there were no candidates, in which case the
        actor keeps its final weights.
    """
    final_state = copy.deepcopy(actor.state_dict())
    init = lab.initial_state()
    best_reward, best = -math.inf, None
    for candidate in candidates:
        actor.load_state_dict(candidate.state_dict)
        reward = lab.steady_state_reward(policy, env_name, observation_lag=observation_lag, init_state=init)
        if reward > best_reward:
            best_reward, best = reward, candidate
    actor.load_state_dict(best.state_dict if best is not None else final_state)
    return best, best_reward


def _obs_lag(obs_spec: str) -> bool:
    if obs_spec not in OBS_DIM:
        raise ValueError(f"obs_spec must be one of {sorted(OBS_DIM)}, got {obs_spec!r}")
    return obs_spec == "x2"


def train_ddpg_cell(lab: PolicyLab, env_name: str, obs_spec: str, policy_variant: str) -> tuple[TrainedCell, pd.DataFrame]:
    """Architecture search for one DDPG cell.

    Trains one agent per (critic size, actor size) pair, each from the same seed, picks the best
    checkpoint of each by steady-state reward, and keeps the best architecture.

    Returns:
        ``(cell, log)``: the winning cell and one row per architecture with its steady-state reward
        and number of candidate checkpoints.
    """
    cfg, obs_lag = lab.config, _obs_lag(obs_spec)
    actor_grid = (1,) if policy_variant == "linear" else cfg.actor_node_grid
    best: Optional[TrainedCell] = None
    rows = []
    for critic_nodes in cfg.critic_node_grid:
        for actor_nodes in actor_grid:
            torch.manual_seed(cfg.seed)
            agent = build_ddpg_agent(OBS_DIM[obs_spec], policy_variant, actor_nodes, critic_nodes, cfg)
            tracker = train_off_policy(agent, lab.make_env(env_name, observation_lag=obs_lag, num_envs=cfg.num_envs), cfg)
            policy = DeterministicPolicy(agent.actor)
            _, reward = select_best_candidate(lab, agent.actor, policy, tracker.candidates, env_name, observation_lag=obs_lag)
            rows.append({"Critic nodes": critic_nodes, "Actor nodes": actor_nodes, "SS reward": reward,
                         "Candidates": len(tracker.candidates), "Episodes": len(tracker.episode_rewards)})
            if best is None or reward > best.steady_state_reward:
                best = TrainedCell("DDPG", env_name, obs_spec, policy_variant, critic_nodes, actor_nodes, reward, policy,
                                   cfg.action_high, list(tracker.episode_rewards))
    assert best is not None
    return best, pd.DataFrame(rows)


def train_sac_cell(lab: PolicyLab, env_name: str, obs_spec: str) -> TrainedCell:
    """Train one SAC cell and select its best checkpoint by steady-state reward."""
    cfg, obs_lag = lab.config, _obs_lag(obs_spec)
    torch.manual_seed(cfg.seed)
    agent = build_sac_agent(OBS_DIM[obs_spec], cfg)
    tracker = train_off_policy(agent, lab.make_env(env_name, observation_lag=obs_lag, num_envs=cfg.num_envs), cfg)
    policy = DeterministicPolicy(agent.actor)
    _, reward = select_best_candidate(lab, agent.actor, policy, tracker.candidates, env_name, observation_lag=obs_lag)
    return TrainedCell("SAC", env_name, obs_spec, "nonlinear", cfg.sac_hidden[0], cfg.sac_hidden[0], reward, policy,
                       cfg.action_high, list(tracker.episode_rewards))


@torch.no_grad()
def train_q_cell(lab: PolicyLab, env_name: str) -> TrainedCell:
    """Tabular Q-learning (Watkins, 1989) on the no-lag observation.

    Epsilon-greedy exploration with per-episode decay; the update does not bootstrap past the end of
    an episode. The reported statistic is the mean return of the last 200 training episodes.
    """
    cfg = lab.config
    env = lab.make_env(env_name, observation_lag=False)
    policy = TabularQPolicy(cfg.action_high)
    rng = torch.Generator().manual_seed(cfg.seed)
    n_actions = policy.action_grid.numel()
    epsilon = cfg.q_epsilon
    rewards: list[float] = []

    td = env.reset()
    for _ in range(cfg.q_episodes):
        total = 0.0
        while True:
            obs = td["observation"]
            if float(torch.rand((), generator=rng)) < epsilon:
                a_idx = torch.randint(n_actions, obs.shape[:-1], generator=rng)
            else:
                a_idx = policy.greedy_index(obs)
            td.set("action", policy.action_for(a_idx))
            stepped, td = env.step_and_maybe_reset(td)

            reward = stepped["next", "reward"][..., 0].to(torch.float64)
            done = stepped["next", "done"][..., 0]
            iy, ipi = policy.discretise(obs)
            iy2, ipi2 = policy.discretise(stepped["next", "observation"])
            target = reward + cfg.gamma * policy.Q[iy2, ipi2].max(dim=-1).values * (~done)
            policy.Q[iy, ipi, a_idx] += cfg.q_alpha * (target - policy.Q[iy, ipi, a_idx])

            total += float(reward)
            if bool(done):
                break
        epsilon = max(cfg.q_epsilon_min, epsilon * cfg.q_epsilon_decay)
        rewards.append(total)
    return TrainedCell("Q-learning", env_name, "x1", None, None, None, float(np.mean(rewards[-200:])), policy,
                       cfg.action_high, rewards)

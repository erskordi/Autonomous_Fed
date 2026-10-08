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

"""Reinforcement learning for monetary policy on the estimated economies.

Needs PyTorch and TorchRL (``pip install autofed[rl]``); the rest of the package imports without them.

    from autofed import rl

    lab = rl.PolicyLab.load(nb.workspace.artifacts)
    cell, log = rl.train_ddpg_cell(lab, "SVAR", "x1", "linear")
"""

from __future__ import annotations

import logging

from .cells import load_cell, load_cells, save_cell, save_cells
from .config import RLConfig, cb_reward, is_terminal, normalised_to_rate, rate_to_normalised
from .env import MonetaryEnv, episodic
from .evaluation import coefficient_table, linearise_policy, policy_surface
from .lab import N_INIT, PolicyLab
from .policies import (
    ACTION_SPEC,
    OBS_DIM,
    DeterministicPolicy,
    LinearActorNet,
    NonlinearActorNet,
    ReferenceRule,
    TabularQPolicy,
    TrainedCell,
    as_env_policy,
    make_ddpg_actor,
    make_sac_actor,
)
from .robustness import build_rs99_F, rs99_table, rs99_variances, solve_discrete_lyapunov
from .training import (
    AgentCandidate,
    EpisodeTracker,
    OffPolicyAgent,
    build_ddpg_agent,
    build_sac_agent,
    select_best_candidate,
    train_ddpg_cell,
    train_off_policy,
    train_q_cell,
    train_sac_cell,
)
from .transitions import NARXTransition, NSSMTransition, RecursiveLinearTransition, Transition

# TorchRL logs a line for every replay storage it initialises; one per training run is noise here.
logging.getLogger("torchrl").setLevel(logging.WARNING)

__all__ = [
    "ACTION_SPEC", "N_INIT", "OBS_DIM", "AgentCandidate", "DeterministicPolicy", "EpisodeTracker", "LinearActorNet",
    "MonetaryEnv", "NARXTransition", "NSSMTransition", "NonlinearActorNet", "OffPolicyAgent", "PolicyLab", "RLConfig",
    "RecursiveLinearTransition", "ReferenceRule", "TabularQPolicy", "TrainedCell", "Transition", "as_env_policy",
    "build_ddpg_agent", "build_rs99_F", "build_sac_agent", "cb_reward", "coefficient_table", "episodic", "is_terminal", "linearise_policy",
    "load_cell", "load_cells", "make_ddpg_actor", "make_sac_actor", "normalised_to_rate", "policy_surface",
    "rate_to_normalised", "rs99_table", "rs99_variances", "save_cell", "save_cells", "select_best_candidate",
    "solve_discrete_lyapunov", "train_ddpg_cell", "train_off_policy", "train_q_cell", "train_sac_cell",
]

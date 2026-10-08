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

"""The policy laboratory: the four estimated economies behind one object."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs

from ..data import NAMES, load_macro_panel
from ..linear.svar import RecursiveLinearSpec
from ..nonlinear.economy import NARXEconomy, NSSMEconomy
from ..rules import REFERENCE_RULES
from .config import RLConfig, normalised_to_rate
from .env import MonetaryEnv, episodic
from .policies import ReferenceRule, TrainedCell, as_env_policy
from .transitions import ENV_DTYPE, NARXTransition, NSSMTransition, RecursiveLinearTransition, Transition

__all__ = ["N_INIT", "PolicyLab"]

#: Quarters of initial conditions at the head of every simulated path.
N_INIT: Final = 2


@dataclass
class PolicyLab:
    """The estimated economies, the training data and the settings of the RL exercise.

    Attributes:
        config: Hyper-parameters.
        transitions: Economy name to transition module.
        train: Estimation-window data with columns ``y``, ``pi``, ``i`` on a quarterly index.
    """

    config: RLConfig
    transitions: dict[str, Transition]
    train: pd.DataFrame

    @classmethod
    def load(cls, artifacts: str | os.PathLike[str], *, config: Optional[RLConfig] = None) -> PolicyLab:
        """Assemble the laboratory from the artifacts of notebooks 00 to 07.

        Args:
            artifacts: The shared artifacts directory.
            config: Hyper-parameters; read from ``rl/config.json`` when omitted.

        Raises:
            ArtifactError: If a required artifact is missing; the message names the notebook to run.
        """
        root = Path(artifacts)
        cfg = config if config is not None else RLConfig.load(root / "rl" / "config.json")
        panel = load_macro_panel(root / "data")
        transitions: dict[str, Transition] = {
            "SVAR": RecursiveLinearTransition(RecursiveLinearSpec.load(root / "linear" / "svar_restricted.json")),
            "TVP-SVAR-SV": RecursiveLinearTransition(RecursiveLinearSpec.load(root / "linear" / "tvp_anchor.json")),
            "NARX": NARXTransition(NARXEconomy.load(root / "nonlinear" / "narx.pt")),
            "NSSM": NSSMTransition(NSSMEconomy.load(root / "nonlinear" / "nssm.pt")),
        }
        return cls(cfg, transitions, panel.train)

    @property
    def env_names(self) -> list[str]:
        """Economy names in presentation order."""
        return list(self.transitions)

    @property
    def data(self) -> torch.Tensor:
        """Training observations as a ``(T, 3)`` tensor in ``(y, pi, i)`` order."""
        return torch.tensor(self.train[list(NAMES)].to_numpy(dtype=float), dtype=ENV_DTYPE)

    def initial_state(self) -> dict[str, float]:
        """The first two training quarters, the ``(t-1, t)`` pair every simulated path starts from."""
        first, second = self.train.iloc[0], self.train.iloc[1]
        return {"pi_t": float(second["pi"]), "pi_tm1": float(first["pi"]), "y_t": float(second["y"]),
                "y_tm1": float(first["y"]), "i_t": float(second["i"]), "i_tm1": float(first["i"])}

    def make_env(self, env_name: str, *, observation_lag: bool, seed: Optional[int] = None, num_envs: int = 1,
                 init_state: Optional[dict[str, float]] = None, with_episodes: bool = True) -> EnvBase:
        """Build an economy environment.

        ``with_episodes=True`` adds the stopping band, truncation at ``t_max`` and episode-return
        tracking; ``False`` returns the bare economy for fixed-horizon paths.

        Raises:
            KeyError: If ``env_name`` is not one of the economies.
        """
        if env_name not in self.transitions:
            raise KeyError(f"unknown economy {env_name!r}; expected one of {self.env_names}")
        base = MonetaryEnv(self.transitions[env_name], self.data, self.config, observation_lag=observation_lag,
                           num_envs=num_envs, init_state=init_state, terminate_in_band=with_episodes, seed=seed)
        return episodic(base) if with_episodes else base

    def check(self, env_name: str) -> None:
        """Run TorchRL's spec check on the economy, single and batched, with the episode transforms."""
        for num_envs in (1, 3):
            check_env_specs(self.make_env(env_name, observation_lag=True, num_envs=num_envs))

    def reference_rules(self) -> dict[str, ReferenceRule]:
        """TR93, NPP and BA as policy modules."""
        return {name: ReferenceRule(rule, self.config.action_high) for name, rule in REFERENCE_RULES.items()}

    @torch.no_grad()
    def counterfactual(self, env_name: str, policy: nn.Module, *, observation_lag: bool,
                       seed: Optional[int] = None) -> pd.DataFrame:
        """Simulate the economy under ``policy`` from the first two training quarters.

        The shock sequence depends only on ``seed``, so every policy faces the same shocks within an
        economy. The path spans the training index in the timing of the data: the output gap and
        inflation of the first two quarters and the rate of the first are the realised initial
        conditions; from the second quarter on the rate is the policy's.

        Returns:
            A frame with columns ``y``, ``pi`` and ``i``.
        """
        init = self.initial_state()
        horizon = len(self.train) - N_INIT
        env = self.make_env(env_name, observation_lag=observation_lag, seed=seed, init_state=init, with_episodes=False)
        rollout = env.rollout(max_steps=horizon, policy=as_env_policy(policy), auto_reset=True, break_when_any_done=False)
        simulated = rollout["next", "macro"][0].numpy()                      # (H, 3): pi_{t+1}, y_{t+1}, i_t
        last_action = policy(rollout["next", "observation"][0, -1:])          # the rate set in the final quarter
        last_rate = float(normalised_to_rate(last_action.to(ENV_DTYPE), self.config.action_high).reshape(-1)[0])
        y = np.concatenate([[init["y_tm1"], init["y_t"]], simulated[:, 1]])
        pi = np.concatenate([[init["pi_tm1"], init["pi_t"]], simulated[:, 0]])
        rate = np.concatenate([[init["i_tm1"]], simulated[:, 2], [last_rate]])   # row t holds the rate set in t
        return pd.DataFrame({"y": y, "pi": pi, "i": rate}, index=self.train.index)

    @torch.no_grad()
    def steady_state_reward(self, policy: nn.Module, env_name: str, *, observation_lag: bool,
                            init_state: Optional[dict[str, float]] = None) -> float:
        """Average per-step reward of ``policy`` over ``ss_rollout_steps``, restarting at episode ends."""
        env = self.make_env(env_name, observation_lag=observation_lag, init_state=init_state)
        rollout = env.rollout(max_steps=self.config.ss_rollout_steps, policy=as_env_policy(policy), auto_reset=True,
                              break_when_any_done=False)
        return float(rollout["next", "reward"].mean())

    def loss(self, path: pd.DataFrame) -> dict[str, float]:
        """Mean squared deviations and the loss mirroring the reward, over the quarters after ``N_INIT``."""
        cfg = self.config
        pi, y, i = (path[c].to_numpy(dtype=float) for c in ("pi", "y", "i"))
        dev2_pi = float(np.mean((pi[N_INIT:] - cfg.pi_star) ** 2))
        dev2_y = float(np.mean(y[N_INIT:] ** 2))
        dev2_di = float(np.mean(np.diff(i[N_INIT - 2: -1]) ** 2))        # the rate changes the reward penalises
        return {"dev2 pi": dev2_pi, "dev2 y": dev2_y, "dev2 di": dev2_di,
                "Loss": cfg.omega_pi * dev2_pi + cfg.omega_y * dev2_y + cfg.omega_di * dev2_di}

    def simulate_policies(self, env_name: str, cells: Iterable[TrainedCell] = ()) -> dict[str, pd.DataFrame]:
        """Paths in one economy: the realised data, the three reference rules, and each given cell.

        Cells trained in another economy are ignored. Keys are ``"Actual"``, the rule names and the
        cells' labels; every policy faces the same shock sequence.
        """
        paths: dict[str, pd.DataFrame] = {"Actual": self.train[list(NAMES)]}
        for name, rule in self.reference_rules().items():
            paths[name] = self.counterfactual(env_name, rule, observation_lag=False)
        for cell in cells:
            if cell.env_name == env_name:
                paths[cell.label] = self.counterfactual(env_name, cell.policy, observation_lag=cell.obs_spec == "x2")
        return paths

    def loss_table(self, paths_by_economy: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
        """Loss decomposition of every path, indexed by ``(economy, policy)``."""
        rows = {(economy, policy): self.loss(path) for economy, paths in paths_by_economy.items()
                for policy, path in paths.items()}
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index = pd.MultiIndex.from_tuples(table.index, names=["Economy", "Policy"])
        return table

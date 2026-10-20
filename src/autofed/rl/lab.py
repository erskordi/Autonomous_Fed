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


"""The policy laboratory: the estimated economies behind one object."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
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
from ..exceptions import SpecificationError
from ..linear.svar import RecursiveLinearSpec
from ..nonlinear.economy import NARXEconomy, NSSMEconomy
from ..rules import REFERENCE_RULES
from .config import RLConfig, cb_penalty, cb_reward, normalised_to_rate
from .env import MonetaryEnv, episodic
from .policies import ReferenceRule, TrainedCell, as_env_policy
from .transitions import ENV_DTYPE, NARXTransition, NSSMTransition, RecursiveLinearTransition, Transition

__all__ = ["ECONOMIES", "N_INIT", "PolicyLab"]

#: Quarters of initial conditions at the head of every simulated path.
N_INIT: Final = 2
#: Every estimated economy, in presentation order, with the artifact it is loaded from.
ECONOMIES: Final[dict[str, str]] = {
    "SVAR": "linear/svar_restricted.json",
    "TVP-SVAR-SV": "linear/tvp_anchor.json",
    "NARX": "nonlinear/narx.pt",
    "NSSM": "nonlinear/nssm.pt",
}


def _load_transition(name: str, root: Path, train: pd.DataFrame) -> Transition:
    path = root / ECONOMIES[name]
    if name in ("SVAR", "TVP-SVAR-SV"):
        return RecursiveLinearTransition(RecursiveLinearSpec.load(path))
    if name == "NARX":
        return NARXTransition(NARXEconomy.load(path))
    return NSSMTransition(NSSMEconomy.load(path), train)


@dataclass
class PolicyLab:
    """The estimated economies, the training data and the settings of the RL exercise.

    Attributes:
        config: Hyper-parameters.
        transitions: Economy name to transition module, in presentation order.
        train: Estimation-window data with columns ``y``, ``pi``, ``i`` on a quarterly index.
    """

    config: RLConfig
    transitions: dict[str, Transition]
    train: pd.DataFrame

    @classmethod
    def load(cls, artifacts: str | os.PathLike[str], *, config: Optional[RLConfig] = None,
             economies: Optional[Sequence[str]] = None) -> PolicyLab:
        """Assemble the laboratory from the artifacts of notebooks 00 to 07.

        Args:
            artifacts: The shared artifacts directory.
            config: Hyper-parameters; read from ``rl/config.json`` when omitted.
            economies: Economies to load; ``config.economies`` when omitted. Notebook 07 loads all of
                :data:`ECONOMIES` to run the transmission check and records the passing ones in the
                configuration, which the algorithm notebooks then follow.

        Raises:
            ArtifactError: If a required artifact is missing; the message names the notebook to run.
            KeyError: If an economy name is unknown.
        """
        root = Path(artifacts)
        cfg = config if config is not None else RLConfig.load(root / "rl" / "config.json")
        names = list(cfg.economies if economies is None else economies)
        unknown = [n for n in names if n not in ECONOMIES]
        if unknown:
            raise KeyError(f"unknown economies {unknown}; expected a subset of {list(ECONOMIES)}")
        panel = load_macro_panel(root / "data")
        transitions = {name: _load_transition(name, root, panel.train) for name in ECONOMIES if name in names}
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
                "y_tm1": float(first["y"]), "i_t": float(second["i"]), "i_tm1": float(first["i"]), "index": 1}

    def make_env(self, env_name: str, *, observation_lag: bool, seed: Optional[int] = None, num_envs: int = 1,
                 init_state: Optional[dict[str, float]] = None, with_episodes: bool = True,
                 observe_rate: Optional[bool] = None, noise_scale: float = 1.0) -> EnvBase:
        """Build an economy environment.

        ``with_episodes=True`` adds the stopping band, truncation at ``t_max`` and episode-return
        tracking; ``False`` returns the bare economy for fixed-horizon paths. ``observe_rate`` defaults
        to the configuration; ``noise_scale=0`` gives the deterministic economy.

        Raises:
            KeyError: If ``env_name`` is not one of the economies.
        """
        if env_name not in self.transitions:
            raise KeyError(f"unknown economy {env_name!r}; expected one of {self.env_names}")
        base = MonetaryEnv(self.transitions[env_name], self.data, self.config, observation_lag=observation_lag,
                           observe_rate=observe_rate, num_envs=num_envs, init_state=init_state,
                           terminate_in_band=with_episodes, noise_scale=noise_scale, seed=seed)
        return episodic(base) if with_episodes else base

    def check(self, env_name: str) -> None:
        """Run TorchRL's spec check on the economy, single and batched, with the episode transforms."""
        for num_envs in (1, 3):
            check_env_specs(self.make_env(env_name, observation_lag=True, num_envs=num_envs))

    # ------------------------------------------------------------------ transmission
    @torch.no_grad()
    def transmission(self, env_name: str, *, horizon: Optional[int] = None, shock: float = 1.0) -> pd.DataFrame:
        """Response of the economy to a policy-rate increase held for ``horizon`` quarters.

        From every training quarter (the third on), the deterministic economy (``eps = 0``) is run
        ``horizon`` quarters twice, with the rate held at the realised rate of that quarter and at the
        realised rate plus ``shock``; the difference is averaged over the starting quarters.

        Returns:
            A frame indexed by horizon ``1..H`` with columns ``dy`` and ``dpi`` (percentage points per
            ``shock`` percentage points) and ``cum_dpi``, the cumulative inflation response.
        """
        tr = self.transitions[env_name]
        data, H = self.data, int(self.config.transmission_horizon if horizon is None else horizon)
        T = data.shape[0]
        idx = torch.arange(2, T)
        base = torch.stack([data[idx, 0], data[idx - 1, 0], data[idx, 1], data[idx - 1, 1], data[idx, 2], data[idx - 1, 2]], dim=-1)

        def run(delta: float) -> torch.Tensor:
            state = base.clone()
            state[:, 4] += delta
            hidden = tr.initial_hidden(torch.Size([state.shape[0]]), idx)
            out = []
            for _ in range(H):
                y_new, pi_new, hidden = tr(state, torch.zeros(state.shape[0], 2, dtype=ENV_DTYPE), hidden)
                out.append(torch.stack([y_new, pi_new], dim=-1))
                state = torch.stack([y_new, state[:, 0], pi_new, state[:, 2], state[:, 4], state[:, 4]], dim=-1)
            return torch.stack(out, dim=0)                      # (H, n_starts, 2)

        diff = (run(shock) - run(0.0)).mean(dim=1).numpy()
        frame = pd.DataFrame({"dy": diff[:, 0], "dpi": diff[:, 1]}, index=pd.RangeIndex(1, H + 1, name="h"))
        frame["cum_dpi"] = frame["dpi"].cumsum()
        return frame

    def transmission_table(self, horizons: Sequence[int] = (1, 4, 8, 20)) -> pd.DataFrame:
        """Transmission of every loaded economy at a few horizons, with the pass/fail verdict.

        An economy passes when inflation responds negatively to a sustained rate increase at the
        configured horizon, the channel a stabilising rule needs. The output-gap response is reported
        but not required: H&T's own Table 9 SVAR has a positive one at every horizon.
        """
        H = max(int(self.config.transmission_horizon), *horizons)
        rows = {}
        for name in self.env_names:
            resp = self.transmission(name, horizon=H)
            row = {}
            for h in horizons:
                row[f"dy, h={h}"] = float(resp["dy"].to_numpy()[h - 1])
                row[f"dpi, h={h}"] = float(resp["dpi"].to_numpy()[h - 1])
            row["Passes"] = self.passes_transmission(name, resp)
            rows[name] = row
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "Economy"
        return table

    def passes_transmission(self, env_name: str, response: Optional[pd.DataFrame] = None) -> bool:
        """Whether inflation falls after a sustained rate increase, at the configured horizon."""
        resp = self.transmission(env_name) if response is None else response
        return bool(float(resp["dpi"].to_numpy()[int(self.config.transmission_horizon) - 1]) < 0.0)

    def passing_economies(self) -> tuple[str, ...]:
        """Loaded economies that pass :meth:`passes_transmission`, in presentation order."""
        return tuple(name for name in self.env_names if self.passes_transmission(name))

    def require_transmission(self) -> None:
        """Raise unless every loaded economy passes the transmission check.

        Raises:
            SpecificationError: Naming the failing economies.
        """
        failing = [name for name in self.env_names if not self.passes_transmission(name)]
        if failing:
            raise SpecificationError(
                f"inflation does not fall after a rate increase in {failing}; these economies cannot be used "
                "as policy environments (see notebook 07_rl_environments, transmission check)")

    # ------------------------------------------------------------------ policies and paths
    def reference_rules(self) -> dict[str, ReferenceRule]:
        """TR93, NPP and BA as policy modules."""
        return {name: ReferenceRule(rule, self.config.action_high) for name, rule in REFERENCE_RULES.items()}

    @torch.no_grad()
    def counterfactual(self, env_name: str, policy: nn.Module, *, observation_lag: bool,
                       observe_rate: Optional[bool] = None, seed: Optional[int] = None) -> pd.DataFrame:
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
        env = self.make_env(env_name, observation_lag=observation_lag, observe_rate=observe_rate, seed=seed,
                            init_state=init, with_episodes=False)
        rollout = env.rollout(max_steps=horizon, policy=as_env_policy(policy), auto_reset=True, break_when_any_done=False)
        simulated = rollout["next", "macro"][0].numpy()                      # (H, 3): pi_{t+1}, y_{t+1}, i_t
        last_action = policy(rollout["next", "observation"][0, -1:])          # the rate set in the final quarter
        last_rate = float(normalised_to_rate(last_action.to(ENV_DTYPE), self.config.action_high).reshape(-1)[0])
        y = np.concatenate([[init["y_tm1"], init["y_t"]], simulated[:, 1]])
        pi = np.concatenate([[init["pi_tm1"], init["pi_t"]], simulated[:, 0]])
        rate = np.concatenate([[init["i_tm1"]], simulated[:, 2], [last_rate]])   # row t holds the rate set in t
        return pd.DataFrame({"y": y, "pi": pi, "i": rate}, index=self.train.index)

    @torch.no_grad()
    def steady_state(self, policy: nn.Module, env_name: str, *, observation_lag: bool,
                     observe_rate: Optional[bool] = None) -> Optional[tuple[float, float, float]]:
        """Deterministic closed-loop steady state ``(pi, y, i)`` under ``policy``, or ``None``.

        The economy without shocks is run from the first two training quarters for ``ss_horizon``
        quarters with no termination and no reset; the path has converged if none of its last eight
        quarter-to-quarter changes exceeds ``ss_tol``. A limit cycle or a divergent path returns ``None``.
        """
        cfg = self.config
        env = self.make_env(env_name, observation_lag=observation_lag, observe_rate=observe_rate,
                            init_state=self.initial_state(), with_episodes=False, noise_scale=0.0)
        rollout = env.rollout(max_steps=cfg.ss_horizon, policy=as_env_policy(policy), auto_reset=True,
                              break_when_any_done=False)
        macro = rollout["next", "macro"][0]                                   # (H, 3): pi, y, i
        tail = macro[-9:]
        if not bool(torch.isfinite(tail).all()) or not bool((tail[1:] - tail[:-1]).abs().max() <= cfg.ss_tol):
            return None
        return float(tail[-1, 0]), float(tail[-1, 1]), float(tail[-1, 2])

    def steady_state_reward(self, policy: nn.Module, env_name: str, *, observation_lag: bool,
                            observe_rate: Optional[bool] = None) -> float:
        """H&T (2021) selection statistic: the reward at the closed-loop steady state, ``-inf`` if none.

        At a steady state the rate does not change, so the smoothing term is zero and the statistic is
        eq. 16 (with fn. 13) evaluated at the fixed point.
        """
        ss = self.steady_state(policy, env_name, observation_lag=observation_lag, observe_rate=observe_rate)
        if ss is None:
            return -math.inf
        pi, y, i = (torch.tensor(v, dtype=ENV_DTYPE) for v in ss)
        return float(cb_reward(self.config, pi, y, i, i))

    @torch.no_grad()
    def evaluation_reward(self, policy: nn.Module, env_name: str, *, observation_lag: bool,
                          observe_rate: Optional[bool] = None, episodes: Optional[int] = None,
                          seed: Optional[int] = None) -> float:
        """Mean per-step reward over complete stochastic episodes from random training-data starts.

        Episodes follow the training design (stopping band, truncation at ``t_max``); the statistic is
        the total reward of the episodes divided by their total length.
        """
        cfg = self.config
        n = int(cfg.eval_episodes if episodes is None else episodes)
        env = self.make_env(env_name, observation_lag=observation_lag, observe_rate=observe_rate,
                            seed=cfg.seed + 1 if seed is None else seed)
        rollout = env.rollout(max_steps=n * cfg.t_max, policy=as_env_policy(policy), auto_reset=True,
                              break_when_any_done=False)
        reward = rollout["next", "reward"].reshape(-1)
        done = rollout["next", "done"].reshape(-1)
        ends = torch.nonzero(done).reshape(-1)
        last = int(ends[min(n, ends.numel()) - 1]) + 1 if ends.numel() else reward.numel()
        return float(reward[:last].mean())

    def loss(self, path: pd.DataFrame) -> dict[str, float]:
        """Mean squared deviations, the H&T penalty and the loss, over the quarters after ``N_INIT``.

        ``Loss`` weights the three mean squared deviations as the reward does; ``penalty`` is the mean
        footnote-13 term, which the reward also carries, so ``Loss + penalty`` is minus the mean reward.
        """
        cfg = self.config
        pi, y, i = (path[c].to_numpy(dtype=float) for c in ("pi", "y", "i"))
        dev2_pi = float(np.mean((pi[N_INIT:] - cfg.pi_star) ** 2))
        dev2_y = float(np.mean(y[N_INIT:] ** 2))
        dev2_di = float(np.mean(np.diff(i[N_INIT - 2: -1]) ** 2))        # the rate changes the reward penalises
        penalty = float(cb_penalty(cfg, torch.tensor(pi[N_INIT:]), torch.tensor(y[N_INIT:])).mean())
        return {"dev2 pi": dev2_pi, "dev2 y": dev2_y, "dev2 di": dev2_di,
                "Loss": cfg.omega_pi * dev2_pi + cfg.omega_y * dev2_y + cfg.omega_di * dev2_di, "penalty": penalty}

    def simulate_policies(self, env_name: str, cells: Iterable[TrainedCell] = ()) -> dict[str, pd.DataFrame]:
        """Paths in one economy: the realised data, the three reference rules, and each given cell.

        Cells trained in another economy are ignored. Keys are ``"Actual"``, the rule names and the
        cells' labels; every policy faces the same shock sequence.
        """
        paths: dict[str, pd.DataFrame] = {"Actual": self.train[list(NAMES)]}
        for name, rule in self.reference_rules().items():
            paths[name] = self.counterfactual(env_name, rule, observation_lag=False, observe_rate=False)
        for cell in cells:
            if cell.env_name == env_name:
                paths[cell.label] = self.counterfactual(env_name, cell.policy, observation_lag=cell.obs_spec == "x2",
                                                        observe_rate=cell.lagged_rate)
        return paths

    def loss_table(self, paths_by_economy: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
        """Loss decomposition of every path, indexed by ``(economy, policy)``."""
        rows = {(economy, policy): self.loss(path) for economy, paths in paths_by_economy.items()
                for policy, path in paths.items()}
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index = pd.MultiIndex.from_tuples(table.index, names=["Economy", "Policy"])
        return table

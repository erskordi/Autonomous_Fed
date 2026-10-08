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

"""The monetary-policy environment: one TorchRL ``EnvBase`` around any transition."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import Compose, EnvBase, InitTracker, RewardSum, StepCounter, TransformedEnv

from .config import RLConfig, cb_reward, is_terminal, normalised_to_rate
from .transitions import ENV_DEVICE, ENV_DTYPE, Transition

__all__ = ["MonetaryEnv", "episodic"]

_INIT_KEYS = ("y_t", "y_tm1", "pi_t", "pi_tm1", "i_t", "i_tm1")


class MonetaryEnv(EnvBase):
    """Monetary-policy environment around a :class:`~autofed.rl.transitions.Transition`.

    Timing follows the recursive ordering of the estimated economies: in quarter ``t`` the agent
    observes ``(y_t, pi_t)`` and sets ``i_t``, which enters the equations of quarter ``t + 1`` as the
    first lag of the rate, with ``i_{t-1}`` as the second. The realised rate of the starting quarter is
    therefore not used; the agent's first action takes its place.

    The environment is stateless in the TorchRL sense: the lag state (and a latent state, if the
    economy has one) travels in the TensorDict, so partial resets and any number of parallel copies
    work without extra code.

    TensorDict layout (leading batch dimension ``N = num_envs``)::

        observation  (N, 2 | 4) float32   policy input, x1 = (y, pi) or x2 = (y, y_lag, pi, pi_lag)
        lags         (N, 6)     float64   (y_t, y_{t-1}, pi_t, pi_{t-1}, i_{t-1}, i_{t-2}); state
        hidden       (N, H)     float64   latent state, only for economies that have one; state
        macro        (N, 3)     float64   (pi_{t+1}, y_{t+1}, i_t) after a step, for logging
        action       (N, 1)     float32   normalised rate in [-1, 1]

    Args:
        transition: The economy.
        data: ``(T, 3)`` training observations in ``(y, pi, i)`` column order; episodes start from a
            uniformly drawn pair of consecutive quarters.
        config: Reward, bounds and clamps.
        observation_lag: Use the ``x2`` observation instead of ``x1``.
        num_envs: Number of parallel copies.
        init_state: If given, every reset starts from this state instead of a draw from ``data``.
            Keys: ``pi_t, pi_tm1, y_t, y_tm1, i_t, i_tm1``; ``i_t`` is not used (see the timing note).
        terminate_in_band: Emit ``terminated=True`` inside the stopping band. Disable for
            fixed-horizon paths.
        seed: Seed of the environment's own shock and initial-state generator; the configured seed
            when omitted.

    Raises:
        ValueError: If ``num_envs`` is not positive or ``data`` is not ``(T >= 3, 3)``.
        KeyError: If ``init_state`` lacks a key.
    """

    def __init__(self, transition: Transition, data: torch.Tensor, config: RLConfig, *, observation_lag: bool = False,
                 num_envs: int = 1, init_state: Optional[dict[str, float]] = None, terminate_in_band: bool = True,
                 seed: Optional[int] = None) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] < 3:
            raise ValueError(f"data must have shape (T >= 3, 3), got {tuple(data.shape)}")
        super().__init__(device=ENV_DEVICE, batch_size=torch.Size([num_envs]))
        self.transition = transition
        self.config = config
        self.observation_lag = bool(observation_lag)
        self.terminate_in_band = bool(terminate_in_band)
        self._data = data.to(dtype=ENV_DTYPE, device=ENV_DEVICE)
        self._init_lags: Optional[torch.Tensor] = None
        if init_state is not None:
            missing = [k for k in _INIT_KEYS if k not in init_state]
            if missing:
                raise KeyError(f"init_state is missing {missing}")
            # The rate of quarter t is the agent's first decision, so the lag slots hold i_{t-1}.
            order = ("y_t", "y_tm1", "pi_t", "pi_tm1", "i_tm1", "i_tm1")
            self._init_lags = torch.tensor([float(init_state[k]) for k in order], dtype=ENV_DTYPE)
            self._init_rate = float(init_state["i_t"])
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(config.seed if seed is None else seed))
        self._make_specs()

    @property
    def obs_dim(self) -> int:
        """Width of the policy observation."""
        return 4 if self.observation_lag else 2

    def _make_specs(self) -> None:
        bs, cfg = self.batch_size, self.config
        if math.isfinite(cfg.macro_clamp_pi) and math.isfinite(cfg.macro_clamp_y):
            y_b = (-cfg.macro_clamp_y, cfg.macro_clamp_y)
            pi_b = (cfg.pi_star - cfg.macro_clamp_pi, cfg.pi_star + cfg.macro_clamp_pi)
            bounds = (y_b, y_b, pi_b, pi_b) if self.observation_lag else (y_b, pi_b)
            low = torch.tensor([b[0] for b in bounds], dtype=torch.float32).expand(*bs, -1)
            high = torch.tensor([b[1] for b in bounds], dtype=torch.float32).expand(*bs, -1)
            obs_spec: Any = Bounded(low=low, high=high, shape=(*bs, self.obs_dim), dtype=torch.float32)
        else:
            obs_spec = Unbounded(shape=(*bs, self.obs_dim), dtype=torch.float32)
        state_specs = {"lags": Unbounded(shape=(*bs, 6), dtype=ENV_DTYPE)}
        if self.transition.hidden_dim > 0:
            state_specs["hidden"] = Unbounded(shape=(*bs, self.transition.hidden_dim), dtype=ENV_DTYPE)
        self.observation_spec = Composite(observation=obs_spec, macro=Unbounded(shape=(*bs, 3), dtype=ENV_DTYPE),
                                          **{k: v.clone() for k, v in state_specs.items()}, shape=bs)
        self.state_spec = Composite(**state_specs, shape=bs)
        self.action_spec = Bounded(low=-1.0, high=1.0, shape=(*bs, 1), dtype=torch.float32)
        self.reward_spec = Unbounded(shape=(*bs, 1), dtype=torch.float32)
        self.done_spec = Composite(done=Categorical(2, shape=(*bs, 1), dtype=torch.bool),
                                   terminated=Categorical(2, shape=(*bs, 1), dtype=torch.bool), shape=bs)

    def _observation(self, lags: torch.Tensor) -> torch.Tensor:
        cols = [0, 1, 2, 3] if self.observation_lag else [0, 2]
        return lags[..., cols].to(torch.float32)

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs: Any) -> TensorDictBase:
        """A full batch is always returned; TorchRL keeps only the rows flagged in ``"_reset"``."""
        bs = self.batch_size
        if self._init_lags is not None:
            lags = self._init_lags.expand(*bs, -1).clone()
        else:
            idx = torch.randint(2, self._data.shape[0], tuple(bs), generator=self._rng)
            cur, prev = self._data[idx - 1], self._data[idx - 2]
            # Quarter t's rate is the agent's first decision; the lag slots hold the rate of quarter t-1.
            lags = torch.stack([cur[..., 0], prev[..., 0], cur[..., 1], prev[..., 1], prev[..., 2], prev[..., 2]], dim=-1)
        out = TensorDict(
            {"observation": self._observation(lags), "lags": lags,
             "macro": torch.stack([lags[..., 2], lags[..., 0], lags[..., 4]], dim=-1),
             "done": torch.zeros(*bs, 1, dtype=torch.bool), "terminated": torch.zeros(*bs, 1, dtype=torch.bool)},
            batch_size=bs,
        )
        hidden = self.transition.initial_hidden(bs)
        if hidden is not None:
            out.set("hidden", hidden)
        return out

    @torch.no_grad()
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        bs, cfg = self.batch_size, self.config
        rate = normalised_to_rate(tensordict["action"].to(ENV_DTYPE)[..., 0], cfg.action_high)
        lags = tensordict["lags"]
        rate_prev = lags[..., 4]
        # The rate chosen now is next quarter's first lag; the previous rate becomes its second lag.
        state = torch.stack([lags[..., 0], lags[..., 1], lags[..., 2], lags[..., 3], rate, rate_prev], dim=-1)
        eps = torch.randn(*bs, 2, generator=self._rng, dtype=ENV_DTYPE)
        y_new, pi_new, hidden = self.transition(state, eps, tensordict.get("hidden", None))

        pi_new = pi_new.clamp(cfg.pi_star - cfg.macro_clamp_pi, cfg.pi_star + cfg.macro_clamp_pi)
        y_new = y_new.clamp(-cfg.macro_clamp_y, cfg.macro_clamp_y)

        reward = cb_reward(cfg, pi_new, y_new, rate, rate_prev)
        terminated = (is_terminal(cfg, pi_new, y_new) if self.terminate_in_band
                      else torch.zeros_like(y_new, dtype=torch.bool))
        new_lags = torch.stack([y_new, lags[..., 0], pi_new, lags[..., 2], rate, rate_prev], dim=-1)
        out = TensorDict(
            {"observation": self._observation(new_lags), "lags": new_lags,
             "macro": torch.stack([pi_new, y_new, rate], dim=-1),
             "reward": reward.to(torch.float32).unsqueeze(-1),
             "done": terminated.unsqueeze(-1), "terminated": terminated.unsqueeze(-1)},
            batch_size=bs,
        )
        if hidden is not None:
            out.set("hidden", hidden)
        return out

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng.manual_seed(int(seed))


def episodic(env: MonetaryEnv) -> TransformedEnv:
    """Add the episode transforms: start-of-episode flag, truncation at ``t_max``, episode return."""
    return TransformedEnv(env, Compose(InitTracker(), StepCounter(max_steps=env.config.t_max), RewardSum()))

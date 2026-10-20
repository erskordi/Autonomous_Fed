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

"""Economies as stateless, batch-first transition modules."""

from __future__ import annotations

import copy
import math
from typing import Any, Final, Optional

import pandas as pd
import torch
import torch.nn as nn

from ..linear.svar import REGRESSORS_PI, REGRESSORS_Y, STATE_NAMES, RecursiveLinearSpec
from ..nonlinear.economy import NARXEconomy, NSSMEconomy

__all__ = ["ENV_DEVICE", "ENV_DTYPE", "NARXTransition", "NSSMTransition", "RecursiveLinearTransition", "Transition"]

#: The fitted economies are float64 and tiny; they run on the CPU.
ENV_DTYPE: Final = torch.float64
ENV_DEVICE: Final = torch.device("cpu")


class Transition(nn.Module):
    """One-quarter transition of an estimated economy.

    ``forward(state, eps, hidden) -> (y_new, pi_new, hidden_new)`` with ``state`` of shape ``(N, 6)``
    in :data:`autofed.linear.STATE_NAMES` order (``i_lag1`` is the rate just chosen), ``eps`` of shape
    ``(N, 2)`` standard-normal draws, and ``hidden`` of shape ``(N, hidden_dim)`` or ``None``.

    Attributes:
        sigma: ``(2,)`` innovation standard deviations (output gap, inflation).
        hidden_dim: Width of the latent state carried between quarters.
    """

    hidden_dim: int = 0
    sigma: torch.Tensor

    def __init__(self, sigma_y: float, sigma_pi: float) -> None:
        super().__init__()
        for label, value in (("sigma_y", sigma_y), ("sigma_pi", sigma_pi)):
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
        self.register_buffer("sigma", torch.tensor([sigma_y, sigma_pi], dtype=ENV_DTYPE))

    def initial_hidden(self, batch_size: torch.Size, index: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Latent state at reset; ``None`` for economies without one.

        Args:
            batch_size: Batch shape of the reset.
            index: Row of the training data each episode starts from, shape ``batch_size``; economies
                with a latent path start from the latent state of that quarter.
        """
        return None

    def forward(self, state: torch.Tensor, eps: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


def _frozen_copy(module: nn.Module) -> nn.Module:
    """Detached, evaluation-mode copy, so a loaded economy is never mutated."""
    clone = copy.deepcopy(module).to(dtype=ENV_DTYPE, device=ENV_DEVICE).eval()
    clone.requires_grad_(False)
    return clone


class RecursiveLinearTransition(Transition):
    """Recursive linear economy: the output gap first, then inflation with contemporaneous output gap."""

    def __init__(self, spec: RecursiveLinearSpec) -> None:
        super().__init__(spec.sigma_y, spec.sigma_pi)
        self.label = spec.label
        self.register_buffer("b_y", torch.tensor(spec.series("y").reindex(list(REGRESSORS_Y)).to_numpy(), dtype=ENV_DTYPE))
        self.register_buffer("b_pi", torch.tensor(spec.series("pi").reindex(list(REGRESSORS_PI)).to_numpy(), dtype=ENV_DTYPE))
        assert len(STATE_NAMES) == self.b_y.numel() - 1

    def forward(self, state: torch.Tensor, eps: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        y_new = self.b_y[0] + state @ self.b_y[1:] + self.sigma[0] * eps[..., 0]
        pi_new = self.b_pi[0] + self.b_pi[1] * y_new + state @ self.b_pi[2:] + self.sigma[1] * eps[..., 1]
        return y_new, pi_new, hidden


def _scaler_affine(scaler: Any, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``(scale, offset)`` of a fitted scaler, such that ``x_scaled = x * scale + offset``."""
    if not getattr(scaler, "fitted_", False):
        raise RuntimeError(f"scaler {label} is not fitted")
    if bool((scaler.data_range_ < scaler.eps).any()):
        raise ValueError(f"scaler {label} has a constant feature; its affine map is not invertible")
    return (scaler.scale_.detach().to(dtype=ENV_DTYPE, device=ENV_DEVICE).clone(),
            scaler.min_.detach().to(dtype=ENV_DTYPE, device=ENV_DEVICE).clone())


class NARXTransition(Transition):
    """Non-linear economy driven by the two restricted NARX networks.

    Args:
        economy: The fitted networks with their scalers.
        clamp_inputs: Clip each network input to the range it was fitted on (the scaler's feature
            range), so the environment never evaluates the networks outside the training support.
    """

    _Y_INPUTS: Final = [0, 2, 4]            # y_lag1, pi_lag1, i_lag1
    _PI_INPUTS: Final = [0, 1, 2, 3, 4]     # y_lag1, y_lag2, pi_lag1, pi_lag2, i_lag1 (after contemporaneous y)
    xs_y_scale: torch.Tensor
    xs_y_offset: torch.Tensor
    ys_y_scale: torch.Tensor
    ys_y_offset: torch.Tensor
    xs_pi_scale: torch.Tensor
    xs_pi_offset: torch.Tensor
    ys_pi_scale: torch.Tensor
    ys_pi_offset: torch.Tensor
    x_range: torch.Tensor

    def __init__(self, economy: NARXEconomy, *, clamp_inputs: bool = True) -> None:
        super().__init__(economy.sigma_y, economy.sigma_pi)
        if economy.net_y.input_size != 3 or economy.net_pi.input_size != 6:
            raise ValueError(f"expected NARX input sizes (3, 6), got ({economy.net_y.input_size}, {economy.net_pi.input_size})")
        self.net_y = _frozen_copy(economy.net_y)
        self.net_pi = _frozen_copy(economy.net_pi)
        self.clamp_inputs = bool(clamp_inputs)
        for label, scaler in (("xs_y", economy.x_scaler_y), ("ys_y", economy.y_scaler_y),
                              ("xs_pi", economy.x_scaler_pi), ("ys_pi", economy.y_scaler_pi)):
            scale, offset = _scaler_affine(scaler, label)
            self.register_buffer(f"{label}_scale", scale)
            self.register_buffer(f"{label}_offset", offset)
        lo, hi = economy.x_scaler_y.feature_range
        self.register_buffer("x_range", torch.tensor([float(lo), float(hi)], dtype=ENV_DTYPE))

    def _scaled(self, x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        z = x * scale + offset
        return z.clamp(float(self.x_range[0]), float(self.x_range[1])) if self.clamp_inputs else z

    def forward(self, state: torch.Tensor, eps: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        y_scaled = self.net_y(self._scaled(state[..., self._Y_INPUTS], self.xs_y_scale, self.xs_y_offset))
        y_new = ((y_scaled - self.ys_y_offset) / self.ys_y_scale)[..., 0] + self.sigma[0] * eps[..., 0]
        feat_pi = torch.cat([y_new.unsqueeze(-1), state[..., self._PI_INPUTS]], dim=-1)
        pi_scaled = self.net_pi(self._scaled(feat_pi, self.xs_pi_scale, self.xs_pi_offset))
        pi_new = ((pi_scaled - self.ys_pi_offset) / self.ys_pi_scale)[..., 0] + self.sigma[1] * eps[..., 1]
        return y_new, pi_new, hidden


class NSSMTransition(Transition):
    """Non-linear economy driven by the two-block neural state-space model.

    An episode that starts from quarter ``t`` of the training data starts from the latent state the
    model reached in quarter ``t`` when run over the data (``economy.run``); without the data every
    episode starts from the latent state at the end of the estimation window.

    Args:
        economy: The fitted model with its anchored latent states.
        train: Estimation-window data with columns ``y``, ``pi``, ``i``; gives the latent path.
    """

    h0: torch.Tensor
    h_path: torch.Tensor

    def __init__(self, economy: NSSMEconomy, train: Optional[pd.DataFrame] = None) -> None:
        super().__init__(economy.sigma_y, economy.sigma_pi)
        self.model = _frozen_copy(economy.model)
        h_y0 = economy.terminal_y.detach().reshape(-1).to(dtype=ENV_DTYPE, device=ENV_DEVICE)
        h_pi0 = economy.terminal_pi.detach().reshape(-1).to(dtype=ENV_DTYPE, device=ENV_DEVICE)
        if h_y0.numel() != self.model.W_h_y.in_features or h_pi0.numel() != self.model.W_h_pi.in_features:
            raise ValueError("anchored latent states do not match the NSSM state dimensions")
        self._dim_y = int(h_y0.numel())
        self.hidden_dim = int(h_y0.numel() + h_pi0.numel())
        self.register_buffer("h0", torch.cat([h_y0, h_pi0]))
        if train is None:
            self.register_buffer("h_path", self.h0.unsqueeze(0).clone())
        else:
            _, states_y, states_pi = economy.run(train)
            path = torch.cat([states_y, states_pi], dim=-1).detach().to(dtype=ENV_DTYPE, device=ENV_DEVICE)
            # run() returns one state per quarter from the third on; the first two quarters take the first.
            self.register_buffer("h_path", torch.cat([path[:1], path[:1], path], dim=0))

    def initial_hidden(self, batch_size: torch.Size, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if index is None:
            return self.h0.expand(*batch_size, -1).clone()
        return self.h_path[index.clamp(0, self.h_path.shape[0] - 1)].clone()

    def forward(self, state: torch.Tensor, eps: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if hidden is None:
            raise ValueError("NSSMTransition needs the latent state")
        m = self.model
        h_y, h_pi = hidden[..., : self._dim_y], hidden[..., self._dim_y:]
        h_y = m.activation(m.W_h_y(h_y) + m.W_x_y(state))          # STATE_NAMES is the block's input order
        y_new = m.W_o_y(h_y)[..., 0] + self.sigma[0] * eps[..., 0]
        x_pi = torch.cat([y_new.unsqueeze(-1), state], dim=-1)     # realised contemporaneous y first
        h_pi = m.activation(m.W_h_pi(h_pi) + m.W_x_pi(x_pi))
        pi_new = m.W_o_pi(h_pi)[..., 0] + self.sigma[1] * eps[..., 1]
        return y_new, pi_new, torch.cat([h_y, h_pi], dim=-1)


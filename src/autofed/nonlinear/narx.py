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

"""The NARX network and the restricted design matrices of the two transition equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ._tensor import _to_2d_tensor
from .initialization import nguyen_widrow_

__all__ = ["NARXDesignMatrices", "NARXNet", "build_restricted_narx_design_matrices"]


class NARXNet(nn.Module):
    """
    Simple feedforward NARX network.

    Inputs are externally constructed lagged regressors, e.g.
    [y_{t-1}, pi_{t-1}, i_{t-1}] or
    [y_t, y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1}].

    Architecture:
        input -> hidden(tanh) -> output(linear)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, bias: bool = True) -> None:
        """
        Args:
            input_size: Number of lagged/exogenous regressors.
            hidden_size: Number of hidden units.
            output_size: Number of outputs.
            bias: Whether to use bias terms.
        """
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

        self.hidden = nn.Linear(input_size, hidden_size, bias=bias)
        self.output = nn.Linear(hidden_size, output_size, bias=bias)
        self.activation = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nguyen_widrow_(self.hidden, nonlinearity="tanh")
        nn.init.xavier_uniform_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, output_size)
        """
        x = _to_2d_tensor(x)
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {x.shape[1]}."
            )

        hidden = self.activation(self.hidden(x))
        output = self.output(hidden)
        return output


@dataclass
class NARXDesignMatrices:
    """Container for restricted nonlinear-environment design matrices."""
    X_y: torch.Tensor
    y_target: torch.Tensor
    X_pi: torch.Tensor
    pi_target: torch.Tensor
    time_index: list


def build_restricted_narx_design_matrices(df, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> NARXDesignMatrices:
    """
    Build restricted NARX design matrices using the lag-dropped linear specification.

    Output-gap equation:
        y_t <- [y_{t-1}, pi_{t-1}, i_{t-1}]

    Inflation equation:
        pi_t <- [y_t, y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1}]

    Args:
        df: pandas.DataFrame with columns ['y', 'pi', 'i'] and quarterly index.
        dtype: Torch dtype.
        device: Optional torch device.

    Returns:
        NARXDesignMatrices
    """
    required = {"y", "pi", "i"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    work = df.loc[:, ["y", "pi", "i"]].copy().sort_index()

    rows_y = []
    rows_pi = []
    y_targets = []
    pi_targets = []
    time_index = []

    for t in range(2, len(work)):
        y_t = float(work["y"].iloc[t])
        pi_t = float(work["pi"].iloc[t])
        y_lag1 = float(work["y"].iloc[t - 1])
        y_lag2 = float(work["y"].iloc[t - 2])
        pi_lag1 = float(work["pi"].iloc[t - 1])
        pi_lag2 = float(work["pi"].iloc[t - 2])
        i_lag1 = float(work["i"].iloc[t - 1])

        x_y_t = [y_lag1, pi_lag1, i_lag1]
        x_pi_t = [y_t, y_lag1, y_lag2, pi_lag1, pi_lag2, i_lag1]

        rows_y.append(x_y_t)
        rows_pi.append(x_pi_t)
        y_targets.append([y_t])
        pi_targets.append([pi_t])
        time_index.append(work.index[t])

    X_y = torch.tensor(rows_y, dtype=dtype, device=device)
    y_target = torch.tensor(y_targets, dtype=dtype, device=device)
    X_pi = torch.tensor(rows_pi, dtype=dtype, device=device)
    pi_target = torch.tensor(pi_targets, dtype=dtype, device=device)

    return NARXDesignMatrices(
        X_y=X_y,
        y_target=y_target,
        X_pi=X_pi,
        pi_target=pi_target,
        time_index=time_index,
    )

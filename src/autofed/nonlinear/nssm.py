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

"""The two-block neural state-space model, its input sequences and sequence propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

from ._tensor import _to_2d_tensor

__all__ = ["FullNSSMInputs", "TwoBlockNSSM", "build_full_nssm_inputs", "run_two_block_nssm_sequence"]


class TwoBlockNSSM(nn.Module):
    """Two-block neural state-space model: an output-gap block and an inflation block.

    Each block updates its latent state as ``h_t = SiLU(W_h h_{t-1} + W_x x_t)`` and reads the
    observable out linearly, ``x_hat_t = W_o h_t``.

    Args:
        state_dim_y: Latent dimension of the output-gap block.
        state_dim_pi: Latent dimension of the inflation block.
        input_dim_y: Number of output-gap regressors.
        input_dim_pi: Number of inflation regressors.
    """

    def __init__(self, state_dim_y: int, state_dim_pi: int, input_dim_y: int, input_dim_pi: int):
        super().__init__()

        self.W_h_y = nn.Linear(state_dim_y, state_dim_y)
        self.W_x_y = nn.Linear(input_dim_y, state_dim_y)
        self.W_o_y = nn.Linear(state_dim_y, 1)

        self.W_h_pi = nn.Linear(state_dim_pi, state_dim_pi)
        self.W_x_pi = nn.Linear(input_dim_pi, state_dim_pi)
        self.W_o_pi = nn.Linear(state_dim_pi, 1)

        self.activation = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_h_y.weight)
        nn.init.orthogonal_(self.W_h_pi.weight)

        nn.init.xavier_uniform_(self.W_x_y.weight)
        nn.init.xavier_uniform_(self.W_x_pi.weight)
        nn.init.xavier_uniform_(self.W_o_y.weight)
        nn.init.xavier_uniform_(self.W_o_pi.weight)

    def forward(self, h_y: torch.Tensor, h_pi: torch.Tensor, x_y: torch.Tensor, x_pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_y = self.activation(self.W_h_y(h_y) + self.W_x_y(x_y))
        y_t = self.W_o_y(h_y)

        h_pi = self.activation(self.W_h_pi(h_pi) + self.W_x_pi(x_pi))
        pi_t = self.W_o_pi(h_pi)

        return h_y, h_pi, y_t, pi_t


def run_two_block_nssm_sequence(model: TwoBlockNSSM, X_y: torch.Tensor, X_pi: torch.Tensor, h0_y: torch.Tensor, h0_pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Roll the two-block NSSM forward over the full sample.

    Args:
        model: Fitted two-block NSSM model.
        X_y: Output-gap regressor sequence of shape (T, input_dim_y).
        X_pi: Inflation regressor sequence of shape (T, input_dim_pi).
        h0_y: Initial output-gap latent state of shape (1, state_dim_y).
        h0_pi: Initial inflation latent state of shape (1, state_dim_pi).

    Returns:
        Tuple containing:
            - predicted output-gap path of shape (T, 1)
            - predicted inflation path of shape (T, 1)
            - output-gap latent-state path of shape (T, state_dim_y)
            - inflation latent-state path of shape (T, state_dim_pi)
    """
    X_y = _to_2d_tensor(X_y)
    X_pi = _to_2d_tensor(X_pi)
    h_y = _to_2d_tensor(h0_y)
    h_pi = _to_2d_tensor(h0_pi)

    if X_y.shape[0] != X_pi.shape[0]:
        raise ValueError("X_y and X_pi must have the same number of rows.")

    y_preds: List[torch.Tensor] = []
    pi_preds: List[torch.Tensor] = []
    states_y: List[torch.Tensor] = []
    states_pi: List[torch.Tensor] = []

    for t in range(X_y.shape[0]):
        x_y_t = X_y[t:t + 1]
        x_pi_t = X_pi[t:t + 1]

        h_y, h_pi, y_t, pi_t = model(
            h_y,
            h_pi,
            x_y_t,
            x_pi_t,
        )

        y_preds.append(y_t)
        pi_preds.append(pi_t)
        states_y.append(h_y)
        states_pi.append(h_pi)

    return (
        torch.cat(y_preds, dim=0),
        torch.cat(pi_preds, dim=0),
        torch.cat(states_y, dim=0),
        torch.cat(states_pi, dim=0),
    )


@dataclass
class FullNSSMInputs:
    """Container for full-regressor NSSM inputs."""
    X_y: torch.Tensor
    y_target: torch.Tensor
    X_pi: torch.Tensor
    pi_target: torch.Tensor
    time_index: list


def build_full_nssm_inputs(df: pd.DataFrame, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> FullNSSMInputs:
    """
    Build full-information NSSM input sequences.

    Output-gap block:
        s_t^y = [y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1}, i_{t-2}]

    Inflation block:
        s_t^pi = [y_t, y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1}, i_{t-2}]
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

        y_l1 = float(work["y"].iloc[t - 1])
        y_l2 = float(work["y"].iloc[t - 2])

        pi_l1 = float(work["pi"].iloc[t - 1])
        pi_l2 = float(work["pi"].iloc[t - 2])

        i_l1 = float(work["i"].iloc[t - 1])
        i_l2 = float(work["i"].iloc[t - 2])

        x_y_t = [y_l1, y_l2, pi_l1, pi_l2, i_l1, i_l2]
        x_pi_t = [y_t, y_l1, y_l2, pi_l1, pi_l2, i_l1, i_l2]

        rows_y.append(x_y_t)
        rows_pi.append(x_pi_t)
        y_targets.append([y_t])
        pi_targets.append([pi_t])
        time_index.append(work.index[t])

    return FullNSSMInputs(
        X_y=torch.tensor(rows_y, dtype=dtype, device=device),
        y_target=torch.tensor(y_targets, dtype=dtype, device=device),
        X_pi=torch.tensor(rows_pi, dtype=dtype, device=device),
        pi_target=torch.tensor(pi_targets, dtype=dtype, device=device),
        time_index=time_index,
    )

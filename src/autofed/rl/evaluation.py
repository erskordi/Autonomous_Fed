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

"""Evaluation of trained policies: effective Taylor coefficients and reaction-function surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..rules import REFERENCE_RULES
from .policies import TrainedCell

__all__ = ["coefficient_table", "linearise_policy", "policy_surface"]


def _grid_observations(y: torch.Tensor, pi: torch.Tensor, obs_spec: str) -> torch.Tensor:
    """Observations for flattened ``(y, pi)`` grids; under ``x2`` the lag equals the current value."""
    return torch.stack([y, y, pi, pi] if obs_spec == "x2" else [y, pi], dim=-1)


def linearise_policy(cell: TrainedCell, *, y_grid: Optional[torch.Tensor] = None,
                     pi_grid: Optional[torch.Tensor] = None) -> dict[str, float]:
    """OLS approximation of a trained policy in the level form ``i = alpha0 + beta_pi pi + beta_y y``.

    The *clipped* policy is evaluated on a ``(y, pi)`` grid, so the coefficients describe the rate
    the economy actually receives and read directly against H&T (2021, Table 4).

    Args:
        cell: The trained cell.
        y_grid: Output-gap grid; -4 to 4 in 17 points when omitted.
        pi_grid: Inflation grid; 0 to 6 in 13 points when omitted.

    Returns:
        ``alpha0``, ``beta_pi``, ``beta_y`` and the ``R2`` of the approximation.
    """
    y_values = torch.linspace(-4, 4, 17, dtype=torch.float64) if y_grid is None else y_grid.to(torch.float64)
    pi_values = torch.linspace(0, 6, 13, dtype=torch.float64) if pi_grid is None else pi_grid.to(torch.float64)
    Y, PI = torch.meshgrid(y_values, pi_values, indexing="ij")
    y, pi = Y.reshape(-1), PI.reshape(-1)
    rate = cell.rate(_grid_observations(y, pi, cell.obs_spec))
    X = torch.stack([torch.ones_like(pi), pi, y], dim=-1)
    coef = torch.linalg.lstsq(X, rate.unsqueeze(-1)).solution.squeeze(-1)
    resid = rate - X @ coef
    r2 = 1.0 - float(resid.var(correction=0)) / max(float(rate.var(correction=0)), 1e-12)
    return {"alpha0": float(coef[0]), "beta_pi": float(coef[1]), "beta_y": float(coef[2]), "R2": r2}


def policy_surface(cell: TrainedCell, *, y_range: tuple[float, float] = (-5.0, 5.0),
                   pi_range: tuple[float, float] = (0.0, 5.0), points: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The learned reaction function on a ``(pi, y)`` grid.

    Returns:
        ``(PI, Y, RATE)`` arrays of shape ``(points, points)``, the rate in percent.
    """
    y_values = torch.linspace(*y_range, points, dtype=torch.float64)
    pi_values = torch.linspace(*pi_range, points, dtype=torch.float64)
    Y, PI = torch.meshgrid(y_values, pi_values, indexing="ij")
    rate = cell.rate(_grid_observations(Y.reshape(-1), PI.reshape(-1), cell.obs_spec)).reshape(Y.shape)
    return PI.numpy(), Y.numpy(), rate.numpy()


def coefficient_table(cells: Iterable[TrainedCell], *, include_reference: bool = True) -> pd.DataFrame:
    """Effective Taylor coefficients of trained cells, grouped by economy (the layout of H&T Table 4).

    Args:
        cells: Trained cells.
        include_reference: Lead with the three reference rules, whose coefficients are exact.

    Returns:
        A frame indexed by ``(group, policy)`` with columns ``alpha0``, ``beta_pi``, ``beta_y`` and ``R2``.
    """
    rows: dict[tuple[str, str], dict[str, float]] = {}
    if include_reference:
        for name, rule in REFERENCE_RULES.items():
            rows[("Reference rules", name)] = {"alpha0": rule.alpha0, "beta_pi": rule.beta_pi, "beta_y": rule.beta_y, "R2": 1.0}
    for cell in cells:
        rows[(cell.env_name, cell.label)] = linearise_policy(cell)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["Economy", "Policy"])
    return table.rename(columns={"alpha0": r"$\alpha_0$", "beta_pi": r"$\beta_\pi$", "beta_y": r"$\beta_y$", "R2": r"$R^2$"})

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

import math
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..rules import REFERENCE_RULES
from .policies import TrainedCell

__all__ = ["coefficient_table", "linearise_policy", "policy_surface"]

#: Rate at which ``i_{t-1}`` is held when a policy that observes it is evaluated on a macro grid:
#: the long-run nominal rate, ``r* + pi*``.
RATE_LEVEL: float = 4.0


def _grid_observations(y: torch.Tensor, pi: torch.Tensor, cell: TrainedCell, *, y_lag: Optional[torch.Tensor] = None,
                       pi_lag: Optional[torch.Tensor] = None, rate_level: float = RATE_LEVEL) -> torch.Tensor:
    """Observations for flattened grids; under ``x2`` a lag defaults to the current value."""
    cols = [y, pi] if cell.obs_spec == "x1" else [y, y if y_lag is None else y_lag, pi, pi if pi_lag is None else pi_lag]
    if cell.lagged_rate:
        cols.append(torch.full_like(y, rate_level))
    return torch.stack(cols, dim=-1)


def linearise_policy(cell: TrainedCell, *, y_grid: Optional[torch.Tensor] = None, pi_grid: Optional[torch.Tensor] = None,
                     lagged: bool = True, rate_level: float = RATE_LEVEL) -> dict[str, float]:
    """OLS approximation of a trained policy in the level form of H&T (2021, Table 4).

    The *clipped* policy is evaluated on a grid, so the coefficients describe the rate the economy
    actually receives. Under ``x1`` the rule is ``i = alpha0 + beta_pi pi + beta_y y``; under ``x2``
    with ``lagged=True`` the grid spans the lags as well and the rule is
    ``i = alpha0 + beta_pi pi_t + beta_pi_1 pi_{t-1} + beta_y y_t + beta_y_1 y_{t-1}``. A policy that
    observes ``i_{t-1}`` is evaluated with it held at ``rate_level``.

    Args:
        cell: The trained cell.
        y_grid: Output-gap grid; -4 to 4 in 17 points when omitted (9 points per lag dimension).
        pi_grid: Inflation grid; 0 to 6 in 13 points when omitted (9 points per lag dimension).
        lagged: Fit the lag coefficients of an ``x2`` policy instead of setting the lags equal to the
            current values.
        rate_level: Level of ``i_{t-1}`` on the grid, percent.

    Returns:
        ``alpha0``, ``beta_pi``, ``beta_y``, ``beta_pi_1``, ``beta_y_1`` (the lag coefficients are zero
        for ``x1`` policies and when ``lagged`` is false) and the ``R2`` of the approximation.
    """
    with_lags = lagged and cell.obs_spec == "x2"
    if y_grid is None:
        y_grid = torch.linspace(-4, 4, 9 if with_lags else 17, dtype=torch.float64)
    if pi_grid is None:
        pi_grid = torch.linspace(0, 6, 9 if with_lags else 13, dtype=torch.float64)
    y_grid, pi_grid = y_grid.to(torch.float64), pi_grid.to(torch.float64)
    if with_lags:
        Y, YL, PI, PIL = torch.meshgrid(y_grid, y_grid, pi_grid, pi_grid, indexing="ij")
        y, y_lag, pi, pi_lag = (g.reshape(-1) for g in (Y, YL, PI, PIL))
        rate = cell.rate(_grid_observations(y, pi, cell, y_lag=y_lag, pi_lag=pi_lag, rate_level=rate_level))
        X = torch.stack([torch.ones_like(pi), pi, pi_lag, y, y_lag], dim=-1)
    else:
        Y, PI = torch.meshgrid(y_grid, pi_grid, indexing="ij")
        y, pi = Y.reshape(-1), PI.reshape(-1)
        rate = cell.rate(_grid_observations(y, pi, cell, rate_level=rate_level))
        X = torch.stack([torch.ones_like(pi), pi, y], dim=-1)
    coef = torch.linalg.lstsq(X, rate.unsqueeze(-1)).solution.squeeze(-1)
    resid = rate - X @ coef
    r2 = 1.0 - float(resid.var(correction=0)) / max(float(rate.var(correction=0)), 1e-12)
    if with_lags:
        return {"alpha0": float(coef[0]), "beta_pi": float(coef[1]), "beta_pi_1": float(coef[2]),
                "beta_y": float(coef[3]), "beta_y_1": float(coef[4]), "R2": r2}
    return {"alpha0": float(coef[0]), "beta_pi": float(coef[1]), "beta_pi_1": 0.0, "beta_y": float(coef[2]),
            "beta_y_1": 0.0, "R2": r2}


def policy_surface(cell: TrainedCell, *, y_range: tuple[float, float] = (-5.0, 5.0),
                   pi_range: tuple[float, float] = (0.0, 5.0), points: int = 41,
                   rate_level: float = RATE_LEVEL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The learned reaction function on a ``(pi, y)`` grid, lags equal to current values.

    Returns:
        ``(PI, Y, RATE)`` arrays of shape ``(points, points)``, the rate in percent.
    """
    y_values = torch.linspace(*y_range, points, dtype=torch.float64)
    pi_values = torch.linspace(*pi_range, points, dtype=torch.float64)
    Y, PI = torch.meshgrid(y_values, pi_values, indexing="ij")
    rate = cell.rate(_grid_observations(Y.reshape(-1), PI.reshape(-1), cell, rate_level=rate_level)).reshape(Y.shape)
    return PI.numpy(), Y.numpy(), rate.numpy()


def coefficient_table(cells: Iterable[TrainedCell], *, include_reference: bool = True,
                      lagged: bool = True) -> pd.DataFrame:
    """Effective Taylor coefficients of trained cells, grouped by economy (the layout of H&T Table 4).

    Args:
        cells: Trained cells.
        include_reference: Lead with the three reference rules, whose coefficients are exact.
        lagged: Report the lag coefficients of ``x2`` policies (columns are dropped when no cell has any).

    Returns:
        A frame indexed by ``(group, policy)`` with columns ``alpha0``, ``beta_pi``, (``beta_pi_1``,)
        ``beta_y``, (``beta_y_1``,) and ``R2``.
    """
    rows: dict[tuple[str, str], dict[str, float]] = {}
    if include_reference:
        for name, rule in REFERENCE_RULES.items():
            rows[("Reference rules", name)] = {"alpha0": rule.alpha0, "beta_pi": rule.beta_pi, "beta_pi_1": math.nan,
                                               "beta_y": rule.beta_y, "beta_y_1": math.nan, "R2": 1.0}
    any_lags = False
    for cell in cells:
        fit = linearise_policy(cell, lagged=lagged)
        if cell.obs_spec != "x2" or not lagged:
            fit["beta_pi_1"], fit["beta_y_1"] = math.nan, math.nan
        else:
            any_lags = True
        rows[(cell.env_name, cell.label)] = fit
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["Economy", "Policy"])
    columns = ["alpha0", "beta_pi", "beta_pi_1", "beta_y", "beta_y_1", "R2"] if any_lags else ["alpha0", "beta_pi", "beta_y", "R2"]
    return table[columns].rename(columns={"alpha0": r"$\alpha_0$", "beta_pi": r"$\beta_\pi^0$" if any_lags else r"$\beta_\pi$",
                                          "beta_pi_1": r"$\beta_\pi^1$", "beta_y": r"$\beta_y^0$" if any_lags else r"$\beta_y$",
                                          "beta_y_1": r"$\beta_y^1$", "R2": r"$R^2$"})

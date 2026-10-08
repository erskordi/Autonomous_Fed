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

"""Robustness in the Rudebusch-Svensson (1999) model: closed-loop variances of a linear rule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Optional

import pandas as pd
import torch

__all__ = ["RS99_SIGMA_PI", "RS99_SIGMA_Y", "build_rs99_F", "rs99_table", "rs99_variances", "solve_discrete_lyapunov"]

RS99_SIGMA_Y: Final = 0.819
RS99_SIGMA_PI: Final = 1.013
_SHOCK_COV: Final = torch.diag(torch.tensor([RS99_SIGMA_Y**2, 0.0, RS99_SIGMA_PI**2, 0.0, 0.0, 0.0], dtype=torch.float64))


def build_rs99_F(coeffs: Mapping[str, float]) -> torch.Tensor:
    """Closed-loop transition matrix of RS99 under ``i = b_pi0 pi_t + b_pi1 pi_{t-1} + b_y0 y_t + b_y1 y_{t-1}``.

    State ``z = (y_t, y_{t-1}, pi_t, pi_{t-1}, pi_{t-2}, pi_{t-3})``. Coefficients are read from the
    keys ``beta_pi_0``, ``beta_pi_1``, ``beta_y_0`` and ``beta_y_1``; missing keys are zero.
    """
    c_pi0, c_pi1 = coeffs.get("beta_pi_0", 0.0), coeffs.get("beta_pi_1", 0.0)
    c_y0, c_y1 = coeffs.get("beta_y_0", 0.0), coeffs.get("beta_y_1", 0.0)
    # y_{t+1} = 1.1597 y_t - 0.2588 y_{t-1} - 0.0908 (i_t - pi_bar_t) + eps_y,  pi_bar_t = mean(pi_t..pi_{t-3})
    quarter = 0.0908 / 4.0
    return torch.tensor([
        [1.1597 - 0.0908 * c_y0, -0.2588 - 0.0908 * c_y1, -0.0908 * c_pi0 + quarter, -0.0908 * c_pi1 + quarter, quarter, quarter],
        [1.0,    0.0, 0.0,   0.0, 0.0,  0.0],
        [0.1397, 0.0, 0.701, 0.3, -0.1, 0.1],
        [0.0,    0.0, 1.0,   0.0, 0.0,  0.0],
        [0.0,    0.0, 0.0,   1.0, 0.0,  0.0],
        [0.0,    0.0, 0.0,   0.0, 1.0,  0.0],
    ], dtype=torch.float64)


def solve_discrete_lyapunov(F: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Solve ``V = F V F' + Q`` through ``vec(V) = (I - F kron F)^{-1} vec(Q)``.

    Raises:
        ValueError: If ``F`` and ``Q`` are not square matrices of one size.
    """
    n = F.shape[0]
    if F.shape != (n, n) or Q.shape != (n, n):
        raise ValueError(f"F and Q must be square and conformable, got {tuple(F.shape)} and {tuple(Q.shape)}")
    lhs = torch.eye(n * n, dtype=F.dtype) - torch.kron(F, F)
    return torch.linalg.solve(lhs, Q.reshape(-1)).reshape(n, n)


def rs99_variances(coeffs: Mapping[str, float]) -> Optional[dict[str, float]]:
    """Unconditional variances of inflation, the output gap and the rate change under a rule.

    Returns:
        ``{"var_pi", "var_y", "var_di"}``, or ``None`` if the closed loop is unstable.
    """
    F = build_rs99_F(coeffs)
    if float(torch.linalg.eigvals(F).abs().max()) >= 1.0 - 1e-10:
        return None
    V = solve_discrete_lyapunov(F, _SHOCK_COV)
    K = torch.tensor([coeffs.get("beta_y_0", 0.0), coeffs.get("beta_y_1", 0.0), coeffs.get("beta_pi_0", 0.0),
                      coeffs.get("beta_pi_1", 0.0), 0.0, 0.0], dtype=torch.float64)
    var_di = float(K @ (2.0 * V - F @ V - V @ F.T) @ K)             # var(Delta i) = K (2V - FV - VF') K'
    return {"var_pi": float(V[2, 2]), "var_y": float(V[0, 0]), "var_di": max(var_di, 0.0)}


def rs99_table(policy_coeffs: Mapping[str, Mapping[str, float]], *, baseline: str = "TR93") -> pd.DataFrame:
    """Closed-loop variances of every rule as ratios to the baseline rule; ``inf`` marks an unstable loop.

    Args:
        policy_coeffs: Rule name to its coefficients (keys as in :func:`build_rs99_F`).
        baseline: Name of the rule the ratios are taken against.

    Raises:
        KeyError: If the baseline is not among the rules.
        RuntimeError: If the baseline itself is unstable.
    """
    if baseline not in policy_coeffs:
        raise KeyError(f"baseline {baseline!r} is not among the rules")
    results = {label: rs99_variances(coeffs) for label, coeffs in policy_coeffs.items()}
    base = results[baseline]
    if base is None:
        raise RuntimeError(f"the {baseline} benchmark is unstable in RS99; the ratios are undefined")
    rows = {}
    for label, res in results.items():
        if res is None:
            rows[label] = {"var(pi)": math.inf, "var(y)": math.inf, "var(Delta i)": math.inf, "Mean": math.inf}
            continue
        ratios = [res[k] / max(base[k], 1e-12) for k in ("var_pi", "var_y", "var_di")]
        rows[label] = {"var(pi)": ratios[0], "var(y)": ratios[1], "var(Delta i)": ratios[2], "Mean": sum(ratios) / 3.0}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Policy"
    return table

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


"""Robustness in the Rudebusch-Svensson (1999) model: closed-loop variances of a linear rule.

The model, in deviations from steady state (Rudebusch & Svensson 1999, eq. 1-2; standard errors of
the residuals 1.009 and 0.819)::

    pi_{t+1} = 0.70 pi_t - 0.10 pi_{t-1} + 0.28 pi_{t-2} + 0.12 pi_{t-3} + 0.14 y_t + e_pi
    y_{t+1}  = 1.16 y_t - 0.25 y_{t-1} - 0.10 (ibar_t - pibar_t) + e_y

with ``ibar`` and ``pibar`` the four-quarter averages of the policy rate and of inflation. The policy
rule is ``i_t = b_pi0 pi_t + b_pi1 pi_{t-1} + b_y0 y_t + b_y1 y_{t-1}``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Optional

import pandas as pd
import torch

__all__ = ["RS99_SIGMA_PI", "RS99_SIGMA_Y", "RS99_STATE", "build_rs99_F", "rs99_rule_vector", "rs99_table",
           "rs99_variances", "solve_discrete_lyapunov"]

RS99_SIGMA_Y: Final = 0.819
RS99_SIGMA_PI: Final = 1.009
#: Order of the closed-loop state.
RS99_STATE: Final[tuple[str, ...]] = ("y_t", "y_{t-1}", "pi_t", "pi_{t-1}", "pi_{t-2}", "pi_{t-3}", "i_{t-1}", "i_{t-2}", "i_{t-3}")
_RS99_Y: Final = (1.16, -0.25)                   # y_t, y_{t-1}
_RS99_PI: Final = (0.70, -0.10, 0.28, 0.12)      # pi_t .. pi_{t-3}
_RS99_PI_Y: Final = 0.14                         # y_t in the inflation equation
_RS99_R: Final = 0.10                            # real-rate coefficient
_N: Final = len(RS99_STATE)
_SHOCK_COV: Final = torch.diag(torch.tensor([RS99_SIGMA_Y**2, 0.0, RS99_SIGMA_PI**2] + [0.0] * (_N - 3), dtype=torch.float64))


def rs99_rule_vector(coeffs: Mapping[str, float]) -> torch.Tensor:
    """``K`` such that ``i_t = K z_t`` for the state :data:`RS99_STATE`; missing keys are zero."""
    K = torch.zeros(_N, dtype=torch.float64)
    K[0], K[1] = coeffs.get("beta_y_0", 0.0), coeffs.get("beta_y_1", 0.0)
    K[2], K[3] = coeffs.get("beta_pi_0", 0.0), coeffs.get("beta_pi_1", 0.0)
    return K


def build_rs99_F(coeffs: Mapping[str, float]) -> torch.Tensor:
    """Closed-loop transition matrix of RS99 under the rule read from ``coeffs``.

    Coefficients are the keys ``beta_pi_0``, ``beta_pi_1``, ``beta_y_0`` and ``beta_y_1``; missing keys
    are zero. The four-quarter average rate needs ``i_{t-1}``, ``i_{t-2}`` and ``i_{t-3}`` in the state.
    """
    K = rs99_rule_vector(coeffs)
    ibar = K / 4.0                                   # (i_t + i_{t-1} + i_{t-2} + i_{t-3}) / 4
    ibar[6:9] += 0.25
    pibar = torch.zeros(_N, dtype=torch.float64)
    pibar[2:6] = 0.25
    F = torch.zeros(_N, _N, dtype=torch.float64)
    F[0, 0], F[0, 1] = _RS99_Y
    F[0] -= _RS99_R * (ibar - pibar)
    F[1, 0] = 1.0
    F[2, 2:6] = torch.tensor(_RS99_PI, dtype=torch.float64)
    F[2, 0] = _RS99_PI_Y
    F[3, 2] = F[4, 3] = F[5, 4] = 1.0
    F[6] = K                                         # i_{t-1}' = i_t
    F[7, 6] = F[8, 7] = 1.0
    return F


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
        ``{"var_pi", "var_y", "var_di", "rho"}`` with ``rho`` the spectral radius of the closed loop,
        or ``None`` if the closed loop is unstable.
    """
    F = build_rs99_F(coeffs)
    rho = float(torch.linalg.eigvals(F).abs().max())
    if rho >= 1.0 - 1e-10:
        return None
    V = solve_discrete_lyapunov(F, _SHOCK_COV)
    K = rs99_rule_vector(coeffs)
    var_di = float(K @ (2.0 * V - F @ V - V @ F.T) @ K)             # var(Delta i) = K (2V - FV - VF') K'
    return {"var_pi": float(V[2, 2]), "var_y": float(V[0, 0]), "var_di": max(var_di, 0.0), "rho": rho}


def rs99_table(policy_coeffs: Mapping[str, Mapping[str, float]], *, baseline: str = "TR93") -> pd.DataFrame:
    """Closed-loop variances of every rule as ratios to the baseline rule; ``inf`` marks an unstable loop.

    Args:
        policy_coeffs: Rule name to its coefficients (keys as in :func:`build_rs99_F`).
        baseline: Name of the rule the ratios are taken against.

    Returns:
        Columns ``var(pi)``, ``var(y)``, ``var(Delta i)``, ``Mean`` and ``rho``, the spectral radius of
        the closed loop (a rule is stable when ``rho < 1``).

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
        rho = float(torch.linalg.eigvals(build_rs99_F(policy_coeffs[label])).abs().max())
        if res is None:
            rows[label] = {"var(pi)": math.inf, "var(y)": math.inf, "var(Delta i)": math.inf, "Mean": math.inf, "rho": rho}
            continue
        ratios = [res[k] / max(base[k], 1e-12) for k in ("var_pi", "var_y", "var_di")]
        rows[label] = {"var(pi)": ratios[0], "var(y)": ratios[1], "var(Delta i)": ratios[2], "Mean": sum(ratios) / 3.0,
                       "rho": rho}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Policy"
    return table

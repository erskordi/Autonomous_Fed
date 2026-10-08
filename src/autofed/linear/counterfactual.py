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

"""Historical counterfactuals: swap the policy equation, keep the economy and its shocks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..data import NAMES
from ..rules import PolicyRule
from .svar import RecursiveLinearSpec, structural_design

__all__ = ["closed_loop_radius", "simulate", "structural_residuals", "target_loss"]

N_INIT = 2


def structural_residuals(frame: pd.DataFrame, spec: RecursiveLinearSpec) -> pd.DataFrame:
    """Residuals of both equations of ``spec`` on realised data (the historical structural shocks).

    Returns:
        A frame with columns ``y`` and ``pi`` on ``frame``'s index; the first two rows are ``NaN``.
    """
    y_t, X_y, pi_t, X_pi = structural_design(frame)
    out = pd.DataFrame(index=frame.index, columns=["y", "pi"], dtype=float)
    out.loc[y_t.index, "y"] = y_t.to_numpy() - X_y.to_numpy() @ spec.series("y").to_numpy()
    out.loc[pi_t.index, "pi"] = pi_t.to_numpy() - X_pi.to_numpy() @ spec.series("pi").to_numpy()
    return out


def simulate(frame: pd.DataFrame, spec: RecursiveLinearSpec, rule: PolicyRule | None, *,
             residuals: pd.DataFrame | None = None, zlb: bool = False) -> pd.DataFrame:
    """Run the economy forward under ``rule`` with the historical structural shocks.

    The first two quarters are the realised data. From the third on, the output gap and inflation
    follow ``spec`` plus the shocks, and the policy rate follows ``rule``.

    Args:
        frame: Realised data with columns ``y``, ``pi`` and ``i`` on a gap-free index.
        spec: The economy.
        rule: Rule replacing the policy equation; ``None`` keeps the realised rate, which must
            reproduce the realised data exactly.
        residuals: Shocks from :func:`structural_residuals`; computed from ``frame`` when omitted.
        zlb: Floor the rule's prescription at zero.

    Returns:
        A frame with columns ``y``, ``pi`` and ``i``.

    Raises:
        FloatingPointError: If the simulated path is not finite.
    """
    shocks = structural_residuals(frame, spec) if residuals is None else residuals
    cy, cp = spec.coef_y, spec.coef_pi
    y, pi, i = (frame[c].to_numpy(dtype=float).copy() for c in NAMES)
    e_y, e_pi = shocks["y"].to_numpy(dtype=float), shocks["pi"].to_numpy(dtype=float)
    for t in range(N_INIT, len(frame)):
        lags = (cy["y_lag1"] * y[t - 1] + cy["y_lag2"] * y[t - 2] + cy["pi_lag1"] * pi[t - 1]
                + cy["pi_lag2"] * pi[t - 2] + cy["i_lag1"] * i[t - 1] + cy["i_lag2"] * i[t - 2])
        y[t] = cy["const"] + lags + e_y[t]
        lags = (cp["y_lag1"] * y[t - 1] + cp["y_lag2"] * y[t - 2] + cp["pi_lag1"] * pi[t - 1]
                + cp["pi_lag2"] * pi[t - 2] + cp["i_lag1"] * i[t - 1] + cp["i_lag2"] * i[t - 2])
        pi[t] = cp["const"] + cp["y"] * y[t] + lags + e_pi[t]
        if rule is not None:
            i[t] = float(rule.prescribe(float(pi[t]), float(y[t]), zlb=zlb))
    out = pd.DataFrame({"y": y, "pi": pi, "i": i}, index=frame.index)
    if not np.isfinite(out.to_numpy()).all():
        raise FloatingPointError(f"non-finite counterfactual path under {'the realised rate' if rule is None else rule.name}")
    return out


def closed_loop_radius(spec: RecursiveLinearSpec, rule: PolicyRule) -> float:
    """Spectral radius of the economy closed with ``rule`` (lower bound ignored).

    Stacking ``z_t = (y_t, pi_t, i_t)``, the closed system is ``A0 z_t = c + A1 z_{t-1} + A2 z_{t-2}``;
    a radius of one or more means the counterfactual path diverges mechanically.
    """
    cy, cp = spec.coef_y, spec.coef_pi
    a0 = np.array([[1.0, 0.0, 0.0], [-cp["y"], 1.0, 0.0], [-rule.beta_y, -rule.beta_pi, 1.0]])
    lag = lambda k: np.array([[cy[f"y_lag{k}"], cy[f"pi_lag{k}"], cy[f"i_lag{k}"]],          # noqa: E731
                              [cp[f"y_lag{k}"], cp[f"pi_lag{k}"], cp[f"i_lag{k}"]],
                              [0.0, 0.0, 0.0]])
    a0_inv = np.linalg.inv(a0)
    companion = np.block([[a0_inv @ lag(1), a0_inv @ lag(2)], [np.eye(3), np.zeros((3, 3))]])
    return float(np.abs(np.linalg.eigvals(companion)).max())


def target_loss(paths: Mapping[str, pd.DataFrame], *, pi_star: float = 2.0, omega_pi: float = 0.5,
                omega_y: float = 0.5, window: slice = slice(None)) -> pd.DataFrame:
    """Mean squared target deviations and the central-bank loss per policy (paper Table 5 layout).

    Args:
        paths: Simulated (or realised) frames with columns ``pi`` and ``y``, keyed by policy name.
        pi_star: Inflation target.
        omega_pi: Loss weight on inflation deviations.
        omega_y: Loss weight on the output gap.
        window: Label slice of the quarters to evaluate.
    """
    rows = {}
    for name, path in paths.items():
        part = path.loc[window]
        d2_pi = float(((part["pi"] - pi_star) ** 2).mean())
        d2_y = float((part["y"] ** 2).mean())
        rows[name] = {"dev2 pi": d2_pi, "dev2 y": d2_y, "Loss": omega_pi * d2_pi + omega_y * d2_y}
    return pd.DataFrame.from_dict(rows, orient="index")

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

"""Helpers around the ``cultivars`` TVP-VAR-SV posterior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data import NAMES
from ..exceptions import SpecificationError
from .svar import RecursiveLinearSpec

__all__ = ["tvp_conditional_fit", "tvp_structural_spec"]


def _check(result: Any) -> None:
    names = tuple(result.names)
    if names != NAMES:
        raise SpecificationError(f"TVP model must be ordered {NAMES}, got {names}")
    if int(result.order) != 2:
        raise SpecificationError(f"TVP model must have two lags, got order {result.order}")


def tvp_structural_spec(result: Any, t: int, *, label: str = "TVP-SVAR-SV", **meta: Any) -> RecursiveLinearSpec:
    """Posterior-mean structural equations of a TVP-VAR-SV at estimation-sample row ``t``.

    With ``Sigma_t = A^{-1} H_t A^{-T}`` and the ordering ``y, pi, i``, the contemporaneous
    coefficient is ``a = (A^{-1})_{pi,y}``, the structural inflation coefficients are
    ``beta_pi - a * beta_y``, and the shock scales are the posterior means of ``exp(h_t / 2)``.

    Raises:
        SpecificationError: If the model is not a VAR(2) ordered ``y, pi, i``.
        IndexError: If ``t`` is outside the estimation sample.
    """
    _check(result)
    if not 0 <= t < int(result.nobs):
        raise IndexError(f"t must be in 0..{int(result.nobs) - 1}, got {t}")
    labels = {"const": "const"}
    labels.update({f"{v}.L{lag}": f"{v}_lag{lag}" for v in NAMES for lag in (1, 2)})
    reduced = {eq: pd.Series({ours: float(result.coefficient_path(eq, theirs)[t, 1]) for theirs, ours in labels.items()})
               for eq in ("y", "pi")}
    impact = np.asarray(result.impact_draws, dtype=float)
    a_y0 = float(np.mean(impact[:, 1, 0] / impact[:, 0, 0]))
    coef_pi = (reduced["pi"] - a_y0 * reduced["y"]).to_dict()
    coef_pi["y"] = a_y0
    return RecursiveLinearSpec(label, reduced["y"].to_dict(), coef_pi,
                               float(result.volatility_path("y")[t, 1]), float(result.volatility_path("pi")[t, 1]),
                               {"anchor_row": int(t), **meta})


def tvp_conditional_fit(result: Any, frame: pd.DataFrame, *, training: int,
                        band: tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
    """Posterior-mean fitted paths with credible bands; inflation conditional on realised ``y``.

    Args:
        result: The fitted ``TVPVARSVResult``.
        frame: The panel handed to the model, training sample and initial lags included.
        training: Number of leading rows used to calibrate the priors.
        band: Lower and upper quantiles of the pointwise band.

    Returns:
        A frame on the estimation-sample index with ``y``, ``pi``, ``y_hat``, ``pi_hat`` and the band
        columns ``y_hat_lo``, ``y_hat_hi``, ``pi_hat_lo``, ``pi_hat_hi``.

    Raises:
        SpecificationError: If the rebuilt design does not reproduce the model's fitted values.
    """
    _check(result)
    k, order = len(NAMES), int(result.order)
    width, first = 1 + order * k, training + order
    Y = frame[list(NAMES)].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(Y) - first)] + [Y[first - lag: len(Y) - lag] for lag in range(1, order + 1)])
    if X.shape != (int(result.nobs), width):
        raise SpecificationError(f"design has shape {X.shape}; the model expects {(int(result.nobs), width)}")
    beta = np.asarray(result.beta_draws, dtype=float)
    mu = np.stack([np.einsum("tw,stw->st", X, beta[:, :, j * width:(j + 1) * width]) for j in range(k)], axis=-1)
    if not np.allclose(mu.mean(axis=0), result.fittedvalues, atol=1e-8):
        raise SpecificationError("rebuilt design does not reproduce the model's fitted values")
    impact = np.asarray(result.impact_draws, dtype=float)
    a_draws = impact[:, 1, 0] / impact[:, 0, 0]
    y_obs = Y[first:, 0]
    mu_pi = mu[:, :, 1] + a_draws[:, None] * (y_obs[None, :] - mu[:, :, 0])
    lo, hi = band
    return pd.DataFrame(
        {"y": y_obs, "pi": Y[first:, 1], "y_hat": mu[:, :, 0].mean(axis=0), "pi_hat": mu_pi.mean(axis=0),
         "y_hat_lo": np.quantile(mu[:, :, 0], lo, axis=0), "y_hat_hi": np.quantile(mu[:, :, 0], hi, axis=0),
         "pi_hat_lo": np.quantile(mu_pi, lo, axis=0), "pi_hat_hi": np.quantile(mu_pi, hi, axis=0)},
        index=frame.index[first:],
    )

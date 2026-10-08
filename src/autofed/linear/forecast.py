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

"""One-step fits of the recursive SVAR(2) under fixed, rolling and expanding estimation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..data import NAMES

__all__ = ["error_metrics", "expanding_one_step", "fit_unrestricted_svar", "one_step", "one_step_from",
           "rolling_one_step"]

#: Smallest sample on which a three-variable VAR(2) has a non-singular residual covariance.
MIN_WINDOW = 11


def fit_unrestricted_svar(sample: np.ndarray, *, names: Sequence[str] = NAMES) -> tuple[Any, Any]:
    """Fit the unrestricted recursive SVAR(2) on one sample with ``cultivars``.

    Returns:
        ``(VARResult, SVARResult)``.

    Raises:
        ValueError: If the sample is shorter than :data:`MIN_WINDOW` rows.
    """
    from cultivars.multivariate import VAR, RecursiveSVAR

    if sample.shape[0] < MIN_WINDOW:
        raise ValueError(f"a VAR(2) in three variables needs at least {MIN_WINDOW} rows, got {sample.shape[0]}")
    var = VAR(sample, order=2, trend="c", names=list(names)).fit()
    return var, RecursiveSVAR(var, order=list(names)).identify()


def one_step_from(var: Any, svar: Any, X: np.ndarray, t: int) -> tuple[float, float]:
    """One-step ``(y_hat, pi_hat)`` at row ``t`` from a fitted VAR and its recursive identification.

    Uses realised lags only; ``pi_hat`` is conditional on the realised ``y_t`` through the
    contemporaneous structural coefficient implied by the impact matrix.
    """
    constant = np.asarray(var.deterministic)[0]
    pred = constant + sum(var.coefficients[lag] @ X[t - 1 - lag] for lag in range(var.order))
    a_y0 = float(svar.impact[1, 0] / svar.impact[0, 0])
    return float(pred[0]), float(pred[1] + a_y0 * (X[t, 0] - pred[0]))


def _frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=["y_hat", "pi_hat"], dtype=float)


def _first(frame: pd.DataFrame, start: str, minimum: int) -> int:
    period = pd.Period(start, "Q")
    if period not in frame.index:
        raise KeyError(f"{start} is not in the frame ({frame.index[0]} to {frame.index[-1]})")
    return max(int(frame.index.get_loc(period)), minimum)


def one_step(var: Any, svar: Any, frame: pd.DataFrame, start: str) -> pd.DataFrame:
    """One-step fitted ``(y_hat, pi_hat)`` from ``start`` on, with fixed coefficients."""
    X = frame[list(NAMES)].to_numpy(dtype=float)
    out = _frame(frame.index)
    for t in range(_first(frame, start, int(var.order)), len(frame)):
        out.iloc[t, 0], out.iloc[t, 1] = one_step_from(var, svar, X, t)
    return out


def rolling_one_step(frame: pd.DataFrame, window: int, start: str) -> pd.DataFrame:
    """One-step fits from ``start`` on, re-estimating on the trailing ``window`` quarters ``[t-window, t-1]``."""
    X = frame[list(NAMES)].to_numpy(dtype=float)
    out = _frame(frame.index)
    for t in range(_first(frame, start, window), len(frame)):
        var, svar = fit_unrestricted_svar(X[t - window: t])
        out.iloc[t, 0], out.iloc[t, 1] = one_step_from(var, svar, X, t)
    return out


def expanding_one_step(frame: pd.DataFrame, start: str, *, minimum: int = 12) -> pd.DataFrame:
    """One-step fits from ``start`` on, re-estimating on all data through ``t-1``."""
    X = frame[list(NAMES)].to_numpy(dtype=float)
    out = _frame(frame.index)
    for t in range(_first(frame, start, minimum), len(frame)):
        var, svar = fit_unrestricted_svar(X[:t])
        out.iloc[t, 0], out.iloc[t, 1] = one_step_from(var, svar, X, t)
    return out


def error_metrics(frame: pd.DataFrame, y_col: str = "y_hat", pi_col: str = "pi_hat") -> dict[str, float]:
    """MSE and RMSE of the output-gap and inflation fits in ``frame``, with the sample size."""
    out: dict[str, float] = {}
    for column, fit_column in (("y", y_col), ("pi", pi_col)):
        error = (frame[column] - frame[fit_column]).to_numpy(dtype=float)
        out[f"MSE {column}"] = float(np.mean(error**2))
        out[f"RMSE {column}"] = float(np.sqrt(np.mean(error**2)))
    out["n"] = float(len(frame))
    return out

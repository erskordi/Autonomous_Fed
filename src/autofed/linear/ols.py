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

"""Single-equation least squares with the diagnostics the notebooks report."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["EquationFit", "breusch_godfrey_lm", "durbin_watson", "ols"]


@dataclass(frozen=True)
class EquationFit:
    """OLS fit of one structural equation on a named design matrix.

    Attributes:
        name: Dependent variable label (``"y"`` or ``"pi"``).
        params: Coefficient estimates, indexed by regressor name.
        bse: Conventional (non-robust) standard errors.
        tvalues: t statistics.
        pvalues: Two-sided p-values under the t distribution with ``df_resid`` degrees of freedom.
        resid: Residuals over the estimation sample.
        fittedvalues: Fitted values over the estimation sample.
        design: The design matrix as estimated (constant included).
        nobs: Number of observations.
        df_resid: Residual degrees of freedom ``nobs - k``.
        ssr: Sum of squared residuals.
        rsquared: Coefficient of determination.
        rsquared_adj: Adjusted coefficient of determination.
    """

    name: str
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    resid: pd.Series
    fittedvalues: pd.Series
    design: pd.DataFrame
    nobs: int
    df_resid: int
    ssr: float
    rsquared: float
    rsquared_adj: float

    @property
    def mse_resid(self) -> float:
        """Residual variance estimate ``SSR / (n - k)``."""
        return self.ssr / self.df_resid

    @property
    def sigma(self) -> float:
        """Residual standard deviation ``sqrt(SSR / (n - k))``."""
        return float(np.sqrt(self.mse_resid))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Linear prediction on a design matrix that contains (at least) this fit's regressors.

        Raises:
            KeyError: If ``X`` lacks a regressor of the fit.
        """
        missing = [c for c in self.params.index if c not in X.columns]
        if missing:
            raise KeyError(f"design matrix lacks regressors {missing}")
        values = X[list(self.params.index)].to_numpy(dtype=float) @ self.params.to_numpy(dtype=float)
        return pd.Series(values, index=X.index, name=f"{self.name}_hat")

    def summary_frame(self) -> pd.DataFrame:
        """Coefficient table in the ``coef / std err / t / P>|t|`` layout."""
        return pd.DataFrame({"coef": self.params, "std err": self.bse, "t": self.tvalues, "P>|t|": self.pvalues})


def ols(y: pd.Series, X: pd.DataFrame, *, name: str) -> EquationFit:
    """Ordinary least squares with conventional standard errors.

    Args:
        y: Dependent variable.
        X: Design matrix with a ``const`` column already present; must share ``y``'s index.
        name: Label for the dependent variable.

    Returns:
        The fit.

    Raises:
        ValueError: If the indices disagree, values are missing, there are too few observations,
            or the design is rank deficient.
    """
    if not y.index.equals(X.index):
        raise ValueError("y and X must share the same index")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("y and X must not contain missing values")
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    n, k = Xv.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than regressors ({k})")
    if np.linalg.matrix_rank(Xv) < k:
        raise ValueError(f"design matrix is rank deficient: {list(X.columns)}")

    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    fitted = Xv @ beta
    resid = yv - fitted
    ssr = float(resid @ resid)
    df_resid = n - k
    cov = ssr / df_resid * np.linalg.inv(Xv.T @ Xv)
    bse = np.sqrt(np.diag(cov))
    tvalues = beta / bse
    pvalues = 2.0 * stats.t.sf(np.abs(tvalues), df_resid)
    tss = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ssr / tss
    cols = list(X.columns)
    return EquationFit(
        name=name,
        params=pd.Series(beta, index=cols, name="coef"),
        bse=pd.Series(bse, index=cols, name="std err"),
        tvalues=pd.Series(tvalues, index=cols, name="t"),
        pvalues=pd.Series(pvalues, index=cols, name="P>|t|"),
        resid=pd.Series(resid, index=X.index, name=f"eps_{name}"),
        fittedvalues=pd.Series(fitted, index=X.index, name=f"{name}_fit"),
        design=X.copy(),
        nobs=n,
        df_resid=df_resid,
        ssr=ssr,
        rsquared=r2,
        rsquared_adj=1.0 - (1.0 - r2) * (n - 1) / df_resid,
    )


def durbin_watson(resid: pd.Series) -> float:
    """Durbin-Watson statistic ``sum (e_t - e_{t-1})^2 / sum e_t^2``."""
    e = np.asarray(resid, dtype=float)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e**2))


def breusch_godfrey_lm(fit: EquationFit, nlags: int = 1) -> tuple[float, float]:
    """Breusch-Godfrey LM test for residual serial correlation up to ``nlags``.

    Lagged residuals are padded with zeros, the auxiliary regression uses the original regressors
    plus the lagged residuals, and ``LM = n R^2`` is referred to a chi-squared distribution with
    ``nlags`` degrees of freedom.

    Returns:
        ``(statistic, p-value)``.

    Raises:
        ValueError: If ``nlags`` is not positive.
    """
    if nlags < 1:
        raise ValueError(f"nlags must be positive, got {nlags}")
    e = fit.resid.to_numpy(dtype=float)
    lagged = np.column_stack([np.concatenate([np.zeros(lag), e[:-lag]]) for lag in range(1, nlags + 1)])
    Z = np.column_stack([fit.design.to_numpy(dtype=float), lagged])
    gamma, *_ = np.linalg.lstsq(Z, e, rcond=None)
    u = e - Z @ gamma
    r2 = 1.0 - float(u @ u) / float((e - e.mean()) @ (e - e.mean()))
    lm = e.size * r2
    return float(lm), float(stats.chi2.sf(lm, nlags))

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

"""The restricted recursive SVAR(2) of the paper and its portable specification."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from ..data import NAMES
from ..exceptions import ArtifactError, SpecificationError
from .ols import EquationFit, breusch_godfrey_lm, durbin_watson, ols

__all__ = [
    "PARAMETER_TEX", "PAPER_TABLE9", "REGRESSORS_PI", "REGRESSORS_Y", "STATE_NAMES", "RecursiveLinearSpec", "SVARFit",
    "equation_tex", "fit_linear_svar", "identify_insignificant_lag2_terms", "structural_design",
    "structural_equations", "table_nine",
]

#: Lag state shared by both equations, in the order used throughout the package.
STATE_NAMES: Final[tuple[str, ...]] = ("y_lag1", "y_lag2", "pi_lag1", "pi_lag2", "i_lag1", "i_lag2")
REGRESSORS_Y: Final[tuple[str, ...]] = ("const", *STATE_NAMES)
REGRESSORS_PI: Final[tuple[str, ...]] = ("const", "y", *STATE_NAMES)
_CULTIVARS_LABELS: Final[dict[str, str]] = {
    "const": "const", "y": "y",
    "y.L1": "y_lag1", "y.L2": "y_lag2", "pi.L1": "pi_lag1", "pi.L2": "pi_lag2", "i.L1": "i_lag1", "i.L2": "i_lag2",
}
_TERM_TEX: Final[dict[str, str]] = {
    "const": "", "y": "y_t", "y_lag1": "y_{t-1}", "y_lag2": "y_{t-2}", "pi_lag1": r"\pi_{t-1}",
    "pi_lag2": r"\pi_{t-2}", "i_lag1": "i_{t-1}", "i_lag2": "i_{t-2}",
}
#: Mathtext symbol of each coefficient, per equation, in the paper's notation.
PARAMETER_TEX: Final[dict[str, dict[str, str]]] = {
    "y": {"const": r"$C^y$", "y_lag1": r"$a^y_{y,1}$", "y_lag2": r"$a^y_{y,2}$", "pi_lag1": r"$a^y_{\pi,1}$",
          "pi_lag2": r"$a^y_{\pi,2}$", "i_lag1": r"$a^y_{i,1}$", "i_lag2": r"$a^y_{i,2}$"},
    "pi": {"const": r"$C^\pi$", "y": r"$a^\pi_{y,0}$", "y_lag1": r"$a^\pi_{y,1}$", "y_lag2": r"$a^\pi_{y,2}$",
           "pi_lag1": r"$a^\pi_{\pi,1}$", "pi_lag2": r"$a^\pi_{\pi,2}$", "i_lag1": r"$a^\pi_{i,1}$",
           "i_lag2": r"$a^\pi_{i,2}$"},
}


def structural_design(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Targets and design matrices of the two structural equations on a common index.

    The first two rows are dropped so both equations share one effective sample.

    Args:
        frame: Data with columns ``y``, ``pi`` and ``i``.

    Returns:
        ``(y_target, X_y, pi_target, X_pi)``; both design matrices carry a leading ``const`` column
        and ``X_pi`` the contemporaneous ``y``.

    Raises:
        KeyError: If a column is missing.
    """
    missing = [c for c in NAMES if c not in frame.columns]
    if missing:
        raise KeyError(f"frame lacks columns {missing}")
    work = frame[list(NAMES)].copy().sort_index()
    for column in NAMES:
        work[f"{column}_lag1"] = work[column].shift(1)
        work[f"{column}_lag2"] = work[column].shift(2)
    work = work.dropna()
    X_y = work[list(STATE_NAMES)].copy()
    X_y.insert(0, "const", 1.0)
    X_pi = work[["y", *STATE_NAMES]].copy()
    X_pi.insert(0, "const", 1.0)
    return work["y"], X_y, work["pi"], X_pi


def identify_insignificant_lag2_terms(fit: EquationFit, *, alpha: float = 0.10, suffix: str = "_lag2") -> list[str]:
    """Lag-2 regressors of ``fit`` whose p-value exceeds ``alpha``."""
    return [k for k in fit.params.index if k.endswith(suffix) and float(fit.pvalues[k]) > alpha]


@dataclass(frozen=True)
class RecursiveLinearSpec:
    """A recursive linear economy: the output gap first, inflation with contemporaneous output gap.

    ``y_t = b_y' z_t + sigma_y eps_y`` and ``pi_t = a y_t + b_pi' z_t + sigma_pi eps_pi``, with
    ``z_t`` the constant and two lags of ``y``, ``pi`` and ``i``. The fixed SVAR, the paper's
    estimates and a TVP model frozen at one date are all instances, and it is the object passed
    between notebooks.

    Attributes:
        label: Name of the economy.
        coef_y: Output-gap coefficients keyed by :data:`REGRESSORS_Y`; omitted terms are zero.
        coef_pi: Inflation coefficients keyed by :data:`REGRESSORS_PI`; omitted terms are zero.
        sigma_y: Standard deviation of the output-gap shock.
        sigma_pi: Standard deviation of the inflation shock.
        meta: Free-form provenance (sample, restriction, vintage).

    Raises:
        SpecificationError: If a coefficient name is unknown, or a value is not finite.
    """

    label: str
    coef_y: Mapping[str, float]
    coef_pi: Mapping[str, float]
    sigma_y: float
    sigma_pi: float
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr, names in (("coef_y", REGRESSORS_Y), ("coef_pi", REGRESSORS_PI)):
            given = dict(getattr(self, attr))
            unknown = sorted(set(given) - set(names))
            if unknown:
                raise SpecificationError(f"{attr} has unknown regressors {unknown}; expected a subset of {list(names)}")
            full = {name: float(given.get(name, 0.0)) for name in names}
            if not all(math.isfinite(v) for v in full.values()):
                raise SpecificationError(f"{attr} has non-finite coefficients")
            object.__setattr__(self, attr, full)
        for attr in ("sigma_y", "sigma_pi"):
            value = float(getattr(self, attr))
            if not (math.isfinite(value) and value >= 0.0):
                raise SpecificationError(f"{attr} must be finite and non-negative, got {value!r}")
            object.__setattr__(self, attr, value)

    def series(self, equation: str) -> pd.Series:
        """Coefficients of ``"y"`` or ``"pi"`` as a Series in canonical order."""
        if equation not in ("y", "pi"):
            raise KeyError(f"equation must be 'y' or 'pi', got {equation!r}")
        return pd.Series(self.coef_y if equation == "y" else self.coef_pi, dtype=float, name=equation)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One-step fitted ``y_hat`` and ``pi_hat`` on realised data (inflation conditional on realised ``y``)."""
        _, X_y, _, X_pi = structural_design(frame)
        return pd.DataFrame({"y_hat": X_y.to_numpy() @ self.series("y").to_numpy(),
                             "pi_hat": X_pi.to_numpy() @ self.series("pi").to_numpy()}, index=X_y.index)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {"label": self.label, "coef_y": dict(self.coef_y), "coef_pi": dict(self.coef_pi),
                "sigma_y": self.sigma_y, "sigma_pi": self.sigma_pi, "meta": dict(self.meta)}

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the specification to a JSON file and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> RecursiveLinearSpec:
        """Read a specification written by :meth:`save`.

        Raises:
            ArtifactError: If the file is missing or malformed.
        """
        source = Path(path)
        if not source.is_file():
            raise ArtifactError(f"{source} not found; run the notebook that estimates this economy first")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            return cls(raw["label"], raw["coef_y"], raw["coef_pi"], raw["sigma_y"], raw["sigma_pi"], raw.get("meta", {}))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"{source} is not a valid economy specification: {exc}") from exc


#: Restricted SVAR estimates published in Hinterlang & Tänzer (2021, Table 9).
PAPER_TABLE9: Final = RecursiveLinearSpec(
    label="H&T (2021) Table 9",
    coef_y={"const": 0.3834, "y_lag1": 0.9084, "pi_lag1": -0.1437, "i_lag1": 0.2726, "i_lag2": -0.2896},
    coef_pi={"const": 0.1035, "y": -0.0655, "y_lag1": 0.1970, "y_lag2": -0.1121, "pi_lag1": 1.2970,
             "pi_lag2": -0.3116, "i_lag1": -0.0122},
    sigma_y=math.sqrt(0.2136),
    sigma_pi=math.sqrt(0.0330),
    meta={"sample": "1987Q3-2007Q2", "source": "Deutsche Bundesbank Discussion Paper 51/2021, Table 9"},
)


@dataclass(frozen=True)
class SVARFit:
    """Full and restricted equation-by-equation fits of the recursive SVAR(2).

    Attributes:
        full_y: Unrestricted output-gap equation.
        full_pi: Unrestricted (structural) inflation equation.
        mod_y: Restricted output-gap refit.
        mod_pi: Restricted inflation refit.
        zeroed_y: Regressors dropped from the output-gap equation.
        zeroed_pi: Regressors dropped from the inflation equation.
    """

    full_y: EquationFit
    full_pi: EquationFit
    mod_y: EquationFit
    mod_pi: EquationFit
    zeroed_y: tuple[str, ...]
    zeroed_pi: tuple[str, ...]

    def _pair(self, equation: str) -> tuple[EquationFit, EquationFit]:
        if equation not in ("y", "pi"):
            raise KeyError(f"equation must be 'y' or 'pi', got {equation!r}")
        return (self.full_y, self.mod_y) if equation == "y" else (self.full_pi, self.mod_pi)

    def coef(self, equation: str) -> pd.Series:
        """Restricted coefficients on the full regressor set: dropped terms appear as ``0.0``."""
        full, mod = self._pair(equation)
        return mod.params.reindex(full.params.index, fill_value=0.0)

    def pvalues(self, equation: str) -> pd.Series:
        """Restricted p-values; dropped terms carry the full-model p-value that justified the drop."""
        full, mod = self._pair(equation)
        return mod.pvalues.reindex(full.params.index).fillna(full.pvalues)

    def to_spec(self, label: str = "Restricted SVAR(2)", **meta: Any) -> RecursiveLinearSpec:
        """Portable specification of the restricted system, with ``sqrt(SSR / (n - k))`` shock scales."""
        return RecursiveLinearSpec(label, self.coef("y").to_dict(), self.coef("pi").to_dict(),
                                   self.mod_y.sigma, self.mod_pi.sigma,
                                   {"zeroed_y": list(self.zeroed_y), "zeroed_pi": list(self.zeroed_pi), **meta})


def fit_linear_svar(frame: pd.DataFrame, *, alpha_drop: float = 0.10) -> SVARFit:
    """Estimate the paper's restricted recursive SVAR(2).

    1. Fit the full SVAR(2) equation by equation (OLS).
    2. Drop second-lag terms whose full-model p-value exceeds ``alpha_drop``.
    3. Refit the restricted equations without those regressors.

    Args:
        frame: Estimation sample with columns ``y``, ``pi`` and ``i``.
        alpha_drop: Significance threshold of the restriction.

    Raises:
        ValueError: If ``alpha_drop`` is outside ``(0, 1)``.
    """
    if not 0.0 < alpha_drop < 1.0:
        raise ValueError(f"alpha_drop must be in (0, 1), got {alpha_drop}")
    y_t, X_y, pi_t, X_pi = structural_design(frame)
    full_y = ols(y_t, X_y, name="y")
    drop_y = identify_insignificant_lag2_terms(full_y, alpha=alpha_drop)
    full_pi = ols(pi_t, X_pi, name="pi")
    drop_pi = identify_insignificant_lag2_terms(full_pi, alpha=alpha_drop)
    return SVARFit(full_y=full_y, full_pi=full_pi, mod_y=ols(y_t, X_y.drop(columns=drop_y), name="y"),
                   mod_pi=ols(pi_t, X_pi.drop(columns=drop_pi), name="pi"),
                   zeroed_y=tuple(drop_y), zeroed_pi=tuple(drop_pi))


def structural_equations(var: Any, svar: Any) -> dict[str, dict[str, float]]:
    """Structural-form coefficients from a fitted ``cultivars`` VAR and its recursive identification.

    Args:
        var: Fitted reduced-form ``VAR`` ordered ``y, pi, i``.
        svar: Its identified ``RecursiveSVAR``.

    Returns:
        ``{"y": {...}, "pi": {...}}`` keyed by the package's regressor labels.
    """
    impact = np.asarray(svar.impact, dtype=float)
    red_y, red_pi = var.equation("y"), var.equation("pi")
    a_y0 = float(impact[1, 0] / impact[0, 0])
    struct_pi = {"const": red_pi["const"] - a_y0 * red_y["const"], "y": a_y0}
    struct_pi.update({k: red_pi[k] - a_y0 * red_y[k] for k in red_pi if k != "const"})
    return {"y": {_CULTIVARS_LABELS[k]: float(v) for k, v in red_y.items()},
            "pi": {_CULTIVARS_LABELS[k]: float(v) for k, v in struct_pi.items()}}


def equation_tex(lhs: str, coef: pd.Series | Mapping[str, float]) -> str:
    """Display-math string of an estimated equation; zero (dropped) terms are omitted."""
    values = dict(coef)
    body = f"{values.get('const', 0.0):.4f}"
    for name, value in values.items():
        if name == "const" or value == 0.0:
            continue
        body += f" {'-' if value < 0 else '+'} {abs(value):.4f}{_TERM_TEX[name]}"
    return f"$${lhs} = {body}$$"


def table_nine(fit: SVARFit, reference: RecursiveLinearSpec | None = PAPER_TABLE9) -> pd.DataFrame:
    """Restricted estimates, p-values and residual diagnostics in the layout of the paper's Table 9.

    Args:
        fit: The estimated system.
        reference: Published coefficients shown alongside; ``None`` omits the column.

    Returns:
        A frame indexed by ``(equation, parameter)`` with columns ``Estimate``, ``p-value`` and,
        if requested, ``H&T (2021)``.
    """
    blocks: Sequence[tuple[str, str, str, EquationFit]] = (
        ("y", "Output gap equation", r"$\hat\sigma^2_{\varepsilon_1}$", fit.mod_y),
        ("pi", "Inflation equation", r"$\hat\sigma^2_{\varepsilon_2}$", fit.mod_pi),
    )
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for equation, title, sigma_tex, model in blocks:
        published = reference.series(equation) if reference is not None else None
        for name in fit.coef(equation).index:
            kept = name in model.params.index
            in_paper = published is not None and published[name] != 0.0
            if not (kept or in_paper):
                continue
            rows[(title, PARAMETER_TEX[equation][name])] = {
                "Estimate": float(model.params[name]) if kept else float("nan"),
                "p-value": float(model.pvalues[name]) if kept else float("nan"),
                "H&T (2021)": float(published[name]) if in_paper and published is not None else float("nan"),
            }
        lm, lm_p = breusch_godfrey_lm(model, 1)
        nan = float("nan")
        rows[(title, r"$\bar{R}^2$")] = {"Estimate": model.rsquared_adj, "p-value": nan, "H&T (2021)": nan}
        rows[(title, "MSE")] = {"Estimate": model.ssr / model.nobs, "p-value": nan, "H&T (2021)": nan}
        rows[(title, sigma_tex)] = {"Estimate": model.mse_resid, "p-value": nan, "H&T (2021)": nan}
        rows[(title, "DW")] = {"Estimate": durbin_watson(model.resid), "p-value": nan, "H&T (2021)": nan}
        rows[(title, "LM(1)")] = {"Estimate": lm, "p-value": lm_p, "H&T (2021)": nan}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["Equation", "Parameter"])
    return table if reference is not None else table.drop(columns="H&T (2021)")

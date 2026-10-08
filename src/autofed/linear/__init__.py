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

"""Linear economies: the restricted recursive SVAR, re-estimation schemes, TVP helpers, counterfactuals."""

from __future__ import annotations

from .counterfactual import closed_loop_radius, simulate, structural_residuals, target_loss
from .forecast import (
    error_metrics,
    expanding_one_step,
    fit_unrestricted_svar,
    one_step,
    one_step_from,
    rolling_one_step,
)
from .ols import EquationFit, breusch_godfrey_lm, durbin_watson, ols
from .svar import (
    PAPER_TABLE9,
    PARAMETER_TEX,
    REGRESSORS_PI,
    REGRESSORS_Y,
    STATE_NAMES,
    RecursiveLinearSpec,
    SVARFit,
    equation_tex,
    fit_linear_svar,
    identify_insignificant_lag2_terms,
    structural_design,
    structural_equations,
    table_nine,
)
from .tvp import tvp_conditional_fit, tvp_structural_spec

__all__ = [
    "PAPER_TABLE9", "PARAMETER_TEX", "REGRESSORS_PI", "REGRESSORS_Y", "STATE_NAMES", "EquationFit", "RecursiveLinearSpec", "SVARFit",
    "breusch_godfrey_lm", "closed_loop_radius", "durbin_watson", "equation_tex", "error_metrics",
    "expanding_one_step", "fit_linear_svar", "fit_unrestricted_svar", "identify_insignificant_lag2_terms", "ols",
    "one_step", "one_step_from", "rolling_one_step", "simulate", "structural_design", "structural_equations",
    "structural_residuals", "table_nine", "target_loss", "tvp_conditional_fit", "tvp_structural_spec",
]

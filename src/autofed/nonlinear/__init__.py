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

"""Non-linear economies: the restricted NARX networks and the two-block neural state-space model.

Needs PyTorch (``pip install autofed[nonlinear]``); the rest of the package imports without it.
"""

from __future__ import annotations

from .economy import NARXEconomy, NSSMEconomy, NSSMTrainingResult, train_nssm
from .initialization import nguyen_widrow_
from .lm import (
    LevenbergMarquardtTrainer,
    LMTrainingResult,
    LMValidationTrainingResult,
    TimeSeriesSplit,
    ValidationLevenbergMarquardtTrainer,
    make_last_block_validation_split,
)
from .narx import NARXDesignMatrices, NARXNet, build_restricted_narx_design_matrices
from .nssm import FullNSSMInputs, TwoBlockNSSM, build_full_nssm_inputs, run_two_block_nssm_sequence
from .scaling import MapMinMaxScaler
from .search import HiddenSearchSummary, HiddenSearchTrialResult, run_hidden_unit_search

__all__ = [
    "FullNSSMInputs", "HiddenSearchSummary", "HiddenSearchTrialResult", "LMTrainingResult",
    "LMValidationTrainingResult", "LevenbergMarquardtTrainer", "MapMinMaxScaler", "NARXDesignMatrices",
    "NARXEconomy", "NARXNet", "NSSMEconomy", "NSSMTrainingResult", "TimeSeriesSplit", "TwoBlockNSSM",
    "ValidationLevenbergMarquardtTrainer", "build_full_nssm_inputs", "build_restricted_narx_design_matrices",
    "make_last_block_validation_split", "nguyen_widrow_", "run_hidden_unit_search",
    "run_two_block_nssm_sequence", "train_nssm",
]

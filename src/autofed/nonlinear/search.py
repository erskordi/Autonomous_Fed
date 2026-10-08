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

"""Hidden-unit selection by repeated validation-aware Levenberg-Marquardt fits."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .lm import ValidationLevenbergMarquardtTrainer, make_last_block_validation_split
from .narx import NARXNet
from .scaling import MapMinMaxScaler

__all__ = ["HiddenSearchSummary", "HiddenSearchTrialResult", "run_hidden_unit_search"]


@dataclass
class HiddenSearchTrialResult:
    """Single trial result for one hidden size."""
    hidden_size: int
    trial_seed: int
    best_epoch: int
    train_mse: float
    val_mse: float
    overall_mse: float
    epochs_completed: int
    model_state_dict: Dict[str, torch.Tensor]
    x_scaler: MapMinMaxScaler
    y_scaler: MapMinMaxScaler

@dataclass
class HiddenSearchSummary:
    """Summary across all hidden sizes and trials."""
    equation_name: str
    selected_hidden_size: int
    mean_val_mse_by_hidden: Dict[int, float]
    best_trial_by_hidden: Dict[int, HiddenSearchTrialResult]
    all_trials: List[HiddenSearchTrialResult]


def _emit(enabled: bool, message: str) -> None:
    """Print a progress line when requested."""
    if enabled:
        print(message)


def _compute_mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Compute MSE for a fitted model."""
    with torch.no_grad():
        residuals = model(X) - y
        return float(torch.mean(residuals ** 2).item())


def _fit_one_hidden_search_trial(*, X: torch.Tensor, y: torch.Tensor, time_index: list, hidden_size: int, trial_seed: int, patience: int = 6,
                                 max_epochs: int = 200, verbose: bool = False, val_fraction: float = 0.15,
                                 dtype: torch.dtype = torch.float64,
                                 device: Optional[torch.device] = None) -> HiddenSearchTrialResult:
    """
    Run one hidden-search trial for a single hidden size.
    """
    torch.manual_seed(trial_seed)

    split = make_last_block_validation_split(
        X=X,
        y=y,
        time_index=time_index,
        val_fraction=val_fraction,
    )

    x_scaler = MapMinMaxScaler(feature_range=(-1.0, 1.0))
    y_scaler = MapMinMaxScaler(feature_range=(-1.0, 1.0))

    X_train_scaled = x_scaler.fit_transform(split.X_train)
    y_train_scaled = y_scaler.fit_transform(split.y_train)

    X_val_scaled = x_scaler.transform(split.X_val)
    y_val_scaled = y_scaler.transform(split.y_val)

    model = NARXNet(
        input_size=X.shape[1],
        hidden_size=hidden_size,
        output_size=1,
    ).to(device=device, dtype=dtype)

    trainer = ValidationLevenbergMarquardtTrainer(
        mu_init=1e-2,
        mu_increase=10.0,
        mu_decrease=0.1,
        mu_max=1e12,
        max_epochs=max_epochs,
        tol_grad=1e-8,
        tol_step=1e-8,
        tol_loss=1e-12,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )

    fit_result = trainer.fit_with_validation(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        patience=patience,
        min_delta=0.0,
    )

    model.load_state_dict(fit_result.best_state_dict)

    train_mse = _compute_mse(model, X_train_scaled, y_train_scaled)
    val_mse = _compute_mse(model, X_val_scaled, y_val_scaled)

    X_all_scaled = x_scaler.transform(X)
    y_all_scaled = y_scaler.transform(y)
    overall_mse = _compute_mse(model, X_all_scaled, y_all_scaled)

    return HiddenSearchTrialResult(
        hidden_size=hidden_size,
        trial_seed=trial_seed,
        best_epoch=fit_result.best_epoch,
        train_mse=train_mse,
        val_mse=val_mse,
        overall_mse=overall_mse,
        epochs_completed=fit_result.epochs_completed,
        model_state_dict=copy.deepcopy(fit_result.best_state_dict),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def run_hidden_unit_search(*, equation_name: str, X: torch.Tensor, y: torch.Tensor, time_index: list,
                           hidden_min: int = 1, hidden_max: int = 10, n_trials: int = 30, base_seed: int = 123,
                           patience: int = 6, max_epochs: int = 200, verbose: bool = False,
                           val_fraction: float = 0.15, progress: bool = False,
                           dtype: torch.dtype = torch.float64,
                           device: Optional[torch.device] = None) -> HiddenSearchSummary:
    """
    Run the Bundesbank-style hidden search for one equation.

    Selection rule:
        choose hidden size with lowest mean validation MSE across 30 trials.

    After choosing hidden size:
        retain the trial with the lowest overall MSE among that hidden size's trials.
    """
    if hidden_min <= 0 or hidden_max < hidden_min:
        raise ValueError("Invalid hidden-unit range.")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    all_trials: List[HiddenSearchTrialResult] = []
    mean_val_mse_by_hidden: Dict[int, float] = {}
    best_trial_by_hidden: Dict[int, HiddenSearchTrialResult] = {}

    for hidden_size in range(hidden_min, hidden_max + 1):
        trials_this_hidden: List[HiddenSearchTrialResult] = []

        for trial_idx in range(n_trials):
            trial_seed = base_seed + 10_000 * hidden_size + trial_idx

            trial_result = _fit_one_hidden_search_trial(
                X=X,
                y=y,
                time_index=time_index,
                hidden_size=hidden_size,
                trial_seed=trial_seed,
                patience=patience,
                max_epochs=max_epochs,
                verbose=verbose,
                val_fraction=val_fraction,
                dtype=dtype,
                device=device,
            )

            trials_this_hidden.append(trial_result)
            all_trials.append(trial_result)

        mean_val_mse = sum(t.val_mse for t in trials_this_hidden) / len(trials_this_hidden)
        mean_val_mse_by_hidden[hidden_size] = mean_val_mse

        best_trial = min(trials_this_hidden, key=lambda t: t.overall_mse)
        best_trial_by_hidden[hidden_size] = best_trial

        _emit(progress, 
            f"[{equation_name}] hidden={hidden_size:2d} | "
            f"mean validation MSE={mean_val_mse:.8f} | "
            f"best overall MSE={best_trial.overall_mse:.8f}"
        )

    selected_hidden_size = min(mean_val_mse_by_hidden, key=mean_val_mse_by_hidden.get)

    _emit(progress, 
        f"\n[{equation_name}] selected hidden size = {selected_hidden_size} "
        f"(lowest mean validation MSE across {n_trials} trials)"
    )

    return HiddenSearchSummary(
        equation_name=equation_name,
        selected_hidden_size=selected_hidden_size,
        mean_val_mse_by_hidden=mean_val_mse_by_hidden,
        best_trial_by_hidden=best_trial_by_hidden,
        all_trials=all_trials,
    )

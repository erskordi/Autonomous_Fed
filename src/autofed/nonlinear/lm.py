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

"""Levenberg-Marquardt training, with a chronological validation split and early stopping."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ._tensor import (
    _assign_flat_parameters,
    _flatten_parameters,
    _num_parameters,
    _to_2d_tensor,
    _validate_finite_tensor,
)

__all__ = ["LMTrainingResult", "LMValidationTrainingResult", "LevenbergMarquardtTrainer", "TimeSeriesSplit", "ValidationLevenbergMarquardtTrainer", "make_last_block_validation_split"]


@dataclass
class LMTrainingResult:
    """Container for LM training results."""
    best_loss: float
    final_loss: float
    epochs_completed: int
    mu_final: float
    loss_history: List[float]
    mu_history: List[float]
    best_state_dict: Dict[str, torch.Tensor]


class LevenbergMarquardtTrainer:
    """
    Jacobian-based Levenberg-Marquardt trainer for small PyTorch models.

    The objective is sum-of-squares / mean-squared-error style training based on
    residual vector e = y_pred - y_true.

    For n residuals and p parameters:
        J: shape (n, p)
        e: shape (n, 1)

    Update:
        delta = -(J^T J + mu I)^(-1) J^T e
    """

    def __init__(self, mu_init: float = 1e-2, mu_increase: float = 10.0, mu_decrease: float = 0.1,
                 mu_max: float = 1e12, max_epochs: int = 200, tol_grad: float = 1e-8, tol_step: float = 1e-8,
                 tol_loss: float = 1e-12, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64, verbose: bool = False) -> None:
        if mu_init <= 0:
            raise ValueError("mu_init must be positive.")
        if mu_increase <= 1:
            raise ValueError("mu_increase must be > 1.")
        if not (0 < mu_decrease < 1):
            raise ValueError("mu_decrease must satisfy 0 < mu_decrease < 1.")
        if mu_max <= mu_init:
            raise ValueError("mu_max must be greater than mu_init.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")

        self.mu_init = float(mu_init)
        self.mu_increase = float(mu_increase)
        self.mu_decrease = float(mu_decrease)
        self.mu_max = float(mu_max)
        self.max_epochs = int(max_epochs)
        self.tol_grad = float(tol_grad)
        self.tol_step = float(tol_step)
        self.tol_loss = float(tol_loss)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.verbose = verbose

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Move model to desired device/dtype."""
        return model.to(device=self.device, dtype=self.dtype)

    def _prepare_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move data to desired device/dtype and validate shapes."""
        x = _to_2d_tensor(x).to(device=self.device, dtype=self.dtype)
        y = _to_2d_tensor(y).to(device=self.device, dtype=self.dtype)

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have the same number of rows. Got {x.shape[0]} and {y.shape[0]}."
            )

        _validate_finite_tensor(x, "x")
        _validate_finite_tensor(y, "y")

        return x, y

    def _residual_vector(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Residual vector e = y_pred - y, flattened to shape (n_residuals,).
        """
        y_pred = model(x)
        residuals = (y_pred - y).reshape(-1)
        return residuals

    def _jacobian(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Jacobian of residual vector with respect to model parameters.

        Returns:
            J: shape (n_residuals, n_params)
            e: shape (n_residuals,)
        """
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("Model has no trainable parameters.")

        e = self._residual_vector(model, x, y)
        n_residuals = e.numel()
        n_params = _num_parameters(parameters)

        J = torch.zeros(
            (n_residuals, n_params),
            device=self.device,
            dtype=self.dtype
        )

        for i in range(n_residuals):
            model.zero_grad(set_to_none=True)
            e[i].backward(retain_graph=True)

            grad_parts: List[torch.Tensor] = []
            for parameter in parameters:
                if parameter.grad is None:
                    grad_parts.append(torch.zeros_like(parameter).reshape(-1))
                else:
                    grad_parts.append(parameter.grad.reshape(-1))

            J[i, :] = torch.cat(grad_parts)

        model.zero_grad(set_to_none=True)
        return J, e.detach()

    def _loss_from_residuals(self, e: torch.Tensor) -> torch.Tensor:
        """
        Mean squared error from residual vector.
        """
        return torch.mean(e ** 2)

    def fit(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> LMTrainingResult:
        """
        Fit a model using LM.
        """
        model = self._prepare_model(model)
        x, y = self._prepare_data(x, y)

        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("Model has no trainable parameters.")

        mu = self.mu_init
        loss_history: List[float] = []
        mu_history: List[float] = []

        best_state_dict = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        previous_loss: Optional[float] = None

        for epoch in range(1, self.max_epochs + 1):
            J, e = self._jacobian(model, x, y)

            loss = self._loss_from_residuals(e)
            loss_value = float(loss.item())

            if loss_value < best_loss:
                best_loss = loss_value
                best_state_dict = copy.deepcopy(model.state_dict())

            JT = J.transpose(0, 1)
            H_approx = JT @ J
            g = JT @ e

            grad_norm = torch.norm(g).item()
            if grad_norm < self.tol_grad:
                if self.verbose:
                    print(f"[LM] Stopping at epoch {epoch}: gradient norm below tolerance.")
                break

            flat_params = _flatten_parameters(parameters)
            identity = torch.eye(
                H_approx.shape[0],
                device=self.device,
                dtype=self.dtype
            )

            accepted = False
            step_norm = None

            while not accepted:
                system_matrix = H_approx + mu * identity

                try:
                    delta = torch.linalg.solve(system_matrix, -g)
                except RuntimeError:
                    mu *= self.mu_increase
                    if mu > self.mu_max:
                        if self.verbose:
                            print("[LM] Aborting: mu exceeded mu_max after singular solve.")
                        model.load_state_dict(best_state_dict)
                        final_e = self._residual_vector(model, x, y)
                        final_loss = float(self._loss_from_residuals(final_e).item())
                        return LMTrainingResult(
                            best_loss=best_loss,
                            final_loss=final_loss,
                            epochs_completed=epoch,
                            mu_final=mu,
                            loss_history=loss_history,
                            mu_history=mu_history,
                            best_state_dict=best_state_dict
                        )
                    continue

                step_norm = torch.norm(delta).item()
                if step_norm < self.tol_step:
                    if self.verbose:
                        print(f"[LM] Stopping at epoch {epoch}: step norm below tolerance.")
                    model.load_state_dict(best_state_dict)
                    final_e = self._residual_vector(model, x, y)
                    final_loss = float(self._loss_from_residuals(final_e).item())
                    return LMTrainingResult(
                        best_loss=best_loss,
                        final_loss=final_loss,
                        epochs_completed=epoch,
                        mu_final=mu,
                        loss_history=loss_history,
                        mu_history=mu_history,
                        best_state_dict=best_state_dict
                    )

                trial_params = flat_params + delta
                _assign_flat_parameters(parameters, trial_params)

                with torch.no_grad():
                    e_trial = self._residual_vector(model, x, y)
                    trial_loss = self._loss_from_residuals(e_trial)
                    trial_loss_value = float(trial_loss.item())

                if trial_loss_value < loss_value:
                    accepted = True
                    mu *= self.mu_decrease
                    mu = max(mu, 1e-30)
                    loss_value = trial_loss_value
                else:
                    _assign_flat_parameters(parameters, flat_params)
                    mu *= self.mu_increase

                    if mu > self.mu_max:
                        if self.verbose:
                            print("[LM] Aborting: mu exceeded mu_max after repeated rejections.")
                        model.load_state_dict(best_state_dict)
                        final_e = self._residual_vector(model, x, y)
                        final_loss = float(self._loss_from_residuals(final_e).item())
                        return LMTrainingResult(
                            best_loss=best_loss,
                            final_loss=final_loss,
                            epochs_completed=epoch,
                            mu_final=mu,
                            loss_history=loss_history,
                            mu_history=mu_history,
                            best_state_dict=best_state_dict
                        )

            loss_history.append(loss_value)
            mu_history.append(mu)

            if self.verbose:
                print(
                    f"[LM] epoch={epoch:03d} "
                    f"loss={loss_value:.12f} "
                    f"mu={mu:.3e} "
                    f"grad_norm={grad_norm:.3e} "
                    f"step_norm={step_norm:.3e}"
                )

            if previous_loss is not None:
                improvement = abs(previous_loss - loss_value)
                if improvement < self.tol_loss:
                    if self.verbose:
                        print(f"[LM] Stopping at epoch {epoch}: loss improvement below tolerance.")
                    break

            previous_loss = loss_value

        model.load_state_dict(best_state_dict)
        final_e = self._residual_vector(model, x, y)
        final_loss = float(self._loss_from_residuals(final_e).item())

        return LMTrainingResult(
            best_loss=best_loss,
            final_loss=final_loss,
            epochs_completed=len(loss_history),
            mu_final=mu,
            loss_history=loss_history,
            mu_history=mu_history,
            best_state_dict=best_state_dict
        )


@dataclass
class TimeSeriesSplit:
    """Chronological train-validation split for time-series ANN estimation."""
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    train_index: list
    val_index: list


def make_last_block_validation_split(X: torch.Tensor, y: torch.Tensor, time_index: list, *, val_fraction: float = 0.15) -> TimeSeriesSplit:
    """
    Split the sample chronologically, reserving the last val_fraction for validation.

    Args:
        X: Input tensor of shape (n_samples, n_features).
        y: Target tensor of shape (n_samples, 1).
        time_index: List-like aligned time index.
        val_fraction: Fraction of final observations used for validation.

    Returns:
        TimeSeriesSplit
    """
    X = _to_2d_tensor(X)
    y = _to_2d_tensor(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if len(time_index) != X.shape[0]:
        raise ValueError("time_index length must match number of rows in X and y.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must satisfy 0 < val_fraction < 1.")

    n = X.shape[0]
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val

    if n_train <= 0:
        raise ValueError("Validation fraction leaves no training observations.")

    return TimeSeriesSplit(
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_val=X[n_train:],
        y_val=y[n_train:],
        train_index=time_index[:n_train],
        val_index=time_index[n_train:],
    )


@dataclass
class LMValidationTrainingResult:
    """Container for validation-aware LM training results."""
    best_train_loss: float
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float
    epochs_completed: int
    best_epoch: int
    mu_final: float
    train_loss_history: List[float]
    val_loss_history: List[float]
    mu_history: List[float]
    best_state_dict: Dict[str, torch.Tensor]


class ValidationLevenbergMarquardtTrainer(LevenbergMarquardtTrainer):
    """
    Validation-aware LM trainer with early stopping on validation MSE.
    """

    def fit_with_validation(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, 
                            y_val: torch.Tensor, *, patience: int = 6, min_delta: float = 0.0,) -> LMValidationTrainingResult:
        """
        Fit a model using LM while monitoring validation loss.

        Early stopping:
            Stop when validation MSE fails to improve by more than min_delta
            for 'patience' consecutive epochs.

        Args:
            model: PyTorch model.
            X_train: Training inputs.
            y_train: Training targets.
            X_val: Validation inputs.
            y_val: Validation targets.
            patience: Early stopping patience in epochs.
            min_delta: Minimum improvement required to reset patience.

        Returns:
            LMValidationTrainingResult
        """
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")

        model = self._prepare_model(model)
        X_train, y_train = self._prepare_data(X_train, y_train)
        X_val, y_val = self._prepare_data(X_val, y_val)

        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("Model has no trainable parameters.")

        mu = self.mu_init
        train_loss_history: List[float] = []
        val_loss_history: List[float] = []
        mu_history: List[float] = []

        best_state_dict = copy.deepcopy(model.state_dict())
        best_train_loss = float("inf")
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            J, e = self._jacobian(model, X_train, y_train)
            train_loss = self._loss_from_residuals(e)
            train_loss_value = float(train_loss.item())

            JT = J.transpose(0, 1)
            H_approx = JT @ J
            g = JT @ e

            grad_norm = torch.norm(g).item()
            if grad_norm < self.tol_grad:
                if self.verbose:
                    print(f"[LM-VAL] Stopping at epoch {epoch}: gradient norm below tolerance.")
                break

            flat_params = _flatten_parameters(parameters)
            identity = torch.eye(
                H_approx.shape[0],
                device=self.device,
                dtype=self.dtype,
            )

            accepted = False
            step_norm = None

            while not accepted:
                system_matrix = H_approx + mu * identity

                try:
                    delta = torch.linalg.solve(system_matrix, -g)
                except RuntimeError:
                    mu *= self.mu_increase
                    if mu > self.mu_max:
                        if self.verbose:
                            print("[LM-VAL] Aborting: mu exceeded mu_max after singular solve.")
                        model.load_state_dict(best_state_dict)

                        final_train = float(self._loss_from_residuals(
                            self._residual_vector(model, X_train, y_train)
                        ).item())
                        final_val = float(self._loss_from_residuals(
                            self._residual_vector(model, X_val, y_val)
                        ).item())

                        return LMValidationTrainingResult(
                            best_train_loss=best_train_loss,
                            best_val_loss=best_val_loss,
                            final_train_loss=final_train,
                            final_val_loss=final_val,
                            epochs_completed=epoch,
                            best_epoch=best_epoch,
                            mu_final=mu,
                            train_loss_history=train_loss_history,
                            val_loss_history=val_loss_history,
                            mu_history=mu_history,
                            best_state_dict=best_state_dict,
                        )
                    continue

                step_norm = torch.norm(delta).item()
                if step_norm < self.tol_step:
                    if self.verbose:
                        print(f"[LM-VAL] Stopping at epoch {epoch}: step norm below tolerance.")
                    break

                trial_params = flat_params + delta
                _assign_flat_parameters(parameters, trial_params)

                with torch.no_grad():
                    e_trial = self._residual_vector(model, X_train, y_train)
                    trial_train_loss = self._loss_from_residuals(e_trial)
                    trial_train_loss_value = float(trial_train_loss.item())

                if trial_train_loss_value < train_loss_value:
                    accepted = True
                    mu *= self.mu_decrease
                    mu = max(mu, 1e-30)
                    train_loss_value = trial_train_loss_value
                else:
                    _assign_flat_parameters(parameters, flat_params)
                    mu *= self.mu_increase

                    if mu > self.mu_max:
                        if self.verbose:
                            print("[LM-VAL] Aborting: mu exceeded mu_max after repeated rejections.")
                        model.load_state_dict(best_state_dict)

                        final_train = float(self._loss_from_residuals(
                            self._residual_vector(model, X_train, y_train)
                        ).item())
                        final_val = float(self._loss_from_residuals(
                            self._residual_vector(model, X_val, y_val)
                        ).item())

                        return LMValidationTrainingResult(
                            best_train_loss=best_train_loss,
                            best_val_loss=best_val_loss,
                            final_train_loss=final_train,
                            final_val_loss=final_val,
                            epochs_completed=epoch,
                            best_epoch=best_epoch,
                            mu_final=mu,
                            train_loss_history=train_loss_history,
                            val_loss_history=val_loss_history,
                            mu_history=mu_history,
                            best_state_dict=best_state_dict,
                        )

            with torch.no_grad():
                val_e = self._residual_vector(model, X_val, y_val)
                val_loss_value = float(self._loss_from_residuals(val_e).item())

            train_loss_history.append(train_loss_value)
            val_loss_history.append(val_loss_value)
            mu_history.append(mu)

            if train_loss_value < best_train_loss:
                best_train_loss = train_loss_value

            if val_loss_value < (best_val_loss - min_delta):
                best_val_loss = val_loss_value
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(
                    f"[LM-VAL] epoch={epoch:03d} "
                    f"train={train_loss_value:.12f} "
                    f"val={val_loss_value:.12f} "
                    f"mu={mu:.3e} "
                    f"grad_norm={grad_norm:.3e} "
                    f"step_norm={0.0 if step_norm is None else step_norm:.3e}"
                )

            if epochs_without_improvement >= patience:
                if self.verbose:
                    print(
                        f"[LM-VAL] Early stopping at epoch {epoch}: "
                        f"validation loss failed to improve for {patience} consecutive epochs."
                    )
                break

        model.load_state_dict(best_state_dict)

        final_train_loss = float(self._loss_from_residuals(
            self._residual_vector(model, X_train, y_train)
        ).item())
        final_val_loss = float(self._loss_from_residuals(
            self._residual_vector(model, X_val, y_val)
        ).item())

        return LMValidationTrainingResult(
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            epochs_completed=len(train_loss_history),
            best_epoch=best_epoch,
            mu_final=mu,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            mu_history=mu_history,
            best_state_dict=best_state_dict,
        )

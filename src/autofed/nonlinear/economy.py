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

"""Fitted non-linear economies: one-step prediction, NSSM training and persistence.

``NARXEconomy`` and ``NSSMEconomy`` are what the notebooks build, save and reload. Each bundles the
fitted networks with everything needed to use them elsewhere (scalers, latent states, shock scales).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

from ..exceptions import ArtifactError, SpecificationError
from .narx import NARXNet, build_restricted_narx_design_matrices
from .nssm import FullNSSMInputs, TwoBlockNSSM, build_full_nssm_inputs, run_two_block_nssm_sequence
from .scaling import MapMinMaxScaler
from .search import HiddenSearchSummary

__all__ = ["NARXEconomy", "NSSMEconomy", "NSSMTrainingResult", "train_nssm"]

_FORMAT_VERSION = 1


def _scaler_state(scaler: MapMinMaxScaler) -> dict[str, Any]:
    if not scaler.fitted_ or scaler.data_min_ is None or scaler.data_max_ is None:
        raise SpecificationError("cannot save a scaler that has not been fitted")
    return {"feature_range": list(scaler.feature_range), "eps": scaler.eps,
            "data_min": scaler.data_min_.detach().cpu(), "data_max": scaler.data_max_.detach().cpu()}


def _scaler_from_state(state: dict[str, Any], dtype: torch.dtype) -> MapMinMaxScaler:
    lower, upper = state["feature_range"]
    bounds = torch.stack([state["data_min"], state["data_max"]]).to(dtype)
    return MapMinMaxScaler(feature_range=(float(lower), float(upper)), eps=float(state["eps"])).fit(bounds)


def _load(path: str | os.PathLike[str], kind: str, producer: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"{source} not found; run {producer} first")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for a damaged file
        raise ArtifactError(f"{source} could not be read: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ArtifactError(f"{source} is not a saved {kind} economy")
    if payload.get("version") != _FORMAT_VERSION:
        raise ArtifactError(f"{source} has format version {payload.get('version')}; expected {_FORMAT_VERSION}")
    return payload


def _frame(time_index: list, y: torch.Tensor, pi: torch.Tensor, y_hat: torch.Tensor, pi_hat: torch.Tensor) -> pd.DataFrame:
    flat = lambda t: t.detach().cpu().reshape(-1).numpy()                     # noqa: E731
    return pd.DataFrame({"y": flat(y), "pi": flat(pi), "y_hat": flat(y_hat), "pi_hat": flat(pi_hat)},
                        index=pd.PeriodIndex(time_index, freq="Q", name="quarter"))


def _residual_sigma(fit: pd.DataFrame) -> tuple[float, float]:
    return (float((fit["y"] - fit["y_hat"]).std(ddof=1)), float((fit["pi"] - fit["pi_hat"]).std(ddof=1)))


@dataclass
class NARXEconomy:
    """The two restricted NARX transition equations with their scalers.

    Attributes:
        net_y: Output-gap network on ``(y_{t-1}, pi_{t-1}, i_{t-1})``.
        net_pi: Inflation network on ``(y_t, y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1})``.
        x_scaler_y: Input scaler of the output-gap network.
        y_scaler_y: Target scaler of the output-gap network.
        x_scaler_pi: Input scaler of the inflation network.
        y_scaler_pi: Target scaler of the inflation network.
        sigma_y: Standard deviation of the output-gap residuals over the estimation window.
        sigma_pi: Standard deviation of the inflation residuals over the estimation window.
        meta: Free-form provenance.
    """

    net_y: NARXNet
    net_pi: NARXNet
    x_scaler_y: MapMinMaxScaler
    y_scaler_y: MapMinMaxScaler
    x_scaler_pi: MapMinMaxScaler
    y_scaler_pi: MapMinMaxScaler
    sigma_y: float = 0.0
    sigma_pi: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_search(cls, search_y: HiddenSearchSummary, search_pi: HiddenSearchSummary, *,
                    dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None,
                    **meta: Any) -> NARXEconomy:
        """Instantiate the selected networks from two hidden-unit searches.

        For each equation the hidden size with the lowest mean validation MSE is used, with the
        weights of the trial that achieved the lowest overall MSE at that size.
        """
        nets, scalers = [], []
        for search in (search_y, search_pi):
            trial = search.best_trial_by_hidden[search.selected_hidden_size]
            n_inputs = int(trial.model_state_dict["hidden.weight"].shape[1])
            net = NARXNet(input_size=n_inputs, hidden_size=trial.hidden_size, output_size=1).to(device=device, dtype=dtype)
            net.load_state_dict(trial.model_state_dict)
            nets.append(net.eval())
            scalers.extend([trial.x_scaler, trial.y_scaler])
        return cls(nets[0], nets[1], scalers[0], scalers[1], scalers[2], scalers[3], meta=dict(meta))

    def _spec(self) -> tuple[torch.dtype, torch.device]:
        parameter = next(self.net_y.parameters())
        return parameter.dtype, parameter.device

    @torch.no_grad()
    def predict_y(self, X: torch.Tensor) -> torch.Tensor:
        """Output-gap prediction in percent for rows of ``(y_{t-1}, pi_{t-1}, i_{t-1})``."""
        return self.y_scaler_y.inverse_transform(self.net_y(self.x_scaler_y.transform(X))).reshape(-1)

    @torch.no_grad()
    def predict_pi(self, X: torch.Tensor) -> torch.Tensor:
        """Inflation prediction in percent for rows of ``(y_t, y_{t-1}, y_{t-2}, pi_{t-1}, pi_{t-2}, i_{t-1})``."""
        return self.y_scaler_pi.inverse_transform(self.net_pi(self.x_scaler_pi.transform(X))).reshape(-1)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One-step fits on realised lags; inflation is conditional on the realised output gap.

        Returns:
            A frame from the third quarter of ``frame`` on, with columns ``y``, ``pi``, ``y_hat``, ``pi_hat``.
        """
        dtype, device = self._spec()
        design = build_restricted_narx_design_matrices(frame, dtype=dtype, device=device)
        return _frame(design.time_index, design.y_target, design.pi_target,
                      self.predict_y(design.X_y), self.predict_pi(design.X_pi))

    def set_sigma(self, fit: pd.DataFrame) -> None:
        """Set the shock scales to the sample standard deviations of the residuals in ``fit``."""
        self.sigma_y, self.sigma_pi = _residual_sigma(fit)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the economy to one file readable with ``weights_only=True``; returns the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dtype, _ = self._spec()
        payload = {
            "kind": "narx", "version": _FORMAT_VERSION, "dtype": str(dtype).removeprefix("torch."),
            "hidden": [self.net_y.hidden_size, self.net_pi.hidden_size],
            "inputs": [self.net_y.input_size, self.net_pi.input_size],
            "state_y": {k: v.detach().cpu() for k, v in self.net_y.state_dict().items()},
            "state_pi": {k: v.detach().cpu() for k, v in self.net_pi.state_dict().items()},
            "scalers": [_scaler_state(s) for s in (self.x_scaler_y, self.y_scaler_y, self.x_scaler_pi, self.y_scaler_pi)],
            "sigma": [float(self.sigma_y), float(self.sigma_pi)], "meta": dict(self.meta),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> NARXEconomy:
        """Read an economy written by :meth:`save`.

        Raises:
            ArtifactError: If the file is missing, damaged, or not a NARX economy.
        """
        payload = _load(path, "narx", "notebook 05_nonlinear_narx")
        dtype = getattr(torch, payload["dtype"])
        nets = []
        for n_inputs, n_hidden, key in zip(payload["inputs"], payload["hidden"], ("state_y", "state_pi")):
            net = NARXNet(input_size=n_inputs, hidden_size=n_hidden, output_size=1).to(dtype=dtype)
            net.load_state_dict(payload[key])
            nets.append(net.eval())
        scalers = [_scaler_from_state(state, dtype) for state in payload["scalers"]]
        return cls(nets[0], nets[1], *scalers, sigma_y=float(payload["sigma"][0]), sigma_pi=float(payload["sigma"][1]),
                   meta=dict(payload.get("meta", {})))


@dataclass
class NSSMTrainingResult:
    """Outcome of :func:`train_nssm`.

    Attributes:
        model: The trained model, in evaluation mode.
        h0_y: Initial latent state of the output-gap block, shape ``(1, state_dim_y)``.
        h0_pi: Initial latent state of the inflation block, shape ``(1, state_dim_pi)``.
        loss_history: Per-epoch losses with columns ``total``, ``y`` and ``pi``.
    """

    model: TwoBlockNSSM
    h0_y: torch.Tensor
    h0_pi: torch.Tensor
    loss_history: pd.DataFrame


def train_nssm(inputs: FullNSSMInputs, *, state_dim_y: int = 4, state_dim_pi: int = 4, epochs: int = 1000,
               lr: float = 1e-3, weight_decay: float = 1e-4, seed: Optional[int] = None) -> NSSMTrainingResult:
    """Estimate the two-block NSSM by full-sequence AdamW on the joint mean squared error.

    Args:
        inputs: Lag-aligned input and target sequences.
        state_dim_y: Latent dimension of the output-gap block.
        state_dim_pi: Latent dimension of the inflation block.
        epochs: Number of full-sequence gradient steps.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        seed: Seed of the weight initialisation; ``None`` leaves the generator untouched.

    Raises:
        ValueError: If a dimension, ``epochs`` or ``lr`` is not positive.
    """
    if min(state_dim_y, state_dim_pi, epochs) < 1 or lr <= 0:
        raise ValueError("state dimensions, epochs and lr must be positive")
    if seed is not None:
        torch.manual_seed(seed)
    dtype, device = inputs.X_y.dtype, inputs.X_y.device
    model = TwoBlockNSSM(state_dim_y=state_dim_y, state_dim_pi=state_dim_pi, input_dim_y=inputs.X_y.shape[1],
                         input_dim_pi=inputs.X_pi.shape[1]).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    h0_y = torch.zeros((1, state_dim_y), dtype=dtype, device=device)
    h0_pi = torch.zeros((1, state_dim_pi), dtype=dtype, device=device)
    history: list[tuple[float, float, float]] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        y_hat, pi_hat, _, _ = run_two_block_nssm_sequence(model=model, X_y=inputs.X_y, X_pi=inputs.X_pi, h0_y=h0_y, h0_pi=h0_pi)
        loss_y = torch.mean((y_hat - inputs.y_target) ** 2)
        loss_pi = torch.mean((pi_hat - inputs.pi_target) ** 2)
        loss = loss_y + loss_pi
        loss.backward()
        optimizer.step()
        history.append((float(loss.item()), float(loss_y.item()), float(loss_pi.item())))
    frame = pd.DataFrame(history, columns=["total", "y", "pi"])
    frame.index.name = "epoch"
    return NSSMTrainingResult(model.eval(), h0_y, h0_pi, frame)


@dataclass
class NSSMEconomy:
    """The fitted two-block NSSM with its initial and anchored latent states.

    Attributes:
        model: The fitted model.
        h0_y: Initial latent state of the output-gap block, shape ``(1, d_y)``.
        h0_pi: Initial latent state of the inflation block, shape ``(1, d_pi)``.
        terminal_y: Output-gap latent state at the last estimation-window quarter, shape ``(1, d_y)``.
        terminal_pi: Inflation latent state at the last estimation-window quarter, shape ``(1, d_pi)``.
        sigma_y: Standard deviation of the output-gap residuals over the estimation window.
        sigma_pi: Standard deviation of the inflation residuals over the estimation window.
        meta: Free-form provenance.
    """

    model: TwoBlockNSSM
    h0_y: torch.Tensor
    h0_pi: torch.Tensor
    terminal_y: torch.Tensor
    terminal_pi: torch.Tensor
    sigma_y: float = 0.0
    sigma_pi: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_training(cls, result: NSSMTrainingResult, train: pd.DataFrame, **meta: Any) -> NSSMEconomy:
        """Build the economy from a training result: anchor the latent states and set the shock scales."""
        economy = cls(result.model, result.h0_y, result.h0_pi, result.h0_y.clone(), result.h0_pi.clone(), meta=dict(meta))
        fit, states_y, states_pi = economy.run(train)
        economy.terminal_y, economy.terminal_pi = states_y[-1:].clone(), states_pi[-1:].clone()
        economy.sigma_y, economy.sigma_pi = _residual_sigma(fit)
        return economy

    @torch.no_grad()
    def run(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
        """Propagate the model over ``frame`` from the initial latent state.

        Returns:
            ``(fit, states_y, states_pi)``: one-step fits with columns ``y``, ``pi``, ``y_hat``,
            ``pi_hat`` from the third quarter on, and the latent-state paths.
        """
        parameter = next(self.model.parameters())
        inputs = build_full_nssm_inputs(frame, dtype=parameter.dtype, device=parameter.device)
        y_hat, pi_hat, states_y, states_pi = run_two_block_nssm_sequence(
            model=self.model, X_y=inputs.X_y, X_pi=inputs.X_pi, h0_y=self.h0_y, h0_pi=self.h0_pi)
        return _frame(inputs.time_index, inputs.y_target, inputs.pi_target, y_hat, pi_hat), states_y, states_pi

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One-step fits over ``frame``; see :meth:`run`."""
        return self.run(frame)[0]

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the economy to one file readable with ``weights_only=True``; returns the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model = self.model
        payload = {
            "kind": "nssm", "version": _FORMAT_VERSION, "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "dims": [model.W_h_y.in_features, model.W_h_pi.in_features, model.W_x_y.in_features, model.W_x_pi.in_features],
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "h0": [self.h0_y.detach().cpu(), self.h0_pi.detach().cpu()],
            "terminal": [self.terminal_y.detach().cpu(), self.terminal_pi.detach().cpu()],
            "sigma": [float(self.sigma_y), float(self.sigma_pi)], "meta": dict(self.meta),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> NSSMEconomy:
        """Read an economy written by :meth:`save`.

        Raises:
            ArtifactError: If the file is missing, damaged, or not an NSSM economy.
        """
        payload = _load(path, "nssm", "notebook 06_nonlinear_nssm")
        dtype = getattr(torch, payload["dtype"])
        d_y, d_pi, in_y, in_pi = payload["dims"]
        model = TwoBlockNSSM(state_dim_y=d_y, state_dim_pi=d_pi, input_dim_y=in_y, input_dim_pi=in_pi).to(dtype=dtype)
        model.load_state_dict(payload["state"])
        return cls(model.eval(), payload["h0"][0], payload["h0"][1], payload["terminal"][0], payload["terminal"][1],
                   sigma_y=float(payload["sigma"][0]), sigma_pi=float(payload["sigma"][1]), meta=dict(payload.get("meta", {})))

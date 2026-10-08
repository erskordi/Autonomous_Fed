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

"""PyTorch-native analogue of MATLAB's ``mapminmax``."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._tensor import _to_2d_tensor, _validate_finite_tensor

__all__ = ["MapMinMaxScaler"]


class MapMinMaxScaler:
    """
    PyTorch-native analogue of MATLAB's mapminmax.

    Transforms each feature independently from observed data range
    [x_min, x_max] into [feature_range[0], feature_range[1]].

    Default output range is [-1, 1], which is appropriate for tanh networks.
    """

    def __init__(self, feature_range: Tuple[float, float] = (-1.0, 1.0), eps: float = 1e-12) -> None:
        """
        Args:
            feature_range: Desired output range.
            eps: Small constant to avoid division by zero.
        """
        lower, upper = feature_range
        if lower >= upper:
            raise ValueError("feature_range must satisfy lower < upper.")
        if eps <= 0:
            raise ValueError("eps must be strictly positive.")

        self.feature_range = feature_range
        self.eps = float(eps)

        self.data_min_: Optional[torch.Tensor] = None
        self.data_max_: Optional[torch.Tensor] = None
        self.data_range_: Optional[torch.Tensor] = None
        self.scale_: Optional[torch.Tensor] = None
        self.min_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.fitted_: bool = False

    def fit(self, x: torch.Tensor) -> "MapMinMaxScaler":
        """
        Fit the scaler on 2D data of shape (n_samples, n_features).
        """
        x = _to_2d_tensor(x)
        _validate_finite_tensor(x, "x")

        self.data_min_ = x.min(dim=0).values
        self.data_max_ = x.max(dim=0).values
        self.data_range_ = self.data_max_ - self.data_min_

        lower, upper = self.feature_range
        safe_range = torch.where(
            self.data_range_ < self.eps,
            torch.ones_like(self.data_range_),
            self.data_range_
        )

        self.scale_ = (upper - lower) / safe_range
        self.min_ = lower - self.data_min_ * self.scale_
        self.n_features_in_ = x.shape[1]
        self.fitted_ = True

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform data using fitted scaling parameters.
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before calling transform().")

        x = _to_2d_tensor(x)
        _validate_finite_tensor(x, "x")

        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {x.shape[1]}."
            )

        transformed = x * self.scale_ + self.min_

        lower, upper = self.feature_range
        midpoint = (lower + upper) / 2.0
        constant_mask = self.data_range_ < self.eps
        if constant_mask.any():
            transformed[:, constant_mask] = midpoint

        return transformed

    def inverse_transform(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """
        Invert scaling back to original units.
        """
        if not self.fitted_:
            raise RuntimeError("Scaler must be fitted before inverse_transform().")

        x_scaled = _to_2d_tensor(x_scaled)
        _validate_finite_tensor(x_scaled, "x_scaled")

        if x_scaled.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {x_scaled.shape[1]}."
            )

        original = (x_scaled - self.min_) / self.scale_

        constant_mask = self.data_range_ < self.eps
        if constant_mask.any():
            original[:, constant_mask] = self.data_min_[constant_mask]

        return original

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fit and transform in one step.
        """
        return self.fit(x).transform(x)

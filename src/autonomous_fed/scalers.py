"""
Module defining scalers for data normalization.
"""

from typing import Optional
from dataclasses import dataclass
import torch

@dataclass
class MapMinMax:
    """
    Min-max scaler mapping features to a target range.

    Attributes:
        in_min (Optional[torch.Tensor]): Minimum values for each feature in the input data.
        in_max (Optional[torch.Tensor]): Maximum values for each feature in the input data.
        out_lo (float): Lower bound of the output range.
        out_hi (float): Upper bound of the output range.
        scale_ (Optional[torch.Tensor]): Scaling factors for each feature.
        mid_ (Optional[torch.Tensor]): Midpoint adjustments for each feature.

    Methods:
        fit: Compute the min and max values for scaling.
        transform: Scale the input data to the specified range.
        inverse_transform: Revert the scaled data back to the original range.
    """
    in_min: Optional[torch.Tensor] = None
    in_max: Optional[torch.Tensor] = None
    out_lo: float = -1.0
    out_hi: float =  1.0
    scale_: Optional[torch.Tensor] = None
    mid_:   Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "MapMinMax":
        """
        Fit the scaler to the data x.

        Args:
            x (torch.Tensor): Input data to compute min and max values.

        Returns:
            MapMinMax: The fitted scaler instance.

        Raises:
            ValueError: If x is not a 2D tensor.
        """
        if x.ndim != 2:
            raise ValueError("x must be 2-dimensional.")

        # NaN-safe min/max using nan_to_num
        x_min = torch.min(torch.nan_to_num(x, nan=float("inf")), dim=0).values
        x_max = torch.max(torch.nan_to_num(x, nan=-float("inf")), dim=0).values
        # If a column is all-NaN: set min/max = 0 (or choose your default)
        bad = ~torch.isfinite(x_min) | ~torch.isfinite(x_max)
        x_min = torch.where(bad, torch.zeros_like(x_min), x_min)
        x_max = torch.where(bad, torch.zeros_like(x_max), x_max)
        self.in_min = x_min
        self.in_max = x_max

        # Handle constant columns
        rng = self.in_max - self.in_min
        rng_safe = torch.where(rng == 0.0, torch.ones_like(rng), rng)

        # Compute scale + mid
        self.scale_ = (self.out_hi - self.out_lo) / rng_safe
        in_mid  = (self.in_min + self.in_max) * 0.5
        out_mid = (self.out_lo + self.out_hi) * 0.5
        self.mid_ = out_mid - self.scale_ * in_mid

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the input data to the specified range.

        Args:
            x (torch.Tensor): Input data to be scaled.
        Returns:
            torch.Tensor: Scaled data.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if self.scale_ is None:
            raise ValueError("Scaler not fitted. Call .fit(x) first.")
        return self.scale_ * x + self.mid_ # type: ignore[operator]

    def inverse_transform(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Revert the scaled data back to the original range.

        Args:
            xs (torch.Tensor): Scaled data to be reverted.

        Returns:
            torch.Tensor: Data in the original scale.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if self.scale_ is None:
            raise ValueError("Scaler not fitted. Call .fit(x) first.")

        # Prevent div-by-zero
        inv_scale = torch.where(self.scale_ == 0.0, torch.ones_like(self.scale_), self.scale_)

        return (xs - self.mid_) / inv_scale # type: ignore[operator]

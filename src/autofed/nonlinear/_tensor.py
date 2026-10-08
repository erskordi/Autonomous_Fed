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

"""Tensor shape, finiteness and parameter-vector helpers (private)."""

from __future__ import annotations

from typing import List, Sequence

import torch

__all__ = ["_assign_flat_parameters", "_flatten_parameters", "_num_parameters", "_to_2d_tensor", "_validate_finite_tensor"]


def _to_2d_tensor(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2D."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}.")
    if x.ndim == 1:
        return x.unsqueeze(1)
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(x.shape)}.")
    return x


def _validate_finite_tensor(x: torch.Tensor, name: str) -> None:
    """Validate that a tensor contains only finite values."""
    if not torch.isfinite(x).all():
        raise ValueError(f"Tensor '{name}' contains NaN or infinite values.")


def _flatten_parameters(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten a sequence of parameters into a single 1D tensor."""
    flat_parts: List[torch.Tensor] = []
    for parameter in parameters:
        flat_parts.append(parameter.detach().reshape(-1))
    if not flat_parts:
        raise ValueError("No parameters were provided for flattening.")
    return torch.cat(flat_parts)


def _assign_flat_parameters(parameters: Sequence[torch.nn.Parameter], flat_vector: torch.Tensor) -> None:
    """Assign a flat parameter vector back into model parameters in-place."""
    if flat_vector.ndim != 1:
        raise ValueError("Flat parameter vector must be 1D.")

    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            numel = parameter.numel()
            parameter.copy_(flat_vector[offset:offset + numel].view_as(parameter))
            offset += numel

    if offset != flat_vector.numel():
        raise ValueError(
            "Flat parameter vector size does not match total model parameter size."
        )


def _num_parameters(parameters: Sequence[torch.nn.Parameter]) -> int:
    """Return total number of scalar parameters."""
    return sum(parameter.numel() for parameter in parameters)

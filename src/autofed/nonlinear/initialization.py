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

"""Nguyen-Widrow initialisation for shallow tanh networks."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["nguyen_widrow_"]


@torch.no_grad()
def nguyen_widrow_(layer: nn.Linear, nonlinearity: str = "tanh") -> nn.Linear:
    """
    Apply Nguyen-Widrow initialization to a Linear layer.

    This is classically used for single-hidden-layer feedforward networks with
    bounded nonlinear activations such as tanh.

    Args:
        layer: nn.Linear layer to initialize.
        nonlinearity: Currently only 'tanh' is supported explicitly.

    Returns:
        The initialized layer.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError("nguyen_widrow_ expects an nn.Linear layer.")

    if nonlinearity.lower() != "tanh":
        raise ValueError(
            "This Nguyen-Widrow implementation currently supports nonlinearity='tanh' only."
        )

    out_features = layer.out_features
    in_features = layer.in_features

    if out_features <= 0 or in_features <= 0:
        raise ValueError("Linear layer must have positive in_features and out_features.")

    layer.weight.uniform_(-0.5, 0.5)

    weight = layer.weight.data
    norms = torch.norm(weight, dim=1, keepdim=True).clamp_min(1e-12)

    beta = 0.7 * (out_features ** (1.0 / in_features))
    layer.weight.copy_(beta * weight / norms)

    if layer.bias is not None:
        layer.bias.uniform_(-beta, beta)

    return layer

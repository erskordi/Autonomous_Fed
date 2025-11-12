"""
Module for custom neural network architectures.
"""

import torch
from torch import nn
from .initializers import nguyen_widrow_

class SingleHiddenLayerNet(nn.Module):
    """
    Single hidden layer neural network.

    Attributes:
        h (nn.Linear): Hidden layer.
        o (nn.Linear): Output layer.

    Methods:
        forward: Forward pass through the network.
    """
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.h = nn.Linear(n_in, n_hidden)
        nguyen_widrow_(self.h) # only hidden layer
        self.o = nn.Linear(n_hidden, n_out)
        # Small random init for output layer (paper-style)
        bound = 0.03
        nn.init.uniform_(self.o.weight, -bound, bound)
        nn.init.uniform_(self.o.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.

        Raises:
            None
        """
        if x.dtype != self.h.weight.dtype:
            x = x.to(self.h.weight.dtype)
        return self.o(torch.tanh(self.h(x)))

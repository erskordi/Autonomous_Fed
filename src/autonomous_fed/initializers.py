"""
Module for custom neural network initializers.
"""

import torch
from torch import nn

@torch.no_grad()
def nguyen_widrow_(layer: nn.Linear) -> None:
    """
    Nguyen-Widrow initialization for a single linear layer feeding a tanh/sigmoid block. Scales incoming weight vectors to beta = 0.7 * (n_out)^(1/n_in). Bias in [-beta, beta].

    Args:
        layer (nn.Linear): Linear layer to initialize.

    Returns:
        None

    Raises:
        None
    """
    n_out, n_in = layer.weight.shape
    with torch.no_grad():
        w = torch.empty_like(layer.weight).uniform_(-0.5, 0.5)
        norms = torch.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
        w = w / norms
        beta = 0.7 * (n_out ** (1.0 / max(1, n_in)))
        layer.weight.copy_(w * beta)
        layer.bias.uniform_(-beta, beta)

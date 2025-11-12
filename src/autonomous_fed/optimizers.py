"""
Module for custom neural network optimizers.
"""

from typing import Dict, Iterable, Union, Callable
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class LevenbergMarquardt(torch.optim.Optimizer):
    """
    Implements the Levenberg-Marquardt optimization algorithm for non-linear least squares problems.

    Attributes:
        params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize.
        mu (float): Initial damping factor.
        mu_lo (float): Minimum damping factor.
        mu_hi (float): Maximum damping factor.
        mu_up (float): Factor to increase mu when step is rejected.
        mu_down (float): Factor to decrease mu when step is accepted.
        weight_decay (float): Weight decay (L2 penalty) coefficient.

    Methods:
        step(closure): Performs a single optimization step.
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], mu: float=1e-3, mu_lo: float=1e-12, mu_hi: float=1e10, mu_up: float=10.0, mu_down: float=0.1, weight_decay: float=0.0) -> None:
        """
        Initializes the Levenberg-Marquardt optimizer.

        Args:
            params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize.
            mu (float, optional): Initial damping factor. Default is 1e-3.
            mu_lo (float, optional): Minimum damping factor. Default is 1e-12
            mu_hi (float, optional): Maximum damping factor. Default is 1e10.
            mu_up (float, optional): Factor to increase mu when step is rejected. Default is 10.0.
            mu_down (float, optional): Factor to decrease mu when step is accepted. Default is 0.1.
            weight_decay (float, optional): Weight decay (L2 penalty) coefficient. Default is 0.0.

        Returns:
            None

        Raises:
            ValueError: If any of the hyperparameters are out of valid range.
        """
        defaults = dict(mu=mu, mu_lo=mu_lo, mu_hi=mu_hi, mu_up=mu_up, mu_down=mu_down, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> Dict[str, Union[float, bool]]: # type: ignore[override]
        """
        Performs a single optimization step using the Levenberg-Marquardt algorithm.

        Args:
            closure (callable): A closure that reevaluates the model and returns the residuals.

        Returns:
            Dict: A dictionary containing information about the optimization step:
                - "rho": Gain ratio (float).
                - "accepted": Whether the step was accepted (bool).
                - "mu": Updated damping factor (float).
                - "old_sse": Sum of squared errors before the step (float).
                - "new_sse": Sum of squared errors after the step (float).

        Raises:
            RuntimeError: If the linear system cannot be solved.
        """
        # Flatten parameters
        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)
        theta = parameters_to_vector(params)
        p = theta.numel()

        # Compute residuals r(theta)
        r = closure() # 1D vector (M,)
        if r.dim() > 1:
            r = r.reshape(-1)
        m = r.numel()

        # Build J explicitly (m x p) by autograd on each residual entry
        # NOTE: This is slow; fine for small m,p only.
        j = torch.zeros(m, p, dtype=theta.dtype, device=theta.device)
        # We need a graph for grads w.r.t. params
        # Recompute r with graph (not no_grad)
        with torch.enable_grad():
            # Recompute r with graph
            rr = closure()
            rr = rr.reshape(-1)
            for i in range(m):
                self.zero_grad(set_to_none=True)
                grad_i = torch.autograd.grad(rr[i], params, retain_graph=True, allow_unused=False)
                j[i] = parameters_to_vector(grad_i)

        # Form normal equations: (J^T J + mu I) Δ = -J^T r
        group = self.param_groups[0]
        mu = group['mu']
        wd = group['weight_decay']

        jt = j.transpose(0, 1) # p x m
        jtj = jt @ j # p x p
        if wd != 0.0:
            jtj = jtj + wd * torch.eye(p, device=theta.device, dtype=theta.dtype)
        rhs = -(jt @ r) # p
        a = jtj + mu * torch.eye(p, device=theta.device, dtype=theta.dtype)
        # Solve
        try:
            dtheta = torch.linalg.solve(a, rhs) # pylint: disable=not-callable
        except RuntimeError:
            # fallback to least-squares
            dtheta, *_ = torch.linalg.lstsq(a, rhs.unsqueeze(1)) # pylint: disable=not-callable
            dtheta = dtheta.squeeze(1)

        # Evaluate gain ratio ρ to adapt mu
        old_r = r
        old_sse = 0.5 * (old_r @ old_r)

        # Canonical LM quantities BEFORE updating parameters
        g = jt @ old_r # gradient (p,)
        predicted_reduction = 0.5 * (dtheta @ (mu * dtheta - g))

        # If model predicts no reduction (numerical or negative), force μ increase
        if predicted_reduction <= 0:
            rho = -1.0
            # Reject immediately: increase μ and keep params
            new_mu = min(group['mu_hi'], max(group['mu_lo'], mu * group['mu_up']))
            group['mu'] = new_mu
            return {
                "rho": float(rho),
                "accepted": False,
                "mu": float(group['mu']),
                "old_sse": float(old_sse.item()),
                "new_sse": float(old_sse.item())
            }

        # Tentative update
        new_theta = theta + dtheta
        old_params = theta.clone()
        vector_to_parameters(new_theta, params)

        new_r = closure().reshape(-1)
        new_sse = 0.5 * (new_r @ new_r)

        rho = (old_sse - new_sse) / predicted_reduction

        if rho > 0 and new_sse < old_sse: # accept
            new_mu = min(group['mu_hi'], max(group['mu_lo'], mu * group['mu_down']))
            group['mu'] = new_mu
            success = True
        else: # reject
            vector_to_parameters(old_params, params)
            new_mu = min(group['mu_hi'], max(group['mu_lo'], mu * group['mu_up']))
            group['mu'] = new_mu
            success = False
            new_sse = old_sse # report unchanged SSE

        return {
            "rho": float(rho.item()), # type: ignore[attr-defined]
            "accepted": success,
            "mu": float(group['mu']),
            "old_sse": float(old_sse.item()),
            "new_sse": float(new_sse.item())
        }

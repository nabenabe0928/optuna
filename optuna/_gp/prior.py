from __future__ import annotations

import math
from typing import TYPE_CHECKING

from optuna._gp import gp


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


DEFAULT_MINIMUM_NOISE_VAR = 1e-6


def default_log_prior(kernel_params: "gp.KernelParamsTensor") -> "torch.Tensor":
    # Log of prior distribution of kernel parameters.
    dim = len(kernel_params.inverse_squared_lengthscales)
    sigma = math.sqrt(3)
    mu = math.sqrt(2) + math.log(dim)

    def log_gamma_prior(x: "torch.Tensor", concentration: float, rate: float) -> "torch.Tensor":
        # We omit the constant factor `rate ** concentration / Gamma(concentration)`.
        return (concentration - 1) * torch.log(x) - rate * x
    
    def log_normal_inv_squared_prior(x: "torch.Tensor") -> "torch.Tensor":
        # Suppose the lengthscales Y follows the log normal distribution, i.e.,
        # log(Y) ~ N(mu, sigma^2), the inverse lengthscales X=1/Y**2 follows
        # log(X)=-2log(Y) ~ N(-2*mu, 4sigma^2).
        # The reference below uses LN(sqrt(2) + log sqrt(D), 3).
        # https://github.com/pytorch/botorch/discussions/2451
        log_x = torch.log(x)
        first_term = -0.5 * math.log(2 * math.pi) - log_x - math.log(2 * sigma)
        second_term = - (log_x + 2 * mu)**2 / (8.0 * sigma**2)
        return first_term + second_term

    # NOTE(contramundum53): The priors below (params and function
    # shape for inverse_squared_lengthscales) were picked by heuristics.
    # TODO(contramundum53): Check whether these priors are appropriate.
    return (
        log_normal_inv_squared_prior(kernel_params.inverse_squared_lengthscales).sum()
        + log_gamma_prior(kernel_params.kernel_scale, 2, 1)
        + log_gamma_prior(kernel_params.noise_var, 1.1, 30)
    )

"""Variable Definitions in Gaussian Process Regressor.

* X: The normalized observed parameter values. The shape is (len(trials), len(params)).
* Y: The standardized observed objective values. The shape is (len(trials), ).
     Note that Y is modified so that larger Y is better.
* x: The parameter values to be evaluated. Possibly batched.
     The shape is (len(params), ) or (batch_size, len(params)).
* cov_fX_fX: The kernel matrix V[f(X)] of X. The shape is (len(trials), len(trials)).
* cov_fx_fX: The kernel vector Cov[f(x), f(X)] of x and X.
             The shape is (len(trials), ) or (batch_size, len(trials)).
* cov_fx_fx: The kernel value of x = V[f(x)]. Since we use a Matern 5/2 kernel,
             we assume this value to be a constant.
* cov_Y_Y_inv: The inverse of the covariance matrix of Y = (V[f(X) + noise])^-1.
               The shape is (len(trials), len(trials)).
* cov_Y_Y_inv_Y: cov_Y_Y_inv @ Y. The shape is (len(trials), ).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import typing
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

from optuna.logging import get_logger


if TYPE_CHECKING:
    import scipy.optimize as so
    import torch
else:
    from optuna._imports import _LazyImport

    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")

logger = get_logger(__name__)


@dataclass(frozen=True)
class Matern52Kernel(torch.autograd.Function):
    is_categorical: np.ndarray
    # Kernel parameters to fit.
    inverse_squared_lengthscales: torch.Tensor  # (len(params), )
    scale: torch.Tensor  # Scalar
    noise_var: torch.Tensor  # Scalar

    @staticmethod
    def forward(ctx: typing.Any, squared_distance: torch.Tensor) -> torch.Tensor:  # type: ignore
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        # Notice that the derivative is taken w.r.t. d^2, but not w.r.t. d.
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: typing.Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function,
        # then deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        (deriv,) = ctx.saved_tensors
        return deriv * grad

    def compute(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        PyTorch cannot differentiate the expression below:
            exp(sqrt5d) * (5/3 * squared_distance + sqrt(squared_distance) + 1),
        because its gradient runs into 0/0 at squared_distance=0.
        """
        squared_diffs = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
        # Use the Hamming distance for categorical parameters.
        squared_diffs[..., self.is_categorical] = (
            squared_diffs[..., self.is_categorical] > 0.0
        ).type(torch.float64)
        squared_distance = (squared_diffs * self.inverse_squared_lengthscales).sum(dim=-1)
        return self.scale * self.apply(squared_distance)  # type: ignore


def marginal_log_likelihood(
    kernel: Matern52Kernel, X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:  # Scaler
    # -0.5 * log((2pi)^n |C|) - 0.5 * Y^T C^-1 Y, where C^-1 = cov_Y_Y_inv
    # We apply the cholesky decomposition to efficiently compute log(|C|) and C^-1.

    cov_fX_fX = kernel.compute(X, X)
    cov_Y_Y_chol = torch.linalg.cholesky(
        cov_fX_fX + kernel.noise_var * torch.eye(X.shape[0], dtype=torch.float64)
    )
    # log |L| = 0.5 * log|L^T L| = 0.5 * log|C|
    logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
    # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = Y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ Y
    cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[:, 0]
    # Y^T C^-1 Y = Y^T inv(L^T L) Y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
    return -0.5 * (
        logdet + X.shape[0] * math.log(2 * math.pi) + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
    )


def _fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[Matern52Kernel], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    initial_kernel: Matern52Kernel,
    gtol: float,
) -> Matern52Kernel:
    n_params = X.shape[1]

    # We apply log transform to enforce the positivity of the kernel parameters.
    # Note that we cannot just use the constraint because of the numerical unstability
    # of the marginal log likelihood.
    # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
    # pathological behavior of maximum likelihood estimation.
    initial_raw_params = np.concatenate(
        [
            np.log(initial_kernel.inverse_squared_lengthscales.detach().numpy()),
            [
                np.log(initial_kernel.scale.item()),
                # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
                np.log(initial_kernel.noise_var.item() - 0.99 * minimum_noise),
            ],
        ]
    )

    def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
        raw_params_tensor = torch.from_numpy(raw_params)
        raw_params_tensor.requires_grad_(True)
        kernel = Matern52Kernel(
            inverse_squared_lengthscales=torch.exp(raw_params_tensor[:n_params]),
            scale=torch.exp(raw_params_tensor[n_params]),
            noise_var=(
                torch.tensor(minimum_noise, dtype=torch.float64)
                if deterministic_objective
                else torch.exp(raw_params_tensor[n_params + 1]) + minimum_noise
            ),
            is_categorical=is_categorical,
        )
        loss = -log_prior(kernel) - marginal_log_likelihood(
            kernel, torch.from_numpy(X), torch.from_numpy(Y)
        )
        loss.backward()  # type: ignore
        # scipy.minimize requires all the gradients to be zero for termination.
        raw_noise_var_grad = raw_params_tensor.grad[n_params + 1]  # type: ignore
        assert not deterministic_objective or raw_noise_var_grad == 0
        return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

    # jac=True means loss_func returns the gradient for gradient descent.
    res = so.minimize(
        # Too small `gtol` causes instability in loss_func optimization.
        loss_func,
        initial_raw_params,
        jac=True,
        method="l-bfgs-b",
        options={"gtol": gtol},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    raw_params_opt_tensor = torch.from_numpy(res.x)

    res = Matern52Kernel(
        inverse_squared_lengthscales=torch.exp(raw_params_opt_tensor[:n_params]),
        scale=torch.exp(raw_params_opt_tensor[n_params]),
        noise_var=(
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else minimum_noise + torch.exp(raw_params_opt_tensor[n_params + 1])
        ),
        is_categorical=is_categorical,
    )
    return res


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[Matern52Kernel], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    initial_kernel: Matern52Kernel | None = None,
    gtol: float = 1e-2,
) -> Matern52Kernel:
    default_initial_kernel = Matern52Kernel(
        inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
        scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(1.0, dtype=torch.float64),
        is_categorical=is_categorical,
    )
    if initial_kernel is None:
        initial_kernel = default_initial_kernel

    error = None
    # First try optimizing the kernel params with the provided initial_kernel,
    # but if it fails, rerun the optimization with the default initial_kernel.
    # This increases the robustness of the optimization.
    for init_kernel in [initial_kernel, default_initial_kernel]:
        try:
            return _fit_kernel_params(
                X=X,
                Y=Y,
                is_categorical=is_categorical,
                log_prior=log_prior,
                minimum_noise=minimum_noise,
                initial_kernel=init_kernel,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as e:
            error = e

    logger.warn(
        f"The optimization of kernel_params failed: \n{error}\n"
        "The default initial kernel params will be used instead."
    )
    return default_initial_kernel

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


class Matern52Kernel(torch.autograd.Function):
    def __init__(
        self,
        is_categorical: np.ndarray,
        inverse_squared_lengthscales: torch.Tensor | None = None,  # (len(params), )
        scale: torch.Tensor | None = None,  # Scalar
        noise_var: torch.Tensor | None = None,  # Scalar
    ):
        n_params = is_categorical.size
        self.is_categorical = is_categorical
        self.inverse_squared_lengthscales = (
            torch.ones(n_params, dtype=torch.float64)
            if inverse_squared_lengthscales is None
            else inverse_squared_lengthscales
        )
        self.scale = torch.tensor(1.0, dtype=torch.float64) if scale is None else scale
        self.noise_var = torch.tensor(1.0, dtype=torch.float64) if noise_var is None else noise_var

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

    @property
    def params_array(self) -> np.ndarray:
        # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
        inv_squared_lscales = self.inverse_squared_lengthscales.detach().numpy()
        return np.concatenate([inv_squared_lscales, [self.scale.item(), self.noise_var.item()]])

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


def _revert_raw_params(
    raw_params_tensor: torch.Tensor, n_params: int, min_noise: float, deterministic: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inv_squared_lscales = torch.exp(raw_params_tensor[:n_params])
    scale = torch.exp(raw_params_tensor[n_params])
    noise_var = torch.tensor(min_noise, dtype=torch.float64)
    noise_var += 0.0 if deterministic else torch.exp(raw_params_tensor[n_params + 1])
    return inv_squared_lscales, scale, noise_var


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

    initial_raw_params = initial_kernel.params_array
    # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
    initial_raw_params[-1] -= 0.99 * minimum_noise
    # Log transformation to enforce the positivity of the kernel parameters.
    initial_raw_params = np.log(initial_raw_params)

    def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
        raw_params_tensor = torch.from_numpy(raw_params)
        raw_params_tensor.requires_grad_(True)
        inv_squared_lscales, scale, noise_var = _revert_raw_params(
            raw_params_tensor, n_params, minimum_noise, deterministic_objective
        )
        kernel = Matern52Kernel(is_categorical, inv_squared_lscales, scale, noise_var)
        loss = -log_prior(kernel) - marginal_log_likelihood(
            kernel, torch.from_numpy(X), torch.from_numpy(Y)
        )
        loss.backward()  # type: ignore
        # scipy.minimize requires all the gradients to be zero for termination.
        raw_noise_var_grad = raw_params_tensor.grad[n_params + 1]  # type: ignore
        assert not deterministic_objective or raw_noise_var_grad == 0
        return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

    # jac=True means loss_func returns the gradient for gradient descent.
    # Too small `gtol` causes instability in loss_func optimization.
    res = so.minimize(
        loss_func, initial_raw_params, jac=True, method="l-bfgs-b", options={"gtol": gtol}
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    raw_params_opt_tensor = torch.from_numpy(res.x)
    inv_squared_lscales, scale, noise_var = _revert_raw_params(
        raw_params_opt_tensor, n_params, minimum_noise, deterministic_objective
    )
    return Matern52Kernel(is_categorical, inv_squared_lscales, scale, noise_var)


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
    default_initial_kernel = Matern52Kernel(is_categorical=is_categorical)
    if initial_kernel is None:
        initial_kernel = default_initial_kernel

    error = None
    # First try optimizing the provided kernel params. If it fails, try the default initial_kernel.
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

"""This GP implementation uses the following notation:

X: Observed parameter values with the shape of (len(trials), len(params)).
Y: Observed objective values with the shape of (len(trials), ).
x: Parameter value(s) to evaluate with the shape of (..., len(params)).
cov_fX_fX: A kernel matrix V[f(X)] with the shape of (len(trials), len(trials)).
cov_fx_fX: A kernel vector Cov[f(x), f(X)] with the shape of (..., len(trials)).
cov_fx_fx: A kernel value (scalar) V[f(x)]. As we use Matern 5/2 kernel, this value is constant.
cov_Y_Y_inv: inverse of V[f(X) + noise].
cov_Y_Y_inv_Y[len(trials)]: cov_Y_Y_inv @ Y.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.logging import get_logger


if TYPE_CHECKING:
    import scipy.optimize as so
    import torch
else:
    from optuna._imports import _LazyImport

    prior = _LazyImport("optuna._gp.prior")
    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")

logger = get_logger(__name__)


class DifferentiableMatern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:  # type: ignore
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        # Notice that the derivative is taken w.r.t. d^2, but not w.r.t. d.
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function,
        # then deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        (deriv,) = ctx.saved_tensors
        return deriv * grad


class Matern52Kernel:
    def __init__(
        self,
        is_categorical: torch.Tensor,
        inverse_squared_lengthscales: torch.Tensor | None = None,  # (len(params), )
        kernel_scale: torch.Tensor | None = None,  # Scalar
        noise_var: torch.Tensor | None = None,  # Scalar
    ) -> None:
        self.n_params = len(is_categorical)
        self.is_categorical = is_categorical
        self.inverse_squared_lengthscales = (
            inverse_squared_lengthscales or torch.ones(self.n_params, dtype=torch.float64)
        )
        self.kernel_scale = kernel_scale or torch.tensor(1.0, dtype=torch.float64)
        self.noise_var = noise_var or torch.tensor(1.0, dtype=torch.float64)

    def __call__(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        # d2(x1, x2) = sum_i d2_i(x1_i, x2_i)
        # d2_i(x1_i, x2_i) = (x1_i - x2_i) ** 2  # if x_i is continuous
        # d2_i(x1_i, x2_i) = 1 if x1_i != x2_i else 0  # if x_i is categorical

        d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
        # Use the Hamming distance for categorical parameters.
        d2[..., self.is_categorical] = (d2[..., self.is_categorical] > 0.0).type(torch.float64)
        d2 = (d2 * self.inverse_squared_lengthscales).sum(dim=-1)
        # sqrt5d = sqrt(5 * d2) where d2 is squared distance.
        # exp(sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)
        # We cannot let PyTorch differentiate the above expression because
        # the gradient runs into 0/0 at squared_distance=0.
        return self.kernel_scale * DifferentiableMatern52Kernel.apply(d2)  # type: ignore

    def _marginal_log_likelihood(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # -0.5 * log((2pi)^n |C|) - 0.5 * Y^T C^-1 Y, where C^-1 = cov_Y_Y_inv
        # We apply the cholesky decomposition to efficiently compute log(|C|) and C^-1.
        cov_fX_fX = self(X, X)
        cov_Y_Y_chol = torch.linalg.cholesky(
            cov_fX_fX + self.noise_var * torch.eye(X.shape[0], dtype=torch.float64)
        )
        # log |L| = 0.5 * log|L^T L| = 0.5 * log|C|
        logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
        # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = Y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ Y
        cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(
            cov_Y_Y_chol, Y[:, None], upper=False
        )[:, 0]
        # The return is a scalar value.
        return -0.5 * (
            logdet
            + X.shape[0] * math.log(2 * math.pi)
            # Y^T C^-1 Y = Y^T inv(L^T L) Y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
            + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
        )

    def _update_with_raw_params(
        self, raw_params: torch.Tensor, deterministic_objective: bool, minimum_noise: float
    ) -> None:
        self.inverse_squared_lengthscales = torch.exp(raw_params[:self.n_params])
        self.kernel_scale = torch.exp(raw_params[self.n_params])
        self.noise_var = (
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else torch.exp(raw_params[self.n_params + 1]) + minimum_noise
        )

    def _get_raw_params(self, minimum_noise: float) -> np.ndarray:
        # We apply log transform to enforce the positivity of the kernel parameters.
        # Note that we cannot just use the constraint because of the numerical unstability
        # of the marginal log likelihood.
        # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
        # pathological behavior of maximum likelihood estimation.
        return np.concatenate(
            [
                np.log(self.inverse_squared_lengthscales.detach().numpy()),
                [np.log(self.kernel_scale.item())],
                # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
                [np.log(self.noise_var.item() - 0.99 * minimum_noise)],
            ]
        )

    def fit(
        self,
        X: np.ndarray,  # (len(trials), len(params))
        Y: np.ndarray,  # (len(trials), )
        log_prior: Callable[[KernelParamsTensor], torch.Tensor],
        deterministic_objective: bool,
        minimum_noise: float | None = None,
        gtol: float = 1e-2,
    ) -> None:
        minimum_noise = minimum_noise or prior.DEFAULT_MINIMUM_NOISE_VAR
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)

        def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
            raw_params_tensor = torch.from_numpy(raw_params)
            raw_params_tensor.requires_grad_(True)
            with torch.enable_grad():
                self._update_with_raw_params(
                    raw_params_tensor, deterministic_objective, minimum_noise
                )
                loss = -self._marginal_log_likelihood(X_tensor, Y_tensor) - log_prior(self)
                loss.backward()  # type: ignore
                # scipy.minimize requires all the gradients to be zero for termination.
                raw_noise_var_grad = raw_params_tensor.grad[self.n_params + 1]  # type: ignore
                assert not deterministic_objective or raw_noise_var_grad == 0
            return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

        # jac=True means loss_func returns the gradient for gradient descent.
        # Too small `gtol` causes instability in loss_func optimization.
        res = so.minimize(
            loss_func,
            self._get_raw_params(minimum_noise),
            jac=True,
            method="l-bfgs-b",
            options={"gtol": gtol},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self._update_with_raw_params(
            torch.from_numpy(res.x), deterministic_objective, minimum_noise
        )


def posterior(
    kernel: Matern52Kernel,
    X: torch.Tensor,  # [len(trials), len(params)]
    cov_Y_Y_inv: torch.Tensor,  # [len(trials), len(trials)]
    cov_Y_Y_inv_Y: torch.Tensor,  # [len(trials)]
    x: torch.Tensor,  # [(batch,) len(params)]
) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: [(batch,)], var: [(batch,)])
    cov_fx_fX = kernel(x[..., None, :], X)[..., 0, :]
    cov_fx_fx = kernel.kernel_scale

    # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
    # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T
    mean = cov_fx_fX @ cov_Y_Y_inv_Y  # [batch]
    var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ cov_Y_Y_inv)).sum(dim=-1)  # [batch]
    # We need to clamp the variance to avoid negative values due to numerical errors.
    return (mean, torch.clamp(var, min=0.0))


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[KernelParamsTensor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    cached_kernel: Matern52Kernel | None = None,
    gtol: float = 1e-2,
) -> KernelParamsTensor:
    cached_kernel = cached_kernel or Matern52Kernel(is_categorical)
    error = None
    # First try optimizing the kernel params with the provided initial_kernel_params,
    # but if it fails, rerun the optimization with the default initial_kernel_params.
    # This increases the robustness of the optimization.
    for kernel in [cached_kernel, Matern52Kernel(is_categorical, minimum_noise=minimum_noise)]:
        try:
            kernel.fit(
                X=X,
                Y=Y,
                log_prior=log_prior,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as e:
            error = e

    logger.warning(
        f"The optimization of kernel_params failed: \n{error}\n"
        "The default initial kernel params will be used instead."
    )
    return default_initial_kernel_params


if __name__ == "__main__":
    from optuna._gp.gp import fit_kernel_params
    rng = np.random.RandomState(42)
    n_params = 2
    n_trials = 30
    X = rng.random((n_trials, n_params)) * 10 - 5
    Y = np.sum(X**2, axis=-1)
    kernel = Matern52Kernel(np.array([False]*n_params))
    kernel.fit(
        X=(X + 5) / 10,
        Y=(Y - np.mean(Y)) / np.std(Y),
        log_prior=prior.default_log_prior,
        deterministic_objective=False,
    )

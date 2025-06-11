"""Notations in this Gaussian process implementation

X: Observed parameter values with the shape of (len(trials), len(params)).
Y: Observed objective values with the shape of (len(trials), ).
x: (Possibly batched) parameter value(s) to evaluate with the shape of (..., len(params)).
cov_fX_fX: Kernel matrix X = V[f(X)] with the shape of (len(trials), len(trials)).
cov_fx_fX: Kernel matrix Cov[f(x), f(X)] with the shape of (..., len(trials)).
cov_fx_fx: Kernel scalar value x = V[f(x)]. This value is constant for the Matern 5/2 kernel.
cov_Y_Y_inv:
    The inverse of the covariance matrix (V[f(X) + noise])^-1 with the shape of
    (len(trials), len(trials)).
cov_Y_Y_inv_Y: `cov_Y_Y_inv @ Y` with the shape of (len(trials), ).
max_Y: The maximum of Y (Note that we transform the objective values such that it is maximized.)
d2: The squared distance between two points.
is_categorical:
    A boolean array with the shape of (len(params), ). If is_categorical[i] is True, the i-th
    parameter is categorical.
"""

from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    import scipy.optimize as so
    import torch
else:
    from optuna._imports import _LazyImport

    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")

logger = get_logger(__name__)


def warn_and_convert_inf(values: np.ndarray) -> np.ndarray:
    is_values_finite = np.isfinite(values)
    if np.all(is_values_finite):
        return values

    warnings.warn("Clip non-finite values to the min/max finite values for GP fittings.")
    is_any_finite = np.any(is_values_finite, axis=0)
    # NOTE(nabenabe): values cannot include nan to apply np.clip properly, but Optuna anyways won't
    # pass nan in values by design.
    return np.clip(
        values,
        np.where(is_any_finite, np.min(np.where(is_values_finite, values, np.inf), axis=0), 0.0),
        np.where(is_any_finite, np.max(np.where(is_values_finite, values, -np.inf), axis=0), 0.0),
    )


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:
        """
        This method calculates `exp(sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)` where
        `sqrt5d = sqrt(5 * squared_distance)`.

        Please note that automatic differentiation by PyTorch does not work well at
        `squared_distance = 0` due to zero division, so we manually save the derivative, i.e.,
        `-5/6 * (1 + sqrt5d) * exp(-sqrt5d)`, for the exact derivative calculation.
        """
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        # Notice that the derivative is taken w.r.t. d^2, but not w.r.t. d.
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        # Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function,
        # then deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        (deriv,) = ctx.saved_tensors
        return deriv * grad


class GaussianProcessRegressor:
    def __init__(
        self,
        *,
        is_categorical: torch.Tensor,
        inverse_squared_lengthscales: torch.Tensor | None = None,
        kernel_scale: torch.Tensor | None = None,
        noise_var: torch.Tensor | None = None,
    ) -> None:
        self._is_categorical = is_categorical
        self._n_params = len(is_categorical)
        self._inverse_squared_lengthscales: torch.Tensor = (
            torch.ones(self._n_params, dtype=torch.float64)
            if inverse_squared_lengthscales is None
            else inverse_squared_lengthscales
        )
        self._kernel_scale = (
            torch.tensor(1.0, dtype=torch.float64) if kernel_scale is None else kernel_scale
        )
        self._noise_var = (
            torch.tensor(1.0, dtype=torch.float64) if noise_var is None else noise_var
        )

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Return the kernel matrix with the shape of (..., n_A, n_B) given X1 and X2 each with the
        shapes of (..., n_A, len(params)) and (..., n_B, len(params)).

        If x1 and x2 have the shape of (len(params), ), kernel(x1, x2) is computed as:
            kernel_scale * Matern52Kernel.apply(
                d2(x1, x2) @ inverse_squared_lengthscales
            )
        where if x1[i] is continuous, d2(x1, x2)[i] = (x1[i] - x2[i]) ** 2 and if x1[i] is
        categorical, d2(x1, x2)[i] = int(x1[i] != x2[i]).
        Note that the distance for categorical parameters is the Hamming distance.
        """
        d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
        d2[..., is_categorical] = (d2[..., is_categorical] > 0.0).type(torch.float64)
        d2 = (d2 * self._inverse_squared_lengthscales).sum(dim=-1)
        return Matern52Kernel.apply(d2) * self._kernel_scale  # type: ignore

    def posterior(
        self,
        X: torch.Tensor,
        # TODO(nabenabe): Make them instance attributes.
        cov_Y_Y_inv: torch.Tensor,
        cov_Y_Y_inv_Y: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: (...,), var: (...,))
        cov_fx_fX = self._kernel(x[..., None, :], X)[..., 0, :]
        cov_fx_fx = self._kernel_scale  # kernel(x, x) = kernel_scale

        # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
        # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T
        # The shape of both mean and var is (..., ).
        mean = cov_fx_fX @ cov_Y_Y_inv_Y
        var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ cov_Y_Y_inv)).sum(dim=-1)
        # We need to clamp the variance to avoid negative values due to numerical errors.
        return mean, torch.clamp(var, min=0.0)

    def _marginal_log_likelihood(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:  # Scalar
        # -0.5 * log((2pi)^n |C|) - 0.5 * Y^T C^-1 Y, where C^-1 = cov_Y_Y_inv
        # We apply the cholesky decomposition to efficiently compute log(|C|) and C^-1.
        cov_fX_fX = kernel(X, X)
        cov_Y_Y_chol = torch.linalg.cholesky(
            cov_fX_fX + self._noise_var * torch.eye(X.shape[0], dtype=torch.float64)
        )
        # log |L| = 0.5 * log|L^T L| = 0.5 * log|C|
        logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
        # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = Y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ Y
        cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[
            :, 0
        ]
        return -0.5 * (
            logdet
            + X.shape[0] * math.log(2 * math.pi)
            # Y^T C^-1 Y = Y^T inv(L^T L) Y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
            + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
        )

    def _disable_grad(self) -> None:
        self._inverse_squared_lengthscales = self._inverse_squared_lengthscales.detach()
        self._inverse_squared_lengthscales.grad = None
        self._kernel_scale = self._kernel_scale.detach()
        self._kernel_scale.grad = None
        self._noise_var = self._noise_var.detach()
        self._noise_var.grad = None

    def _set_kernel_params_from_raw_params_tensor(
        self, raw_params_tensor: torch.Tensor, minimum_noise: float, deterministic_objective: bool
    ) -> None:
        exp_raw_params = torch.exp(raw_params_tensor)
        self._inverse_squared_lengthscales = exp_raw_params[:-2]
        self._kernel_scale = exp_raw_params[-2]
        self._noise_var = (
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else exp_raw_params[-1] + minimum_noise
        )

    def _loss_func(
        self,
        raw_params: np.ndarray,
        log_prior: Callable[["GaussianProcessRegressor"], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
    ) -> tuple[float, np.ndarray]:
        raw_params_tensor = torch.from_numpy(raw_params).requires_grad_(True)
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            self._set_kernel_params_from_raw_params_tensor(
                raw_params_tensor, minimum_noise, deterministic_objective
            )
            loss = -self._marginal_log_likelihood(
                torch.from_numpy(X), torch.from_numpy(Y)
            ) - log_prior(self)
            loss.backward()  # type: ignore
            # scipy.minimize requires all the gradients to be zero for termination.
            grad = raw_params_tensor.grad.detach().numpy()
            raw_noise_var_grad = grad[self._n_params + 1]  # type: ignore
            assert not deterministic_objective or raw_noise_var_grad == 0
        self._disable_grad()
        return loss.item(), grad  # type: ignore

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        log_prior: Callable[["GaussianProcessRegressor"], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float,
    ) -> None:
        raw_params = np.empty(self._n_params + 2, dtype=float)
        raw_params[:-2] = self._inverse_squared_lengthscales.detach().numpy()
        raw_params[-2] = self._kernel_scale.item()
        # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
        raw_params[-1] = self._noise_var.item() - 0.99 * minimum_noise
        # We apply log transform to enforce the positivity of the kernel parameters.
        # Note that we cannot just use the constraint because of the numerical unstability
        # of the marginal log likelihood.
        # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
        # pathological behavior of maximum likelihood estimation.
        raw_params = np.log(raw_params)
        # jac=True means loss_func returns the gradient for gradient descent.
        res = so.minimize(
            # Too small `gtol` causes instability in loss_func optimization.
            lambda x: self._loss_func(x, log_prior, minimum_noise, deterministic_objective),
            x0=self._get_raw_params(),
            jac=True,
            method="l-bfgs-b",
            options={"gtol": gtol},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self._set_kernel_params_from_raw_params_tensor(
            torch.from_numpy(res.x), minimum_noise, deterministic_objective
        )


def fit_gp_regressor(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[GaussianProcessRegressor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    last_gp_regressor: GaussianProcessRegressor | None = None,
    gtol: float = 1e-2,
) -> GaussianProcessRegressor:
    error = None
    # First try optimizing the kernel params with the provided initial_kernel_params,
    # but if it fails, rerun the optimization with the default initial_kernel_params.
    # This increases the robustness of the optimization.
    for old_gpr in [last_gp_regressor, None]:
        try:
            new_gpr = GaussianProcessRegressor(
                is_categorical=torch.from_numpy(is_categorical),
                inverse_squared_lengthscales=(
                    None if old_gpr is None else old_gpr._inverse_squared_lengthscales
                ),
                kernel_scale=None if old_gpr is None else old_gpr._kernel_scale,
                noise_var=None if old_gpr is None else old_gpr._noise_var,
            )
            new_gpr.fit(X, Y, log_prior, minimum_noise, deterministic_objective, gtol)
            return new_gpr
        except RuntimeError as e:
            error = e

    logger.warning(
        f"The optimization of kernel_params failed: \n{error}\n"
        "The default initial kernel params will be used instead."
    )
    return GaussianProcessRegressor(is_categorical=torch.from_numpy(is_categorical))

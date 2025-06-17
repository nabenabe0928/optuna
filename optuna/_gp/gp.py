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
        This method calculates `exp(-sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)` where
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


class GPRegressor:
    def __init__(
        self,
        is_categorical: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        kernel_params: torch.Tensor | None = None,
    ) -> None:
        default_kernel_params = torch.ones(X_train.shape[-1] + 2, dtype=torch.float64)
        assert (
            kernel_params is None
            or len(X_train) == 0
            or kernel_params.shape == default_kernel_params.shape
        )
        self._is_categorical = is_categorical
        self._X_train = X_train
        self._y_train = y_train
        self._cov_Y_Y_inv: torch.Tensor | None = None
        self._cov_Y_Y_inv_Y: torch.Tensor | None = None
        # TODO(nabenabe): Separate kernel params class if necessary.
        self._kernel_params = default_kernel_params if kernel_params is None else kernel_params

    @staticmethod
    def _from_raw_params(
        raw_params: torch.Tensor, minimum_noise: float, deterministic_objective: bool
    ) -> torch.Tensor:
        min_noise_t = torch.tensor([minimum_noise], dtype=torch.float64)
        noise_var = min_noise_t if deterministic_objective else raw_params[-1].exp() + min_noise_t
        return torch.concat([raw_params[:-1].exp(), noise_var])

    def _to_raw_params(self, minimum_noise: float) -> np.ndarray:
        """
        We apply log transform to enforce the positivity of the kernel parameters. Note that we
        cannot just use the constraint because of the numerical unstability of the marginal
        log likelihood. We also enforce the noise parameter to be greater than `minimum_noise` to
        avoid pathological behavior of maximum likelihood estimation.
        """
        exp_raw_params = self._kernel_params.detach().numpy()
        # We add 0.01 * minimum_noise to noise_var to avoid instability.
        exp_raw_params[-1] -= 0.99 * minimum_noise
        return np.log(exp_raw_params)

    @property
    def _inverse_squared_lengthscales(self) -> torch.Tensor:
        return self._kernel_params[:-2]

    @property
    def _kernel_scale(self) -> torch.Tensor:
        return self._kernel_params[-2]

    @property
    def _noise_var(self) -> torch.Tensor:
        return self._kernel_params[-1]

    @property
    def kernel_params(self) -> torch.Tensor:
        return self._kernel_params.clone()

    def _cache_matrix(self) -> None:
        with torch.no_grad():
            cov_Y_Y = self._kernel(self._X_train, self._X_train).detach().numpy()

        cov_Y_Y[np.diag_indices(self._X_train.shape[0])] += self._noise_var.item()
        cov_Y_Y_inv = np.linalg.inv(cov_Y_Y)
        cov_Y_Y_inv_Y = cov_Y_Y_inv @ self._y_train.numpy()
        # NOTE(nabenabe): Here we use NumPy to guarantee the reproducibility from the past.
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv_Y)

    @property
    def length_scales(self) -> np.ndarray:
        return 1.0 / np.sqrt(self._inverse_squared_lengthscales.detach().numpy())

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
        d2[..., self._is_categorical] = (d2[..., self._is_categorical] > 0.0).type(torch.float64)
        d2 = (d2 * self._inverse_squared_lengthscales).sum(dim=-1)
        return Matern52Kernel.apply(d2) * self._kernel_scale  # type: ignore

    def posterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The shape of mean and var is x.shape[:-1].
        assert (
            self._cov_Y_Y_inv is not None and self._cov_Y_Y_inv_Y is not None
        ), "Call cache_matrix before calling posterior."
        cov_fx_fX = self._kernel(x[..., None, :], self._X_train)[..., 0, :]
        cov_fx_fx = self._kernel_scale  # kernel(x, x) = kernel_scale

        # mean = cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ y
        # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ cov_fx_fX.T
        # The shape of both mean and var is (..., ).
        mean = cov_fx_fX @ self._cov_Y_Y_inv_Y
        var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ self._cov_Y_Y_inv)).sum(dim=-1)
        # We need to clamp the variance to avoid negative values due to numerical errors.
        return mean, torch.clamp(var, min=0.0)

    def _marginal_log_likelihood(self) -> torch.Tensor:  # Scalar
        # -0.5 * log((2*pi)**n * det(C)) - 0.5 * y.T @ inv(C) @ y, where inv(C) = cov_Y_Y_inv.
        # We apply the cholesky decomposition to efficiently compute log(det(C)) and inv(C).
        cov_fX_fX = self._kernel(self._X_train, self._X_train)
        n_points = self._X_train.shape[0]
        cov_Y_Y_chol = torch.linalg.cholesky(
            cov_fX_fX + self._noise_var * torch.eye(n_points, dtype=torch.float64)
        )
        # log(det(L)) = 0.5 * log(det(L.T @ L)) = 0.5 * log(det(C))
        logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
        # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ y
        cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(
            cov_Y_Y_chol, self._y_train[:, None], upper=False
        )[:, 0]
        return -0.5 * (
            logdet
            + n_points * math.log(2 * math.pi)
            # y.T @ inv(C) @ y = y.T @ inv(L.T @ L) @ y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
            + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
        )

    def _fit_kernel_params(
        self,
        log_prior: Callable[["GPRegressor"], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float,
    ) -> None:
        def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
            raw_params_tensor = torch.from_numpy(raw_params).requires_grad_(True)
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                self._kernel_params = self._from_raw_params(
                    raw_params_tensor, minimum_noise, deterministic_objective
                )
                # self._update_kernel_params(kernel_params)
                loss = -self._marginal_log_likelihood() - log_prior(self)
                loss.backward()  # type: ignore
                # scipy.minimize requires all the gradients to be zero for termination.
                raw_noise_var_grad = raw_params_tensor.grad[-1]  # type: ignore
                assert not deterministic_objective or raw_noise_var_grad == 0
            return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

        # jac=True means loss_func returns the gradient for gradient descent.
        res = so.minimize(
            # Too small `gtol` causes instability in loss_func optimization.
            loss_func,
            x0=self._to_raw_params(minimum_noise),
            jac=True,
            method="l-bfgs-b",
            options={"gtol": gtol},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self._kernel_params = self._from_raw_params(
            torch.from_numpy(res.x), minimum_noise, deterministic_objective
        )
        self._cache_matrix()

    def fit_kernel_params(
        self,
        log_prior: Callable[["GPRegressor"], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float = 1e-2,
    ) -> "GPRegressor":
        """
        First try optimizing the kernel params with the provided kernel parameters in cache, but if
        it fails, rerun the optimization with the default kernel parameters for the robustness.
        """
        default_kernel_params = torch.ones(self._X_train.shape[1] + 2, dtype=torch.float64)
        error = None
        for _ in range(2):
            try:
                self._fit_kernel_params(log_prior, minimum_noise, deterministic_objective, gtol)
                return self
            except RuntimeError as e:
                error = e
                self._kernel_params = default_kernel_params.clone()

        logger.warning(
            f"The optimization of kernel parameters failed: \n{error}\n"
            "The default initial kernel parameters will be used instead."
        )
        self._kernel_params = default_kernel_params
        self._cache_matrix()
        return self

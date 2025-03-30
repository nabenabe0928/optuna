from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.gp import kernel
from optuna._gp.gp import KernelParamsTensor
from optuna._gp.gp import posterior
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


class BaseAcquisitionFunc(metaclass=ABCMeta):
    def __init__(
        self, X: np.ndarray, search_space: SearchSpace, length_scales: np.ndarray
    ) -> None:
        self._is_categorical = torch.from_numpy(search_space.scale_types == ScaleType.CATEGORICAL)
        self._X = torch.from_numpy(X)
        self.search_space = search_space
        self.length_scales = length_scales

    @abstractmethod
    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_with_grad(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert x.ndim == 1
        x_tensor = torch.from_numpy(x)
        x_tensor.requires_grad_(True)
        val = self._calculate(x_tensor)
        val.backward()  # type: ignore
        return val.item(), x_tensor.grad.detach().numpy()  # type: ignore

    def eval_with_no_grad(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._calculate(torch.from_numpy(x)).detach().numpy(), None


def _calculate_cov_Y_Y_inv(
    kernel_params: KernelParamsTensor, X: np.ndarray, is_categorical: torch.Tensor
) -> np.ndarray:
    X_tensor = torch.from_numpy(X)
    with torch.no_grad():
        cov_Y_Y = kernel(is_categorical, kernel_params, X_tensor, X_tensor).detach().numpy()

    cov_Y_Y[np.diag_indices(X.shape[0])] += kernel_params.noise_var.item()
    cov_Y_Y_inv = np.linalg.inv(cov_Y_Y)
    return cov_Y_Y_inv


class LogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        X: np.ndarray,
        Y: np.ndarray,
        search_space: SearchSpace,
        threshold: float | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        length_scales = 1 / np.sqrt(kernel_params.inverse_squared_lengthscales.detach().numpy())
        super().__init__(X=X, search_space=search_space, length_scales=length_scales)
        cov_Y_Y_inv = _calculate_cov_Y_Y_inv(kernel_params, X, self._is_categorical)
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv @ Y)
        self._kernel_params = kernel_params
        self._threshold = threshold if threshold is not None else np.max(Y)
        self._stabilizing_noise = stabilizing_noise

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        if np.isneginf(self._threshold):
            # If there are no feasible trials, threshold is set to -np.inf.
            # Then we return logEI=0 to ignore the contribution from the objective.
            return torch.tensor(0.0, dtype=torch.float64)

        mean, var = posterior(
            kernel_params=self._kernel_params,
            X=self._X,
            is_categorical=self._is_categorical,
            cov_Y_Y_inv=self._cov_Y_Y_inv,
            cov_Y_Y_inv_Y=self._cov_Y_Y_inv_Y,
            x=x,
        )

        def _standard_logei(z: torch.Tensor) -> torch.Tensor:
            # Return E_{x ~ N(0, 1)}[max(0, x+z)]

            # We switch the implementation depending on the value of z to
            # avoid numerical instability.
            small = z < -25

            vals = torch.empty_like(z)
            # Eq. (9) in ref: https://arxiv.org/pdf/2310.20708.pdf
            # NOTE: We do not use the third condition because ours is good enough.
            z_small = z[small]
            z_normal = z[~small]
            sqrt_2pi = math.sqrt(2 * math.pi)
            # First condition
            cdf = 0.5 * torch.special.erfc(-z_normal * math.sqrt(0.5))
            pdf = torch.exp(-0.5 * z_normal**2) * (1 / sqrt_2pi)
            vals[~small] = torch.log(z_normal * cdf + pdf)
            # Second condition
            r = math.sqrt(0.5 * math.pi) * torch.special.erfcx(-z_small * math.sqrt(0.5))
            vals[small] = -0.5 * z_small**2 + torch.log((z_small * r + 1) * (1 / sqrt_2pi))
            return vals

        # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
        sigma = torch.sqrt(var + self._stabilizing_noise)
        st_val = _standard_logei((mean - self._threshold) / sigma)
        return torch.log(sigma) + st_val


class LogPI(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        X: np.ndarray,
        Y: np.ndarray,
        search_space: SearchSpace,
        threshold: float | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        length_scales = 1 / np.sqrt(kernel_params.inverse_squared_lengthscales.detach().numpy())
        super().__init__(X=X, search_space=search_space, length_scales=length_scales)
        cov_Y_Y_inv = _calculate_cov_Y_Y_inv(kernel_params, X, self._is_categorical)
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv @ Y)
        self._kernel_params = kernel_params
        self._threshold = threshold if threshold is not None else np.max(Y)
        self._stabilizing_noise = stabilizing_noise

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = posterior(
            kernel_params=self._kernel_params,
            X=self._X,
            is_categorical=self._is_categorical,
            cov_Y_Y_inv=self._cov_Y_Y_inv,
            cov_Y_Y_inv_Y=self._cov_Y_Y_inv_Y,
            x=x,
        )
        # Return the integral of N(mean, var) from -inf to f0
        # This is identical to the integral of N(0, 1) from -inf to (f0-mean)/sigma
        # Return E_{y ~ N(mean, var)}[bool(y <= f0)]
        sigma = torch.sqrt(var + self._stabilizing_noise)
        return torch.special.log_ndtr((self._threshold - mean) / sigma)


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        X: np.ndarray,
        Y: np.ndarray,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        length_scales = 1 / np.sqrt(kernel_params.inverse_squared_lengthscales.detach().numpy())
        super().__init__(X=X, search_space=search_space, length_scales=length_scales)
        cov_Y_Y_inv = _calculate_cov_Y_Y_inv(kernel_params, X, self._is_categorical)
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv @ Y)
        self._kernel_params = kernel_params
        self._beta = beta

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = posterior(
            kernel_params=self._kernel_params,
            X=self._X,
            is_categorical=self._is_categorical,
            cov_Y_Y_inv=self._cov_Y_Y_inv,
            cov_Y_Y_inv_Y=self._cov_Y_Y_inv_Y,
            x=x,
        )
        return mean + torch.sqrt(self._beta * var)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        X: np.ndarray,
        Y: np.ndarray,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        length_scales = 1 / np.sqrt(kernel_params.inverse_squared_lengthscales.detach().numpy())
        super().__init__(X=X, search_space=search_space, length_scales=length_scales)
        cov_Y_Y_inv = _calculate_cov_Y_Y_inv(kernel_params, X, self._is_categorical)
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv @ Y)
        self._kernel_params = kernel_params
        self._beta = beta

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = posterior(
            kernel_params=self._kernel_params,
            X=self._X,
            is_categorical=self._is_categorical,
            cov_Y_Y_inv=self._cov_Y_Y_inv,
            cov_Y_Y_inv_Y=self._cov_Y_Y_inv_Y,
            x=x,
        )
        return mean - torch.sqrt(self._beta * var)


class ConstrainedLogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        constraint_kernel_params_list: list[KernelParamsTensor],
        X: np.ndarray,
        constraint_vals: np.ndarray,
        search_space: SearchSpace,
        objective_acqf: LogEI,
        constraint_thresholds: list[float],
        stabilizing_noise: float = 1e-12,
    ) -> None:
        length_scales = 1 / np.sqrt(kernel_params.inverse_squared_lengthscales.detach().numpy())
        super().__init__(X=X, search_space=search_space, length_scales=length_scales)
        assert constraint_vals.shape == (X.shape[0], len(constraint_kernel_params_list))
        self._acqf_list = [objective_acqf] + [
            LogPI(kernel_params, X, c_vals, search_space, threshold, stabilizing_noise)
            for kernel_params, c_vals, threshold in zip(
                constraint_kernel_params_list, constraint_vals.T, constraint_thresholds
            )
        ]

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        return sum(acqf_._calculate(x) for acqf_ in self._acqf_list)

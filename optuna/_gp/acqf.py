from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.gp import Matern52Kernel
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


def standard_logei(z: torch.Tensor) -> torch.Tensor:
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


class BaseAcquisitionFunc(metaclass=ABCMeta):
    def __init__(
        self, kernel: Matern52Kernel, search_space: SearchSpace, X: np.ndarray, Y: np.ndarray
    ):
        self.search_space = search_space
        self._X = torch.from_numpy(X)
        self._kernel = kernel
        self._is_categorical = torch.from_numpy(search_space.scale_types == ScaleType.CATEGORICAL)
        with torch.no_grad():
            cov_Y_Y = self._kernel.compute(self._X, self._X).detach().numpy()
            cov_Y_Y[np.diag_indices(X.shape[0])] += self._kernel.noise_var.item()

        cov_Y_Y_inv = np.linalg.inv(cov_Y_Y)
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv @ Y)

    def _gp_poterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cov_fx_fX = self._kernel.compute(x[..., None, :], self._X)[..., 0, :]
        cov_fx_fx = self._kernel.scale

        # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
        mean = cov_fx_fX @ self._cov_Y_Y_inv_Y
        # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T
        var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ self._cov_Y_Y_inv)).sum(dim=-1)
        # We need to clamp the variance to avoid negative values due to numerical errors.
        return (mean, torch.clamp(var, min=0.0))

    @abstractmethod
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_with_no_grad(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.eval(torch.from_numpy(x)).detach().numpy()

    def eval_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        assert x.ndim == 1
        x_tensor = torch.from_numpy(x)
        x_tensor.requires_grad_(True)
        val = self.eval(x_tensor)
        val.backward()  # type: ignore
        return val.item(), x_tensor.grad.detach().numpy()  # type: ignore


class LogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel: Matern52Kernel,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        stabilizing_noise: float = 1e-12,
    ):
        super().__init__(kernel=kernel, search_space=search_space, X=X, Y=Y)
        self._max_Y = np.max(Y)
        self._stabilizing_noise = stabilizing_noise

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
        mean, variance = self._gp_poterior(x)
        sigma = torch.sqrt(variance + self._stabilizing_noise)
        st_val = standard_logei((mean - self._max_Y) / sigma)
        val = torch.log(sigma) + st_val
        return val


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel: Matern52Kernel,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        beta: float,
    ):
        assert beta >= 0, "beta must be non-negative."
        super().__init__(kernel=kernel, search_space=search_space, X=X, Y=Y)
        self._beta = beta

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        mean, variance = self._gp_poterior(x)
        return mean + torch.sqrt(self._beta * variance)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel: Matern52Kernel,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        beta: float,
    ):
        assert beta >= 0, "beta must be non-negative."
        super().__init__(kernel=kernel, search_space=search_space, X=X, Y=Y)
        self._beta = beta

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        mean, variance = self._gp_poterior(x)
        return mean - torch.sqrt(self._beta * variance)

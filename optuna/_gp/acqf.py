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
from optuna._hypervolume import get_non_dominated_hyper_rectangle_bounds
from optuna.study._multi_objective import _is_pareto_front


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


def _sample_from_normal_sobol(dim: int, n_samples: int, seed: int | None = None) -> torch.Tensor:
    # Ref.: https://github.com/pytorch/botorch/blob/466da73a18731d45b034bfd36011bb3eb150fdd8/botorch/sampling/qmc.py#L26  # NOQA: E501
    sobol_engine = torch.quasirandom.SobolEngine(
        dimension=dim, scramble=True, seed=seed
    )  # type: ignore
    # The Sobol sequence in [-1, 1].
    samples = 2.0 * (sobol_engine.draw(n_samples, dtype=torch.float64) - 0.5)
    # Inverse transform to standard normal (values to close to 0/1 result in inf values).
    return torch.erfinv(samples) * float(np.sqrt(2))


def logehvi(
    means: torch.Tensor,  # (..., n_objectives)
    cov: torch.Tensor,  # (..., n_objectives, n_objectives)
    fixed_samples: torch.Tensor,  # (n_qmc_samples, n_objectives)
    non_dominated_lower_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    non_dominated_upper_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    is_objective_independent: bool = False,
) -> torch.Tensor:  # (..., )
    def _loghvi(
        loss_vals: torch.Tensor,  # (..., n_objectives)
        non_dominated_lower_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
        non_dominated_upper_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    ) -> torch.Tensor:  # shape = (..., )
        # NOTE: [Daulton20] is available at https://arxiv.org/abs/2006.05078.
        # This function calculates Eq. (1) of [Daulton20].
        # TODO(nabenabe): Adapt to Eq. (3) of [Daulton20] when we support batch optimization.
        diff = torch.nn.functional.relu(
            non_dominated_upper_bounds
            - torch.maximum(loss_vals[..., torch.newaxis, :], non_dominated_lower_bounds)
        )
        return torch.special.logsumexp(diff.log().sum(dim=-1), dim=-1)

    def _logehvi(
        loss_vals: torch.Tensor,  # (..., n_qmc_samples, n_objectives)
        non_dominated_lower_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
        non_dominated_upper_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    ) -> torch.Tensor:  # shape = (..., )
        log_n_qmc_samples = float(np.log(loss_vals.shape[-2]))
        log_hvi_vals = _loghvi(loss_vals, non_dominated_lower_bounds, non_dominated_upper_bounds)
        return -log_n_qmc_samples + torch.special.logsumexp(log_hvi_vals, dim=-1)

    # NOTE(nabenabe): By using fixed samples from the Sobol sequence, EHVI becomes deterministic,
    # making it possible to optimize the acqf by l-BFGS.
    n_objectives = means.shape[-1]
    assert cov.shape[-1] == n_objectives and cov.shape[-2] == n_objectives
    if is_objective_independent:
        L = torch.zeros_like(cov, dtype=torch.float64)
        objective_indices = torch.arange(n_objectives)
        L[..., objective_indices, objective_indices] = torch.sqrt(
            cov[..., objective_indices, objective_indices]
        )
    else:
        L = torch.linalg.cholesky(cov)

    # NOTE(nabenabe): matmul (+squeeze) below is equivalent to einsum("BMM,NM->BNM").
    loss_vals_from_qmc = (
        means[..., torch.newaxis, :]
        + torch.matmul(L[..., torch.newaxis, :, :], fixed_samples[..., torch.newaxis]).squeeze()
    )
    return _logehvi(loss_vals_from_qmc, non_dominated_lower_bounds, non_dominated_upper_bounds)


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


def logei(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
    sigma = torch.sqrt(var)
    st_val = standard_logei((mean - f0) / sigma)
    val = torch.log(sigma) + st_val
    return val


def logpi(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    # Return the integral of N(mean, var) from -inf to f0
    # This is identical to the integral of N(0, 1) from -inf to (f0-mean)/sigma
    # Return E_{y ~ N(mean, var)}[bool(y <= f0)]
    sigma = torch.sqrt(var)
    return torch.special.log_ndtr((f0 - mean) / sigma)


def ucb(mean: torch.Tensor, var: torch.Tensor, beta: float) -> torch.Tensor:
    return mean + torch.sqrt(beta * var)


def lcb(mean: torch.Tensor, var: torch.Tensor, beta: float) -> torch.Tensor:
    return mean - torch.sqrt(beta * var)


class BaseAcquisitionFunc(metaclass=ABCMeta):
    def __init__(self, X: np.ndarray, search_space: SearchSpace) -> None:
        self._is_categorical = torch.from_numpy(search_space.scale_types == ScaleType.CATEGORICAL)
        self._X = torch.from_numpy(X)

    @abstractmethod
    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x: np.ndarray, with_grad: bool) -> tuple[np.ndarray, np.ndarray | None]:
        if with_grad:
            assert x.ndim == 1
            x_tensor = torch.from_numpy(x)
            x_tensor.requires_grad_(True)
            val = self._calculate(x_tensor)
            val.backward()  # type: ignore
            return val.item(), x_tensor.grad.detach().numpy()  # type: ignore
        else:
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
        super().__init__(X=X, search_space=search_space)
        cov_Y_Y_inv = _calculate_cov_Y_Y_inv(kernel_params, X, self._is_categorical)
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv @ Y)
        self._kernel_params = kernel_params
        self._threshold = threshold if threshold is not None else np.max(Y)
        self._stabilizing_noise = stabilizing_noise

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        if not np.isneginf(self._threshold):
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
        return logei(mean=mean, var=var + self._stabilizing_noise, f0=self._threshold)


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
        super().__init__(X=X, search_space=search_space)
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
        return logpi(mean=mean, var=var + self._stabilizing_noise, f0=self._threshold)


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        X: np.ndarray,
        Y: np.ndarray,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        super().__init__(X=X, search_space=search_space)
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
        return ucb(mean=mean, var=var, beta=self._beta)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        kernel_params: KernelParamsTensor,
        X: np.ndarray,
        Y: np.ndarray,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        super().__init__(X=X, search_space=search_space)
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
        return lcb(mean=mean, var=var, beta=self._beta)


class LogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        objective_kernel_params_list: list[KernelParamsTensor],
        X: np.ndarray,
        Y: np.ndarray,
        search_space: SearchSpace,
        pareto_sols: np.ndarray | None = None,
        n_qmc_samples: int = 128,
        ref_point: np.ndarray | None = None,
        seed: int | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        super().__init__(X=X, search_space=search_space)
        assert len(X) == len(Y) and len(Y.shape) == 2
        loss_vals = -Y  # NOTE(nabenabe): Y is to be maximized, loss_vals is to be minimized.
        if pareto_sols is None:
            pareto_sols = loss_vals[_is_pareto_front(loss_vals, assume_unique_lexsorted=False)]

        ref_point = np.max(loss_vals, axis=0) * 1.1 if ref_point is None else ref_point
        lbs, ubs = get_non_dominated_hyper_rectangle_bounds(pareto_sols, ref_point)
        self._non_dominated_lower_bounds = torch.from_numpy(lbs)
        self._non_dominated_upper_bounds = torch.from_numpy(ubs)
        self._fixed_samples = _sample_from_normal_sobol(
            dim=loss_vals.shape[-1], n_samples=n_qmc_samples, seed=seed
        )
        self._cov_Y_Y_inv_list: list[torch.Tensor] = []
        self._cov_Y_Y_inv_Y_list: list[torch.Tensor] = []
        self._objective_kernel_params_list = objective_kernel_params_list
        self._stabilizing_noise = stabilizing_noise
        for kernel_params, loss in zip(objective_kernel_params_list, loss_vals.T):
            cov_Y_Y_inv = _calculate_cov_Y_Y_inv(kernel_params, X, self._is_categorical)
            self._cov_Y_Y_inv_list.append(torch.from_numpy(cov_Y_Y_inv))
            # NOTE(nabenabe): We follow minimization for multi-objective.
            self._cov_Y_Y_inv_Y_list.append(torch.from_numpy(cov_Y_Y_inv @ loss))

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        means_ = []
        vars_ = []
        for kernel_params, cov_Y_Y_inv, cov_Y_Y_inv_Y in zip(
            self._objective_kernel_params_list, self._cov_Y_Y_inv_list, self._cov_Y_Y_inv_Y_list
        ):
            mean, var = posterior(
                kernel_params, self._X, self._is_categorical, cov_Y_Y_inv, cov_Y_Y_inv_Y, x
            )
            means_.append(mean)
            vars_.append(var + self._stabilizing_noise)

        return logehvi(
            means=torch.stack(means_, axis=-1),
            cov=torch.diag_embed(torch.stack(vars_, axis=-1)),
            fixed_samples=self._fixed_samples,
            non_dominated_lower_bounds=self._non_dominated_lower_bounds,
            non_dominated_upper_bounds=self._non_dominated_upper_bounds,
            is_objective_independent=True,  # TODO(nabenabe): Introduce Multi-task GP.
        )


class ConstrainedLogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        constraint_kernel_params_list: list[KernelParamsTensor],
        X: np.ndarray,
        constraint_vals: np.ndarray,
        search_space: SearchSpace,
        objective_acqf: LogEI | LogEHVI,
        constraint_thresholds: list[float],
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert constraint_vals.shape == (X.shape[0], len(constraint_kernel_params_list))
        self._acqf_list = [objective_acqf] + [
            LogPI(kernel_params, X, c_vals, search_space, threshold, stabilizing_noise)
            for kernel_params, c_vals, threshold in zip(
                constraint_kernel_params_list, constraint_vals.T, constraint_thresholds
            )
        ]

    def _calculate(self, x: torch.Tensor) -> torch.Tensor:
        return sum(acqf_._calculate(x) for acqf_ in self._acqf_list)

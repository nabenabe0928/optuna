from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.gp import GPRegressor
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace
from optuna._hypervolume import get_non_dominated_box_bounds
from optuna.study._multi_objective import _is_pareto_front


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


def _sample_from_normal_sobol(dim: int, n_samples: int, seed: int | None) -> torch.Tensor:
    # NOTE(nabenabe): Normal Sobol sampling based on BoTorch.
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/sampling/qmc.py#L26-L97
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/sampling.py#L109-L138
    sobol_samples = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=dim, scramble=True, seed=seed
    ).draw(n_samples, dtype=torch.float64)
    samples = 2.0 * (sobol_samples - 0.5)  # The Sobol sequence in [-1, 1].
    # Inverse transform to standard normal (values to close to -1 or 1 result in infinity).
    return torch.erfinv(samples) * float(np.sqrt(2))


def logehvi(
    Y_post: torch.Tensor,  # (..., n_qmc_samples, n_objectives)
    non_dominated_box_lower_bounds: torch.Tensor,  # (n_boxes, n_objectives)
    non_dominated_box_upper_bounds: torch.Tensor,  # (n_boxes, n_objectives)
) -> torch.Tensor:  # (..., )
    log_n_qmc_samples = float(np.log(Y_post.shape[-2]))
    # This function calculates Eq. (1) of https://arxiv.org/abs/2006.05078.
    # TODO(nabenabe): Adapt to Eq. (3) when we support batch optimization.
    # TODO(nabenabe): Make the calculation here more numerically stable.
    # cf. https://arxiv.org/abs/2310.20708
    # Check the implementations here:
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/safe_math.py
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/acquisition/multi_objective/logei.py#L146-L266
    _EPS = torch.tensor(1e-12, dtype=torch.float64)  # NOTE(nabenabe): grad becomes nan when EPS=0.
    diff = torch.maximum(
        _EPS,
        torch.minimum(Y_post[..., torch.newaxis, :], non_dominated_box_upper_bounds)
        - non_dominated_box_lower_bounds,
    )
    # NOTE(nabenabe): logsumexp with dim=-1 is for the HVI calculation and that with dim=-2 is for
    # expectation of the HVIs over the fixed_samples.
    return torch.special.logsumexp(diff.log().sum(dim=-1), dim=(-2, -1)) - log_n_qmc_samples


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


class BaseAcquisitionFunc(ABC):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        is_categorical = torch.from_numpy(search_space.scale_types == ScaleType.CATEGORICAL)
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            cov_Y_Y = gpr.kernel(is_categorical, X_tensor, X_tensor).detach().numpy()
        cov_Y_Y[np.diag_indices(X.shape[0])] += gpr.noise_var.item()

        # TODO(nabenabe): Make the attributes private by `_` if necessary.
        self.is_categorical = is_categorical
        self.gpr = gpr
        self.X = torch.from_numpy(X)
        self.search_space = search_space
        self.cov_Y_Y_inv = torch.from_numpy(np.linalg.inv(cov_Y_Y))
        self.cov_Y_Y_inv_Y = self.cov_Y_Y_inv @ torch.from_numpy(Y)

    @abstractmethod
    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        # TODO(nabenabe): Rename the method.
        with torch.no_grad():
            return self.eval_acqf(torch.from_numpy(x)).detach().numpy()

    def eval_acqf_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        # TODO(nabenabe): Rename the method.
        assert x.ndim == 1
        x_tensor = torch.from_numpy(x)
        x_tensor.requires_grad_(True)
        val = self.eval_acqf(x_tensor)
        val.backward()  # type: ignore
        return val.item(), x_tensor.grad.detach().numpy()  # type: ignore


class LogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        # TODO(kAIto47802): Want to change the name to a generic name like threshold,
        # since it is not actually in operation as max_Y
        acqf_stabilizing_noise: float = 1e-12,
        max_Y: float | None = None,
    ) -> None:
        super().__init__(gpr=gpr, search_space=search_space, X=X, Y=Y)
        # TODO(nabenabe): Rename.
        self.acqf_stabilizing_noise = acqf_stabilizing_noise
        self.max_Y = max_Y if max_Y is not None else np.max(Y)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.gpr.posterior(
            self.X, self.is_categorical, self.cov_Y_Y_inv, self.cov_Y_Y_inv_Y, x
        )
        # If there are no feasible trials, max_Y is set to -np.inf.
        # If max_Y is set to -np.inf, we set logEI to zero to ignore it.
        return (
            logei(mean=mean, var=var + self.acqf_stabilizing_noise, f0=self.max_Y)
            if not np.isneginf(self.max_Y)
            else torch.tensor(0.0, dtype=torch.float64)
        )


class LogPI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        acqf_stabilizing_noise: float = 1e-12,
        max_Y: float | None = None,
    ) -> None:
        super().__init__(gpr=gpr, search_space=search_space, X=X, Y=Y)
        # TODO(nabenabe): Rename.
        self.acqf_stabilizing_noise = acqf_stabilizing_noise
        self.max_Y = max_Y if max_Y is not None else np.max(Y)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.gpr.posterior(
            self.X, self.is_categorical, self.cov_Y_Y_inv, self.cov_Y_Y_inv_Y, x
        )
        return logpi(mean=mean, var=var + self.acqf_stabilizing_noise, f0=self.max_Y)


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        beta: float,
    ) -> None:
        super().__init__(gpr=gpr, search_space=search_space, X=X, Y=Y)
        # TODO(nabenabe): Rename.
        self.beta = beta

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.gpr.posterior(
            self.X, self.is_categorical, self.cov_Y_Y_inv, self.cov_Y_Y_inv_Y, x
        )
        return ucb(mean=mean, var=var, beta=self.beta)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,
        beta: float,
    ) -> None:
        super().__init__(gpr=gpr, search_space=search_space, X=X, Y=Y)
        # TODO(nabenabe): Rename.
        self.beta = beta

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.gpr.posterior(
            self.X, self.is_categorical, self.cov_Y_Y_inv, self.cov_Y_Y_inv_Y, x
        )
        return lcb(mean=mean, var=var, beta=self.beta)


class LogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gprs_list: list[GPRegressor],
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,  # NOTE: 2D array. Refactor here later.
        n_qmc_samples: int,
        qmc_seed: int | None,
        acqf_stabilizing_noise: float = 1e-12,
    ) -> None:
        # Since all the objectives are equally important, we simply use the mean of
        # inverse of squared mean lengthscales over all the objectives.
        mean_lengthscales = np.mean(
            [1 / np.sqrt(gpr.inverse_squared_lengthscales.detach().numpy()) for gpr in gprs_list],
            axis=0,
        )
        dummy_gpr = GPRegressor(
            # inverse_squared_lengthscales is used in optim_mixed.py.
            # cf. https://github.com/optuna/optuna/blob/v4.3.0/optuna/_gp/optim_mixed.py#L200-L209
            inverse_squared_lengthscales=torch.from_numpy(1.0 / mean_lengthscales**2),
            # These parameters will not be used anywhere.
            kernel_scale=torch.empty(0),
            noise_var=torch.empty(0),
        )

        def _get_non_dominated_box_bounds() -> tuple[torch.Tensor, torch.Tensor]:
            loss_vals = -Y  # NOTE(nabenabe): Y is to be maximized, loss_vals is to be minimized.
            pareto_sols = loss_vals[_is_pareto_front(loss_vals, assume_unique_lexsorted=False)]
            ref_point = np.max(loss_vals, axis=0)
            ref_point = np.nextafter(np.maximum(1.1 * ref_point, 0.9 * ref_point), np.inf)
            lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
            # NOTE(nabenabe): Flip back the sign to make them compatible with maximization.
            return torch.from_numpy(-ubs), torch.from_numpy(-lbs)

        # TODO(nabenabe): Make them private later.
        self.gprs_list = gprs_list
        self.fixed_samples = _sample_from_normal_sobol(
            dim=Y.shape[-1], n_samples=n_qmc_samples, seed=qmc_seed
        )
        self.non_dominated_box_lower_bounds, self.non_dominated_box_upper_bounds = (
            _get_non_dominated_box_bounds()
        )
        self.acqf_stabilizing_noise = acqf_stabilizing_noise
        super().__init__(gpr=dummy_gpr, search_space=search_space, X=X, Y=Y[:, 0])

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        Y_post = []
        for i, gpr in enumerate(self.gprs_list):
            mean, var = gpr.posterior(
                self.X, self.is_categorical, self.cov_Y_Y_inv, self.cov_Y_Y_inv_Y, x
            )
            stdev = torch.sqrt(var + self.acqf_stabilizing_noise)
            # NOTE(nabenabe): By using fixed samples from the Sobol sequence, EHVI becomes
            # deterministic, making it possible to optimize the acqf by l-BFGS.
            # Sobol is better than the standard Monte-Carlo w.r.t. the approximation stability.
            # cf. Appendix D of https://arxiv.org/pdf/2006.05078
            Y_post.append(
                mean[..., torch.newaxis] + stdev[..., torch.newaxis] * self.fixed_samples[..., i]
            )

        # NOTE(nabenabe): Use the following once multi-task GP is supported.
        # L = torch.linalg.cholesky(cov)
        # Y_post = means[..., torch.newaxis, :] + torch.einsum("...MM,SM->...SM", L, fixed_samples)
        return logehvi(
            Y_post=torch.stack(Y_post, dim=-1),
            non_dominated_box_lower_bounds=self.non_dominated_box_lower_bounds,
            non_dominated_box_upper_bounds=self.non_dominated_box_upper_bounds,
        )


class ConstrainedLogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        # TODO(nabenabe): Refactor here later.
        # TODO(nabenabe): Adapt here to multi-objective later.
        search_space: SearchSpace,
        X: np.ndarray,
        Y: np.ndarray,  # NOTE: Refactor here later.
        objective_acqf: LogEI,
        constraints_acqf_list: list[LogPI],
    ) -> None:
        self.objective_acqf = objective_acqf
        self.constraints_acqf_list = constraints_acqf_list
        super().__init__(
            gpr=objective_acqf.gpr,
            search_space=search_space,
            X=X,
            Y=Y,
        )

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        acqf_val = self.objective_acqf.eval_acqf(x)
        for constraint_acqf in self.constraints_acqf_list:
            acqf_val += constraint_acqf.eval_acqf(x)
        return acqf_val

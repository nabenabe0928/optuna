from __future__ import annotations

import numpy as np
import pytest
import torch

from optuna._gp import acqf as acqf_module
from optuna._gp.gp import GPRegressor
from optuna._gp.search_space import SearchSpace
from optuna.distributions import FloatDistribution


def verify_eval_acqf(x: np.ndarray, acqf: acqf_module.BaseAcquisitionFunc) -> None:
    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad_(True)
    acqf_value = acqf.eval_acqf(x_tensor)
    acqf_value.sum().backward()  # type: ignore
    acqf_grad = x_tensor.grad
    assert acqf_grad is not None
    assert acqf_value.shape == x.shape[:-1]
    assert torch.all(torch.isfinite(acqf_value))
    assert torch.all(torch.isfinite(acqf_grad))


def get_gpr(y_train: np.ndarray) -> GPRegressor:
    gpr = GPRegressor(
        is_categorical=torch.tensor([False, False]),
        X_train=torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]], dtype=torch.float64),
        y_train=torch.from_numpy(y_train),
        inverse_squared_lengthscales=torch.tensor([2.0, 3.0], dtype=torch.float64),
        kernel_scale=torch.tensor(4.0, dtype=torch.float64),
        noise_var=torch.tensor(0.1, dtype=torch.float64),
    )
    gpr._cache_matrix()
    return gpr


@pytest.fixture
def search_space() -> SearchSpace:
    n_dims = 2
    return SearchSpace({chr(ord("a") + i): FloatDistribution(0.0, 1.0) for i in range(n_dims)})


parametrized_x = pytest.mark.parametrize(
    "x",
    [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])],  # unbatched  # batched
)

parametrized_additional_values = pytest.mark.parametrize(
    "additional_values",
    [
        np.array([[0.2], [0.3], [-0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, -0.4], [-0.2, -0.3, -0.4]]),
        np.array([[-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4]]),
    ],
)


@pytest.mark.parametrize(
    "acqf_cls", [acqf_module.LogEI, acqf_module.LCB, acqf_module.UCB, acqf_module.LogPI]
)
@parametrized_x
def test_eval_acqf(
    acqf_cls: type[acqf_module.BaseAcquisitionFunc],
    x: np.ndarray,
    search_space: SearchSpace,
) -> None:
    Y = np.array([1.0, 2.0, 3.0])
    kwargs = dict(gpr=get_gpr(Y), search_space=search_space)
    if acqf_cls in [acqf_module.LCB, acqf_module.UCB]:
        kwargs.update(beta=2.0)
    else:
        kwargs.update(threshold=np.max(Y))

    verify_eval_acqf(x, acqf_cls(**kwargs))  # type: ignore[arg-type]


@pytest.mark.parametrize("n_running", [1, 4])
def test_conditional_gpr_matches_joint(n_running: int) -> None:
    N, D, S = 10, 3, 512
    torch.manual_seed(42)
    X_train = torch.rand(N, D, dtype=torch.float64)
    y_train = torch.sin(X_train.sum(-1))
    gpr = GPRegressor(
        is_categorical=torch.zeros(D, dtype=torch.bool),
        X_train=X_train,
        y_train=y_train,
        inverse_squared_lengthscales=torch.ones(D, dtype=torch.float64),
        kernel_scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(0.01, dtype=torch.float64),
    )
    gpr._cache_matrix()

    x_running = torch.rand(n_running, D, dtype=torch.float64)
    x_new = torch.rand(D, dtype=torch.float64)
    fixed_samples = acqf_module._sample_from_normal_sobol(dim=n_running + 1, n_samples=S, seed=7)

    # Joint approach.
    joint_x = torch.cat([x_running, x_new.unsqueeze(0)], dim=0)
    mu_j, cov_j = gpr.posterior(joint_x, joint=True)
    cov_j.diagonal().add_(1e-12)
    L_j = torch.linalg.cholesky(cov_j)
    samples_joint = mu_j + fixed_samples @ L_j.mT

    # Conditional approach.
    cond = acqf_module.ConditionalGPRegressor(gpr, x_running, fixed_samples, 1e-12)
    samples_cond = cond.posterior_samples(x_new)

    torch.testing.assert_close(samples_joint, samples_cond)


@parametrized_x
def test_eval_qlogei(x: np.ndarray, search_space: SearchSpace) -> None:
    Y = np.array([1.0, 2.0, 3.0])
    acqf = acqf_module.qLogEI(
        gpr=get_gpr(Y),
        search_space=search_space,
        threshold=np.max(Y),
        n_qmc_samples=32,
        qmc_seed=42,
        normalized_params_of_running_trials=np.array([[0.4, 0.6]]),
        stabilizing_noise=0.0,
    )
    verify_eval_acqf(x, acqf)


@parametrized_x
@parametrized_additional_values
def test_eval_acqf_with_constraints(
    x: np.ndarray,
    additional_values: np.ndarray,
    search_space: SearchSpace,
) -> None:
    c = additional_values.copy()
    Y = np.array([1.0, 2.0, 3.0])
    is_feasible = np.all(c <= 0, axis=1)
    is_all_infeasible = not np.any(is_feasible)
    acqf = acqf_module.ConstrainedLogEI(
        gpr=get_gpr(Y),
        search_space=search_space,
        threshold=-np.inf if is_all_infeasible else np.max(Y[is_feasible]),
        stabilizing_noise=0.0,
        constraints_gpr_list=[get_gpr(vals) for vals in c.T],
        constraints_threshold_list=[0.0] * len(c.T),
    )
    verify_eval_acqf(x, acqf)


@parametrized_x
@parametrized_additional_values
def test_eval_multi_objective_acqf(
    x: np.ndarray,
    additional_values: np.ndarray,
    search_space: SearchSpace,
) -> None:
    Y = np.hstack([np.array([1.0, 2.0, 3.0])[:, np.newaxis], additional_values])
    n_objectives = Y.shape[-1]
    acqf = acqf_module.LogEHVI(
        gpr_list=[get_gpr(Y[:, i]) for i in range(n_objectives)],
        search_space=search_space,
        Y_train=torch.from_numpy(Y),
        n_qmc_samples=32,
        qmc_seed=42,
        stabilizing_noise=0.0,
    )
    verify_eval_acqf(x, acqf)


@parametrized_x
@parametrized_additional_values
def test_eval_multi_objective_acqf_with_constraints(
    x: np.ndarray,
    additional_values: np.ndarray,
    search_space: SearchSpace,
) -> None:
    c = additional_values.copy()
    Y = np.hstack([np.array([1.0, 2.0, 3.0])[:, np.newaxis], additional_values])
    n_objectives = Y.shape[-1]
    is_feasible = np.all(c <= 0, axis=1)
    is_all_infeasible = not np.any(is_feasible)
    acqf = acqf_module.ConstrainedLogEHVI(
        gpr_list=[get_gpr(Y[:, i]) for i in range(n_objectives)],
        search_space=search_space,
        Y_feasible=None if is_all_infeasible else torch.from_numpy(Y[is_feasible]),
        n_qmc_samples=32,
        qmc_seed=42,
        constraints_gpr_list=[get_gpr(vals) for vals in c.T],
        constraints_threshold_list=[0.0] * len(c.T),
        stabilizing_noise=0.0,
    )
    verify_eval_acqf(x, acqf)

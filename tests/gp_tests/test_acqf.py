from __future__ import annotations

import numpy as np
import pytest
import torch

from optuna._gp import acqf as acqf_module
from optuna._gp.gp import GPRegressor
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


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


@pytest.fixture
def X() -> np.ndarray:
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]])


@pytest.fixture
def gpr() -> GPRegressor:
    return GPRegressor(
        inverse_squared_lengthscales=torch.tensor([2.0, 3.0], dtype=torch.float64),
        kernel_scale=torch.tensor(4.0, dtype=torch.float64),
        noise_var=torch.tensor(0.1, dtype=torch.float64),
    )


@pytest.fixture
def search_space() -> SearchSpace:
    n_dims = 2
    return SearchSpace(
        scale_types=np.full(n_dims, ScaleType.LINEAR),
        bounds=np.array([[0.0, 1.0] * n_dims]),
        steps=np.zeros(n_dims),
    )


parametrized_x = pytest.mark.parametrize(
    "x", [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])]  # unbatched  # batched
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
    "acqf_cls", [acqf_module.LogEI, acqf_module.UCB, acqf_module.LCB, acqf_module.LogPI]
)
@parametrized_x
def test_eval_acqf(
    acqf_cls: type[acqf_module.BaseAcquisitionFunc],
    x: np.ndarray,
    gpr: GPRegressor,
    search_space: SearchSpace,
    X: np.ndarray,
) -> None:
    Y = np.array([1.0, 2.0, 3.0])
    kwargs = dict(gpr=gpr, search_space=search_space, X=X, Y=Y)
    if acqf_cls in [acqf_module.LCB, acqf_module.UCB]:
        kwargs.update(beta=2.0)
    else:
        kwargs.update(acqf_stabilizing_noise=0.0)

    acqf = acqf_cls(**kwargs)  # type: ignore[arg-type]
    verify_eval_acqf(x, acqf)


@parametrized_x
@parametrized_additional_values
def test_eval_acqf_with_constraints(
    x: np.ndarray,
    additional_values: np.ndarray,
    gpr: GPRegressor,
    search_space: SearchSpace,
    X: np.ndarray,
) -> None:
    c = additional_values.copy()
    Y = np.array([1.0, 2.0, 3.0])
    is_feasible = np.all(c <= 0, axis=1)
    is_all_infeasible = not np.any(is_feasible)
    objective_acqf = acqf_module.LogEI(
        gpr=gpr,
        search_space=search_space,
        X=X,
        Y=Y,
        max_Y=-np.inf if is_all_infeasible else np.max(Y[is_feasible]),
        acqf_stabilizing_noise=0.0,
    )
    constraints_acqf_list = [
        acqf_module.LogPI(
            gpr=gpr, search_space=search_space, X=X, Y=vals, acqf_stabilizing_noise=0.0, max_Y=0.0
        )
        for vals in c.T
    ]
    acqf = acqf_module.ConstrainedLogEI(
        search_space=search_space,
        X=X,
        Y=Y,
        objective_acqf=objective_acqf,
        constraints_acqf_list=constraints_acqf_list,
    )
    verify_eval_acqf(x, acqf)


@parametrized_x
@parametrized_additional_values
def test_eval_multi_objective_acqf(
    x: np.ndarray,
    additional_values: np.ndarray,
    gpr: GPRegressor,
    search_space: SearchSpace,
    X: np.ndarray,
) -> None:
    Y = np.hstack([np.array([1.0, 2.0, 3.0])[:, np.newaxis], additional_values])
    n_objectives = Y.shape[-1]
    acqf = acqf_module.LogEHVI(
        gprs_list=[gpr for _ in range(n_objectives)],
        search_space=search_space,
        X=X,
        Y=Y,
        acqf_stabilizing_noise=0.0,
        n_qmc_samples=32,
        qmc_seed=42,
    )
    verify_eval_acqf(x, acqf)

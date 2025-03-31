from __future__ import annotations

import numpy as np
import pytest
import torch

import optuna._gp.acqf as acqf_module
from optuna._gp.gp import KernelParamsTensor
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


@pytest.mark.parametrize(
    "acqf_cls", [acqf_module.LogEI, acqf_module.UCB, acqf_module.LCB, acqf_module.LogPI]
)
@pytest.mark.parametrize(
    "x", [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])]  # unbatched  # batched
)
def test_eval_acqf(acqf_cls: type[acqf_module.BaseAcquisitionFunc], x: np.ndarray) -> None:
    n_dims = 2
    X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]])
    Y = np.array([1.0, 2.0, 3.0])
    kernel_params = KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([2.0, 3.0], dtype=torch.float64),
        kernel_scale=torch.tensor(4.0, dtype=torch.float64),
        noise_var=torch.tensor(0.1, dtype=torch.float64),
    )
    search_space = SearchSpace(
        scale_types=np.full(n_dims, ScaleType.LINEAR),
        bounds=np.array([[0.0, 1.0] * n_dims]),
        steps=np.zeros(n_dims),
    )
    acqf: acqf_module.BaseAcquisitionFunc
    if acqf_cls == acqf_module.LogEI:
        acqf = acqf_module.LogEI(
            kernel_params=kernel_params, search_space=search_space, X=X, Y=Y, stabilizing_noise=0.0
        )
    elif acqf_cls == acqf_module.LogPI:
        acqf = acqf_module.LogPI(
            kernel_params=kernel_params, search_space=search_space, X=X, Y=Y, stabilizing_noise=0.0
        )
    elif acqf_cls == acqf_module.LCB:
        acqf = acqf_module.LCB(
            kernel_params=kernel_params, search_space=search_space, X=X, Y=Y, beta=2.0
        )
    elif acqf_cls == acqf_module.UCB:
        acqf = acqf_module.UCB(
            kernel_params=kernel_params, search_space=search_space, X=X, Y=Y, beta=2.0
        )
    else:
        assert False, f"Unknown acqf_cls: {acqf_cls.__name__}"

    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad_(True)
    acqf_value = acqf._calculate(x_tensor)
    acqf_value.sum().backward()  # type: ignore
    acqf_grad = x_tensor.grad
    assert acqf_grad is not None
    assert acqf_value.shape == x.shape[:-1]
    assert torch.all(torch.isfinite(acqf_value))
    assert torch.all(torch.isfinite(acqf_grad))


@pytest.mark.parametrize(
    "x", [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])]  # unbatched  # batched
)
@pytest.mark.parametrize(
    "c",
    [
        np.array([[0.2], [0.3], [-0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, -0.4], [-0.2, -0.3, -0.4]]),
        np.array([[-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4]]),
    ],
)
def test_eval_acqf_with_constraints(x: np.ndarray, c: np.ndarray) -> None:
    n_dims = 2
    X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]])
    Y = np.array([1.0, 2.0, 3.0])
    kernel_params = KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([2.0, 3.0], dtype=torch.float64),
        kernel_scale=torch.tensor(4.0, dtype=torch.float64),
        noise_var=torch.tensor(0.1, dtype=torch.float64),
    )
    search_space = SearchSpace(
        scale_types=np.full(n_dims, ScaleType.LINEAR),
        bounds=np.array([[0.0, 1.0] * n_dims]),
        steps=np.zeros(n_dims),
    )

    is_feasible = np.all(c <= 0, axis=1)
    is_all_infeasible = not np.any(is_feasible)
    objective_acqf = acqf_module.LogEI(
        kernel_params=kernel_params,
        search_space=search_space,
        X=X,
        Y=Y,
        threshold=-np.inf if is_all_infeasible else np.max(Y[is_feasible]),
        stabilizing_noise=0.0,
    )
    n_constraints = c.shape[-1]
    acqf = acqf_module.ConstrainedLogEI(
        objective_acqf=objective_acqf,
        X=X,
        constraint_vals=c,
        constraint_kernel_params_list=[kernel_params for _ in range(n_constraints)],
        constraint_thresholds=[0.0] * n_constraints,
        stabilizing_noise=0.0,
    )

    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad_(True)

    acqf_value = acqf._calculate(x_tensor)
    acqf_value.sum().backward()  # type: ignore
    acqf_grad = x_tensor.grad
    assert acqf_grad is not None
    assert acqf_value.shape == x.shape[:-1]
    assert torch.all(torch.isfinite(acqf_value))
    assert torch.all(torch.isfinite(acqf_grad))

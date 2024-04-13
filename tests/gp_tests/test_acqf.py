from __future__ import annotations

import numpy as np
import pytest
import torch

from optuna._gp.acqf import BaseAcquisitionFunc

from optuna._gp.acqf import LCB
from optuna._gp.acqf import LogEI
from optuna._gp.acqf import UCB
from optuna._gp.gp import KernelParamsTensor
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


@pytest.mark.parametrize("acqf_cls", [LogEI, UCB, LCB])
@pytest.mark.parametrize(
    "x", [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])]  # unbatched  # batched
)
def test_eval_acqf(acqf_cls: BaseAcquisitionFunc, x: np.ndarray) -> None:
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

    kwargs = dict(stabilizing_noise=0.0) if acqf_cls == LogEI else dict(beta=2.0)
    acqf = acqf_cls(
        kernel_params=kernel_params, search_space=search_space, X=X, Y=Y, **kwargs
    )

    if x.ndim == 1:
        acqf_value, grad = acqf.eval_with_grad(x)
        assert grad is not None
        assert grad.shape == x.shape
        assert np.all(np.isfinite(grad))
        assert np.isfinite(acqf_value)
        assert isinstance(acqf_value, float)

    acqf_value = acqf.eval_with_no_grad(x)
    assert acqf_value.shape == x.shape[:-1]
    assert np.all(np.isfinite(acqf_value))

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING

import numpy as np
from packaging.version import Version


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

    import scipy
else:
    from optuna._imports import _LazyImport

    scipy = _LazyImport("scipy")


@contextmanager
def single_blas_thread_if_scipy_v1_15_or_newer() -> Generator[None]:
    """
    This function limits the thread count in the context to 1. We need to do so because the
    L-BFGS-B in SciPy v1.15 or newer uses OpenBLAS and it apparently causes slowdown due to the
    unmatched thread setup. This context manager aims to solve this issue. If the SciPy version is
    1.14.1 or older, this issue does not happen because it uses the Fortran implementation.

    Reference:
        https://github.com/scipy/scipy/issues/22438

    TODO(nabe): Watch the SciPy update and remove this context manager once it becomes unnecessary.
    TODO(nabe): Benchmark the speed without this context manager for any SciPy updates.
    NOTE(nabe): I don't know why, but `fmin_l_bfgs_b` in optim_mixed.py seems unaffected.
    """
    if Version(scipy.__version__) < Version("1.15.0"):
        # If SciPy is older than 1.15.0, the context manager is unnecessary.
        yield
    else:
        old_val = os.environ.get("OPENBLAS_NUM_THREADS")
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        try:
            yield
        finally:
            if old_val is None:
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = old_val


def minimize(
    func: Callable[[np.ndarray], tuple[float, np.ndarray]], x0: np.ndarray, gtol: float
) -> np.ndarray:
    with single_blas_thread_if_scipy_v1_15_or_newer():
        # jac=True means loss_func returns the gradient for gradient descent.
        # Too small `gtol` causes instability in loss_func optimization.
        res = scipy.optimize.minimize(
            func, x0, jac=True, method="l-bfgs-b", options={"gtol": gtol}
        )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return res.x


def fmin_l_bfgs_b(
    func: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    pgtol: float,
    maxiter: int,
) -> tuple[np.ndarray, float, int]:
    with single_blas_thread_if_scipy_v1_15_or_newer():
        x_opt, func_val_opt, info = scipy.optimize.fmin_l_bfgs_b(
            func=func, x0=x0, bounds=bounds, pgtol=pgtol, maxiter=maxiter
        )
    n_iters = info["nit"]
    return x_opt, func_val_opt, n_iters

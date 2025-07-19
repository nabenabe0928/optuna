from __future__ import annotations

import numpy as np
from optuna.samplers._tpe._erf import _ndtr_negative_non_big
from optuna.samplers._tpe._erf import _log_ndtr_negative
from optuna.samplers._tpe._erf import ndtr_negative
from optuna.samplers._tpe._erf import log_ndtr_negative
from scipy.special import log_ndtr as scipy_log_ndtr
from scipy.special import ndtr as scipy_ndtr
import pytest


REL_ERROR_TOL = 1e-12


@pytest.mark.parametrize(
    "x", [
        np.linspace(-10000, 0, 20000),
        np.linspace(-50, 0, 100),
        -(10**np.linspace(-300, 0, 100)),
    ]
)
def test_approx_error_in_log_ndtr_negative(x: np.ndarray) -> None:
    for target_func in [log_ndtr_negative, _log_ndtr_negative]:
        approx = _log_ndtr_negative(x)
        true_value = scipy_log_ndtr(x)
        abs_error = np.abs(approx - true_value)
        rel_error = abs_error / np.abs(true_value)
        assert rel_error.max() < REL_ERROR_TOL, f"{target_func.__name__} failed."


@pytest.mark.parametrize(
    "x", [
        np.linspace(-50, 0, 20000),
        np.linspace(-50, 0, 100),
        -(10**np.linspace(-300, 0, 100)),
    ]
)
def test_approx_error_in_ndtr_negative(x: np.ndarray) -> None:
    for target_func in [ndtr_negative, _ndtr_negative_non_big]:
        approx = _log_ndtr_negative(x)
        true_value = scipy_log_ndtr(x)
        abs_error = np.abs(approx - true_value)
        rel_error = abs_error / np.abs(true_value)
        assert rel_error.max() < REL_ERROR_TOL, f"{target_func.__name__} failed."


@pytest.mark.parametrize("x", [-np.arange(1000), -np.arange(5)])
def test_log_ndtr_negative_with_nan(x: np.ndarray) -> None:
    x = np.append(x, np.nan)
    approx = log_ndtr_negative(x)
    assert np.isnan(approx[-1])


@pytest.mark.parametrize("x", [-np.arange(1000), -np.arange(5)])
def test_ndtr_negative_with_nan(x: np.ndarray) -> None:
    x = np.append(x, np.nan)
    approx = ndtr_negative(x)
    assert np.isnan(approx[-1])


@pytest.mark.parametrize("x", [-np.arange(1000), -np.arange(5)])
def test_ndtr_negative_with_positive(x: np.ndarray) -> None:
    x = np.append(x, 10)
    with pytest.raises(AssertionError):
        log_ndtr_negative(x)


@pytest.mark.parametrize("x", [-np.arange(1000), -np.arange(5)])
def test_ndtr_negative_with_nan(x: np.ndarray) -> None:
    x = np.append(x, 10)
    with pytest.raises(AssertionError):
        ndtr_negative(x)

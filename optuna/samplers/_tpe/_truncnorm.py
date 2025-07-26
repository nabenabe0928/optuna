# This file contains the codes from SciPy project.
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math
import sys

import numpy as np

from optuna.samplers._tpe._erf import log_ndtr_negative as numpy_log_ndtr_negative
from optuna.samplers._tpe._erf import ndtr_negative as numpy_ndtr_negative


_norm_pdf_logC = 0.5 * math.log(2 * math.pi)
_ndtri_exp_approx_C = math.sqrt(3) / math.pi
_log_2 = math.log(2)
_sqrt_2 = 2**0.5


def _log_ndtr_negative_single(a: float) -> float:
    if a > -20:
        return math.log(0.5 * (math.erfc(-a / _sqrt_2) if a < -1 else 1 + math.erf(a / _sqrt_2)))

    log_LHS = -0.5 * a**2 - math.log(-a) - _norm_pdf_logC
    last_total = 0.0
    right_hand_side = 1.0
    numerator = 1.0
    denom_factor = 1.0
    denom_cons = 1 / a**2
    sign = 1
    i = 0
    while abs(last_total - right_hand_side) > sys.float_info.epsilon:
        i += 1
        last_total = right_hand_side
        sign = -sign
        denom_factor *= denom_cons
        numerator *= 2 * i - 1
        right_hand_side += sign * numerator * denom_factor

    return log_LHS + math.log(right_hand_side)


def _log_ndtr_negative(a: np.ndarray) -> np.ndarray:
    if a.size < 300:
        return np.asarray([_log_ndtr_negative_single(v) for v in a.ravel()]).reshape(a.shape)

    return numpy_log_ndtr_negative(a)


def _ndtr_negative(a: np.ndarray) -> np.ndarray:
    if a.size < 2000:
        return 0.5 * (1 + np.asarray([math.erf(v) for v in a.ravel() / _sqrt_2])).reshape(a.shape)

    return numpy_ndtr_negative(a)


def _log_gauss_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Log of Gaussian probability mass within an interval of [a, b]. Calculations in right tail are
    inaccurate, so we'll exploit the symmetry and work only in the left tail. The central part
    where `a <= 0 <= b` is handled separately. The central part was previously implemented as
    logaddexp(_log_gauss_mass(a, 0) + _log_gauss_mass(0, b)), but as the result goes to one,
    catastrophic cancellation occurs. To avoid this, we use an alternative formulation.
    """
    assert a.shape == b.shape
    right_inds = np.nonzero(a > 0)
    a[right_inds], b[right_inds] = -b[right_inds], -a[right_inds]
    out = np.empty_like(a)
    if (left_inds := np.nonzero(b <= 0))[0].size:
        log_ndtr_b = _log_ndtr_negative(b[left_inds])
        log_ndtr_a = _log_ndtr_negative(a[left_inds])
        out[left_inds] = log_ndtr_b + np.log1p(-np.exp(log_ndtr_a - log_ndtr_b))
    if (central_inds := np.nonzero(b > 0))[0].size:
        out[central_inds] = np.log1p(
            -_ndtr_negative(a[central_inds]) - _ndtr_negative(-b[central_inds])
        )
    return out


def _ndtri_exp(y: np.ndarray) -> np.ndarray:
    """
    Use the Newton method to efficiently find the root.

    `ndtri_exp(y)` returns `x` such that `y = log_ndtr(x)`, meaning that the Newton method is
    supposed to find the root of `f(x) := log_ndtr(x) - y = 0`.

    Since `df/dx = d log_ndtr(x)/dx = (ndtr(x))'/ndtr(x) = norm_pdf(x)/ndtr(x)`, the Newton update
    is x[n + 1] := x[n] - (log_ndtr(x) - y) * ndtr(x) / norm_pdf(x).

    As an initial guess, we use the Gaussian tail asymptotic approximation:
        1 - ndtr(x) \\simeq norm_pdf(x) / x
        --> log(norm_pdf(x) / x) = -1/2 * x**2 - 1/2 * log(2pi) - log(x)

    First recall that y is a non-positive value and y = log_ndtr(inf) = 0 and
    y = log_ndtr(-inf) = -inf.

    If abs(y) is very small, x is very large, meaning that x**2 >> log(x) and
    ndtr(x) = exp(y) \\simeq 1 + y --> -y \\simeq 1 - ndtr(x). From this, we can calculate:
        log(1 - ndtr(x)) \\simeq log(-y) \\simeq -1/2 * x**2 - 1/2 * log(2pi) - log(x).
    Because x**2 >> log(x), we can ignore the second and third terms, leading to:
        log(-y) \\simeq -1/2 * x**2 --> x \\simeq sqrt(-2 log(-y)).
    where we take the positive `x` as abs(y) becomes very small only if x >> 0.
    The second order approximation version is sqrt(-2 log(-y) - log(-2 log(-y))).

    If abs(y) is very large, we use log_ndtr(x) \\simeq -1/2 * x**2 - 1/2 * log(2pi) - log(x).
    To solve this equation analytically, we ignore the log term, yielding:
        log_ndtr(x) = y \\simeq -1/2 * x**2 - 1/2 * log(2pi)
        --> y + 1/2 * log(2pi) = -1/2 * x**2 --> x**2 = -2 * (y + 1/2 * log(2pi))
        --> x = sqrt(-2 * (y + 1/2 * log(2pi))

    For the moderate y, we use Eq. (13), i.e., standard logistic CDF, in the following paper:

    - `Approximating the Cumulative Distribution Function of the Normal Distribution
      <https://jsr.isrt.ac.bd/wp-content/uploads/41n1_5.pdf>`__

    The standard logistic CDF approximates ndtr(x) with:
        exp(y) = ndtr(x) \\simeq 1 / (1 + exp(-pi * x / sqrt(3)))
        --> exp(-y) \\simeq 1 + exp(-pi * x / sqrt(3))
        --> log(exp(-y) - 1) \\simeq -pi * x / sqrt(3)
        --> x \\simeq -sqrt(3) / pi * log(exp(-y) - 1).
    """
    # z = log_ndtr(-x) --> z = log1p(-ndtr(x)) --> z = log1p(-exp(y)) --> z = log(-expm1(y)).
    # Since x becomes positive for y > -log(2), we use this formula and flip the sign later.
    flipped = y > -_log_2
    y[flipped] = np.log(-np.expm1(y[flipped]))  # y is always < -log(2) = -0.693...
    x = np.empty_like(y)
    if (small_inds := np.nonzero(y < -5))[0].size:
        x[small_inds] = -np.sqrt(-2.0 * (y[small_inds] + _norm_pdf_logC))
    if (moderate_inds := np.nonzero(y >= -5))[0].size:
        x[moderate_inds] = -_ndtri_exp_approx_C * np.log(np.expm1(-y[moderate_inds]))

    for _ in range(100):
        log_ndtr_x = _log_ndtr_negative(x)
        # NOTE(nabenabe): Use exp(log_ndtr_x - norm_logpdf_x) instead of ndtr_x / norm_pdf_x for
        # numerical stability.
        norm_logpdf_x = -x**2 / 2.0 - _norm_pdf_logC
        dx = (log_ndtr_x - y) * np.exp(log_ndtr_x - norm_logpdf_x)
        x -= dx
        if np.all(np.abs(dx) < 1e-8 * -x):  # NOTE: x is always negative.
            # Equivalent to np.isclose with atol=0.0 and rtol=1e-8.
            break
    x[flipped] *= -1
    return x


def ppf(q: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """
    Compute the percent point function (inverse of cdf) at q of the given truncated Gaussian.

    Namely, this function returns the value `c` such that:
        q = \\int_{a}^{c} f(x) dx

    where `f(x)` is the probability density function of the truncated normal distribution with
    the lower limit `a` and the upper limit `b`.

    More precisely, this function returns `c` such that:
        ndtr(c) = ndtr(a) + q * (ndtr(b) - ndtr(a))
    for the case where `a < 0`, i.e., `case_left`. For `case_right`, we flip the sign for the
    better numerical stability.
    """
    q, a, b = np.atleast_1d(q, a, b)
    q, a, b = np.broadcast_arrays(q, a, b)
    log_mass = _log_gauss_mass(a, b)
    right_inds = np.nonzero(a >= 0)
    a[right_inds] = -b[right_inds]
    log_q = np.log(q)
    log_q[right_inds] = np.log1p(-q[right_inds])
    x = _ndtri_exp(np.logaddexp(_log_ndtr_negative(a), log_mass + log_q))
    x[right_inds] *= -1
    return x


def rvs(
    a: np.ndarray,
    b: np.ndarray,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    This function generates random variates from a truncated normal distribution defined between
    `a` and `b` with the mean of `loc` and the standard deviation of `scale`.
    """
    random_state = random_state or np.random.RandomState()
    size = np.broadcast(a, b, loc, scale).shape
    quantiles = random_state.uniform(low=0, high=1, size=size)
    return ppf(quantiles, a, b) * scale + loc


def logpdf(
    x: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
) -> np.ndarray:
    z, a, b = np.atleast_1d((x - loc) / scale, a, b)
    return -z**2 / 2.0 - _norm_pdf_logC - _log_gauss_mass(a, b) - np.log(scale)

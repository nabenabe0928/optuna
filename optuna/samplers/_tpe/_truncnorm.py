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

import functools
import math
import sys

import numpy as np

from optuna.samplers._tpe._erf import log_ndtr_negative as numpy_log_ndtr_negative
from optuna.samplers._tpe._erf import ndtr_negative as numpy_ndtr_negative


_norm_pdf_C = math.sqrt(2 * math.pi)
_norm_pdf_logC = math.log(_norm_pdf_C)
_ndtri_exp_approx_C = math.sqrt(3) / math.pi


def _log_sum(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    return np.logaddexp(log_p, log_q)


def _log_diff(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    return log_p + np.log1p(-np.exp(log_q - log_p))


def _ndtr_single(a: float) -> float:
    x = a / 2**0.5

    if x < -1 / 2**0.5:
        y = 0.5 * math.erfc(-x)
    elif x < 1 / 2**0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 1.0 - 0.5 * math.erfc(x)

    return y


@functools.lru_cache(1000)
def _log_ndtr_single(a: float) -> float:
    if a > 6:
        return -_ndtr_single(-a)
    if a > -20:
        return math.log(_ndtr_single(a))

    log_LHS = -0.5 * a**2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
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
        return np.asarray([_log_ndtr_single(v) for v in a.ravel()])

    return numpy_log_ndtr_negative(a)


def _ndtr_negative(a: np.ndarray) -> np.ndarray:
    if a.size < 2000:
        return 0.5 * (1 + np.asarray([math.erf(v) for v in a.ravel() / 2**0.5])).reshape(a.shape)
    
    return numpy_ndtr_negative(a)


def _norm_logpdf(x: np.ndarray) -> np.ndarray:
    return -(x**2) / 2.0 - _norm_pdf_logC


def _log_gauss_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Log of Gaussian probability mass within an interval"""

    assert a.shape == b.shape
    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail
    right_inds = np.nonzero(a > 0)
    a[*right_inds], b[*right_inds] = -b[*right_inds], -a[*right_inds]
    left_inds = np.nonzero(b <= 0)
    central_inds = np.nonzero(b > 0)

    def mass_case_left(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _log_diff(_log_ndtr_negative(b), _log_ndtr_negative(a))

    def mass_case_central(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Previously, this was implemented as:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
        # Correct for this with an alternative formulation.
        # We're not concerned with underflow here: if only one term
        # underflows, it was insignificant; if both terms underflow,
        # the result can't accurately be represented in logspace anyway
        # because sc.log1p(x) ~ x for small x.
        return np.log1p(-_ndtr_negative(a) - _ndtr_negative(-b))

    # _lazyselect not working; don't care to debug it
    out = np.full_like(a, fill_value=np.nan, dtype=np.complex128)
    if left_inds[0].size:
        out[*left_inds] = mass_case_left(a[*left_inds], b[*left_inds])
    if central_inds[0].size:
        out[*central_inds] = mass_case_central(a[*central_inds], b[*central_inds])
    return np.real(out)  # discard ~0j


def _ndtri_exp_single(y: float) -> float:
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
    if y > -sys.float_info.min:
        return math.inf if y <= 0 else math.nan

    if y > -1e-2:  # Case 1. abs(y) << 1.
        u = -2.0 * math.log(-y)
        x = math.sqrt(u - math.log(u))
    elif y < -5:  # Case 2. abs(y) >> 1.
        x = -math.sqrt(-2.0 * (y + _norm_pdf_logC))
    else:  # Case 3. Moderate y.
        x = -_ndtri_exp_approx_C * math.log(math.exp(-y) - 1)

    for _ in range(100):
        log_ndtr_x = _log_ndtr_single(x)
        log_norm_pdf_x = -0.5 * x**2 - _norm_pdf_logC
        # NOTE(nabenabe): Use exp(log_ndtr_x - log_norm_pdf_x) instead of ndtr_x / norm_pdf_x for
        # numerical stability.
        dx = (log_ndtr_x - y) * math.exp(log_ndtr_x - log_norm_pdf_x)
        x -= dx
        if abs(dx) < 1e-8 * abs(x):  # Equivalent to np.isclose with atol=0.0 and rtol=1e-8.
            break

    return x


def _ndtri_exp(y: np.ndarray) -> np.ndarray:
    # return np.frompyfunc(_ndtri_exp_single, 1, 1)(y).astype(np.float64)
    # z = log_ndtr(-x) --> z = log1p(-ndtr(x)) --> z = log1p(-exp(y)) --> z = log(-expm1(y)).
    # Since x becomes negative for y > -log(2), we use this formula and flip the sign later.
    flipped = y > -math.log(2)
    z = np.where(flipped, np.log(-np.expm1(y)), y)  # z is always < -log(2) = -0.693...
    case_small = z < -5
    x = np.empty_like(y)
    if (z_small := z[case_small]).size:
        x[case_small] = -np.sqrt(-2.0 * (z_small + _norm_pdf_logC))
    if (z_moderate := z[~case_small]).size:
        x[~case_small] = -_ndtri_exp_approx_C * np.log(np.exp(-z_moderate) - 1)

    for _ in range(100):
        log_ndtr_x = _log_ndtr_negative(x)
        log_norm_pdf_x = _norm_logpdf(x)
        # NOTE(nabenabe): Use exp(log_ndtr_x - log_norm_pdf_x) instead of ndtr_x / norm_pdf_x for
        # numerical stability.
        dx = (log_ndtr_x - z) * np.exp(log_ndtr_x - log_norm_pdf_x)
        x -= dx
        if np.all(np.abs(dx) < 1e-8 * np.abs(x)):
            # Equivalent to np.isclose with atol=0.0 and rtol=1e-8.
            break
    return np.where(flipped, -x, x)


def ppf(q: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    q, a, b = np.atleast_1d(q, a, b)
    q, a, b = np.broadcast_arrays(q, a, b)

    case_left = a < 0
    case_right = ~case_left

    def ppf_left(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        log_Phi_x = _log_sum(_log_ndtr_negative(a), np.log(q) + _log_gauss_mass(a, b))
        return _ndtri_exp(log_Phi_x)

    def ppf_right(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        log_Phi_x = _log_sum(_log_ndtr_negative(-b), np.log1p(-q) + _log_gauss_mass(a, b))
        return -_ndtri_exp(log_Phi_x)

    out = np.empty_like(q)

    q_left = q[case_left]
    q_right = q[case_right]

    if q_left.size:
        out[case_left] = ppf_left(q_left, a[case_left], b[case_left])
    if q_right.size:
        out[case_right] = ppf_right(q_right, a[case_right], b[case_right])

    out[q == 0] = a[q == 0]
    out[q == 1] = b[q == 1]
    out[a == b] = math.nan

    return out


def rvs(
    a: np.ndarray,
    b: np.ndarray,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    random_state = random_state or np.random.RandomState()
    size = np.broadcast(a, b, loc, scale).shape
    percentiles = random_state.uniform(low=0, high=1, size=size)
    return ppf(percentiles, a, b) * scale + loc


def logpdf(
    x: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
) -> np.ndarray:
    x = (x - loc) / scale

    x, a, b = np.atleast_1d(x, a, b)

    out = _norm_logpdf(x) - _log_gauss_mass(a, b) - np.log(scale)

    x, a, b = np.broadcast_arrays(x, a, b)
    out[(x < a) | (b < x)] = -np.inf
    out[a == b] = math.nan

    return out

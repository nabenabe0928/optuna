# This code uses the modified version of erf function in FreeBSD's standard C library.
# origin: FreeBSD /usr/src/lib/msun/src/s_erf.c
# https://github.com/freebsd/freebsd-src/blob/main/lib/msun/src/s_erf.c

# /* @(#)s_erf.c 5.1 93/09/24 */
# /*
#  * ====================================================
#  * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#  *
#  * Developed at SunPro, a Sun Microsystems, Inc. business.
#  * Permission to use, copy, modify, and distribute this
#  * software is freely granted, provided that this notice
#  * is preserved.
#  * ====================================================
#  */

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import Polynomial


if TYPE_CHECKING:
    from collections.abc import Callable


erx = 8.45062911510467529297e-01

# Coefficients for approximation to erf on [0,0.84375]

pp0 = 1.28379167095512558561e-01
pp1 = -3.25042107247001499370e-01
pp2 = -2.84817495755985104766e-02
pp3 = -5.77027029648944159157e-03
pp4 = -2.37630166566501626084e-05
pp = Polynomial([pp0, pp1, pp2, pp3, pp4])
qq1 = 3.97917223959155352819e-01
qq2 = 6.50222499887672944485e-02
qq3 = 5.08130628187576562776e-03
qq4 = 1.32494738004321644526e-04
qq5 = -3.96022827877536812320e-06
qq = Polynomial([1, qq1, qq2, qq3, qq4, qq5])

# Coefficients for approximation to erf in [0.84375,1.25]

pa0 = -2.36211856075265944077e-03
pa1 = 4.14856118683748331666e-01
pa2 = -3.72207876035701323847e-01
pa3 = 3.18346619901161753674e-01
pa4 = -1.10894694282396677476e-01
pa5 = 3.54783043256182359371e-02
pa6 = -2.16637559486879084300e-03
pa = Polynomial([pa0, pa1, pa2, pa3, pa4, pa5, pa6])
qa1 = 1.06420880400844228286e-01
qa2 = 5.40397917702171048937e-01
qa3 = 7.18286544141962662868e-02
qa4 = 1.26171219808761642112e-01
qa5 = 1.36370839120290507362e-02
qa6 = 1.19844998467991074170e-02
qa = Polynomial([1, qa1, qa2, qa3, qa4, qa5, qa6])

# Coefficients for approximation to erfc in [1.25,1/0.35]

ra0 = -9.86494403484714822705e-03
ra1 = -6.93858572707181764372e-01
ra2 = -1.05586262253232909814e01
ra3 = -6.23753324503260060396e01
ra4 = -1.62396669462573470355e02
ra5 = -1.84605092906711035994e02
ra6 = -8.12874355063065934246e01
ra7 = -9.81432934416914548592e00
ra = Polynomial([ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7])
sa1 = 1.96512716674392571292e01
sa2 = 1.37657754143519042600e02
sa3 = 4.34565877475229228821e02
sa4 = 6.45387271733267880336e02
sa5 = 4.29008140027567833386e02
sa6 = 1.08635005541779435134e02
sa7 = 6.57024977031928170135e00
sa8 = -6.04244152148580987438e-02
sa = Polynomial([1, sa1, sa2, sa3, sa4, sa5, sa6, sa7, sa8])

# Coefficients for approximation to erfc in [1/.35,28]

rb0 = -9.86494292470009928597e-03
rb1 = -7.99283237680523006574e-01
rb2 = -1.77579549177547519889e01
rb3 = -1.60636384855821916062e02
rb4 = -6.37566443368389627722e02
rb5 = -1.02509513161107724954e03
rb6 = -4.83519191608651397019e02
rb = Polynomial([rb0, rb1, rb2, rb3, rb4, rb5, rb6])
sb1 = 3.03380607434824582924e01
sb2 = 3.25792512996573918826e02
sb3 = 1.53672958608443695994e03
sb4 = 3.19985821950859553908e03
sb5 = 2.55305040643316442583e03
sb6 = 4.74528541206955367215e02
sb7 = -2.24409524465858183362e01
sb = Polynomial([1, sb1, sb2, sb3, sb4, sb5, sb6, sb7])


def _ndtr_negative_non_big(x: np.ndarray) -> np.ndarray:
    """
    This function compute the cumulative distribution function (CDF) of the standard normal
    distribution in the range where the input is non-big and non-positive, i.e., -38 < x <= 0.

    ndtr is usually computed as 0.5 * erfc(-x / sqrt(2)) for x < -1 and as
    0.5 * (1 + erf(x / sqrt(2))) for x >= -1.
    Since x < 0 in this function and erf is an odd function, we can compute ndtr as:
        0.5 * (1 - erf(a)) for x >= -1 and 0.5 * erfc(a) for x < -1 where a = -x / sqrt(2).

    However, we use a different switching point in our function due to the choice of our piece-wise
    approximation function. In our case, the boundary of each pice is at a=0.84375, 1.25, and
    so on. As a=0.84375, i.e., x = -1.1879..., is the closest point to -1, we use it as the
    switching point.

    NOTE(nabenabe): Our switching point is also fine numerically because erf(-1.1879... / sqrt(2)) 
    = erf(-0.84375) = -0.767..., which is not smaller than -0.9 and hence does not cause underflow.
    To illustrate what I mean,, let's assume a floating number can express only up to 6 digits.
        When erf(z) is -9.12345e-1, 1 + erf(z) is 1e-1 - 1.2345e-2.
        When erf(z) is -9.91234e-1, 1 + erf(z) is 1e-2  - 1.234e-3.
        When erf(z) is -9.99123e-1, 1 + erf(z) is 1e-3 - 1.23e-4.
        When erf(z) is -9.99912e-1, 1 + erf(z) is 1e-4  - 1.2e-5.
    In the example above, the precision reduction is observed in erf(z) as the heading digits are
    filled with 9s. This explains why switching points are fine as long as erf(z) > -0.9, which
    ours fulfills.
    """
    assert len(x.shape) == 1, "Input must be a 1D array."
    assert np.all(x >= 0), "All elements must be non-positive."
    # NOTE(nabenabe): Binning is much quicker than creating individual bool arrays.
    bin_inds = np.count_nonzero(
        (a := x / -2**0.5) >= [[0.84375], [1.25], [1 / 0.35]], axis=0
    )
    out = np.empty_like(x)
    if (target_inds := np.nonzero(bin_inds == 0)[0]).size:  # Small1: a < 0.84375
        # Compute 1 - erf(a) for this range. Use erfc(a) for the other ranges.
        out[target_inds] = 1 - (u := a[target_inds]) * (1 + pp(z := u * u) / qq(z))
    if (target_inds := np.nonzero(bin_inds == 1)[0]).size:  # Small2: 0.84375 <= a < 1.25
        out[target_inds] = 1 - erx - pa(s := a[target_inds] - 1) / qa(s)
    if (target_inds := np.nonzero(bin_inds == 2)[0]).size:  # Med1: 1.25 <= a < 1 / 0.35
        u = a[target_inds]
        out[target_inds] = np.exp(-(z := u * u) - 0.5625 + ra(s := 1 / z) / sa(s)) / u
    if (target_inds := np.nonzero(bin_inds == 3)[0]).size:  # Med2: a > 1 / 0.35
        # The computation is accurate up to a < 27
        u = a[target_inds]
        out[target_inds] = np.exp(-(z := u * u) - 0.5625 + rb(s := 1 / z) / sb(s)) / u
    return 0.5 * out  # Don't forget multiplying 0.5.


def ndtr_negative(x: np.ndarray) -> np.ndarray:
    """
    This function computes the cumulative distribution function (CDF) of the standard normal
    distribution at non-positive values, i.e., x <= 0.

    NOTE(nabenabe): ndtr is used only for non-positive values, but the positive side can be
    computed. Please compute at the negative side first and use ndtr(-x) = 1 - ndtr(x) for
    numerical stability. For positive side, ndtr(-x) = 1 - ndtr(x) can appropriately calculated up
    to x < 9. So please return 1.0 for x >= 9 for the efficiency.
    """
    x_ravel = x.ravel()
    out = np.where(np.isnan(x_ravel), np.nan, 0.0)
    non_big_inds = np.nonzero(-38 < x_ravel)[0]
    out[non_big_inds] = _ndtr_negative_non_big(x_ravel[non_big_inds])
    return out.reshape(x.shape)


def _log_ndtr_negative(x: np.ndarray) -> np.ndarray:
    """
    This function computes the logarithm of the cumulative distribution function (CDF) of the
    standard normal distribution at non-positive values, i.e., x <= 0.

    NOTE(nabenabe): log_ndtr is also used only for non-positive values, but the positive side can
    be computed. Denote z := -ndtr(-x). Then, we can compute log_ndtr at the positive side as:
        0.0 (x > 38), z (8 < x <= 38), log1p(z) (x <= 8).
    z is the first term of the Taylor series expansion of log1p(z). This formula is applicable
    because z goes to zero as x grows.
    """
    assert len(x.shape) == 1, "Input must be a 1D array."
    assert np.all(x <= 0), "All elements must be non-positive."
    # NOTE(nabenabe): Binning is much quicker than creating individual bool arrays.
    bin_inds = np.count_nonzero(
        (a := x / -2**0.5) >= [[0.84375], [1.25], [1 / 0.35]], axis=0
    )
    out = np.empty_like(x)
    # We use the same piece-wise approximation as in _ndtr_negative_non_big.
    # The only difference is that we use med2 for even big values.
    if (target_inds := np.nonzero(bin_inds == 0)[0]).size:  # Small1: a < 0.84375
        out[target_inds] = np.log(1 - (u := a[target_inds]) * (1 + pp(z := u * u) / qq(z)))
    if (target_inds := np.nonzero(bin_inds == 1)[0]).size:  # Small2: 0.84375 <= a < 1.25
        out[target_inds] = np.log(1 - erx - pa(s := a[target_inds] - 1) / qa(s))
    if (target_inds := np.nonzero(bin_inds == 2)[0]).size:  # Med1: 1.25 <= a < 1 / 0.35
        u = a[target_inds]
        out[target_inds] = -(z := u * u) - 0.5625 + ra(s := 1 / z) / sa(s) - np.log(u)
    if (target_inds := np.nonzero(bin_inds == 3)[0]).size:  # Med2: a > 1 / 0.35
        # The computation is accurate for even big a.
        u = a[target_inds]
        out[target_inds] = -(z := u * u) - 0.5625 + rb(s := 1 / z) / sb(s) - np.log(u)
    return out - math.log(2)


def log_ndtr_negative(x: np.ndarray) -> np.ndarray:
    x_ravel = x.ravel()
    out = np.full_like(x_ravel, np.nan, dtype=float)
    not_nan_inds = np.nonzero(~np.isnan(x_ravel))[0]
    out[not_nan_inds] = _log_ndtr_negative(x_ravel[not_nan_inds])
    return out.reshape(x.shape)
